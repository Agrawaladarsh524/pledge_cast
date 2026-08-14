"""Reg 31 event features - the event stream, not the quarterly snapshot.

**Why this module exists.** The 8 pledge features in ``pledge.py`` are computed
from the quarterly shareholding filing - one number per company per quarter.
Measured on this panel that number is UNCHANGED in 90.5% of company-quarters,
which is the mechanical reason the headline result came back negative. SEBI
Regulation 31 requires promoters to disclose each individual pledge action
within days of it happening, so the event stream is the same phenomenon observed
at far higher frequency. This module tests whether that resolution carries
anything the quarterly snapshot loses.

Three things were established by profiling the 35,584 ingested events BEFORE any
feature was written, and each one changed the design:

**1. The raw event table is contaminated.** 20,788 of the 35,584 events belong to
companies whose quarterly filings show a promoter pledge of exactly 0.00% in
every one of the 20 quarters. CDSL alone contributes 6,104 - releases of 4, 30
and 50 shares by "Indian Clearing Corporation Limited", each rounding to 0.00% of
equity. That is clearing and custodial machinery disclosed through the same
endpoint, not promoter loan collateral. A naive ``count_events_90d`` would have
ranked CDSL the most pledge-active company in the study while its actual promoter
pledge is zero - a feature that looks informative and is pure noise.

**2. The materiality rule comes from the data, not from a chosen threshold.**
``pct_equity`` is filed to two decimals, so every event is either exactly 0.00%
or at least 0.01%. Keeping the non-zero ones retains 10,531 events, of which
98.4% sit at companies that genuinely carry a pledge - up from 41.6% unfiltered.
The threshold is configurable but there is nothing arbitrary to tune: it is the
filing's own precision.

**3. The filter also removes a hidden time confound.** Raw event volume swings
nine-fold across years (10,446 in 2023 against 1,165 in 2025) because the
contaminated companies file in bursts. A count feature built on that would encode
*which year it is* - the market-timing confound of sec.9.6 in a new costume.
After filtering, annual volume is flat: 1209 / 1107 / 1316 / 1140 / 902.

**The disclosure lag is the leakage risk here.** ``pledge_events.event_date``
records when the pledge was CREATED, not when it became public. Regulation 31
allows up to 7 working days to disclose it, so an event dated the 1st may not
have been knowable until the 10th. Counting events by ``event_date <=
observation_date`` would hand the model information that did not exist yet -
precisely the failure ``data/panel.py`` exists to prevent. Every window here is
therefore closed ``disclosure_lag_days`` before the observation date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

EVENT_FEATURES = [
    "event_created_90d",
    "event_released_90d",
    "event_net_90d",
    "event_count_90d",
    "event_days_since",
    "event_invocations_365d",
]


def filter_material(events: pd.DataFrame, min_pct_equity: float) -> tuple[pd.DataFrame, dict]:
    """Drop encumbrances too small to be promoter financing.

    Returns ``(material, report)``. The report is logged and printed by
    ``03_build_panel.py`` so the exclusion is an auditable number rather than a
    silent filter.
    """
    if events.empty:
        return events, {"total": 0, "material": 0, "dropped": 0}

    material = events[events["pct_equity"].fillna(0.0) >= min_pct_equity]
    report = {
        "total": len(events),
        "material": len(material),
        "dropped": len(events) - len(material),
        "companies_total": int(events["symbol"].nunique()),
        "companies_material": int(material["symbol"].nunique()),
        "min_pct_equity": min_pct_equity,
    }
    logger.info(
        "Reg 31: %d of %d events are material (>= %.2f%% of equity); %d dropped as "
        "clearing/custodial noise",
        report["material"],
        report["total"],
        min_pct_equity,
        report["dropped"],
    )
    return material, report


def _window_sums(
    event_dates: np.ndarray,
    values: np.ndarray,
    window_start: np.ndarray,
    window_end: np.ndarray,
) -> np.ndarray:
    """Sum ``values`` for events falling in each ``[start, end]`` window.

    A prefix sum plus two binary searches, rather than a row-by-row filter.
    ``event_dates`` must be sorted; ISO date strings compare correctly as
    strings, so no datetime conversion is needed.
    """
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    left = np.searchsorted(event_dates, window_start, side="left")
    right = np.searchsorted(event_dates, window_end, side="right")
    return prefix[right] - prefix[left]


def _shift(dates: np.ndarray, days: int) -> np.ndarray:
    """ISO date strings shifted back by ``days`` calendar days."""
    return (pd.to_datetime(dates) - pd.Timedelta(days=days)).strftime("%Y-%m-%d").to_numpy()


def build_event_features(
    events: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    window_days: int,
    invocation_window_days: int,
    disclosure_lag_days: int,
) -> pd.DataFrame:
    """Attach the 6 event features to every (symbol, observation_date) row.

    ``events`` must already be filtered by :func:`filter_material`.

    A company with no events is NOT missing data - it genuinely had no pledge
    activity, so counts and sums are 0. ``event_days_since`` is the exception:
    "days since the last pledge action" is undefined when there has never been
    one, so it stays NaN. Writing a large number there would be inventing a
    fact, and both XGBoost and the LogReg missingness indicator handle NaN
    properly.
    """
    required = ["symbol", "observation_date"]
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise ValueError(f"observations is missing {missing}")

    out = observations.copy()
    for column in EVENT_FEATURES:
        out[column] = 0.0
    out["event_days_since"] = np.nan

    if events.empty or out.empty:
        logger.warning("no material Reg 31 events - every event feature is zero")
        return out

    # THE point-in-time boundary. Windows close `disclosure_lag_days` before the
    # observation date, because an event is not public the moment it happens.
    observed = out["observation_date"].to_numpy()
    window_end = _shift(observed, disclosure_lag_days)
    window_start = _shift(observed, disclosure_lag_days + window_days)
    invocation_start = _shift(observed, disclosure_lag_days + invocation_window_days)

    by_symbol = dict(iter(events.groupby("symbol")))
    positions = {symbol: np.flatnonzero(out["symbol"].to_numpy() == symbol) for symbol in by_symbol}

    for symbol, group in by_symbol.items():
        rows = positions[symbol]
        if len(rows) == 0:
            continue

        block = group.sort_values("event_date")
        dates = block["event_date"].to_numpy()
        pct = block["pct_equity"].to_numpy(dtype=float)
        kind = block["event_type"].to_numpy()

        starts, ends = window_start[rows], window_end[rows]

        created = _window_sums(dates, np.where(kind == "creation", pct, 0.0), starts, ends)
        released = _window_sums(dates, np.where(kind == "release", pct, 0.0), starts, ends)
        counted = _window_sums(dates, np.ones(len(dates)), starts, ends)
        invoked = _window_sums(
            dates,
            np.where(kind == "invocation", 1.0, 0.0),
            invocation_start[rows],
            ends,
        )

        out.iloc[rows, out.columns.get_loc("event_created_90d")] = created
        out.iloc[rows, out.columns.get_loc("event_released_90d")] = released
        out.iloc[rows, out.columns.get_loc("event_net_90d")] = created - released
        out.iloc[rows, out.columns.get_loc("event_count_90d")] = counted
        out.iloc[rows, out.columns.get_loc("event_invocations_365d")] = invoked

        # Days since the most recent event that was PUBLIC by the window end -
        # measured over all history, not just the window, so it stays defined
        # for a company whose last pledge action was two years ago.
        last = np.searchsorted(dates, ends, side="right") - 1
        known = last >= 0
        if known.any():
            gap = (
                pd.to_datetime(ends[known]) - pd.to_datetime(dates[last[known]])
            ).days.to_numpy(dtype=float)
            target = out.columns.get_loc("event_days_since")
            out.iloc[rows[known], target] = gap

    return out


def coverage_report(frame: pd.DataFrame) -> dict:
    """How sparse the event features actually are - reported, never hidden.

    Measured on the full panel this comes out near 8.6%, which is the honest
    ceiling on what any event-based experiment can demonstrate here.
    """
    if frame.empty:
        return {"rows": 0}
    active = frame["event_count_90d"] > 0
    return {
        "rows": len(frame),
        "rows_with_event": int(active.sum()),
        "pct_with_event": float(active.mean()),
        "rows_with_invocation": int((frame["event_invocations_365d"] > 0).sum()),
        "rows_with_days_since": int(frame["event_days_since"].notna().sum()),
        "companies_with_event": int(frame.loc[active, "symbol"].nunique()),
    }


__all__ = [
    "EVENT_FEATURES",
    "build_event_features",
    "coverage_report",
    "filter_material",
]
