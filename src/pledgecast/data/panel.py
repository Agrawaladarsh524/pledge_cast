"""Point-in-time panel assembly - PLAN.md sec.7, sec.9.3.

    "THIS LAYER IS WHERE LEAKAGE IS PREVENTED."  (sec.7, architecture diagram)

    "The single point where leakage is prevented - one file to audit." (sec.8.1)

Two rules, and nothing else in the project is allowed to bend them:

    observation_date = quarter_end + 30 calendar days, rolled to the next
                       trading day

    a filing may enter that row only if  submission_date <= observation_date

**Why 30 days** (sec.9.3): SEBI requires the shareholding pattern within 21 days
of quarter end and observed lags run 7-18 days, so a 30-day cutoff captures
essentially every filing while staying strictly leak-free. Measured on the pilot
set, 0 of 100 filings were submitted after the cutoff.

**Why a fixed lag rather than each company's own filing date**: it aligns every
company onto ONE observation date per quarter. Within-quarter ROC-AUC (sec.9.6)
asks "on this date, did the model rank the right companies higher?" - a question
that only exists if the companies share a date. Anchoring on each company's own
submission date would silently destroy the primary metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.exceptions import LeakageError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def observation_date_for(
    quarter_end: str,
    lag_days: int,
    trading_days: np.ndarray | None = None,
) -> str | None:
    """``quarter_end + lag_days``, rolled forward to the next trading day.

    Returns ``None`` when the rolled date would fall beyond the available
    trading calendar - that quarter simply has no observable date yet.
    """
    target = (pd.Timestamp(quarter_end) + pd.Timedelta(days=lag_days)).date().isoformat()
    if trading_days is None or len(trading_days) == 0:
        return target

    position = np.searchsorted(trading_days, target, side="left")
    if position >= len(trading_days):
        return None
    return str(trading_days[position])


def build_observation_grid(
    quarters: list[str],
    symbols: list[str],
    *,
    lag_days: int,
    trading_days: np.ndarray | None = None,
) -> pd.DataFrame:
    """The full (symbol x quarter) grid with its shared observation dates."""
    mapping = {
        quarter: observation_date_for(quarter, lag_days, trading_days) for quarter in quarters
    }
    usable = {q: d for q, d in mapping.items() if d is not None}

    dropped = sorted(set(mapping) - set(usable))
    if dropped:
        logger.info("quarters with no observable date yet: %s", dropped)

    grid = pd.MultiIndex.from_product(
        [sorted(symbols), sorted(usable)], names=["symbol", "quarter_end"]
    ).to_frame(index=False)
    grid["observation_date"] = grid["quarter_end"].map(usable)
    return grid


def apply_point_in_time_filter(
    grid: pd.DataFrame,
    features: pd.DataFrame,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Join features onto the grid, dropping anything not yet filed.

    THE rule. A feature row survives only if its ``submission_date`` is on or
    before the row's ``observation_date``. Anything later did not exist when the
    prediction would have been made.
    """
    merged = grid.merge(features, on=["symbol", "quarter_end"], how="left")

    if "submission_date" not in merged.columns:
        raise LeakageError(
            "features carry no submission_date - the point-in-time rule cannot be applied"
        )

    known = merged["submission_date"].notna()
    in_time = merged["submission_date"] <= merged["observation_date"]
    leaked = known & ~in_time

    if leaked.any():
        offenders = merged.loc[
            leaked, ["symbol", "quarter_end", "submission_date", "observation_date"]
        ]
        logger.warning(
            "%d rows filed after their observation date - blanked (e.g. %s)",
            int(leaked.sum()),
            offenders.head(3).to_dict("records"),
        )

    # Blank the features rather than dropping the row: the company still exists
    # on that date, it simply has no pledge data yet. Market features and the
    # label remain perfectly valid.
    feature_columns = [
        c for c in features.columns if c not in ("symbol", "quarter_end", "submission_date")
    ]
    merged.loc[leaked, feature_columns] = np.nan
    merged.loc[leaked, "submission_date"] = pd.NA

    if strict:
        assert_no_leakage(merged)

    return merged


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Hard invariant: no surviving row was filed after its observation date."""
    if "submission_date" not in frame.columns:
        return
    known = frame["submission_date"].notna()
    violations = frame[known & (frame["submission_date"] > frame["observation_date"])]
    if not violations.empty:
        sample = violations.head(5)[
            ["symbol", "quarter_end", "submission_date", "observation_date"]
        ]
        raise LeakageError(
            f"{len(violations)} panel rows carry data filed after their observation "
            f"date:\n{sample.to_string(index=False)}"
        )


def filing_lag_report(features: pd.DataFrame, grid: pd.DataFrame, lag_days: int) -> dict:
    """How much the 30-day cutoff actually costs, measured rather than assumed."""
    merged = grid.merge(
        features[["symbol", "quarter_end", "submission_date"]],
        on=["symbol", "quarter_end"],
        how="inner",
    ).dropna(subset=["submission_date"])

    if merged.empty:
        return {"n": 0}

    lag = (
        pd.to_datetime(merged["submission_date"]) - pd.to_datetime(merged["quarter_end"])
    ).dt.days
    late = int((merged["submission_date"] > merged["observation_date"]).sum())

    return {
        "n": len(merged),
        "lag_min": int(lag.min()),
        "lag_median": float(lag.median()),
        "lag_max": int(lag.max()),
        "lag_p99": float(lag.quantile(0.99)),
        "filed_after_cutoff": late,
        "pct_lost_to_cutoff": 100.0 * late / len(merged),
        "cutoff_days": lag_days,
    }


__all__ = [
    "apply_point_in_time_filter",
    "assert_no_leakage",
    "build_observation_grid",
    "filing_lag_report",
    "observation_date_for",
]
