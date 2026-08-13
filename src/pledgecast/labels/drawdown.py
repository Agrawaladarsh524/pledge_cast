"""Forward drawdown label - PLAN.md sec.9.2.

    Y(i,t) = 1  if  min( P[t+1 .. t+60] ) / P[t] - 1  <=  -0.15
    where P = ADJUSTED close, and 60 is TRADING days, not calendar days.

**Worst decline from entry, never peak-to-trough within the window.** sec.9.2 is
explicit about why: peak-to-trough fires on a stock that rose 40% and then fell
15%, which was not a downside event for anyone holding it from t. The measure
here is what an investor who bought at t actually experienced.

The continuous ``fwd_max_drawdown`` is stored alongside the binary label so the
distribution can be inspected before the -15% threshold is locked (sec.9.2:
"plot the drawdown distribution on the real panel and confirm the base rate
lands near the measured ~25%").

Rows without a full forward window get ``label_is_valid = 0`` and a NULL label -
never a 0, which would silently teach the model that the most recent quarter is
always safe (sec.9.4's embargo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def forward_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Worst return over the next ``horizon`` observations, from each entry.

    ``result[i] = min(prices[i+1 : i+1+horizon]) / prices[i] - 1``, and NaN
    wherever a full window is unavailable.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)
    if n <= horizon:
        return out

    # windows[j] == prices[1 + j : 1 + j + horizon]
    windows = np.lib.stride_tricks.sliding_window_view(prices[1:], horizon)
    forward_min = windows.min(axis=1)

    entry = prices[: len(forward_min)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[: len(forward_min)] = np.where(entry > 0, forward_min / entry - 1.0, np.nan)
    return out


def label_series(
    frame: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
    price_column: str = "adj_close",
    date_column: str = "trade_date",
) -> pd.DataFrame:
    """Add ``fwd_max_drawdown``, ``label`` and ``label_is_valid`` to one company.

    ``frame`` must be a single symbol's price history, ascending by date.
    """
    ordered = frame.sort_values(date_column).reset_index(drop=True)
    drawdown = forward_drawdown(ordered[price_column].to_numpy(dtype=float), horizon)

    ordered["fwd_max_drawdown"] = drawdown
    valid = ~np.isnan(drawdown)
    ordered["label_is_valid"] = valid.astype(int)
    # pandas' nullable Int64 so the invalid rows stay NULL rather than becoming 0.
    ordered["label"] = pd.array(
        np.where(valid, (drawdown <= threshold).astype(float), np.nan), dtype="Int64"
    )
    return ordered


def label_observations(
    observations: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """Attach labels to (symbol, observation_date) rows.

    The observation date is an as-of anchor, so the entry price is the last
    close **on or before** it - a quarter end rolled forward can still land on a
    market holiday. The forward window then starts strictly after that entry bar
    (sec.10 "assert label windows start strictly after"), which is what
    ``forward_drawdown`` computes.
    """
    if observations.empty or prices.empty:
        return observations.assign(fwd_max_drawdown=np.nan, label=pd.NA, label_is_valid=0)

    results: list[pd.DataFrame] = []
    price_groups = dict(iter(prices.groupby("symbol")))

    for symbol, group in observations.groupby("symbol"):
        history = price_groups.get(symbol)
        if history is None or history.empty:
            results.append(group.assign(fwd_max_drawdown=np.nan, label=pd.NA, label_is_valid=0))
            continue

        labelled = label_series(history, horizon=horizon, threshold=threshold)
        block = group.sort_values("observation_date").copy()

        # "The last trading day at or before the observation date."
        # searchsorted rather than merge_asof: dates are ISO strings, and
        # merge_asof rejects object-dtype keys outright.
        trade_dates = labelled["trade_date"].to_numpy()
        position = (
            np.searchsorted(trade_dates, block["observation_date"].to_numpy(), side="right") - 1
        )
        valid = position >= 0

        def pick(values: np.ndarray, position=position, valid=valid) -> np.ndarray:
            out = np.full(len(position), np.nan)
            out[valid] = values[position[valid]]
            return out

        block["entry_price"] = pick(labelled["adj_close"].to_numpy(dtype=float))
        block["fwd_max_drawdown"] = pick(labelled["fwd_max_drawdown"].to_numpy(dtype=float))

        label_values = pick(
            labelled["label"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        )
        block["label"] = pd.array(label_values, dtype="Int64")
        block["label_is_valid"] = (~np.isnan(label_values)).astype(int)

        results.append(block)

    return pd.concat(results, ignore_index=True) if results else observations


def detect_price_breaks(prices: pd.DataFrame, floor: float) -> dict[str, list[str]]:
    """Dates where a single-day move breaches ``floor`` even after adjustment.

    ``adjclose`` absorbs splits, bonuses and dividends but NOT demergers: when a
    company spins off a division, the parent's price drops by the value handed
    to shareholders and no adjustment factor compensates, because holders were
    made whole in shares of the new entity rather than in cash.

    Measured on the full universe: VEDL -64.9% (2026-04-30) and TMPV -40.2%
    (2025-10-14) are both demergers, not crashes.
    """
    breaks: dict[str, list[str]] = {}
    for symbol, group in prices.groupby("symbol"):
        ordered = group.sort_values("trade_date")
        close = ordered["adj_close"].to_numpy(dtype=float)
        if len(close) < 2:
            continue
        returns = np.diff(close) / close[:-1]
        flagged = np.where(returns < floor)[0]
        if len(flagged):
            dates = ordered["trade_date"].to_numpy()
            breaks[symbol] = [str(dates[i + 1]) for i in flagged]
    return breaks


def invalidate_windows_spanning_breaks(
    observations: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    floor: float,
) -> tuple[pd.DataFrame, dict]:
    """Void labels whose forward window contains an unabsorbed corporate action.

    sec.10 requires the assertion; this is what to DO once it fires. A demerger
    inside the window produces a large negative "return" that would be labelled
    a downside event, when the holder lost nothing. The row keeps its features
    and only its label is voided, so the company stays in the cross-section.
    """
    breaks = detect_price_breaks(prices, floor)
    if not breaks or observations.empty:
        return observations, {"symbols_flagged": 0, "rows_voided": 0, "breaks": {}}

    frame = observations.copy()
    voided = np.zeros(len(frame), dtype=bool)
    price_groups = dict(iter(prices.groupby("symbol")))

    for symbol, break_dates in breaks.items():
        history = price_groups.get(symbol)
        if history is None:
            continue
        dates = history.sort_values("trade_date")["trade_date"].to_numpy()
        rows = frame["symbol"].to_numpy() == symbol
        if not rows.any():
            continue

        observed = frame.loc[rows, "observation_date"].to_numpy()
        start = np.searchsorted(dates, observed, side="right")  # first day AFTER entry
        end = np.minimum(start + horizon, len(dates))

        hit = np.zeros(len(observed), dtype=bool)
        for break_date in break_dates:
            position = int(np.searchsorted(dates, break_date, side="left"))
            hit |= (position >= start) & (position < end)
        voided[rows] = hit

    if voided.any():
        frame.loc[voided, "label"] = pd.NA
        frame.loc[voided, "label_is_valid"] = 0
        frame.loc[voided, "fwd_max_drawdown"] = np.nan
        logger.warning(
            "voided %d labels whose forward window spans an unabsorbed corporate action "
            "across %d symbols: %s",
            int(voided.sum()),
            len(breaks),
            sorted(breaks),
        )

    return frame, {
        "symbols_flagged": len(breaks),
        "rows_voided": int(voided.sum()),
        "breaks": breaks,
    }


def summarise(frame: pd.DataFrame) -> dict:
    """Distribution stats for the sec.9.2 pre-training sanity check."""
    valid = frame[frame["label_is_valid"] == 1]
    if valid.empty:
        return {"n": 0, "n_valid": 0, "event_rate": None}

    drawdown = valid["fwd_max_drawdown"].astype(float)
    return {
        "n": len(frame),
        "n_valid": len(valid),
        "n_events": int(valid["label"].sum()),
        "event_rate": float(valid["label"].mean()),
        "drawdown_mean": float(drawdown.mean()),
        "drawdown_median": float(drawdown.median()),
        "drawdown_p05": float(drawdown.quantile(0.05)),
        "drawdown_p25": float(drawdown.quantile(0.25)),
        "drawdown_p75": float(drawdown.quantile(0.75)),
        "drawdown_min": float(drawdown.min()),
        "drawdown_max": float(drawdown.max()),
    }


__all__ = ["forward_drawdown", "label_observations", "label_series", "summarise"]
