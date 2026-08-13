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
    price_groups = dict(prices.groupby("symbol"))

    for symbol, group in observations.groupby("symbol"):
        history = price_groups.get(symbol)
        if history is None or history.empty:
            results.append(group.assign(fwd_max_drawdown=np.nan, label=pd.NA, label_is_valid=0))
            continue

        labelled = label_series(history, horizon=horizon, threshold=threshold)

        # merge_asof gives "the last trading day at or before the observation".
        merged = pd.merge_asof(
            group.sort_values("observation_date"),
            labelled[["trade_date", "adj_close", "fwd_max_drawdown", "label", "label_is_valid"]],
            left_on="observation_date",
            right_on="trade_date",
            direction="backward",
        )
        merged["label_is_valid"] = merged["label_is_valid"].fillna(0).astype(int)
        results.append(merged.rename(columns={"adj_close": "entry_price"}))

    return pd.concat(results, ignore_index=True) if results else observations


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
