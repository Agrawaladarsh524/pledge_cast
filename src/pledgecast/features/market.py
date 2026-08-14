"""The 5 market features - PLAN.md sec.9.1.

    volatility_90d     annualised std of daily log returns
    trailing_dd_60d    realised max drawdown over the PRIOR 60 days
    return_90d         90-day return
    rel_return_90d     return_90d - NIFTY 90-day return
    log_turnover_90d   log(median daily price x volume)   <- size/liquidity proxy

Two of these carry the whole argument of the project. sec.2.1: pledged companies
tend to be leveraged smallcaps, so a model given only ``volatility_90d`` and
``log_turnover_90d`` separates the target almost perfectly while learning
nothing about pledging. They are the ``exp0_null`` feature set, and the headline
result is what the pledge block adds **over** them.

sec.9.1 on the size proxy: "``log_turnover_90d`` replaces market cap
deliberately: it needs only price data (avoiding a whole shares-outstanding
dependency) and is the better liquidity control anyway."

Every window is backward-looking and strictly ends at the observation date, so
none of these can see the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def _as_of_positions(trade_dates: np.ndarray, observation_dates: np.ndarray) -> np.ndarray:
    """Index of the last trading day at or before each observation date.

    ``-1`` where the observation predates the price history.
    """
    return np.searchsorted(trade_dates, observation_dates, side="right") - 1


def _sample_at(series: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Read ``series`` at each position, NaN where the position is out of range."""
    out = np.full(len(position), np.nan)
    valid = position >= 0
    out[valid] = series[position[valid]]
    return out


def compute_market_features(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    volatility_window: int,
    drawdown_window: int,
    return_window: int,
    turnover_window: int,
    trading_days_per_year: int,
) -> pd.DataFrame:
    """Compute the 5 price-derived features for every (symbol, observation_date).

    Returns ``observations`` with the feature columns attached.
    """
    required = ["symbol", "observation_date"]
    missing = [c for c in required if c not in observations.columns]
    if missing:
        raise ValueError(f"observations is missing {missing}")

    if observations.empty or prices.empty:
        return observations.assign(
            volatility_90d=np.nan,
            trailing_dd_60d=np.nan,
            return_90d=np.nan,
            rel_return_90d=np.nan,
            log_turnover_90d=np.nan,
        )

    # --- benchmark return, computed once ----------------------------------
    bench = benchmark.sort_values("trade_date").reset_index(drop=True)
    bench_dates = bench["trade_date"].to_numpy()
    bench_close = bench["adj_close"].to_numpy(dtype=float)
    bench_return = np.full(len(bench_close), np.nan)
    if len(bench_close) > return_window:
        bench_return[return_window:] = (
            bench_close[return_window:] / bench_close[:-return_window] - 1.0
        )

    # dict(iter(...)) not dict(...): a DataFrameGroupBy has a `keys` ATTRIBUTE
    # holding the grouping column name, so dict() mistakes it for a mapping and
    # raises "'str' object is not callable". iter() yields the (name, group)
    # pairs directly.
    price_groups = dict(iter(prices.groupby("symbol")))
    output: list[pd.DataFrame] = []

    for symbol, group in observations.groupby("symbol"):
        block = group.sort_values("observation_date").copy()
        history = price_groups.get(symbol)

        if history is None or len(history) < 2:
            for column in (
                "volatility_90d",
                "trailing_dd_60d",
                "return_90d",
                "rel_return_90d",
                "log_turnover_90d",
            ):
                block[column] = np.nan
            output.append(block)
            continue

        history = history.sort_values("trade_date").reset_index(drop=True)
        dates = history["trade_date"].to_numpy()
        close = history["adj_close"].to_numpy(dtype=float)
        volume = history["volume"].to_numpy(dtype=float)

        # --- rolling series over the full price history -------------------
        log_return = np.full(len(close), np.nan)
        log_return[1:] = np.log(close[1:] / close[:-1])
        volatility = (
            pd.Series(log_return)
            .rolling(volatility_window, min_periods=volatility_window // 2)
            .std()
            * np.sqrt(trading_days_per_year)
        ).to_numpy()

        running_max = pd.Series(close).rolling(drawdown_window, min_periods=2).max().to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            trailing_dd = np.where(running_max > 0, close / running_max - 1.0, np.nan)

        simple_return = np.full(len(close), np.nan)
        if len(close) > return_window:
            simple_return[return_window:] = close[return_window:] / close[:-return_window] - 1.0

        turnover = close * np.nan_to_num(volume, nan=0.0)
        median_turnover = (
            pd.Series(turnover).rolling(turnover_window, min_periods=turnover_window // 2).median()
        ).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_turnover = np.where(median_turnover > 0, np.log(median_turnover), np.nan)

        # --- sample as of each observation date ---------------------------
        observed = block["observation_date"].to_numpy()
        position = _as_of_positions(dates, observed)

        block["volatility_90d"] = _sample_at(volatility, position)
        block["trailing_dd_60d"] = _sample_at(trailing_dd, position)
        block["return_90d"] = _sample_at(simple_return, position)
        block["log_turnover_90d"] = _sample_at(log_turnover, position)

        bench_position = _as_of_positions(bench_dates, observed)
        bench_valid = bench_position >= 0
        bench_sampled = np.full(len(bench_position), np.nan)
        bench_sampled[bench_valid] = bench_return[bench_position[bench_valid]]
        block["rel_return_90d"] = block["return_90d"].to_numpy() - bench_sampled

        output.append(block)

    return pd.concat(output, ignore_index=True) if output else observations


def blank_features_spanning_breaks(
    frame: pd.DataFrame,
    prices: pd.DataFrame,
    breaks: dict[str, list[str]],
    *,
    volatility_window: int,
    drawdown_window: int,
    return_window: int,
    turnover_window: int,
) -> tuple[pd.DataFrame, dict]:
    """Blank market features whose lookback window contains a corporate action.

    sec.10 already requires that a demerger inside the FORWARD window not become
    a fake label. The same break inside the BACKWARD window is the same error
    pointing the other way, and it was live here: VEDL's demerger falls on
    2026-04-30, which is an observation date, so its ``volatility_90d`` read 180%
    and ``return_90d`` read -52% from a corporate action rather than from
    anything the market did. The model then ranked it the single riskiest
    company in the study.

    Each feature is blanked against its OWN window rather than blanking all five
    together - a break 80 trading days back corrupts ``volatility_90d`` but has
    long since left ``trailing_dd_60d``.

    Returns ``(frame, report)``. Features are blanked, not dropped: the row's
    pledge features and label remain perfectly valid.
    """
    if not breaks or frame.empty:
        return frame, {"symbols": 0, "rows_blanked": 0, "cells_blanked": 0}

    windows = {
        "volatility_90d": volatility_window,
        "trailing_dd_60d": drawdown_window,
        "return_90d": return_window,
        "rel_return_90d": return_window,
        "log_turnover_90d": turnover_window,
    }

    price_groups = dict(iter(prices.groupby("symbol")))
    affected: set[int] = set()
    cells = 0

    for symbol, break_dates in breaks.items():
        history = price_groups.get(symbol)
        rows = frame.index[frame["symbol"] == symbol]
        if history is None or len(rows) == 0:
            continue

        dates = np.sort(history["trade_date"].to_numpy())
        observed = frame.loc[rows, "observation_date"].to_numpy()
        position = _as_of_positions(dates, observed)
        break_positions = _as_of_positions(dates, np.array(sorted(break_dates)))

        for break_position in break_positions:
            if break_position < 0:
                continue
            distance = position - break_position
            for column, window in windows.items():
                # The break sits inside this row's lookback window when it is at
                # or before the observation date and no further back than the
                # window reaches.
                hit = (distance >= 0) & (distance <= window)
                if not hit.any():
                    continue
                targets = rows[hit]
                cells += int(frame.loc[targets, column].notna().sum())
                frame.loc[targets, column] = np.nan
                affected.update(targets.tolist())

    report = {
        "symbols": len(breaks),
        "rows_blanked": len(affected),
        "cells_blanked": cells,
        "detail": {s: sorted(d) for s, d in breaks.items()},
    }
    if affected:
        logger.warning(
            "blanked %d market-feature cells across %d rows: the lookback window spans a "
            "corporate action for %s",
            cells,
            len(affected),
            sorted(breaks),
        )
    return frame, report


__all__ = ["blank_features_spanning_breaks", "compute_market_features"]
