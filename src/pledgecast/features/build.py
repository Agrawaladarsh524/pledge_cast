"""Panel orchestration - PLAN.md sec.7 step 4, sec.8.

Wires the layers together in the only order that is safe:

    pledge_state -> trajectory features   (features/pledge.py)
    grid         -> observation dates     (data/panel.py)   <- THE point-in-time rule
    prices       -> market features       (features/market.py)
    prices       -> forward label         (labels/drawdown.py)
    -> panel table

The point-in-time filter is applied to the pledge block BEFORE the market block
and the label are attached, because only the pledge block has a filing date. The
other two derive from prices, which are public the day they happen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.data import panel as panel_module
from pledgecast.exceptions import InsufficientDataError
from pledgecast.features import market, pledge
from pledgecast.labels import drawdown
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

PANEL_COLUMNS = [
    "symbol",
    "observation_date",
    "quarter_end",
    "promoter_holding_pct",
    "pledge_pct_promoter",
    "pledge_pct_equity",
    "pledge_chg_1q",
    "pledge_chg_2q",
    "pledge_accel",
    "consecutive_rising_q",
    "pledge_max_4q",
    "volatility_90d",
    "trailing_dd_60d",
    "return_90d",
    "rel_return_90d",
    "log_turnover_90d",
    "is_stale",
    "fwd_max_drawdown",
    "label",
    "label_is_valid",
]


def build_panel(
    pledge_state: pd.DataFrame,
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    symbols: list[str],
    quarters: list[str],
    settings,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the ML-ready panel. Returns ``(panel, diagnostics)``."""
    diagnostics: dict = {}

    if pledge_state.empty:
        raise InsufficientDataError("pledge_state is empty - run scripts/02_ingest_all.py")
    if prices.empty:
        raise InsufficientDataError("prices are empty - run scripts/02_ingest_all.py")

    trading_days = np.sort(benchmark["trade_date"].unique()) if not benchmark.empty else None

    # -- 1. trajectory features on the canonical quarter grid ---------------
    features = pledge.build_pledge_features(
        pledge_state,
        quarters,
        max_forward_fill_quarters=settings.features.max_forward_fill_quarters,
        rolling_max_quarters=settings.features.pledge_rolling_max_quarters,
    )
    diagnostics["pledge_feature_rows"] = len(features)

    # -- 2. observation grid + THE point-in-time rule -----------------------
    grid = panel_module.build_observation_grid(
        quarters,
        symbols,
        lag_days=settings.point_in_time.observation_lag_days,
        trading_days=trading_days,
    )
    diagnostics["grid_rows"] = len(grid)
    diagnostics["observation_dates"] = sorted(grid["observation_date"].unique())
    diagnostics["filing_lag"] = panel_module.filing_lag_report(
        features, grid, settings.point_in_time.observation_lag_days
    )

    frame = panel_module.apply_point_in_time_filter(grid, features, strict=True)

    # -- 3. market features -------------------------------------------------
    frame = market.compute_market_features(
        prices,
        benchmark,
        frame,
        volatility_window=settings.features.volatility_window_days,
        drawdown_window=settings.features.trailing_drawdown_window_days,
        return_window=settings.features.return_window_days,
        turnover_window=settings.features.turnover_window_days,
        trading_days_per_year=settings.features.trading_days_per_year,
    )

    # -- 4. forward label ---------------------------------------------------
    frame = drawdown.label_observations(
        frame,
        prices,
        horizon=settings.label.horizon_trading_days,
        threshold=settings.label.drawdown_threshold,
    )

    # -- 5. tidy ------------------------------------------------------------
    for column in PANEL_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame["is_stale"] = frame["is_stale"].fillna(0).astype(int)
    frame["label_is_valid"] = frame["label_is_valid"].fillna(0).astype(int)
    frame["consecutive_rising_q"] = frame["consecutive_rising_q"].astype("Int64")

    # A company with no price history at all cannot contribute anything.
    priceless = frame["volatility_90d"].isna() & frame["log_turnover_90d"].isna()
    if priceless.any():
        affected = sorted(frame.loc[priceless, "symbol"].unique())
        diagnostics["symbols_without_prices"] = affected
        logger.warning("%d symbols have no usable price history", len(affected))

    panel_frame = frame[PANEL_COLUMNS].sort_values(["observation_date", "symbol"])
    panel_frame = panel_frame.reset_index(drop=True)

    diagnostics["panel_rows"] = len(panel_frame)
    diagnostics["label_summary"] = drawdown.summarise(panel_frame)
    diagnostics["coverage"] = {
        column: float(panel_frame[column].notna().mean())
        for column in settings.features.all_features
    }
    return panel_frame, diagnostics


def exclude_insufficient_history(
    panel_frame: pd.DataFrame, settings
) -> tuple[pd.DataFrame, list[str]]:
    """Drop companies with too few observed quarters to form a trajectory.

    sec.10: "Company with < 3 quarters cannot produce pledge_accel ->
    InsufficientDataError, excluded with a logged reason."
    """
    minimum = settings.features.min_quarters_per_company
    observed = (
        panel_frame[panel_frame["pledge_pct_promoter"].notna()]
        .groupby("symbol")["quarter_end"]
        .nunique()
    )
    too_short = sorted(observed[observed < minimum].index)
    never_observed = sorted(set(panel_frame["symbol"]) - set(observed.index))
    excluded = sorted(set(too_short) | set(never_observed))

    if excluded:
        logger.warning(
            "excluding %d companies with fewer than %d observed pledge quarters: %s",
            len(excluded),
            minimum,
            excluded[:10],
        )
    return panel_frame[~panel_frame["symbol"].isin(excluded)].copy(), excluded


__all__ = ["PANEL_COLUMNS", "build_panel", "exclude_insufficient_history"]
