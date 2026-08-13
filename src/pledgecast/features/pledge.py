"""The 8 pledge features - PLAN.md sec.9.1, sec.3.2.

These features **are** the GRU replacement. sec.3.2:

    "The features pledge_chg_1q, pledge_chg_2q, pledge_accel,
     consecutive_rising_q and pledge_max_4q ARE the temporal signal, explicitly
     encoded. A GRU would spend its capacity rediscovering them from 8
     timesteps."

    promoter_holding_pct   promoter shares / total shares
    pledge_pct_promoter    pledged / promoter holding      <- primary level
    pledge_pct_equity      pledged / total equity
    pledge_chg_1q          QoQ change in pledge_pct_promoter
    pledge_chg_2q          2-quarter change
    pledge_accel           chg_1q - previous chg_1q
    consecutive_rising_q   streak length of rising quarters
    pledge_max_4q          rolling 4-quarter maximum

Every derived feature is computed on a **complete quarterly grid** per company.
Reindexing matters: if a company skips a filing, a naive ``.diff()`` over the
rows present would compare Q1 against Q3 and label it a one-quarter change,
overstating the trajectory exactly where the data is weakest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

LEVEL_COLUMN = "pledge_pct_promoter"


def _consecutive_rising(values: pd.Series) -> pd.Series:
    """Length of the current run of strictly rising quarters.

    Resets to 0 on any non-increase. A NaN breaks the run rather than extending
    it, because an unknown quarter is not evidence of a rise.
    """
    rising = values.diff() > 0
    unknown = values.isna() | values.shift().isna()

    streak = np.zeros(len(values), dtype=float)
    run = 0
    for i, (is_rising, is_unknown) in enumerate(zip(rising, unknown, strict=True)):
        if is_unknown:
            run = 0
            streak[i] = np.nan
            continue
        run = run + 1 if is_rising else 0
        streak[i] = run
    return pd.Series(streak, index=values.index)


def build_pledge_features(
    pledge_state: pd.DataFrame,
    quarters: list[str],
    *,
    max_forward_fill_quarters: int = 1,
    rolling_max_quarters: int = 4,
) -> pd.DataFrame:
    """Per-company trajectory features on the canonical quarter grid.

    ``pledge_state`` is the parsed table; ``quarters`` is the full ordered list
    of quarter ends in the study window.

    Forward-fill is capped at ``max_forward_fill_quarters`` and every carried
    row is flagged ``is_stale`` (sec.10: "Forward-fill pledge state max 1
    quarter, set is_stale=1, drop beyond").
    """
    if pledge_state.empty:
        return pd.DataFrame()

    grid = pd.Index(quarters, name="quarter_end")
    frames: list[pd.DataFrame] = []

    for symbol, group in pledge_state.groupby("symbol"):
        company = (
            group.drop_duplicates(subset="quarter_end", keep="last")
            .set_index("quarter_end")
            .reindex(grid)
        )
        company["symbol"] = symbol

        # UNAVAILABLE means the filing did not report encumbrance. That is not
        # zero, so it is blanked before any arithmetic touches it.
        unavailable = company["pledge_status"].eq("UNAVAILABLE")
        for column in ("pledge_pct_promoter", "pledge_pct_equity"):
            company.loc[unavailable, column] = np.nan

        observed = company["pledge_status"].notna()

        # --- capped forward fill ------------------------------------------
        carry_columns = [
            "promoter_holding_pct",
            "pledge_pct_promoter",
            "pledge_pct_equity",
            "submission_date",
            "pledge_status",
        ]
        filled = company[carry_columns].ffill(limit=max_forward_fill_quarters)
        company[carry_columns] = filled
        company["is_stale"] = (~observed & company["pledge_status"].notna()).astype(int)

        # --- trajectory (sec.9.1) ------------------------------------------
        level = company[LEVEL_COLUMN].astype(float)
        company["pledge_chg_1q"] = level.diff(1)
        company["pledge_chg_2q"] = level.diff(2)
        company["pledge_accel"] = company["pledge_chg_1q"].diff(1)
        company["consecutive_rising_q"] = _consecutive_rising(level)
        company["pledge_max_4q"] = level.rolling(rolling_max_quarters, min_periods=1).max()

        frames.append(company.reset_index())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["symbol", "quarter_end"]).reset_index(drop=True)


__all__ = ["LEVEL_COLUMN", "build_pledge_features"]
