"""The economic backtest - PLAN.md sec.9.9.

    Each quarter, split the universe into risk quintiles by predicted
    probability, then compare realised event rates:

        Q5 (highest risk)  -> realised event rate  --.
                                                     |-> separation ratio
        Q1 (lowest risk)   -> realised event rate  --'

    "Compute the same table for the exp0_null model and show them side by side.
     If pledge-aware quintiles separate no better than volatility-only
     quintiles, the honest headline is 'pledge trajectory adds no incremental
     early warning once volatility is accounted for' - publish that."

    "Report per quarter, not just pooled - the spread shows whether the edge is
     stable or came from one lucky correction."

**Quintiles are cut WITHIN each observation date**, never pooled. Pooling would
rank a company from a calm quarter against one from a crashing quarter, and Q5
would fill up with "whatever quarter the market fell in" - the same market-timing
confound that makes pooled AUC misleading (sec.9.6). Cutting per date asks the
question a quarterly watchlist actually poses.

**Why the ratio is reported alongside a difference.** Q5/Q1 is the number sec.9.9
names, but it explodes when Q1 has zero events - which happens here, because 6 of
19 observation dates have an event rate under 5%. The difference Q5-Q1 and the
lift Q5/base are always defined, so all three are reported and the ratio is left
empty rather than infinite when its denominator is zero.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from pledgecast.exceptions import InsufficientDataError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def rank_groups(values: pd.Series, n_groups: int) -> pd.Series:
    """Cut ``values`` into ``n_groups`` equal-count bands, 1 = lowest.

    Ranks first with ``method='first'`` rather than calling ``pd.qcut`` on the
    raw probabilities. Predicted probabilities tie constantly - whole blocks of
    never-pledged companies receive identical scores - and ``qcut`` either
    raises on duplicate bin edges or silently returns fewer bins than asked
    for. Ranking breaks ties by position and always yields exactly ``n_groups``.
    """
    if values.empty:
        return pd.Series(dtype=int)
    ranks = values.rank(method="first")
    return np.ceil(ranks / len(values) * n_groups).clip(1, n_groups).astype(int)


def quintile_table(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """Realised event rate per (observation_date, quintile).

    ``predictions`` needs ``symbol``, ``observation_date``, ``probability``;
    ``labels`` needs ``symbol``, ``observation_date``, ``label``. The join is
    inner, so the embargo quarter drops out on its own - it has predictions but
    no realised outcome to compare them against.
    """
    merged = predictions.merge(
        labels[["symbol", "observation_date", "label"]],
        on=["symbol", "observation_date"],
        how="inner",
    ).dropna(subset=["probability", "label"])

    if merged.empty:
        raise InsufficientDataError("no predictions could be matched to a realised label")

    merged["quintile"] = merged.groupby("observation_date")["probability"].transform(
        lambda block: rank_groups(block, n_quintiles)
    )

    grouped = (
        merged.groupby(["observation_date", "quintile"])
        .agg(n_companies=("label", "size"), n_events=("label", "sum"))
        .reset_index()
    )
    grouped["n_events"] = grouped["n_events"].astype(int)
    grouped["event_rate"] = grouped["n_events"] / grouped["n_companies"]
    return grouped


def separation(table: pd.DataFrame, *, n_quintiles: int = 5) -> pd.DataFrame:
    """Per-date Q5 vs Q1 comparison - sec.9.9's "report per quarter"."""
    rows = []
    for date, block in table.groupby("observation_date"):
        rates = block.set_index("quintile")["event_rate"]
        top = float(rates.get(n_quintiles, np.nan))
        bottom = float(rates.get(1, np.nan))
        base = float(block["n_events"].sum() / block["n_companies"].sum())
        rows.append(
            {
                "observation_date": date,
                "base_rate": base,
                "q1_event_rate": bottom,
                "q5_event_rate": top,
                # Empty rather than infinite when nothing happened in Q1.
                "separation_ratio": (top / bottom) if bottom > 0 else None,
                "separation_diff": top - bottom,
                "lift_q5": (top / base) if base > 0 else None,
                "monotonic": _is_monotonic(rates, n_quintiles),
            }
        )
    return pd.DataFrame(rows).sort_values("observation_date").reset_index(drop=True)


def _is_monotonic(rates: pd.Series, n_quintiles: int) -> bool:
    """Does the event rate rise across every quintile step?

    A stronger claim than Q5 > Q1: it says the ranking is informative
    throughout, not just at the extremes.
    """
    ordered = [rates.get(q, np.nan) for q in range(1, n_quintiles + 1)]
    if any(pd.isna(v) for v in ordered):
        return False
    # pairwise, not zip(x, x[1:]) - the two arguments differ in length by one
    # by construction, so strict=False would silently skip the check.
    return all(b >= a for a, b in itertools.pairwise(ordered))


def pooled_separation(table: pd.DataFrame, *, n_quintiles: int = 5) -> dict:
    """One summary row, pooling companies across dates INSIDE each quintile.

    Quintile membership was still decided per date; only the counting is
    pooled. Reported second, after the per-date spread, because sec.9.9 warns
    that a pooled number hides whether the edge came from one lucky correction.
    """
    grouped = table.groupby("quintile").agg(
        n_companies=("n_companies", "sum"), n_events=("n_events", "sum")
    )
    grouped["event_rate"] = grouped["n_events"] / grouped["n_companies"]

    top = float(grouped["event_rate"].get(n_quintiles, np.nan))
    bottom = float(grouped["event_rate"].get(1, np.nan))
    base = float(grouped["n_events"].sum() / grouped["n_companies"].sum())

    per_date = separation(table, n_quintiles=n_quintiles)
    ratios = per_date["separation_ratio"].dropna()

    return {
        "quintiles": grouped.reset_index(),
        "base_rate": base,
        "q1_event_rate": bottom,
        "q5_event_rate": top,
        "separation_ratio": (top / bottom) if bottom > 0 else None,
        "separation_diff": top - bottom,
        "lift_q5": (top / base) if base > 0 else None,
        "monotonic": _is_monotonic(grouped["event_rate"], n_quintiles),
        "n_dates": int(table["observation_date"].nunique()),
        "n_dates_monotonic": int(per_date["monotonic"].sum()),
        "n_dates_q5_beats_q1": int((per_date["separation_diff"] > 0).sum()),
        "ratio_median": float(ratios.median()) if not ratios.empty else None,
        "ratio_min": float(ratios.min()) if not ratios.empty else None,
        "ratio_max": float(ratios.max()) if not ratios.empty else None,
        "n_dates_ratio_undefined": int(per_date["separation_ratio"].isna().sum()),
    }


def compare(headline: dict, null: dict, *, label_headline: str, label_null: str) -> pd.DataFrame:
    """Side-by-side summary - the sec.9.9 comparison, one table.

    If the null model separates as well as the pledge-aware one, that IS the
    result. The table is built to make that readable rather than to hide it.
    """
    fields = [
        ("base_rate", "base event rate"),
        ("q1_event_rate", "Q1 (safest) event rate"),
        ("q5_event_rate", "Q5 (riskiest) event rate"),
        ("separation_diff", "Q5 - Q1"),
        ("separation_ratio", "Q5 / Q1"),
        ("lift_q5", "Q5 / base (lift)"),
        ("ratio_median", "median per-date Q5/Q1"),
        ("n_dates_monotonic", "dates monotonic across quintiles"),
        ("n_dates_q5_beats_q1", "dates where Q5 > Q1"),
    ]
    return pd.DataFrame(
        [
            {
                "measure": text,
                label_headline: headline.get(key),
                label_null: null.get(key),
            }
            for key, text in fields
        ]
    )


def to_rows(table: pd.DataFrame, run_id: str) -> list[dict]:
    """Shape a quintile table for the ``backtest_results`` table."""
    return [
        {
            "run_id": run_id,
            "observation_date": row.observation_date,
            "quintile": int(row.quintile),
            "n_companies": int(row.n_companies),
            "n_events": int(row.n_events),
            "event_rate": float(row.event_rate),
        }
        for row in table.itertuples(index=False)
    ]


def run(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    run_id: str,
    n_quintiles: int = 5,
) -> dict:
    """Quintile table, per-date spread, pooled summary and DB rows in one call."""
    table = quintile_table(predictions, labels, n_quintiles=n_quintiles)
    summary = pooled_separation(table, n_quintiles=n_quintiles)
    logger.info(
        "backtest %s: Q5 %.1f%% vs Q1 %.1f%% (base %.1f%%), monotonic on %d of %d dates",
        run_id,
        100 * summary["q5_event_rate"],
        100 * summary["q1_event_rate"],
        100 * summary["base_rate"],
        summary["n_dates_monotonic"],
        summary["n_dates"],
    )
    return {
        "run_id": run_id,
        "table": table,
        "per_date": separation(table, n_quintiles=n_quintiles),
        "summary": summary,
        "rows": to_rows(table, run_id),
    }


__all__ = [
    "compare",
    "pooled_separation",
    "quintile_table",
    "rank_groups",
    "run",
    "separation",
    "to_rows",
]
