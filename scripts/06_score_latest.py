"""06 - Score the most recent quarter.

This is the watchlist. It scores the observation date that has features but no
label - the embargo quarter of PLAN.md sec.9.4:

    "the final quarter can be featured but never labelled - its label needs 60
     trading days of future prices that don't exist yet."

Which is exactly what makes it the only genuinely forward-looking prediction the
system produces. Every other date has already been scored out-of-fold during
walk-forward training.

Scoring runs through ``inference/service.py`` (sec.7.2), the same path the API
and dashboard use, so the numbers on the scanner cannot disagree with the
numbers from ``POST /predict``.

    python scripts/06_score_latest.py
    python scripts/06_score_latest.py --date 2026-04-30
    python scripts/06_score_latest.py --top 30
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.inference.service import PredictionService
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the latest observation date.")
    parser.add_argument("--date", default=None, help="Observation date; defaults to the latest.")
    parser.add_argument("--top", type=int, default=None, help="How many rows to print.")
    parser.add_argument("--no-persist", action="store_true", help="Print without writing.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    top_n = args.top or settings.dashboard.default_top_n

    with get_connection(settings=settings) as conn:
        service = PredictionService(settings, conn)
        info = service.model_info(conn)
        scored = service.score_date(
            conn, args.date, persist=not args.no_persist, source="backtest"
        )
        counts = repo.table_counts(conn)

    date = scored["observation_date"].iloc[0]
    unlabelled = int((scored["label_is_valid"] == 0).sum())

    print("=" * 96)
    print("PLEDGECAST - LATEST SCORES")
    print("=" * 96)
    print(f"\n  model                 : {info['run_id']}  ({info['model_name']})")
    print(f"  observation date      : {date}")
    print(f"  companies scored      : {len(scored):,}  ({unlabelled} with no realised outcome yet)")
    print(
        f"  walk-forward AUC      : "
        f"{info['metrics'].get('within_quarter_auc', float('nan')):.4f} (within-quarter, sec.9.6)"
    )

    print(f"\n  TOP {top_n} BY PREDICTED RISK")
    print(
        f"    {'#':>3} {'symbol':<14}{'prob':>7}{'decile':>8}{'band':>10}"
        f"{'pledge%prom':>13}{'vol90d':>9}"
    )
    print("    " + "-" * 66)
    for rank, row in enumerate(scored.head(top_n).itertuples(index=False), start=1):
        pledge = "n/a" if row.pledge_pct_promoter is None else f"{row.pledge_pct_promoter:.1f}"
        vol = "n/a" if row.volatility_90d is None else f"{row.volatility_90d:.2f}"
        print(
            f"    {rank:>3} {row.symbol:<14}{row.probability:>7.3f}{row.risk_decile:>8}"
            f"{row.risk_band:>10}{pledge:>13}{vol:>9}"
        )

    flagged = scored[scored["warnings"].map(len) > 0]
    if not flagged.empty:
        print(f"\n  WARNINGS (sec.13.1) - {len(flagged)} of {len(scored)} companies")
        for row in flagged.head(5).itertuples(index=False):
            print(f"    {row.symbol:<14}{row.warnings[0][:74]}")
        if len(flagged) > 5:
            print(f"    ... and {len(flagged) - 5} more")

    band_counts = scored["risk_band"].value_counts()
    print("\n  RISK BANDS")
    for band in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        n = int(band_counts.get(band, 0))
        print(f"    {band:<10}{n:>5}  {'#' * int(40 * n / len(scored))}")

    print(f"\n  predictions in DB     : {counts['predictions']:,}")
    print("\n  next: make api    |    make app")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
