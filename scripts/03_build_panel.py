"""03 - Build the point-in-time panel.

PLAN.md sec.16 Phase 3:

    "point-in-time join (quarter_end + 30d), 8 pledge + 5 market features,
     forward-drawdown label. Then immediately run the leakage tests. Confirm
     ~6,000 rows and ~25% event rate. If the rate is wildly off, the label is
     wrong - stop and fix."

sec.17 names the one exploration step that must not be skipped: checking the
drawdown distribution BEFORE the -15% threshold is locked. That is done here
(sec.9.2), not in a notebook.

    python scripts/03_build_panel.py
    python scripts/03_build_panel.py --no-strict     # report instead of raising
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.data import validate
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.evaluation import leakage
from pledgecast.features import build
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _histogram(values, lo: float = -1.0, hi: float = 0.6, bins: int = 12) -> list[str]:
    """A text histogram - sec.17 wants this check, not a notebook."""
    import numpy as np

    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    peak = max(counts) or 1
    return [
        f"    {edges[i]:>6.2f}..{edges[i + 1]:>6.2f}  {'#' * int(40 * counts[i] / peak):<40} "
        f"{counts[i]:>6,}"
        for i in range(len(counts))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the PledgeCast panel.")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Report validation and leakage problems instead of raising on them.",
    )
    args = parser.parse_args()
    strict = not args.no_strict

    settings = get_settings()
    setup_logging(settings)

    with get_connection(settings=settings) as conn:
        symbols = repo.get_universe_symbols(conn)
        if not symbols:
            logger.error("no companies in the universe - run scripts/01_build_universe.py")
            return 1

        pledge_state = repo.load_pledge_state(conn)
        prices = repo.load_prices(conn)
        benchmark = repo.load_benchmark(conn)
        pledge_events = repo.load_pledge_events(conn)
        quarters = settings.window.quarter_ends()

        panel_frame, diagnostics = build.build_panel(
            pledge_state, prices, benchmark, symbols, quarters, settings, pledge_events
        )
        panel_frame, excluded = build.exclude_insufficient_history(panel_frame, settings)

        validation = validate.validate_panel(panel_frame, settings, strict=strict)
        leak_results = leakage.run_all(
            panel_frame,
            pledge_state,
            prices,
            horizon=settings.label.horizon_trading_days,
            events=pledge_events,
            event_lag_days=settings.features.event_disclosure_lag_days,
            event_window_days=settings.features.event_window_days,
            min_event_pct_equity=settings.features.min_event_pct_equity,
            strict=strict,
        )

        repo.replace_panel(conn, panel_frame)
        counts = repo.table_counts(conn)

    summary = diagnostics["label_summary"]
    lag = diagnostics["filing_lag"]

    # ------------------------------------------------------------------ report
    print("=" * 80)
    print("PLEDGECAST - PANEL BUILT")
    print("=" * 80)
    print(
        f"  companies             : {len(symbols)}  ({len(excluded)} excluded, too little history)"
    )
    print(f"  quarters              : {len(quarters)}  {quarters[0]} .. {quarters[-1]}")
    print(f"  observation dates     : {len(diagnostics['observation_dates'])}")
    print(f"  panel rows            : {counts['panel']:,}")

    print(
        f"\n  POINT-IN-TIME RULE (sec.9.3): quarter_end + "
        f"{settings.point_in_time.observation_lag_days}d, rolled to next trading day"
    )
    if lag.get("n"):
        print(f"    filings matched     : {lag['n']:,}")
        print(
            f"    filing lag (days)   : min={lag['lag_min']} median={lag['lag_median']:.0f} "
            f"p99={lag['lag_p99']:.0f} max={lag['lag_max']}"
        )
        print(
            f"    filed after cutoff  : {lag['filed_after_cutoff']} "
            f"({lag['pct_lost_to_cutoff']:.2f}% of filings)"
        )

    print(
        "\n  LABEL (sec.9.2): worst decline from entry over "
        f"{settings.label.horizon_trading_days} trading days"
    )
    print(f"    labelled rows       : {summary['n_valid']:,} of {summary['n']:,}")
    if summary.get("event_rate") is not None:
        rate = summary["event_rate"]
        expected = settings.label.expected_event_rate
        delta = abs(rate - expected)
        verdict = "OK" if delta <= settings.label.event_rate_tolerance else "OUT OF TOLERANCE"
        print(f"    events (<= {settings.label.drawdown_threshold:.0%}) : {summary['n_events']:,}")
        print(
            f"    EVENT RATE          : {rate:.1%}   (sec.1.1 measured {expected:.1%})  -> {verdict}"
        )
        print(
            f"    drawdown median     : {summary['drawdown_median']:.1%}   "
            f"p05={summary['drawdown_p05']:.1%}  p25={summary['drawdown_p25']:.1%}"
        )

        print("\n  drawdown distribution (sec.9.2 - confirm BEFORE locking the threshold)")
        import numpy as np

        valid = panel_frame[panel_frame["label_is_valid"] == 1]["fwd_max_drawdown"].astype(float)
        for line in _histogram(valid.to_numpy(dtype=float)):
            print(line)
        print(
            f"    threshold {settings.label.drawdown_threshold:.2f} sits at the "
            f"{100 * float(np.mean(valid <= settings.label.drawdown_threshold)):.1f}th pct"
        )

    print("\n  FEATURE COVERAGE (fraction non-null)")
    for name, coverage in diagnostics["coverage"].items():
        flag = "" if coverage > 0.5 else "   <-- sparse"
        print(f"    {name:<24} {coverage:6.1%}{flag}")

    print("\n  LEAKAGE CHECKS (sec.9.8)")
    for result in leak_results:
        print(
            f"    [{'PASS' if result['passed'] else 'FAIL'}] {result['check']} "
            f"({result.get('violations', 0)} violations of {result.get('rows_checked', 0)} checked)"
        )

    print(
        f"\n  VALIDATION            : {'passed' if validation['passed'] else validation['issues']}"
    )

    all_passed = validation["passed"] and all(r["passed"] for r in leak_results)
    print(f"\n  GATE 2                : {'PASSED' if all_passed else 'FAILED'}")
    print("  next: python scripts/04_train_all.py")
    print("=" * 80)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
