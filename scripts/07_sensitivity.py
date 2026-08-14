"""07 - Does the Reg 31 result depend on the knobs I chose?

Phase 10b. Three numbers in the event block were judgement calls: the 90-day
window, the 365-day invocation window and the 0.01% materiality threshold. A
null result at one setting invites the obvious objection - "you picked the wrong
window" - and the only answer that settles it is the whole sweep.

This script also prints the univariate table: every feature's own within-quarter
AUC against the label, with no model, no hyperparameters and no folds. It is the
most direct evidence in the project about which features separate and which do
not, and on the pledged stratum it surfaces something the pooled experiments
average away.

NOTHING here feeds back into config.yaml. Sweeping a parameter and keeping the
setting that scored best is how a null gets tuned into a finding; sec.9.9
forbids it. The sweep exists to show the result is flat.

    python scripts/07_sensitivity.py
    python scripts/07_sensitivity.py --population pledged
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.data.population import apply_population, describe_populations
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.evaluation import sensitivity
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def _fmt(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity sweeps for the Reg 31 block.")
    parser.add_argument(
        "--population",
        default="all",
        help="Which stratum to sweep on (default: all). See config.yaml populations.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    with get_connection(settings=settings) as conn:
        panel = repo.load_panel(conn, valid_only=False)
        events = repo.load_pledge_events(conn)

    if panel.empty:
        logger.error("panel is empty - run scripts/03_build_panel.py first")
        return 1

    strata = describe_populations(panel, settings)
    panel, _ = apply_population(panel, args.population, settings)
    labelled = panel[panel["label_is_valid"] == 1].reset_index(drop=True)

    print("=" * 96)
    print(f"PLEDGECAST - SENSITIVITY SWEEP  [population: {args.population}]")
    print("=" * 96)
    print(f"  labelled rows : {len(labelled):,}")
    print(f"  companies     : {labelled['symbol'].nunique()}")
    print(f"  crash rate    : {labelled['label'].mean():.2%}")
    print(f"  dates         : {labelled['observation_date'].nunique()}")

    print("\n  AVAILABLE STRATA")
    print(strata.to_string(index=False, float_format=lambda v: f"{v:.3f}", max_colwidth=44))

    # ------------------------------------------------------------- window sweep
    print(f"\n  {'=' * 92}")
    print("  1. WINDOW SWEEP - univariate within-quarter AUC at each lookback")
    print(f"  {'=' * 92}")
    print("     configured window is marked; nothing here changes it.\n")
    windows = sensitivity.window_sweep(events, labelled, settings)
    print(windows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    spread = windows[["count", "net", "created"]].stack().dropna()
    if not spread.empty:
        print(
            f"\n     AUC across every window and feature: {spread.min():.4f} .. {spread.max():.4f}"
        )
        if spread.max() - 0.5 < 0.05 and 0.5 - spread.min() < 0.05:
            print(
                "     Every value sits within 0.05 of a coin flip while coverage more than\n"
                "     doubles. The null is not an artefact of the 90-day choice."
            )

    # -------------------------------------------------------- materiality sweep
    print(f"\n  {'=' * 92}")
    print("  2. MATERIALITY SWEEP - what the 0.01% filter buys")
    print(f"  {'=' * 92}")
    print(
        "     The 0.00 row is the unfiltered feature: the clearing and custodial\n"
        "     disclosures come back, coverage jumps, and the AUC does not move.\n"
        "     That is the signature of a feature measuring filing volume, not pledging.\n"
    )
    thresholds = sensitivity.materiality_sweep(events, labelled, settings)
    print(thresholds.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---------------------------------------------------------- univariate table
    print(f"\n  {'=' * 92}")
    print("  3. UNIVARIATE SEPARATION - each feature alone, no model involved")
    print(f"  {'=' * 92}")
    print(
        "     auc_if_inverted is the mirror: 0.44 and 0.56 are equally strong orderings,\n"
        "     they just point opposite ways. `strength` is the distance from 0.5.\n"
    )
    features = [
        f for f in settings.features.all_features if f in labelled.columns
    ]
    table = sensitivity.univariate_table(labelled, features, settings)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not table.empty:
        best = table.iloc[0]
        print(
            f"\n     strongest single feature: {best['feature']} "
            f"(AUC {_fmt(best['auc'])}, strength {_fmt(best['strength'])})"
        )
        inverted = table[table["auc"] < 0.5].head(3)
        if not inverted.empty:
            print(
                "\n     features that separate the OTHER way (higher value -> fewer crashes):"
            )
            for _, row in inverted.iterrows():
                print(
                    f"       {row['feature']:<26} {_fmt(row['auc'])}  "
                    f"(as {_fmt(row['auc_if_inverted'])} inverted)"
                )
            print(
                "\n     An inverse relationship is a finding, not a failure - but read it\n"
                "     against the interval in 04_train_all.py's DETECTABILITY table before\n"
                "     calling it real. A univariate AUC here has no confidence interval and\n"
                "     no out-of-sample discipline; it points, it does not prove."
            )

    print("\n  next: python scripts/04_train_all.py   (for the intervals and ceilings)")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
