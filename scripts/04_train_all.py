"""04 - Train every experiment x model, walk-forward.

PLAN.md sec.16 Phase 4:

    "walk-forward training, all four experiments, three models. Run the
     label-shuffle test the moment the first model trains."

sec.9.8, non-negotiable:

    "Shuffle the labels, retrain, confirm AUC collapses to ~0.50. If it does
     not collapse, you have leakage - stop everything and fix it."

The banner leads with the headline delta rather than the best AUC, because
sec.2.3 says the result of this project IS the delta:

    expB_full.within_quarter_auc - exp0_null.within_quarter_auc

    python scripts/04_train_all.py
    python scripts/04_train_all.py --no-search      # use config hyperparameters
    python scripts/04_train_all.py --no-shuffle     # skip GATE 3 (debugging only)
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.logging_config import get_logger, setup_logging
from pledgecast.training import train

logger = get_logger(__name__)


def _fmt(value, spec: str = ".4f") -> str:
    return "n/a" if value is None or pd.isna(value) else format(value, spec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all PledgeCast models.")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip the fold-1 random search and use the hyperparameters in config.yaml.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Skip the label-shuffle gate. Debugging only - sec.9.8 calls it non-negotiable.",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)

    with get_connection(settings=settings) as conn:
        report = train.train_all(
            conn,
            settings,
            do_search=not args.no_search,
            run_shuffle_gate=not args.no_shuffle,
        )
        counts = repo.table_counts(conn)

    plan = report["plan"]
    winner = report["winner"]
    headline = report["headline"]
    gate = report["shuffle_gate"]

    print("=" * 96)
    print("PLEDGECAST - MODELS TRAINED")
    print("=" * 96)

    # ------------------------------------------------------------------ folds
    print(f"\n  WALK-FORWARD (sec.9.4): expanding window, {len(plan)} folds")
    print(
        f"    labelled dates      : {len(plan.labelled_dates)}  "
        f"{plan.labelled_dates[0]} .. {plan.labelled_dates[-1]}"
    )
    print(f"    embargoed (sec.9.4) : {plan.embargoed_dates or 'none'}")
    print(f"    min train quarters  : {settings.walkforward.min_train_quarters}")
    print(
        f"    folds disjoint      : "
        f"{'PASS' if report['fold_check']['passed'] else 'FAIL'} "
        f"({report['fold_check']['violations']} violations)"
    )
    print()
    print(report["fold_table"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ----------------------------------------------------------------- search
    search = report["search"]
    if not search["table"].empty:
        print(
            f"\n  HYPERPARAMETER SEARCH (sec.9.5): {len(search['table'])} points, "
            f"{settings.training.search_model}, fold {settings.training.search_fold_index} only"
        )
        print(f"    frozen for every experiment and every later fold: {search['overrides']}")
        scored = search["table"]["within_quarter_auc"].dropna()
        if not scored.empty:
            print(
                f"    search-fold AUC across the {len(scored)} points: "
                f"{scored.min():.4f} .. {scored.max():.4f} (median {scored.median():.4f})"
            )
            print(
                "    NOTE: fold "
                f"{settings.training.search_fold_index} is therefore in-sample for the "
                "searched model. The auc_ex_search_fold column below removes it."
            )
    else:
        print("\n  HYPERPARAMETER SEARCH: skipped - using config.yaml hyperparameters")

    # ------------------------------------------------------------- comparison
    print("\n  RESULTS (sec.9.6 - primary metric is within-quarter AUC)")
    comparison = report["comparison"].copy()
    display = comparison[
        [
            "experiment",
            "model",
            "n_features",
            "folds",
            "within_quarter_auc",
            "auc_std",
            "auc_min",
            "auc_max",
            "auc_ex_search_fold",
            "pooled_auc",
            "pr_auc",
            "brier_skill_score",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        "\n    per-fold spread is reported because sec.9.6 requires it: with 8 folds and "
        "\n    time-clustered positives a single pooled number overstates confidence."
    )

    # --------------------------------------------------------------- headline
    print(f"\n  {'=' * 92}")
    print("  HEADLINE RESULT (sec.2.3) - the point of the whole project")
    print(f"  {'=' * 92}")
    print(
        f"    {headline['metric']}:  {settings.headline.experiment} - "
        f"{settings.headline.baseline}, compared model for model"
    )
    print()
    print(headline["table"].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    if headline["searched_model"]:
        print(
            f"\n    tuned_on_treatment marks {headline['searched_model']}: its frozen "
            "hyperparameters were chosen\n"
            f"    on {settings.headline.experiment}'s feature set and then applied to the "
            f"{settings.headline.baseline} baseline too.\n"
            "    That handicaps the control, so its delta is not a clean comparison - which is\n"
            "    why the verdict below is the MEDIAN across models rather than any one of them."
        )

    print(
        f"\n    EVERY experiment against the same null "
        f"({settings.headline.baseline}), model by model:\n"
    )
    print(report["deltas"].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    delta = headline["median_delta"]
    if delta is not None:
        verdict = (
            "pledge trajectory ADDS incremental signal over volatility and size"
            if delta > 0
            else "pledge trajectory adds NO incremental early warning once volatility "
            "is accounted for"
        )
        print(
            f"\n    median delta across {headline['n_models']} models: {delta:+.4f}   "
            f"({headline['n_negative']} of {headline['n_models']} models <= 0)"
        )
        if headline["clean_median_delta"] is not None and headline["searched_model"]:
            print(
                f"    median over the {headline['n_models'] - 1} untuned models only: "
                f"{headline['clean_median_delta']:+.4f}"
            )
        print(f"    -> {verdict}")
        if delta <= 0:
            print(
                "\n    sec.2.2: a negative finding is a legitimate result. It is the honest\n"
                "    headline and sec.9.9 says to publish it, not to tune until it flips."
            )

    # ----------------------------------------------------------------- winner
    print(f"\n  SELECTED MODEL (sec.9.7) - within {settings.headline.experiment} only")
    print(f"    run_id              : {winner.run_id}")
    print(f"    model               : {winner.model_name}   ({len(winner.features)} features)")
    print(f"    within-quarter AUC  : {_fmt(winner.primary)}")
    print(f"    refit on            : {winner.n_train_rows:,} labelled rows (all folds)")
    print(f"    artifact            : models/{winner.run_id}.joblib   [is_active = 1]")
    if winner.skipped_folds:
        print(f"    folds skipped       : {winner.skipped_folds} (single-class training window)")

    # ----------------------------------------------------------------- gate 3
    print(f"\n  {'=' * 92}")
    print("  GATE 3 - LABEL-SHUFFLE TEST (sec.9.8, non-negotiable)")
    print(f"  {'=' * 92}")
    if gate.get("skipped"):
        print("    SKIPPED by --no-shuffle. This is not a pass.")
        passed = False
    else:
        print(
            f"    labels permuted WITHIN each observation date, {len(gate['scores'])} repeats, "
            f"full walk-forward re-run each time"
        )
        print(f"    shuffled AUC per repeat : {[f'{s:.4f}' for s in gate['scores']]}")
        print(f"    mean                    : {gate['mean']:.4f}   (expected 0.50 +/- {gate['tolerance']})")
        print(f"    GATE 3                  : {'PASSED' if gate['passed'] else 'FAILED'}")
        if not gate["passed"]:
            print(
                "\n    sec.9.8: the model still ranks companies after the labels were destroyed.\n"
                "    That is leakage. STOP - do not proceed to evaluation."
            )
        passed = bool(gate["passed"])

    print("\n  PERSISTED")
    print(f"    model_runs          : {counts['model_runs']:,}")
    print(f"    model_metrics       : {counts['model_metrics']:,}")
    print(f"    predictions         : {counts['predictions']:,}  (source='backtest', all runs)")

    print("\n  next: python scripts/05_evaluate_and_explain.py")
    print("=" * 96)
    return 0 if passed and report["fold_check"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
