"""05 - Economic backtest + SHAP explanations.

PLAN.md sec.16 Phase 5:

    "quintile backtest against the null, SHAP global + local, the figures the
     README needs."

sec.9.9 is the reason this script exists in the shape it does:

    "Compute the same table for the exp0_null model and show them side by side.
     If pledge-aware quintiles separate no better than volatility-only
     quintiles, the honest headline is 'pledge trajectory adds no incremental
     early warning once volatility is accounted for' - publish that."

sec.11.1 adds the second obligation - the global SHAP plot IS the confound
audit, and it gets shown "either way".

    python scripts/05_evaluate_and_explain.py
    python scripts/05_evaluate_and_explain.py --no-figures
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

import _bootstrap  # noqa: F401  - must precede pledgecast/config imports
from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.evaluation import backtest
from pledgecast.explain import shap_runner
from pledgecast.logging_config import get_logger, setup_logging
from pledgecast.models import registry
from pledgecast.models.preprocessing import prepare_matrix

logger = get_logger(__name__)


def _pct(value) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{100 * value:5.1f}%"


def find_run(runs: pd.DataFrame, model_name: str, experiment: str) -> dict | None:
    """The most recent run for one (model, experiment) pair."""
    match = runs[(runs["model_name"] == model_name) & (runs["experiment"] == experiment)]
    return None if match.empty else match.iloc[0].to_dict()


def calibration_figure(y_true, y_score, path, n_bins: int):
    """Are the probabilities honest? (sec.12 - matplotlib, README)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    # Quantile bins, not uniform: the predictions cluster hard below 0.4, and
    # uniform bins would leave the top of the range holding a handful of rows.
    observed, predicted = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfectly calibrated")
    plt.plot(predicted, observed, "o-", label="walk-forward out-of-fold")
    plt.xlabel("mean predicted probability")
    plt.ylabel("observed event rate")
    plt.title("Calibration - out-of-fold predictions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    logger.info("wrote %s", path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest and explain the trained models.")
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG generation.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings)
    figures_dir = settings.paths.figures_dir

    with get_connection(settings=settings) as conn:
        payload, active = registry.load_active_model(conn, settings)
        runs = repo.load_model_runs(conn)
        panel = repo.load_panel(conn, valid_only=True)

        # ---------------------------------------------------------- backtest
        null_run = find_run(runs, active["model_name"], settings.headline.baseline)
        if null_run is None:
            logger.error("no %s run for the active model", settings.headline.baseline)
            return 1

        labels = panel[["symbol", "observation_date", "label"]]
        n_quintiles = settings.evaluation.n_quintiles

        headline_bt = backtest.run(
            repo.load_predictions(conn, run_id=active["run_id"], source="backtest"),
            labels,
            run_id=active["run_id"],
            n_quintiles=n_quintiles,
        )
        null_bt = backtest.run(
            repo.load_predictions(conn, run_id=null_run["run_id"], source="backtest"),
            labels,
            run_id=null_run["run_id"],
            n_quintiles=n_quintiles,
        )
        repo.save_backtest_results(conn, headline_bt["rows"] + null_bt["rows"])

        # -------------------------------------------------------------- SHAP
        # sec.11.1's confound audit runs on `explain.model_for_shap`; the
        # per-prediction explanations run on the model actually being served.
        audit_run = find_run(runs, settings.explain.model_for_shap, settings.headline.experiment)
        audit_payload = (
            registry.load_model(audit_run["run_id"], settings)
            if audit_run and audit_run.get("artifact_path")
            else payload
        )
        audit_features = audit_payload["feature_list"]
        audit = shap_runner.explain(
            audit_payload["pipeline"], prepare_matrix(panel, audit_features), audit_features
        )
        confound = shap_runner.confound_audit(
            audit["importance"], settings.features.pledge_features
        )

        # Explanations are persisted for the most recent quarter that has a
        # realised outcome - that is what the dashboard's investigation page
        # opens on, and explaining all 3,296 out-of-fold rows would store four
        # times the data for no extra insight.
        latest_date = panel["observation_date"].max()
        latest = panel[panel["observation_date"] == latest_date].reset_index(drop=True)
        served = shap_runner.explain(
            payload["pipeline"],
            prepare_matrix(latest, payload["feature_list"]),
            payload["feature_list"],
        )
        records = shap_runner.explanation_rows(
            served["values"], served["raw"], served["names"], payload["feature_list"]
        )

        predictions = repo.load_predictions(
            conn, run_id=active["run_id"], observation_date=latest_date
        )
        by_symbol = dict(zip(latest["symbol"], records, strict=True))
        stored = 0
        for row in predictions.itertuples(index=False):
            if row.symbol in by_symbol:
                stored += repo.save_explanations(conn, int(row.prediction_id), by_symbol[row.symbol])

        # Worked example for the banner (sec.11.1 use 3).
        top = predictions.sort_values("probability", ascending=False).iloc[0]
        sentence = shap_runner.summarise(
            by_symbol[top.symbol],
            float(top.probability),
            decile=int(top.risk_decile),
            band=settings.evaluation.band_for(float(top.probability)),
            top_n=settings.explain.top_n_features,
        )

        oof = repo.load_predictions(conn, run_id=active["run_id"], source="backtest").merge(
            labels, on=["symbol", "observation_date"], how="inner"
        )
        counts = repo.table_counts(conn)

    # ------------------------------------------------------------- figures
    written = []
    if not args.no_figures:
        written.append(
            shap_runner.beeswarm(
                audit["values"],
                audit["matrix"],
                audit["names"],
                figures_dir / "shap_beeswarm.png",
                settings.explain.beeswarm_max_display,
            )
        )
        written.append(
            calibration_figure(
                oof["label"],
                oof["probability"],
                figures_dir / "calibration.png",
                settings.evaluation.n_deciles,
            )
        )

    # -------------------------------------------------------------- report
    hs, ns = headline_bt["summary"], null_bt["summary"]

    print("=" * 96)
    print("PLEDGECAST - BACKTEST + EXPLANATIONS")
    print("=" * 96)
    print(f"\n  served model          : {active['run_id']}  ({active['model_name']})")
    print(f"  null comparison       : {null_run['run_id']}  ({settings.headline.baseline})")

    print(f"\n  {'=' * 92}")
    print(f"  ECONOMIC BACKTEST (sec.9.9) - {n_quintiles} risk quintiles, cut WITHIN each quarter")
    print(f"  {'=' * 92}")
    print("\n  pooled event rate by quintile (1 = model says safest)")
    print(f"    {'quintile':<10}{settings.headline.experiment:>16}{settings.headline.baseline:>16}")
    for q in range(1, n_quintiles + 1):
        a = hs["quintiles"].set_index("quintile")["event_rate"].get(q)
        b = ns["quintiles"].set_index("quintile")["event_rate"].get(q)
        print(f"    Q{q:<9}{_pct(a):>16}{_pct(b):>16}")

    print()
    print(
        backtest.compare(
            hs, ns, label_headline=settings.headline.experiment, label_null=settings.headline.baseline
        ).to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )

    print("\n  per-quarter spread (sec.9.9: 'the spread shows whether the edge is stable')")
    per_date = headline_bt["per_date"]
    print(
        per_date[
            ["observation_date", "base_rate", "q1_event_rate", "q5_event_rate", "separation_diff"]
        ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    print(
        f"\n    Q5 > Q1 on {hs['n_dates_q5_beats_q1']} of {hs['n_dates']} dates "
        f"| monotonic across all quintiles on {hs['n_dates_monotonic']} "
        f"| Q5/Q1 undefined on {hs['n_dates_ratio_undefined']} (no Q1 events)"
    )

    verdict = (
        "pledge-aware quintiles separate BETTER than volatility-only"
        if hs["separation_diff"] > ns["separation_diff"]
        else "pledge-aware quintiles separate NO BETTER than volatility-only"
    )
    print(f"\n    -> {verdict}")

    print(f"\n  {'=' * 92}")
    print("  SHAP - GLOBAL (sec.11.1) - 'this is where you audit the confound'")
    print(f"  {'=' * 92}")
    print(
        f"    explained model     : {audit_run['run_id'] if audit_run else active['run_id']}  "
        f"({audit['kind']} explainer, exact)"
    )
    print()
    print(
        audit["importance"]
        .head(settings.explain.beeswarm_max_display)
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print(f"\n    pledge features hold {confound['pledge_share_of_importance']:.1%} of total |SHAP|")
    print(f"    market features hold {confound['market_share_of_importance']:.1%}")
    print(f"    top 3               : {confound['top_3']}")
    print(f"    best pledge feature : rank {confound['best_pledge_feature_rank']} of {len(audit['names'])}")

    print(f"\n  SHAP - LOCAL + TEXT (sec.11.1) - {latest_date}, served model")
    print(f"    explanations stored : {stored:,} rows for {len(predictions)} predictions")
    print(f"\n    highest-risk company: {top.symbol}")
    print(f"    {sentence}")

    if written:
        print("\n  FIGURES (sec.12 - matplotlib for the README)")
        for path in written:
            print(f"    {path}")

    print("\n  PERSISTED")
    print(f"    backtest_results    : {counts['backtest_results']:,}")
    print(f"    explanations        : {counts['explanations']:,}")

    print("\n  next: python scripts/06_score_latest.py")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
