"""The training loop - PLAN.md sec.9.4-9.8.

    experiments x models x folds

    exp0_null      volatility + turnover only     "how much is just small and jumpy?"
    expA_pledge    the 8 pledge features          "standalone signal?"
    expB_full      all 13                         HEADLINE
    ablation_static  levels + market              "does trajectory beat levels?"

    HEADLINE RESULT = expB_full.within_quarter_auc - exp0_null.within_quarter_auc

Four things in here are load-bearing and easy to get subtly wrong:

1. **Preprocessing is inside the pipeline**, so it is re-fit per fold on the
   training rows only (sec.9.4 "fold hygiene"). Nothing is fit globally.

2. **The hyperparameter search runs on ONE fold** (sec.9.5) and is then frozen
   for every experiment and every later fold. Searching per experiment would
   quadruple the selection surface on a 6,000-row panel.

3. **Fold 0's reported score is not clean** once it has been used to choose
   hyperparameters. It is reported anyway, alongside an aggregate that excludes
   it, so the reader can see the size of the effect instead of trusting that it
   is small.

4. **The shuffle test re-runs the whole walk-forward** (sec.9.8), not a single
   fit. A leak that only appears through the fold structure would survive a
   cheaper check.

sec.9.6 insists on per-fold spread rather than a pooled number, so every result
here carries its fold table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Connection

from pledgecast.data.population import apply_population, describe_populations
from pledgecast.db import repository as repo
from pledgecast.evaluation import backtest, leakage, metrics, power
from pledgecast.exceptions import InsufficientDataError
from pledgecast.logging_config import get_logger
from pledgecast.models import definitions, registry
from pledgecast.models.preprocessing import build_pipeline, prepare_matrix
from pledgecast.training import walkforward as wf

logger = get_logger(__name__)


@dataclass
class RunResult:
    """One (model, experiment) pair evaluated across every fold."""

    run_id: str
    model_name: str
    experiment: str
    features: list[str]
    hyperparams: dict[str, Any]
    fold_metrics: pd.DataFrame
    aggregate: dict[str, float | None]
    oof: pd.DataFrame
    n_train_rows: int
    n_folds: int
    skipped_folds: list[int] = field(default_factory=list)
    # Which stratum of the panel this run was fitted and scored on. Two runs are
    # only comparable when this matches.
    population: str = "all"

    @property
    def primary(self) -> float | None:
        return self.aggregate.get("within_quarter_auc")


def make_run_id(stamp: str, model_name: str, experiment: str) -> str:
    """``20260814T1030_xgboost_expB_full`` - sortable, self-describing.

    One timestamp is shared by every run in a training session, so the twelve
    rows of a single ``04_train_all.py`` invocation group together in
    ``model_runs`` without needing a separate batch table.
    """
    return f"{stamp}_{model_name}_{experiment}"


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M")


# --------------------------------------------------------------------------- #
# one fold                                                                     #
# --------------------------------------------------------------------------- #
def fit_fold(
    panel: pd.DataFrame,
    fold: wf.Fold,
    features: list[str],
    model_name: str,
    settings,
    *,
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Fit on the fold's training dates, score its single test date.

    Returns the test rows with a ``probability`` column, or ``None`` when the
    training window carries a single class - AUC would be undefined and the
    estimator would refuse to fit. That is a reportable event, not a crash.
    """
    train, test = wf.split(panel, fold)
    y_train = train["label"].to_numpy(dtype=float)

    if len(np.unique(y_train)) < 2:
        logger.warning(
            "fold %d (%s): training window has a single class - skipped",
            fold.index,
            fold.test_date,
        )
        return None

    spec = settings.models[model_name]
    pipeline = build_pipeline(
        definitions.build_estimator(model_name, settings, overrides=overrides), spec, settings
    )
    pipeline.fit(prepare_matrix(train, features), y_train)

    probability = pipeline.predict_proba(prepare_matrix(test, features))[:, 1]

    out = test[["symbol", "observation_date", "label"]].copy()
    out["probability"] = probability
    out["fold"] = fold.index
    return out


def run_walkforward(
    panel: pd.DataFrame,
    plan: wf.FoldPlan,
    features: list[str],
    model_name: str,
    settings,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Every fold for one (model, experiment). Returns ``(oof, fold_metrics, skipped)``."""
    frames: list[pd.DataFrame] = []
    rows: list[dict] = []
    skipped: list[int] = []

    for fold in plan.folds:
        predicted = fit_fold(panel, fold, features, model_name, settings, overrides=overrides)
        if predicted is None:
            skipped.append(fold.index)
            continue
        frames.append(predicted)

        # A fold has exactly one test date, so this AUC *is* that date's AUC.
        fold_metrics = metrics.evaluate(
            predicted["label"],
            predicted["probability"],
            predicted["observation_date"],
            k=settings.evaluation.precision_at_k,
            min_rows=settings.evaluation.min_rows_per_quarter_for_auc,
        )
        rows.append(
            {
                "fold": fold.index,
                "test_date": fold.test_date,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "test_event_rate": fold.n_test_events / fold.n_test if fold.n_test else None,
                **{k: v for k, v in fold_metrics.items() if k != "base_rate"},
            }
        )

    oof = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["symbol", "observation_date", "label", "probability", "fold"])
    )
    return oof, pd.DataFrame(rows), skipped


def score_oof(oof: pd.DataFrame, settings) -> dict[str, float | None]:
    """Aggregate metrics over the pooled out-of-fold predictions.

    Because every fold contributes exactly one observation date, the
    within-quarter AUC computed here is identical to the mean of the per-fold
    AUCs - the two views cannot disagree, which is why both are reported.
    """
    if oof.empty:
        return {}
    return metrics.evaluate(
        oof["label"],
        oof["probability"],
        oof["observation_date"],
        k=settings.evaluation.precision_at_k,
        min_rows=settings.evaluation.min_rows_per_quarter_for_auc,
    )


# --------------------------------------------------------------------------- #
# hyperparameter search - fold 1 only (sec.9.5)                                #
# --------------------------------------------------------------------------- #
def random_search(
    panel: pd.DataFrame,
    plan: wf.FoldPlan,
    settings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """20-point random search on ONE fold, then freeze (sec.9.5).

    Searched on the headline experiment's feature set and then applied to every
    experiment. The alternative - one search per experiment - multiplies the
    number of choices made from the same 6,000 rows by four, which is precisely
    the overfitting sec.9.5 warns about. Hyperparameters describe how the model
    family behaves on this data, not what the features mean.

    Candidates are ranked on the search fold's TEST date. That makes fold
    ``search_fold_index`` optimistically biased for the searched model; the
    caller reports an aggregate with and without it.
    """
    model_name = settings.training.search_model
    points = definitions.search_points(model_name, settings)
    if not points:
        logger.info("no search_space configured for %s - keeping config defaults", model_name)
        return {}, pd.DataFrame()

    index = settings.training.search_fold_index
    if index >= len(plan.folds):
        raise InsufficientDataError(
            f"training.search_fold_index={index} but only {len(plan.folds)} folds exist"
        )
    fold = plan.folds[index]
    features = settings.experiment_features(settings.headline.experiment)

    logger.info(
        "random search: %d points, model=%s, fold %d (test %s, %d events of %d), features=%s",
        len(points),
        model_name,
        index,
        fold.test_date,
        fold.n_test_events,
        fold.n_test,
        settings.headline.experiment,
    )
    if fold.n_test_events < settings.evaluation.min_rows_per_quarter_for_auc:
        logger.warning(
            "search fold %d has only %d positive labels - an AUC ranked on that few events is "
            "mostly noise, so the frozen hyperparameters are weakly identified. Compare against "
            "`--no-search` (the sec.9.5 configured defaults) before trusting the difference.",
            index,
            fold.n_test_events,
        )

    rows = []
    for position, candidate in enumerate(points):
        predicted = fit_fold(panel, fold, features, model_name, settings, overrides=candidate)
        auc = (
            None
            if predicted is None
            else metrics.within_quarter_auc(
                predicted["label"],
                predicted["probability"],
                predicted["observation_date"],
                min_rows=settings.evaluation.min_rows_per_quarter_for_auc,
            )
        )
        rows.append({"point": position, "within_quarter_auc": auc, **candidate})

    table = pd.DataFrame(rows).sort_values("within_quarter_auc", ascending=False, na_position="last")
    if table["within_quarter_auc"].isna().all():
        logger.warning("every search point scored NaN - keeping config defaults")
        return {}, table

    best = table.iloc[0]
    # Read the winning point out of `points`, NOT out of the DataFrame row.
    # A DataFrame row spanning int and float columns is upcast to float64, so
    # `best["n_estimators"]` comes back as 300.0 and XGBoost raises
    # "'float' object cannot be interpreted as an integer" - a failure that
    # surfaces four models later, nowhere near its cause.
    chosen = dict(points[int(best["point"])])

    scored = table["within_quarter_auc"].dropna()
    logger.info(
        "search best: AUC %.4f with %s (%d points scored, spread %.4f..%.4f)",
        best["within_quarter_auc"],
        chosen,
        len(scored),
        scored.min(),
        scored.max(),
    )
    return chosen, table


# --------------------------------------------------------------------------- #
# persistence                                                                  #
# --------------------------------------------------------------------------- #
def _risk_deciles(oof: pd.DataFrame, n_deciles: int) -> pd.Series:
    """Decile of predicted risk WITHIN each observation date, 1 = safest.

    Ranked per date rather than pooled, for the same reason within-quarter AUC
    is the primary metric: a pooled decile would mostly encode which quarter a
    row came from. Shares ``backtest.rank_groups`` with the quintile backtest so
    the two rankings cannot drift apart.
    """
    return oof.groupby("observation_date")["probability"].transform(
        lambda block: backtest.rank_groups(block, n_deciles)
    )


def persist_run(
    conn: Connection,
    result: RunResult,
    settings,
    *,
    pipeline=None,
) -> None:
    """Write the run row, its metrics, and its out-of-fold predictions.

    Fold metrics are stored with their fold index; the aggregate uses fold -1,
    which is the convention sec.6's ``model_metrics`` table documents. Every
    run's OOF predictions are persisted with ``source='backtest'`` because
    sec.9.9's quintile comparison needs the null model's predictions as well as
    the headline model's.
    """
    if pipeline is not None:
        registry.register(
            conn,
            pipeline,
            run_id=result.run_id,
            model_name=result.model_name,
            experiment=result.experiment,
            feature_list=result.features,
            hyperparams=result.hyperparams,
            settings=settings,
            n_train_rows=result.n_train_rows,
            n_folds=result.n_folds,
        )
    else:
        repo.insert_model_run(
            conn,
            run_id=result.run_id,
            model_name=result.model_name,
            experiment=result.experiment,
            feature_list=result.features,
            hyperparams=result.hyperparams,
            random_seed=settings.training.random_seed,
            n_train_rows=result.n_train_rows,
            n_folds=result.n_folds,
            artifact_path=None,
            config_snapshot=settings.snapshot(),
        )

    payload = [
        {"fold": int(row["fold"]), "metric_name": name, "metric_value": float(row[name])}
        for _, row in result.fold_metrics.iterrows()
        for name in result.fold_metrics.columns
        if name not in ("fold", "test_date") and pd.notna(row[name])
    ]
    payload += [
        {"fold": -1, "metric_name": name, "metric_value": float(value)}
        for name, value in result.aggregate.items()
        if value is not None
    ]
    repo.insert_metrics(conn, result.run_id, payload)

    repo.delete_predictions_for_run(conn, result.run_id)
    if not result.oof.empty:
        oof = result.oof.copy()
        oof["risk_decile"] = _risk_deciles(oof, settings.evaluation.n_deciles)
        repo.save_predictions_bulk(
            conn,
            (
                {
                    "run_id": result.run_id,
                    "symbol": row.symbol,
                    "observation_date": row.observation_date,
                    "probability": float(row.probability),
                    "risk_decile": int(row.risk_decile),
                    "source": "backtest",
                }
                for row in oof.itertuples(index=False)
            ),
        )


# --------------------------------------------------------------------------- #
# selection (sec.9.7)                                                          #
# --------------------------------------------------------------------------- #
def select_best(results: list[RunResult], settings) -> RunResult:
    """sec.9.7: highest mean within-quarter AUC, ties toward the simpler model.

    Selection runs WITHIN the headline experiment. The experiment ladder is a
    research comparison, not a menu of deployable models: ``exp0_null`` exists
    to be beaten or not beaten, and shipping it because it scored highest would
    answer the research question by changing the product. The headline delta is
    reported either way - if the null wins, that is the finding, and sec.2.2
    says to publish it.
    """
    candidates = [r for r in results if r.experiment == settings.headline.experiment]
    if not candidates:
        raise InsufficientDataError(
            f"no successful runs for the headline experiment {settings.headline.experiment!r}"
        )

    scored = [r for r in candidates if r.primary is not None]
    if not scored:
        raise InsufficientDataError("no headline run produced a within-quarter AUC")

    best_auc = max(r.primary for r in scored)
    tied = [
        r for r in scored if best_auc - r.primary <= settings.training.selection_tie_tolerance
    ]
    winner = min(tied, key=lambda r: definitions.simplicity_rank(r.model_name))

    if len(tied) > 1:
        logger.info(
            "%d models within %.3f AUC of the best (%.4f) - sec.9.7 tie-break selects the "
            "simpler: %s",
            len(tied),
            settings.training.selection_tie_tolerance,
            best_auc,
            winner.model_name,
        )
    return winner


def fit_final(
    panel: pd.DataFrame,
    result: RunResult,
    settings,
    *,
    overrides: dict[str, Any] | None = None,
):
    """sec.9.7 step 3: retrain the winner on ALL labelled data."""
    labelled = panel[panel["label_is_valid"] == 1]
    spec = settings.models[result.model_name]
    pipeline = build_pipeline(
        definitions.build_estimator(result.model_name, settings, overrides=overrides),
        spec,
        settings,
    )
    pipeline.fit(
        prepare_matrix(labelled, result.features), labelled["label"].to_numpy(dtype=float)
    )
    logger.info(
        "final model refit on all %d labelled rows (%s / %s)",
        len(labelled),
        result.model_name,
        result.experiment,
    )
    return pipeline, len(labelled)


# --------------------------------------------------------------------------- #
# gate 3 (sec.9.8)                                                             #
# --------------------------------------------------------------------------- #
def shuffle_gate(panel: pd.DataFrame, plan: wf.FoldPlan, settings, *, overrides=None) -> dict:
    """sec.9.8, non-negotiable: shuffled labels must collapse AUC to ~0.50.

    The full walk-forward is re-run per repeat rather than a single fit. A leak
    that entered through the fold structure - a training window reaching into
    its own test date - would survive a single-fit check and fail here.
    """
    model_name = settings.explain.model_for_shap  # the primary model, sec.9.5
    features = settings.experiment_features(settings.headline.experiment)

    def fit_and_score(shuffled: pd.DataFrame) -> float:
        oof, _, _ = run_walkforward(
            shuffled, plan, features, model_name, settings, overrides=overrides
        )
        score = score_oof(oof, settings).get("within_quarter_auc")
        return float("nan") if score is None else float(score)

    return leakage.label_shuffle_test(
        fit_and_score,
        panel,
        seed=settings.training.random_seed,
        tolerance=settings.evaluation.shuffle_test_tolerance,
    )


# --------------------------------------------------------------------------- #
# orchestration                                                                #
# --------------------------------------------------------------------------- #
def train_all(
    conn: Connection,
    settings,
    *,
    do_search: bool = True,
    run_shuffle_gate: bool = True,
) -> dict[str, Any]:
    """Every experiment x every model, walk-forward, then select and register.

    Returns a report dict; ``04_train_all.py`` renders it. Nothing is printed
    from here so the same function is usable from a test.
    """
    panel = repo.load_panel(conn, valid_only=False)
    if panel.empty:
        raise InsufficientDataError("panel is empty - run scripts/03_build_panel.py first")

    plan = wf.generate_folds(panel, min_train_quarters=settings.walkforward.min_train_quarters)
    fold_check = leakage.check_folds_disjoint([f.to_dict() for f in plan.folds])

    overrides, search_table = ({}, pd.DataFrame())
    if do_search:
        overrides, search_table = random_search(panel, plan, settings)

    stamp = timestamp()
    results: list[RunResult] = []

    # Strata are resolved once and shared. The fold PLAN stays the one built
    # from the full panel, so every experiment - stratified or not - is trained
    # and tested on the identical set of observation dates. `wf.split` selects
    # by date, so handing it a stratum panel yields that stratum's rows on those
    # same dates. Regenerating folds per stratum would let two experiments span
    # different quarters, and their delta would then confound the feature set
    # with the calendar.
    panels: dict[str, pd.DataFrame] = {}
    population_reports: dict[str, dict] = {}
    for name in {settings.experiment_population(e) for e in settings.experiments}:
        panels[name], population_reports[name] = apply_population(panel, name, settings)

    for experiment in settings.experiments:
        features = settings.experiment_features(experiment)
        population = settings.experiment_population(experiment)
        stratum = panels[population]
        for model_name in definitions.model_names(settings):
            applied = overrides if model_name == settings.training.search_model else {}
            oof, fold_metrics, skipped = run_walkforward(
                stratum, plan, features, model_name, settings, overrides=applied
            )
            aggregate = score_oof(oof, settings)
            aggregate.update(_search_fold_adjustment(fold_metrics, settings))

            result = RunResult(
                run_id=make_run_id(stamp, model_name, experiment),
                model_name=model_name,
                experiment=experiment,
                features=features,
                hyperparams=definitions.resolved_params(model_name, settings, overrides=applied),
                fold_metrics=fold_metrics,
                aggregate=aggregate,
                oof=oof,
                n_train_rows=_train_rows(stratum, plan),
                n_folds=len(plan.folds) - len(skipped),
                skipped_folds=skipped,
                population=population,
            )
            results.append(result)
            logger.info(
                "%-14s %-18s [%s] within-quarter AUC %s (%d folds)",
                model_name,
                experiment,
                population,
                f"{result.primary:.4f}" if result.primary is not None else "n/a",
                result.n_folds,
            )

    winner = select_best(results, settings)
    winner_overrides = overrides if winner.model_name == settings.training.search_model else {}
    pipeline, n_labelled = fit_final(
        panels[winner.population], winner, settings, overrides=winner_overrides
    )
    winner.n_train_rows = n_labelled

    # sec.11 runs the global confound audit on `explain.model_for_shap`. When the
    # sec.9.7 selection picks a different model - as it does on this data - that
    # model has no artifact to explain, so fit and store it too. It is NOT
    # activated: exactly one run serves, and that is the selected one.
    artifacts = {winner.run_id: pipeline}
    audited = _find(results, settings.explain.model_for_shap, settings.headline.experiment)
    if audited is not None and audited is not winner:
        audit_overrides = (
            overrides if audited.model_name == settings.training.search_model else {}
        )
        audit_pipeline, audit_rows = fit_final(
            panels[audited.population], audited, settings, overrides=audit_overrides
        )
        audited.n_train_rows = audit_rows
        artifacts[audited.run_id] = audit_pipeline

    for result in results:
        persist_run(conn, result, settings, pipeline=artifacts.get(result.run_id))
    registry.set_active(conn, winner.run_id, settings)

    gate = (
        shuffle_gate(
            panels[settings.experiment_population(settings.headline.experiment)],
            plan,
            settings,
            overrides=overrides,
        )
        if run_shuffle_gate
        else {"check": "label-shuffle", "passed": None, "skipped": True}
    )

    return {
        "stamp": stamp,
        "plan": plan,
        "fold_table": wf.describe(plan),
        "fold_check": fold_check,
        "search": {"overrides": overrides, "table": search_table},
        "results": results,
        "winner": winner,
        "headline": headline_delta(results, settings, searched=bool(overrides)),
        "deltas": deltas_vs_baseline(results, settings),
        "power": power_report(results, panels, settings),
        "populations": describe_populations(panel, settings),
        "population_reports": population_reports,
        "comparison": comparison_table(results),
        "shuffle_gate": gate,
    }


def _train_rows(stratum: pd.DataFrame, plan: wf.FoldPlan) -> int:
    """Labelled rows in the widest training window, for THIS stratum.

    ``plan`` carries counts taken from the full panel, so a stratified run must
    not reuse them - reporting 4,000 training rows for a model actually fitted
    on 600 would misstate the study by a factor of six.
    """
    if not plan.folds or stratum.empty:
        return 0
    labelled = stratum[stratum["label_is_valid"] == 1]
    return int(labelled["observation_date"].isin(plan.folds[-1].train_dates).sum())


def power_report(
    results: list[RunResult],
    panels: dict[str, pd.DataFrame],
    settings,
) -> pd.DataFrame:
    """Every experiment-vs-its-baseline delta, with an interval and a ceiling.

    This is what makes the study's conclusion falsifiable. A delta of -0.016 on
    its own cannot be argued with because it makes no claim; the same delta
    written ``-0.016 [-0.045, +0.012], ceiling +0.190`` says three things that
    can each be checked: the effect is not distinguishable from zero, an effect
    larger than about 0.03 would have been, and an effect as large as 0.19 was
    available to be found.
    """
    rows: list[dict] = []
    for experiment in settings.experiments:
        baseline = settings.experiment_baseline(experiment)
        if experiment == baseline:
            continue
        treatment_features = set(settings.experiment_features(experiment))
        extra = sorted(treatment_features - set(settings.experiment_features(baseline)))
        stratum = panels[settings.experiment_population(experiment)]

        for model_name in definitions.model_names(settings):
            treatment = _find(results, model_name, experiment)
            control = _find(results, model_name, baseline)
            if treatment is None or control is None or treatment.oof.empty:
                continue
            assessment = power.assess(treatment.oof, control.oof, stratum, extra, settings)
            rows.append(
                {
                    "experiment": experiment,
                    "baseline": baseline,
                    "population": treatment.population,
                    "model": model_name,
                    "n_extra_features": len(extra),
                    **assessment,
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    keep = [
        c
        for c in (
            "experiment",
            "baseline",
            "population",
            "model",
            "delta",
            "ci_low",
            "ci_high",
            "half_width",
            "ceiling",
            "ceiling_binding",
            "coverage",
            "n_dates",
            "dates_better",
            "dates_worse",
            "verdict",
        )
        if c in frame.columns
    ]
    return frame[keep]


def _search_fold_adjustment(fold_metrics: pd.DataFrame, settings) -> dict[str, float | None]:
    """Mean within-quarter AUC excluding the fold the search was fitted on.

    Computed for EVERY model, not just the searched one, so the column stays a
    like-for-like comparison over the same folds. Only the searched model's
    headline number is optimistically biased, but comparing its clean subset
    against another model's full set would just trade one bias for another.
    """
    if fold_metrics.empty or "within_quarter_auc" not in fold_metrics:
        return {}
    kept = fold_metrics[fold_metrics["fold"] != settings.training.search_fold_index]
    value = float(kept["within_quarter_auc"].mean()) if not kept.empty else None
    return {"within_quarter_auc_ex_search_fold": value}


def headline_delta(results: list[RunResult], settings, *, searched: bool = False) -> dict[str, Any]:
    """HEADLINE = expB_full - exp0_null on the primary metric (sec.2.3).

    Compared model-for-model. Taking the best of one experiment against the best
    of the other would let the difference be a model choice wearing a
    feature-set label.

    **The verdict is the median delta across models, not one model's.** Measured
    on this panel, reading a single model gives the wrong answer: the searched
    model's frozen hyperparameters were chosen using the headline experiment's
    13 features and are then applied to the 2-feature baseline, where they do
    real damage (xgboost exp0_null fell 0.6177 -> 0.5878). That alone flipped
    its delta from -0.0013 to +0.0120 - a positive headline produced by
    handicapping the control rather than by improving the treatment. The other
    two models, whose hyperparameters were fixed a priori, both stayed negative.

    So the searched model's row is flagged and the verdict is taken from the
    median, which agrees across the tuned and untuned runs.
    """
    metric = settings.headline.metric
    rows = []
    for model_name in definitions.model_names(settings):
        full = _find(results, model_name, settings.headline.experiment)
        null = _find(results, model_name, settings.headline.baseline)
        if full is None or null is None:
            continue
        a, b = full.aggregate.get(metric), null.aggregate.get(metric)
        rows.append(
            {
                "model": model_name,
                settings.headline.experiment: a,
                settings.headline.baseline: b,
                "delta": None if a is None or b is None else a - b,
                # True when this model's hyperparameters were selected using the
                # treatment's feature set - its delta is not a clean comparison.
                "tuned_on_treatment": searched and model_name == settings.training.search_model,
            }
        )

    table = pd.DataFrame(rows)
    deltas = [r["delta"] for r in rows if r["delta"] is not None]
    clean = [r["delta"] for r in rows if r["delta"] is not None and not r["tuned_on_treatment"]]

    return {
        "metric": metric,
        "table": table,
        "median_delta": float(np.median(deltas)) if deltas else None,
        "clean_median_delta": float(np.median(clean)) if clean else None,
        "n_models": len(deltas),
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_negative": sum(1 for d in deltas if d <= 0),
        "searched_model": settings.training.search_model if searched else None,
    }


def deltas_vs_baseline(results: list[RunResult], settings) -> pd.DataFrame:
    """Every experiment minus the null, model by model, then the median.

    With seven experiments the single headline pair is no longer the whole
    story: the Reg 31 block asks the same question at event resolution, and
    ``expD_events_market`` is its fair comparison against ``exp0_null`` because
    both carry the identical market control. Reporting one delta per experiment
    keeps every comparison against the SAME baseline rather than against
    whichever experiment happens to sit next to it.

    "The same baseline" now means *the same baseline within the same stratum*.
    A pledged-only experiment resolves to the pledged-only null, because its
    delta against the full-panel null would be dominated by the 81% of rows the
    stratum removed rather than by the features under test.
    """
    metric = settings.headline.metric

    rows = []
    for experiment in settings.experiments:
        baseline = settings.experiment_baseline(experiment)
        if experiment == baseline:
            continue
        per_model = {}
        for model_name in definitions.model_names(settings):
            treatment = _find(results, model_name, experiment)
            control = _find(results, model_name, baseline)
            if treatment is None or control is None:
                continue
            a, b = treatment.aggregate.get(metric), control.aggregate.get(metric)
            if a is not None and b is not None:
                per_model[model_name] = a - b
        if not per_model:
            continue
        rows.append(
            {
                "experiment": experiment,
                "vs": baseline,
                "population": settings.experiment_population(experiment),
                "n_features": len(settings.experiment_features(experiment)),
                **{f"delta_{k}": v for k, v in per_model.items()},
                "median_delta": float(np.median(list(per_model.values()))),
                # Counted, but NOT interpreted as evidence: the three models are
                # fitted on the same rows, so their errors are correlated and
                # "3 of 3 negative" is closer to one observation than to three.
                # power_report's interval is what decides the direction.
                "models_negative": sum(1 for v in per_model.values() if v <= 0),
            }
        )
    return pd.DataFrame(rows).sort_values("median_delta", ascending=False)


def comparison_table(results: list[RunResult]) -> pd.DataFrame:
    """Every run, one row - the Validation page and the console banner use this."""
    return pd.DataFrame(
        [
            {
                "experiment": r.experiment,
                "model": r.model_name,
                "n_features": len(r.features),
                "folds": r.n_folds,
                "within_quarter_auc": r.aggregate.get("within_quarter_auc"),
                "auc_std": r.aggregate.get("within_quarter_auc_std"),
                "auc_min": r.aggregate.get("within_quarter_auc_min"),
                "auc_max": r.aggregate.get("within_quarter_auc_max"),
                "auc_ex_search_fold": r.aggregate.get("within_quarter_auc_ex_search_fold"),
                "pooled_auc": r.aggregate.get("pooled_auc"),
                "pr_auc": r.aggregate.get("pr_auc"),
                "brier_skill_score": r.aggregate.get("brier_skill_score"),
                "run_id": r.run_id,
            }
            for r in results
        ]
    )


def _find(results: list[RunResult], model_name: str, experiment: str) -> RunResult | None:
    for result in results:
        if result.model_name == model_name and result.experiment == experiment:
            return result
    return None


__all__ = [
    "RunResult",
    "comparison_table",
    "deltas_vs_baseline",
    "fit_final",
    "fit_fold",
    "headline_delta",
    "make_run_id",
    "persist_run",
    "power_report",
    "random_search",
    "run_walkforward",
    "score_oof",
    "select_best",
    "shuffle_gate",
    "train_all",
]
