"""SHAP explanations - PLAN.md sec.11.

    1. GLOBAL   beeswarm summary. "This is where you audit the confound: if
                volatility_90d and log_turnover_90d dominate while pledge
                features sit near zero, the honest conclusion is that pledge
                adds little. Show that plot either way."
    2. LOCAL    per-company waterfall on the investigation screen.
    3. TEXT     top 3 SHAP values -> a sentence, via a template. NO LLM.

**Which model gets explained.** sec.11 specifies ``TreeExplainer`` on XGBoost,
and ``explain.model_for_shap`` still points there - that is the global confound
audit, and XGBoost is the right instrument for it because it is the most
flexible model in the study, so if pledge features carry usable structure it has
the best chance of finding it.

Per-prediction explanations are a different job: they must explain the model
that actually produced the number. sec.9.7's selection rule picked logreg on
this data, not the XGBoost sec.9.5 anticipated would win, so an explanation
lifted from XGBoost would describe a model the user is not being served.
:func:`build_explainer` therefore selects the explainer from the fitted
estimator - ``TreeExplainer`` for XGBoost and RandomForest, ``LinearExplainer``
for LogisticRegression. Both are exact, neither needs approximation tuning, and
no extra dependency is involved. sec.11.2's exclusion of "SHAP for the Random
Forest" is about not duplicating the same explanation twice over, not about
explaining a model you never serve.

**Preprocessing is stripped before explaining.** A Pipeline's estimator sees
winsorised, imputed, scaled columns - so SHAP must be handed those same columns,
not the raw panel. And when the imputer adds missingness indicators the matrix
grows past 13 columns, so the feature names have to be recovered from the fitted
pipeline rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from pledgecast.exceptions import ValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

# Presentation text, not a tunable - it belongs next to the code that renders
# it rather than in config.yaml. `{v}` is the feature's value for the row.
FEATURE_PHRASES: dict[str, str] = {
    "promoter_holding_pct": "promoters hold {v:.1f}% of equity",
    "pledge_pct_promoter": "{v:.1f}% of the promoter stake is pledged",
    "pledge_pct_equity": "pledged shares are {v:.1f}% of equity",
    "pledge_chg_1q": "pledge {dir1} {av:.1f}pp over one quarter",
    "pledge_chg_2q": "pledge {dir1} {av:.1f}pp over two quarters",
    "pledge_accel": "pledge change {dir2} by {av:.1f}pp",
    "consecutive_rising_q": "pledge has risen {v:.0f} consecutive quarters",
    "pledge_max_4q": "peak pledge over four quarters was {v:.1f}%",
    "volatility_90d": "90-day volatility {v:.0%}",
    "trailing_dd_60d": "already down {av:.0%} over 60 days",
    "return_90d": "90-day return {v:+.0%}",
    "rel_return_90d": "{dir3} NIFTY by {av:.0%}",
    "log_turnover_90d": "log 90-day turnover {v:.1f}",
}


def _phrase(feature: str, value: float) -> str:
    """One feature-value pair rendered as English."""
    if feature.endswith("_missing") or value is None or pd.isna(value):
        return f"{feature.removesuffix('_missing')} was unavailable"
    template = FEATURE_PHRASES.get(feature)
    if template is None:
        return f"{feature} = {value}"
    return template.format(
        v=value,
        av=abs(value),
        dir1="rose" if value >= 0 else "fell",
        dir2="accelerated" if value >= 0 else "decelerated",
        dir3="outperformed" if value >= 0 else "underperformed",
    )


# --------------------------------------------------------------------------- #
# pipeline introspection                                                       #
# --------------------------------------------------------------------------- #
def transformed_feature_names(pipeline, features: list[str]) -> list[str]:
    """Column names AFTER preprocessing.

    ``SimpleImputer(add_indicator=True)`` appends one binary column per feature
    that had a missing value in the training fold, so the estimator can see 15
    columns where the panel offered 13. Reading the fitted imputer's
    ``indicator_.features_`` recovers exactly which ones, in order - guessing
    would misalign every SHAP value with its label.
    """
    names = list(features)
    imputer = pipeline.named_steps.get("impute")
    if imputer is not None and getattr(imputer, "indicator_", None) is not None:
        names += [f"{features[i]}_missing" for i in imputer.indicator_.features_]
    return names


def transform(pipeline, matrix: np.ndarray) -> np.ndarray:
    """Run the preprocessing steps, stopping short of the estimator."""
    return pipeline[:-1].transform(matrix)


def build_explainer(pipeline, background: np.ndarray) -> tuple[Any, str]:
    """Pick the exact explainer for the fitted estimator. Returns ``(explainer, kind)``."""
    estimator = pipeline[-1]

    if isinstance(estimator, LogisticRegression):
        return shap.LinearExplainer(estimator, background), "linear"

    try:
        return shap.TreeExplainer(estimator), "tree"
    except Exception as exc:  # noqa: BLE001 - surfaced with context, then re-raised
        raise ValidationError(
            f"no exact SHAP explainer for {type(estimator).__name__}: {exc}. "
            "sec.11 uses TreeExplainer for trees and LinearExplainer for logistic "
            "regression; adding a model means choosing its explainer deliberately."
        ) from exc


def shap_values(explainer, matrix: np.ndarray) -> np.ndarray:
    """SHAP values for the positive class, always as ``(n_rows, n_features)``.

    TreeExplainer's output shape moved around across versions and across
    binary/multiclass estimators - a list of two arrays, or a 3-D array with a
    trailing class axis. Both are normalised here so callers never branch on it.
    """
    values = explainer.shap_values(matrix)

    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


# --------------------------------------------------------------------------- #
# 1. global (sec.11.1)                                                         #
# --------------------------------------------------------------------------- #
def global_importance(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature - the confound audit as a table."""
    frame = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    total = frame["mean_abs_shap"].sum()
    frame["share"] = frame["mean_abs_shap"] / total if total else np.nan
    return frame.reset_index(drop=True)


def confound_audit(importance: pd.DataFrame, pledge_features: list[str]) -> dict:
    """sec.11.1's actual question, answered numerically rather than by eye.

    "If volatility_90d and log_turnover_90d dominate while pledge features sit
    near zero, the honest conclusion is that pledge adds little."
    """
    pledge_share = float(importance[importance["feature"].isin(pledge_features)]["share"].sum())
    ranked = importance["feature"].tolist()
    return {
        "pledge_share_of_importance": pledge_share,
        "market_share_of_importance": 1.0 - pledge_share,
        "top_feature": ranked[0] if ranked else None,
        "top_3": ranked[:3],
        "best_pledge_feature_rank": next(
            (i + 1 for i, f in enumerate(ranked) if f in pledge_features), None
        ),
    }


def beeswarm(values: np.ndarray, matrix: np.ndarray, names: list[str], path: Path, max_display: int):
    """Global beeswarm PNG for the README (sec.12: matplotlib, not Plotly)."""
    import matplotlib

    matplotlib.use("Agg")  # no display on a build machine
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        values, matrix, feature_names=names, max_display=max_display, show=False, plot_size=(9, 6)
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    logger.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------- #
# 2. local (sec.11.1)                                                          #
# --------------------------------------------------------------------------- #
def explain_row(
    values: np.ndarray,
    matrix: np.ndarray,
    names: list[str],
    index: int,
    *,
    top_n: int | None = None,
) -> list[dict]:
    """Ranked per-feature contributions for one row, largest |SHAP| first."""
    row = pd.DataFrame(
        {
            "feature_name": names,
            "feature_value": matrix[index],
            "shap_value": values[index],
        }
    )
    row["abs"] = row["shap_value"].abs()
    row = row.sort_values("abs", ascending=False)
    if top_n:
        row = row.head(top_n)
    return row.drop(columns="abs").to_dict("records")


def explanation_rows(
    values: np.ndarray,
    raw: np.ndarray,
    names: list[str],
    features: list[str],
) -> list[list[dict]]:
    """Explanations for every row, carrying the RAW feature values.

    The estimator sees scaled numbers; a user reading "90-day volatility 0.42"
    wants the panel's 42%, not the pipeline's -0.31. Indicator columns have no
    raw counterpart, so they report their transformed 0/1, which is what they
    mean anyway.
    """
    out = []
    for i in range(values.shape[0]):
        records = []
        for j, name in enumerate(names):
            value = float(raw[i, j]) if j < len(features) else None
            records.append(
                {
                    "feature_name": name,
                    "feature_value": value,
                    "shap_value": float(values[i, j]),
                }
            )
        records.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
        out.append(records)
    return out


def waterfall(
    explainer,
    values: np.ndarray,
    matrix: np.ndarray,
    names: list[str],
    index: int,
    path: Path | None = None,
    max_display: int = 10,
    rc: dict | None = None,
):
    """Single-prediction waterfall (sec.12: matplotlib -> ``st.pyplot``).

    ``rc`` overrides matplotlib rcParams for this figure only. The dashboard
    passes its theme through it so the one non-Plotly figure in the app sits on
    the same ground as the other ten instead of arriving on matplotlib's default
    white - which reads as a foreign object dropped into the page, on the tab
    that exists to make the model's reasoning trustworthy.

    It is a parameter rather than an import because this module lives under
    ``src/`` and the theme lives under ``dashboard/``; the dependency runs one
    way only, so the renderer tells the plotter how to look rather than the
    plotter reaching into the app.

    ``rc_context`` scopes the change to this call. Setting rcParams globally
    would leak into the README's beeswarm, which is a report artefact with its
    own committed appearance.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = explainer.expected_value
    if isinstance(base, list | np.ndarray):
        base = np.asarray(base).ravel()[-1]

    explanation = shap.Explanation(
        values=values[index],
        base_values=float(base),
        data=matrix[index],
        feature_names=names,
    )
    with matplotlib.rc_context(rc or {}):
        figure = plt.figure()
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=150)
            plt.close("all")
            return path
    return figure


# --------------------------------------------------------------------------- #
# 3. human-readable (sec.11.1)                                                 #
# --------------------------------------------------------------------------- #
def merge_indicators(records: list[dict]) -> list[dict]:
    """Fold each ``X_missing`` indicator back into ``X``, summing their SHAP.

    ``SimpleImputer(add_indicator=True)`` splits one fact - "this feature was
    unavailable" - across two columns: the indicator, and the median that got
    filled in. SHAP attributes to both, so an unmerged sentence reports the same
    missing feature twice, once as a driver and once as an offset, at two
    different magnitudes. Neither number is the contribution of the fact. Their
    sum is.

    Storage keeps the columns separate - the ``explanations`` table records what
    the model actually saw. This is a presentation step.
    """
    by_name = {r["feature_name"]: dict(r) for r in records}
    merged: list[dict] = []

    for record in records:
        name = record["feature_name"]
        if name.endswith("_missing"):
            continue
        indicator = by_name.get(f"{name}_missing")
        if indicator is not None:
            record = dict(record)
            record["shap_value"] += indicator["shap_value"]
        merged.append(record)

    merged.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return merged


def summarise(
    records: list[dict],
    probability: float,
    *,
    decile: int | None = None,
    band: str | None = None,
    top_n: int = 3,
) -> str:
    """Top SHAP values as a sentence. A template, no LLM (sec.11.1).

    Drivers and offsets are separated by the SIGN of the SHAP value, not by the
    feature's own direction - "already down 18% over 60 days" can push risk
    either way depending on what the model learned, and the explanation has to
    report what the model did rather than what the reader expects.
    """
    records = merge_indicators(records)
    label = band or ("Elevated risk" if probability >= 0.5 else "Risk")
    head = f"{label} (probability {probability:.2f}"
    if decile is not None:
        head += f", decile {decile}"
    head += ")."

    raising = [r for r in records if r["shap_value"] > 0][:top_n]
    lowering = [r for r in records if r["shap_value"] < 0][:top_n]

    parts = [head]
    if raising:
        drivers = ", ".join(
            f"{_phrase(r['feature_name'], r['feature_value'])} ({r['shap_value']:+.2f})"
            for r in raising
        )
        parts.append(f"Main drivers: {drivers}.")
    if lowering:
        offsets = ", ".join(
            f"{_phrase(r['feature_name'], r['feature_value'])} ({r['shap_value']:+.2f})"
            for r in lowering
        )
        parts.append(f"Offsetting: {offsets}.")
    if not raising and not lowering:
        parts.append("No feature moved this prediction away from the base rate.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# orchestration                                                                #
# --------------------------------------------------------------------------- #
def explain(
    pipeline, raw: np.ndarray, features: list[str], *, background: np.ndarray | None = None
) -> dict:
    """Everything the callers need, computed once.

    ``background`` is the reference population SHAP measures deviation FROM. It
    defaults to the matrix being explained, which is right for a batch call over
    the whole panel and catastrophically wrong for a single row: LinearExplainer
    computes ``coef * (x - mean(background))``, so explaining one row against
    itself returns exactly zero for every feature. The API hit precisely that -
    a well-formed response whose every SHAP value was 0.0. Single-row callers
    must pass the population explicitly.
    """
    matrix = transform(pipeline, raw)
    reference = matrix if background is None else transform(pipeline, background)
    names = transformed_feature_names(pipeline, features)

    if matrix.shape[1] != len(names):
        raise ValidationError(
            f"recovered {len(names)} feature names for a {matrix.shape[1]}-column matrix; "
            "the SHAP values would be mislabelled"
        )

    explainer, kind = build_explainer(pipeline, reference)
    values = shap_values(explainer, matrix)
    logger.info("SHAP: %s explainer, %d rows x %d features", kind, *values.shape)

    return {
        "explainer": explainer,
        "kind": kind,
        "values": values,
        "matrix": matrix,
        "raw": raw,
        "names": names,
        "importance": global_importance(values, names),
    }


__all__ = [
    "FEATURE_PHRASES",
    "beeswarm",
    "build_explainer",
    "confound_audit",
    "explain",
    "explain_row",
    "explanation_rows",
    "global_importance",
    "merge_indicators",
    "shap_values",
    "summarise",
    "transform",
    "transformed_feature_names",
    "waterfall",
]
