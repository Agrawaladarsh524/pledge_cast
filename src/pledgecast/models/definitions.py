"""Model definitions - PLAN.md sec.9.5.

    Logistic Regression   interpretable baseline. L2, scaled, class_weight balanced
    Random Forest         second reference. 300 trees, depth 8, min_samples_leaf 10
    XGBoost               PRIMARY. depth 3, 400 trees, lr 0.04, subsample 0.8, ...

    "Hyperparameter tuning: a 20-point random search on fold 1 only, then
     freeze. At ~6,000 rows a large search *is* overfitting."

**Why there is no hardcoded MODELS grid here.** sec.8's sketch describes this
file as "MODELS dict: name -> (estimator, param grid)", but sec.8.1 and sec.10
are stricter: *every* hyperparameter lives in config.yaml, "zero magic numbers
in code". A grid written twice is a grid that will disagree with itself. So the
only thing this module hardcodes is ``ESTIMATORS`` - the mapping from a config
string to a Python class, which genuinely cannot live in YAML. Parameters,
search space and preprocessing flags all come from ``settings.models[name]``.

``SIMPLICITY_ORDER`` encodes sec.9.7 rule 2 ("ties broken toward the simpler
model") as data rather than as a comment.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pledgecast.exceptions import ValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

# config.yaml `estimator:` string -> class. The one unavoidable piece of
# code-side knowledge; everything else about a model is configuration.
ESTIMATORS: dict[str, type] = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

# sec.9.7 rule 2: simplest first. Used only to break a tie in model selection.
SIMPLICITY_ORDER: tuple[str, ...] = ("logreg", "random_forest", "xgboost")

# Estimators that take a `random_state`. All three do, but stating it is
# cheaper than discovering the exception later.
_SEEDED = frozenset(ESTIMATORS)


def simplicity_rank(model_name: str) -> int:
    """Position in ``SIMPLICITY_ORDER``; unknown models sort last."""
    try:
        return SIMPLICITY_ORDER.index(model_name)
    except ValueError:
        return len(SIMPLICITY_ORDER)


def model_names(settings) -> list[str]:
    """Configured model names, in config order."""
    return list(settings.models)


def build_estimator(model_name: str, settings, *, overrides: dict[str, Any] | None = None):
    """Instantiate one configured estimator.

    ``overrides`` carries the frozen result of the fold-1 random search
    (sec.9.5); it is applied on top of the config defaults so the config file
    always shows the starting point and the run row records what actually ran.
    """
    if model_name not in settings.models:
        raise ValidationError(
            f"unknown model {model_name!r}. Configured: {sorted(settings.models)}"
        )

    spec = settings.models[model_name]
    if spec.estimator not in ESTIMATORS:
        raise ValidationError(
            f"model {model_name!r} names estimator {spec.estimator!r}, which is not one of "
            f"{sorted(ESTIMATORS)}. Adding one is a code change, deliberately."
        )

    params = dict(spec.params)
    params.update(overrides or {})
    if spec.estimator in _SEEDED:
        params.setdefault("random_state", settings.training.random_seed)

    return ESTIMATORS[spec.estimator](**params)


def resolved_params(model_name: str, settings, *, overrides: dict[str, Any] | None = None) -> dict:
    """Exactly the parameters ``build_estimator`` would use - stored on the run row."""
    spec = settings.models[model_name]
    params = dict(spec.params)
    params.update(overrides or {})
    if spec.estimator in _SEEDED:
        params.setdefault("random_state", settings.training.random_seed)
    return params


def search_points(model_name: str, settings) -> list[dict[str, Any]]:
    """``training.search_n_points`` distinct draws from the configured space.

    The full grid is enumerated and then sampled WITHOUT replacement, rather
    than drawing each axis independently. Independent draws can repeat a
    combination, which would quietly turn a 20-point search into a 17-point
    one; enumeration makes "20 points" mean 20 distinct configurations. The
    configured space is 3,888 combinations, so enumerating it is trivial.
    """
    spec = settings.models[model_name]
    if not spec.search_space:
        return []

    axes = sorted(spec.search_space)
    grid = [
        dict(zip(axes, values, strict=True))
        for values in itertools.product(*(spec.search_space[a] for a in axes))
    ]

    n = settings.training.search_n_points
    if len(grid) <= n:
        logger.info(
            "search space has %d combinations, <= the %d requested - evaluating all of them",
            len(grid),
            n,
        )
        return grid

    rng = np.random.default_rng(settings.training.random_seed)
    chosen = rng.choice(len(grid), size=n, replace=False)
    return [grid[int(i)] for i in chosen]


__all__ = [
    "ESTIMATORS",
    "SIMPLICITY_ORDER",
    "build_estimator",
    "model_names",
    "resolved_params",
    "search_points",
    "simplicity_rank",
]
