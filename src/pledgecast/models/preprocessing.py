"""Fold-local preprocessing - PLAN.md sec.9.4, sec.10.

    "Fold hygiene: scaling, imputation and winsorisation are fit on the TRAINING
     FOLD ONLY and then applied to the test fold. Never compute these statistics
     globally."

Fitting a scaler on all the data before splitting is the quietest form of
leakage there is: the test fold's distribution reaches the model through the
mean and variance, and nothing about the result looks wrong. Every transformer
here is fit inside ``Pipeline.fit`` on the training fold, so the discipline is
structural rather than remembered.

**Per-model paths, decided by measurement.** sec.10 specifies NaN handling for
XGBoost (native) and LogReg (median-impute + missingness indicator) but is
silent on RandomForest. Measured on 2026-08-13 against this exact stack:
sklearn 1.5.2's RandomForest accepts NaN natively, so it takes the same raw path
as XGBoost. Only LogReg imputes, and only LogReg scales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to training-fold quantiles (sec.10 "Outliers").

    scikit-learn has no winsoriser, and the two obvious substitutes are both
    wrong here: ``FunctionTransformer`` cannot learn quantiles from the training
    fold, and ``RobustScaler`` rescales rather than clips.

    **NaN-safe by design.** Uses ``nanpercentile`` to learn the bounds and
    ``np.clip``, which leaves NaN untouched. That matters because XGBoost and
    RandomForest rely on receiving real NaNs - a winsoriser that filled them
    would silently disable native missing-value handling and turn "unknown" into
    a hard number.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):  # noqa: N803, ARG002 - sklearn API
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        with np.errstate(invalid="ignore"):
            self.lower_ = np.nanpercentile(values, self.lower_quantile * 100, axis=0)
            self.upper_ = np.nanpercentile(values, self.upper_quantile * 100, axis=0)
        # An all-NaN column yields NaN bounds; clipping to those would erase the
        # column, so fall back to leaving it alone.
        self.lower_ = np.where(np.isnan(self.lower_), -np.inf, self.lower_)
        self.upper_ = np.where(np.isnan(self.upper_), np.inf, self.upper_)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        return np.clip(values, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [], dtype=object)


def build_pipeline(estimator, spec, settings) -> Pipeline:
    """Assemble the fold-local preprocessing chain for one model.

    Order is winsorise -> impute -> scale. Winsorising first means the imputed
    median is computed on already-clipped data, so a single extreme outlier
    cannot drag the fill value.
    """
    steps: list[tuple[str, object]] = [
        (
            "winsorize",
            Winsorizer(
                lower_quantile=settings.preprocessing.winsorize_lower_quantile,
                upper_quantile=settings.preprocessing.winsorize_upper_quantile,
            ),
        )
    ]

    if spec.requires_imputation:
        steps.append(
            (
                "impute",
                SimpleImputer(
                    strategy=settings.preprocessing.imputation_strategy,
                    add_indicator=settings.preprocessing.add_missing_indicator,
                    keep_empty_features=True,
                ),
            )
        )
    if spec.requires_scaling:
        steps.append(("scale", StandardScaler()))

    steps.append(("model", estimator))
    return Pipeline(steps)


def prepare_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature matrix with explicit float dtype (sec.10 "Incorrect feature types").

    Missing columns fail loudly rather than arriving as a silent NaN column.
    """
    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"panel is missing feature column(s): {missing}")
    return frame[features].to_numpy(dtype=float)


__all__ = ["Winsorizer", "build_pipeline", "prepare_matrix"]
