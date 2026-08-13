"""Metrics - PLAN.md sec.9.6.

    Within-quarter ROC-AUC  PRIMARY. "The only metric immune to the
                            market-timing confound. Pooled AUC looks better and
                            means less."
    PR-AUC                  correct shape for a screening tool
    Precision@20            literally what the dashboard shows
    Brier                   calibration, against a base-rate-only predictor
    Accuracy                NEVER - 75% by always predicting "no event"

**Why within-quarter is not a refinement but a correction.** Measured on this
panel, the event rate per observation date runs from 1.0% (2023-10-30) to 60.7%
(2022-05-02) - a 60x spread. A pooled AUC over that is largely rewarded for
telling one quarter from another, which is market timing rather than company
selection. Computing AUC inside each date and averaging asks the only question
a watchlist cares about: on this date, did the model rank the right companies
higher?

Every function returns per-group detail alongside the mean, because sec.9.6
insists: "Report per-fold spread, not just the mean."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def _clean(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = ~(pd.isna(y_true) | pd.isna(y_score))
    return np.asarray(y_true)[mask].astype(float), np.asarray(y_score)[mask].astype(float)


def pooled_auc(y_true, y_score) -> float | None:
    """Plain ROC-AUC. Reported for contrast only - never as the headline."""
    y_true, y_score = _clean(y_true, y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def within_quarter_auc(
    y_true,
    y_score,
    groups,
    *,
    min_rows: int = 10,
    return_detail: bool = False,
) -> float | tuple[float | None, pd.DataFrame]:
    """PRIMARY METRIC. Mean of per-observation-date ROC-AUC.

    A date is skipped when it has fewer than ``min_rows`` rows or only one
    class present - AUC is undefined there, not zero. Skips are counted and
    returned so they can be reported rather than hidden.
    """
    frame = pd.DataFrame({"y": y_true, "p": y_score, "g": groups}).dropna()

    rows = []
    for group, block in frame.groupby("g"):
        n = len(block)
        positives = int(block["y"].sum())
        if n < min_rows or positives == 0 or positives == n:
            rows.append(
                {
                    "group": group,
                    "n": n,
                    "n_positive": positives,
                    "auc": np.nan,
                    "skipped": True,
                    "reason": "too few rows" if n < min_rows else "single class",
                }
            )
            continue
        rows.append(
            {
                "group": group,
                "n": n,
                "n_positive": positives,
                "auc": float(roc_auc_score(block["y"], block["p"])),
                "skipped": False,
                "reason": "",
            }
        )

    detail = pd.DataFrame(rows)
    scored = detail[~detail["skipped"]]
    mean = float(scored["auc"].mean()) if not scored.empty else None

    if return_detail:
        return mean, detail
    return mean


def precision_at_k(y_true, y_score, k: int = 20) -> float | None:
    """Share of the top-k highest-risk companies that actually had an event."""
    y_true, y_score = _clean(y_true, y_score)
    if len(y_true) == 0:
        return None
    k = min(k, len(y_true))
    top = np.argsort(-y_score)[:k]
    return float(y_true[top].mean())


def precision_at_k_by_group(
    y_true, y_score, groups, k: int = 20
) -> tuple[float | None, pd.DataFrame]:
    """Precision@k computed per date - what the watchlist actually delivers."""
    frame = pd.DataFrame({"y": y_true, "p": y_score, "g": groups}).dropna()
    rows = []
    for group, block in frame.groupby("g"):
        rows.append(
            {
                "group": group,
                "n": len(block),
                "precision_at_k": precision_at_k(block["y"], block["p"], k),
                "base_rate": float(block["y"].mean()),
            }
        )
    detail = pd.DataFrame(rows)
    mean = float(detail["precision_at_k"].mean()) if not detail.empty else None
    return mean, detail


def pr_auc(y_true, y_score) -> float | None:
    """Average precision. Report against the base rate, not against 0.5."""
    y_true, y_score = _clean(y_true, y_score)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def brier(y_true, y_score) -> float | None:
    y_true, y_score = _clean(y_true, y_score)
    if len(y_true) == 0:
        return None
    return float(brier_score_loss(y_true, y_score))


def brier_skill_score(y_true, y_score) -> float | None:
    """Brier against a base-rate-only predictor (sec.9.6).

    Positive means better calibrated than always predicting the base rate;
    zero or negative means the model adds nothing.
    """
    y_true, y_score = _clean(y_true, y_score)
    if len(y_true) == 0:
        return None
    reference = brier_score_loss(y_true, np.full_like(y_true, y_true.mean()))
    if reference == 0:
        return None
    return float(1.0 - brier_score_loss(y_true, y_score) / reference)


def evaluate(
    y_true,
    y_score,
    groups,
    *,
    k: int = 20,
    min_rows: int = 10,
) -> dict[str, float | None]:
    """Every metric sec.9.6 asks for, in one call. Accuracy is deliberately absent."""
    y_array = np.asarray(y_true, dtype=float)
    wq, detail = within_quarter_auc(y_true, y_score, groups, min_rows=min_rows, return_detail=True)
    pak, _ = precision_at_k_by_group(y_true, y_score, groups, k=k)

    scored = detail[~detail["skipped"]] if not detail.empty else detail
    return {
        "within_quarter_auc": wq,
        "within_quarter_auc_std": float(scored["auc"].std()) if len(scored) > 1 else None,
        "within_quarter_auc_min": float(scored["auc"].min()) if not scored.empty else None,
        "within_quarter_auc_max": float(scored["auc"].max()) if not scored.empty else None,
        "n_dates_scored": int(len(scored)),
        "n_dates_skipped": int(detail["skipped"].sum()) if not detail.empty else 0,
        "pooled_auc": pooled_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "base_rate": float(np.nanmean(y_array)) if len(y_array) else None,
        f"precision_at_{k}": pak,
        "brier": brier(y_true, y_score),
        "brier_skill_score": brier_skill_score(y_true, y_score),
    }


__all__ = [
    "brier",
    "brier_skill_score",
    "evaluate",
    "pooled_auc",
    "pr_auc",
    "precision_at_k",
    "precision_at_k_by_group",
    "within_quarter_auc",
]
