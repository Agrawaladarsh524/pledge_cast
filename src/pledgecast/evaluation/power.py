"""Uncertainty and ceiling for every reported difference.

**The problem this module fixes.** The study reports differences between
experiments - ``expB_full`` minus ``exp0_null`` and so on - and until now those
differences were reported as bare numbers. A bare number invites two mistakes,
and the first version of the Reg 31 write-up made both:

*Reading a direction out of noise.* Measured here, a within-quarter AUC averaged
over 19 observation dates has a block-bootstrap SD of about 0.017, so its 95%
interval is roughly +/-0.033. The Reg 31 deltas came in at -0.014 to -0.018 and
were described as "18 of 18 negative". They are not negative. They are zero,
reported to more decimal places than the measurement supports. Counting signs
across models does not help: the models are fitted on the same rows, so their
errors agree, and 18 correlated coin flips are not 18 pieces of evidence.

*Mistaking no power for no effect.* A null result means nothing unless the
design could have detected an effect had one existed. With event features
present on only 8.25% of rows, that is a real worry rather than a rhetorical
one - and it is answerable. :func:`oracle_ceiling` fits nothing and cheats
completely: it hands the model the TRUE label for every row the treatment can
see, and leaves the baseline untouched everywhere else. No real model can beat
that. On this panel it returns +0.19, which says the design had room to find a
large effect and did not - so the null is a bounded measurement rather than a
shrug.

**Why dates are the resampling unit.** Rows inside one observation date share a
market regime; in a quarter where the index fell 20%, most companies crash
together. Resampling rows would treat those as independent observations and
report an interval several times too narrow. The block bootstrap resamples whole
dates, which keeps that correlation intact. There are only 19 of them, and that
small number is precisely the point - it is the honest sample size of this study
and the reason the intervals are as wide as they are.

**Why the delta bootstrap is paired.** Both experiments are scored on the same
dates and the same rows, so their per-date AUCs move together. Differencing
within each date first cancels that shared movement; taking two independent
intervals and comparing them would discard the pairing and overstate the
uncertainty by a factor of about two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.evaluation import metrics
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

# Verdict labels. A delta is only given a direction when its interval excludes
# zero; otherwise it is ZERO, however many decimal places it has.
ZERO = "ZERO"
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
UNKNOWN = "UNKNOWN"


def per_date_auc(oof: pd.DataFrame, *, min_rows: int = 10) -> pd.Series:
    """Within-date ROC-AUC, indexed by observation date.

    Delegates to :func:`metrics.within_quarter_auc` rather than recomputing, so
    a date this module scores is exactly a date the headline metric scored -
    including the skip rules for single-class and thin dates.
    """
    if oof.empty:
        return pd.Series(dtype=float, name="auc")

    _, detail = metrics.within_quarter_auc(
        oof["label"],
        oof["probability"],
        oof["observation_date"],
        min_rows=min_rows,
        return_detail=True,
    )
    scored = detail[~detail["skipped"]]
    return pd.Series(
        scored["auc"].to_numpy(dtype=float),
        index=scored["group"].to_numpy(),
        name="auc",
    ).sort_index()


def _bootstrap_means(values: np.ndarray, *, n_bootstrap: int, seed: int) -> np.ndarray:
    """Means of ``n_bootstrap`` resamples of ``values``, drawn with replacement."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return values[draws].mean(axis=1)


def _interval(samples: np.ndarray, confidence_level: float) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    return (
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    )


def verdict(low: float | None, high: float | None) -> str:
    """A direction only when the interval excludes zero."""
    if low is None or high is None or np.isnan(low) or np.isnan(high):
        return UNKNOWN
    if low > 0:
        return POSITIVE
    if high < 0:
        return NEGATIVE
    return ZERO


def auc_ci(
    oof: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_rows: int = 10,
) -> dict:
    """Block-bootstrap interval for one experiment's within-quarter AUC."""
    scores = per_date_auc(oof, min_rows=min_rows)
    if len(scores) < 2:
        return {"auc": float(scores.iloc[0]) if len(scores) == 1 else None, "n_dates": len(scores)}

    values = scores.to_numpy(dtype=float)
    samples = _bootstrap_means(values, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _interval(samples, confidence_level)
    return {
        "auc": float(values.mean()),
        "n_dates": int(len(values)),
        "sd": float(samples.std(ddof=1)),
        "ci_low": low,
        "ci_high": high,
        "half_width": float((high - low) / 2.0),
        "per_date_min": float(values.min()),
        "per_date_max": float(values.max()),
    }


def paired_delta_ci(
    treatment_oof: pd.DataFrame,
    control_oof: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
    min_rows: int = 10,
) -> dict:
    """Interval for ``treatment - control`` on the primary metric.

    Paired by observation date. Only dates scored by BOTH experiments are used;
    a date one side skipped for having a single class cannot contribute a
    difference, and quietly treating its absence as a zero would pull every
    delta toward zero.
    """
    treatment = per_date_auc(treatment_oof, min_rows=min_rows)
    control = per_date_auc(control_oof, min_rows=min_rows)

    shared = treatment.index.intersection(control.index)
    dropped = sorted(set(treatment.index).symmetric_difference(control.index))
    if dropped:
        logger.info(
            "paired delta: %d date(s) scored by only one side, excluded: %s", len(dropped), dropped
        )

    if len(shared) < 2:
        return {"delta": None, "n_dates": len(shared), "verdict": UNKNOWN, "dropped_dates": dropped}

    differences = (treatment.loc[shared] - control.loc[shared]).to_numpy(dtype=float)
    samples = _bootstrap_means(differences, n_bootstrap=n_bootstrap, seed=seed)
    low, high = _interval(samples, confidence_level)

    return {
        "delta": float(differences.mean()),
        "n_dates": int(len(shared)),
        "sd": float(samples.std(ddof=1)),
        "ci_low": low,
        "ci_high": high,
        "half_width": float((high - low) / 2.0),
        # The smallest effect this design could call non-zero. Report it next to
        # every null: "we found nothing" and "nothing this small is findable
        # here" are different claims.
        "min_detectable_effect": float((high - low) / 2.0),
        "dates_better": int((differences > 0).sum()),
        "dates_worse": int((differences < 0).sum()),
        "verdict": verdict(low, high),
        "dropped_dates": dropped,
    }


def oracle_ceiling(
    control_oof: pd.DataFrame,
    defined: pd.Series | np.ndarray,
    *,
    min_rows: int = 10,
) -> dict:
    """The largest delta the treatment could achieve if it were always right.

    ``defined`` marks the rows the treatment can actually see - rows where its
    extra features carry a value. The oracle keeps the control's ranking on
    every other row and sorts the visible rows perfectly: crashes to the top,
    survivors to the bottom. Nothing is fitted and nothing is validated, because
    this is not a model - it is an upper bound.

    The control's probabilities are converted to a within-date percentile rank
    first, so the oracle's +/-1 offsets are commensurate with them regardless of
    how the control's probabilities happen to be scaled. Ranking within the date
    also matches how the primary metric reads the scores: AUC cares only about
    order inside a date.

    Reading the result: a ceiling near zero means the design could never have
    detected anything and any null is uninformative. A large ceiling next to a
    zero measurement means the effect is genuinely absent.
    """
    if control_oof.empty:
        return {"ceiling": None, "coverage": 0.0}

    frame = control_oof.copy()
    frame["_defined"] = np.asarray(defined, dtype=bool)

    baseline_rank = frame.groupby("observation_date")["probability"].rank(pct=True)
    base = metrics.within_quarter_auc(
        frame["label"], baseline_rank, frame["observation_date"], min_rows=min_rows
    )

    # label 1 -> rank + 1 (above every unseen row); label 0 -> rank - 1 (below).
    offset = np.where(frame["_defined"], (frame["label"].to_numpy(dtype=float) - 0.5) * 2.0, 0.0)
    oracle = metrics.within_quarter_auc(
        frame["label"], baseline_rank + offset, frame["observation_date"], min_rows=min_rows
    )

    if base is None or oracle is None:
        return {"ceiling": None, "coverage": float(frame["_defined"].mean())}

    return {
        "baseline_auc": float(base),
        "oracle_auc": float(oracle),
        "ceiling": float(oracle - base),
        "coverage": float(frame["_defined"].mean()),
        "rows_visible": int(frame["_defined"].sum()),
        "rows_total": int(len(frame)),
    }


def rows_with_any_feature(panel: pd.DataFrame, features: list[str]) -> pd.Series:
    """Which rows carry usable information for ``features``.

    A count feature sitting at 0 is genuinely "no activity", not missing - but
    for a ceiling it is also indistinguishable from every other zero row, so it
    conveys nothing that could re-rank anything. A row counts as visible when at
    least one of the features is present AND non-zero. This is deliberately the
    generous reading: it makes the ceiling as high as it can honestly be, which
    is the conservative direction when the ceiling is being used to defend a
    null.
    """
    present = [f for f in features if f in panel.columns]
    if not present:
        return pd.Series(False, index=panel.index)
    block = panel[present]
    return (block.notna() & (block != 0)).any(axis=1)


def assess(
    treatment_oof: pd.DataFrame,
    control_oof: pd.DataFrame,
    panel: pd.DataFrame,
    extra_features: list[str],
    settings,
) -> dict:
    """Delta, interval, verdict and ceiling for one experiment pair.

    ``extra_features`` is what the treatment has and the control does not - the
    features whose value is being measured.
    """
    result = paired_delta_ci(
        treatment_oof,
        control_oof,
        n_bootstrap=settings.power.n_bootstrap,
        confidence_level=settings.power.confidence_level,
        seed=settings.power.bootstrap_seed,
        min_rows=settings.evaluation.min_rows_per_quarter_for_auc,
    )

    if control_oof.empty or not extra_features:
        result["ceiling"] = None
        return result

    # Align the visibility mask to the control's out-of-fold rows. A left merge
    # on (symbol, date) rather than positional indexing: OOF rows are assembled
    # fold by fold and are not in panel order.
    visible = panel[["symbol", "observation_date"]].copy()
    visible["_visible"] = rows_with_any_feature(panel, extra_features).to_numpy()
    joined = control_oof.merge(visible, on=["symbol", "observation_date"], how="left")

    result.update(
        oracle_ceiling(
            control_oof,
            joined["_visible"].fillna(False).to_numpy(dtype=bool),
            min_rows=settings.evaluation.min_rows_per_quarter_for_auc,
        )
    )
    return result


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Assessment rows as a printable table, widest-evidence first."""
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    order = [
        c
        for c in (
            "experiment",
            "model",
            "delta",
            "ci_low",
            "ci_high",
            "min_detectable_effect",
            "ceiling",
            "coverage",
            "n_dates",
            "verdict",
        )
        if c in frame.columns
    ]
    return frame[order]


__all__ = [
    "NEGATIVE",
    "POSITIVE",
    "UNKNOWN",
    "ZERO",
    "assess",
    "auc_ci",
    "oracle_ceiling",
    "paired_delta_ci",
    "per_date_auc",
    "rows_with_any_feature",
    "summarise",
    "verdict",
]
