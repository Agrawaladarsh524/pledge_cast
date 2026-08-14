"""Leakage proofs - PLAN.md sec.8.1, sec.9.8, sec.10, sec.15.

    "Proves the pipeline is honest rather than claiming it."  (sec.8.1)

Four checks, in increasing order of how much they would hurt to fail:

  1. every panel row's ``submission_date <= observation_date``
  2. every label window starts strictly AFTER its observation date
  3. train and test fold dates are disjoint
  4. **shuffle the labels, retrain, and AUC must collapse to ~0.50** (sec.9.8)

Check 4 is the one that matters. sec.9.8 calls it non-negotiable: "Run this the
moment the first model trains. If it does not collapse, you have leakage - stop
everything and fix it before proceeding." A model that still scores well on
shuffled labels is reading something it should not be able to see.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pledgecast.exceptions import LeakageError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def check_submission_before_observation(
    panel_frame: pd.DataFrame, pledge_state: pd.DataFrame
) -> dict:
    """Check 1: nothing in the panel was filed after its observation date."""
    merged = panel_frame.merge(
        pledge_state[["symbol", "quarter_end", "submission_date"]],
        on=["symbol", "quarter_end"],
        how="left",
    )
    # Only rows that actually carry pledge data can violate the rule.
    has_data = merged["pledge_pct_promoter"].notna() & merged["submission_date"].notna()
    violations = merged[has_data & (merged["submission_date"] > merged["observation_date"])]

    result = {
        "check": "submission_date <= observation_date",
        "rows_checked": int(has_data.sum()),
        "violations": len(violations),
        "passed": violations.empty,
    }
    if not violations.empty:
        result["sample"] = violations.head(5)[
            ["symbol", "quarter_end", "submission_date", "observation_date"]
        ].to_dict("records")
    return result


def check_label_window_starts_after_observation(
    panel_frame: pd.DataFrame, prices: pd.DataFrame, horizon: int
) -> dict:
    """Check 2: the forward window opens strictly after the entry bar.

    Recomputes the label independently for a sample of rows and confirms it
    matches - if the pipeline had used ``P[t..t+h]`` instead of ``P[t+1..t+h]``,
    the entry bar itself would be inside the window and a row could be labelled
    on information from its own observation date.
    """
    labelled = panel_frame[panel_frame["label_is_valid"] == 1]
    if labelled.empty or prices.empty:
        return {"check": "label window starts after observation", "rows_checked": 0, "passed": True}

    sample = labelled.sample(min(200, len(labelled)), random_state=0)
    price_groups = {
        symbol: group.sort_values("trade_date") for symbol, group in prices.groupby("symbol")
    }

    mismatches = []
    for row in sample.itertuples():
        history = price_groups.get(row.symbol)
        if history is None:
            continue
        dates = history["trade_date"].to_numpy()
        close = history["adj_close"].to_numpy(dtype=float)

        position = np.searchsorted(dates, row.observation_date, side="right") - 1
        if position < 0 or position + horizon >= len(close):
            continue

        window = close[position + 1 : position + 1 + horizon]
        expected = window.min() / close[position] - 1.0
        if not np.isclose(expected, row.fwd_max_drawdown, atol=1e-6):
            mismatches.append(
                {
                    "symbol": row.symbol,
                    "observation_date": row.observation_date,
                    "stored": float(row.fwd_max_drawdown),
                    "recomputed": float(expected),
                }
            )

    return {
        "check": "label window starts strictly after observation",
        "rows_checked": len(sample),
        "violations": len(mismatches),
        "passed": not mismatches,
        "sample": mismatches[:5],
    }


def check_events_respect_disclosure_lag(
    panel_frame: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lag_days: int,
    window_days: int,
    min_pct_equity: float,
) -> dict:
    """Check 5: no Reg 31 event was counted before it could have been public.

    ``pledge_events.event_date`` is when the pledge was CREATED. Regulation 31
    allows up to 7 working days to disclose it, so an event dated the 1st may
    not have been knowable until the 10th. This is a different leak from the
    quarterly one in check 1 - the filing date is not stored anywhere, so the
    only defence is a time buffer, and the only way to know the buffer was
    applied is to recompute the feature both ways.

    For each sampled row the count is recomputed WITH the buffer and WITHOUT
    it. The stored value must match the buffered version. A row matching the
    unbuffered version where the two disagree is counting information that did
    not exist yet.
    """
    column = "event_count_90d"
    if column not in panel_frame.columns or events.empty:
        return {"check": "Reg 31 events respect the disclosure lag", "rows_checked": 0,
                "passed": True}

    material = events[events["pct_equity"].fillna(0.0) >= min_pct_equity]
    if material.empty:
        return {"check": "Reg 31 events respect the disclosure lag", "rows_checked": 0,
                "passed": True}

    by_symbol = {
        symbol: np.sort(group["event_date"].to_numpy())
        for symbol, group in material.groupby("symbol")
    }
    sample = panel_frame.sample(min(500, len(panel_frame)), random_state=0)

    violations = []
    checked = 0
    for row in sample.itertuples():
        dates = by_symbol.get(row.symbol)
        if dates is None or pd.isna(getattr(row, column)):
            continue
        checked += 1

        observation = pd.Timestamp(row.observation_date)
        buffered_end = (observation - pd.Timedelta(days=lag_days)).strftime("%Y-%m-%d")
        buffered_start = (
            observation - pd.Timedelta(days=lag_days + window_days)
        ).strftime("%Y-%m-%d")
        naive_end = observation.strftime("%Y-%m-%d")
        naive_start = (observation - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

        buffered = int(
            np.searchsorted(dates, buffered_end, "right")
            - np.searchsorted(dates, buffered_start, "left")
        )
        naive = int(
            np.searchsorted(dates, naive_end, "right")
            - np.searchsorted(dates, naive_start, "left")
        )
        stored = int(getattr(row, column))

        if stored != buffered:
            violations.append(
                {
                    "symbol": row.symbol,
                    "observation_date": row.observation_date,
                    "stored": stored,
                    "with_buffer": buffered,
                    "without_buffer": naive,
                }
            )

    return {
        "check": "Reg 31 events respect the disclosure lag",
        "rows_checked": checked,
        "violations": len(violations),
        "passed": not violations,
        "sample": violations[:5],
    }


def check_folds_disjoint(folds: list[dict]) -> dict:
    """Check 3: no observation date appears in both train and test of a fold."""
    violations = []
    for fold in folds:
        overlap = set(fold["train_dates"]) & set(fold["test_dates"])
        if overlap:
            violations.append({"fold": fold.get("fold"), "overlap": sorted(overlap)[:5]})
        # Walk-forward also requires every training date to precede the test date.
        if (
            fold["train_dates"]
            and fold["test_dates"]
            and max(fold["train_dates"]) >= min(fold["test_dates"])
        ):
            violations.append(
                {
                    "fold": fold.get("fold"),
                    "reason": "a training date is not strictly before the test date",
                    "max_train": max(fold["train_dates"]),
                    "min_test": min(fold["test_dates"]),
                }
            )

    return {
        "check": "train/test fold dates disjoint and ordered",
        "folds_checked": len(folds),
        "violations": len(violations),
        "passed": not violations,
        "sample": violations[:5],
    }


def label_shuffle_test(
    fit_and_score,
    frame: pd.DataFrame,
    *,
    seed: int,
    tolerance: float,
    n_repeats: int = 3,
) -> dict:
    """Check 4 (sec.9.8): shuffled labels must collapse AUC to ~0.50.

    ``fit_and_score(frame) -> float`` trains and returns the primary metric.
    Labels are shuffled WITHIN each observation date, which is the strict form
    of the test: it destroys the company-level signal while preserving the
    per-date class balance, so a model cannot score by learning "this quarter
    was bad for everyone".

    **Missing labels stay missing.** The permutation runs only over the non-null
    labels in each date. A plain permutation moves NaN around, so a row that was
    deliberately unlabelled (the embargo quarter, or the four rows voided
    because their drawdown window spans a demerger) can hand its NaN to a
    perfectly good training row - and XGBoost then rejects the fold with
    "Invalid classes inferred from unique values of y". Permuting in place keeps
    the valid/invalid partition exactly as the panel defined it.
    """
    rng = np.random.default_rng(seed)
    scores = []

    def permute_in_place(block: pd.Series) -> np.ndarray:
        values = block.to_numpy(dtype=float).copy()
        present = ~np.isnan(values)
        values[present] = rng.permutation(values[present])
        return values

    for _ in range(n_repeats):
        shuffled = frame.copy()
        shuffled["label"] = shuffled.groupby("observation_date")["label"].transform(
            permute_in_place
        )
        scores.append(float(fit_and_score(shuffled)))

    mean_score = float(np.mean(scores))
    passed = abs(mean_score - 0.5) <= tolerance

    result = {
        "check": "label-shuffle collapses AUC to ~0.50",
        "scores": scores,
        "mean": mean_score,
        "tolerance": tolerance,
        "passed": passed,
    }
    if not passed:
        logger.error(
            "LABEL-SHUFFLE TEST FAILED: mean AUC %.4f on shuffled labels (expected 0.50 +/- %.2f). "
            "This is leakage. sec.9.8: stop everything and fix it.",
            mean_score,
            tolerance,
        )
    return result


def run_all(
    panel_frame: pd.DataFrame,
    pledge_state: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizon: int,
    folds: list[dict] | None = None,
    events: pd.DataFrame | None = None,
    event_lag_days: int | None = None,
    event_window_days: int | None = None,
    min_event_pct_equity: float = 0.01,
    strict: bool = True,
) -> list[dict]:
    """Run every static check. Raises ``LeakageError`` if any fail and ``strict``."""
    results = [
        check_submission_before_observation(panel_frame, pledge_state),
        check_label_window_starts_after_observation(panel_frame, prices, horizon),
    ]
    if events is not None and event_lag_days is not None and event_window_days is not None:
        results.append(
            check_events_respect_disclosure_lag(
                panel_frame,
                events,
                lag_days=event_lag_days,
                window_days=event_window_days,
                min_pct_equity=min_event_pct_equity,
            )
        )
    if folds is not None:
        results.append(check_folds_disjoint(folds))

    failed = [r for r in results if not r["passed"]]
    for result in results:
        (logger.info if result["passed"] else logger.error)(
            "%s %s (%d violations)",
            "PASS" if result["passed"] else "FAIL",
            result["check"],
            result.get("violations", 0),
        )

    if failed and strict:
        raise LeakageError(f"{len(failed)} leakage check(s) failed: {[r['check'] for r in failed]}")
    return results


__all__ = [
    "check_events_respect_disclosure_lag",
    "check_folds_disjoint",
    "check_label_window_starts_after_observation",
    "check_submission_before_observation",
    "label_shuffle_test",
    "run_all",
]
