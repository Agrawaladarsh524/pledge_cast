"""Uncertainty and ceiling - the machinery that makes the null falsifiable.

Three properties here are load-bearing, and each corresponds to a mistake the
first Reg 31 write-up actually made:

**A delta inside its interval is ZERO, not "slightly negative".** The reported
deltas were -0.014 to -0.018 against an interval half-width of about 0.033, and
were described as "18 of 18 negative". :func:`verdict` exists so that reading is
no longer possible.

**The interval must be paired.** Both experiments score the same dates, so their
errors move together. Differencing per date first cancels that; two independent
intervals would be roughly twice too wide and would hide a real effect.

**A null needs a ceiling.** With event features present on 8% of rows, "we found
nothing" and "nothing was findable" look identical in the results table. The
oracle separates them, so it is tested at both extremes: full visibility must
give a large ceiling, zero visibility must give exactly zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pledgecast.evaluation import power

pytestmark = pytest.mark.leakage


def _oof(n_dates: int = 8, n_per_date: int = 40, *, signal: float = 0.0, seed: int = 0):
    """Synthetic out-of-fold predictions with a tunable amount of real signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2024-{d + 1:02d}-01"
        label = rng.integers(0, 2, n_per_date)
        noise = rng.normal(0, 1, n_per_date)
        rows.append(
            pd.DataFrame(
                {
                    "symbol": [f"S{i:03d}" for i in range(n_per_date)],
                    "observation_date": date,
                    "label": label.astype(float),
                    "probability": noise + signal * label,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# verdict - the rule that replaces reading a sign                              #
# --------------------------------------------------------------------------- #
def test_an_interval_straddling_zero_is_zero_whatever_the_point_estimate():
    assert power.verdict(-0.045, 0.012) == power.ZERO
    assert power.verdict(-0.001, 0.400) == power.ZERO


def test_a_direction_is_only_given_when_the_interval_excludes_zero():
    assert power.verdict(0.01, 0.09) == power.POSITIVE
    assert power.verdict(-0.16, -0.09) == power.NEGATIVE


def test_a_missing_interval_is_unknown_rather_than_zero():
    assert power.verdict(None, None) == power.UNKNOWN
    assert power.verdict(float("nan"), 0.1) == power.UNKNOWN


# --------------------------------------------------------------------------- #
# per-date AUC                                                                 #
# --------------------------------------------------------------------------- #
def test_per_date_auc_returns_one_score_per_observation_date():
    scores = power.per_date_auc(_oof(n_dates=5), min_rows=10)
    assert len(scores) == 5
    assert scores.index.is_monotonic_increasing


def test_a_single_class_date_is_dropped_not_scored_as_zero():
    frame = _oof(n_dates=3)
    frame.loc[frame["observation_date"] == "2024-02-01", "label"] = 1.0
    scores = power.per_date_auc(frame, min_rows=10)
    assert "2024-02-01" not in scores.index
    assert len(scores) == 2


def test_a_thin_date_is_dropped_by_min_rows():
    frame = _oof(n_dates=2, n_per_date=40)
    frame = frame[~((frame["observation_date"] == "2024-02-01") & (frame.index % 40 > 4))]
    assert len(power.per_date_auc(frame, min_rows=10)) == 1


# --------------------------------------------------------------------------- #
# the paired delta                                                             #
# --------------------------------------------------------------------------- #
def test_two_identical_experiments_give_exactly_zero_delta_and_a_zero_verdict():
    frame = _oof(seed=1)
    result = power.paired_delta_ci(frame, frame, n_bootstrap=200, seed=0)
    assert result["delta"] == pytest.approx(0.0, abs=1e-12)
    assert result["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert result["verdict"] == power.ZERO


def test_a_large_real_improvement_is_reported_as_positive():
    control = _oof(n_dates=12, n_per_date=80, signal=0.0, seed=2)
    treatment = control.copy()
    # Same rows, same dates - only the ranking improves.
    treatment["probability"] = treatment["probability"] + 2.0 * treatment["label"]

    result = power.paired_delta_ci(treatment, control, n_bootstrap=500, seed=0)
    assert result["delta"] > 0.15
    assert result["verdict"] == power.POSITIVE
    assert result["ci_low"] > 0


def test_pure_noise_between_two_experiments_is_not_called_a_direction():
    """The exact failure mode being guarded: a small delta from nothing at all."""
    control = _oof(n_dates=11, n_per_date=60, seed=3)
    treatment = control.copy()
    rng = np.random.default_rng(9)
    treatment["probability"] = treatment["probability"] + rng.normal(0, 0.05, len(treatment))

    result = power.paired_delta_ci(treatment, control, n_bootstrap=1000, seed=0)
    assert abs(result["delta"]) < 0.05
    assert result["verdict"] == power.ZERO, "noise was reported as a direction"


def test_pairing_makes_the_interval_tighter_than_treating_the_runs_as_independent():
    """Why the bootstrap is paired at all.

    Per-date AUC varies far more across dates than the two experiments differ on
    any one date. Differencing inside each date removes that shared variation;
    an unpaired interval would carry it and be much wider.
    """
    control = _oof(n_dates=12, n_per_date=60, seed=4)
    treatment = control.copy()
    treatment["probability"] = treatment["probability"] + 0.3 * treatment["label"]

    paired = power.paired_delta_ci(treatment, control, n_bootstrap=1000, seed=0)
    a = power.auc_ci(treatment, n_bootstrap=1000, seed=0)
    b = power.auc_ci(control, n_bootstrap=1000, seed=0)
    unpaired_half_width = float(np.hypot(a["half_width"], b["half_width"]))

    assert paired["half_width"] < unpaired_half_width


def test_a_date_scored_by_only_one_side_is_excluded_and_reported():
    control = _oof(n_dates=4, seed=5)
    treatment = control.copy()
    treatment.loc[treatment["observation_date"] == "2024-03-01", "label"] = 1.0

    result = power.paired_delta_ci(treatment, control, n_bootstrap=200, seed=0)
    assert result["n_dates"] == 3
    assert "2024-03-01" in result["dropped_dates"]


def test_the_bootstrap_is_reproducible_for_a_fixed_seed():
    treatment, control = _oof(seed=6), _oof(seed=7)
    a = power.paired_delta_ci(treatment, control, n_bootstrap=300, seed=11)
    b = power.paired_delta_ci(treatment, control, n_bootstrap=300, seed=11)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


def test_min_detectable_effect_is_reported_so_a_null_can_be_read():
    result = power.paired_delta_ci(_oof(seed=8), _oof(seed=8), n_bootstrap=300, seed=0)
    assert "min_detectable_effect" in result
    assert result["min_detectable_effect"] >= 0


# --------------------------------------------------------------------------- #
# the oracle ceiling                                                           #
# --------------------------------------------------------------------------- #
def test_an_oracle_that_sees_nothing_can_improve_nothing():
    """Zero coverage must give exactly zero ceiling, not a small positive number."""
    control = _oof(n_dates=6, seed=10)
    result = power.oracle_ceiling(control, np.zeros(len(control), dtype=bool))
    assert result["ceiling"] == pytest.approx(0.0, abs=1e-12)
    assert result["coverage"] == 0.0


def test_an_oracle_that_sees_everything_reaches_a_perfect_ranking():
    control = _oof(n_dates=6, seed=11)
    result = power.oracle_ceiling(control, np.ones(len(control), dtype=bool))
    assert result["oracle_auc"] == pytest.approx(1.0)
    assert result["ceiling"] > 0.3


def test_partial_visibility_gives_a_ceiling_between_the_two_extremes():
    """THE number that decides whether a null means anything.

    With the treatment able to see only a minority of rows, the best it could
    possibly do is bounded well below perfect - and the size of that bound is
    what says whether the design had a chance.
    """
    control = _oof(n_dates=8, n_per_date=50, seed=12)
    rng = np.random.default_rng(0)
    visible = rng.random(len(control)) < 0.25

    partial = power.oracle_ceiling(control, visible)
    full = power.oracle_ceiling(control, np.ones(len(control), dtype=bool))

    assert 0 < partial["ceiling"] < full["ceiling"]
    assert partial["coverage"] == pytest.approx(visible.mean())


def test_the_ceiling_never_penalises_the_baseline_on_rows_it_cannot_see():
    """Unseen rows must keep their baseline ordering exactly.

    If the oracle disturbed them, the ceiling would mix "how much the treatment
    could add" with "how much the control was damaged", and could even come out
    negative.
    """
    control = _oof(n_dates=6, n_per_date=40, signal=1.5, seed=13)
    rng = np.random.default_rng(1)
    visible = rng.random(len(control)) < 0.3

    result = power.oracle_ceiling(control, visible)
    assert result["ceiling"] >= 0, "the oracle made a well-ranked baseline worse"
    assert result["oracle_auc"] >= result["baseline_auc"]


# --------------------------------------------------------------------------- #
# visibility mask                                                              #
# --------------------------------------------------------------------------- #
def test_a_row_whose_features_are_all_zero_is_not_counted_as_visible():
    """A count feature at 0 is real information but cannot re-rank anything -
    every other zero row is tied with it."""
    panel = pd.DataFrame({"event_count_90d": [0.0, 0.0, 3.0], "event_net_90d": [0.0, 0.0, -1.0]})
    visible = power.rows_with_any_feature(panel, ["event_count_90d", "event_net_90d"])
    assert visible.tolist() == [False, False, True]


def test_a_row_visible_through_any_one_feature_counts():
    panel = pd.DataFrame({"a": [0.0, 0.0], "b": [np.nan, 7.0]})
    assert power.rows_with_any_feature(panel, ["a", "b"]).tolist() == [False, True]


def test_missing_is_not_visible():
    panel = pd.DataFrame({"a": [np.nan, np.nan]})
    assert power.rows_with_any_feature(panel, ["a"]).tolist() == [False, False]


def test_unknown_feature_names_do_not_crash_the_mask():
    panel = pd.DataFrame({"a": [1.0]})
    assert power.rows_with_any_feature(panel, ["nope"]).tolist() == [False]


# --------------------------------------------------------------------------- #
# assemble                                                                     #
# --------------------------------------------------------------------------- #
def test_assess_joins_the_mask_on_symbol_and_date_not_by_position(settings):
    """OOF rows are assembled fold by fold, so they are NOT in panel order.

    A positional join would silently attach the wrong company's visibility to
    every row, and the ceiling would still look plausible.
    """
    control = _oof(n_dates=4, n_per_date=30, seed=14)
    treatment = control.copy()

    panel = control[["symbol", "observation_date"]].copy()
    panel["event_count_90d"] = 0.0
    panel.loc[panel["symbol"] == "S000", "event_count_90d"] = 5.0
    # Shuffle so position no longer matches the OOF frame.
    panel = panel.sample(frac=1.0, random_state=0).reset_index(drop=True)

    result = power.assess(treatment, control, panel, ["event_count_90d"], settings)
    assert result["coverage"] == pytest.approx(1 / 30, abs=1e-9)


def test_assess_returns_no_ceiling_when_the_treatment_adds_no_features(settings):
    frame = _oof(n_dates=4, seed=15)
    assert power.assess(frame, frame, frame, [], settings)["ceiling"] is None


def test_summarise_orders_the_columns_for_reading():
    rows = [{"experiment": "x", "delta": 0.1, "verdict": "ZERO", "ci_low": 0.0, "ci_high": 0.2}]
    table = power.summarise(rows)
    assert list(table.columns)[0] == "experiment"
    assert list(table.columns)[-1] == "verdict"


def test_summarise_of_nothing_is_an_empty_frame_not_an_error():
    assert power.summarise([]).empty
