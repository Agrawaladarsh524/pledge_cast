"""Leakage proofs - PLAN.md sec.15, the highest-value tests in the suite.

    (4) submission_date <= observation_date for every panel row ·
        label window starts strictly after observation ·
        train/test fold dates disjoint ·
        label-shuffle collapses AUC to ~0.5                       [***]

sec.9.8 is unambiguous about the last one: "Shuffle the labels, retrain, confirm
AUC collapses to ~0.50. If it does not collapse, you have leakage - stop
everything and fix it."

Each test here is written to FAIL on a leak that is deliberately introduced,
not merely to pass on clean data. A test that only ever sees correct input
proves nothing about what it would catch - so every check has a negative twin
that plants the exact violation and asserts it is caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pledgecast.data import panel as panel_module
from pledgecast.evaluation import leakage
from pledgecast.exceptions import InsufficientDataError, LeakageError
from pledgecast.training import walkforward as wf

pytestmark = [pytest.mark.critical, pytest.mark.leakage]


# --------------------------------------------------------------------------- #
# 1. submission_date <= observation_date                                       #
# --------------------------------------------------------------------------- #
def test_every_panel_row_was_filed_before_its_observation_date():
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "quarter_end": ["2024-03-31", "2024-03-31"],
            "observation_date": ["2024-04-30", "2024-04-30"],
            "submission_date": ["2024-04-15", "2024-04-29"],
        }
    )
    panel_module.assert_no_leakage(frame)  # must not raise


def test_a_filing_after_the_observation_date_is_caught():
    """The negative twin: one row filed a day late must raise."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "quarter_end": ["2024-03-31", "2024-03-31"],
            "observation_date": ["2024-04-30", "2024-04-30"],
            # BBB filed the day AFTER the prediction would have been made.
            "submission_date": ["2024-04-15", "2024-05-01"],
        }
    )
    with pytest.raises(LeakageError, match="filed after"):
        panel_module.assert_no_leakage(frame)


def test_late_filings_are_blanked_rather_than_silently_kept():
    """sec.9.3: a late filing must not reach the row, but the row survives."""
    grid = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "quarter_end": ["2024-03-31", "2024-03-31"],
            "observation_date": ["2024-04-30", "2024-04-30"],
        }
    )
    features = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "quarter_end": ["2024-03-31", "2024-03-31"],
            "submission_date": ["2024-04-15", "2024-05-20"],
            "pledge_pct_promoter": [30.0, 90.0],
        }
    )
    merged = panel_module.apply_point_in_time_filter(grid, features, strict=True)

    assert len(merged) == 2, "the company must not be dropped, only its unfiled data"
    late = merged[merged["symbol"] == "BBB"].iloc[0]
    assert pd.isna(late["pledge_pct_promoter"]), "the future value leaked into the panel"
    on_time = merged[merged["symbol"] == "AAA"].iloc[0]
    assert on_time["pledge_pct_promoter"] == 30.0


# --------------------------------------------------------------------------- #
# 2. the label window starts strictly after the observation                    #
# --------------------------------------------------------------------------- #
def _prices(symbol: str, start: str, n: int, step: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": [d.date().isoformat() for d in dates],
            "adj_close": 100.0 + step * np.arange(n),
            "volume": 1_000.0,
        }
    )


def _dip_at_entry(n: int = 200, entry: int = 50) -> pd.DataFrame:
    """A series whose single lowest point IS the observation bar.

    Constructed so the off-by-one actually changes the answer. With the dip at
    the entry bar:

        correct   min(P[t+1 .. t+60]) / P[t] - 1  =  100/50 - 1  = +1.00
        leaked    min(P[t   .. t+60]) / P[t] - 1  =   50/50 - 1  =  0.00

    On a monotonic series both give the same number, so a test built on one
    would pass whether the code was right or wrong.
    """
    frame = _prices("AAA", "2024-01-01", n, step=0.0)
    frame.loc[entry, "adj_close"] = 50.0
    return frame


def test_label_window_starts_strictly_after_the_observation_date():
    prices = _dip_at_entry()
    close = prices["adj_close"].to_numpy()
    entry = 50
    correct = close[entry + 1 : entry + 61].min() / close[entry] - 1.0

    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "observation_date": [prices["trade_date"].iloc[entry]],
            "fwd_max_drawdown": [correct],
            "label": [0],
            "label_is_valid": [1],
        }
    )
    result = leakage.check_label_window_starts_after_observation(frame, prices, horizon=60)
    assert result["passed"], result
    assert result["rows_checked"] == 1


def test_a_label_window_that_included_the_entry_bar_is_caught():
    """The negative twin.

    Stores the drawdown a leaky pipeline would produce - one measured over
    ``P[t .. t+h]`` rather than ``P[t+1 .. t+h]``, so the entry bar sits inside
    its own window. Check 2 recomputes independently and must reject it.
    """
    prices = _dip_at_entry()
    close = prices["adj_close"].to_numpy()
    entry = 50
    leaked = close[entry : entry + 60].min() / close[entry] - 1.0

    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "observation_date": [prices["trade_date"].iloc[entry]],
            "fwd_max_drawdown": [leaked],
            "label": [1],
            "label_is_valid": [1],
        }
    )
    result = leakage.check_label_window_starts_after_observation(frame, prices, horizon=60)
    assert not result["passed"], "a window containing its own entry bar was reported clean"
    assert result["violations"] == 1


# --------------------------------------------------------------------------- #
# 3. train/test fold dates disjoint and ordered                                #
# --------------------------------------------------------------------------- #
def _panel(n_dates: int = 12, n_symbols: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [f"2022-{m:02d}-01" for m in range(1, n_dates + 1)]
    rows = []
    for date in dates:
        for i in range(n_symbols):
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "observation_date": date,
                    "label": int(rng.random() < 0.25),
                    "label_is_valid": 1,
                    "volatility_90d": float(rng.normal(0.4, 0.1)),
                    "log_turnover_90d": float(rng.normal(18, 1)),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_folds_are_disjoint_and_ordered():
    plan = wf.generate_folds(_panel(), min_train_quarters=4)
    result = leakage.check_folds_disjoint([f.to_dict() for f in plan.folds])
    assert result["passed"], result
    assert len(plan.folds) == 8, "12 labelled dates minus 4 held for training"


def test_a_fold_whose_training_set_contains_its_test_date_is_caught():
    """The negative twin - the exact mistake k-fold CV would make."""
    corrupted = [
        {
            "fold": 0,
            "train_dates": ["2022-01-01", "2022-02-01", "2022-03-01"],
            # the test date is also in the training set
            "test_dates": ["2022-02-01"],
        }
    ]
    result = leakage.check_folds_disjoint(corrupted)
    assert not result["passed"], "an overlapping fold was reported as clean"


def test_training_dates_always_precede_the_test_date():
    plan = wf.generate_folds(_panel(), min_train_quarters=4)
    for fold in plan.folds:
        assert max(fold.train_dates) < fold.test_date, f"fold {fold.index} trains on the future"


def test_the_embargo_quarter_is_never_trainable():
    """sec.9.4: the final quarter can be featured but never labelled."""
    frame = _panel()
    last = frame["observation_date"].max()
    frame.loc[frame["observation_date"] == last, ["label", "label_is_valid"]] = [np.nan, 0]

    plan = wf.generate_folds(frame, min_train_quarters=4)
    assert last in plan.embargoed_dates
    assert all(last not in fold.train_dates for fold in plan.folds)
    assert all(fold.test_date != last for fold in plan.folds)


def test_folds_cannot_be_built_without_enough_history():
    with pytest.raises(InsufficientDataError):
        wf.generate_folds(_panel(n_dates=3), min_train_quarters=8)


# --------------------------------------------------------------------------- #
# 4. the label-shuffle test (sec.9.8)                                          #
# --------------------------------------------------------------------------- #
def test_label_shuffle_collapses_a_real_model_to_chance():
    """sec.9.8, the non-negotiable check, on a model that genuinely learns.

    The panel is built so ``volatility_90d`` really does predict the label, so
    the unshuffled model scores well above chance. Shuffling must destroy that
    and nothing else.
    """
    from sklearn.linear_model import LogisticRegression

    from pledgecast.evaluation import metrics

    rng = np.random.default_rng(7)
    rows = []
    for date in [f"2022-{m:02d}-01" for m in range(1, 13)]:
        for i in range(40):
            volatility = float(rng.uniform(0.1, 0.9))
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "observation_date": date,
                    "volatility_90d": volatility,
                    # a genuine relationship the model can find
                    "label": int(rng.random() < volatility),
                    "label_is_valid": 1,
                }
            )
    frame = pd.DataFrame(rows)

    def fit_and_score(data: pd.DataFrame) -> float:
        model = LogisticRegression(max_iter=500)
        x = data[["volatility_90d"]].to_numpy(dtype=float)
        y = data["label"].to_numpy(dtype=float)
        model.fit(x, y)
        score = metrics.within_quarter_auc(
            y, model.predict_proba(x)[:, 1], data["observation_date"], min_rows=10
        )
        return 0.5 if score is None else float(score)

    honest = fit_and_score(frame)
    assert honest > 0.6, f"the synthetic signal was not learnable ({honest:.3f})"

    result = leakage.label_shuffle_test(fit_and_score, frame, seed=42, tolerance=0.05)
    assert result["passed"], f"shuffled AUC {result['mean']:.4f} did not collapse to 0.50"
    assert abs(result["mean"] - 0.5) <= 0.05


def test_label_shuffle_preserves_the_per_date_class_balance():
    """Shuffling WITHIN a date must not change how many events that date had.

    If it did, the test would be measuring a different panel and a model could
    still score by learning "this quarter was bad for everyone".
    """
    frame = _panel(seed=3)
    captured: list[pd.DataFrame] = []

    def capture(data: pd.DataFrame) -> float:
        captured.append(data.copy())
        return 0.5

    leakage.label_shuffle_test(capture, frame, seed=1, tolerance=0.05, n_repeats=1)

    before = frame.groupby("observation_date")["label"].sum()
    after = captured[0].groupby("observation_date")["label"].sum()
    # check_dtype=False: the permutation goes through a float array so that NaN
    # survives it, which is deliberate. The COUNTS are what must be identical.
    pd.testing.assert_series_equal(before, after, check_dtype=False)


def test_label_shuffle_leaves_unlabelled_rows_unlabelled():
    """The demerger-voided and embargo rows must stay NaN through a shuffle.

    A plain permutation moves NaN around, handing it to a valid training row -
    which is exactly how this broke in Phase 5, with XGBoost rejecting the fold.
    """
    frame = _panel(seed=5)
    frame["label"] = frame["label"].astype(float)
    frame.loc[frame.index[:4], ["label", "label_is_valid"]] = [np.nan, 0]
    captured: list[pd.DataFrame] = []

    leakage.label_shuffle_test(
        lambda data: captured.append(data.copy()) or 0.5,
        frame,
        seed=1,
        tolerance=0.05,
        n_repeats=1,
    )

    shuffled = captured[0]
    assert shuffled.loc[frame.index[:4], "label"].isna().all(), "NaN was permuted onto other rows"
    assert shuffled["label"].isna().sum() == frame["label"].isna().sum()


def test_run_all_raises_when_a_check_fails_and_strict():
    """Check 1 counts a row only if it actually carries pledge data, so the
    fixture must supply ``pledge_pct_promoter`` - a row with no pledge value
    cannot violate the point-in-time rule and is correctly ignored."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "quarter_end": ["2024-03-31"],
            "observation_date": ["2024-04-30"],
            "pledge_pct_promoter": [42.0],
            "fwd_max_drawdown": [-0.2],
            "label": [1],
            "label_is_valid": [1],
        }
    )
    state = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "quarter_end": ["2024-03-31"],
            # filed ten days AFTER the observation date
            "submission_date": ["2024-05-10"],
        }
    )
    with pytest.raises(LeakageError):
        leakage.run_all(frame, state, _prices("AAA", "2024-01-01", 200), horizon=60, strict=True)


def test_run_all_ignores_rows_that_carry_no_pledge_data():
    """A never-pledged company cannot leak a filing it never made."""
    frame = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "quarter_end": ["2024-03-31"],
            "observation_date": ["2024-04-30"],
            "pledge_pct_promoter": [np.nan],
            "fwd_max_drawdown": [-0.2],
            "label": [1],
            "label_is_valid": [0],
        }
    )
    state = pd.DataFrame(
        {"symbol": ["AAA"], "quarter_end": ["2024-03-31"], "submission_date": ["2024-05-10"]}
    )
    results = leakage.run_all(
        frame, state, _prices("AAA", "2024-01-01", 200), horizon=60, strict=True
    )
    assert all(r["passed"] for r in results)
