"""Two API contract defects found by audit, pinned so they cannot return.

Neither affects a research number. Both are the kind of defect that survives
indefinitely because the endpoint returns 200 and the response looks plausible.

**Pagination counted a different population than it paged.** ``GET /predictions``
accepts ``symbol``, ``observation_date`` and ``source``, and applied all three
when selecting rows - but counted on ``symbol`` alone. A request for one
observation date therefore returned that date's rows above a ``total`` of every
prediction ever stored, so any caller doing ``ceil(total / limit)`` paged into
emptiness.

**Raw feature vectors were unchecked.** ``POST /predict`` accepts
``dict[str, float]``, which establishes only that a value is a number. The
service verified that the required feature NAMES were present; nothing verified
the values could mean what they claim. ``pledge_pct_promoter: -500`` - a
promoter with minus five hundred percent of their holding pledged - scored, and
came back with a confident probability attached.
"""

from __future__ import annotations

import pytest

from pledgecast.data.validate import FEATURE_BOUNDS, check_feature_vector
from pledgecast.db import repository as repo


# --------------------------------------------------------------------------- #
# 1. the count must match the page                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def predictions(seeded_conn):
    """Nine predictions spanning three dates and two sources."""
    repo.insert_model_run(
        seeded_conn,
        run_id="R1",
        model_name="logreg",
        experiment="exp0_null",
        feature_list=["volatility_90d"],
        hyperparams={},
        random_seed=42,
        n_train_rows=10,
        n_folds=1,
    )
    rows = [
        {
            "run_id": "R1",
            "symbol": symbol,
            "observation_date": date,
            "probability": 0.5,
            "risk_decile": 5,
            "source": source,
        }
        for symbol in ("AAA", "BBB")
        for date, source in (
            ("2024-01-30", "backtest"),
            ("2024-04-30", "backtest"),
            ("2024-04-30", "api"),
        )
    ]
    repo.save_predictions_bulk(seeded_conn, rows)
    return seeded_conn


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"symbol": "AAA"},
        {"observation_date": "2024-04-30"},
        {"source": "api"},
        {"symbol": "AAA", "observation_date": "2024-04-30"},
        {"symbol": "AAA", "observation_date": "2024-04-30", "source": "api"},
    ],
)
def test_the_count_matches_the_rows_for_every_filter_combination(predictions, filters):
    """The defect: count accepted a subset of the filters that select accepted.

    Parameterised over each combination because the original bug was invisible
    for the one case that was tested - filtering by symbol alone.
    """
    counted = repo.count_predictions(predictions, **filters)
    selected = repo.load_predictions(predictions, **filters)
    assert counted == len(selected), f"count {counted} != {len(selected)} rows for {filters}"


def test_the_count_is_not_simply_the_table_size(predictions):
    """Guards the trivial regression where every filter is ignored again."""
    everything = repo.count_predictions(predictions)
    filtered = repo.count_predictions(predictions, observation_date="2024-01-30")
    assert filtered < everything, "a date filter must reduce the count"
    assert filtered == 2


# --------------------------------------------------------------------------- #
# 2. raw feature vectors get the panel's range rules                           #
# --------------------------------------------------------------------------- #
def test_a_valid_vector_reports_no_problems():
    assert check_feature_vector({"pledge_pct_promoter": 42.5, "volatility_90d": 0.31}) == []


@pytest.mark.parametrize(
    ("feature", "value"),
    [
        ("pledge_pct_promoter", -500.0),  # the case from the audit
        ("pledge_pct_promoter", 101.0),
        ("pledge_pct_equity", -0.1),
        ("promoter_holding_pct", 250.0),
        ("volatility_90d", -2.0),  # a negative standard deviation
        ("trailing_dd_60d", 0.5),  # a drawdown that went up
        ("trailing_dd_60d", -1.5),  # lost more than everything
        ("return_90d", -3.0),
        ("consecutive_rising_q", -1.0),
        ("event_count_90d", -5.0),
    ],
)
def test_impossible_values_are_rejected(feature, value):
    problems = check_feature_vector({feature: value})
    assert problems, f"{feature}={value} should have been rejected"
    assert feature in problems[0]


def test_unknown_keys_are_not_an_error():
    """The service already rejects a vector missing a required feature.

    Duplicating that check here would report the same fault twice, in two
    different formats, from two different layers.
    """
    assert check_feature_vector({"not_a_feature": 1e9}) == []


def test_nan_is_left_to_the_imputer():
    """Missing is a state the pipeline handles; it is not a domain violation."""
    assert check_feature_vector({"volatility_90d": float("nan")}) == []


def test_every_bounded_feature_admits_its_own_boundary():
    """An inclusive bound must accept the bound itself, or valid rows fail."""
    for name, (low, high) in FEATURE_BOUNDS.items():
        for edge in (low, high):
            if edge is not None:
                assert check_feature_vector({name: edge}) == [], f"{name} rejected its own bound {edge}"


def test_the_request_schema_refuses_an_out_of_domain_vector():
    """It must surface at the schema layer, which FastAPI renders as a 422."""
    from pydantic import ValidationError as PydanticValidationError

    from pledgecast.api import schemas

    with pytest.raises(PydanticValidationError):
        schemas.PredictRequest(features={"pledge_pct_promoter": -500.0})

    ok = schemas.PredictRequest(features={"pledge_pct_promoter": 50.0})
    assert ok.features == {"pledge_pct_promoter": 50.0}
