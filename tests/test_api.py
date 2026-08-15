"""API tests - PLAN.md sec.15.

    (4) /health 200 · /predict valid -> 200 with probability in [0,1] ·
        unknown symbol -> 404 · malformed body -> 422              [**]

**These run against a COPY of the real database.** The alternative - a synthetic
panel plus a model trained inside the fixture - would test a model that has
never existed and skip the two things most likely to break in practice: that the
stored artifact still unpickles, and that its feature list still matches the
``model_runs`` row. Copying costs a few hundred milliseconds and keeps the
suite from writing ``source='api'`` rows into the working database.

The whole module skips cleanly on a fresh clone, where no model has been
trained yet.
"""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from config import Settings, get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.inference.service import PredictionService


@pytest.fixture(scope="module")
def api_settings(tmp_path_factory) -> Settings:
    """Settings pointed at a throwaway copy of the project database.

    ``models_dir`` is left alone deliberately - the artifact on disk is part of
    what these tests are checking.
    """
    live = get_settings()
    if not live.db_path.exists():
        pytest.skip("no database - run `make init-db && make ingest && make build`")

    copy = tmp_path_factory.mktemp("api") / "pledgecast.db"
    shutil.copy2(live.db_path, copy)
    return Settings(db_path=copy)


@pytest.fixture(scope="module")
def client(api_settings):
    from pledgecast.api.main import app

    with TestClient(app) as test_client:
        # Redirect the app at the copy AFTER lifespan has run, so no test can
        # write to the working database.
        app.state.settings = api_settings
        app.state.service = PredictionService(api_settings)
        with get_connection(settings=api_settings) as conn:
            if repo.get_active_run(conn) is None:
                pytest.skip("no active model - run `make train`")
            app.state.service.load(conn)
        yield test_client


@pytest.fixture(scope="module")
def a_symbol(api_settings) -> str:
    with get_connection(settings=api_settings) as conn:
        symbols = repo.get_universe_symbols(conn)
    if not symbols:
        pytest.skip("no companies in the universe")
    return symbols[0]


# --------------------------------------------------------------------------- #
# /health                                                                      #
# --------------------------------------------------------------------------- #
def test_health_returns_200_and_reports_both_dependencies(client):
    """sec.13: "proves the system knows its own state" - DB and model together."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["database"] is True
    assert body["model"] is True
    assert body["run_id"]


def test_model_info_reports_what_is_actually_serving(client):
    response = client.get("/model-info")
    assert response.status_code == 200

    body = response.json()
    assert body["features"], "an empty feature list would score nonsense"
    assert body["random_seed"] is not None, "sec.10: seeds are fixed and stored"
    assert body["run_id"] == client.get("/health").json()["run_id"]


# --------------------------------------------------------------------------- #
# /predict                                                                     #
# --------------------------------------------------------------------------- #
def test_predict_returns_a_probability_in_the_unit_interval(client, a_symbol):
    response = client.post("/predict", json={"symbol": a_symbol})
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert 1 <= body["risk_decile"] <= 10
    assert body["risk_band"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert body["symbol"] == a_symbol
    assert body["model"]["run_id"]


def test_predict_with_an_explanation_returns_ranked_non_zero_contributions(client, a_symbol):
    """Regression: a single-row LinearExplainer built against the row itself
    returns 0.0 for every feature - a perfectly well-formed answer that says
    nothing. At least one contribution must be non-zero."""
    response = client.post(
        "/predict", json={"symbol": a_symbol, "include_explanation": True}
    )
    assert response.status_code == 200

    explanation = response.json()["explanation"]
    assert explanation is not None
    assert explanation["summary"]
    contributions = [f["shap"] for f in explanation["top_features"]]
    assert any(abs(value) > 1e-9 for value in contributions), "every SHAP value was zero"
    assert contributions == sorted(contributions, key=abs, reverse=True), "not ranked"


def test_predict_persists_the_prediction(client, a_symbol, api_settings):
    """sec.13: "Prediction history is a side effect of serving"."""
    with get_connection(settings=api_settings) as conn:
        before = repo.count_predictions(conn, symbol=a_symbol)

    body = client.post("/predict", json={"symbol": a_symbol}).json()

    with get_connection(settings=api_settings) as conn:
        after = repo.count_predictions(conn, symbol=a_symbol)
        stored = repo.load_predictions(conn, symbol=a_symbol, source="api")

    assert after == before + 1
    assert body["prediction_id"] in set(stored["prediction_id"])


def _plausible_vector(features: list[str]) -> dict[str, float]:
    """A feature vector that could describe a real company.

    This used to be ``dict.fromkeys(features, 0.5)``, which was convenient and
    physically impossible: ``trailing_dd_60d`` is a drawdown from entry and is
    never positive, so 0.5 described a company that fell upward. It scored
    anyway, because nothing checked. Now the domain bounds are enforced at the
    schema layer, so the filler has to be clamped into each feature's range.
    """
    from pledgecast.data.validate import FEATURE_BOUNDS

    vector = {}
    for name in features:
        low, high = FEATURE_BOUNDS.get(name, (None, None))
        value = 0.5
        if low is not None:
            value = max(value, low)
        if high is not None:
            value = min(value, high)
        vector[name] = value
    return vector


def test_predict_accepts_a_raw_feature_vector_without_a_decile(client):
    features = client.get("/model-info").json()["features"]
    response = client.post("/predict", json={"features": _plausible_vector(features)})
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_decile"] is None, "an ad-hoc vector belongs to no cohort"
    assert body["warnings"]


def test_predict_rejects_an_impossible_feature_vector(client):
    """sec.10: invalid input is a 422 with field-level detail, not a confident score.

    A promoter cannot have minus five hundred percent of their holding pledged.
    Before the domain bounds reached the API this scored happily.
    """
    features = client.get("/model-info").json()["features"]
    vector = _plausible_vector(features)
    if "pledge_pct_promoter" not in vector:
        pytest.skip("active model does not use pledge_pct_promoter")
    vector["pledge_pct_promoter"] = -500.0

    response = client.post("/predict", json={"features": vector})
    assert response.status_code == 422
    assert "pledge_pct_promoter" in response.text


# --------------------------------------------------------------------------- #
# sec.13.2 error contract                                                      #
# --------------------------------------------------------------------------- #
def test_unknown_symbol_returns_404(client):
    response = client.post("/predict", json={"symbol": "NOSUCHCOMPANY"})
    assert response.status_code == 404

    body = response.json()
    assert body["error_type"] == "InsufficientDataError"
    assert body["request_id"], "sec.13.2 requires a request_id for log correlation"


def test_unknown_company_history_returns_404(client):
    assert client.get("/companies/NOSUCHCOMPANY/history").status_code == 404


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ({}, "neither symbol nor features"),
        ({"symbol": "AAA", "features": {"volatility_90d": 0.4}}, "both at once"),
        ({"symbol": 5}, "symbol is not a string"),
        ({"features": {"volatility_90d": "not a number"}}, "feature is not numeric"),
    ],
)
def test_malformed_body_returns_422(client, body, why):
    assert client.post("/predict", json=body).status_code == 422, why


def test_missing_features_in_a_raw_vector_is_rejected(client):
    """A short feature dict must not be silently padded with zeros."""
    response = client.post("/predict", json={"features": {"volatility_90d": 0.4}})
    assert response.status_code == 422
    assert "missing required feature" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# listing endpoints                                                            #
# --------------------------------------------------------------------------- #
def test_predictions_are_paginated(client, a_symbol):
    client.post("/predict", json={"symbol": a_symbol})
    response = client.get("/predictions", params={"symbol": a_symbol, "limit": 2})
    assert response.status_code == 200

    body = response.json()
    assert body["limit"] == 2
    assert len(body["items"]) <= 2
    assert body["total"] >= len(body["items"])


def test_page_size_is_capped_by_config(client, api_settings):
    response = client.get(
        "/predictions", params={"limit": api_settings.api.predictions_max_page_size + 500}
    )
    assert response.status_code == 200
    assert response.json()["limit"] == api_settings.api.predictions_max_page_size


def test_company_history_returns_the_investigation_payload(client, a_symbol):
    response = client.get(f"/companies/{a_symbol.lower()}/history")
    assert response.status_code == 200, "the symbol must be matched case-insensitively"

    body = response.json()
    assert body["symbol"] == a_symbol
    assert isinstance(body["pledge_history"], list)
    assert isinstance(body["events"], list)
    assert isinstance(body["predictions"], list)
