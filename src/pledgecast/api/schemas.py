"""Request/response models - PLAN.md sec.13.1.

Pydantic v2. These types ARE the API contract: FastAPI derives the OpenAPI
schema from them, and sec.13.2's 422 ("invalid request body, field-level
detail") is produced by validation here rather than by hand-written checks in
the routes.

Every field name matches the sec.13.1 example exactly. Where the plan shows a
value the service cannot always supply - ``risk_decile`` for a raw feature dict,
which belongs to no cohort - the field is optional rather than faked.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelRef(BaseModel):
    """Which model produced a number. Present on every prediction (sec.13.1)."""

    run_id: str
    model_name: str

    model_config = ConfigDict(protected_namespaces=())


class FeatureContribution(BaseModel):
    feature: str
    value: float | None = None
    shap: float
    direction: Literal["increases_risk", "reduces_risk"]


class Explanation(BaseModel):
    top_features: list[FeatureContribution]
    summary: str


class PredictRequest(BaseModel):
    """``POST /predict`` - by symbol (features from the DB) or by raw features."""

    symbol: str | None = Field(default=None, examples=["JPPOWER"])
    observation_date: str | None = Field(
        default=None,
        description="Defaults to the company's most recent observation date.",
        examples=["2026-07-30"],
    )
    features: dict[str, float] | None = Field(
        default=None,
        description="Score an arbitrary feature vector instead of a stored company.",
    )
    include_explanation: bool = False

    @model_validator(mode="after")
    def _exactly_one_input(self) -> PredictRequest:
        """Reject both-or-neither at the schema layer, so it surfaces as a 422."""
        if (self.symbol is None) == (self.features is None):
            raise ValueError("supply exactly one of 'symbol' or 'features'")
        return self


class PredictResponse(BaseModel):
    symbol: str | None = None
    observation_date: str | None = None
    probability: float
    risk_decile: int | None = None
    risk_band: str
    model: ModelRef
    explanation: Explanation | None = None
    # sec.13.1: stale pledge state, unavailable features, no realised outcome yet.
    warnings: list[str] = Field(default_factory=list)
    prediction_id: int | None = None

    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    """``GET /model-info`` - "model transparency" (sec.13)."""

    run_id: str
    model_name: str
    experiment: str
    features: list[str]
    hyperparams: dict[str, Any]
    random_seed: int
    n_train_rows: int | None = None
    n_folds: int | None = None
    trained_at: str
    metrics: dict[str, float | None]

    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(BaseModel):
    """``GET /health`` - "proves the system knows its own state" (sec.13)."""

    status: Literal["healthy", "unhealthy"]
    database: bool
    model: bool
    run_id: str | None = None
    panel_rows: int | None = None
    predictions: int | None = None
    error: str | None = None

    model_config = ConfigDict(protected_namespaces=())


class PredictionRecord(BaseModel):
    prediction_id: int
    run_id: str
    symbol: str
    observation_date: str
    probability: float
    risk_decile: int | None = None
    source: str
    created_at: str


class PredictionsResponse(BaseModel):
    """``GET /predictions`` - paginated history."""

    total: int
    limit: int
    offset: int
    items: list[PredictionRecord]


class PledgeStateRecord(BaseModel):
    quarter_end: str
    submission_date: str | None = None
    promoter_holding_pct: float | None = None
    pledge_pct_promoter: float | None = None
    pledge_pct_equity: float | None = None
    pledge_status: str


class PledgeEventRecord(BaseModel):
    event_date: str
    promoter_name: str | None = None
    event_type: str
    shares: float | None = None
    pct_equity: float | None = None


class CompanyHistoryResponse(BaseModel):
    """``GET /companies/{symbol}/history`` - backs the investigation screen."""

    symbol: str
    company_name: str | None = None
    industry: str | None = None
    pledge_history: list[PledgeStateRecord]
    events: list[PledgeEventRecord]
    predictions: list[PredictionRecord]


class ErrorResponse(BaseModel):
    """sec.13.2's error body. ``request_id`` correlates a 500 with the log line."""

    detail: str
    error_type: str
    request_id: str | None = None


__all__ = [
    "CompanyHistoryResponse",
    "ErrorResponse",
    "Explanation",
    "FeatureContribution",
    "HealthResponse",
    "ModelInfoResponse",
    "ModelRef",
    "PledgeEventRecord",
    "PledgeStateRecord",
    "PredictRequest",
    "PredictResponse",
    "PredictionRecord",
    "PredictionsResponse",
]
