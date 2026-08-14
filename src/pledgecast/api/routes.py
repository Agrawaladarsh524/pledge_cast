"""The 5 endpoints - PLAN.md sec.13.

    GET  /health                        DB reachable + active model loaded
    GET  /model-info                    active run: model, features, metrics
    POST /predict                       score by symbol or by raw features
    GET  /predictions                   history, filterable, paginated
    GET  /companies/{symbol}/history    pledge trajectory + past predictions

sec.7.1: "API / Dashboard | Transport + presentation | Never contains business
logic." Every route here opens a connection, delegates to
``PredictionService``, and shapes the result. There is no scoring code in this
file and there must never be - the dashboard reaches the same service directly,
so anything implemented here would exist only for API callers.

**Excluded on purpose** (sec.13): ``POST /train`` (training is not a web
operation), auth (no users), ``/batch-predict`` (a script's job).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from pledgecast.api import schemas
from pledgecast.db import repository as repo
from pledgecast.db.connection import get_connection
from pledgecast.exceptions import InsufficientDataError, ModelNotFoundError
from pledgecast.exceptions import ValidationError as PledgeCastValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


def _service(request: Request):
    """The service built once during lifespan (sec.13 "lifespan model loading")."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="scoring service is not initialised")
    return service


def _settings(request: Request):
    return request.app.state.settings


@router.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health(request: Request):
    """Never raises. A health check that 500s tells you nothing you can act on."""
    settings = _settings(request)
    service = getattr(request.app.state, "service", None)
    if service is None:
        return schemas.HealthResponse(status="unhealthy", database=False, model=False,
                                      error="service not initialised")
    try:
        with get_connection(settings=settings) as conn:
            return schemas.HealthResponse(**service.health(conn))
    except Exception as exc:  # noqa: BLE001 - reported as a body, not a status code
        logger.exception("health check failed")
        return schemas.HealthResponse(
            status="unhealthy", database=False, model=False, error=str(exc)
        )


@router.get("/model-info", response_model=schemas.ModelInfoResponse, tags=["model"])
def model_info(request: Request):
    settings = _settings(request)
    with get_connection(settings=settings) as conn:
        return schemas.ModelInfoResponse(**_service(request).model_info(conn))


@router.post("/predict", response_model=schemas.PredictResponse, tags=["scoring"])
def predict(request: Request, body: schemas.PredictRequest):
    """Score one company, or one supplied feature vector.

    sec.13: "Every prediction is persisted ... Prediction history is a side
    effect of serving, not a separate feature to build." Symbol-based scores are
    written with ``source='api'``; a raw feature vector is not persisted because
    it belongs to no company and no observation date.
    """
    settings = _settings(request)
    service = _service(request)

    with get_connection(settings=settings) as conn:
        if body.features is not None:
            return schemas.PredictResponse(**service.score_features(conn, body.features))

        result = service.score(
            conn,
            body.symbol,
            observation_date=body.observation_date,
            include_explanation=body.include_explanation,
            persist=True,
            source="api",
        )
    return schemas.PredictResponse(**result)


@router.get("/predictions", response_model=schemas.PredictionsResponse, tags=["scoring"])
def predictions(
    request: Request,
    symbol: str | None = None,
    observation_date: str | None = None,
    source: str | None = None,
    limit: int | None = Query(default=None, ge=1),
    offset: int = Query(default=0, ge=0),
):
    settings = _settings(request)
    page = limit or settings.api.predictions_page_size
    page = min(page, settings.api.predictions_max_page_size)

    with get_connection(settings=settings) as conn:
        total = repo.count_predictions(conn, symbol=symbol)
        frame = repo.load_predictions(
            conn,
            symbol=symbol,
            observation_date=observation_date,
            source=source,
            limit=page,
            offset=offset,
        )
    return schemas.PredictionsResponse(
        total=total, limit=page, offset=offset, items=frame.to_dict("records")
    )


@router.get(
    "/companies/{symbol}/history",
    response_model=schemas.CompanyHistoryResponse,
    tags=["companies"],
)
def company_history(request: Request, symbol: str):
    settings = _settings(request)
    with get_connection(settings=settings) as conn:
        history = _service(request).company_history(conn, symbol.upper())
    return schemas.CompanyHistoryResponse(**history)


# --------------------------------------------------------------------------- #
# sec.13.2 error contract                                                      #
# --------------------------------------------------------------------------- #
# 422 is produced by pydantic before a route body runs, so it needs no handler.
STATUS_FOR: dict[type[Exception], int] = {
    ModelNotFoundError: 503,
    InsufficientDataError: 404,
    PledgeCastValidationError: 422,
}


__all__ = ["STATUS_FOR", "router"]
