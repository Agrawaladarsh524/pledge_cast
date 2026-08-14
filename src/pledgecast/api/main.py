"""FastAPI application - PLAN.md sec.13.

    "FastAPI, 5 endpoints. All logic delegates to src/inference/service.py."

Two things this file exists to do, and nothing else:

**Lifespan model loading.** The active model is loaded once at startup rather
than per request. A 300-company cohort score costs a millisecond; unpickling the
artifact does not. If no model is active, startup still succeeds and ``/health``
reports ``model: false`` - a server that refuses to boot cannot tell you why it
refused, which is the opposite of sec.13's "proves the system knows its own
state".

**The sec.13.2 error contract**, applied centrally:

    422  invalid request body (pydantic, field-level detail)
    404  unknown symbol
    503  model or database unavailable
    500  anything else, with request_id for log correlation

Mapping domain exceptions here rather than try/except in each route means a new
endpoint inherits the contract instead of re-implementing it.

    uvicorn pledgecast.api.main:app --reload
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config import get_settings
from pledgecast.api import routes
from pledgecast.api.schemas import ErrorResponse
from pledgecast.db.connection import get_connection
from pledgecast.exceptions import DatabaseError, PledgeCastError
from pledgecast.inference.service import PredictionService
from pledgecast.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings)
    app.state.settings = settings
    app.state.service = PredictionService(settings)

    try:
        with get_connection(settings=settings) as conn:
            app.state.service.load(conn)
    except PledgeCastError as exc:
        # Boot anyway. /health will say model: false and explain why, which is
        # more useful than a container that will not start.
        logger.error("starting without an active model: %s", exc)

    yield
    logger.info("shutting down")


app = FastAPI(
    title=get_settings().api.title,
    version=get_settings().api.version,
    description=(
        "Explainable early-warning scores for promoter-pledge-driven downside risk. "
        "All scoring runs through src/inference/service.py, which the Streamlit "
        "dashboard imports directly - the two cannot drift apart."
    ),
    lifespan=lifespan,
)
app.include_router(routes.router)


@app.exception_handler(PledgeCastError)
async def domain_error_handler(request: Request, exc: PledgeCastError):
    """sec.13.2: every domain exception has one agreed status code."""
    status = routes.STATUS_FOR.get(type(exc), 503 if isinstance(exc, DatabaseError) else 500)
    request_id = str(uuid.uuid4())

    log = logger.warning if status < 500 else logger.error
    log("%s %s -> %d [%s] %s", request.method, request.url.path, status, request_id, exc)

    return JSONResponse(
        status_code=status,
        content=ErrorResponse(
            detail=str(exc), error_type=type(exc).__name__, request_id=request_id
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """sec.13.2's 500 - "with request_id for log correlation"."""
    request_id = str(uuid.uuid4())
    logger.exception("%s %s -> 500 [%s]", request.method, request.url.path, request_id)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="internal server error",
            error_type=type(exc).__name__,
            request_id=request_id,
        ).model_dump(),
    )


def run() -> None:
    """``make api`` entry point."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "pledgecast.api.main:app",
        host=settings.api.host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
