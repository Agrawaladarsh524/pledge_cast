"""Domain exceptions - PLAN.md sec.8.

Every layer fails loudly with a named exception rather than returning a silent
zero or an empty frame (sec.18 "Robustness": "Every layer fails loudly and logs
why").
"""

from __future__ import annotations


class PledgeCastError(Exception):
    """Base for everything this project raises. Catch this to catch them all."""


class DataIngestionError(PledgeCastError):
    """Network/HTTP/session failure while fetching from NSE or Yahoo.

    Raised only once retries and session refresh are exhausted (sec.10
    "Network failure mid-ingest").
    """


class ParseError(PledgeCastError):
    """An XBRL file could not be parsed into a typed row.

    The offending file is copied to ``data/quarantine/`` with the reason, so
    data loss becomes an auditable list rather than a silent gap (sec.8.1).
    """

    def __init__(self, message: str, *, path: str | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason or message


class ModelNotFoundError(PledgeCastError):
    """No model run is flagged ``is_active = 1``, or its artifact is missing.

    sec.10: the API turns this into HTTP 503; the dashboard shows
    "no active model - run `make train`".
    """


class InsufficientDataError(PledgeCastError):
    """Not enough history to compute what was asked for.

    sec.10: a company with fewer than ``features.min_quarters_per_company``
    quarters cannot produce ``pledge_accel`` and is excluded with a logged
    reason rather than silently imputed.
    """


class ValidationError(PledgeCastError):
    """A row failed a range, null or duplicate check.

    NOTE: deliberately shares its name with ``pydantic.ValidationError`` because
    PLAN.md sec.8 names it so. Always import it qualified::

        from pledgecast.exceptions import ValidationError as PledgeCastValidationError
    """


class DatabaseError(PledgeCastError):
    """The database is unreachable, locked, or structurally wrong.

    sec.10: the API turns this into HTTP 503; the dashboard degrades to a
    readable error, never a traceback.
    """


class LeakageError(PledgeCastError):
    """A point-in-time invariant was violated.

    sec.9.8 makes this non-negotiable: if the label-shuffle test does not
    collapse AUC to ~0.50, everything stops until it is fixed.
    """


__all__ = [
    "DataIngestionError",
    "DatabaseError",
    "InsufficientDataError",
    "LeakageError",
    "ModelNotFoundError",
    "ParseError",
    "PledgeCastError",
    "ValidationError",
]
