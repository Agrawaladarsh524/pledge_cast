"""Engine and connection management - PLAN.md sec.6, sec.10.

Two PRAGMAs matter and they behave differently:

  ``journal_mode = WAL``  is PERSISTENT - stored in the database file itself, so
      it only has to be set once. It is what lets the API and the dashboard read
      concurrently without file-locking problems (sec.5.1).

  ``foreign_keys = ON``   is PER-CONNECTION - SQLite defaults it OFF and forgets
      it on every new connection. It is therefore re-applied on every connect via
      an event listener, not once at init.

sec.10 ("Database failure"): the context manager commits on success and rolls
back on any exception, so a partial write can never be left behind.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.engine import Connection

from pledgecast.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from config import Settings

logger = get_logger(__name__)

_ENGINES: dict[str, Engine] = {}


def _configure_connection(dbapi_connection: sqlite3.Connection, _record: object) -> None:
    """Applied to every new DBAPI connection."""
    cursor = dbapi_connection.cursor()
    try:
        # Per-connection: SQLite forgets this every time (sec.6).
        cursor.execute("PRAGMA foreign_keys = ON")
        # Persistent, but setting it is idempotent and cheap.
        cursor.execute("PRAGMA journal_mode = WAL")
        # Wait rather than raise "database is locked" when the dashboard and a
        # script touch the DB at the same moment.
        cursor.execute("PRAGMA busy_timeout = 5000")
        # WAL + NORMAL is the standard durable-enough/fast pairing.
        cursor.execute("PRAGMA synchronous = NORMAL")
    finally:
        cursor.close()


def get_engine(db_path: Path | str | None = None, *, settings: Settings | None = None) -> Engine:
    """Return the cached engine for ``db_path``. Creates the parent directory."""
    if db_path is None:
        if settings is None:
            from config import get_settings

            settings = get_settings()
        db_path = settings.db_path

    resolved = Path(db_path).resolve()
    key = str(resolved)

    if key not in _ENGINES:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        # Default pooling is right here: the PRAGMAs below are applied on the
        # "connect" event, which fires when a real DBAPI connection is opened -
        # not on every pool checkout. Since both settings persist for that
        # connection's lifetime, a pooled connection stays correctly configured.
        engine = create_engine(f"sqlite+pysqlite:///{resolved}", future=True)
        event.listen(engine, "connect", _configure_connection)
        _ENGINES[key] = engine
        logger.debug("created engine for %s", resolved)

    return _ENGINES[key]


@contextmanager
def get_connection(
    db_path: Path | str | None = None,
    *,
    settings: Settings | None = None,
) -> Iterator[Connection]:
    """Transactional connection. Commits on success, rolls back on any error.

    Usage::

        with get_connection() as conn:
            repository.upsert_pledge_state(conn, rows)
    """
    engine = get_engine(db_path, settings=settings)
    conn = engine.connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        logger.exception("transaction rolled back")
        raise
    finally:
        conn.close()


def create_all(conn: Connection) -> None:
    """Create every table and index. Idempotent (``checkfirst=True``)."""
    from pledgecast.db.schema import metadata

    metadata.create_all(conn, checkfirst=True)


def drop_all(conn: Connection) -> None:
    """Drop every table. Only for tests and an explicit ``--force`` re-init."""
    from pledgecast.db.schema import metadata

    metadata.drop_all(conn, checkfirst=True)


def healthcheck(db_path: Path | str | None = None, *, settings: Settings | None = None) -> dict:
    """Is the database reachable and correctly configured? Backs ``GET /health``."""
    from pledgecast.db.schema import ALL_TABLES

    try:
        with get_connection(db_path, settings=settings) as conn:
            journal = conn.execute(text("PRAGMA journal_mode")).scalar()
            fk = conn.execute(text("PRAGMA foreign_keys")).scalar()
            found = {
                row[0]
                for row in conn.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))
            }
            missing = sorted(set(ALL_TABLES) - found)
            return {
                "reachable": True,
                "journal_mode": journal,
                "foreign_keys": bool(fk),
                "tables_present": len(set(ALL_TABLES) & found),
                "tables_expected": len(ALL_TABLES),
                "missing_tables": missing,
            }
    except Exception as exc:  # noqa: BLE001 - health must report, never raise
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def dispose_engines() -> None:
    """Close every pooled connection. Used by test teardown."""
    for engine in _ENGINES.values():
        engine.dispose()
    _ENGINES.clear()


__all__ = [
    "create_all",
    "dispose_engines",
    "drop_all",
    "get_connection",
    "get_engine",
    "healthcheck",
]
