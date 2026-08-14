"""ALL SQL in the project lives here - PLAN.md sec.8.1.

    "No SQL scattered through the codebase; swappable backend later."

That is also the answer to sec.18 "Scaling": moving ``prices`` and ``panel`` to
Postgres or DuckDB is a one-module change because nothing outside this file
knows SQL exists.

Two conventions every function here honours:

  * **Empty means empty.** Loaders return a zero-row DataFrame with the correct
    columns, never ``None`` and never an ``IndexError`` (sec.10 "Empty datasets",
    and a sec.15 ``test_repository`` case).
  * **Upserts are idempotent.** Composite primary keys make duplicates
    structurally impossible; writes use ``INSERT OR REPLACE`` so re-running any
    ingest script is safe (sec.10 "Duplicate records").
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import Table, delete, func, insert, select, text, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Connection

from pledgecast.db import schema
from pledgecast.exceptions import ModelNotFoundError, ValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """ISO-8601 UTC timestamp. One source of 'now' for every table."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _columns(table: Table) -> list[str]:
    return [c.name for c in table.columns]


def _empty(table: Table, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns) if columns else _columns(table))


def _read(conn: Connection, stmt: Any, table: Table) -> pd.DataFrame:
    """Run a select and always return a DataFrame with the right columns."""
    result = conn.execute(stmt)
    rows = result.fetchall()
    cols = list(result.keys())
    if not rows:
        return _empty(table, cols)
    return pd.DataFrame(rows, columns=cols)


def _upsert(
    conn: Connection,
    table: Table,
    rows: Iterable[dict],
    *,
    on_conflict: str = "update",
) -> int:
    """Idempotent batch write. Returns the number of rows submitted.

    Uses SQLite's native ``INSERT ... ON CONFLICT``, **never ``INSERT OR
    REPLACE``**. REPLACE looks equivalent and is not: it DELETEs the conflicting
    row and inserts a fresh one, which has two destructive consequences here.

    1. Columns absent from the payload are reset to their defaults rather than
       left alone. Re-running filing discovery would wipe ``local_path``,
       ``sha256`` and reset ``status`` to pending on every already-downloaded
       filing.
    2. The delete-and-reinsert allocates a NEW autoincrement id, so
       ``pledge_state.filing_id`` would point at a row that no longer exists -
       observed as "FOREIGN KEY constraint failed" partway through a re-run.

    ``on_conflict='nothing'`` preserves the existing row entirely, which is what
    re-discovery wants: the ledger already knows more about that filing than the
    discovery payload does.
    """
    payload = [r for r in rows if r]
    if not payload:
        return 0

    allowed = set(_columns(table))
    cleaned: list[dict] = []
    for row in payload:
        unknown = set(row) - allowed
        if unknown:
            raise ValidationError(
                f"{table.name}: unknown column(s) {sorted(unknown)}; "
                f"valid columns are {sorted(allowed)}"
            )
        cleaned.append(row)

    keys = schema.CONFLICT_KEYS.get(table.name)
    if keys is None:
        keys = tuple(c.name for c in table.primary_key)

    # executemany needs a homogeneous column set, so group by signature.
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in cleaned:
        groups.setdefault(tuple(sorted(row)), []).append(row)

    for signature, group in groups.items():
        stmt = sqlite_insert(table)
        if not keys:
            conn.execute(stmt, group)
            continue

        updatable = {c: stmt.excluded[c] for c in signature if c not in keys}
        if on_conflict == "nothing" or not updatable:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(keys))
        else:
            stmt = stmt.on_conflict_do_update(index_elements=list(keys), set_=updatable)
        conn.execute(stmt, group)

    return len(cleaned)


# ========================================================================== #
# companies                                                                  #
# ========================================================================== #
def upsert_companies(conn: Connection, rows: Iterable[dict]) -> int:
    prepared = []
    for row in rows:
        row = dict(row)
        row.setdefault("added_at", utc_now())
        row.setdefault("in_universe", 1)
        prepared.append(row)
    n = _upsert(conn, schema.companies, prepared)
    logger.info("upserted %d companies", n)
    return n


def get_universe_symbols(conn: Connection, *, in_universe: bool = True) -> list[str]:
    stmt = select(schema.companies.c.symbol)
    if in_universe:
        stmt = stmt.where(schema.companies.c.in_universe == 1)
    return [r[0] for r in conn.execute(stmt.order_by(schema.companies.c.symbol))]


def load_companies(conn: Connection, *, in_universe: bool | None = None) -> pd.DataFrame:
    stmt = select(schema.companies)
    if in_universe is not None:
        stmt = stmt.where(schema.companies.c.in_universe == int(in_universe))
    return _read(conn, stmt.order_by(schema.companies.c.symbol), schema.companies)


def get_company(conn: Connection, symbol: str) -> dict | None:
    row = (
        conn.execute(select(schema.companies).where(schema.companies.c.symbol == symbol))
        .mappings()
        .first()
    )
    return dict(row) if row else None


def symbol_exists(conn: Connection, symbol: str) -> bool:
    """sec.10: validate a symbol against ``companies`` before any query."""
    return (
        conn.execute(
            select(func.count())
            .select_from(schema.companies)
            .where(schema.companies.c.symbol == symbol)
        ).scalar_one()
        > 0
    )


def set_in_universe(conn: Connection, symbols: Sequence[str], flag: bool) -> int:
    """Exclude symbols without deleting them - keeps the audit trail (sec.6)."""
    if not symbols:
        return 0
    result = conn.execute(
        update(schema.companies)
        .where(schema.companies.c.symbol.in_(list(symbols)))
        .values(in_universe=int(flag))
    )
    return result.rowcount or 0


# ========================================================================== #
# filings  (the ingestion ledger - raw XML stays on disk, sec.5.2)           #
# ========================================================================== #
def upsert_filings(conn: Connection, rows: Iterable[dict], *, preserve_state: bool = True) -> int:
    """Record filings in the ledger.

    ``preserve_state`` (the default) leaves an existing ledger row untouched.
    Discovery re-runs on every ``01_build_universe.py`` invocation and only
    knows the URL and dates; the stored row also knows where the file landed,
    its hash and whether it parsed. Overwriting that would force a full
    re-download and orphan ``pledge_state.filing_id``.
    """
    prepared = []
    for row in rows:
        row = dict(row)
        row.setdefault("status", "pending")
        if row.get("status") not in schema.FILING_STATUSES:
            raise ValidationError(
                f"invalid filing status {row.get('status')!r}; "
                f"expected one of {schema.FILING_STATUSES}"
            )
        prepared.append(row)
    n = _upsert(
        conn, schema.filings, prepared, on_conflict="nothing" if preserve_state else "update"
    )
    logger.debug("upserted %d filings", n)
    return n


def load_filings(
    conn: Connection,
    *,
    symbol: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    stmt = select(schema.filings)
    if symbol:
        stmt = stmt.where(schema.filings.c.symbol == symbol)
    if status:
        stmt = stmt.where(schema.filings.c.status == status)
    stmt = stmt.order_by(schema.filings.c.symbol, schema.filings.c.quarter_end)
    return _read(conn, stmt, schema.filings)


def update_filing_status(
    conn: Connection,
    filing_id: int,
    status: str,
    *,
    local_path: str | None = None,
    sha256: str | None = None,
    error_message: str | None = None,
    fetched_at: str | None = None,
) -> None:
    if status not in schema.FILING_STATUSES:
        raise ValidationError(f"invalid filing status {status!r}")

    values: dict[str, Any] = {"status": status}
    if local_path is not None:
        values["local_path"] = local_path
    if sha256 is not None:
        values["sha256"] = sha256
    if error_message is not None:
        values["error_message"] = error_message
    if fetched_at is not None or status == "downloaded":
        values["fetched_at"] = fetched_at or utc_now()

    conn.execute(
        update(schema.filings).where(schema.filings.c.filing_id == filing_id).values(**values)
    )


def count_filings_by_status(conn: Connection) -> dict[str, int]:
    stmt = select(schema.filings.c.status, func.count()).group_by(schema.filings.c.status)
    return dict(conn.execute(stmt).all())


# ========================================================================== #
# pledge_state                                                               #
# ========================================================================== #
def upsert_pledge_state(conn: Connection, rows: Iterable[dict]) -> int:
    prepared = []
    for row in rows:
        row = dict(row)
        if row.get("pledge_status") not in schema.PLEDGE_STATUSES:
            raise ValidationError(
                f"invalid pledge_status {row.get('pledge_status')!r}; "
                f"expected one of {schema.PLEDGE_STATUSES}. A missing pledge flag "
                f"must become UNAVAILABLE, never a silent zero."
            )
        prepared.append(row)
    n = _upsert(conn, schema.pledge_state, prepared)
    logger.debug("upserted %d pledge_state rows", n)
    return n


def load_pledge_state(conn: Connection, *, symbol: str | None = None) -> pd.DataFrame:
    stmt = select(schema.pledge_state)
    if symbol:
        stmt = stmt.where(schema.pledge_state.c.symbol == symbol)
    stmt = stmt.order_by(schema.pledge_state.c.symbol, schema.pledge_state.c.quarter_end)
    return _read(conn, stmt, schema.pledge_state)


# ========================================================================== #
# pledge_events  (Reg 31 feature layer)                                      #
# ========================================================================== #
def replace_pledge_events(conn: Connection, symbol: str, rows: Iterable[dict]) -> int:
    """Replace one company's Reg 31 events wholesale.

    The natural key includes ``promoter_name``, ``event_type`` and ``shares``,
    all nullable - and SQLite treats NULL as distinct from NULL in a UNIQUE
    index, so ON CONFLICT would never match a row with a missing promoter name
    and every re-run would accumulate duplicates. Deleting the symbol's events
    first makes re-ingestion exactly idempotent regardless of NULLs.
    """
    conn.execute(delete(schema.pledge_events).where(schema.pledge_events.c.symbol == symbol))
    payload = list(rows)
    if not payload:
        return 0
    conn.execute(insert(schema.pledge_events), payload)
    return len(payload)


def upsert_pledge_events(conn: Connection, rows: Iterable[dict]) -> int:
    n = _upsert(conn, schema.pledge_events, rows)
    logger.debug("upserted %d pledge_events", n)
    return n


def load_pledge_events(
    conn: Connection,
    *,
    symbol: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    stmt = select(schema.pledge_events)
    if symbol:
        stmt = stmt.where(schema.pledge_events.c.symbol == symbol)
    if start:
        stmt = stmt.where(schema.pledge_events.c.event_date >= start)
    if end:
        stmt = stmt.where(schema.pledge_events.c.event_date <= end)
    stmt = stmt.order_by(schema.pledge_events.c.symbol, schema.pledge_events.c.event_date)
    return _read(conn, stmt, schema.pledge_events)


# ========================================================================== #
# prices + benchmark                                                         #
# ========================================================================== #
def upsert_prices(conn: Connection, rows: Iterable[dict]) -> int:
    return _upsert(conn, schema.prices, rows)


def upsert_benchmark(conn: Connection, rows: Iterable[dict]) -> int:
    return _upsert(conn, schema.benchmark, rows)


def load_prices(
    conn: Connection,
    *,
    symbols: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    stmt = select(schema.prices)
    if symbols:
        stmt = stmt.where(schema.prices.c.symbol.in_(list(symbols)))
    if start:
        stmt = stmt.where(schema.prices.c.trade_date >= start)
    if end:
        stmt = stmt.where(schema.prices.c.trade_date <= end)
    stmt = stmt.order_by(schema.prices.c.symbol, schema.prices.c.trade_date)
    return _read(conn, stmt, schema.prices)


def load_benchmark(
    conn: Connection, *, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    stmt = select(schema.benchmark)
    if start:
        stmt = stmt.where(schema.benchmark.c.trade_date >= start)
    if end:
        stmt = stmt.where(schema.benchmark.c.trade_date <= end)
    return _read(conn, stmt.order_by(schema.benchmark.c.trade_date), schema.benchmark)


def get_trading_days(conn: Connection) -> list[str]:
    """Distinct benchmark dates - the trading calendar used to roll dates (sec.9.3)."""
    return [
        r[0]
        for r in conn.execute(
            select(schema.benchmark.c.trade_date).order_by(schema.benchmark.c.trade_date)
        )
    ]


def count_price_rows(conn: Connection) -> dict[str, int]:
    stmt = select(schema.prices.c.symbol, func.count()).group_by(schema.prices.c.symbol)
    return dict(conn.execute(stmt).all())


# ========================================================================== #
# panel  (the ML-ready table)                                                #
# ========================================================================== #
def replace_panel(conn: Connection, frame: pd.DataFrame) -> int:
    """Rebuild the panel wholesale. ``03_build_panel.py`` is idempotent."""
    conn.execute(delete(schema.panel))
    if frame.empty:
        logger.warning("replace_panel called with an empty frame")
        return 0
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    return _upsert(conn, schema.panel, rows)


def load_panel(
    conn: Connection,
    experiment: str | None = None,
    *,
    valid_only: bool = True,
) -> pd.DataFrame:
    """Load the ML-ready panel (PLAN.md sec.5.3).

    ``valid_only`` keeps only labelled rows. Pass ``False`` to include the
    embargo quarter, which is featured but deliberately unlabelled (sec.9.4) -
    that is what ``06_score_latest.py`` scores.

    ``experiment`` narrows the feature columns to that experiment's set; keys,
    label and quality flags are always retained.
    """
    stmt = select(schema.panel)
    if valid_only:
        stmt = stmt.where(schema.panel.c.label_is_valid == 1)
    stmt = stmt.order_by(schema.panel.c.observation_date, schema.panel.c.symbol)
    frame = _read(conn, stmt, schema.panel)

    if experiment:
        from config import get_settings

        keep = [
            "symbol",
            "observation_date",
            "quarter_end",
            "is_stale",
            "fwd_max_drawdown",
            "label",
            "label_is_valid",
        ]
        features = get_settings().experiment_features(experiment)
        frame = frame[[c for c in keep + features if c in frame.columns]]

    return frame


def get_observation_dates(conn: Connection, *, valid_only: bool = True) -> list[str]:
    stmt = select(schema.panel.c.observation_date).distinct()
    if valid_only:
        stmt = stmt.where(schema.panel.c.label_is_valid == 1)
    return [r[0] for r in conn.execute(stmt.order_by(schema.panel.c.observation_date))]


def get_panel_row(
    conn: Connection, symbol: str, observation_date: str | None = None
) -> dict | None:
    """Latest (or a specific) panel row for one company - backs ``POST /predict``."""
    stmt = select(schema.panel).where(schema.panel.c.symbol == symbol)
    if observation_date:
        stmt = stmt.where(schema.panel.c.observation_date == observation_date)
    stmt = stmt.order_by(schema.panel.c.observation_date.desc()).limit(1)
    row = conn.execute(stmt).mappings().first()
    return dict(row) if row else None


# ========================================================================== #
# model_runs + model_metrics  (this is the MLflow replacement)               #
# ========================================================================== #
def insert_model_run(
    conn: Connection,
    *,
    run_id: str,
    model_name: str,
    experiment: str,
    feature_list: Sequence[str],
    hyperparams: dict,
    random_seed: int,
    n_train_rows: int | None = None,
    n_folds: int | None = None,
    artifact_path: str | None = None,
    config_snapshot: str | None = None,
    created_at: str | None = None,
) -> str:
    conn.execute(
        insert(schema.model_runs).prefix_with("OR REPLACE"),
        {
            "run_id": run_id,
            "created_at": created_at or utc_now(),
            "model_name": model_name,
            "experiment": experiment,
            "feature_list": json.dumps(list(feature_list)),
            "hyperparams": json.dumps(hyperparams, default=str),
            "random_seed": random_seed,
            "n_train_rows": n_train_rows,
            "n_folds": n_folds,
            "artifact_path": artifact_path,
            "config_snapshot": config_snapshot,
            "is_active": 0,
        },
    )
    logger.info("recorded model run %s (%s / %s)", run_id, model_name, experiment)
    return run_id


def get_model_run(conn: Connection, run_id: str) -> dict | None:
    row = (
        conn.execute(select(schema.model_runs).where(schema.model_runs.c.run_id == run_id))
        .mappings()
        .first()
    )
    if not row:
        return None
    out = dict(row)
    out["feature_list"] = json.loads(out["feature_list"])
    out["hyperparams"] = json.loads(out["hyperparams"])
    return out


def get_active_run(conn: Connection) -> dict | None:
    """The one run flagged ``is_active = 1`` (sec.9.7). ``None`` if unset."""
    row = (
        conn.execute(select(schema.model_runs).where(schema.model_runs.c.is_active == 1))
        .mappings()
        .first()
    )
    if not row:
        return None
    out = dict(row)
    out["feature_list"] = json.loads(out["feature_list"])
    out["hyperparams"] = json.loads(out["hyperparams"])
    return out


def require_active_run(conn: Connection) -> dict:
    """``get_active_run`` or raise - the API turns this into HTTP 503 (sec.10)."""
    run = get_active_run(conn)
    if run is None:
        raise ModelNotFoundError(
            "no active model run (is_active = 1). Run `make train` to create one."
        )
    return run


def set_active_run(conn: Connection, run_id: str) -> None:
    """Flag exactly one run active. Atomic - clears all, then sets one (sec.9.7)."""
    exists = conn.execute(
        select(func.count())
        .select_from(schema.model_runs)
        .where(schema.model_runs.c.run_id == run_id)
    ).scalar_one()
    if not exists:
        raise ModelNotFoundError(f"cannot activate unknown run_id {run_id!r}")

    conn.execute(update(schema.model_runs).values(is_active=0))
    conn.execute(
        update(schema.model_runs).where(schema.model_runs.c.run_id == run_id).values(is_active=1)
    )
    logger.info("active model run set to %s", run_id)


def load_model_runs(conn: Connection, *, experiment: str | None = None) -> pd.DataFrame:
    stmt = select(schema.model_runs)
    if experiment:
        stmt = stmt.where(schema.model_runs.c.experiment == experiment)
    return _read(conn, stmt.order_by(schema.model_runs.c.created_at.desc()), schema.model_runs)


def insert_metrics(conn: Connection, run_id: str, metrics: Iterable[dict]) -> int:
    """``metrics`` rows are ``{fold, metric_name, metric_value}``; fold -1 = aggregate."""
    rows = [{"run_id": run_id, **m} for m in metrics]
    return _upsert(conn, schema.model_metrics, rows)


def load_metrics(
    conn: Connection,
    *,
    run_id: str | None = None,
    metric_name: str | None = None,
    fold: int | None = None,
) -> pd.DataFrame:
    stmt = select(schema.model_metrics)
    if run_id:
        stmt = stmt.where(schema.model_metrics.c.run_id == run_id)
    if metric_name:
        stmt = stmt.where(schema.model_metrics.c.metric_name == metric_name)
    if fold is not None:
        stmt = stmt.where(schema.model_metrics.c.fold == fold)
    return _read(conn, stmt, schema.model_metrics)


def load_model_comparison(conn: Connection, metric_name: str, *, fold: int = -1) -> pd.DataFrame:
    """Runs joined to one aggregate metric - backs the Validation page table."""
    stmt = (
        select(
            schema.model_runs.c.run_id,
            schema.model_runs.c.model_name,
            schema.model_runs.c.experiment,
            schema.model_runs.c.n_folds,
            schema.model_runs.c.is_active,
            schema.model_metrics.c.metric_value.label(metric_name),
        )
        .select_from(
            schema.model_runs.join(
                schema.model_metrics,
                schema.model_runs.c.run_id == schema.model_metrics.c.run_id,
            )
        )
        .where(schema.model_metrics.c.metric_name == metric_name)
        .where(schema.model_metrics.c.fold == fold)
        .order_by(schema.model_metrics.c.metric_value.desc())
    )
    return _read(conn, stmt, schema.model_metrics)


# ========================================================================== #
# predictions + explanations                                                 #
# ========================================================================== #
def save_prediction(
    conn: Connection,
    *,
    run_id: str,
    symbol: str,
    observation_date: str,
    probability: float,
    source: str,
    risk_decile: int | None = None,
    created_at: str | None = None,
) -> int:
    """Append one prediction and return its ``prediction_id``.

    sec.13: "Every prediction is persisted ... Prediction history is a side
    effect of serving, not a separate feature to build."
    """
    if source not in schema.PREDICTION_SOURCES:
        raise ValidationError(
            f"invalid source {source!r}; expected one of {schema.PREDICTION_SOURCES}"
        )
    if not 0.0 <= probability <= 1.0:
        raise ValidationError(f"probability {probability} outside [0, 1]")

    result = conn.execute(
        insert(schema.predictions),
        {
            "run_id": run_id,
            "symbol": symbol,
            "observation_date": observation_date,
            "probability": float(probability),
            "risk_decile": risk_decile,
            "source": source,
            "created_at": created_at or utc_now(),
        },
    )
    return int(result.inserted_primary_key[0])


def save_predictions_bulk(conn: Connection, rows: Iterable[dict]) -> int:
    """Batch insert for walk-forward out-of-fold predictions (``source='backtest'``)."""
    payload = []
    now = utc_now()
    for row in rows:
        row = dict(row)
        row.setdefault("created_at", now)
        row.setdefault("source", "backtest")
        payload.append(row)
    if not payload:
        return 0
    conn.execute(insert(schema.predictions), payload)
    return len(payload)


def delete_predictions_for_run(conn: Connection, run_id: str) -> int:
    """Clear a run's predictions so re-training is idempotent."""
    result = conn.execute(delete(schema.predictions).where(schema.predictions.c.run_id == run_id))
    return result.rowcount or 0


def delete_predictions_for_run_date(
    conn: Connection, run_id: str, observation_date: str, source: str
) -> int:
    """Clear one date's batch scores so re-scoring is idempotent.

    Narrower than :func:`delete_predictions_for_run` on purpose. Re-running
    ``06_score_latest.py`` must not remove the walk-forward out-of-fold rows the
    backtest depends on, and those share the run id - only the observation date
    and source separate them.
    """
    result = conn.execute(
        delete(schema.predictions)
        .where(schema.predictions.c.run_id == run_id)
        .where(schema.predictions.c.observation_date == observation_date)
        .where(schema.predictions.c.source == source)
    )
    return result.rowcount or 0


def load_predictions(
    conn: Connection,
    *,
    run_id: str | None = None,
    symbol: str | None = None,
    observation_date: str | None = None,
    source: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> pd.DataFrame:
    stmt = select(schema.predictions)
    if run_id:
        stmt = stmt.where(schema.predictions.c.run_id == run_id)
    if symbol:
        stmt = stmt.where(schema.predictions.c.symbol == symbol)
    if observation_date:
        stmt = stmt.where(schema.predictions.c.observation_date == observation_date)
    if source:
        stmt = stmt.where(schema.predictions.c.source == source)
    stmt = stmt.order_by(
        schema.predictions.c.observation_date.desc(),
        schema.predictions.c.probability.desc(),
    )
    if limit is not None:
        stmt = stmt.limit(limit).offset(offset)
    return _read(conn, stmt, schema.predictions)


def count_predictions(
    conn: Connection, *, run_id: str | None = None, symbol: str | None = None
) -> int:
    stmt = select(func.count()).select_from(schema.predictions)
    if run_id:
        stmt = stmt.where(schema.predictions.c.run_id == run_id)
    if symbol:
        stmt = stmt.where(schema.predictions.c.symbol == symbol)
    return conn.execute(stmt).scalar_one()


def save_explanations(conn: Connection, prediction_id: int, rows: Iterable[dict]) -> int:
    """``rows`` are ``{feature_name, feature_value, shap_value}``."""
    payload = [{"prediction_id": prediction_id, **r} for r in rows]
    return _upsert(conn, schema.explanations, payload)


def load_explanations(conn: Connection, prediction_id: int) -> pd.DataFrame:
    stmt = (
        select(schema.explanations)
        .where(schema.explanations.c.prediction_id == prediction_id)
        .order_by(func.abs(schema.explanations.c.shap_value).desc())
    )
    return _read(conn, stmt, schema.explanations)


# ========================================================================== #
# backtest_results                                                           #
# ========================================================================== #
def save_backtest_results(conn: Connection, rows: Iterable[dict]) -> int:
    return _upsert(conn, schema.backtest_results, rows)


def load_backtest_results(
    conn: Connection,
    *,
    run_id: str | None = None,
    observation_date: str | None = None,
) -> pd.DataFrame:
    stmt = select(schema.backtest_results)
    if run_id:
        stmt = stmt.where(schema.backtest_results.c.run_id == run_id)
    if observation_date:
        stmt = stmt.where(schema.backtest_results.c.observation_date == observation_date)
    stmt = stmt.order_by(
        schema.backtest_results.c.observation_date, schema.backtest_results.c.quintile
    )
    return _read(conn, stmt, schema.backtest_results)


# ========================================================================== #
# summary                                                                    #
# ========================================================================== #
def table_counts(conn: Connection) -> dict[str, int]:
    """Row count per table - used by every script's completion banner."""
    counts: dict[str, int] = {}
    for name in schema.ALL_TABLES:
        counts[name] = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
    return counts


# Explicit, not derived from dir(): a computed __all__ is opaque to type
# checkers and IDEs, which then cannot resolve `from repository import ...`.
__all__ = [
    # companies
    "get_company",
    "get_universe_symbols",
    "load_companies",
    "set_in_universe",
    "symbol_exists",
    "upsert_companies",
    # filings ledger
    "count_filings_by_status",
    "load_filings",
    "update_filing_status",
    "upsert_filings",
    # pledge state + events
    "load_pledge_events",
    "load_pledge_state",
    "upsert_pledge_events",
    "upsert_pledge_state",
    # prices + benchmark
    "count_price_rows",
    "get_trading_days",
    "load_benchmark",
    "load_prices",
    "upsert_benchmark",
    "upsert_prices",
    # panel
    "get_observation_dates",
    "get_panel_row",
    "load_panel",
    "replace_panel",
    # model runs + metrics
    "get_active_run",
    "get_model_run",
    "insert_metrics",
    "insert_model_run",
    "load_metrics",
    "load_model_comparison",
    "load_model_runs",
    "require_active_run",
    "set_active_run",
    # predictions + explanations
    "count_predictions",
    "delete_predictions_for_run",
    "delete_predictions_for_run_date",
    "load_explanations",
    "load_predictions",
    "save_explanations",
    "save_prediction",
    "save_predictions_bulk",
    # backtest
    "load_backtest_results",
    "save_backtest_results",
    # misc
    "table_counts",
    "utc_now",
]
