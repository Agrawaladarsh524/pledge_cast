"""SQLAlchemy Core table definitions - a 1:1 port of PLAN.md sec.6 DDL.

PLAN.md sec.4: "SQLAlchemy Core - Typed schema in one place, parameterised
queries, pandas.read_sql integration. **Core only - no ORM**."

Conventions carried over from sec.6:
  * every date is TEXT in ISO 'YYYY-MM-DD' form - SQLite has no date type, and
    ISO strings sort and compare correctly as text
  * composite primary keys make duplicate rows structurally impossible (sec.10)
  * raw XBRL stays on disk; ``filings`` is only the ledger (sec.5.2)
"""

from __future__ import annotations

from sqlalchemy import (
    REAL,
    Column,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)

metadata = MetaData()

# ============================================================================
# REFERENCE
# ============================================================================
companies = Table(
    "companies",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("company_name", Text, nullable=False),
    Column("isin", Text),
    Column("industry", Text),
    # 0 = excluded, keeps the audit trail rather than deleting the row.
    Column("in_universe", Integer, nullable=False, server_default=text("1")),
    Column("added_at", Text, nullable=False),
)

# ============================================================================
# INGESTION LEDGER  (raw files stay on disk - sec.5.2)
# ============================================================================
filings = Table(
    "filings",
    metadata,
    Column("filing_id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", Text, ForeignKey("companies.symbol"), nullable=False),
    Column("quarter_end", Text, nullable=False),
    # THE point-in-time anchor. Nothing may enter a panel row whose
    # submission_date is after that row's observation_date (sec.9.3).
    Column("submission_date", Text, nullable=False),
    Column("xbrl_url", Text, nullable=False),
    Column("local_path", Text),
    Column("sha256", Text),
    # pending | downloaded | parsed | quarantined
    Column("status", Text, nullable=False),
    Column("error_message", Text),
    Column("fetched_at", Text),
    UniqueConstraint("symbol", "quarter_end", "xbrl_url", name="uq_filings_symbol_q_url"),
    sqlite_autoincrement=True,
)
Index("idx_filings_symbol_q", filings.c.symbol, filings.c.quarter_end)
Index("idx_filings_status", filings.c.status)

# ============================================================================
# PARSED PLEDGE STATE
# ============================================================================
pledge_state = Table(
    "pledge_state",
    metadata,
    Column("symbol", Text, ForeignKey("companies.symbol"), primary_key=True),
    Column("quarter_end", Text, primary_key=True),
    Column("submission_date", Text, nullable=False),
    Column("promoter_shares", REAL),
    Column("pledged_shares", REAL),
    Column("total_shares", REAL),
    Column("promoter_holding_pct", REAL),
    Column("pledge_pct_promoter", REAL),  # pledged / promoter holding
    Column("pledge_pct_equity", REAL),  # pledged / total equity
    # PLEDGE_PRESENT | NO_PLEDGE | UNAVAILABLE. The three-way split matters:
    # a missing flag must never collapse into a silent zero (sec.15 parser test).
    Column("pledge_status", Text, nullable=False),
    Column("filing_id", Integer, ForeignKey("filings.filing_id")),
)
Index("idx_pledge_qend", pledge_state.c.quarter_end)

# ============================================================================
# REG 31 EVENTS  (feature layer only - sec.2.4)
# ============================================================================
pledge_events = Table(
    "pledge_events",
    metadata,
    Column("event_id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", Text, ForeignKey("companies.symbol"), nullable=False),
    Column("event_date", Text, nullable=False),
    Column("promoter_name", Text),
    Column("event_type", Text),  # creation | release | invocation
    Column("shares", REAL),
    Column("pct_equity", REAL),
    Column("lender", Text),
    Column("reason", Text),
    UniqueConstraint(
        "symbol",
        "event_date",
        "promoter_name",
        "event_type",
        "shares",
        name="uq_events_natural_key",
    ),
    sqlite_autoincrement=True,
)
Index("idx_events_symbol_date", pledge_events.c.symbol, pledge_events.c.event_date)

# ============================================================================
# PRICES
# ============================================================================
prices = Table(
    "prices",
    metadata,
    Column("symbol", Text, primary_key=True),
    Column("trade_date", Text, primary_key=True),
    # ALWAYS adjusted close. sec.10: raw close makes a 1:2 split look like an
    # exact -50% crash and would silently corrupt every label.
    Column("adj_close", REAL, nullable=False),
    Column("volume", REAL),
)
Index("idx_prices_date", prices.c.trade_date)

benchmark = Table(  # NIFTY 50 (^NSEI)
    "benchmark",
    metadata,
    Column("trade_date", Text, primary_key=True),
    Column("adj_close", REAL, nullable=False),
)

# ============================================================================
# ML-READY PANEL  (features + label in one row)
# ============================================================================
panel = Table(
    "panel",
    metadata,
    Column("symbol", Text, ForeignKey("companies.symbol"), primary_key=True),
    # quarter_end + 30 days, rolled to the next trading day (sec.9.3).
    Column("observation_date", Text, primary_key=True),
    Column("quarter_end", Text, nullable=False),
    # --- pledge features (8) ------------------------------------------------
    Column("promoter_holding_pct", REAL),
    Column("pledge_pct_promoter", REAL),
    Column("pledge_pct_equity", REAL),
    Column("pledge_chg_1q", REAL),
    Column("pledge_chg_2q", REAL),
    Column("pledge_accel", REAL),
    Column("consecutive_rising_q", Integer),
    Column("pledge_max_4q", REAL),
    # --- market features (5) ------------------------------------------------
    Column("volatility_90d", REAL),
    Column("trailing_dd_60d", REAL),
    Column("return_90d", REAL),
    Column("rel_return_90d", REAL),
    Column("log_turnover_90d", REAL),
    # --- data-quality flags -------------------------------------------------
    Column("is_stale", Integer, nullable=False, server_default=text("0")),
    # --- label --------------------------------------------------------------
    Column("fwd_max_drawdown", REAL),  # continuous, for distribution analysis
    Column("label", Integer),  # 1 if fwd_max_drawdown <= threshold
    # 0 = insufficient future prices (the embargo quarter, sec.9.4)
    Column("label_is_valid", Integer, nullable=False, server_default=text("1")),
)
Index("idx_panel_obsdate", panel.c.observation_date)
Index("idx_panel_label", panel.c.label)

# ============================================================================
# EXPERIMENT TRACKING  (this is what replaces MLflow - sec.4.1)
# ============================================================================
model_runs = Table(
    "model_runs",
    metadata,
    Column("run_id", Text, primary_key=True),  # e.g. 20260813T1430_xgb_expB
    Column("created_at", Text, nullable=False),
    Column("model_name", Text, nullable=False),  # logreg|random_forest|xgboost
    # exp0_null | expA_pledge | expB_full | ablation_static
    Column("experiment", Text, nullable=False),
    Column("feature_list", Text, nullable=False),  # JSON array
    Column("hyperparams", Text, nullable=False),  # JSON
    Column("random_seed", Integer, nullable=False),
    Column("n_train_rows", Integer),
    Column("n_folds", Integer),
    Column("artifact_path", Text),  # models/<run_id>.joblib
    Column("config_snapshot", Text),  # full JSON config at run time (sec.10)
    # Exactly one row may hold 1 - enforced in repository.set_active_run().
    Column("is_active", Integer, nullable=False, server_default=text("0")),
)

model_metrics = Table(
    "model_metrics",
    metadata,
    Column(
        "run_id",
        Text,
        ForeignKey("model_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("fold", Integer, primary_key=True),  # -1 = aggregate across folds
    Column("metric_name", Text, primary_key=True),
    Column("metric_value", REAL),
)

# ============================================================================
# PREDICTIONS + EXPLANATIONS
# ============================================================================
predictions = Table(
    "predictions",
    metadata,
    Column("prediction_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Text, ForeignKey("model_runs.run_id"), nullable=False),
    Column("symbol", Text, nullable=False),
    Column("observation_date", Text, nullable=False),
    Column("probability", REAL, nullable=False),
    Column("risk_decile", Integer),
    Column("source", Text, nullable=False),  # backtest | api | dashboard
    Column("created_at", Text, nullable=False),
    sqlite_autoincrement=True,
)
Index("idx_pred_run_sym", predictions.c.run_id, predictions.c.symbol)
Index("idx_pred_obsdate", predictions.c.observation_date)
Index("idx_pred_created", predictions.c.created_at)

explanations = Table(
    "explanations",
    metadata,
    Column(
        "prediction_id",
        Integer,
        ForeignKey("predictions.prediction_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("feature_name", Text, primary_key=True),
    Column("feature_value", REAL),
    Column("shap_value", REAL, nullable=False),
)

backtest_results = Table(
    "backtest_results",
    metadata,
    Column(
        "run_id",
        Text,
        ForeignKey("model_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("observation_date", Text, primary_key=True),
    Column("quintile", Integer, primary_key=True),  # 1 = lowest risk, 5 = highest
    Column("n_companies", Integer, nullable=False),
    Column("n_events", Integer, nullable=False),
    Column("event_rate", REAL, nullable=False),
)

# ---------------------------------------------------------------------------
ALL_TABLES: tuple[str, ...] = tuple(metadata.tables)

# Valid enum-ish values, kept next to the schema so callers never invent new ones.
FILING_STATUSES = ("pending", "downloaded", "parsed", "quarantined")
PLEDGE_STATUSES = ("PLEDGE_PRESENT", "NO_PLEDGE", "UNAVAILABLE")
PREDICTION_SOURCES = ("backtest", "api", "dashboard")
EVENT_TYPES = ("creation", "release", "invocation")

__all__ = [
    "ALL_TABLES",
    "EVENT_TYPES",
    "FILING_STATUSES",
    "PLEDGE_STATUSES",
    "PREDICTION_SOURCES",
    "backtest_results",
    "benchmark",
    "companies",
    "explanations",
    "filings",
    "metadata",
    "model_metrics",
    "model_runs",
    "panel",
    "pledge_events",
    "pledge_state",
    "predictions",
    "prices",
]
