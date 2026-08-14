# PledgeCast — Implementation Blueprint

**An explainable ML early-warning system for promoter-pledge-driven downside risk in Indian equities.**

> This document is the single source of truth for the build. It supersedes all earlier specs and the
> preliminary prototype in `d:\pledge_cast` (kept only as reference — nothing from it is reused except
> the XBRL parsing logic, which is described here).

---

## Table of contents

1. [Verified data reality](#1-verified-data-reality)
2. [The research question (corrected)](#2-the-research-question-corrected)
3. [GRU decision](#3-gru-decision)
4. [Technology stack](#4-technology-stack)
5. [Storage: SQLite](#5-storage-sqlite)
6. [Database schema](#6-database-schema)
7. [Architecture](#7-architecture)
8. [Folder structure](#8-folder-structure)
9. [ML pipeline](#9-ml-pipeline)
10. [Robustness](#10-robustness)
11. [Explainability](#11-explainability)
12. [Visualizations](#12-visualizations)
13. [Backend & API](#13-backend--api)
14. [Dashboard](#14-dashboard)
15. [Testing](#15-testing)
16. [One-day build sequence](#16-one-day-build-sequence)
17. [Feature prioritization](#17-feature-prioritization)
18. [Interview positioning](#18-interview-positioning)

---

## 1. Verified data reality

Every figure below was **measured against the live APIs on 13 August 2026**, not assumed.
Re-verify §1.1 before starting — the archive boundary may have moved forward.

### 1.1 Findings

| Check | Result | Consequence |
|---|---|---|
| **SHP XBRL archive depth** | Oldest record = **30-Sep-2021** for JPPOWER, TCS and HFCL alike. Requesting 2015→2021 returns **1 record**, so this is a real archive boundary, not a 20-row page cap. | Study window is forced to **2021-Q3 → 2026-Q2 = 20 quarters**. The original "2015–2025" plan is impossible. |
| **Point-in-time anchor** | Every record carries `submissionDate`. Observed filing lag: **7–14 days** after quarter end (SEBI allows 21). | Leakage control is straightforward. Anchor observations at **quarter_end + 30 days**. |
| **Target event rate**<br>(60-trading-day drawdown ≥ 15%) | Mean **25.5%**. RELIANCE 3.8% · ITC 5.1% · TCS 14.0% · ADANIPOWER 23.3% · JPPOWER 34.3% · JTLIND 40.4% · IDEA 41.1% · HFCL 41.9% | **Not a rare event** — drop all class-imbalance machinery. But the **10× spread** is a confound that can fake a good model. See §2. |
| **Reg 31 event stream** | Works. ADANIPOWER back to **2017-03-24**; 199 events across 5 pilot companies. Fields include lender, reason, post-event %. | Longer history than the panel, but **per-promoter** and the % is of total equity. Use as a feature layer, not the state layer. |
| **Pledge universe snapshot**<br>`/api/corporate-pledgedata` | **1,540** companies. 528 above 5% pledge · 150 above 20% · 23 above 50%. Snapshot only, no history. | **Trap:** lists *only companies that reported pledge*. Building the universe from it leaves **no zero-pledge control group**. |
| **Financial statements**<br>`/api/corporates-financial-results` | 105 quarterly records for JPPOWER with `filingDate` + Ind-AS XBRL links. Consolidated and standalone both present. | P&L is quarterly, but **3 of the 4 originally-planned ratios need the balance sheet**, which is half-yearly at best. Deferred — see §17. |
| **Price + benchmark** | Yahoo chart API returns **1,239 daily bars over 5y** for `JPPOWER.NS` and `^NSEI`, with `adjclose` present. | Solved. Price history aligns almost exactly with the pledge window. **No `yfinance` dependency needed** — call the chart API directly. |
| **Download throughput** | **0.21 s/file** at 4 concurrent workers (0.26 s sequential). | 300 companies × 20 quarters ≈ 6,000 files ≈ **~35 min** including polite delays. A real universe is feasible in one day. |

### 1.2 Working endpoints

```
Shareholding master (quarterly filings + XBRL URLs)
  GET https://www.nseindia.com/api/corporate-share-holdings-master
      ?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY
      &symbol=SYMBOL&issuer=FULL+COMPANY+NAME

Regulation 31 pledge events
  GET https://www.nseindia.com/api/corporate-shareholding-disclosure
      ?type=reg31&index=equities&symbol=SYMBOL&issuer=FULL+COMPANY+NAME

Current pledge snapshot (whole market, no params)
  GET https://www.nseindia.com/api/corporate-pledgedata

Daily prices / benchmark
  GET https://query1.finance.yahoo.com/v8/finance/chart/SYMBOL.NS?range=5y&interval=1d
  GET https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI?range=5y&interval=1d
```

**NSE requires a session bootstrap:** GET `https://www.nseindia.com/` with a browser `User-Agent`
first to set cookies, then reuse that `requests.Session`. Include a `Referer` matching the
relevant corporate-filings page. Refresh the session on any 401/403.

### 1.3 Scope parameters

| Parameter | Value |
|---|---|
| Universe | ~300 NSE companies (NIFTY 500 spine, filtered for data completeness) |
| Window | 2021-Q3 → 2026-Q2 (20 quarters) |
| Panel size | ~6,000 company-quarter rows |
| Base event rate | ~25% |
| Walk-forward folds | ~8 |

---

## 2. The research question (corrected)

### 2.1 The confound

The measured event rate ranges from **3.8% (Reliance) to 41.9% (HFCL)**. A model given only
volatility and size separates those almost perfectly — while learning **nothing** about pledging.
Because promoter pledging correlates with being a leveraged smallcap, a naive build of this
project would report ROC-AUC ≈ 0.80 that is **almost entirely confound**.

### 2.2 The corrected question

> ❌ *"Can we predict downside events from pledge data?"*
>
> ✅ **"Does pledge trajectory add anything over what volatility and size already tell you?"**

This is sharper, is what a quant interviewer will actually probe, and costs one extra baseline model.

### 2.3 Experiment ladder

| Experiment | Features | Question |
|---|---|---|
| `exp0_null` | `volatility_90d`, `log_turnover_90d` | **The null.** How much of the label is just "small and jumpy"? |
| `expA_pledge` | 8 pledge features | Does pledge behaviour carry standalone signal? |
| `expB_full` | All 13 | **Headline result = `expB` − `exp0`** |
| `ablation_static` | 3 pledge *levels* + market | Does trajectory beat static levels? (replaces the GRU — see §3) |

**The headline number is `expB_within_quarter_AUC − exp0_within_quarter_AUC`.**
If it is near zero, that is a **legitimate negative finding** — report it. A negative result honestly
reported is a better interview conversation than a suspiciously good AUC.

### 2.4 Other corrections to the original spec

| Change | From → To | Reason |
|---|---|---|
| Study window | 2015–2025 → **2021-Q3 – 2026-Q2** | Forced by the archive boundary |
| Null baseline | — → **Experiment 0** | Without it the confound invalidates every result |
| Headline metric | Pooled ROC-AUC → **within-quarter ROC-AUC** | Pooled AUC is inflated by market timing |
| Label definition | Ambiguous "max drawdown" → **worst decline from entry** | Peak-to-trough fires on stocks that rallied then dipped |
| Observation date | Quarter end → **quarter end + 30 days** | Leak-free *and* cross-sectionally aligned |
| Size control | — → **`log_turnover_90d`** | The other half of the confound control |
| ROCE, OCF growth | Cut → **YoY revenue & PAT growth** (deferred) | Balance-sheet items are half-yearly |
| Universe source | Pledge list → **NIFTY 500 spine** | The pledge endpoint has no zero-pledge control group |
| Class imbalance handling | Planned → **dropped** | Measured base rate is 25%, not rare. **No SMOTE.** |
| Screener / Trendlyne | Candidate source → **not used** | NSE path verified; keeps provenance primary-source and licensing clean |
| Reg 31 backfill | Reconstruct pre-2021 state → **feature layer only** | Per-promoter events with a different denominator; reconstruction is error-prone |
| GRU | Planned → **rejected** | See §3 |

### 2.5 Known limitations (state these openly in the README)

- **Survivorship bias.** Today's NIFTY 500 excludes companies that collapsed or delisted during
  2021–2026 — precisely the most interesting cases. Point-in-time historical constituent lists
  are not freely available, so this **cannot be fully fixed**. Mitigate by adding back delisted or
  surveillance-moved names where NSE still serves their filings, and **disclose the residual bias**.
- **Only ~8 walk-forward folds**, with positives clustered in time around market-wide corrections.
  Report per-fold spread, never just a pooled mean.
- **5-year window** limits any claim about behaviour across a full market cycle.

---

## 3. GRU decision

> ## FINAL DECISION: **DO NOT USE GRU**

This is the technically correct call, not a scope compromise.

### 3.1 Why

A GRU needs long sequences, many of them, and temporal structure too complex to hand-engineer.

| Requirement | Reality | Verdict |
|---|---|---|
| Sequence length | 20 quarters max; walk-forward training starts at ~8 | Too short |
| Independent sequences | ~300 companies | Marginal |
| Complex temporal structure | Pledge % is a slow **step function** — flat, then a jump on a filing | Trivially captured by lags |
| Effective training samples (fold 1) | ~300 companies × ~2 usable windows | Severely underpowered |

A GRU with 6-step sequences trained on ~8 quarters fits thousands of parameters to a few hundred
effective examples. It will lose to XGBoost, and the only way to make it "win" is to tune on the test
folds — the exact methodological sin the rest of this project is built to avoid.

### 3.2 The replacement — test the same hypothesis, better

The sequence hypothesis does **not** require a sequence model. The features `pledge_chg_1q`,
`pledge_chg_2q`, `pledge_accel`, `consecutive_rising_q` and `pledge_max_4q` **are** the temporal
signal, explicitly encoded. A GRU would spend its capacity rediscovering them from 8 timesteps.

So run an **ablation** instead:

- **Static-only** — pledge levels at time *t* (3 features)
- **Trajectory** — levels + change / acceleration / streak (8 features)

If trajectory beats static, **the sequence carries information**. That answers the actual research
question with a method appropriate to the sample size.

### 3.3 Why traditional ML is the right choice

- ~6,000 rows × 13 features is squarely gradient-boosting territory
- Trees handle `NaN` natively (missing quarters are expected)
- SHAP `TreeExplainer` is **exact** for trees and runs in milliseconds
- Training takes seconds, so 3 models × 4 experiments × 8 folds is affordable

### 3.4 The interview answer

> "I evaluated a GRU and rejected it on sample-size grounds — 20 quarters gives roughly 8 training
> timesteps, which cannot support a recurrent model. And the temporal signal is a slow step function,
> which lags capture completely. So I tested the temporal hypothesis directly via a static-versus-
> trajectory feature ablation. Choosing the model that fits the data is the engineering decision;
> choosing the model that sounds impressive is the mistake."

This is **stronger** than "I built a GRU." Most candidates cannot articulate why they *didn't* use
deep learning.

---

## 4. Technology stack

Everything below is used. Nothing is decorative.

| Technology | Purpose | Where used | Why required |
|---|---|---|---|
| **Python 3.11** | Language | Everything | Widest wheel availability |
| **requests** | HTTP + NSE cookie session | `src/ingest/` | NSE requires a homepage hit to set cookies before API calls |
| **lxml** | XBRL parsing | `src/ingest/xbrl.py` | 3–5× faster than stdlib `ElementTree` across 6,000 files; tolerant of malformed XML |
| **pandas** | Panel assembly, feature engineering | `src/features/`, `src/data/` | Core data structure |
| **numpy** | Numeric ops, rolling windows | `src/features/`, `src/labels/` | Drawdown and volatility math |
| **sqlite3** (stdlib) | Storage engine | `src/db/` | Zero install, one file, transactional |
| **SQLAlchemy Core** | Schema definition + safe queries | `src/db/schema.py` | Typed schema in one place, parameterised queries, `pandas.read_sql` integration. **Core only — no ORM** |
| **scikit-learn** | LogReg, RandomForest, preprocessing, metrics | `src/models/`, `src/evaluation/` | Baseline models + metric implementations |
| **xgboost** | Primary model | `src/models/` | Best-in-class small-tabular; native NaN handling |
| **shap** | Explainability | `src/explain/` | `TreeExplainer` — exact for trees |
| **pydantic v2** | Config + API schemas + row validation | `config.py`, `src/api/`, `src/data/validate.py` | One validation library for three jobs |
| **pydantic-settings** | Typed settings from YAML + `.env` | `config.py` | Config as a typed object, not a dict |
| **PyYAML** | Config file parsing | `config.py` | Human-editable config |
| **fastapi** | Inference API | `src/api/` | Enforces train/inference separation; automatic request validation |
| **uvicorn** | ASGI server | Runtime | Runs FastAPI |
| **streamlit** | Dashboard | `dashboard/` | **Zero HTML/CSS/JS** — see §14 |
| **plotly** | Interactive charts | `dashboard/` | Native Streamlit integration; hover/zoom free |
| **matplotlib** | Static figures | `src/explain/`, `reports/` | SHAP emits matplotlib natively; README figures |
| **joblib** | Model serialization | `src/models/registry.py` | sklearn-standard; efficient with numpy arrays |
| **pytest** | Testing | `tests/` | ~22 tests (§15) |
| **ruff** | Lint + format | dev | Replaces black + flake8 + isort in one tool |
| **tqdm** | Progress bars | `src/ingest/` | You will watch a 35-minute download; make it legible |
| **python-dotenv** | Env var loading | `config.py` | Keeps machine-specific paths out of git |
| **logging** (stdlib) | Structured logs | everywhere | No dependency needed |
| **git** | Version control | repo | — |
| **venv** + **pip** | Environment + packages | — | Standard; no Poetry/conda overhead |

### 4.1 Deliberately excluded

| Excluded | Why |
|---|---|
| PyTorch / GRU | §3 |
| LightGBM | Redundant with XGBoost — same API, no new insight |
| Parquet | §5 |
| PostgreSQL / MongoDB | SQLite is sufficient and simpler at this scale |
| Alembic | Schema is created once |
| Airflow / Prefect | This is a quarterly batch job |
| MLflow / W&B | The `model_runs` + `model_metrics` tables do this |
| Docker | Optional, 20 min at the very end — adds nothing to the ML story |
| Redis / Celery | No async workload |
| Auth | No users, no sensitive data |
| React / any JS framework | §14 |
| LLMs / MCP / Kafka / Kubernetes | Correctly excluded in the original spec — keep them out |
| `yfinance` | The Yahoo chart API works directly with `requests` |

### 4.2 `requirements.txt`

```
pandas==2.2.3
numpy==1.26.4
requests==2.32.3
lxml==5.3.0
SQLAlchemy==2.0.36
scikit-learn==1.5.2
xgboost==2.1.3
shap==0.46.0
pydantic==2.10.3
pydantic-settings==2.6.1
PyYAML==6.0.2
fastapi==0.115.6
uvicorn==0.34.0
streamlit==1.41.1
plotly==5.24.1
matplotlib==3.9.3
joblib==1.4.2
pytest==8.3.4
ruff==0.8.4
tqdm==4.67.1
python-dotenv==1.0.1
```

> ### ⚠️ Do this before writing any other code
>
> The old venv had **pandas 3.0.5 / numpy 2.4.6**. `shap` is historically slow to follow major
> pandas/numpy releases. The pins above deliberately step **back** to pandas 2.2 / numpy 1.26 —
> the combination `shap` is known-good against.
>
> **Build a fresh venv, install, and run the smoke test in Phase 1.**
> If `shap` still breaks, do not debug it — adjust pins and move on.

---

## 5. Storage: SQLite

### 5.1 Why SQLite is genuinely right here (not a downgrade from Parquet)

| Requirement | Parquet | SQLite |
|---|---|---|
| Append prediction history with timestamps | Rewrite the whole file | `INSERT` ✅ |
| Query one company's history for the dashboard | Load everything, then filter | Indexed lookup ✅ |
| Relate predictions → model run → metrics | Manual joins across files | Foreign keys ✅ |
| Transactional partial-failure safety | None | ACID ✅ |
| Concurrent read from API + dashboard | File-locking problems | WAL mode ✅ |

At ~6,000 panel rows and ~375,000 price rows this is **far** below SQLite's comfortable ceiling.
Parquet's advantages (columnar scans over 100M+ rows) are irrelevant at this scale, while its
weaknesses (no indexes, no relations, no appends) hurt every day.

### 5.2 What goes where

| Location | Contents | Rationale |
|---|---|---|
| **SQLite** `data/pledgecast.db` | Every structured table, all predictions, metrics, model metadata, explanations | Queried, related, appended |
| **Filesystem** `data/raw/xbrl/` | Raw downloaded XML (~2.4 GB) | Immutable research data. As BLOBs it would bloat the DB and slow every query for no benefit. SQLite holds the **ledger** — what was downloaded, when, its hash, its parse status |
| **Filesystem** `models/` | `.joblib` artifacts | Binary blobs the DB never queries; `model_runs.artifact_path` points to them |
| **Filesystem** `data/quarantine/` | Files the parser refused + reason | Audit trail for data loss |

> **Interview answer:** *"Structured, queried, related data goes in SQLite. Large immutable binaries
> stay on disk with a pointer and a checksum in the database."*

### 5.3 How the ML pipeline reads it

One function in `src/pledgecast/db/repository.py`:

```python
def load_panel(conn, experiment: str) -> pd.DataFrame:
    """Load the ML-ready panel for a given experiment's feature set."""
    return pd.read_sql(
        "SELECT * FROM panel WHERE label_is_valid = 1 ORDER BY observation_date, symbol",
        conn,
    )
```

Walk-forward folds are then pure pandas date filters. No ORM, no lazy loading, no surprises.

---

## 6. Database schema

```sql
PRAGMA journal_mode = WAL;      -- concurrent API + dashboard reads
PRAGMA foreign_keys = ON;

-- ============================================================
-- REFERENCE
-- ============================================================
CREATE TABLE companies (
    symbol          TEXT PRIMARY KEY,
    company_name    TEXT NOT NULL,
    isin            TEXT,
    industry        TEXT,
    in_universe     INTEGER NOT NULL DEFAULT 1,   -- 0 = excluded, keeps audit trail
    added_at        TEXT NOT NULL
);

-- ============================================================
-- INGESTION LEDGER  (raw files stay on disk)
-- ============================================================
CREATE TABLE filings (
    filing_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL REFERENCES companies(symbol),
    quarter_end     TEXT NOT NULL,               -- ISO 'YYYY-MM-DD'
    submission_date TEXT NOT NULL,               -- ★ POINT-IN-TIME ANCHOR
    xbrl_url        TEXT NOT NULL,
    local_path      TEXT,
    sha256          TEXT,
    status          TEXT NOT NULL,               -- pending|downloaded|parsed|quarantined
    error_message   TEXT,
    fetched_at      TEXT,
    UNIQUE (symbol, quarter_end, xbrl_url)
);
CREATE INDEX idx_filings_symbol_q ON filings(symbol, quarter_end);
CREATE INDEX idx_filings_status   ON filings(status);

-- ============================================================
-- PARSED PLEDGE STATE
-- ============================================================
CREATE TABLE pledge_state (
    symbol               TEXT NOT NULL REFERENCES companies(symbol),
    quarter_end          TEXT NOT NULL,
    submission_date      TEXT NOT NULL,
    promoter_shares      REAL,
    pledged_shares       REAL,
    total_shares         REAL,
    promoter_holding_pct REAL,
    pledge_pct_promoter  REAL,                   -- pledged / promoter holding
    pledge_pct_equity    REAL,                   -- pledged / total equity
    pledge_status        TEXT NOT NULL,          -- PLEDGE_PRESENT|NO_PLEDGE|UNAVAILABLE
    filing_id            INTEGER REFERENCES filings(filing_id),
    PRIMARY KEY (symbol, quarter_end)
);
CREATE INDEX idx_pledge_qend ON pledge_state(quarter_end);

-- ============================================================
-- REG 31 EVENTS  (feature layer)
-- ============================================================
CREATE TABLE pledge_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL REFERENCES companies(symbol),
    event_date      TEXT NOT NULL,
    promoter_name   TEXT,
    event_type      TEXT,                        -- creation|release|invocation
    shares          REAL,
    pct_equity      REAL,
    lender          TEXT,
    reason          TEXT,
    UNIQUE (symbol, event_date, promoter_name, event_type, shares)
);
CREATE INDEX idx_events_symbol_date ON pledge_events(symbol, event_date);

-- ============================================================
-- PRICES
-- ============================================================
CREATE TABLE prices (
    symbol      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,
    adj_close   REAL NOT NULL,
    volume      REAL,
    PRIMARY KEY (symbol, trade_date)
);
CREATE INDEX idx_prices_date ON prices(trade_date);

CREATE TABLE benchmark (                          -- NIFTY 50 (^NSEI)
    trade_date  TEXT PRIMARY KEY,
    adj_close   REAL NOT NULL
);

-- ============================================================
-- ML-READY PANEL  (features + label in one row)
-- ============================================================
CREATE TABLE panel (
    symbol                TEXT NOT NULL REFERENCES companies(symbol),
    observation_date      TEXT NOT NULL,          -- ★ quarter_end + 30 days
    quarter_end           TEXT NOT NULL,

    -- pledge features (8)
    promoter_holding_pct  REAL,
    pledge_pct_promoter   REAL,
    pledge_pct_equity     REAL,
    pledge_chg_1q         REAL,
    pledge_chg_2q         REAL,
    pledge_accel          REAL,
    consecutive_rising_q  INTEGER,
    pledge_max_4q         REAL,

    -- market features (5)
    volatility_90d        REAL,
    trailing_dd_60d       REAL,
    return_90d            REAL,
    rel_return_90d        REAL,
    log_turnover_90d      REAL,

    -- data-quality flags
    is_stale              INTEGER NOT NULL DEFAULT 0,

    -- label
    fwd_max_drawdown      REAL,                   -- continuous, for analysis
    label                 INTEGER,                -- 1 if fwd_max_drawdown <= -0.15
    label_is_valid        INTEGER NOT NULL DEFAULT 1,  -- 0 = insufficient future prices

    PRIMARY KEY (symbol, observation_date)
);
CREATE INDEX idx_panel_obsdate ON panel(observation_date);
CREATE INDEX idx_panel_label   ON panel(label);

-- ============================================================
-- EXPERIMENT TRACKING
-- ============================================================
CREATE TABLE model_runs (
    run_id          TEXT PRIMARY KEY,            -- e.g. 20260813T1430_xgb_expB
    created_at      TEXT NOT NULL,
    model_name      TEXT NOT NULL,               -- logreg|random_forest|xgboost
    experiment      TEXT NOT NULL,               -- exp0_null|expA_pledge|expB_full|ablation_static
    feature_list    TEXT NOT NULL,               -- JSON array
    hyperparams     TEXT NOT NULL,               -- JSON
    random_seed     INTEGER NOT NULL,
    n_train_rows    INTEGER,
    n_folds         INTEGER,
    artifact_path   TEXT,                        -- models/<run_id>.joblib
    config_snapshot TEXT,                        -- full JSON config at run time
    is_active       INTEGER NOT NULL DEFAULT 0   -- exactly one = the serving model
);

CREATE TABLE model_metrics (
    run_id       TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    fold         INTEGER NOT NULL,               -- -1 = aggregate across folds
    metric_name  TEXT NOT NULL,
    metric_value REAL,
    PRIMARY KEY (run_id, fold, metric_name)
);

-- ============================================================
-- PREDICTIONS + EXPLANATIONS
-- ============================================================
CREATE TABLE predictions (
    prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES model_runs(run_id),
    symbol           TEXT NOT NULL,
    observation_date TEXT NOT NULL,
    probability      REAL NOT NULL,
    risk_decile      INTEGER,
    source           TEXT NOT NULL,              -- backtest|api|dashboard
    created_at       TEXT NOT NULL
);
CREATE INDEX idx_pred_run_sym ON predictions(run_id, symbol);
CREATE INDEX idx_pred_obsdate ON predictions(observation_date);
CREATE INDEX idx_pred_created ON predictions(created_at);

CREATE TABLE explanations (
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id) ON DELETE CASCADE,
    feature_name  TEXT NOT NULL,
    feature_value REAL,
    shap_value    REAL NOT NULL,
    PRIMARY KEY (prediction_id, feature_name)
);

CREATE TABLE backtest_results (
    run_id           TEXT NOT NULL REFERENCES model_runs(run_id) ON DELETE CASCADE,
    observation_date TEXT NOT NULL,
    quintile         INTEGER NOT NULL,           -- 1 = lowest risk, 5 = highest
    n_companies      INTEGER NOT NULL,
    n_events         INTEGER NOT NULL,
    event_rate       REAL NOT NULL,
    PRIMARY KEY (run_id, observation_date, quintile)
);
```

### 6.1 Storage decisions summary

| Data | Stored in SQLite? |
|---|---|
| Source data (raw XML) | ❌ Filesystem + ledger row |
| Cleaned/parsed data | ✅ `pledge_state`, `pledge_events`, `prices`, `benchmark` |
| Features | ✅ `panel` |
| Predictions + timestamps | ✅ `predictions` |
| Model version + metadata | ✅ `model_runs` |
| Model metrics | ✅ `model_metrics` |
| Explanation results | ✅ `explanations` |
| Backtest results | ✅ `backtest_results` |
| Model binaries | ❌ Filesystem, path in `model_runs.artifact_path` |

---

## 7. Architecture

```
NSE API + Yahoo API                          ← external sources
        │
        ▼
┌─────────────────────┐
│ 1. INGESTION        │  concurrent, resumable, rate-limited
│  src/ingest/        │  raw XML → data/raw/  ·  ledger → filings table
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. PARSE + VALIDATE │  lxml → pydantic row validation → quarantine on failure
│  src/ingest/xbrl.py │  → pledge_state, pledge_events, prices, benchmark
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. PANEL ASSEMBLY   │  ★ observation_date = quarter_end + 30d
│  src/data/panel.py  │  ★ filter: submission_date <= observation_date
└─────────┬───────────┘     THIS LAYER IS WHERE LEAKAGE IS PREVENTED
          ▼
┌─────────────────────┐
│ 4. FEATURES + LABEL │  8 pledge · 5 market · forward 60d drawdown from entry
│  src/features/      │  → panel table
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 5. TRAINING         │  walk-forward (~8 folds) × 3 models × 4 experiments
│  src/training/      │  fold-local scaling/imputation — never global
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 6. EVALUATION       │  within-quarter AUC · PR-AUC · P@20 · Brier
│  src/evaluation/    │  quintile backtest → backtest_results
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 7. PERSISTENCE      │  models/<run_id>.joblib + model_runs + model_metrics
│  src/models/        │  exactly one run flagged is_active = 1
└─────────┬───────────┘
          ▼
┌──────────────────────────────────────────────┐
│ 8. INFERENCE SERVICE   src/inference/service.py│  ← THE ONLY CODE THAT SCORES
│    load active model · validate · predict      │
│    · SHAP · persist prediction + explanation   │
└──────┬────────────────────────────────┬────────┘
       ▼                                ▼
┌──────────────┐                 ┌──────────────┐
│ 9a. FastAPI  │                 │ 9b. Streamlit│
│  src/api/    │                 │  dashboard/  │
└──────────────┘                 └──────────────┘
```

### 7.1 Layer responsibilities

| Layer | Owns | Never does |
|---|---|---|
| Ingestion | Network I/O, retries, raw persistence | Parse or interpret |
| Parse | XML → typed rows, quarantine | Network calls |
| Panel | The point-in-time join rule | Compute features |
| Features | Derived columns + label | Touch raw files |
| Training | Fold splitting, fitting, selection | Persist predictions |
| Evaluation | Metrics, backtest | Fit models |
| Inference | Score + explain + persist | Fit or train |
| API / Dashboard | Transport + presentation | Contain business logic |

### 7.2 The critical design choice

**Both FastAPI and Streamlit import the same `src/inference/service.py`.**

- One scoring path, tested once
- The dashboard does **not** depend on the API being up — the demo cannot break
- The API exists as the clean answer to *"how do you separate training from inference?"*

---

## 8. Folder structure

```
pledgecast/
├── README.md                    Result first, then method, then stack
├── requirements.txt             Exact == pins
├── config.yaml                  ALL tunable parameters — zero magic numbers in code
├── .env.example                 DB_PATH, API_PORT, LOG_LEVEL
├── .gitignore                   data/raw/, *.db, models/*.joblib, .venv/, logs/
├── pytest.ini                   test discovery + markers
├── ruff.toml                    Lint + format rules
├── Makefile                     make setup | ingest | build | train | api | app | test
│
├── .github/workflows/ci.yml     ruff → critical tests → full suite, on every push
├── .streamlit/config.toml       Dashboard theme. Configuration, not CSS (§14)
│
├── config.py                    pydantic-settings: YAML + .env → typed Settings object
│
├── docs/
│   └── PLAN.md                  ← this file
│
├── data/
│   ├── pledgecast.db            ← THE database (gitignored)
│   ├── raw/xbrl/                Immutable downloaded XML (gitignored)
│   ├── quarantine/              Files the parser refused + reason (audit trail)
│   └── universe.csv             Symbol list — COMMITTED, makes runs reproducible
│
├── src/pledgecast/
│   ├── __init__.py
│   ├── logging_config.py        Console + rotating file handler, one setup function
│   ├── exceptions.py            DataIngestionError, ParseError, ModelNotFoundError,
│   │                            InsufficientDataError, ValidationError
│   │
│   ├── db/
│   │   ├── schema.py            SQLAlchemy Core table definitions (§6)
│   │   ├── connection.py        get_connection() — WAL, foreign_keys, context manager
│   │   └── repository.py        ALL SQL lives here: upsert_pledge_state(), load_panel(),
│   │                            save_prediction(), get_active_run()
│   │
│   ├── ingest/
│   │   ├── nse_session.py       Cookie bootstrap, backoff, session refresh on 401/403
│   │   ├── universe.py          Build company list → companies table
│   │   ├── shareholding.py      Master API → filing URLs → concurrent download → ledger
│   │   ├── xbrl.py              Universal parser (port the working prototype) + quarantine
│   │   ├── reg31.py             Event disclosures → pledge_events
│   │   └── prices.py            Yahoo chart API → prices + benchmark
│   │
│   ├── data/
│   │   ├── validate.py          Pydantic row models + range/duplicate/null checks
│   │   ├── panel.py             ★ Point-in-time assembly. The most important file.
│   │   └── population.py        Declarative panel strata (column + bounds)
│   │
│   ├── features/
│   │   ├── pledge.py            8 trajectory features
│   │   ├── market.py            5 price-derived features
│   │   ├── events.py            Reg 31 event features — the second disclosure frequency
│   │   └── build.py             Orchestrates → panel table
│   │
│   ├── labels/
│   │   └── drawdown.py          Forward 60-day max drawdown from entry
│   │
│   ├── models/
│   │   ├── definitions.py       MODELS dict: name → (estimator, param grid)
│   │   ├── preprocessing.py     sklearn Pipeline: impute → winsorize → scale
│   │   └── registry.py          save_model(), load_active_model(), set_active()
│   │
│   ├── training/
│   │   ├── walkforward.py       Fold generator + embargo logic
│   │   └── train.py             Main loop: experiments × models × folds
│   │
│   ├── evaluation/
│   │   ├── metrics.py           within_quarter_auc(), precision_at_k(), brier
│   │   ├── backtest.py          Quintile event-rate separation
│   │   ├── power.py             ★ Bootstrap-t intervals, oracle ceiling, verdicts —
│   │   │                        what makes the null falsifiable rather than asserted
│   │   ├── sensitivity.py       Window + materiality sweeps. Diagnostic, never selection.
│   │   └── leakage.py           ★ Assertions + label-shuffle test
│   │
│   ├── explain/
│   │   └── shap_runner.py       TreeExplainer: global + local + human-readable text
│   │
│   ├── inference/
│   │   └── service.py           ★ SHARED by API and dashboard. Single scoring path.
│   │
│   └── api/
│       ├── main.py              FastAPI app, lifespan model loading
│       ├── schemas.py           Pydantic request/response models
│       └── routes.py            The 5 endpoints (§13)
│
├── dashboard/
│   ├── _bootstrap.py            Puts repo root + src/ on sys.path (twin of scripts/)
│   ├── app.py                   Entry + sidebar + shared cache
│   ├── theme.py                 Palette + the shared Plotly template (§14)
│   ├── components.py            Shared rendering: metrics, tables, charts, forest plot
│   └── pages/
│       ├── 1_Risk_Scanner.py
│       ├── 2_Company_Investigation.py
│       └── 3_Model_Validation.py
│
├── scripts/                     Numbered, idempotent, run top-to-bottom
│   ├── _bootstrap.py            Puts repo root + src/ on sys.path
│   ├── 00_init_db.py
│   ├── 01_build_universe.py
│   ├── 02_ingest_all.py
│   ├── 03_build_panel.py
│   ├── 04_train_all.py
│   ├── 05_evaluate_and_explain.py
│   ├── 06_score_latest.py
│   └── 07_sensitivity.py        Window + materiality sweeps, univariate table
│
├── tests/
│   ├── conftest.py              In-memory SQLite fixture, sample panel fixture
│   ├── fixtures/                3 committed XBRL files (pledge / no-pledge / malformed)
│   ├── test_xbrl_parser.py
│   ├── test_labels.py
│   ├── test_features.py
│   ├── test_event_features.py
│   ├── test_leakage.py          ★ Highest-value tests
│   ├── test_repository.py
│   ├── test_api.py
│   ├── test_power.py            Interval coverage, oracle ceiling, positive control
│   ├── test_population.py       Panel strata, and the cross-population guard
│   ├── test_sensitivity.py
│   ├── test_retention.py        Deletion order, and that VACUUM reclaims
│   ├── test_registry_prune.py
│   ├── test_warnings.py         ★ The sec.13.1 contract, selected by code not position
│   ├── test_theme.py            Palette agreement + the two encoding rules
│   └── test_dashboard.py        Every page renders; no HTML, checked on the AST
│
├── notebooks/                   NOT BUILT — exploration happened in scripts/07_sensitivity.py
│   └── 01_eda.ipynb             (planned: exploration only — no logic lives here)
│
├── reports/figures/             PNGs for README
└── logs/                        Rotating log files (gitignored)
```

### 8.1 Why key files exist

| File | Responsibility | Why it exists |
|---|---|---|
| `config.yaml` | Every threshold, path, seed, hyperparameter | *"Re-run at 20% / 90 days"* becomes a one-line edit — a strong interview answer |
| `db/repository.py` | All SQL in one module | No SQL scattered through the codebase; swappable backend later |
| `data/panel.py` | The `quarter_end + 30d` rule | The single point where leakage is prevented — **one file to audit** |
| `evaluation/leakage.py` | Assertions + shuffle test | **Proves** the pipeline is honest rather than claiming it |
| `inference/service.py` | The only scoring path | API and dashboard share it — no drift between them |
| `models/registry.py` | `is_active` model resolution | Model versioning without MLflow |
| `data/quarantine/` | Rejected files + reasons | Turns silent data loss into an auditable list |
| `data/universe.csv` | Committed symbol list | Makes the whole run reproducible from a clean clone |

---

## 9. ML pipeline

### 9.1 Features (13 total)

| Group | Feature | Definition |
|---|---|---|
| **Pledge** | `promoter_holding_pct` | Promoter shares / total shares |
| | `pledge_pct_promoter` | Pledged / promoter holding ← **primary level** |
| | `pledge_pct_equity` | Pledged / total equity |
| | `pledge_chg_1q` | QoQ change in `pledge_pct_promoter` |
| | `pledge_chg_2q` | 2-quarter change |
| | `pledge_accel` | `chg_1q` − previous `chg_1q` |
| | `consecutive_rising_q` | Streak length of rising quarters |
| | `pledge_max_4q` | Rolling 4-quarter maximum |
| **Market** | `volatility_90d` | Annualised std of daily log returns |
| | `trailing_dd_60d` | Realised max drawdown over the prior 60 days |
| | `return_90d` | 90-day return |
| | `rel_return_90d` | `return_90d` − NIFTY 90-day return |
| | `log_turnover_90d` | log(median daily price × volume) ← **size/liquidity proxy** |

> `log_turnover_90d` replaces market cap deliberately: it needs only price data (avoiding a whole
> shares-outstanding dependency) and is the better liquidity control anyway.

### 9.2 Target variable

```
Y(i,t) = 1  if  min( P[t+1 .. t+60] ) / P[t] − 1  ≤  −0.15
Y(i,t) = 0  otherwise

where P = ADJUSTED close  (adjclose — never raw close)
      60 = trading days, not calendar days
```

**Use "worst decline from entry," not "peak-to-trough within window."** The latter fires on a stock
that rose 40% then fell 15% — not a downside event for anyone holding it.

Also persist `fwd_max_drawdown` as a continuous column for distribution analysis.

> **Before locking the 15% threshold**, plot the drawdown distribution on the real panel and confirm
> the base rate lands near the measured ~25%. If it is wildly different, the label is wrong — stop
> and fix it before training anything.

### 9.3 The point-in-time rule

```
observation_date = quarter_end + 30 calendar days   (rolled to next trading day)

  Q ends 30-Jun-2026 ──► observation 30-Jul-2026 ──► label = next 60 trading days
                              │
                              └── include ONLY filings with submission_date <= 30-Jul-2026
```

**Why 30 days:** SEBI requires the shareholding pattern within 21 days of quarter end, and observed
lags are 7–14 days. A 30-day cutoff captures essentially everything while remaining strictly
leak-free — and it **aligns every company onto the same observation date**, which is required for
within-quarter evaluation (§9.6).

### 9.4 Validation — walk-forward

```
train Q1..Q8   ─► test Q9      ← ~8 folds total
train Q1..Q9   ─► test Q10
train Q1..Q10  ─► test Q11
                    ⋮
train Q1..Q18  ─► test Q19

EMBARGO: the final quarter can be featured but never labelled —
         its label needs 60 trading days of future prices that don't exist yet.
```

> **Standard k-fold cross-validation is forbidden here.** Random folds would place future quarters
> in the training set. Walk-forward is the time-series-correct equivalent — and being able to explain
> *why* is worth more in an interview than using CV.

**Fold hygiene:** scaling, imputation and winsorisation are fit on the **training fold only** and
then applied to the test fold. Never compute these statistics globally.

### 9.5 Models

| Model | Role | Configuration |
|---|---|---|
| **Logistic Regression** | Interpretable baseline | L2, scaled inputs, `class_weight='balanced'` |
| **Random Forest** | Second reference | `n_estimators=300`, `max_depth=8`, `min_samples_leaf=10` |
| **XGBoost** | **Primary** | `max_depth=3`, `n_estimators=400`, `learning_rate=0.04`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, `tree_method='hist'` |

**Hyperparameter tuning:** a **20-point random search on fold 1 only**, then freeze.
At ~6,000 rows a large search *is* overfitting. Say this out loud in the interview — it is a
maturity signal.

### 9.6 Metrics — and why each

| Metric | Why this one |
|---|---|
| **Within-quarter ROC-AUC** ★ | **PRIMARY.** Computed per observation date, then averaged across dates. The only metric immune to the market-timing confound — it asks *"on this date, did the model rank the right companies higher?"* Pooled AUC looks better and means less. |
| PR-AUC | Correct shape for a screening tool; report against the ~25% base rate |
| Precision@20 | Literally what the dashboard shows — the top-20 watchlist |
| Brier score | Calibration; report against a base-rate-only predictor |
| Quintile separation ratio | The economic result: Q5 event rate ÷ Q1 event rate |
| ~~Accuracy~~ | **NEVER** — 75% by always predicting "no event" |

**Report per-fold spread, not just the mean.** With ~8 folds and time-clustered positives, a single
pooled number overstates confidence.

### 9.7 Model selection

1. Highest **mean within-quarter AUC** across folds
2. Ties broken toward the **simpler** model
3. Retrain the winner on all labelled data
4. Save to `models/<run_id>.joblib`, insert `model_runs` row, set `is_active = 1`

### 9.8 The non-negotiable sanity check

> **Shuffle the labels, retrain, confirm AUC collapses to ~0.50.**
>
> Run this the moment the first model trains. If it does not collapse, **you have leakage** —
> stop everything and fix it before proceeding.

### 9.9 The economic backtest

Each quarter, split the universe into risk quintiles by predicted probability, then compare
realised event rates:

```
Q5 (highest risk)  ──►  realised event rate  ──┐
                                                ├──►  separation ratio
Q1 (lowest risk)   ──►  realised event rate  ──┘
```

**Compute the same table for the `exp0_null` model and show them side by side.** If pledge-aware
quintiles separate no better than volatility-only quintiles, the honest headline is *"pledge
trajectory adds no incremental early warning once volatility is accounted for"* — publish that.

Report **per quarter**, not just pooled — the spread shows whether the edge is stable or came from
one lucky correction.

---

## 10. Robustness

Proportionate to an easy-to-medium project. Not enterprise architecture.

| Failure mode | Handling |
|---|---|
| Missing input values | XGBoost handles `NaN` natively. LogReg: median-impute from **training-fold statistics only** + missingness indicator |
| Invalid input types | Pydantic coercion at the API boundary → HTTP 422 with field-level detail |
| Invalid ranges | Validators: `pledge_pct ∈ [0,100]`, `volatility > 0`, `probability ∈ [0,1]` |
| Missing dataset values | Forward-fill pledge state **max 1 quarter**, set `is_stale=1`, drop beyond |
| Duplicate records | Composite primary keys make duplicates structurally impossible; `INSERT OR REPLACE` |
| Outliers | Winsorise at 1st/99th percentile **per training fold**. Never clip the label |
| **Data leakage** | Dedicated `evaluation/leakage.py`: assert `submission_date <= observation_date`; assert label windows start strictly after; label-shuffle test; fold disjointness |
| Incorrect feature types | Explicit dtype enforcement on panel load; fail loudly |
| Model loading failure | `ModelNotFoundError` → API returns 503 with a clear message; dashboard shows *"no active model — run `make train`"* |
| Database failure | Connection context manager with rollback; API returns 503; dashboard degrades to a readable error, never a traceback |
| API errors | Global exception handler → structured JSON `{error, detail, request_id}` |
| Unexpected user input | Symbol validated against the `companies` table before any query |
| Empty datasets | Explicit empty-result checks → *"no data for this selection"*, not `IndexError` |
| Insufficient data | Company with < 3 quarters cannot produce `pledge_accel` → `InsufficientDataError`, excluded with a logged reason |
| Prediction failure | Try/except around scoring; log with traceback; return partial response with `explanation: null` |
| **Corporate actions** | Assert no single-day return < −35% without a matching split. A 1:2 split in raw prices looks like an exact −50% "crash" and would silently corrupt labels. **Always use `adjclose`.** |
| Network failure mid-ingest | Resumable downloader: skip files already on disk, exponential backoff, session refresh |

**Plus:**

- Stdlib `logging` → console + rotating file; `request_id` on all API logs
- All configuration in `config.yaml` — zero magic numbers in code
- Seeds fixed and stored in `model_runs.random_seed`
- Full config snapshot stored per run in `model_runs.config_snapshot`
- `GET /health` checks DB reachability **and** active-model availability together
- Raw files never overwritten — `data/raw/` is immutable research data

---

## 11. Explainability

**SHAP `TreeExplainer` on XGBoost** — exact for trees, milliseconds per row, no approximation tuning.

### 11.1 Three uses

1. **Global** — beeswarm summary plot.
   Critically, **this is where you audit the confound**: if `volatility_90d` and `log_turnover_90d`
   dominate while pledge features sit near zero, the honest conclusion is that pledge adds little.
   **Show that plot either way.**

2. **Local** — per-company waterfall on the investigation screen.

3. **Human-readable** — turn the top 3 SHAP values into a sentence via a template:

   > *"Elevated risk (probability 0.68, decile 9). Main drivers: promoter pledge rose 11.2pp over two
   > quarters (+0.14), pledge has risen 3 consecutive quarters (+0.09), 90-day volatility 52%
   > (+0.07). Offsetting: outperformed NIFTY by 4% (−0.03)."*

   Generated from `explanations` rows — **no LLM**, just a template over ranked SHAP values.
   This is what makes it feel like a risk tool instead of a black box.

### 11.2 Not doing

LIME (redundant with SHAP) · counterfactual explanations (scope) · SHAP interaction values
(slow, hard to defend) · SHAP for the Random Forest (TreeExplainer works, but one model's
explanations is enough).

---

## 12. Visualizations

**Rule:** Plotly for anything on a dashboard screen (interactivity earns its keep).
matplotlib for anything destined for the README or a report (static, reproducible, and what SHAP
emits natively). Do not use one library for both.

| Graph | Purpose | Data | Library | Where |
|---|---|---|---|---|
| Ranked risk table | The watchlist | `predictions` ⋈ `panel` | Streamlit native | Scanner |
| Risk distribution histogram | Is this a risky quarter overall? | `predictions` | Plotly | Scanner |
| **Pledge % vs. risk scatter, coloured by volatility** | ★ Visually exposes the confound | `panel` + `predictions` | Plotly | Scanner |
| Pledge trajectory line | The core narrative object | `pledge_state` | Plotly | Company |
| Price + shaded event windows | Where drawdowns actually happened | `prices`, `panel` | Plotly | Company |
| Reg 31 event markers on pledge line | Creation/release/invocation overlay | `pledge_events` | Plotly | Company |
| SHAP waterfall | Why *this* prediction | `explanations` | matplotlib → `st.pyplot` | Company |
| **Quintile event-rate bars (model vs. null)** | ★ **THE HEADLINE RESULT** | `backtest_results` | Plotly | Validation |
| Separation ratio over time | Stable edge or one lucky correction? | `backtest_results` | Plotly | Validation |
| ROC + PR curves by experiment | Whole contribution in one image | fold predictions | Plotly | Validation + README |
| Model comparison table | LogReg / RF / XGB across metrics | `model_metrics` | Streamlit native | Validation |
| SHAP beeswarm | Global importance + direction | SHAP values | matplotlib | README |
| Correlation heatmap | Feature redundancy | `panel` | matplotlib | Notebook |
| Label distribution by quarter | Justifies within-quarter AUC | `panel` | matplotlib | Notebook |
| Calibration curve | Are probabilities honest? | fold predictions | matplotlib | README |

**Excluded:** confusion matrix — threshold-dependent for a ranking tool. Include only at an explicitly
stated operating point, if at all.

---

## 13. Backend & API

**FastAPI**, 5 endpoints. All logic delegates to `src/inference/service.py`.

| Endpoint | Purpose | Needed? |
|---|---|---|
| `GET /health` | DB reachable + active model loaded | ✅ Proves the system knows its own state |
| `GET /model-info` | Active run: model, experiment, features, metrics, trained-at | ✅ Model transparency |
| `POST /predict` | Score by symbol (features from DB) or raw feature dict | ✅ Core |
| `GET /predictions` | History, filterable by symbol/date, paginated | ✅ Backs the history view |
| `GET /companies/{symbol}/history` | Pledge trajectory + past predictions | ✅ Backs the investigation screen |

**Excluded:** `POST /train` (training is not a web operation) · auth (no users) ·
`/batch-predict` (a script's job).

### 13.1 Request / response contract

```jsonc
// POST /predict  — request
{ "symbol": "JPPOWER", "include_explanation": true }

// response
{
  "symbol": "JPPOWER",
  "observation_date": "2026-07-30",
  "probability": 0.684,
  "risk_decile": 9,
  "risk_band": "HIGH",
  "model": { "run_id": "20260813T1430_xgb_expB", "model_name": "xgboost" },
  "explanation": {
    "top_features": [
      { "feature": "pledge_chg_2q", "value": 11.2, "shap": 0.141,
        "direction": "increases_risk" },
      { "feature": "consecutive_rising_q", "value": 3, "shap": 0.089,
        "direction": "increases_risk" }
    ],
    "summary": "Elevated risk driven by a 11.2pp pledge increase over two quarters..."
  },
  "warnings": ["pledge_state is 1 quarter stale"]
}
```

### 13.2 Error contract

| Status | Condition |
|---|---|
| 422 | Invalid request body (pydantic, field-level detail) |
| 404 | Unknown symbol |
| 503 | Model or database unavailable |
| 500 | Anything else, with `request_id` for log correlation |

**Every prediction is persisted** to `predictions` + `explanations` with `source='api'`.
Prediction history is a side effect of serving, not a separate feature to build.

---

## 14. Dashboard

> # You will not write a single line of HTML, CSS, or JavaScript.

Streamlit is pure Python. `st.dataframe(df)` renders a table; `st.plotly_chart(fig)` renders a chart;
`st.columns(3)` makes a three-column layout. Streamlit generates all the HTML/CSS/JS internally and
serves it. **There is no `.html` file, no stylesheet, no `npm`, no build step.**

**Minimum frontend knowledge required: none.** If you can write a Python function, you can build
this dashboard.

**One rule:** never use `st.markdown(..., unsafe_allow_html=True)`. The moment you do, you have
started writing CSS you would have to defend. Everything needed is covered by `st.columns`,
`st.tabs`, `st.metric`, `st.expander`, `st.selectbox`, `st.dataframe`.

### 14.1 Three pages

**1 — Risk Scanner**
Quarter selector · top-N slider · ranked table (symbol, probability, decile, pledge %, Δ pledge,
volatility) · risk histogram · pledge-vs-risk scatter coloured by volatility.

**2 — Company Investigation**
Symbol dropdown · metric row (current probability, pledge %, QoQ change, volatility) · pledge
trajectory with Reg 31 markers · price chart with shaded event windows · SHAP waterfall ·
human-readable explanation · prediction history table.

**3 — Model Validation**
Model comparison table · ROC/PR curves by experiment · **quintile bars (model vs. null)** ·
separation ratio over time · active model metadata · **a plainly-worded limitations box**
(20 quarters, survivorship bias, volatility confound).

### 14.2 Implementation notes

- Every data load wrapped in `@st.cache_data(ttl=300)`
- Reads SQLite directly through `repository.py`
- Scores through `inference/service.py` — the same code the API uses
- Handles all four failure modes from §10 with readable messages

---

## 15. Testing

~22 tests. Priority ordered — if time runs out, the ★★★ rows are the ones that matter.

| Module | Tests | Priority |
|---|---|---|
| `test_leakage.py` | (4) `submission_date <= observation_date` for every panel row · label window starts strictly after observation · train/test fold dates disjoint · **label-shuffle collapses AUC to ~0.5** | ★★★ |
| `test_xbrl_parser.py` | (5) pledge-present · explicit no-pledge · missing flag → `UNAVAILABLE` · malformed → quarantine not crash · numeric scale correctness | ★★★ |
| `test_labels.py` | (4) known series → known drawdown · exact −15% boundary · insufficient future data → null not 0 · adjusted-price sanity | ★★★ |
| `test_features.py` | (4) QoQ change on synthetic panel · acceleration needs 3 quarters · consecutive-rise counter · stale forward-fill capped at 1 quarter | ★★ |
| `test_repository.py` | (3) upsert idempotency · duplicate PK rejected · empty result → empty DataFrame not exception | ★★ |
| `test_api.py` | (4) `/health` 200 · `/predict` valid → 200 with `probability ∈ [0,1]` · unknown symbol → 404 · malformed body → 422 | ★★ |

`conftest.py` provides an in-memory SQLite fixture and a synthetic 3-company panel.
**Fixture XBRL files are committed** to `tests/fixtures/` — they make parser tests real, not mocked.

---

## 16. One-day build sequence

**Total ≈ 10–13 hours.** This is a long day, not a comfortable one.
A cut line is marked at ~7 hours that still produces a complete, defensible project.

| Phase | Time | Work |
|---|---|---|
| **1 · Setup** | 45 min | Fresh venv → `pip install -r requirements.txt` → **smoke test: import xgboost, shap, sklearn, streamlit, fastapi and fit a 100-row model.** If `shap` breaks, fix pins **now**, not at hour 8. Then folder skeleton, `git init`, `.gitignore` (`data/raw/`, `*.db`), `config.yaml`. |
| **2 · DB + ingest** | 2.5 h | `00_init_db.py` (schema, WAL) → `01_build_universe.py` (300 symbols → `companies`) → `02_ingest_all.py`. **Start the download and write the parser while it runs.** ~35 min XBRL, ~5 min prices, ~5 min Reg 31. Port the prototype parser — it already handles two XBRL variants. |
| **3 · Panel** | 1.5 h | `03_build_panel.py`: point-in-time join (`quarter_end + 30d`), 8 pledge + 5 market features, forward-drawdown label. **Then immediately run the leakage tests.** Confirm ~6,000 rows and ~25% event rate. If the rate is wildly off, the label is wrong — stop and fix. |
| **4 · Train** | 2 h | Walk-forward harness (~8 folds) → 3 models × 3 experiments + the static ablation → within-quarter AUC, PR-AUC, P@20, Brier → **label-shuffle check** → persist to `model_runs`/`model_metrics` → flag active. **Compute `expB − exp0`. That number is the result.** |
| **5 · Backtest + SHAP** | 1 h | Quintile event rates per quarter, model **and** null → `backtest_results`. TreeExplainer: beeswarm PNG, per-company waterfalls, human-readable summaries → `explanations`. |
| — | — | ⬛ **~7 h CUT LINE — a complete, defensible project exists here.** Everything below is presentation and engineering polish. |
| **6 · API** | 1 h | `inference/service.py` **first**, then FastAPI wrapping it. 5 endpoints, pydantic schemas, exception handlers, `/health`. |
| **7 · Dashboard** | 1.5 h | Three Streamlit pages, charts from §12, `@st.cache_data`, the four failure modes handled. |
| **8 · Tests** | 1 h | Write the 22 tests. **Leakage and parser tests first** — if time runs out, those are the ones that matter. |
| **9 · Finalize** | 1 h | README leading with the result. Architecture diagram. **The GRU rejection paragraph.** Limitations stated openly. Screenshots. `Makefile`. Final commit. |

### 16.1 Parallelization wins

- Start the download at the top of Phase 2 and **write the parser while it runs**
- Write tests during model training
- Draft the README while SHAP computes

### 16.2 If you fall behind — cut in this order

1. Reg 31 events + overlay
2. Random Forest (keep LogReg + XGBoost)
3. FastAPI (the dashboard already uses the service layer directly)
4. Streamlit page 3
5. Universe down to 150 companies

---

## 17. Feature prioritization

### ✅ MUST HAVE — in the build, non-negotiable

SQLite schema · ingestion with resume + rate limiting · XBRL parser with quarantine ·
point-in-time panel assembly · 13 features + label · walk-forward validation · LogReg + XGBoost ·
Exp 0 / A / B · within-quarter AUC + PR-AUC + Brier · leakage tests + label-shuffle ·
model persistence with `is_active` · SHAP global + local · Streamlit pages 1 & 2 · `config.yaml` ·
logging · README

### ✅ SHOULD HAVE — planned for today; this is the cut list if behind

Random Forest · static-vs-trajectory ablation · quintile backtest vs. null · Streamlit page 3 ·
FastAPI 5 endpoints · full 22 tests · Reg 31 events + overlay · human-readable explanations ·
prediction history · Precision@20 · calibration curve · Makefile

### ❌ NICE TO HAVE — **NOT being built.** Future-work list only.

| Item | What it is | Why it is cut |
|---|---|---|
| **Fundamentals block / Experiment C** | Parsing Ind-AS XBRL for debt/equity, interest coverage, revenue & PAT growth | A **second parser against a messier schema**. Ind-AS taxonomy varies by industry (banks file completely differently), and 3 of 4 ratios need half-yearly balance-sheet items. Realistically 4–6 hours with a high failure rate across 300 companies. **The biggest time sink on this list.** |
| **NCSKEW / DUVOL** | Academic crash-risk measures. NCSKEW = negative skewness of firm-specific weekly returns; DUVOL = down-period to up-period volatility ratio. Alternative *target variables* in the finance literature | ~50 lines of code, but **high explanation burden**: requires regressing each stock against the market for residual returns, then computing skewness. You would have to defend NCSKEW over drawdown in an interview — and the drawdown label is more intuitive. Academic garnish, zero demo value. |
| **TabPFN row** | A tabular foundation model — a transformer pretrained on synthetic tabular datasets that classifies via in-context learning, no gradient training | 5 lines to run, but a large model download plus a concept you must be able to explain. If asked *"what is a tabular foundation model?"* and you cannot answer, it is **worse than not having it**. Know it exists; mention it only if the topic comes up. |
| **Dockerfile** | Containerizing the app | Writing it is 20 min; *debugging* a Docker build on Windows can eat an hour. Adds nothing to the ML or research story. Easy to learn later. |
| **Streamlit Cloud deploy** | Free hosting → a live public URL | The blocker is not difficulty: the SQLite DB is gitignored and will be 50–150 MB, so you would need to commit a trimmed DB or rebuild on deploy. Fiddly — and fiddly-at-hour-13 is where projects break. **Worth doing on a later day**; a live link is genuinely valuable. |
| **Notebook EDA** | A polished exploration notebook | The exploration happens anyway inside the pipeline scripts. **One exception you must not skip:** checking the drawdown distribution before locking the 15% threshold — but that is 10 lines inside `03_build_panel.py`, not a notebook. |
| **Hyperparameter search beyond 20 points** | A larger grid/random search | **Not a time cut — methodologically correct to omit.** At ~6,000 rows with 8 folds, a 500-point search *is* overfitting: you would be selecting on noise. Keeping the search small makes the project **better**, and saying so is a maturity signal. |

**The pattern:** every item is either a large hidden time cost (fundamentals, deploy), a concept you
would have to defend without payoff (NCSKEW, TabPFN), or something genuinely better left out
(large hyperparameter search). **None of them strengthen the three sentences that carry the project**
(§18.1).

---

## 18. Interview positioning

**Problem.** Indian promoters pledge shares as loan collateral. When the price falls, lenders issue
margin calls, invoke the pledge, and dump shares — pushing the price down further. It is a documented
reflexive loop. Commercial screeners *display* the current pledge number; none test whether it
predicts anything. PledgeCast is the evaluation layer.

**Data.** Built from primary regulatory filings, not a Kaggle CSV. ~6,000 company-quarters across
300 NSE companies, assembled from NSE's XBRL shareholding disclosures, Regulation 31 pledge events,
and adjusted daily prices. I verified the archive depth empirically before designing anything — it
starts September 2021, which forced a 20-quarter window rather than the 10 years originally assumed.

**ML.** XGBoost, with logistic regression as an interpretable baseline and random forest as a second
reference. Six thousand rows across thirteen features is squarely gradient-boosting territory; trees
also handle missing quarters natively, and SHAP TreeExplainer gives exact attributions instantly.

**Why not GRU.** I evaluated it and rejected it on sample-size grounds: 20 quarters means walk-forward
training starts at roughly 8 timesteps, which cannot support a recurrent model. And the temporal
signal is a slow step function — flat, then a jump on a filing — which lags capture completely. So I
tested the sequence hypothesis directly with a static-versus-trajectory feature ablation instead.
Choosing the model that fits the data is the engineering decision.

**The methodological problem I found, and fixed.** I measured the base rate of my target across
companies and found it ranged from 4% for Reliance to 42% for HFCL. A model given only volatility and
size would separate those almost perfectly while learning nothing about pledging — and since pledge
correlates with being a leveraged smallcap, a naive version of this project would report a great AUC
that was pure confound. So I added a null baseline (volatility + size only) and made my headline
result the *incremental* AUC over it, evaluated within each quarter's cross-section rather than
pooled, because pooled AUC is inflated by market-wide timing.

**Architecture.** A linear batch pipeline with a strict boundary: nothing downstream of panel assembly
can see data filed after the observation date. Training and inference are separated by a shared
service layer that both the API and the dashboard call, so there is exactly one scoring path and no
drift between them.

**SQLite over Parquet.** I need to append prediction history with timestamps, query one company's
trajectory by index, and relate predictions to model runs to metrics. Parquet does none of those well
— no indexes, no relations, no appends. At this scale SQLite's columnar disadvantages are irrelevant,
and I get ACID and WAL concurrency for free. Raw XML stays on disk with a checksum ledger in the DB,
because 2 GB of BLOBs would bloat every query for no benefit.

**Explainability.** SHAP answers "why is this company flagged" per prediction, and — more importantly
— it is how I audit the confound. If volatility dominates the global beeswarm and pledge features sit
near zero, that is the honest finding, and I report it.

**Robustness.** Invalid input is rejected at the API boundary by pydantic with field-level errors.
Missing quarters forward-fill at most one period and get flagged stale. Companies with insufficient
history are excluded with a logged reason rather than silently imputed. Unparseable filings go to a
quarantine directory with the failure reason instead of becoming a silent zero. Every layer fails
loudly and logs why.

**Result.** A ranked quarterly watchlist, validated by walk-forward out-of-sample testing, with an
economic backtest comparing realised event rates in the highest-risk quintile versus the lowest —
reported alongside the same table for the null model, so the comparison is honest.

**Scaling.** The batch pipeline already *is* the production shape; you would add a scheduler for the
quarterly run. If the panel grew past a few million rows I would move `prices` and `panel` to Postgres
or DuckDB — the repository layer isolates all SQL, so that is a one-module change. The model itself
would not need to change.

### 18.1 The three sentences that matter most

1. **"I verified the data before designing the system."**
   You probed the API and discovered a 5-year archive, not 10. Most candidates design against
   assumptions.

2. **"I found a confound that would have faked a good result, and built the baseline that exposes it."**
   This is senior-level thinking and the single strongest thing in the project.

3. **"I rejected deep learning on sample-size grounds and tested the same hypothesis a better way."**
   Judgment beats tool-collecting.

### 18.2 Novelty claim — phrase it defensibly

> ❌ *"Nobody has done this."*
>
> ✅ **"I did not find an existing public project that combines promoter-pledge trajectory features
> with walk-forward downside-risk validation and an economic backtest for Indian equities."**

The first gets you caught. The second is defensible.

---

*Endpoint behaviour, archive depth, event rates and download throughput in this document were
measured against live NSE and Yahoo APIs on 13 August 2026. Re-verify §1.1 before starting — the
archive boundary may extend forward, and NSE occasionally changes endpoint contracts without notice.*
