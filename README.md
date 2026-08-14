# PledgeCast

**An explainable early-warning system for promoter-pledge-driven downside risk in Indian equities —
and an honest test of whether pledge data predicts anything at all.**

---

## The result

> ### Pledge trajectory adds **no** incremental early warning once volatility and size are accounted for.

The whole project answers one question: *does pledge trajectory tell you anything that volatility and
company size do not already tell you?* The answer is the difference between two models scored on the
same metric — the full 13-feature model minus a null baseline given only `volatility_90d` and
`log_turnover_90d`.

| model | expB_full (13 features) | exp0_null (volatility + size) | **delta** |
|---|---|---|---|
| XGBoost *(deployed)* | 0.6240 | 0.6383 | **−0.0143** |
| Random Forest | 0.6173 | 0.6356 | **−0.0183** |
| Logistic Regression | 0.6132 | 0.6359 | **−0.0227** |

**Median delta −0.0183. All three models are negative.** Within-quarter ROC-AUC, 11 walk-forward
folds, 5,696 labelled company-quarters.

Three independent lines of evidence agree:

1. **The ablation.** Pledge levels + market features score 0.6243; adding the five trajectory
   features (change, acceleration, streak, rolling max) moves it to 0.6240. The *sequence* carries
   nothing beyond the *level* — which is the hypothesis §3.2 was built to test.
2. **Pledge features alone** reach 0.5506 — barely above the 0.50 coin flip.
3. **The SHAP confound audit.** Market features hold **73.9%** of total |SHAP|; the top three are
   `volatility_90d`, `trailing_dd_60d`, `return_90d`. The best pledge feature ranks 4th of 13.

### Why this is the finding and not a failure

Measured on this panel, **the quarterly pledge percentage does not change in 90.5% of
company-quarters.** JPPOWER sits at exactly 79.20% for fourteen consecutive quarters. Four of the
eight pledge features are therefore zero for nine rows in ten.

That is a fact about India's quarterly disclosure regime, not about the model. Promoters pledge in a
block and leave it; anything happening between filings is invisible at this frequency. A system built
to *display* that number — as commercial screeners do — cannot be early warning, and this project is
the measurement that shows why.

A negative result was a designed-for outcome. The experiment ladder exists precisely so the answer
can come back "no" and still be worth reporting.

---

## The methodological problem this project exists to solve

Pledged companies are overwhelmingly **leveraged smallcaps**. A model handed only volatility and
turnover separates the target almost as well as the full model while learning nothing whatsoever
about pledging.

A naive version of this project would report AUC ≈ 0.62, call it a pledge-risk model, and be
**entirely confounded**. Three deliberate choices prevent that:

| Choice | Why |
|---|---|
| **A null baseline** (`exp0_null`: volatility + size only) | The headline is the *incremental* AUC over it, never the raw score |
| **Within-quarter AUC as the primary metric** | The event rate per observation date ranges **1.0% → 60.7%**, a 60× spread. Pooled AUC is largely rewarded for telling one quarter from another — market timing, not company selection |
| **The universe is the NIFTY 500, not the pledge list** | 267 of 300 companies have quarters with no pledge at all. Without them there is no control group |

---

## Data

Built from primary regulatory filings, not a packaged dataset.

| | |
|---|---|
| **Companies** | 300 (NIFTY 500 spine) |
| **Window** | 2021-09-30 → 2026-06-30, 20 quarters |
| **Panel** | 6,000 company-quarters · 5,696 labelled · **22.9% event rate** |
| **XBRL filings** | 5,995 parsed, **0 quarantined** |
| **Reg 31 events** | 35,584 pledge creation / release / invocation disclosures |
| **Prices** | 373,802 adjusted daily bars + NIFTY 50 benchmark |

**The archive depth was verified before anything was designed.** NSE's XBRL shareholding archive
begins 30 September 2021 — five years, not the ten originally assumed. That measurement forced the
20-quarter window, which in turn is why a recurrent model was rejected.

### Three XBRL taxonomy eras, one of them silent

The parser handles three incompatible generations of NSE's shareholding schema. The third change is
the dangerous one: **from 2025-09-30, percentages are filed as fractions** — `0.24` where the same
field previously read `24.00`. Both use `unitRef="pure"` and `decimals="INF"`, so nothing in the
document declares the change. It is a silent 100× error.

It is detected per file using a fact true by definition — all shareholders together hold 100% — and
caught a real scale bug during the build gate.

---

## Architecture

```
NSE XBRL / Reg 31 / Yahoo prices
            │
      ingest/          network I/O, retries, resumable, immutable raw files
            │
      db/repository.py ALL SQL lives here. SQLite + WAL, SQLAlchemy Core
            │
   ★  data/panel.py    THE POINT-IN-TIME RULE — the one file to audit
            │              observation_date = quarter_end + 30 days
            │              a filing enters only if submission_date <= that date
            │
      features/        8 pledge trajectory + 5 market features
      labels/          forward 60-day max drawdown from entry, threshold −15%
            │
      training/        walk-forward, expanding window, 11 folds, embargo
      evaluation/      within-quarter AUC · quintile backtest · ★ leakage proofs
      explain/         SHAP: global beeswarm, local waterfall, templated text
            │
   ★  inference/service.py   the ONLY scoring path
            │
      ┌─────┴─────┐
   api/         dashboard/     FastAPI (5 endpoints) · Streamlit (3 pages)
```

Two boundaries carry the design:

- **Nothing downstream of panel assembly can see data filed after the observation date.** One file
  enforces it, and `evaluation/leakage.py` proves it rather than claiming it.
- **The API and the dashboard import the same service.** One scoring path, no drift — and the
  dashboard does not require the API to be running.

---

## Validation

### Walk-forward only — k-fold is forbidden here

```
train Q1..Q8   ─► test Q9        11 folds over 19 labelled dates
train Q1..Q9   ─► test Q10
        ⋮
train Q1..Q18  ─► test Q19

EMBARGO: the final quarter is featured but never labelled —
         its label needs 60 trading days of prices that do not exist yet.
```

Random folds would place future quarters in the training set. Scaling, imputation and winsorisation
are fit on the **training fold only**, inside a `Pipeline`, so the discipline is structural rather
than remembered.

### The non-negotiable check

> **Shuffle the labels, retrain, confirm AUC collapses to ~0.50.**

```
repeats  0.4968  0.4969  0.5015      mean 0.4984      PASSED
```

Labels are permuted *within each observation date* — the strict form, which preserves per-date class
balance so a model cannot score by learning "this quarter was bad for everyone". The full
walk-forward is re-run per repeat, not a single fit, so a leak entering through the fold structure
would still fail it.

### Per-fold spread, not a pooled mean

Within-quarter AUC ranges **0.389 → 0.758** across the 11 folds. With positives clustered around
market-wide corrections, a single number overstates confidence considerably.

### Economic backtest (quintiles cut *within* each quarter)

| | expB_full | exp0_null |
|---|---|---|
| Q1 (safest) event rate | 11.1% | 12.0% |
| Q5 (riskiest) event rate | 33.8% | 33.3% |
| Q5 − Q1 | 0.227 | 0.213 |
| monotonic across all 5 quintiles | 5 of 11 dates | 6 of 11 |

The pledge-aware model separates marginally better pooled and marginally worse on monotonicity — both
differences sit far inside the per-quarter spread, which runs from roughly 0 to 0.53. Neither claim
survives the noise.

---

## Why not a GRU

Evaluated and **rejected on sample-size grounds**, before any code was written.

| Requirement | Reality |
|---|---|
| Sequence length | 20 quarters; walk-forward training starts at ~8 timesteps |
| Independent sequences | ~300 companies |
| Complex temporal structure | Pledge % is a slow **step function** — flat, then a jump on a filing |

A GRU with 6-step sequences trained on 8 quarters fits thousands of parameters to a few hundred
effective examples. It would lose to XGBoost, and the only way to make it win is to tune on the test
folds — the exact methodological sin the rest of this project is built to avoid.

**The sequence hypothesis does not require a sequence model.** `pledge_chg_1q`, `pledge_chg_2q`,
`pledge_accel`, `consecutive_rising_q` and `pledge_max_4q` *are* the temporal signal, explicitly
encoded. So it was tested directly as a static-versus-trajectory ablation — and answered: **0.6243
static, 0.6240 with trajectory.** The sequence carries nothing here.

Choosing the model that fits the data is the engineering decision.

---

## Stack

Python 3.11 · pandas 2.2.3 / numpy 1.26.4 (stepped back deliberately — the combination `shap` is
known-good against) · SQLite + SQLAlchemy **Core, no ORM** · scikit-learn 1.5.2 · XGBoost 2.1.3 ·
SHAP 0.46 · FastAPI · Streamlit · Plotly (screens) / matplotlib (report figures) · pydantic v2 ·
pytest · ruff.

**SQLite over Parquet**, deliberately: this workload appends prediction history with timestamps,
queries one company's trajectory by index, and relates predictions → model runs → metrics. Parquet
does none of those well — no indexes, no relations, no appends. At 6,000 rows its columnar advantage
is irrelevant, and ACID plus WAL concurrency come free. Raw XML stays on disk with a checksum ledger
in the DB; 2 GB of BLOBs would bloat every query for nothing.

**No MLflow.** `model_runs` + `model_metrics` + a `config_snapshot` per run is experiment tracking at
this scale.

---

## Running it

```bash
make setup        # venv (outside the repo) + exact pins
cp .env.example .env

make init-db      # schema, WAL
make universe     # 300 companies -> companies + data/universe.csv
make ingest       # XBRL + Reg 31 + prices. Resumable. ~20 min, ~2 GB
make build        # point-in-time panel + leakage tests      GATE 2
make train        # walk-forward, 3 models x 4 experiments   GATE 3
make evaluate     # quintile backtest + SHAP + figures
make score        # score the latest quarter

make api          # http://127.0.0.1:8000/docs
make app          # http://localhost:8501
make test         # 85 tests
make test-critical  # the 38 that matter most: leakage, parser, labels
```

Every threshold, path, seed and hyperparameter lives in `config.yaml`. *"Re-run at 20% over 90 days"*
is a one-line edit.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | DB reachable **and** active model loaded, together |
| `GET /model-info` | active run: model, experiment, features, hyperparams, fold metrics |
| `POST /predict` | score by symbol (features from the DB) or by raw feature vector |
| `GET /predictions` | history, filterable, paginated |
| `GET /companies/{symbol}/history` | pledge trajectory + past predictions |

```jsonc
// POST /predict  { "symbol": "JPPOWER", "include_explanation": true }
{
  "symbol": "JPPOWER", "observation_date": "2026-07-30",
  "probability": 0.439, "risk_decile": 10, "risk_band": "HIGH",
  "model": { "run_id": "20260814T0431_xgboost_expB_full", "model_name": "xgboost" },
  "explanation": {
    "top_features": [{ "feature": "volatility_90d", "value": 0.60,
                       "shap": 0.42, "direction": "increases_risk" }],
    "summary": "HIGH (probability 0.44, decile 10). Main drivers: 90-day volatility 60% (+0.42)..."
  },
  "warnings": ["this observation date has no realised outcome yet - it is a forward prediction"]
}
```

Errors follow one contract: **422** invalid body (pydantic, field-level) · **404** unknown symbol ·
**503** model or database unavailable · **500** with a `request_id` for log correlation.

The plain-English summary is a **template over ranked SHAP values — no LLM anywhere in this project.**

---

## Dashboard

Three Streamlit pages, and **not one line of HTML, CSS or JavaScript**:

- **Risk Scanner** — ranked watchlist, risk histogram, and a pledge-vs-risk scatter coloured by
  volatility that makes the confound visible at a glance
- **Company Investigation** — pledge trajectory with Reg 31 markers, price chart with drawdown
  windows shaded, SHAP waterfall, plain-English explanation
- **Model Validation** — quintile bars model-vs-null, separation per quarter, ROC/PR, and the
  limitations below stated on the page itself

![SHAP beeswarm](reports/figures/shap_beeswarm.png)

*The confound audit. Market features dominate; pledge features cluster near zero.*

![Calibration](reports/figures/calibration.png)

---

## Limitations — stated openly

1. **Survivorship bias.** The universe is *today's* NIFTY 500. Companies that collapsed or delisted
   during 2021–2026 are absent — precisely the most interesting cases. Point-in-time constituent
   lists are not freely available, so this **cannot be fully fixed**. The realised event rate here is
   a floor.
2. **Twenty quarters, eleven folds.** Per-fold AUC ranges 0.389–0.758. The mean summarises a very
   wide spread, and positives cluster around market-wide corrections rather than arriving
   independently.
3. **The volatility confound is mitigated, not eliminated.** The null baseline measures it; it does
   not remove it.
4. **Quarterly disclosure may simply be too slow.** The pledge percentage is unchanged in 90.5% of
   company-quarters. Reg 31 event disclosures are far more granular and are ingested (35,584 of
   them) but are not used as features — that is the most promising extension.
5. **Five years is less than a full market cycle.**
6. **Not investment advice.** This is a research artefact.

---

## Testing

85 tests. The priority order is deliberate — `make test-critical` runs the 38 that matter most.

| File | Covers | Priority |
|---|---|---|
| `test_leakage.py` | point-in-time rule · label window · fold disjointness · label-shuffle | ★★★ |
| `test_xbrl_parser.py` | pledge present · explicit zero · unavailable ≠ zero · malformed → quarantine · **scale correctness** | ★★★ |
| `test_labels.py` | known series → known drawdown · exact −15% boundary · insufficient future → null not 0 | ★★★ |
| `test_features.py` | QoQ change · acceleration needs 3 quarters · rise counter · forward-fill cap | ★★ |
| `test_repository.py` | upsert idempotency · surrogate-id preservation · empty → empty frame | ★★ |
| `test_api.py` | health · predict · 404 · 422 · persistence | ★★ |

Leakage tests come in **positive/negative pairs**: each plants the exact violation it claims to catch,
because a test that only ever sees correct input proves nothing about what it would catch. Parser
tests run against **committed real XBRL files**, not mocks.

---

## Novelty claim, phrased defensibly

> I did not find an existing public project that combines promoter-pledge **trajectory** features with
> walk-forward downside-risk validation and an economic backtest for Indian equities.

Commercial screeners *display* the current pledge number. None of them test whether it predicts
anything. **PledgeCast is the evaluation layer** — and its finding is that, at quarterly frequency,
it does not.

---

*Endpoint behaviour, archive depth and event rates were measured against live NSE and Yahoo APIs in
August 2026. NSE occasionally changes endpoint contracts without notice; re-verify before a fresh
run.*
