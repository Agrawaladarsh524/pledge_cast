# PledgeCast

## An explainable early-warning system for promoter-pledge-driven downside risk in Indian equities

---

## The result

> ### Pledge trajectory adds **nothing distinguishable from zero** once volatility and size are accounted for — and an effect up to **+0.19 AUC** would have been detected had one existed.

The whole project answers one question: *does pledge trajectory tell you anything that volatility and
company size do not already tell you?* The answer is the difference between two models scored on the
same metric — the full 13-feature model minus a null baseline given only `volatility_90d` and
`log_turnover_90d`.

| model | expB_full (13 features) | exp0_null (volatility + size) | delta | 95% interval | verdict |
|---|---|---|---|---|---|
| XGBoost *(deployed)* | 0.6240 | 0.6383 | −0.0143 | [−0.0936, +0.0137] | **ZERO** |
| Random Forest | 0.6173 | 0.6356 | −0.0183 | [−0.0490, +0.0097] | **ZERO** |
| Logistic Regression | 0.6132 | 0.6359 | −0.0227 | [−0.1288, +0.0008] | **ZERO** |

Within-quarter ROC-AUC, 11 walk-forward folds, 5,696 labelled company-quarters. Intervals are a
studentised (bootstrap-t) block bootstrap over observation dates.

**The deltas are inside their own intervals, so the finding is zero — not "slightly negative".**
An earlier version of this README counted signs and reported "all three models are negative". That was
wrong: with 11 test dates the interval half-width is about 0.03–0.06 and the deltas are about 0.02, so
the sign carries no information. Counting three correlated models fitted on the same rows does not make
it three pieces of evidence either. Every difference in this README is now reported with an interval
and a verdict, and the verdict is read off the interval rather than off the sign.

**The interval is itself validated.** An interval is the one kind of code that always looks like it
works — it returns two plausible numbers whether or not it is calibrated. So `tests/test_power.py`
measures the actual coverage of the nominal 95% interval by simulation at this study's real sample
size. The first implementation used a plain percentile bootstrap and **failed**:

| method | normal | skewed | heavy-tailed | false positives at a true zero |
|---|---|---|---|---|
| percentile *(first attempt)* | 90.8% | 86.9% | 90.4% | **7.2%** |
| **bootstrap-t** *(shipped)* | **95.1%** | **94.4%** | **93.0%** | **3.6%** |
| student-t | 95.0% | 90.4% | 96.2% | 3.9% |

A 90% interval was wearing a 95% label. Fixing it cost real power — detection of a true +0.03 effect
falls from 64% to 48% — and **flipped one published verdict**: `expH_pledged_events` (logreg) went from
NEGATIVE −0.0408 [−0.0716, −0.0061] to **ZERO** [−0.0754, +0.0102]. It was a false positive
manufactured by the narrow interval. Student-t was rejected despite being simpler because per-date AUC
differences have no reason to be symmetric, and it drops to 90.4% coverage under skew.

**And the test had room to find something.** A null is only worth reporting if the design could have
detected an effect, so an oracle is handed the *true label* for every row the treatment can see —
cheating completely, unbeatable by any real model. Verified against all 24 real comparisons: **zero
violations**, and 300 deliberately label-peeking attacks reach the bound exactly without exceeding it.

That ceiling is **only evidence where sparsity actually binds it**. For the event block it holds the
oracle to **0.829**, well below a perfect ranking — so `expC_events` returning nothing is a genuine
measurement. For the dense experiments the oracle sorts the whole panel and reaches 1.000, making the
"+0.36 ceiling" arithmetic rather than proof; those verdicts rest on the interval alone.
`scripts/04_train_all.py` reports the two kinds separately instead of quoting the flattering number.

**The positive control.** An instrument that only ever reads zero is indistinguishable from a broken
one, so a feature of *known* strength is planted in the real panel and run through the same
walk-forward, the same folds and the same verdict rule:

| planted feature | resulting AUC | delta | 95% interval | verdict |
|---|---|---|---|---|
| label + 0.3σ noise | 0.9929 | +0.3570 | [+0.3052, +0.4248] | POSITIVE |
| label + 0.8σ noise | 0.8223 | +0.1864 | [+0.1616, +0.2384] | POSITIVE |
| label + 1.5σ noise | 0.7294 | +0.0936 | [+0.0609, +0.1849] | POSITIVE |
| label + 3.0σ noise | 0.6558 | **+0.0199** | [+0.0023, +0.0444] | **POSITIVE** |
| label + 8.0σ noise | 0.6401 | +0.0042 | [−0.0038, +0.0147] | ZERO |
| pure noise | 0.6338 | −0.0020 | [−0.0052, +0.0003] | ZERO |

The fourth row is the one that matters. **A true effect of +0.0199 is detected.** The pledge deltas
this project reports as ZERO are −0.014 to −0.036 — the same magnitude. The apparatus is demonstrably
sensitive enough to have found what pledge data would have needed to show, and pure noise still comes
back ZERO. `tests/test_power.py` runs this control end to end on every test run.

Three independent lines of evidence agree:

1. **The ablation.** Pledge levels + market features score 0.6243; adding the five trajectory
   features (change, acceleration, streak, rolling max) moves it to 0.6240. The *sequence* carries
   nothing beyond the *level* — which is the hypothesis §3.2 was built to test.
2. **Pledge features alone** reach 0.5506 — barely above the 0.50 coin flip.
3. **The SHAP confound audit.** Market features hold **73.9%** of total |SHAP|; the top three are
   `volatility_90d`, `trailing_dd_60d`, `return_90d`. The best pledge feature ranks 4th of 13.

### And it holds at event resolution too

The obvious objection to the above is that quarterly snapshots are simply too blunt. So the study was
extended to **SEBI Regulation 31 disclosures** — the filing a promoter must make within days of each
individual pledge action. Six features were built from that event stream and run through the same
walk-forward protocol.

Every experiment against its own baseline, median across the three models, with the interval and the
oracle ceiling:

| experiment | features | median delta | verdicts | oracle bound |
|---|---|---|---|---|
| `ablation_static` | 8 | −0.0140 | 3 × ZERO | 1.000 *(not binding)* |
| `expD_events_market` | 11 | −0.0156 | 3 × ZERO | 1.000 *(not binding)* |
| `expE_everything` | 19 | −0.0161 | 3 × ZERO | 1.000 *(not binding)* |
| `expB_full` | 13 | −0.0183 | 3 × ZERO | 1.000 *(not binding)* |
| `expG_pledged_full` | 13 | −0.0199 | 3 × ZERO | 1.000 *(not binding)* |
| `expH_pledged_events` | 19 | −0.0354 | 3 × ZERO | 1.000 *(not binding)* |
| `expA_pledge` | 8 | −0.0877 | 3 × **NEGATIVE** | 0.999 |
| `expC_events` | 6 | −0.1453 | 3 × **NEGATIVE** | **0.829** |

**18 ZERO, 6 NEGATIVE.** Two different results are visible here, and separating them is the whole
point of the intervals:

- **Adding pledge data to the market baseline changes nothing.** Every such comparison is ZERO. Not
  worse, not better — indistinguishable.
- **Pledge data used *instead of* market data is measurably worse.** `expA_pledge` at −0.124
  [−0.167, −0.075] and `expC_events` at −0.145 [−0.185, −0.088] both exclude zero comfortably — these
  are three to four interval-widths from the boundary, not marginal calls. Event data alone scores
  **0.478–0.492 — below a coin flip.** This is a real, directional finding: a pledge-only screen is
  worse than screening on volatility alone.

So the finding is not an artefact of quarterly frequency. At both resolutions Indian regulatory
filings offer, pledge behaviour adds nothing once volatility and size are accounted for.

> **The event features are not deployed, and that is the correct outcome.** The active model is
> `xgboost / expB_full`, which uses **13 features and zero event features**. The Reg 31 block was
> built, tested, and failed its test — so the served model ignores it. It is a negative control that
> closes an objection, not a component of the prediction path. Anyone reading this repository should
> not describe it as improving accuracy, because it does not.

### It is not an artefact of the window either

The 90-day event window was a judgement call, so `scripts/07_sensitivity.py` sweeps it. Coverage more
than doubles across the sweep and the signal never appears:

| window | coverage | count AUC | net AUC | created AUC |
|---|---|---|---|---|
| 30d | 5.1% | 0.4987 | 0.4898 | 0.4993 |
| **90d** *(configured)* | 8.3% | 0.4979 | 0.4868 | 0.4970 |
| 180d | 11.1% | 0.4968 | 0.4938 | 0.4980 |
| 365d | 14.9% | 0.4991 | 0.4943 | 0.5017 |
| 730d | 19.7% | 0.5071 | 0.4846 | 0.4985 |

Nothing in that table feeds back into the configuration. Sweeping a parameter and keeping whichever
value scored best is how a null gets tuned into a finding; the sweep exists to show the result is flat
and would be reported just as prominently if it were not.

### Tested again on the population where the question is meaningful

The panel is built on the NIFTY 500 spine deliberately, so the study is not conditioned on being
pledged — selecting the universe from the pledge list would have manufactured a positive result. The
cost is that **212 of the 300 companies never carry a promoter pledge at all**, so most rows have every
pledge feature pinned at zero.

`config.yaml` therefore defines a `pledged` stratum (`pledge_pct_promoter >= 1%`): 858 labelled rows,
72 companies, and a base crash rate of 23.1% against the full panel's 22.9% — a narrower population
answering the same question, not a different one. Event coverage inside it is **42.9% rather than
8.3%**, which makes it the fairest test the Reg 31 block ever gets.

The answer does not change: `expG_pledged_full` and `expH_pledged_events` both come back ZERO against
the stratum's own null. Config validation *refuses* to compare a stratified experiment against a
full-panel baseline, because that delta would measure the population change rather than the feature
set.

One thing does surface there, and it is reported because it points the wrong way: among pledged
companies, `event_net_90d` scores **0.438** univariately — equivalent to 0.562 with the sign flipped.
**Promoters increasing their pledge is associated with *fewer* crashes**, matching the raw crosstab
(15.3% against a 22.9% base). That contradicts the folk wisdom this project set out to test. It is a
univariate signal with no out-of-sample discipline and no interval, and the models that could use it
still come back ZERO, so it is offered as a direction worth investigating and not as a result.

### Why this is the finding and not a failure

Measured on this panel, **the quarterly pledge percentage does not change in 90.5% of
company-quarters.** JPPOWER sits at exactly 79.20% for fourteen consecutive quarters. Four of the
eight pledge features are therefore zero for nine rows in ten.

That is a fact about India's quarterly disclosure regime, not about the model. Promoters pledge in a
block and leave it; anything happening between filings is invisible at this frequency. A system built
to *display* that number — as commercial screeners do — cannot be early warning, and this project is
the measurement that shows why.

A null result was a designed-for outcome. The experiment ladder exists precisely so the answer can
come back "no" and still be worth reporting.

### What a bounded null is worth, and what it is not

The claim this project can defend is narrow and it is stated narrowly:

> On 300 NIFTY 500 companies over 19 quarters, promoter-pledge data — quarterly *and* at Reg 31 event
> resolution, on the full panel *and* restricted to pledged companies — adds no measurable warning
> about 60-day drawdowns beyond what 90-day volatility and turnover already carry. Any true effect is
> smaller than roughly 0.03–0.06 AUC, on intervals measured to hold 93–95% coverage at this sample
> size, and the one comparison where sparsity could have hidden an effect had headroom to 0.83 AUC.

It is **not** a claim that promoter pledging is harmless, that pledge data is useless for any purpose,
or that the result generalises past this universe and window. Twenty quarters is a short study, the
event features are non-zero on only 8.3% of full-panel rows, and the invocation feature rests on 49.
Those limits are listed in full under [Limitations](#limitations--stated-openly).

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
| **Reg 31 events** | 35,584 disclosures; **10,531 material** after filtering clearing noise |
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
      features/        8 pledge trajectory + 5 market + 6 Reg 31 event features
      labels/          forward 60-day max drawdown from entry, threshold −15%
      data/population  panel strata — the rows where a question is answerable
            │
      training/        walk-forward, expanding window, 11 folds, embargo
      evaluation/      within-quarter AUC · quintile backtest · ★ leakage proofs
                       ★ power.py — bootstrap intervals + oracle ceilings
                         sensitivity.py — window and materiality sweeps
      explain/         SHAP: global beeswarm, local waterfall, templated text
            │
   ★  inference/service.py   the ONLY scoring path
            │
      ┌─────┴─────┐
   api/         dashboard/     FastAPI (5 endpoints) · Streamlit (3 pages)
```

Two boundaries carry the design:

- **Nothing downstream of panel assembly can see data filed after the observation date.** One file
  enforces it, and `evaluation/leakage.py` proves it rather than claiming it. The Reg 31 block adds a
  second, stricter boundary: `event_date` records when a pledge was *created*, not when it became
  public, and SEBI allows 7 working days to disclose it — so every event window closes 11 calendar
  days early, and a dedicated check recomputes the feature both ways to prove the buffer was applied.
- **The API and the dashboard import the same service.** One scoring path, no drift — and the
  dashboard does not require the API to be running.
- **No difference is reported without an interval.** `evaluation/power.py` attaches a block-bootstrap
  interval and an oracle ceiling to every experiment comparison, and the Model Validation page
  re-derives them from the stored out-of-fold predictions rather than trusting a cached number. A
  delta whose interval straddles zero is labelled ZERO in the console, in the README and in the UI.

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
make build        # point-in-time panel + leakage tests       GATE 2
make train        # walk-forward, 3 models x 10 experiments,  GATE 3
                  #   with intervals and oracle ceilings
make evaluate     # quintile backtest + SHAP + figures
make score        # score the latest quarter
make sensitivity  # window + materiality sweeps, univariate table

make train-clean  # train, then drop superseded sessions + artifacts and VACUUM
make api          # http://127.0.0.1:8000/docs
make app          # http://localhost:8501
make test         # 205 tests
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
4. **Event coverage is thin.** The Reg 31 features are non-zero for only 8.3% of full-panel rows
   (42.9% inside the pledged stratum), because material pledge actions are genuinely rare. The
   invocation feature rests on 49 rows. This is quantified rather than asserted: the oracle ceiling
   for the event block is **+0.19**, so the design could have found a large effect — but a null on
   thin coverage remains weaker evidence than the quarterly one.
5. **Eleven test dates set the resolution of every conclusion.** The studentised half-width runs
   0.03–0.09 AUC depending on the pair, so **no effect smaller than that is detectable here by any
   method**. Differences below that in any table in this repository should be read as zero, and more
   modelling would not change it — only more data would. Simulated power at n=11: a true +0.03 effect
   is detected 48% of the time, +0.05 88%, +0.08 99.8%.
6. **The intervals themselves are approximations.** Measured coverage of the nominal 95% interval is
   93.0–95.1% depending on the shape of the per-date differences — good, but not exact, and it
   degrades below n≈8 dates. Any future experiment that scores fewer dates than this one should not
   assume these verdicts transfer.
7. **Five years is less than a full market cycle.**
8. **Not investment advice.** This is a research artefact.

---

## Testing

205 tests. The priority order is deliberate — `make test-critical` runs the 38 that matter most.

| File | Covers | Priority |
|---|---|---|
| `test_leakage.py` | point-in-time rule · label window · fold disjointness · label-shuffle | ★★★ |
| `test_xbrl_parser.py` | pledge present · explicit zero · unavailable ≠ zero · malformed → quarantine · **scale correctness** | ★★★ |
| `test_labels.py` | known series → known drawdown · exact −15% boundary · insufficient future → null not 0 | ★★★ |
| `test_power.py` | **measured interval coverage (93–95% at n=11)** · **end-to-end positive control** · paired bootstrap is tighter than unpaired · oracle ceiling is 0 at 0 coverage · mask joins on key not position | ★★★ |
| `test_features.py` | QoQ change · acceleration needs 3 quarters · rise counter · forward-fill cap | ★★ |
| `test_repository.py` | upsert idempotency · surrogate-id preservation · empty → empty frame | ★★ |
| `test_api.py` | health · predict · 404 · 422 · persistence | ★★ |
| `test_event_features.py` | materiality filter · **disclosure buffer** · window arithmetic | ★★ |
| `test_population.py` | NaN is not "pledged" · empty stratum raises · **cross-population comparison rejected at config load** | ★★ |
| `test_retention.py` | **the active session is never deleted** · predictions go before runs (no cascade) · `keep_sessions=0` disables rather than wipes · VACUUM really shrinks | ★★ |
| `test_registry_prune.py` | the active artifact survives · refusal deletes nothing · the survivor still loads and predicts | ★★ |
| `test_sensitivity.py` | coverage rises with the window · inverted features scored by strength · **the sweep mutates no configuration** | ★★ |

Leakage tests come in **positive/negative pairs**: each plants the exact violation it claims to catch,
because a test that only ever sees correct input proves nothing about what it would catch. Parser
tests run against **committed real XBRL files**, not mocks.

**The test suite validates the validator.** `test_power.py` measures the actual coverage of the
nominal 95% interval by simulation, and runs a positive control end to end — a planted effect must be
found, planted noise must not be. Both exist because a confidence interval is the one kind of code
that always looks like it works: it returns two plausible numbers whether or not it is calibrated. The
coverage test is what caught the percentile bootstrap undercovering, and the correction flipped a
published verdict from NEGATIVE to ZERO.

Three of the newer tests exist because the code was wrong in ways that still looked right:

- `test_a_row_with_an_unknown_pledge_is_excluded_rather_than_assumed` — a company whose pledge is
  unknown must not enter a stratum *defined* by pledging. NaN comparisons are silently False in
  pandas, so this behaviour was correct by accident and is now correct on purpose.
- `test_assess_joins_the_mask_on_symbol_and_date_not_by_position` — out-of-fold rows are assembled
  fold by fold and are **not** in panel order. A positional join would attach the wrong company's
  event coverage to every row and still produce a plausible-looking ceiling.
- `test_the_95_percent_interval_really_covers_about_95_percent` — the original percentile bootstrap
  covered 86.9–90.8%, so a nominal 95% interval was really a 90% one. Nothing in the results table
  looked wrong, and one verdict was a false positive because of it.

---

## Novelty claim, phrased defensibly

> I did not find an existing public project that combines promoter-pledge **trajectory** features with
> walk-forward downside-risk validation and an economic backtest for Indian equities.

Commercial screeners *display* the current pledge number. None of them test whether it predicts
anything. **PledgeCast is the evaluation layer** — and its finding is that, at both frequencies Indian
regulatory filings offer and on both the full index and the pledged subset, it does not carry
measurable incremental warning.

The part worth defending is not the null itself but that it is **bounded**. A study that reports "we
found nothing" is unfalsifiable; this one reports what it could have found (+0.19 to +0.36), what it
did find (0.00 ± 0.03), that the answer survives sweeping the window from 30 to 730 days, and that it
survives restricting to the population where the question is meaningful. Those four together are what
make a negative result usable by someone else.

---

*Endpoint behaviour, archive depth and event rates were measured against live NSE and Yahoo APIs in
August 2026. NSE occasionally changes endpoint contracts without notice; re-verify before a fresh
run.*
