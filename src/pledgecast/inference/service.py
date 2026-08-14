"""The single scoring path - PLAN.md sec.7.2.

    "Both FastAPI and Streamlit import the same src/inference/service.py.
     One scoring path, tested once. The dashboard does NOT depend on the API
     being up - the demo cannot break."

sec.7.1 draws the boundary precisely: this layer owns **score + explain +
persist**, and never fits or trains. Nothing here touches a training routine,
and every model it uses arrives already fitted from the registry.

**Why the decile needs the whole cohort.** A probability on its own says very
little - 0.31 is a high score in a calm quarter and a low one in a falling
market, which is the same market-timing confound that makes within-quarter AUC
the primary metric (sec.9.6). So a single-symbol request scores every company
sharing that observation date and ranks within it. That is 300 rows through a
fitted pipeline, which costs about a millisecond, and it makes the returned
decile mean the same thing as the decile stored during the backtest.

**Warnings are part of the contract** (sec.13.1). Three conditions are reported
rather than hidden: forward-filled pledge state, pledge state that was never
determinable, and features blanked because a corporate action fell inside their
lookback window (sec.10). The last one matters more than it looks - a row whose
market features are missing is scored largely by the imputer's missingness
indicators, so the number is real but weakly founded, and the caller is told.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.engine import Connection

from pledgecast.db import repository as repo
from pledgecast.evaluation import backtest
from pledgecast.exceptions import InsufficientDataError, ValidationError
from pledgecast.explain import shap_runner
from pledgecast.logging_config import get_logger
from pledgecast.models import registry
from pledgecast.models.preprocessing import prepare_matrix

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# warning codes (sec.13.1)                                                     #
# --------------------------------------------------------------------------- #
# The wire format stays ``list[str]`` - sec.13.1 specifies a ``warnings`` array
# and ``api/schemas.py`` publishes it - so codes travel BESIDE the messages
# rather than replacing them.
#
# They exist because two of these conditions describe the DATA and one describes
# the CALENDAR, and consumers need to tell them apart. Callers used to do that
# by position, assuming the calendar warning was always last; it is appended only
# when ``label_is_valid == 0``, so on every labelled date the assumption silently
# selected the wrong warnings. Selecting by code cannot drift when the order or
# the wording changes.
STALE_PLEDGE = "stale_pledge_state"
IMPUTED_FEATURES = "imputed_features"
NO_REALISED_OUTCOME = "no_realised_outcome"
SUPPLIED_FEATURES = "supplied_features"

#: Codes that say something about data quality, as opposed to the embargo
#: quarter's structural absence of a label.
DATA_QUALITY_CODES = frozenset({STALE_PLEDGE, IMPUTED_FEATURES})


class PredictionService:
    """Scores companies with the active model. Construct once, reuse.

    The model is loaded lazily on first use and then held, so FastAPI's lifespan
    and Streamlit's cache both get a warm object without either of them knowing
    how loading works.
    """

    def __init__(self, settings, conn: Connection | None = None) -> None:
        self.settings = settings
        self._payload: dict[str, Any] | None = None
        self._run: dict[str, Any] | None = None
        self._cohorts: dict[tuple[str, str], pd.DataFrame] = {}
        self._bg: np.ndarray | None = None
        if conn is not None:
            self.load(conn)

    # ------------------------------------------------------------------ model
    def load(self, conn: Connection) -> None:
        """Load (or reload) the active model. Raises ``ModelNotFoundError`` if none."""
        self._payload, self._run = registry.load_active_model(conn, self.settings)
        self._cohorts.clear()
        self._bg = None
        logger.info("scoring with %s", self._run["run_id"])

    def _ensure(self, conn: Connection) -> tuple[dict, dict]:
        if self._payload is None or self._run is None:
            self.load(conn)
        return self._payload, self._run  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        return self._payload is not None

    def model_info(self, conn: Connection) -> dict[str, Any]:
        """Backs ``GET /model-info`` (sec.13) - full transparency about what is serving."""
        _, run = self._ensure(conn)
        metrics = repo.load_metrics(conn, run_id=run["run_id"], fold=-1)
        return {
            "run_id": run["run_id"],
            "model_name": run["model_name"],
            "experiment": run["experiment"],
            "features": run["feature_list"],
            "hyperparams": run["hyperparams"],
            "random_seed": run["random_seed"],
            "n_train_rows": run["n_train_rows"],
            "n_folds": run["n_folds"],
            "trained_at": run["created_at"],
            "metrics": dict(zip(metrics["metric_name"], metrics["metric_value"], strict=True)),
        }

    def health(self, conn: Connection) -> dict[str, Any]:
        """Backs ``GET /health`` - "proves the system knows its own state" (sec.13)."""
        status: dict[str, Any] = {"database": False, "model": False, "status": "unhealthy"}
        try:
            counts = repo.table_counts(conn)
            status["database"] = True
            status["panel_rows"] = counts.get("panel", 0)
            status["predictions"] = counts.get("predictions", 0)
        except Exception as exc:  # noqa: BLE001 - health must report, never raise
            status["error"] = str(exc)
            return status

        try:
            _, run = self._ensure(conn)
            status["model"] = True
            status["run_id"] = run["run_id"]
            status["status"] = "healthy"
        except Exception as exc:  # noqa: BLE001
            status["error"] = str(exc)
        return status

    # -------------------------------------------------------------- cohort
    def _cohort(self, conn: Connection, observation_date: str) -> pd.DataFrame:
        """Every company on one observation date, scored. Cached per (run, date).

        Safe to cache indefinitely: a past observation date's features never
        change, and :meth:`load` clears the cache when the model does.
        """
        payload, run = self._ensure(conn)
        key = (run["run_id"], observation_date)
        if key in self._cohorts:
            return self._cohorts[key]

        panel = repo.load_panel(conn, valid_only=False)
        block = panel[panel["observation_date"] == observation_date].reset_index(drop=True)
        if block.empty:
            raise InsufficientDataError(f"no panel rows for observation date {observation_date}")

        features = payload["feature_list"]
        block["probability"] = payload["pipeline"].predict_proba(
            prepare_matrix(block, features)
        )[:, 1]
        block["risk_decile"] = backtest.rank_groups(
            block["probability"], self.settings.evaluation.n_deciles
        )
        self._cohorts[key] = block
        return block

    # -------------------------------------------------------------- warnings
    def _warning_items(self, row: pd.Series, features: list[str]) -> list[tuple[str, str]]:
        """``(code, message)`` for every condition this row triggers.

        The single source of truth for sec.13.1's warnings. Both the string list
        the API publishes and the data-quality subset the dashboard renders are
        derived from here, so the two can never disagree about what a warning is
        or how many there are.
        """
        items: list[tuple[str, str]] = []
        if row.get("is_stale") == 1:
            items.append((
                STALE_PLEDGE,
                f"pledge_state forward-filled from an earlier quarter "
                f"(max {self.settings.features.max_forward_fill_quarters})",
            ))

        blank = [f for f in features if pd.isna(row.get(f))]
        if blank:
            items.append((
                IMPUTED_FEATURES,
                f"{len(blank)} feature(s) unavailable and median-imputed: {', '.join(blank)}. "
                "A corporate action inside the lookback window blanks market features "
                "(sec.10), and the score then leans on missingness rather than on data.",
            ))
        if row.get("label_is_valid") == 0:
            items.append((
                NO_REALISED_OUTCOME,
                "this observation date has no realised outcome yet - it is a forward "
                "prediction, not a scored backtest row",
            ))
        return items

    def _warnings(self, row: pd.Series, features: list[str]) -> list[str]:
        """sec.13.1's ``warnings`` array. Everything the caller should discount."""
        return [message for _, message in self._warning_items(row, features)]

    def _data_warnings(self, row: pd.Series, features: list[str]) -> list[str]:
        """Only the conditions that describe the DATA behind the score.

        Excludes the embargo quarter's missing label, which is reported once per
        page rather than 300 times, and which is not a defect in the row.
        """
        return [
            message
            for code, message in self._warning_items(row, features)
            if code in DATA_QUALITY_CODES
        ]

    # ---------------------------------------------------------------- scoring
    def score(
        self,
        conn: Connection,
        symbol: str,
        *,
        observation_date: str | None = None,
        include_explanation: bool = False,
        persist: bool = True,
        source: str = "api",
    ) -> dict[str, Any]:
        """Score one company. The sec.13.1 response contract, as a dict."""
        payload, run = self._ensure(conn)

        row = repo.get_panel_row(conn, symbol, observation_date)
        if row is None:
            raise InsufficientDataError(
                f"no panel row for {symbol!r}"
                + (f" on {observation_date}" if observation_date else "")
            )

        date = row["observation_date"]
        cohort = self._cohort(conn, date)
        scored = cohort[cohort["symbol"] == symbol]
        if scored.empty:
            raise InsufficientDataError(f"{symbol} is not in the cohort for {date}")
        scored = scored.iloc[0]

        probability = float(scored["probability"])
        decile = int(scored["risk_decile"])
        features = payload["feature_list"]

        result: dict[str, Any] = {
            "symbol": symbol,
            "observation_date": date,
            "probability": probability,
            "risk_decile": decile,
            "risk_band": self.settings.evaluation.band_for(probability),
            "model": {"run_id": run["run_id"], "model_name": run["model_name"]},
            "explanation": None,
            "warnings": self._warnings(scored, features),
        }

        prediction_id = None
        if persist:
            prediction_id = repo.save_prediction(
                conn,
                run_id=run["run_id"],
                symbol=symbol,
                observation_date=date,
                probability=probability,
                risk_decile=decile,
                source=source,
            )
            result["prediction_id"] = prediction_id

        if include_explanation:
            records = self.explain_row(conn, scored, features)
            result["explanation"] = {
                "top_features": [
                    {
                        "feature": r["feature_name"],
                        "value": r["feature_value"],
                        "shap": r["shap_value"],
                        "direction": "increases_risk" if r["shap_value"] > 0 else "reduces_risk",
                    }
                    for r in shap_runner.merge_indicators(records)[
                        : self.settings.explain.top_n_features
                    ]
                ],
                "summary": shap_runner.summarise(
                    records,
                    probability,
                    decile=decile,
                    band=result["risk_band"],
                    top_n=self.settings.explain.top_n_features,
                ),
            }
            if persist and prediction_id is not None:
                repo.save_explanations(conn, prediction_id, records)

        return result

    def _background(self, conn: Connection) -> np.ndarray:
        """The reference population SHAP measures deviation from.

        All labelled panel rows - the same data the served model was refit on
        (sec.9.7 step 3), so the explanation's baseline is the study population
        rather than an arbitrary sample. Cached; cleared when the model reloads.
        """
        payload, _ = self._ensure(conn)
        if self._bg is None:
            panel = repo.load_panel(conn, valid_only=True)
            self._bg = prepare_matrix(panel, payload["feature_list"])
        return self._bg

    def explain_detail(self, conn: Connection, row: pd.Series, features: list[str]) -> dict:
        """The full SHAP computation for one row - explainer, values, matrix, names.

        The dashboard's waterfall needs the explainer object and the transformed
        matrix, not just the ranked records. Exposing it here keeps sec.7.1's
        rule intact: the page renders, the service computes.
        """
        payload = self._payload
        if payload is None:
            raise ValidationError("no model loaded")
        matrix = np.asarray([[row.get(f) for f in features]], dtype=float)
        return shap_runner.explain(
            payload["pipeline"], matrix, features, background=self._background(conn)
        )

    def explain_row(self, conn: Connection, row: pd.Series, features: list[str]) -> list[dict]:
        """SHAP records for one already-scored row, largest |contribution| first.

        The background is passed explicitly. Without it a single-row
        LinearExplainer explains the row against itself and every SHAP value
        comes back as 0.0 - a response that looks perfectly well formed and
        says nothing.
        """
        computed = self.explain_detail(conn, row, features)
        return shap_runner.explanation_rows(
            computed["values"], computed["raw"], computed["names"], features
        )[0]

    def score_features(
        self, conn: Connection, features: dict[str, float]
    ) -> dict[str, Any]:
        """Score a raw feature dict (sec.13 "or raw feature dict").

        No decile: a probability only becomes a rank against a cohort, and an
        ad-hoc feature vector belongs to no observation date. Returning a
        fabricated one would be worse than returning none.
        """
        payload, run = self._ensure(conn)
        expected = payload["feature_list"]
        missing = [f for f in expected if f not in features]
        if missing:
            raise ValidationError(f"missing required feature(s): {missing}")

        matrix = np.asarray([[float(features[f]) for f in expected]], dtype=float)
        probability = float(payload["pipeline"].predict_proba(matrix)[0, 1])
        return {
            "symbol": None,
            "observation_date": None,
            "probability": probability,
            "risk_decile": None,
            "risk_band": self.settings.evaluation.band_for(probability),
            "model": {"run_id": run["run_id"], "model_name": run["model_name"]},
            "explanation": None,
            "warnings": ["scored from supplied features - no cohort, so no decile"],
        }

    def score_date(
        self,
        conn: Connection,
        observation_date: str | None = None,
        *,
        persist: bool = True,
        source: str = "backtest",
    ) -> pd.DataFrame:
        """Score every company on one date. Backs ``06_score_latest.py`` and the scanner."""
        payload, run = self._ensure(conn)
        if observation_date is None:
            dates = repo.get_observation_dates(conn, valid_only=False)
            if not dates:
                raise InsufficientDataError("the panel holds no observation dates")
            observation_date = dates[-1]

        cohort = self._cohort(conn, observation_date).copy()
        cohort["risk_band"] = cohort["probability"].map(self.settings.evaluation.band_for)
        cohort["run_id"] = run["run_id"]
        # Two columns, one computation. `warnings` is everything (what the API
        # publishes); `data_warnings` is the data-quality subset the dashboard
        # renders. Consumers pick a column instead of slicing a list, which is
        # what made the old page-level filter wrong on every labelled date.
        items = [
            self._warning_items(row, payload["feature_list"]) for _, row in cohort.iterrows()
        ]
        cohort["warnings"] = [[message for _, message in row] for row in items]
        cohort["data_warnings"] = [
            [message for code, message in row if code in DATA_QUALITY_CODES] for row in items
        ]

        if persist:
            repo.delete_predictions_for_run_date(conn, run["run_id"], observation_date, source)
            repo.save_predictions_bulk(
                conn,
                (
                    {
                        "run_id": run["run_id"],
                        "symbol": row.symbol,
                        "observation_date": observation_date,
                        "probability": float(row.probability),
                        "risk_decile": int(row.risk_decile),
                        "source": source,
                    }
                    for row in cohort.itertuples(index=False)
                ),
            )
        return cohort.sort_values("probability", ascending=False).reset_index(drop=True)

    # ---------------------------------------------------------------- history
    def company_history(self, conn: Connection, symbol: str) -> dict[str, Any]:
        """Backs ``GET /companies/{symbol}/history`` and the investigation page."""
        company = repo.get_company(conn, symbol)
        if company is None:
            raise InsufficientDataError(f"unknown symbol {symbol!r}")

        state = repo.load_pledge_state(conn, symbol=symbol)
        events = repo.load_pledge_events(conn, symbol=symbol)
        predictions = repo.load_predictions(conn, symbol=symbol)
        panel = repo.load_panel(conn, valid_only=False)
        panel = panel[panel["symbol"] == symbol]

        return {
            "symbol": symbol,
            "company_name": company.get("company_name"),
            "industry": company.get("industry"),
            "pledge_history": state.to_dict("records"),
            "events": events.to_dict("records"),
            "predictions": predictions.to_dict("records"),
            "panel": panel.to_dict("records"),
        }


__all__ = ["PredictionService"]
