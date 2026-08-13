"""Model registry - PLAN.md sec.8.1, sec.9.7.

    "models/registry.py | `is_active` model resolution | Model versioning
     without MLflow"

    sec.9.7 step 4: "Save to models/<run_id>.joblib, insert model_runs row,
     set is_active = 1"

The split of responsibility is deliberate: the *artifact* lives on disk, the
*metadata* lives in ``model_runs``, and this module is the only place that knows
they belong together. Nothing else in the codebase opens a .joblib file.

**Why the payload repeats the feature list.** The artifact stores its own
``feature_list``, and :func:`load_active_model` refuses to return a model whose
stored list disagrees with the database row. A pickled sklearn pipeline has no
idea what its columns mean - hand it 13 numbers in the wrong order and it
returns a confident, wrong probability. The check turns a silent mis-scoring
into a startup error, which is what sec.10 asks for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import sklearn
import xgboost
from sqlalchemy.engine import Connection

from pledgecast.db import repository as repo
from pledgecast.exceptions import ModelNotFoundError, ValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

# Bumped only when the payload layout changes in a way older files cannot satisfy.
PAYLOAD_VERSION = 1


def artifact_path(run_id: str, settings) -> Path:
    """``models/<run_id>.joblib`` (sec.9.7)."""
    return Path(settings.paths.models_dir) / f"{run_id}.joblib"


def save_model(
    pipeline,
    *,
    run_id: str,
    model_name: str,
    experiment: str,
    feature_list: list[str],
    hyperparams: dict[str, Any],
    settings,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted pipeline and return its path.

    The pipeline is stored whole - winsoriser, imputer, scaler and estimator
    together - so the serving path cannot reconstruct preprocessing differently
    from the training path. That is the entire point of using a Pipeline
    (sec.9.4 "fold hygiene").
    """
    path = artifact_path(run_id, settings)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "payload_version": PAYLOAD_VERSION,
        "run_id": run_id,
        "model_name": model_name,
        "experiment": experiment,
        "feature_list": list(feature_list),
        "hyperparams": hyperparams,
        "created_at": repo.utc_now(),
        "versions": {
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "pipeline": pipeline,
        **(extra or {}),
    }
    joblib.dump(payload, path, compress=3)
    logger.info("saved model artifact %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_model(run_id: str, settings) -> dict[str, Any]:
    """Load one artifact by run id. Raises if it is missing or stale."""
    path = artifact_path(run_id, settings)
    if not path.exists():
        raise ModelNotFoundError(
            f"model artifact {path} not found. The model_runs row exists but the file does "
            "not - re-run `make train`."
        )

    payload = joblib.load(path)
    if payload.get("payload_version") != PAYLOAD_VERSION:
        raise ModelNotFoundError(
            f"{path} was written by payload version {payload.get('payload_version')}, "
            f"this build expects {PAYLOAD_VERSION}. Re-train rather than guessing."
        )

    if payload.get("versions", {}).get("sklearn") != sklearn.__version__:
        logger.warning(
            "artifact %s was trained on scikit-learn %s but this process runs %s; "
            "unpickling across versions is not guaranteed",
            path.name,
            payload.get("versions", {}).get("sklearn"),
            sklearn.__version__,
        )
    return payload


def load_active_model(conn: Connection, settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """The one model flagged ``is_active`` plus its ``model_runs`` row.

    Returns ``(payload, run)``. Raises :class:`ModelNotFoundError` when no run
    is active - sec.10 and sec.13.2 turn that into an HTTP 503 rather than a
    crash.
    """
    run = repo.require_active_run(conn)
    payload = load_model(run["run_id"], settings)

    if list(payload["feature_list"]) != list(run["feature_list"]):
        raise ValidationError(
            f"run {run['run_id']}: artifact feature list does not match the database row.\n"
            f"  artifact: {payload['feature_list']}\n"
            f"  database: {run['feature_list']}\n"
            "Scoring with mismatched columns produces confident nonsense - refusing."
        )
    return payload, run


def set_active(conn: Connection, run_id: str, settings) -> None:
    """Promote a run to serving (sec.9.7 step 4).

    Verifies the artifact is loadable BEFORE flipping the flag. Activating a run
    whose file is missing would take the API down at the next restart with no
    obvious cause.
    """
    load_model(run_id, settings)
    repo.set_active_run(conn, run_id)


def register(
    conn: Connection,
    pipeline,
    *,
    run_id: str,
    model_name: str,
    experiment: str,
    feature_list: list[str],
    hyperparams: dict[str, Any],
    settings,
    n_train_rows: int | None = None,
    n_folds: int | None = None,
    activate: bool = False,
) -> str:
    """Save artifact + insert the ``model_runs`` row in one call.

    ``config_snapshot`` records the entire configuration at training time
    (sec.10), so a run stays reproducible even after config.yaml moves on.
    """
    path = save_model(
        pipeline,
        run_id=run_id,
        model_name=model_name,
        experiment=experiment,
        feature_list=feature_list,
        hyperparams=hyperparams,
        settings=settings,
    )
    repo.insert_model_run(
        conn,
        run_id=run_id,
        model_name=model_name,
        experiment=experiment,
        feature_list=feature_list,
        hyperparams=hyperparams,
        random_seed=settings.training.random_seed,
        n_train_rows=n_train_rows,
        n_folds=n_folds,
        artifact_path=str(path),
        config_snapshot=settings.snapshot(),
    )
    if activate:
        set_active(conn, run_id, settings)
    return run_id


def describe(conn: Connection) -> str:
    """One-line description of the serving model - used by the API banner."""
    run = repo.get_active_run(conn)
    if run is None:
        return "no active model"
    return (
        f"{run['run_id']} ({run['model_name']} / {run['experiment']}, "
        f"{len(run['feature_list'])} features, seed {run['random_seed']})"
    )


def hyperparams_json(run: dict) -> str:
    """Pretty JSON of a run's hyperparameters - the Validation page shows this."""
    return json.dumps(run.get("hyperparams", {}), indent=2, sort_keys=True, default=str)


__all__ = [
    "PAYLOAD_VERSION",
    "artifact_path",
    "describe",
    "hyperparams_json",
    "load_active_model",
    "load_model",
    "register",
    "save_model",
    "set_active",
]
