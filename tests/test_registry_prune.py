"""Artifact pruning - the only code in this project that deletes user files.

``make train`` writes one ``.joblib`` per run and removes none, so ``models/``
grows forever. Six runs during one afternoon of debugging left seven artifacts
of which exactly one was reachable.

Deletion deserves more care than the thing it cleans up, so the properties
tested here are all about what must SURVIVE: the active artifact, anything
outside the models directory, and every file when the database has no active
run at all.
"""

from __future__ import annotations

import pytest

from pledgecast.db import repository as repo
from pledgecast.exceptions import ModelNotFoundError
from pledgecast.models import registry


@pytest.fixture
def models_dir(tmp_path, settings):
    """A settings object pointing at a throwaway models directory."""
    directory = tmp_path / "models"
    directory.mkdir()
    paths = settings.paths.model_copy(update={"models_dir": directory})
    return directory, settings.model_copy(update={"paths": paths})


def _register(conn, settings, run_id: str, *, activate: bool) -> None:
    """Write a run row plus a stand-in artifact file."""
    repo.insert_model_run(
        conn,
        run_id=run_id,
        model_name="logreg",
        experiment="expB_full",
        feature_list=["volatility_90d"],
        hyperparams={},
        random_seed=42,
        n_train_rows=100,
        n_folds=11,
        artifact_path=str(registry.artifact_path(run_id, settings)),
        config_snapshot="{}",
    )
    registry.artifact_path(run_id, settings).write_bytes(b"x" * 1024)
    if activate:
        repo.set_active_run(conn, run_id)


def test_the_active_artifact_survives_and_the_rest_do_not(seeded_conn, models_dir):
    directory, settings = models_dir
    for run_id in ("20260101T0000_a", "20260101T0100_b"):
        _register(seeded_conn, settings, run_id, activate=False)
    _register(seeded_conn, settings, "20260101T0200_active", activate=True)

    result = registry.prune_artifacts(seeded_conn, settings)

    survivors = sorted(p.name for p in directory.glob("*.joblib"))
    assert survivors == ["20260101T0200_active.joblib"]
    assert len(result["removed"]) == 2
    assert result["kept"] == "20260101T0200_active.joblib"
    assert result["freed_bytes"] == 2048


def test_nothing_is_deleted_when_no_run_is_active(seeded_conn, models_dir):
    """'Nothing is active' almost always means the database is in an odd state,
    not that every artifact is garbage. Refusing is the safe reading."""
    directory, settings = models_dir
    _register(seeded_conn, settings, "20260101T0000_a", activate=False)

    with pytest.raises(ModelNotFoundError, match="refusing to prune"):
        registry.prune_artifacts(seeded_conn, settings)

    assert len(list(directory.glob("*.joblib"))) == 1, "a refusal still deleted something"


def test_dry_run_reports_without_deleting(seeded_conn, models_dir):
    directory, settings = models_dir
    _register(seeded_conn, settings, "20260101T0000_a", activate=False)
    _register(seeded_conn, settings, "20260101T0200_active", activate=True)

    result = registry.prune_artifacts(seeded_conn, settings, dry_run=True)

    assert result["removed"] == ["20260101T0000_a.joblib"]
    assert result["dry_run"] is True
    assert len(list(directory.glob("*.joblib"))) == 2, "dry run deleted a file"


def test_pruning_twice_is_a_no_op_rather_than_an_error(seeded_conn, models_dir):
    directory, settings = models_dir
    _register(seeded_conn, settings, "20260101T0000_a", activate=False)
    _register(seeded_conn, settings, "20260101T0200_active", activate=True)

    registry.prune_artifacts(seeded_conn, settings)
    second = registry.prune_artifacts(seeded_conn, settings)

    assert second["removed"] == []
    assert len(list(directory.glob("*.joblib"))) == 1


def test_only_joblib_files_are_touched(seeded_conn, models_dir):
    """A stray README or .gitkeep in models/ is not a model artifact."""
    directory, settings = models_dir
    _register(seeded_conn, settings, "20260101T0200_active", activate=True)
    (directory / ".gitkeep").write_text("keep", encoding="utf-8")
    (directory / "notes.txt").write_text("hello", encoding="utf-8")

    registry.prune_artifacts(seeded_conn, settings)

    assert (directory / ".gitkeep").exists()
    assert (directory / "notes.txt").exists()


def test_the_surviving_artifact_still_loads(seeded_conn, models_dir, sample_panel):
    """Pruning must leave a WORKING model, not just a file with the right name."""
    from sklearn.linear_model import LogisticRegression

    directory, settings = models_dir
    _register(seeded_conn, settings, "20260101T0000_stale", activate=False)

    run_id = "20260101T0200_active"
    estimator = LogisticRegression().fit([[0.1], [0.9]], [0, 1])
    registry.save_model(
        estimator,
        run_id=run_id,
        model_name="logreg",
        experiment="expB_full",
        feature_list=["volatility_90d"],
        hyperparams={},
        settings=settings,
    )
    repo.insert_model_run(
        seeded_conn,
        run_id=run_id,
        model_name="logreg",
        experiment="expB_full",
        feature_list=["volatility_90d"],
        hyperparams={},
        random_seed=42,
        n_train_rows=100,
        n_folds=11,
        artifact_path=str(registry.artifact_path(run_id, settings)),
        config_snapshot="{}",
    )
    repo.set_active_run(seeded_conn, run_id)

    registry.prune_artifacts(seeded_conn, settings)

    payload = registry.load_model(run_id, settings)
    assert payload["feature_list"] == ["volatility_90d"]
    assert payload["pipeline"].predict([[0.5]]) is not None
