"""Training-session retention - the code that deletes database rows.

Every ``04_train_all.py`` run writes ~28 model runs and ~73,000 out-of-fold
prediction rows and removes none, so the database keeps one complete copy of the
study per invocation. Six runs in one afternoon reached 435,351 prediction rows.

Deleting rows deserves more care than the growth it fixes, so the properties
tested hardest are about what must SURVIVE:

**The active session is never dropped**, even when it is old enough to fall
outside the retention window. Dropping it would leave the API booting with an
``is_active`` row pointing at nothing while every table still looked populated.

**Deletion order is load-bearing.** ``predictions.run_id`` is the one child
foreign key without ``ON DELETE CASCADE``, and the connection runs with
``PRAGMA foreign_keys = ON``. Deleting ``model_runs`` first raises. There is a
test that fails loudly if someone reorders it.

**``keep_sessions = 0`` disables pruning** rather than deleting everything.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pledgecast.db import repository as repo
from pledgecast.models import registry


def _session(conn, stamp: str, *, models=("logreg", "xgboost"), activate: str | None = None):
    """Write a training session: runs, metrics, predictions, explanations."""
    for model in models:
        run_id = f"{stamp}_{model}_expB_full"
        repo.insert_model_run(
            conn,
            run_id=run_id,
            model_name=model,
            experiment="expB_full",
            feature_list=["volatility_90d"],
            hyperparams={},
            random_seed=42,
            n_train_rows=100,
            n_folds=11,
            artifact_path=None,
            config_snapshot="{}",
        )
        repo.insert_metrics(
            conn, run_id, [{"fold": -1, "metric_name": "within_quarter_auc", "metric_value": 0.62}]
        )
        repo.save_predictions_bulk(
            conn,
            [
                {
                    "run_id": run_id,
                    "symbol": "AAA",
                    "observation_date": "2024-01-30",
                    "probability": 0.4,
                    "risk_decile": 5,
                    "source": "backtest",
                }
            ],
        )
        if activate == model:
            repo.set_active_run(conn, run_id)


def _settings_keeping(settings, n: int):
    training = settings.training.model_copy(update={"keep_sessions": n})
    return settings.model_copy(update={"training": training})


# --------------------------------------------------------------------------- #
# session identification                                                       #
# --------------------------------------------------------------------------- #
def test_the_session_is_the_run_id_stamp():
    assert repo.session_of("20260814T1334_xgboost_expB_full") == "20260814T1334"


def test_an_experiment_name_containing_underscores_does_not_confuse_the_split():
    """`expB_full` and `expH_pledged_events` both contain underscores - splitting
    on the FIRST one is what keeps the stamp intact."""
    assert repo.session_of("20260814T1334_random_forest_expH_pledged_events") == "20260814T1334"


def test_sessions_are_listed_newest_first(seeded_conn):
    for stamp in ("20260101T0000", "20260301T0000", "20260201T0000"):
        _session(seeded_conn, stamp)
    listed = repo.list_training_sessions(seeded_conn)

    assert list(listed["session"]) == ["20260301T0000", "20260201T0000", "20260101T0000"]
    assert (listed["n_runs"] == 2).all()


def test_the_latest_session_is_the_newest_stamp(seeded_conn):
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000")
    assert repo.latest_training_session(seeded_conn) == "20260301T0000"


def test_listing_an_empty_database_returns_an_empty_frame_not_an_error(conn):
    assert repo.list_training_sessions(conn).empty
    assert repo.latest_training_session(conn) is None


def test_the_active_session_is_flagged(seeded_conn):
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000", activate="xgboost")
    listed = repo.list_training_sessions(seeded_conn).set_index("session")

    assert listed.loc["20260301T0000", "has_active"]
    assert not listed.loc["20260101T0000", "has_active"]


# --------------------------------------------------------------------------- #
# deletion                                                                     #
# --------------------------------------------------------------------------- #
def test_deleting_a_session_removes_its_runs_and_everything_hanging_off_them(seeded_conn):
    """THE ordering test.

    `predictions.run_id` has no ON DELETE CASCADE and foreign keys are enforced,
    so deleting model_runs before predictions raises an IntegrityError. If this
    test starts failing with a foreign-key error, the deletion order was
    reversed.
    """
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000")

    result = repo.delete_training_sessions(seeded_conn, ["20260101T0000"])

    assert result["model_runs"] == 2
    assert result["predictions"] == 2
    remaining = repo.load_model_runs(seeded_conn)
    assert set(remaining["run_id"].map(repo.session_of)) == {"20260301T0000"}
    assert repo.load_metrics(seeded_conn).empty is False  # the survivor's metrics remain


def test_child_metrics_are_removed_with_their_run(seeded_conn):
    _session(seeded_conn, "20260101T0000")
    repo.delete_training_sessions(seeded_conn, ["20260101T0000"])
    assert repo.load_metrics(seeded_conn).empty


def test_deleting_an_unknown_session_is_a_no_op(seeded_conn):
    _session(seeded_conn, "20260101T0000")
    result = repo.delete_training_sessions(seeded_conn, ["19990101T0000"])
    assert result["model_runs"] == 0
    assert len(repo.load_model_runs(seeded_conn)) == 2


def test_deleting_nothing_is_a_no_op(seeded_conn):
    _session(seeded_conn, "20260101T0000")
    assert repo.delete_training_sessions(seeded_conn, [])["model_runs"] == 0


# --------------------------------------------------------------------------- #
# the retention policy                                                         #
# --------------------------------------------------------------------------- #
def test_only_the_newest_n_sessions_survive(seeded_conn, settings):
    for stamp in ("20260101T0000", "20260201T0000", "20260301T0000", "20260401T0000"):
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, "20260401T0000_xgboost_expB_full")

    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 2))

    assert result["kept"] == ["20260401T0000", "20260301T0000"]
    assert sorted(result["removed"]) == ["20260101T0000", "20260201T0000"]
    assert len(repo.list_training_sessions(seeded_conn)) == 2


def test_the_active_session_survives_even_when_it_falls_outside_the_window(
    seeded_conn, settings
):
    """The rule that matters most.

    Retention ranks by recency, but the ACTIVE run may be an older session -
    someone can activate a previous model deliberately. Dropping it would leave
    an is_active row pointing at nothing and the API unable to serve while every
    table still looked populated.
    """
    for stamp in ("20260101T0000", "20260201T0000", "20260301T0000"):
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, "20260101T0000_xgboost_expB_full")  # the OLDEST

    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1))

    assert "20260101T0000" in result["kept"], "the active session was deleted"
    assert repo.get_active_run(seeded_conn) is not None
    assert "20260101T0000" not in result["removed"]


def test_keep_sessions_zero_disables_pruning_rather_than_wiping_everything(
    seeded_conn, settings
):
    """A retention setting that empties the database at zero is a foot-gun."""
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000")

    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 0))

    assert result["removed"] == []
    assert len(repo.list_training_sessions(seeded_conn)) == 2


def test_pruning_is_a_no_op_when_there_is_less_history_than_the_limit(seeded_conn, settings):
    _session(seeded_conn, "20260301T0000", activate="xgboost")
    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 3))
    assert result["removed"] == []


def test_dry_run_reports_without_deleting(seeded_conn, settings):
    for stamp in ("20260101T0000", "20260201T0000", "20260301T0000"):
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, "20260301T0000_xgboost_expB_full")

    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1), dry_run=True)

    assert sorted(result["removed"]) == ["20260101T0000", "20260201T0000"]
    assert len(repo.list_training_sessions(seeded_conn)) == 3, "dry run deleted rows"


def test_pruning_an_empty_database_does_not_raise(conn, settings):
    assert registry.prune_sessions(conn, _settings_keeping(settings, 3))["removed"] == []


def test_pruning_twice_is_stable(seeded_conn, settings):
    for stamp in ("20260101T0000", "20260201T0000", "20260301T0000"):
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, "20260301T0000_xgboost_expB_full")
    kept = _settings_keeping(settings, 1)

    registry.prune_sessions(seeded_conn, kept)
    second = registry.prune_sessions(seeded_conn, kept)

    assert second["removed"] == []
    assert len(repo.list_training_sessions(seeded_conn)) == 1


def test_the_surviving_session_still_serves_after_pruning(seeded_conn, settings):
    """Pruning must leave a WORKING database, not just a smaller one."""
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000", activate="xgboost")

    registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1))

    active = repo.get_active_run(seeded_conn)
    assert active is not None
    assert repo.session_of(active["run_id"]) == "20260301T0000"
    assert not repo.load_predictions(seeded_conn, run_id=active["run_id"]).empty
    assert not repo.load_metrics(seeded_conn).empty


def test_the_dashboards_session_lookup_agrees_with_the_pruner(seeded_conn, settings):
    """Both read the same helper, so they cannot drift apart."""
    for stamp in ("20260101T0000", "20260301T0000"):
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, "20260301T0000_xgboost_expB_full")

    latest = repo.latest_training_session(seeded_conn)
    result = registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1))

    assert latest == "20260301T0000"
    assert result["kept"] == [latest]
    assert repo.latest_training_session(seeded_conn) == latest


def test_a_session_of_many_runs_is_counted_and_removed_whole(seeded_conn, settings):
    """A real session is ~28 runs, not 2 - the pruner must not leave partials."""
    _session(seeded_conn, "20260101T0000", models=("logreg", "random_forest", "xgboost"))
    _session(seeded_conn, "20260301T0000", activate="xgboost")

    listed = repo.list_training_sessions(seeded_conn).set_index("session")
    assert listed.loc["20260101T0000", "n_runs"] == 3

    registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1))
    surviving = repo.load_model_runs(seeded_conn)
    assert set(surviving["run_id"].map(repo.session_of)) == {"20260301T0000"}


def test_prediction_rows_really_go_away(seeded_conn, settings):
    """The whole point: the row count must actually fall."""
    _session(seeded_conn, "20260101T0000")
    _session(seeded_conn, "20260301T0000", activate="xgboost")
    before = repo.table_counts(seeded_conn)["predictions"]

    registry.prune_sessions(seeded_conn, _settings_keeping(settings, 1))

    after = repo.table_counts(seeded_conn)["predictions"]
    assert after < before
    assert after == 2, "one session of two runs should remain"


def test_vacuum_shrinks_the_file_after_a_large_delete(tmp_path, settings):
    """Without VACUUM the retention policy looks like it did nothing.

    SQLite marks freed pages reusable but does not shrink the file, so deleting
    215,850 rows on the real database left it at 149 MB until VACUUM ran - after
    which the same database was 91 MB.
    """
    from pledgecast.db import connection as db

    path = tmp_path / "vacuum.db"
    with db.get_connection(path) as conn:
        db.create_all(conn)
        repo.upsert_companies(conn, [{"symbol": "AAA", "company_name": "A", "industry": "X"}])
        repo.insert_model_run(
            conn,
            run_id="20260101T0000_logreg_expB_full",
            model_name="logreg",
            experiment="expB_full",
            feature_list=["volatility_90d"],
            hyperparams={},
            random_seed=42,
            n_train_rows=1,
            n_folds=1,
            artifact_path=None,
            config_snapshot="{}",
        )
        repo.save_predictions_bulk(
            conn,
            [
                {
                    "run_id": "20260101T0000_logreg_expB_full",
                    "symbol": "AAA",
                    "observation_date": f"2024-01-{(i % 28) + 1:02d}",
                    "probability": 0.5,
                    "risk_decile": 5,
                    "source": "backtest",
                }
                for i in range(20_000)
            ],
        )

    db.dispose_engines()
    grown = path.stat().st_size

    with db.get_connection(path) as conn:
        repo.delete_training_sessions(conn, ["20260101T0000"])
    db.dispose_engines()

    assert path.stat().st_size >= grown * 0.9, "SQLite shrank without VACUUM - assumption changed"

    result = db.vacuum(path)
    assert result["reclaimed_bytes"] > 0
    assert path.stat().st_size < grown


def test_vacuum_on_a_missing_file_is_a_no_op(tmp_path):
    from pledgecast.db import connection as db

    assert db.vacuum(tmp_path / "nope.db")["reclaimed_bytes"] == 0


def test_shipped_config_keeps_a_sane_number_of_sessions(settings):
    assert isinstance(settings.training.keep_sessions, int)
    assert settings.training.keep_sessions >= 0


@pytest.mark.parametrize("keep", [1, 2, 5])
def test_the_number_kept_never_exceeds_the_limit_plus_the_active_one(
    seeded_conn, settings, keep
):
    stamps = [f"2026{month:02d}01T0000" for month in range(1, 7)]
    for stamp in stamps:
        _session(seeded_conn, stamp)
    repo.set_active_run(seeded_conn, f"{stamps[0]}_xgboost_expB_full")  # oldest active

    registry.prune_sessions(seeded_conn, _settings_keeping(settings, keep))

    remaining = repo.list_training_sessions(seeded_conn)
    assert len(remaining) <= keep + 1
    assert pd.Series(remaining["has_active"]).any()
