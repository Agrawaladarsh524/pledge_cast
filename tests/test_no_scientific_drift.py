"""The published snapshot and the database must agree.

Some changes are supposed to move a research number and some are not, and the
difference matters more than either. A documentation pass, an API fix or a
dashboard relabel must leave every published result exactly where it was; a
methodology change must move it, and must be reported.

Nothing enforced that distinction. This does: ``scripts/08_snapshot_results.py``
writes the published numbers to ``reports/snapshot_<label>.json``, and this test
recomputes them from the database and demands they match.

**When this test fails, it is asking a question, not reporting a fault.** Either
a change moved a result that was not supposed to move - in which case revert it -
or the change was a deliberate methodology intervention, in which case rerun the
snapshot under a NEW label and commit both, so the before and after sit in the
repository side by side.

It skips when there is no snapshot or no database, so a fresh clone and CI - where
``DB_PATH`` deliberately points at a database that does not exist - stay green.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS = PROJECT_ROOT / "reports"
SCRIPTS = PROJECT_ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _snapshots() -> list[Path]:
    return sorted(REPORTS.glob("snapshot_*.json"))


@pytest.fixture(scope="module")
def published() -> dict:
    found = _snapshots()
    if not found:
        pytest.skip("no reports/snapshot_*.json to compare against")
    # Newest by modification time: the current baseline is whichever was written
    # last, so a Phase 2 snapshot supersedes the Phase 0 one without edits here.
    newest = max(found, key=lambda p: p.stat().st_mtime)
    return json.loads(newest.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live() -> dict:
    """Recompute the same numbers from the database, right now."""
    try:
        import importlib

        module = importlib.import_module("08_snapshot_results")
        from config import get_settings

        return module.key_numbers(get_settings())
    except Exception as exc:  # noqa: BLE001 - no database is a skip, not a failure
        pytest.skip(f"cannot read live results: {type(exc).__name__}: {exc}")


REGENERATE = (
    "If this change was a deliberate methodology intervention, rerun "
    "`python scripts/08_snapshot_results.py --label <new_label>` and commit both "
    "snapshots so the before and after are both in the repository."
)


# --------------------------------------------------------------------------- #
# the scientific configuration                                                 #
# --------------------------------------------------------------------------- #
def test_the_scientific_configuration_is_unchanged(published, live):
    """Label, event boundary, bootstrap. Moving any of these moves every result."""
    assert live["config"] == published["config"], (
        f"scientific configuration changed.\n{REGENERATE}"
    )


def test_the_experiment_definitions_are_unchanged(published, live):
    """An experiment is its feature list; changing one changes what was compared."""
    assert live["experiments"] == published["experiments"], (
        f"experiment feature lists changed.\n{REGENERATE}"
    )


# --------------------------------------------------------------------------- #
# the published results                                                        #
# --------------------------------------------------------------------------- #
def test_the_active_model_is_unchanged(published, live):
    assert live["active_run_id"] == published["active_run_id"], (
        f"the serving model changed.\n{REGENERATE}"
    )
    assert live["training_session"] == published["training_session"]


def test_the_panel_is_unchanged(published, live):
    for field in ("n_labelled_rows", "n_observation_dates", "base_rate"):
        assert live[field] == published[field], f"{field} changed.\n{REGENERATE}"


def test_every_published_metric_still_holds(published, live):
    """The headline artefact: within-quarter AUC per experiment and model.

    Compared exactly rather than approximately. These are deterministic - the
    seeds are fixed and the folds are fixed - so a difference in the sixth
    decimal is a real change in the pipeline, not floating-point weather.
    """
    was, now = published["primary_metric"], live["primary_metric"]
    assert set(now) == set(was), (
        f"the set of experiment/model pairs changed: "
        f"added {sorted(set(now) - set(was))}, lost {sorted(set(was) - set(now))}.\n{REGENERATE}"
    )
    moved = {key: (was[key], now[key]) for key in was if was[key] != now[key]}
    assert not moved, f"published metrics moved: {moved}\n{REGENERATE}"


# --------------------------------------------------------------------------- #
# the snapshot itself                                                          #
# --------------------------------------------------------------------------- #
def test_each_markdown_snapshot_has_its_json_twin():
    """The readable half and the checkable half are written together or not at all."""
    for path in REPORTS.glob("snapshot_*.md"):
        assert path.with_suffix(".json").exists(), f"{path.name} has no .json twin"


def test_the_snapshot_records_the_commit_it_describes():
    """A result with no provenance cannot be reproduced or superseded."""
    for path in REPORTS.glob("snapshot_*.md"):
        text = path.read_text(encoding="utf-8")
        assert "| commit |" in text, f"{path.name} records no commit"


def test_the_snapshot_is_reproducible():
    """Running the generator twice on an unchanged database must agree."""
    try:
        import importlib

        module = importlib.import_module("08_snapshot_results")
        from config import get_settings

        settings = get_settings()
        first = module.key_numbers(settings)
        second = module.key_numbers(settings)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot read live results: {type(exc).__name__}: {exc}")
    assert first == second, "the snapshot is not deterministic"


def test_git_provenance_never_raises():
    """A snapshot outside a git checkout must still be written."""
    import importlib

    module = importlib.import_module("08_snapshot_results")
    assert isinstance(module._git("rev-parse", "HEAD"), str)
    assert module._git("this-is-not-a-git-command") == "unavailable"
