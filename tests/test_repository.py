"""Repository tests - PLAN.md sec.15.

    (3) upsert idempotency · duplicate PK rejected ·
        empty result -> empty DataFrame not exception            [**]

The idempotency test is not ceremonial. The upsert was written with
``INSERT OR REPLACE`` first, and it broke the full ingest with "FOREIGN KEY
constraint failed": REPLACE *deletes* the conflicting row before reinserting, so
a surrogate autoincrement id is reallocated and every child row pointing at the
old id is orphaned. Phase 2's checks missed it because they only exercised
tables with no children and no surrogate key. These tests cover both.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy.exc import IntegrityError

from pledgecast.db import repository as repo


# --------------------------------------------------------------------------- #
# idempotency                                                                  #
# --------------------------------------------------------------------------- #
def test_upserting_the_same_rows_twice_changes_nothing(seeded_conn):
    rows = [
        {"symbol": "AAA", "quarter_end": "2024-03-31", "submission_date": "2024-04-10",
         "pledge_pct_promoter": 40.0, "pledge_status": "PLEDGE_PRESENT"},
        {"symbol": "BBB", "quarter_end": "2024-03-31", "submission_date": "2024-04-12",
         "pledge_pct_promoter": 0.0, "pledge_status": "NO_PLEDGE"},
    ]
    repo.upsert_pledge_state(seeded_conn, rows)
    first = repo.load_pledge_state(seeded_conn)

    repo.upsert_pledge_state(seeded_conn, rows)
    second = repo.load_pledge_state(seeded_conn)

    assert len(first) == len(second) == 2, "a re-run duplicated rows"
    pd.testing.assert_frame_equal(first, second)


def test_upsert_updates_in_place_rather_than_appending(seeded_conn):
    # submission_date is NOT NULL by design - a filing with no submission date
    # cannot take part in the point-in-time rule (sec.9.3), so the schema
    # refuses to store one.
    key = {"symbol": "AAA", "quarter_end": "2024-03-31", "submission_date": "2024-04-10"}
    repo.upsert_pledge_state(
        seeded_conn, [{**key, "pledge_pct_promoter": 40.0, "pledge_status": "PLEDGE_PRESENT"}]
    )
    repo.upsert_pledge_state(
        seeded_conn, [{**key, "pledge_pct_promoter": 55.0, "pledge_status": "PLEDGE_PRESENT"}]
    )

    state = repo.load_pledge_state(seeded_conn)
    assert len(state) == 1
    assert state.iloc[0]["pledge_pct_promoter"] == pytest.approx(55.0)


def test_re_upserting_a_filing_keeps_its_surrogate_id_and_its_children(seeded_conn):
    """The exact bug INSERT OR REPLACE caused, as a regression test.

    ``filings`` has an autoincrement ``filing_id`` and ``pledge_state`` points at
    it. If the upsert deletes and reinserts, the id changes and the foreign key
    breaks.
    """
    filing = {
        "symbol": "AAA",
        "quarter_end": "2024-03-31",
        "xbrl_url": "https://example.test/a.xml",
        "submission_date": "2024-04-10",
        "status": "downloaded",
        "local_path": "data/raw/xbrl/a.xml",
    }
    repo.upsert_filings(seeded_conn, [filing])
    original_id = int(repo.load_filings(seeded_conn).iloc[0]["filing_id"])

    repo.upsert_pledge_state(
        seeded_conn,
        [{"symbol": "AAA", "quarter_end": "2024-03-31", "submission_date": "2024-04-10",
          "pledge_status": "PLEDGE_PRESENT", "pledge_pct_promoter": 40.0,
          "filing_id": original_id}],
    )

    # Re-running the discovery step must not renumber the filing.
    repo.upsert_filings(seeded_conn, [filing])
    filings = repo.load_filings(seeded_conn)

    assert len(filings) == 1, "the same filing was inserted twice"
    assert int(filings.iloc[0]["filing_id"]) == original_id, "the surrogate id was reallocated"
    assert repo.load_pledge_state(seeded_conn).iloc[0]["filing_id"] == original_id


def _discovered(**overrides) -> dict:
    return {
        "symbol": "AAA",
        "quarter_end": "2024-03-31",
        "xbrl_url": "https://example.test/a.xml",
        "submission_date": "2024-04-10",
        **overrides,
    }


def test_rediscovery_leaves_a_downloaded_filing_completely_untouched(seeded_conn):
    """``preserve_state=True`` is the default and it means exactly that.

    ``01_build_universe.py`` re-runs discovery every time and knows only the URL
    and the dates. The stored row also knows where the file landed, its hash and
    whether it parsed - so the default is DO NOTHING, not DO UPDATE.
    """
    repo.upsert_filings(
        seeded_conn,
        [_discovered(status="parsed", local_path="data/raw/xbrl/a.xml", sha256="abc123")],
    )
    repo.upsert_filings(seeded_conn, [_discovered(submission_date="2024-04-11")])

    row = repo.load_filings(seeded_conn).iloc[0]
    assert len(repo.load_filings(seeded_conn)) == 1
    assert row["local_path"] == "data/raw/xbrl/a.xml"
    assert row["sha256"] == "abc123"
    assert row["status"] == "parsed", "a re-run would have forced a full re-download"
    assert row["submission_date"] == "2024-04-10", "preserve_state must not update"


def test_an_explicit_update_applies_new_values_but_still_keeps_omitted_ones(seeded_conn):
    """``preserve_state=False`` updates ONLY the columns actually supplied.

    This is what INSERT OR REPLACE got wrong: REPLACE deletes the row first, so
    ``local_path`` and ``sha256`` would come back as NULL even though the second
    write never mentioned them.
    """
    repo.upsert_filings(
        seeded_conn,
        [_discovered(status="parsed", local_path="data/raw/xbrl/a.xml", sha256="abc123")],
    )
    repo.upsert_filings(
        seeded_conn, [_discovered(submission_date="2024-04-11")], preserve_state=False
    )

    row = repo.load_filings(seeded_conn).iloc[0]
    assert row["submission_date"] == "2024-04-11", "the new value was not applied"
    assert row["local_path"] == "data/raw/xbrl/a.xml", "an unmentioned column was wiped"
    assert row["sha256"] == "abc123", "an unmentioned column was wiped"


# --------------------------------------------------------------------------- #
# constraints                                                                  #
# --------------------------------------------------------------------------- #
def test_a_foreign_key_to_a_missing_company_is_rejected(conn):
    """Foreign keys are ON per connection (sec.6) - prove it, do not assume it."""
    with pytest.raises(IntegrityError):
        repo.upsert_pledge_state(
            conn,
            [{"symbol": "NOSUCH", "quarter_end": "2024-03-31", "pledge_status": "NO_PLEDGE"}],
        )


def test_activating_an_unknown_run_is_refused(conn):
    from pledgecast.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        repo.set_active_run(conn, "does-not-exist")


def test_exactly_one_run_can_be_active(seeded_conn):
    for name in ("run-a", "run-b"):
        repo.insert_model_run(
            seeded_conn,
            run_id=name,
            model_name="xgboost",
            experiment="expB_full",
            feature_list=["volatility_90d"],
            hyperparams={"max_depth": 3},
            random_seed=42,
        )
    repo.set_active_run(seeded_conn, "run-a")
    repo.set_active_run(seeded_conn, "run-b")

    runs = repo.load_model_runs(seeded_conn)
    assert int(runs["is_active"].sum()) == 1
    assert repo.get_active_run(seeded_conn)["run_id"] == "run-b"


def test_an_invalid_prediction_source_is_rejected(seeded_conn):
    from pledgecast.exceptions import ValidationError

    repo.insert_model_run(
        seeded_conn,
        run_id="run-a",
        model_name="xgboost",
        experiment="expB_full",
        feature_list=["volatility_90d"],
        hyperparams={},
        random_seed=42,
    )
    with pytest.raises(ValidationError):
        repo.save_prediction(
            seeded_conn,
            run_id="run-a",
            symbol="AAA",
            observation_date="2024-04-30",
            probability=0.5,
            source="not-a-source",
        )


def test_a_probability_outside_zero_one_is_rejected(seeded_conn):
    from pledgecast.exceptions import ValidationError

    repo.insert_model_run(
        seeded_conn,
        run_id="run-a",
        model_name="xgboost",
        experiment="expB_full",
        feature_list=["volatility_90d"],
        hyperparams={},
        random_seed=42,
    )
    with pytest.raises(ValidationError):
        repo.save_prediction(
            seeded_conn,
            run_id="run-a",
            symbol="AAA",
            observation_date="2024-04-30",
            probability=1.5,
            source="api",
        )


# --------------------------------------------------------------------------- #
# empty results                                                                #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "loader",
    [
        repo.load_pledge_state,
        repo.load_prices,
        repo.load_panel,
        repo.load_predictions,
        repo.load_model_runs,
        repo.load_backtest_results,
        repo.load_filings,
        repo.load_companies,
        repo.load_pledge_events,
    ],
)
def test_an_empty_table_returns_an_empty_frame_not_an_exception(conn, loader):
    """sec.10: "Empty datasets -> 'no data for this selection', not IndexError".

    And crucially it must still have its COLUMNS - the dashboard indexes into
    them before it ever checks whether there are rows.
    """
    frame = loader(conn)
    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert len(frame.columns) > 0, "an empty frame with no columns breaks every caller"


def test_missing_lookups_return_none_rather_than_raising(conn):
    assert repo.get_company(conn, "NOSUCH") is None
    assert repo.get_active_run(conn) is None
    assert repo.get_model_run(conn, "nope") is None
    assert repo.get_panel_row(conn, "NOSUCH") is None
    assert repo.get_universe_symbols(conn) == []
