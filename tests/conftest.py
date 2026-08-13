"""Shared fixtures - PLAN.md sec.15.

"conftest.py provides an in-memory SQLite fixture and a synthetic 3-company
 panel. Fixture XBRL files are committed to tests/fixtures/ - they make
 parser tests real, not mocked."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import get_settings
from pledgecast.db import repository as repo
from pledgecast.db.connection import create_all, dispose_engines, get_connection

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def db(tmp_path):
    """A real SQLite file per test - schema created, nothing in it.

    A file rather than ``:memory:`` because the connection helper opens and
    closes connections per transaction, which would discard an in-memory
    database between calls.
    """
    path = tmp_path / "test.db"
    with get_connection(path) as conn:
        create_all(conn)
    yield path
    dispose_engines()


@pytest.fixture
def conn(db):
    with get_connection(db) as connection:
        yield connection


@pytest.fixture
def seeded_conn(db):
    """Three companies, so foreign keys are satisfiable."""
    with get_connection(db) as connection:
        repo.upsert_companies(
            connection,
            [
                {"symbol": "AAA", "company_name": "Alpha Ltd", "industry": "Power"},
                {"symbol": "BBB", "company_name": "Beta Ltd", "industry": "IT"},
                {"symbol": "CCC", "company_name": "Gamma Ltd", "industry": "Metals"},
            ],
        )
        yield connection


@pytest.fixture
def sample_panel() -> pd.DataFrame:
    """Synthetic 3-company x 8-quarter panel with a known structure.

    AAA has a rising pledge and falls; BBB is flat and clean; CCC is volatile
    but unpledged - which is the confound sec.2.1 warns about, in miniature.
    """
    rng = np.random.default_rng(42)
    quarters = pd.date_range("2022-03-31", periods=8, freq="QE")
    rows = []
    for symbol, base_pledge, drift, vol in (
        ("AAA", 40.0, 4.0, 0.45),
        ("BBB", 0.0, 0.0, 0.22),
        ("CCC", 0.0, 0.0, 0.60),
    ):
        for index, quarter_end in enumerate(quarters):
            pledge = base_pledge + drift * index
            observation = (quarter_end + pd.Timedelta(days=30)).date().isoformat()
            rows.append(
                {
                    "symbol": symbol,
                    "observation_date": observation,
                    "quarter_end": quarter_end.date().isoformat(),
                    "promoter_holding_pct": 55.0,
                    "pledge_pct_promoter": pledge,
                    "pledge_pct_equity": pledge * 0.55,
                    "volatility_90d": vol + rng.normal(0, 0.01),
                    "log_turnover_90d": 18.0 + rng.normal(0, 0.2),
                    "trailing_dd_60d": -0.08,
                    "return_90d": 0.02,
                    "rel_return_90d": 0.01,
                    "is_stale": 0,
                    "label_is_valid": 1,
                }
            )
    return pd.DataFrame(rows)


def _fixture(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture {name} not present - run scripts/02_ingest_all.py")
    return path


@pytest.fixture
def xbrl_pledge_legacy() -> Path:
    """Legacy taxonomy, pledge present. Verified against the master API."""
    return _fixture("pledge_present_legacy.xml")


@pytest.fixture
def xbrl_pledge_fraction_scale() -> Path:
    """Era-3 taxonomy, where percentages are filed as fractions."""
    return _fixture("pledge_present_fraction_scale.xml")


@pytest.fixture
def xbrl_no_pledge() -> Path:
    """An EXPLICIT zero - must parse to NO_PLEDGE, never UNAVAILABLE."""
    return _fixture("no_pledge.xml")


@pytest.fixture
def xbrl_malformed() -> Path:
    """Truncated mid-document; must quarantine rather than crash."""
    return _fixture("malformed.xml")
