"""Reg 31 event feature tests.

Two things here are load-bearing and neither is obvious:

**The materiality filter.** Without it, ``event_count_90d`` ranks CDSL as the
most pledge-active company in the study while its promoter pledge is 0.00% in
every quarter. The test below reproduces that exact shape - a company with
hundreds of 0.00% clearing disclosures - and asserts it is excluded.

**The disclosure buffer.** ``event_date`` is when a pledge was created, not when
it became public, and Reg 31 allows up to 7 working days to disclose it. Every
window must close early. As with ``test_leakage.py``, the buffer gets a NEGATIVE
TWIN: an event planted inside the buffer zone must NOT be counted.

The buffer was 11 days and 11 was too short - see the WORST-CASE CALENDAR
section at the foot of this module, which plants the widest real stretch in the
ingested NSE calendar and fails at any buffer below 14.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import get_settings
from pledgecast.evaluation import leakage
from pledgecast.features.events import (
    EVENT_FEATURES,
    build_event_features,
    coverage_report,
    filter_material,
)

pytestmark = pytest.mark.leakage

WINDOW = 90
INVOCATION_WINDOW = 365
# Bound to the configured value rather than restated, so this module cannot
# drift away from the buffer the panel is actually built with.
LAG = get_settings().features.event_disclosure_lag_days


def _events(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """``(symbol, event_date, event_type, pct_equity)`` rows."""
    return pd.DataFrame(rows, columns=["symbol", "event_date", "event_type", "pct_equity"])


def _observations(symbol: str = "AAA", date: str = "2024-06-30") -> pd.DataFrame:
    return pd.DataFrame({"symbol": [symbol], "observation_date": [date]})


def _build(events: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    return build_event_features(
        events,
        observations,
        window_days=WINDOW,
        invocation_window_days=INVOCATION_WINDOW,
        disclosure_lag_days=LAG,
    )


# --------------------------------------------------------------------------- #
# materiality                                                                  #
# --------------------------------------------------------------------------- #
def test_zero_percent_clearing_disclosures_are_excluded():
    """The CDSL shape: hundreds of events, none of them promoter financing."""
    noise = _events([("CDSL", "2024-05-01", "release", 0.0)] * 300)
    real = _events([("AAA", "2024-05-01", "creation", 0.45)])

    material, report = filter_material(pd.concat([noise, real]), min_pct_equity=0.01)

    assert len(material) == 1, "0.00% clearing disclosures survived the filter"
    assert material.iloc[0]["symbol"] == "AAA"
    assert report["dropped"] == 300
    assert report["companies_material"] == 1


def test_the_filter_is_the_filings_own_precision_not_a_tuned_cutoff():
    """pct_equity is filed to 2 decimals, so 0.01 is the smallest real value."""
    events = _events(
        [
            ("AAA", "2024-05-01", "creation", 0.00),
            ("AAA", "2024-05-02", "creation", 0.01),
            ("AAA", "2024-05-03", "creation", 1.50),
        ]
    )
    material, _ = filter_material(events, min_pct_equity=0.01)
    assert len(material) == 2
    assert material["pct_equity"].min() == pytest.approx(0.01)


def test_an_empty_event_table_is_handled_rather_than_crashing():
    material, report = filter_material(pd.DataFrame(), min_pct_equity=0.01)
    assert material.empty
    assert report["total"] == 0


# --------------------------------------------------------------------------- #
# the disclosure buffer - THE leakage control                                  #
# --------------------------------------------------------------------------- #
def test_an_event_comfortably_inside_the_window_is_counted():
    events = _events([("AAA", "2024-05-01", "creation", 2.0)])
    out = _build(events, _observations(date="2024-06-30"))

    assert out.iloc[0]["event_count_90d"] == 1
    assert out.iloc[0]["event_created_90d"] == pytest.approx(2.0)


def test_an_event_inside_the_disclosure_buffer_is_NOT_counted():
    """The negative twin, and the whole reason this module needs care.

    The event happened 5 days before the observation date. Reg 31 gives the
    promoter up to 7 working days to disclose it, so on the observation date it
    may not have been public. Counting it would be leakage.
    """
    events = _events([("AAA", "2024-06-25", "creation", 2.0)])
    out = _build(events, _observations(date="2024-06-30"))

    assert out.iloc[0]["event_count_90d"] == 0, "an undisclosed event was counted"
    assert out.iloc[0]["event_created_90d"] == 0.0


def test_the_buffer_boundary_is_exact():
    """Dated exactly `lag` days before: on the boundary, so it counts."""
    boundary = (pd.Timestamp("2024-06-30") - pd.Timedelta(days=LAG)).strftime("%Y-%m-%d")
    just_inside = (pd.Timestamp("2024-06-30") - pd.Timedelta(days=LAG - 1)).strftime("%Y-%m-%d")

    assert _build(_events([("AAA", boundary, "creation", 1.0)]), _observations()).iloc[0][
        "event_count_90d"
    ] == 1
    assert _build(_events([("AAA", just_inside, "creation", 1.0)]), _observations()).iloc[0][
        "event_count_90d"
    ] == 0


def test_an_event_older_than_the_window_is_not_counted():
    events = _events([("AAA", "2023-01-01", "creation", 5.0)])
    out = _build(events, _observations(date="2024-06-30"))
    assert out.iloc[0]["event_count_90d"] == 0
    # ...but it still sets days-since, which looks back over all history.
    assert out.iloc[0]["event_days_since"] > 400


def test_a_future_event_never_leaks_backwards():
    events = _events([("AAA", "2025-01-01", "creation", 5.0)])
    out = _build(events, _observations(date="2024-06-30"))
    assert out.iloc[0]["event_count_90d"] == 0
    assert pd.isna(out.iloc[0]["event_days_since"]), "a future event set days-since"


# --------------------------------------------------------------------------- #
# the arithmetic                                                               #
# --------------------------------------------------------------------------- #
def test_creations_and_releases_are_summed_by_size_and_netted():
    events = _events(
        [
            ("AAA", "2024-05-01", "creation", 3.0),
            ("AAA", "2024-05-10", "creation", 1.5),
            ("AAA", "2024-05-20", "release", 2.0),
        ]
    )
    row = _build(events, _observations(date="2024-06-30")).iloc[0]

    assert row["event_created_90d"] == pytest.approx(4.5)
    assert row["event_released_90d"] == pytest.approx(2.0)
    assert row["event_net_90d"] == pytest.approx(2.5)
    assert row["event_count_90d"] == 3


def test_net_is_negative_when_more_was_released_than_created():
    events = _events(
        [("AAA", "2024-05-01", "creation", 1.0), ("AAA", "2024-05-02", "release", 4.0)]
    )
    assert _build(events, _observations()).iloc[0]["event_net_90d"] == pytest.approx(-3.0)


def test_invocations_use_their_own_longer_window():
    """169 material invocations exist in the whole archive; at 90 days only 14
    panel rows carry one. The wider window is why the feature is usable."""
    events = _events([("AAA", "2023-10-01", "invocation", 1.0)])
    row = _build(events, _observations(date="2024-06-30")).iloc[0]

    assert row["event_count_90d"] == 0, "outside the 90-day window"
    assert row["event_invocations_365d"] == 1, "but inside the 365-day invocation window"


def test_days_since_is_null_when_a_company_never_filed_rather_than_zero():
    """No event ever is not 'an event today'. Writing 0 invents a fact."""
    events = _events([("BBB", "2024-05-01", "creation", 1.0)])
    out = _build(events, _observations(symbol="AAA"))

    assert pd.isna(out.iloc[0]["event_days_since"])
    # Counts, by contrast, ARE genuinely zero - no activity is real information.
    assert out.iloc[0]["event_count_90d"] == 0
    assert out.iloc[0]["event_net_90d"] == 0.0


def test_each_company_only_sees_its_own_events():
    events = _events(
        [
            ("AAA", "2024-05-01", "creation", 1.0),
            ("BBB", "2024-05-01", "creation", 9.0),
        ]
    )
    observations = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "observation_date": ["2024-06-30", "2024-06-30"]}
    )
    out = _build(events, observations).set_index("symbol")

    assert out.loc["AAA", "event_created_90d"] == pytest.approx(1.0)
    assert out.loc["BBB", "event_created_90d"] == pytest.approx(9.0)


def test_every_declared_feature_column_is_produced():
    out = _build(_events([("AAA", "2024-05-01", "creation", 1.0)]), _observations())
    for column in EVENT_FEATURES:
        assert column in out.columns


def test_no_events_at_all_still_returns_the_full_column_set():
    out = _build(pd.DataFrame(columns=["symbol", "event_date", "event_type", "pct_equity"]),
                 _observations())
    for column in EVENT_FEATURES:
        assert column in out.columns
    assert out.iloc[0]["event_count_90d"] == 0


def test_coverage_report_measures_sparsity_rather_than_hiding_it():
    events = _events([("AAA", "2024-05-01", "creation", 1.0)])
    observations = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "observation_date": ["2024-06-30"] * 3,
        }
    )
    report = coverage_report(_build(events, observations))

    assert report["rows"] == 3
    assert report["rows_with_event"] == 1
    assert report["pct_with_event"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------- #
# the leakage check itself                                                     #
# --------------------------------------------------------------------------- #
def test_the_disclosure_lag_check_passes_on_correctly_built_features():
    events = _events(
        [("AAA", "2024-05-01", "creation", 2.0), ("AAA", "2024-06-25", "creation", 2.0)]
    )
    panel = _build(events, _observations(date="2024-06-30"))
    result = leakage.check_events_respect_disclosure_lag(
        panel, events, lag_days=LAG, window_days=WINDOW, min_pct_equity=0.01
    )
    assert result["passed"], result


def test_the_disclosure_lag_check_catches_an_unbuffered_count():
    """The negative twin for the CHECK, not just the feature.

    Simulates what a naive implementation would store: the count with no buffer,
    which includes the event dated 5 days before the observation date.
    """
    events = _events(
        [("AAA", "2024-05-01", "creation", 2.0), ("AAA", "2024-06-25", "creation", 2.0)]
    )
    panel = _build(events, _observations(date="2024-06-30"))
    assert panel.loc[0, "event_count_90d"] == 1

    leaked = panel.copy()
    leaked.loc[0, "event_count_90d"] = 2  # what no-buffer would have produced

    result = leakage.check_events_respect_disclosure_lag(
        leaked, events, lag_days=LAG, window_days=WINDOW, min_pct_equity=0.01
    )
    assert not result["passed"], "an unbuffered event count was reported as clean"
    assert result["sample"][0]["without_buffer"] == 2


# --------------------------------------------------------------------------- #
# WORST-CASE CALENDAR - why the buffer is 14 and not 11                        #
# --------------------------------------------------------------------------- #
# Regulation 31 is written in WORKING days; the buffer is applied in CALENDAR
# days. The two are not interchangeable, and the gap between them is not
# constant - it depends on where in the calendar the event falls. The original
# 11-day buffer was reasoned as "7 working days plus a weekend", which silently
# assumed at most one weekend and no holidays inside the window.
#
# The stretch below is the widest one in the ingested NSE calendar, and it is
# real, not constructed: a pledge created on Monday 2023-03-27 has its seventh
# trading session on Monday 2023-04-10, FOURTEEN calendar days later, because
# two weekends and three exchange holidays land inside the window.
#
# Measured over all 1,235 ingested sessions the deadline falls 8 to 14 calendar
# days after creation, median 10. 1,325 of the 5,211 material events in the
# study period - 25.4% - have a deadline more than 11 days out. None exceeds 14.
#
# Trading sessions are a SUBSET of working days (the exchange closes on days
# offices do not), so measuring in sessions overstates the true span. The error
# is in the safe direction.
CREATED = "2023-03-27"  # Monday

# The real NSE sessions in that stretch, in order. Absent days are the closures.
SESSIONS_AFTER = [
    "2023-03-28",  # Tue   1
    "2023-03-29",  # Wed   2   (Thu 03-30 closed - Ram Navami)
    "2023-03-31",  # Fri   3
    "2023-04-03",  # Mon   4   (Sat 04-01, Sun 04-02 weekend)
    "2023-04-05",  # Wed   5   (Tue 04-04 closed - Mahavir Jayanti)
    "2023-04-06",  # Thu   6
    "2023-04-10",  # Mon   7   (Fri 04-07 closed - Good Friday; 04-08/09 weekend)
]
DEADLINE = SESSIONS_AFTER[6]  # the seventh working day: the latest lawful disclosure


def test_seven_working_days_can_span_fourteen_calendar_days():
    """The arithmetic the old 11-day buffer got wrong, stated on real dates."""
    assert len(SESSIONS_AFTER) == 7
    span = (pd.Timestamp(DEADLINE) - pd.Timestamp(CREATED)).days
    assert span == 14, "the planted stretch is no longer 14 days wide"
    assert span > 11, "an 11-day buffer would close before this deadline"


def test_the_configured_buffer_covers_the_widest_disclosure_deadline():
    """THE regression test. Fails at 11; passes at 14.

    A buffer shorter than the widest lawful disclosure gap does not merely lose
    accuracy - it counts events the market could not yet have seen, which is the
    one failure mode this whole module exists to prevent. Asserted against
    ``config.yaml`` so lowering the buffer breaks the build rather than quietly
    reintroducing the leak.
    """
    span = (pd.Timestamp(DEADLINE) - pd.Timestamp(CREATED)).days
    assert span <= LAG, (
        f"event_disclosure_lag_days={LAG} closes the window {span - LAG} day(s) "
        f"before an event created {CREATED} was necessarily public ({DEADLINE})"
    )


@pytest.mark.parametrize(
    "observed",
    ["2023-04-09", "2023-04-08", "2023-04-05", "2023-03-28"],
    ids=["day before deadline", "two days before", "mid-window", "day after creation"],
)
def test_an_event_is_not_counted_before_its_disclosure_deadline(observed):
    """Every observation date strictly before the deadline must be blind to it.

    ``2023-04-08`` is the one that matters: the old 11-day buffer counted the
    event there, two days before the promoter was obliged to file.
    """
    events = _events([("AAA", CREATED, "creation", 2.0)])
    row = _build(events, _observations(date=observed)).iloc[0]

    assert row["event_count_90d"] == 0, (
        f"an event created {CREATED} was counted on {observed}, but it was not "
        f"required to be public until {DEADLINE}"
    )
    assert row["event_created_90d"] == 0.0
    assert pd.isna(row["event_days_since"]), "days-since leaked the undisclosed event"


def test_the_same_event_IS_counted_from_the_deadline_onwards():
    """The positive twin: a buffer that never counts anything is not a fix.

    The window closes exactly when the disclosure obligation matures - the
    deadline day itself counts, matching ``test_the_buffer_boundary_is_exact``.
    Excluding it too would be conservatism with nothing behind it.
    """
    for observed in (DEADLINE, "2023-04-11", "2023-05-01"):
        row = _build(
            _events([("AAA", CREATED, "creation", 2.0)]), _observations(date=observed)
        ).iloc[0]
        assert row["event_count_90d"] == 1, f"the event went missing on {observed}"
        assert row["event_created_90d"] == pytest.approx(2.0)


def test_the_old_eleven_day_buffer_would_have_counted_it():
    """Proof the bug was real, kept as an executable record.

    Pinned at 11 explicitly - this asserts what the SUPERSEDED setting did, so
    it must not follow ``LAG``. Observed 2023-04-08: two days before the
    disclosure deadline, and an 11-day window reaches back to 2023-03-28, which
    is one day past the creation date.
    """
    observed = "2023-04-08"
    assert (pd.Timestamp(observed) - pd.Timestamp(CREATED)).days < 14

    leaked = build_event_features(
        _events([("AAA", CREATED, "creation", 2.0)]),
        _observations(date=observed),
        window_days=WINDOW,
        invocation_window_days=INVOCATION_WINDOW,
        disclosure_lag_days=11,
    )
    assert leaked.iloc[0]["event_count_90d"] == 1, (
        "the 11-day buffer no longer reproduces the leak this test documents"
    )


def test_window_sums_scale_to_the_real_panel_shape():
    """A company with many events must not be O(n^2) or silently truncated."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-01", periods=600)
    events = _events(
        [
            ("AAA", d.strftime("%Y-%m-%d"), "creation", float(rng.uniform(0.01, 0.5)))
            for d in dates
        ]
    )
    observations = pd.DataFrame(
        {
            "symbol": "AAA",
            "observation_date": [d.strftime("%Y-%m-%d") for d in dates[::20]],
        }
    )
    out = _build(events, observations)

    assert len(out) == len(observations)
    assert out["event_count_90d"].max() > 0
    assert (out["event_created_90d"] >= 0).all()
