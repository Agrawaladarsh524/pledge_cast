"""Feature tests - PLAN.md sec.15.

    (4) QoQ change on synthetic panel · acceleration needs 3 quarters ·
        consecutive-rise counter · stale forward-fill capped at 1 quarter  [**]

The trajectory features are computed on a REINDEXED complete quarterly grid, and
that is the thing most worth testing. A naive ``.diff()`` over only the rows
present would compare Q1 against Q3 whenever Q2 is missing and report the result
as a one-quarter change - silently, with no error and a plausible-looking
number. Several tests here exist specifically to catch that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pledgecast.features import market
from pledgecast.features.pledge import build_pledge_features

QUARTERS = [
    "2022-03-31",
    "2022-06-30",
    "2022-09-30",
    "2022-12-31",
    "2023-03-31",
    "2023-06-30",
]


def _state(rows: list[dict]) -> pd.DataFrame:
    """A pledge_state frame with sensible defaults for the untouched columns."""
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "promoter_holding_pct": 55.0,
                "pledge_pct_equity": row.get("pledge_pct_promoter", 0.0) * 0.55,
                "pledge_status": "PLEDGE_PRESENT",
                "submission_date": (
                    pd.Timestamp(row["quarter_end"]) + pd.Timedelta(days=15)
                ).date().isoformat(),
                **row,
            }
            for row in rows
        ]
    )


# --------------------------------------------------------------------------- #
# quarter-on-quarter change                                                    #
# --------------------------------------------------------------------------- #
def test_quarter_on_quarter_change_is_the_difference_in_percentage_points():
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2022-06-30", "pledge_pct_promoter": 25.0},
            {"quarter_end": "2022-09-30", "pledge_pct_promoter": 20.0},
        ]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")

    assert pd.isna(out.loc["2022-03-31", "pledge_chg_1q"]), "the first quarter has no predecessor"
    assert out.loc["2022-06-30", "pledge_chg_1q"] == pytest.approx(15.0)
    assert out.loc["2022-09-30", "pledge_chg_1q"] == pytest.approx(-5.0)
    assert out.loc["2022-09-30", "pledge_chg_2q"] == pytest.approx(10.0)


def test_a_missing_quarter_does_not_masquerade_as_a_one_quarter_change():
    """The bug the reindexed grid exists to prevent.

    Q2 is absent entirely. Beyond the one quarter of permitted forward fill,
    the change must be NaN - never Q3 minus Q1 reported as a single-quarter
    move.
    """
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 10.0},
            # 2022-06-30 and 2022-09-30 missing
            {"quarter_end": "2022-12-31", "pledge_pct_promoter": 40.0},
        ]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")

    # Q2 is carried from Q1 (one quarter of fill is allowed), so its change is 0.
    assert out.loc["2022-06-30", "pledge_chg_1q"] == pytest.approx(0.0)
    assert out.loc["2022-06-30", "is_stale"] == 1
    # Q3 is beyond the cap - the level is gone and so is any change.
    assert pd.isna(out.loc["2022-09-30", "pledge_pct_promoter"])
    assert pd.isna(out.loc["2022-09-30", "pledge_chg_1q"])
    # Q4 has a real filing but its predecessor is unknown, so no false 30pp jump.
    assert pd.isna(out.loc["2022-12-31", "pledge_chg_1q"])


# --------------------------------------------------------------------------- #
# acceleration                                                                 #
# --------------------------------------------------------------------------- #
def test_acceleration_needs_three_quarters_of_history():
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2022-06-30", "pledge_pct_promoter": 15.0},
            {"quarter_end": "2022-09-30", "pledge_pct_promoter": 25.0},
        ]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")

    assert pd.isna(out.loc["2022-03-31", "pledge_accel"])
    assert pd.isna(out.loc["2022-06-30", "pledge_accel"]), "two quarters cannot give acceleration"
    # change went +5 then +10, so acceleration is +5
    assert out.loc["2022-09-30", "pledge_accel"] == pytest.approx(5.0)


def test_a_steady_rise_has_zero_acceleration():
    state = _state(
        [{"quarter_end": q, "pledge_pct_promoter": 10.0 * (i + 1)} for i, q in enumerate(QUARTERS)]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")
    steady = out.loc["2022-09-30":, "pledge_accel"].dropna()
    assert (steady.abs() < 1e-9).all(), "a constant rise is not accelerating"


# --------------------------------------------------------------------------- #
# consecutive-rise counter                                                     #
# --------------------------------------------------------------------------- #
def test_consecutive_rise_counter_counts_and_resets():
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2022-06-30", "pledge_pct_promoter": 12.0},  # rise 1
            {"quarter_end": "2022-09-30", "pledge_pct_promoter": 14.0},  # rise 2
            {"quarter_end": "2022-12-31", "pledge_pct_promoter": 18.0},  # rise 3
            {"quarter_end": "2023-03-31", "pledge_pct_promoter": 11.0},  # fall -> reset
            {"quarter_end": "2023-06-30", "pledge_pct_promoter": 13.0},  # rise 1 again
        ]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")

    # The first quarter is NaN, not 0. With no predecessor, "has it risen?" is
    # unknown - and 0 would assert it did not, which is a claim the data does
    # not support. Same reasoning as pledge_chg_1q being NaN there.
    assert pd.isna(out["consecutive_rising_q"].iloc[0])
    assert list(out["consecutive_rising_q"].iloc[1:]) == [1, 2, 3, 0, 1]


def test_an_unchanged_pledge_does_not_count_as_a_rise():
    """90.5% of real company-quarters are unchanged - flat must not read as rising."""
    state = _state([{"quarter_end": q, "pledge_pct_promoter": 30.0} for q in QUARTERS])
    out = build_pledge_features(state, QUARTERS)
    assert (out["consecutive_rising_q"].iloc[1:] == 0).all()
    assert pd.isna(out["consecutive_rising_q"].iloc[0])


def test_rolling_max_looks_back_four_quarters_only():
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 90.0},  # the spike
            {"quarter_end": "2022-06-30", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2022-09-30", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2022-12-31", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2023-03-31", "pledge_pct_promoter": 10.0},
            {"quarter_end": "2023-06-30", "pledge_pct_promoter": 10.0},
        ]
    )
    out = build_pledge_features(state, QUARTERS, rolling_max_quarters=4).set_index("quarter_end")
    assert out.loc["2022-12-31", "pledge_max_4q"] == pytest.approx(90.0), "still inside the window"
    assert out.loc["2023-03-31", "pledge_max_4q"] == pytest.approx(10.0), "spike has aged out"


# --------------------------------------------------------------------------- #
# forward fill capped at one quarter (sec.10)                                  #
# --------------------------------------------------------------------------- #
def test_forward_fill_is_capped_and_flagged():
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 20.0},
            # three quarters missing
            {"quarter_end": "2023-03-31", "pledge_pct_promoter": 20.0},
        ]
    )
    out = build_pledge_features(state, QUARTERS, max_forward_fill_quarters=1).set_index(
        "quarter_end"
    )

    assert out.loc["2022-03-31", "is_stale"] == 0, "a real filing is not stale"
    assert out.loc["2022-06-30", "pledge_pct_promoter"] == pytest.approx(20.0)
    assert out.loc["2022-06-30", "is_stale"] == 1, "a carried row must be flagged"
    assert pd.isna(out.loc["2022-09-30", "pledge_pct_promoter"]), "fill exceeded its cap"
    assert out.loc["2022-09-30", "is_stale"] == 0, "dropped, not stale"


def test_unavailable_is_never_treated_as_zero():
    """sec.2.4's three-way status: UNAVAILABLE is unknown, not 'no pledge'."""
    state = _state(
        [
            {"quarter_end": "2022-03-31", "pledge_pct_promoter": 40.0},
            {"quarter_end": "2022-06-30", "pledge_pct_promoter": 0.0,
             "pledge_status": "UNAVAILABLE"},
        ]
    )
    out = build_pledge_features(state, QUARTERS).set_index("quarter_end")
    # Blanked before arithmetic, then forward-filled from Q1 - so it must never
    # produce a -40pp collapse that never happened.
    assert out.loc["2022-06-30", "pledge_chg_1q"] != pytest.approx(-40.0)


# --------------------------------------------------------------------------- #
# market features                                                              #
# --------------------------------------------------------------------------- #
def test_market_features_are_blanked_when_a_break_sits_in_the_lookback():
    """sec.10 corporate actions, applied to the BACKWARD window."""
    dates = pd.bdate_range("2024-01-01", periods=150)
    prices = pd.DataFrame(
        {
            "symbol": "AAA",
            "trade_date": [d.date().isoformat() for d in dates],
            "adj_close": 100.0,
            "volume": 1000.0,
        }
    )
    break_date = prices["trade_date"].iloc[100]
    observations = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "observation_date": [prices["trade_date"].iloc[99], prices["trade_date"].iloc[101]],
            "volatility_90d": [0.4, 0.4],
            "trailing_dd_60d": [-0.1, -0.1],
            "return_90d": [0.05, 0.05],
            "rel_return_90d": [0.02, 0.02],
            "log_turnover_90d": [18.0, 18.0],
        }
    )

    out, report = market.blank_features_spanning_breaks(
        observations.copy(),
        prices,
        {"AAA": [break_date]},
        volatility_window=90,
        drawdown_window=60,
        return_window=90,
        turnover_window=90,
    )

    before, after = out.iloc[0], out.iloc[1]
    assert before["volatility_90d"] == pytest.approx(0.4), "a break in the FUTURE is not a problem"
    assert np.isnan(after["volatility_90d"]), "a break inside the lookback must blank the feature"
    assert report["rows_blanked"] == 1


def test_a_break_older_than_the_window_no_longer_blanks():
    dates = pd.bdate_range("2024-01-01", periods=300)
    prices = pd.DataFrame(
        {
            "symbol": "AAA",
            "trade_date": [d.date().isoformat() for d in dates],
            "adj_close": 100.0,
            "volume": 1000.0,
        }
    )
    observations = pd.DataFrame(
        {
            "symbol": ["AAA"],
            # 120 trading days after the break - beyond the 90-day window
            "observation_date": [prices["trade_date"].iloc[130]],
            "volatility_90d": [0.4],
            "trailing_dd_60d": [-0.1],
            "return_90d": [0.05],
            "rel_return_90d": [0.02],
            "log_turnover_90d": [18.0],
        }
    )
    out, report = market.blank_features_spanning_breaks(
        observations.copy(),
        prices,
        {"AAA": [prices["trade_date"].iloc[10]]},
        volatility_window=90,
        drawdown_window=60,
        return_window=90,
        turnover_window=90,
    )
    assert out.iloc[0]["volatility_90d"] == pytest.approx(0.4)
    assert report["rows_blanked"] == 0
