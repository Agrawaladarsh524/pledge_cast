"""Forward-drawdown label tests - PLAN.md sec.15, priority ***.

    "(4) known series -> known drawdown . exact -15% boundary .
     insufficient future data -> null not 0 . adjusted-price sanity"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pledgecast.labels import drawdown

pytestmark = pytest.mark.critical


# --------------------------------------------------------------------------- #
# 1. known series -> known drawdown                                           #
# --------------------------------------------------------------------------- #
def test_known_series_gives_known_drawdown():
    """Entry 100, trough 80 inside the window -> exactly -20%."""
    prices = np.array([100.0, 95.0, 80.0, 90.0, 110.0])
    result = drawdown.forward_drawdown(prices, horizon=3)

    # From index 0: min(95, 80, 90) = 80  ->  80/100 - 1 = -0.20
    assert result[0] == pytest.approx(-0.20)
    # From index 1: min(80, 90, 110) = 80 ->  80/95 - 1
    assert result[1] == pytest.approx(80 / 95 - 1)
    # No full window remains from index 2 onward.
    assert np.isnan(result[2:]).all()


def test_worst_decline_from_entry_not_peak_to_trough():
    """sec.9.2's central distinction.

    A stock that rises 40% then falls 15% from that peak is NOT a downside event
    for someone holding from entry - they are still up. Peak-to-trough would
    fire here; worst-decline-from-entry must not.
    """
    prices = np.array([100.0, 120.0, 140.0, 119.0, 125.0])
    result = drawdown.forward_drawdown(prices, horizon=4)

    # Trough 119 is -15% from the 140 peak, but +19% from the 100 entry.
    assert result[0] == pytest.approx(0.19)
    assert result[0] > 0, "an investor entering at 100 never lost money"


# --------------------------------------------------------------------------- #
# 2. the exact threshold boundary                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("trough", "expected_label"),
    [
        (85.0, 1),      # exactly -15.0% -> the boundary is inclusive (<=)
        (84.99, 1),     # just past it
        (85.01, 0),     # just short of it
        (90.0, 0),
    ],
)
def test_threshold_boundary_is_inclusive(trough, expected_label):
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "adj_close": [100.0, trough, 100.0],
        }
    )
    labelled = drawdown.label_series(frame, horizon=2, threshold=-0.15)
    assert int(labelled["label"].iloc[0]) == expected_label


# --------------------------------------------------------------------------- #
# 3. insufficient future data -> NULL, never 0                                #
# --------------------------------------------------------------------------- #
def test_insufficient_future_data_is_null_not_zero():
    """sec.9.4's embargo.

    Labelling the final quarter 0 because its future has not happened yet would
    teach the model that the most recent quarter is always safe - precisely the
    rows the live model is asked to score.
    """
    frame = pd.DataFrame(
        {
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "adj_close": [100.0, 99.0, 98.0],
        }
    )
    labelled = drawdown.label_series(frame, horizon=60, threshold=-0.15)

    assert labelled["label_is_valid"].sum() == 0
    assert labelled["label"].isna().all(), "must be NULL, not 0"
    assert labelled["fwd_max_drawdown"].isna().all()


def test_partial_window_is_not_labelled():
    """A window that runs off the end of the series is not a short window."""
    prices = np.array([100.0, 90.0, 80.0])
    result = drawdown.forward_drawdown(prices, horizon=5)
    assert np.isnan(result).all()


# --------------------------------------------------------------------------- #
# 4. adjusted-price sanity                                                    #
# --------------------------------------------------------------------------- #
def test_unadjusted_split_would_fabricate_an_event():
    """sec.10's corporate-action trap, demonstrated.

    A 1:2 split in RAW prices looks like an exact -50% crash and fires the
    label. Correctly adjusted prices show the flat series it really was. This is
    why prices.py refuses to fall back to raw close.
    """
    raw = np.array([100.0, 100.0, 50.0, 50.0, 50.0])         # split, unadjusted
    adjusted = np.array([50.0, 50.0, 50.0, 50.0, 50.0])      # same economics

    raw_result = drawdown.forward_drawdown(raw, horizon=3)
    adjusted_result = drawdown.forward_drawdown(adjusted, horizon=3)

    assert raw_result[0] == pytest.approx(-0.50), "raw prices manufacture a fake -50% event"
    assert raw_result[0] <= -0.15, "which would be labelled a downside event"
    assert adjusted_result[0] == pytest.approx(0.0), "adjusted prices show no move at all"
    assert adjusted_result[0] > -0.15, "and correctly produce no event"


def test_flat_series_produces_no_event():
    result = drawdown.forward_drawdown(np.full(10, 100.0), horizon=5)
    assert np.nanmax(np.abs(result)) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# summary helper                                                              #
# --------------------------------------------------------------------------- #
def test_summarise_reports_event_rate_over_valid_rows_only():
    frame = pd.DataFrame(
        {
            "label_is_valid": [1, 1, 1, 1, 0],
            "label": pd.array([1, 0, 1, 0, None], dtype="Int64"),
            "fwd_max_drawdown": [-0.30, -0.05, -0.20, 0.10, np.nan],
        }
    )
    summary = drawdown.summarise(frame)

    assert summary["n"] == 5
    assert summary["n_valid"] == 4
    assert summary["n_events"] == 2
    assert summary["event_rate"] == pytest.approx(0.5)
    assert summary["drawdown_median"] == pytest.approx(-0.125)
