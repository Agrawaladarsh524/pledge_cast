"""Sensitivity sweeps - and the discipline that they must not become tuning.

The sweeps exist to answer "you picked the wrong window" with a table. The
danger is that a sweep is one line away from a parameter search: run every
window, keep the one that scored best, and a null becomes a finding without
anybody deciding to cheat. sec.9.9 forbids that, so the property tested hardest
here is that the sweep is inert - it reports, and nothing reads its output back
into the configuration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pledgecast.evaluation import sensitivity


@pytest.fixture
def events() -> pd.DataFrame:
    """A mix of material events and 0.00% clearing noise, as the real table has."""
    rng = np.random.default_rng(0)
    rows = []
    for day in pd.bdate_range("2023-01-02", "2024-05-31"):
        date = day.strftime("%Y-%m-%d")
        symbol = f"S{rng.integers(0, 25):02d}"
        rows.append(
            {
                "symbol": symbol,
                "event_date": date,
                "event_type": "creation" if rng.random() < 0.6 else "release",
                "pct_equity": float(rng.uniform(0.02, 3.0)),
            }
        )
        # The CDSL shape - many disclosures, all rounding to zero.
        rows.append(
            {
                "symbol": "NOISE",
                "event_date": date,
                "event_type": "release",
                "pct_equity": 0.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def labelled() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for quarter in pd.date_range("2023-06-30", periods=4, freq="QE"):
        observation = (quarter + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        for index in range(30):
            rows.append(
                {
                    "symbol": f"S{index:02d}",
                    "observation_date": observation,
                    "label": float(rng.random() < 0.25),
                    "label_is_valid": 1,
                    "volatility_90d": float(rng.uniform(0.2, 0.9)),
                    "pledge_pct_promoter": float(rng.uniform(0, 40)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# window sweep                                                                 #
# --------------------------------------------------------------------------- #
def test_the_sweep_reports_one_row_per_window(events, labelled, settings):
    table = sensitivity.window_sweep(events, labelled, settings, windows=(30, 90, 365))
    assert len(table) == 3
    assert list(table["window_days"]) == [30, 90, 365]


def test_coverage_rises_monotonically_with_the_window(events, labelled, settings):
    """The point of the sweep: a wider window really does see more, so a flat
    AUC across it is evidence rather than an artefact of nothing changing."""
    table = sensitivity.window_sweep(events, labelled, settings, windows=(30, 90, 180, 365))
    coverage = table["coverage"].to_numpy()
    assert (np.diff(coverage) >= 0).all(), "a wider window saw fewer events"


def test_the_configured_window_is_marked_so_the_reader_can_find_it(events, labelled, settings):
    table = sensitivity.window_sweep(events, labelled, settings, windows=(30, 90, 365))
    assert table["configured"].sum() == 1
    assert table.loc[table["configured"], "window_days"].iloc[0] == (
        settings.features.event_window_days
    )


def test_the_sweep_does_not_mutate_the_configured_window(events, labelled, settings):
    """The discipline check: a sweep must not become a parameter search."""
    before = settings.features.event_window_days
    sensitivity.window_sweep(events, labelled, settings, windows=(30, 730))
    assert settings.features.event_window_days == before


def test_the_sweep_does_not_mutate_the_panel_it_was_given(events, labelled, settings):
    snapshot = labelled.copy()
    sensitivity.window_sweep(events, labelled, settings, windows=(90,))
    pd.testing.assert_frame_equal(labelled, snapshot)


# --------------------------------------------------------------------------- #
# materiality sweep                                                            #
# --------------------------------------------------------------------------- #
def test_a_higher_threshold_keeps_strictly_fewer_events(events, labelled, settings):
    table = sensitivity.materiality_sweep(events, labelled, settings, thresholds=(0.0, 0.01, 0.5))
    kept = table["events_kept"].to_numpy()
    assert (np.diff(kept) <= 0).all()


def test_the_zero_threshold_row_readmits_the_clearing_noise(events, labelled, settings):
    """The row that shows what the materiality filter is for."""
    table = sensitivity.materiality_sweep(events, labelled, settings, thresholds=(0.0, 0.01))
    unfiltered = table[table["min_pct_equity"] == 0.0].iloc[0]
    filtered = table[table["min_pct_equity"] == 0.01].iloc[0]

    assert unfiltered["events_kept"] > filtered["events_kept"]
    assert unfiltered["companies"] > filtered["companies"], "the 0.00% filer survived the filter"


def test_dropped_plus_kept_always_equals_the_whole_table(events, labelled, settings):
    table = sensitivity.materiality_sweep(events, labelled, settings, thresholds=(0.0, 0.01, 0.5))
    assert (table["events_kept"] + table["events_dropped"] == len(events)).all()


# --------------------------------------------------------------------------- #
# univariate table                                                             #
# --------------------------------------------------------------------------- #
def test_a_feature_that_is_the_label_scores_a_perfect_auc(labelled, settings):
    frame = labelled.copy()
    frame["volatility_90d"] = frame["label"]
    table = sensitivity.univariate_table(frame, ["volatility_90d"], settings)
    assert table.iloc[0]["auc"] == pytest.approx(1.0)


def test_an_inverted_feature_is_reported_as_equally_strong_not_as_useless(labelled, settings):
    """0.44 and 0.56 order the data equally well, in opposite directions.

    Reporting only the raw AUC would bury the strongest event-side result in the
    study - event_net_90d at 0.438 on the pledged stratum.
    """
    frame = labelled.copy()
    frame["volatility_90d"] = 1.0 - frame["label"]
    row = sensitivity.univariate_table(frame, ["volatility_90d"], settings).iloc[0]

    assert row["auc"] == pytest.approx(0.0)
    assert row["auc_if_inverted"] == pytest.approx(1.0)
    assert row["strength"] == pytest.approx(0.5)


def test_the_table_is_ordered_by_strength_not_by_raw_auc(labelled, settings):
    frame = labelled.copy()
    frame["volatility_90d"] = 1.0 - frame["label"]  # perfectly inverted, strength 0.5
    frame["pledge_pct_promoter"] = 0.5  # constant, strength 0
    table = sensitivity.univariate_table(
        frame, ["pledge_pct_promoter", "volatility_90d"], settings
    )
    assert table.iloc[0]["feature"] == "volatility_90d"


def test_a_constant_feature_scores_a_coin_flip(labelled, settings):
    frame = labelled.copy()
    frame["volatility_90d"] = 7.0
    assert sensitivity.univariate_table(frame, ["volatility_90d"], settings).iloc[0][
        "auc"
    ] == pytest.approx(0.5)


def test_missing_features_are_skipped_rather_than_crashing(labelled, settings):
    table = sensitivity.univariate_table(labelled, ["volatility_90d", "not_a_column"], settings)
    assert set(table["feature"]) == {"volatility_90d"}


def test_coverage_and_nonzero_are_reported_separately(labelled, settings):
    """A feature can be 100% present and 92% zero - the event block is exactly
    that, and conflating the two would hide its sparsity."""
    frame = labelled.copy()
    frame["volatility_90d"] = 0.0
    frame.loc[frame.index[:10], "volatility_90d"] = 1.0
    row = sensitivity.univariate_table(frame, ["volatility_90d"], settings).iloc[0]

    assert row["coverage"] == pytest.approx(1.0)
    assert row["nonzero"] < 0.15


def test_the_public_surface_is_only_what_production_calls():
    """`event_feature_names` was removed for having no caller."""
    assert set(sensitivity.__all__) == {
        "DEFAULT_THRESHOLDS",
        "DEFAULT_WINDOWS",
        "materiality_sweep",
        "univariate_table",
        "window_sweep",
    }
