"""Panel strata, and the config rule that stops them being compared wrongly.

A stratum is a small feature with one large way to go wrong: comparing a model
trained on the stratum against a model trained on the full panel. The delta then
measures the population change - here, dropping 85% of the rows - and not the
feature set under test. Nothing about the resulting number looks wrong, which is
why the check lives in config validation and fails at load rather than four runs
later.

The other trap is quieter. Rows whose filter column is missing must NOT land in
a stratum defined by that very column: "we do not know whether this company is
pledged" is not "this company is pledged".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from config import Settings
from pledgecast.data.population import apply_population, describe_populations
from pledgecast.exceptions import InsufficientDataError


@pytest.fixture
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # 300 rows so the pledged third clears apply_population's 50-row floor -
    # a fixture below it exercises the guard rather than the filter.
    for index in range(300):
        rows.append(
            {
                "symbol": f"S{index % 20:02d}",
                "observation_date": f"2024-{(index % 6) + 1:02d}-01",
                # A third pledged, a third at zero, a third unknown.
                "pledge_pct_promoter": (
                    float(rng.uniform(5, 40))
                    if index % 3 == 0
                    else (0.0 if index % 3 == 1 else np.nan)
                ),
                "volatility_90d": float(rng.uniform(0.2, 0.8)),
                "label": float(index % 4 == 0),
                "label_is_valid": 1,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# applying a stratum                                                           #
# --------------------------------------------------------------------------- #
def test_the_all_population_returns_the_panel_untouched(panel, settings):
    subset, report = apply_population(panel, "all", settings)
    assert len(subset) == len(panel)
    assert report["dropped"] == 0
    assert report["filter"] == "none"


def test_the_pledged_stratum_keeps_only_rows_above_the_threshold(panel, settings):
    subset, report = apply_population(panel, "pledged", settings)
    assert (subset["pledge_pct_promoter"] >= 1.0).all()
    assert report["rows_after"] < report["rows_before"]
    assert report["dropped"] > 0


def test_a_row_with_an_unknown_pledge_is_excluded_rather_than_assumed(panel, settings):
    """The quiet trap: NaN must not silently join a stratum defined by pledging."""
    subset, _ = apply_population(panel, "pledged", settings)
    assert subset["pledge_pct_promoter"].notna().all()


def test_include_missing_puts_unknown_rows_back_when_asked(panel, settings):
    spec = settings.populations["pledged"].model_copy(update={"include_missing": True})
    patched = settings.model_copy(update={"populations": {**settings.populations, "x": spec}})

    subset, _ = apply_population(panel, "x", patched)
    assert subset["pledge_pct_promoter"].isna().any()


def test_a_filter_that_empties_the_panel_raises_rather_than_scoring_three_rows(panel, settings):
    """An AUC on 3 rows is not a measurement, but it looks exactly like one."""
    spec = settings.populations["pledged"].model_copy(update={"min_value": 999.0})
    patched = settings.model_copy(update={"populations": {**settings.populations, "x": spec}})

    with pytest.raises(InsufficientDataError, match="not a measurement"):
        apply_population(panel, "x", patched)


def test_an_unknown_population_name_fails_loudly(panel, settings):
    with pytest.raises(KeyError, match="unknown population"):
        apply_population(panel, "nonexistent", settings)


def test_a_filter_column_absent_from_the_panel_is_an_error(settings):
    thin = pd.DataFrame({"symbol": ["A"], "observation_date": ["2024-01-01"], "label_is_valid": [1]})
    with pytest.raises(InsufficientDataError, match="not a panel column"):
        apply_population(thin, "pledged", settings)


def test_describe_reports_every_stratum_with_its_own_event_rate(panel, settings):
    table = describe_populations(panel, settings)
    assert set(table["population"]) == set(settings.populations)
    assert {"rows", "labelled", "companies", "event_rate", "filter"} <= set(table.columns)


def test_describe_does_not_raise_on_a_stratum_too_small_to_train(panel, settings):
    """The description is a diagnostic - it must show a tiny stratum, not refuse."""
    spec = settings.populations["pledged"].model_copy(update={"min_value": 999.0})
    patched = settings.model_copy(update={"populations": {**settings.populations, "x": spec}})
    table = describe_populations(panel, patched)
    assert (table["population"] == "x").any()


# --------------------------------------------------------------------------- #
# THE config rule                                                              #
# --------------------------------------------------------------------------- #
def _config_dict() -> dict:
    from config import CONFIG_PATH

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _settings_from(raw: dict, tmp_path) -> Settings:
    """Build Settings from ``raw`` ALONE, with the real config.yaml out of the way.

    ``Settings(**raw)`` is not enough for these tests. pydantic-settings merges
    its sources per key, so a key deleted from ``raw`` is quietly restored from
    the real config.yaml and a validator guarding against its absence can never
    fire. Modifying a value works; removing one does not. Pointing a subclass at
    a temporary file removes the merge entirely.
    """
    from pydantic_settings import SettingsConfigDict

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    class Isolated(Settings):
        model_config = SettingsConfigDict(
            yaml_file=path,
            yaml_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    return Isolated()


def test_comparing_across_two_populations_is_rejected_at_config_load():
    """The mistake most likely to be made when someone adds a stratum later."""
    raw = _config_dict()
    raw["experiments"]["expG_pledged_full"]["baseline"] = "exp0_null"  # full-panel null

    with pytest.raises(ValueError, match="measures the population change"):
        Settings(**raw)


def test_an_experiment_naming_an_undefined_population_is_rejected():
    raw = _config_dict()
    raw["experiments"]["expB_full"]["population"] = "imaginary"
    with pytest.raises(ValueError, match="undefined population"):
        Settings(**raw)


def test_an_experiment_naming_an_undefined_baseline_is_rejected():
    raw = _config_dict()
    raw["experiments"]["expB_full"]["baseline"] = "imaginary"
    with pytest.raises(ValueError, match="undefined baseline"):
        Settings(**raw)


def test_a_population_filtering_on_a_non_feature_column_is_rejected():
    raw = _config_dict()
    raw["populations"]["pledged"]["column"] = "not_a_feature"
    with pytest.raises(ValueError, match="not a known feature"):
        Settings(**raw)


def test_the_all_population_is_mandatory(tmp_path):
    raw = _config_dict()
    raw["populations"].pop("all")
    with pytest.raises(ValueError, match="must define 'all'"):
        _settings_from(raw, tmp_path)


def test_inverted_bounds_are_rejected():
    raw = _config_dict()
    raw["populations"]["pledged"]["min_value"] = 50.0
    raw["populations"]["pledged"]["max_value"] = 10.0
    with pytest.raises(ValueError, match="min_value above max_value"):
        Settings(**raw)


def test_every_shipped_experiment_resolves_to_a_baseline_in_its_own_stratum(settings):
    """Guards the shipped config itself, not just the validator."""
    for name in settings.experiments:
        baseline = settings.experiment_baseline(name)
        assert settings.experiment_population(baseline) == settings.experiment_population(name)
