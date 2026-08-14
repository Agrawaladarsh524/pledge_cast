"""The sec.13.1 warnings contract.

Warnings are the only channel through which a score admits it is weakly founded,
so their selection has to be exact. It was not: both dashboard pages, and the
scoring script, picked warnings out of the list by POSITION - filtering on
``len(warnings) > 1`` and slicing ``[:-1]`` - on the assumption that the "no
realised outcome yet" notice is always the last element.

It is appended only when ``label_is_valid == 0``. On the embargo quarter the
assumption holds; on every labelled date it does not, so a company whose only
warning was a forward-filled pledge state failed the ``> 1`` test and disappeared,
and a company with two warnings lost the median-imputation message - the one the
service's own docstring calls the more serious.

Measured on the panel before the fix: 653 data-quality warnings never reached the
dashboard, across all 19 labelled dates.

The fix moves classification into the service, where the warnings are built, and
gives consumers a ``data_warnings`` column to select instead of a list to slice.
These tests pin that contract down so it cannot regress into position again.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pledgecast.inference.service import (
    DATA_QUALITY_CODES,
    IMPUTED_FEATURES,
    NO_REALISED_OUTCOME,
    STALE_PLEDGE,
    PredictionService,
)

FEATURES = ["volatility_90d", "log_turnover_90d"]


@pytest.fixture
def service(settings) -> PredictionService:
    """No connection: warning construction needs settings and a row, nothing else."""
    return PredictionService(settings)


def row(*, stale: int = 0, labelled: int = 1, blank: list[str] | None = None) -> pd.Series:
    values: dict[str, object] = {
        "is_stale": stale,
        "label_is_valid": labelled,
        "volatility_90d": 0.42,
        "log_turnover_90d": 18.1,
    }
    for feature in blank or []:
        values[feature] = None
    return pd.Series(values)


def codes(service: PredictionService, series: pd.Series) -> list[str]:
    return [code for code, _ in service._warning_items(series, FEATURES)]


# --------------------------------------------------------------------------- #
# what each condition produces                                                 #
# --------------------------------------------------------------------------- #
def test_a_clean_labelled_row_warns_about_nothing(service):
    assert codes(service, row()) == []
    assert service._warnings(row(), FEATURES) == []
    assert service._data_warnings(row(), FEATURES) == []


def test_forward_filled_pledge_state_is_a_data_warning(service):
    assert codes(service, row(stale=1)) == [STALE_PLEDGE]
    assert len(service._data_warnings(row(stale=1), FEATURES)) == 1


def test_imputed_features_are_a_data_warning(service):
    series = row(blank=["volatility_90d"])
    assert codes(service, series) == [IMPUTED_FEATURES]
    assert len(service._data_warnings(series, FEATURES)) == 1


def test_the_embargo_notice_is_not_a_data_warning(service):
    """It describes the calendar, not the row. Reported once per page, not 300 times."""
    series = row(labelled=0)
    assert codes(service, series) == [NO_REALISED_OUTCOME]
    assert service._warnings(series, FEATURES) != []
    assert service._data_warnings(series, FEATURES) == []


def test_no_realised_outcome_is_excluded_from_the_data_quality_set():
    assert NO_REALISED_OUTCOME not in DATA_QUALITY_CODES
    assert set(DATA_QUALITY_CODES) == {STALE_PLEDGE, IMPUTED_FEATURES}


# --------------------------------------------------------------------------- #
# the bug itself, pinned                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.critical
def test_a_single_data_warning_on_a_labelled_date_is_never_dropped(service):
    """The exact case that vanished: one warning, on a labelled date.

    The old filter required ``len(warnings) > 1``. A labelled row with a stale
    pledge state has exactly one, so it failed the test and was never rendered -
    on all 19 labelled dates.
    """
    series = row(stale=1, labelled=1)
    warnings = service._warnings(series, FEATURES)
    assert len(warnings) == 1, "precondition: this is the single-warning case"

    old_behaviour = warnings[:-1] if len(warnings) > 1 else []
    assert old_behaviour == [], "precondition: the old filter dropped it"
    assert service._data_warnings(series, FEATURES) == warnings


@pytest.mark.critical
def test_the_more_serious_warning_is_never_truncated_away(service):
    """The second case: two warnings, and ``[:-1]`` amputated the imputation one.

    Order matters to the old code and must not matter to the new code, so this
    asserts on content rather than on index.
    """
    series = row(stale=1, labelled=1, blank=["volatility_90d"])
    data = service._data_warnings(series, FEATURES)
    assert len(data) == 2
    assert any("forward-filled" in message for message in data)
    assert any("median-imputed" in message for message in data), (
        "the median-imputation warning is the one the service calls more serious; "
        "it was the one the old [:-1] slice removed"
    )


@pytest.mark.critical
def test_data_warnings_do_not_depend_on_position(service):
    """Whether the embargo notice is present must not change the data subset.

    This is the invariant the old positional logic violated: the same row on a
    labelled date and on an unlabelled one reported different data problems.
    """
    labelled = row(stale=1, labelled=1)
    embargo = row(stale=1, labelled=0)

    assert service._data_warnings(labelled, FEATURES) == service._data_warnings(
        embargo, FEATURES
    )
    assert len(service._warnings(labelled, FEATURES)) == 1
    assert len(service._warnings(embargo, FEATURES)) == 2


# --------------------------------------------------------------------------- #
# the published contract must not have changed shape                           #
# --------------------------------------------------------------------------- #
def test_the_api_warnings_array_is_still_a_list_of_strings(service):
    """sec.13.1 specifies a ``warnings`` array and api/schemas.py types it.

    Codes travel beside the messages precisely so this stays true.
    """
    series = row(stale=1, labelled=0, blank=["volatility_90d"])
    warnings = service._warnings(series, FEATURES)
    assert isinstance(warnings, list)
    assert all(isinstance(message, str) for message in warnings)
    assert len(warnings) == 3


def test_every_message_is_reachable_from_exactly_one_code(service):
    """No message may be duplicated across codes, or counts drift between views."""
    series = row(stale=1, labelled=0, blank=["volatility_90d"])
    items = service._warning_items(series, FEATURES)
    assert len({code for code, _ in items}) == len(items)
    assert [message for _, message in items] == service._warnings(series, FEATURES)
