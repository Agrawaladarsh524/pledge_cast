"""Walk-forward validation - PLAN.md sec.9.4.

    train Q1..Q8   -> test Q9
    train Q1..Q9   -> test Q10
                        :

    "Standard k-fold cross-validation is FORBIDDEN here. Random folds would
     place future quarters in the training set."

    EMBARGO: "the final quarter can be featured but never labelled - its label
     needs 60 trading days of future prices that don't exist yet."

**On the fold count.** sec.1.3 says "~8 folds" while the sec.9.4 diagram implies
11 (test on Q9..Q19). Both are right, for different definitions of a usable
quarter. 20 quarters minus the embargo leaves 19 labelled; the first FULLY
featured quarter is the 4th, because pledge_accel and pledge_max_4q need four
quarters of history. Requiring 8 fully-featured training quarters gives test
folds Q12..Q19 - exactly the 8 sec.1.3 states.

Rather than pick a reading, the count is derived from config
(``walkforward.min_train_quarters``) and logged, so it follows the data instead
of a hardcoded number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from pledgecast.exceptions import InsufficientDataError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One expanding-window split, addressed by observation date."""

    index: int
    train_dates: list[str]
    test_date: str
    n_train: int = 0
    n_test: int = 0
    n_train_events: int = 0
    n_test_events: int = 0

    @property
    def test_dates(self) -> list[str]:
        return [self.test_date]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.index,
            "train_dates": list(self.train_dates),
            "test_dates": self.test_dates,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_train_events": self.n_train_events,
            "n_test_events": self.n_test_events,
        }


@dataclass
class FoldPlan:
    folds: list[Fold] = field(default_factory=list)
    labelled_dates: list[str] = field(default_factory=list)
    embargoed_dates: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.folds)


def generate_folds(
    panel: pd.DataFrame,
    *,
    min_train_quarters: int,
    date_column: str = "observation_date",
    label_column: str = "label",
    valid_column: str = "label_is_valid",
) -> FoldPlan:
    """Expanding-window folds over the labelled observation dates.

    The embargo is structural rather than a parameter: a date is trainable only
    if it carries valid labels, and the final quarter has none because its
    forward window has not elapsed. Nothing needs to subtract it by hand.
    """
    if panel.empty:
        raise InsufficientDataError("cannot build folds from an empty panel")

    all_dates = sorted(panel[date_column].unique())
    labelled = panel[panel[valid_column] == 1]
    labelled_dates = sorted(labelled[date_column].unique())
    embargoed = [d for d in all_dates if d not in set(labelled_dates)]

    if len(labelled_dates) <= min_train_quarters:
        raise InsufficientDataError(
            f"{len(labelled_dates)} labelled dates but min_train_quarters="
            f"{min_train_quarters}; no fold can be formed"
        )

    counts = labelled.groupby(date_column).agg(
        n=(label_column, "size"), events=(label_column, "sum")
    )

    folds: list[Fold] = []
    for position in range(min_train_quarters, len(labelled_dates)):
        train_dates = labelled_dates[:position]
        test_date = labelled_dates[position]
        folds.append(
            Fold(
                index=len(folds),
                train_dates=train_dates,
                test_date=test_date,
                n_train=int(counts.loc[train_dates, "n"].sum()),
                n_test=int(counts.loc[test_date, "n"]),
                n_train_events=int(counts.loc[train_dates, "events"].sum()),
                n_test_events=int(counts.loc[test_date, "events"]),
            )
        )

    logger.info(
        "walk-forward: %d folds over %d labelled dates (%d embargoed: %s)",
        len(folds),
        len(labelled_dates),
        len(embargoed),
        embargoed,
    )
    return FoldPlan(folds=folds, labelled_dates=labelled_dates, embargoed_dates=embargoed)


def split(panel: pd.DataFrame, fold: Fold, date_column: str = "observation_date"):
    """``(train, test)`` frames for one fold. Labelled rows only."""
    labelled = panel[panel["label_is_valid"] == 1]
    train = labelled[labelled[date_column].isin(fold.train_dates)]
    test = labelled[labelled[date_column] == fold.test_date]
    return train, test


def describe(plan: FoldPlan) -> pd.DataFrame:
    """Fold table for the training report."""
    return pd.DataFrame(
        [
            {
                "fold": f.index,
                "train_through": f.train_dates[-1],
                "test_date": f.test_date,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "train_event_rate": f.n_train_events / f.n_train if f.n_train else None,
                "test_event_rate": f.n_test_events / f.n_test if f.n_test else None,
            }
            for f in plan.folds
        ]
    )


__all__ = ["Fold", "FoldPlan", "describe", "generate_folds", "split"]
