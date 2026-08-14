"""Panel strata - restricting an experiment to the rows where its question means something.

**Why this exists.** The panel is built on the NIFTY 500 spine (sec.2.4), chosen
deliberately so the study is not conditioned on being pledged - selecting the
universe from the pledge list would have guaranteed a positive-looking result by
construction. That choice is right for the headline question and wrong for a
second one.

Measured on the built panel, 212 of the 300 companies never carry a promoter
pledge at all: 4,621 of 5,696 labelled rows have ``pledge_pct_promoter`` at or
near zero. Asking "does pledge trajectory predict crashes?" across all of them
is asking a question about pledging on a population that mostly does not pledge.
Every pledge feature is constant at zero there, so those rows can only dilute -
they contribute noise to the average and nothing to the comparison.

A stratum answers the sharper question: *among companies that actually carry a
pledge*, does pledge behaviour add anything? That is also the population a real
user of this system would be looking at.

**The trap this module is written to avoid.** A stratum changes the rows, so a
model trained on stratum rows and compared against a model trained on all rows
measures the population change and not the feature set. ``Settings`` rejects
that pairing at config load; this module only applies the filter, and reports
exactly what it removed so the reduction is auditable rather than silent.
"""

from __future__ import annotations

import pandas as pd

from pledgecast.exceptions import InsufficientDataError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


def apply_population(
    panel: pd.DataFrame,
    name: str,
    settings,
    *,
    min_rows: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """Restrict ``panel`` to the named stratum. Returns ``(subset, report)``.

    ``min_rows`` guards against a filter that silently empties the panel: an
    experiment on 3 rows would still produce an AUC, and that number would be
    meaningless while looking exactly like every other number in the table.
    """
    if name not in settings.populations:
        raise KeyError(f"unknown population '{name}'. Known: {sorted(settings.populations)}")

    spec = settings.populations[name]
    report = {
        "population": name,
        "description": spec.description,
        "rows_before": len(panel),
        "companies_before": int(panel["symbol"].nunique()) if not panel.empty else 0,
    }

    if spec.is_identity or panel.empty:
        report.update(
            rows_after=len(panel),
            companies_after=report["companies_before"],
            dropped=0,
            filter="none",
        )
        return panel, report

    if spec.column not in panel.columns:
        raise InsufficientDataError(
            f"population '{name}' filters on '{spec.column}', which is not a panel column"
        )

    values = panel[spec.column]
    keep = pd.Series(True, index=panel.index)
    bounds = []
    if spec.min_value is not None:
        keep &= values >= spec.min_value
        bounds.append(f">= {spec.min_value}")
    if spec.max_value is not None:
        keep &= values <= spec.max_value
        bounds.append(f"<= {spec.max_value}")

    # NaN comparisons are False, so a missing value is already excluded by the
    # bounds above. `include_missing` puts it back deliberately.
    if spec.include_missing:
        keep |= values.isna()

    subset = panel[keep]
    report.update(
        rows_after=len(subset),
        companies_after=int(subset["symbol"].nunique()) if not subset.empty else 0,
        dropped=len(panel) - len(subset),
        filter=f"{spec.column} {' and '.join(bounds) or 'any'}"
        + (" or missing" if spec.include_missing else ""),
    )

    labelled = int(subset.get("label_is_valid", pd.Series(dtype=float)).eq(1).sum())
    report["labelled_rows_after"] = labelled

    if labelled < min_rows:
        raise InsufficientDataError(
            f"population '{name}' leaves only {labelled} labelled rows "
            f"(minimum {min_rows}). Widen the filter or drop the stratum - an AUC on "
            "that few rows is not a measurement."
        )

    logger.info(
        "population '%s': %d -> %d rows (%d companies), filter [%s]",
        name,
        report["rows_before"],
        report["rows_after"],
        report["companies_after"],
        report["filter"],
    )
    return subset, report


def describe_populations(panel: pd.DataFrame, settings) -> pd.DataFrame:
    """One row per configured stratum - printed by ``04_train_all.py``.

    A stratum that removes 85% of the panel needs to say so on every run, not
    in a comment. The event rate column is the one that matters: if a filter
    also shifts the base rate a long way, the stratum is answering a different
    question and not just a narrower one.
    """
    rows = []
    for name in settings.populations:
        try:
            subset, report = apply_population(panel, name, settings, min_rows=0)
        except (KeyError, InsufficientDataError) as exc:  # pragma: no cover - config guards this
            rows.append({"population": name, "error": str(exc)})
            continue
        labelled = subset[subset["label_is_valid"] == 1] if not subset.empty else subset
        rows.append(
            {
                "population": name,
                "rows": report["rows_after"],
                "labelled": len(labelled),
                "companies": report["companies_after"],
                "event_rate": float(labelled["label"].mean()) if not labelled.empty else None,
                "dates": int(labelled["observation_date"].nunique()) if not labelled.empty else 0,
                "filter": report["filter"],
            }
        )
    return pd.DataFrame(rows)


__all__ = ["apply_population", "describe_populations"]
