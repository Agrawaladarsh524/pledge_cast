"""Do the Reg 31 results depend on the knobs I chose?

Three numbers in the event block were set by judgement rather than derived:
the 90-day window, the 365-day invocation window, and the 0.01% materiality
threshold. The threshold has a defence - it is the filing's own decimal
precision - but the windows do not. "You picked the wrong window" is the first
objection any reader will raise against a null result, and it deserves a table
rather than a paragraph.

So this module re-derives the event features at every window and every threshold
and reports what changes. Measured on this panel, nothing does: widening the
window from 90 to 730 days lifts coverage from 8.3% to 19.7% and leaves the
univariate within-quarter AUC between 0.485 and 0.507 throughout. The null
survives the sweep, which is a stronger statement than the null at one setting.

**This is a diagnostic, not a selection procedure.** Nothing here feeds back
into the configured window. Sweeping a parameter and keeping whichever value
scored best is how a null gets tuned into a finding - sec.9.9 forbids exactly
that. The sweep exists to show the result is flat, and it would be reported
just as prominently if it were not.
"""

from __future__ import annotations

import pandas as pd

from pledgecast.evaluation import metrics
from pledgecast.features.events import build_event_features, filter_material
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)

# Windows swept. 90 is configured; 730 is deliberately absurd for a "recent
# activity" feature and is included so the table brackets the plausible range.
DEFAULT_WINDOWS = (30, 90, 180, 365, 730)
# 0.01 is the filing precision; 0.0 is "no filter at all", which reproduces the
# contaminated feature the materiality rule exists to prevent.
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.10, 0.50)


def _univariate_auc(
    panel: pd.DataFrame, feature: pd.Series, *, min_rows: int
) -> float | None:
    """Within-quarter AUC of a single raw feature, no model involved.

    A model could always find nothing because it was badly fitted. A raw
    univariate AUC cannot - it is the feature's own ordering against the label,
    with no fitting to blame.
    """
    frame = pd.DataFrame(
        {
            "label": panel["label"].to_numpy(dtype=float),
            "score": pd.to_numeric(feature, errors="coerce").fillna(0.0).to_numpy(),
            "date": panel["observation_date"].to_numpy(),
        }
    )
    return metrics.within_quarter_auc(
        frame["label"], frame["score"], frame["date"], min_rows=min_rows
    )


def window_sweep(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    settings,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    features: tuple[str, ...] = ("event_count_90d", "event_net_90d", "event_created_90d"),
) -> pd.DataFrame:
    """Rebuild the event block at each window; report coverage and univariate AUC.

    ``panel`` must be the labelled rows only - an unlabelled row has no label to
    score against and would be dropped by every AUC anyway.
    """
    material, _ = filter_material(events, settings.features.min_event_pct_equity)
    observations = panel[["symbol", "observation_date"]].copy()
    min_rows = settings.evaluation.min_rows_per_quarter_for_auc

    rows = []
    for window in windows:
        built = build_event_features(
            material,
            observations,
            window_days=window,
            invocation_window_days=settings.features.event_invocation_window_days,
            disclosure_lag_days=settings.features.event_disclosure_lag_days,
        ).reset_index(drop=True)

        row = {
            "window_days": window,
            "coverage": float((built["event_count_90d"] > 0).mean()),
            "configured": window == settings.features.event_window_days,
        }
        for feature in features:
            row[feature.replace("event_", "").replace("_90d", "")] = _univariate_auc(
                panel.reset_index(drop=True), built[feature], min_rows=min_rows
            )
        rows.append(row)

    return pd.DataFrame(rows)


def materiality_sweep(
    events: pd.DataFrame,
    panel: pd.DataFrame,
    settings,
    *,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Vary the materiality cut. Shows what the filter buys and what it costs.

    The ``0.0`` row is the important one: it is the unfiltered feature, and it
    is where the CDSL-style clearing disclosures come back. Its coverage jumps
    while its AUC does not, which is the signature of a feature that measures
    filing volume rather than pledging.
    """
    observations = panel[["symbol", "observation_date"]].copy()
    min_rows = settings.evaluation.min_rows_per_quarter_for_auc

    rows = []
    for threshold in thresholds:
        material, report = filter_material(events, threshold)
        built = build_event_features(
            material,
            observations,
            window_days=settings.features.event_window_days,
            invocation_window_days=settings.features.event_invocation_window_days,
            disclosure_lag_days=settings.features.event_disclosure_lag_days,
        ).reset_index(drop=True)

        rows.append(
            {
                "min_pct_equity": threshold,
                "events_kept": report.get("material", 0),
                "events_dropped": report.get("dropped", 0),
                "companies": report.get("companies_material", 0),
                "coverage": float((built["event_count_90d"] > 0).mean()),
                "count_auc": _univariate_auc(
                    panel.reset_index(drop=True), built["event_count_90d"], min_rows=min_rows
                ),
                "net_auc": _univariate_auc(
                    panel.reset_index(drop=True), built["event_net_90d"], min_rows=min_rows
                ),
                "configured": threshold == settings.features.min_event_pct_equity,
            }
        )

    return pd.DataFrame(rows)


def univariate_table(panel: pd.DataFrame, features: list[str], settings) -> pd.DataFrame:
    """Every feature's own within-quarter AUC on whatever rows it is given.

    Run on a stratum this is the most direct answer available to "which of these
    actually separates?", with no model, no hyperparameters and no folds to
    argue about. An AUC below 0.5 is not a failure - it is the same strength of
    ordering pointing the other way, so both the value and its mirror are
    reported.
    """
    min_rows = settings.evaluation.min_rows_per_quarter_for_auc
    frame = panel.reset_index(drop=True)

    rows = []
    for feature in features:
        if feature not in frame.columns:
            continue
        values = pd.to_numeric(frame[feature], errors="coerce")
        auc = _univariate_auc(frame, values, min_rows=min_rows)
        if auc is None:
            continue
        rows.append(
            {
                "feature": feature,
                "auc": auc,
                # Direction-free strength: 0.44 and 0.56 are equally informative.
                "auc_if_inverted": 1.0 - auc,
                "strength": abs(auc - 0.5),
                "coverage": float(values.notna().mean()),
                "nonzero": float((values.fillna(0) != 0).mean()),
            }
        )

    table = pd.DataFrame(rows)
    return table.sort_values("strength", ascending=False) if not table.empty else table


__all__ = [
    "DEFAULT_THRESHOLDS",
    "DEFAULT_WINDOWS",
    "materiality_sweep",
    "univariate_table",
    "window_sweep",
]
