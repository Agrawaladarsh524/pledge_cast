"""Row validation - PLAN.md sec.8, sec.10.

    "Pydantic row models + range/duplicate/null checks"

sec.4 justifies pydantic here by reuse: "One validation library for three jobs" -
config, API schemas, and these row models. The same range rules that reject a
bad API request also reject a bad parsed filing, so the two cannot drift.

The frame-level helpers below are what the pipeline actually calls; the pydantic
models exist for single-row validation at the API boundary (sec.10 "Invalid
input types -> HTTP 422 with field-level detail").
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from pledgecast.exceptions import ValidationError
from pledgecast.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# row models                                                                  #
# --------------------------------------------------------------------------- #
class PledgeStateRow(BaseModel):
    """One parsed filing, as stored in ``pledge_state``."""

    symbol: str = Field(min_length=1)
    quarter_end: str
    submission_date: str
    promoter_shares: float | None = Field(default=None, ge=0)
    pledged_shares: float | None = Field(default=None, ge=0)
    total_shares: float | None = Field(default=None, ge=0)
    promoter_holding_pct: float | None = Field(default=None, ge=0, le=100)
    pledge_pct_promoter: float | None = Field(default=None, ge=0, le=100)
    pledge_pct_equity: float | None = Field(default=None, ge=0, le=100)
    pledge_status: str
    filing_id: int | None = None

    @field_validator("pledge_status")
    @classmethod
    def _known_status(cls, v: str) -> str:
        allowed = {"PLEDGE_PRESENT", "NO_PLEDGE", "UNAVAILABLE"}
        if v not in allowed:
            raise ValueError(f"pledge_status must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("quarter_end", "submission_date")
    @classmethod
    def _iso_date(cls, v: str) -> str:
        try:
            pd.Timestamp(v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"not an ISO date: {v!r}") from exc
        return v


class PanelRow(BaseModel):
    """One ML-ready panel row."""

    symbol: str = Field(min_length=1)
    observation_date: str
    quarter_end: str
    promoter_holding_pct: float | None = Field(default=None, ge=0, le=100)
    pledge_pct_promoter: float | None = Field(default=None, ge=0, le=100)
    pledge_pct_equity: float | None = Field(default=None, ge=0, le=100)
    volatility_90d: float | None = Field(default=None, ge=0)
    label: int | None = Field(default=None, ge=0, le=1)
    label_is_valid: int = Field(ge=0, le=1)


# --------------------------------------------------------------------------- #
# frame-level checks                                                          #
# --------------------------------------------------------------------------- #
def check_ranges(frame: pd.DataFrame, settings) -> list[dict]:
    """sec.10: pledge_pct in [0,100], volatility > 0, probability in [0,1]."""
    issues: list[dict] = []
    bounds: dict[str, tuple[float, float]] = {
        "promoter_holding_pct": (
            settings.validation.pledge_pct_min,
            settings.validation.pledge_pct_max,
        ),
        "pledge_pct_promoter": (
            settings.validation.pledge_pct_min,
            settings.validation.pledge_pct_max,
        ),
        "pledge_pct_equity": (
            settings.validation.pledge_pct_min,
            settings.validation.pledge_pct_max,
        ),
        "pledge_max_4q": (settings.validation.pledge_pct_min, settings.validation.pledge_pct_max),
        "volatility_90d": (settings.validation.min_volatility, np.inf),
        "trailing_dd_60d": (-1.0, 0.0),
        "fwd_max_drawdown": (-1.0, np.inf),
    }

    for column, (low, high) in bounds.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        outside = values.notna() & ((values < low) | (values > high))
        if outside.any():
            issues.append(
                {
                    "column": column,
                    "rule": f"[{low}, {high}]",
                    "n_violations": int(outside.sum()),
                    "min": float(values[outside].min()),
                    "max": float(values[outside].max()),
                }
            )
    return issues


def check_duplicates(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
    """Composite keys make duplicates impossible in SQLite - catch them earlier."""
    present = [k for k in keys if k in frame.columns]
    if len(present) != len(keys):
        return []
    duplicated = frame.duplicated(subset=present, keep=False)
    if not duplicated.any():
        return []
    return [
        {
            "rule": f"unique on {present}",
            "n_violations": int(duplicated.sum()),
            "sample": frame.loc[duplicated, present].head(5).to_dict("records"),
        }
    ]


def check_nulls(frame: pd.DataFrame, required: list[str]) -> list[dict]:
    """Columns that must never be null, whatever the data quality."""
    issues = []
    for column in required:
        if column not in frame.columns:
            issues.append({"column": column, "rule": "present", "n_violations": len(frame)})
            continue
        n_null = int(frame[column].isna().sum())
        if n_null:
            issues.append({"column": column, "rule": "not null", "n_violations": n_null})
    return issues


def check_dtypes(frame: pd.DataFrame, settings) -> list[dict]:
    """sec.10 "Incorrect feature types": explicit dtype enforcement, fail loudly."""
    issues = []
    for column in settings.features.all_features:
        if column not in frame.columns:
            issues.append({"column": column, "rule": "feature present", "n_violations": 1})
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            issues.append(
                {
                    "column": column,
                    "rule": "numeric dtype",
                    "n_violations": 1,
                    "actual": str(frame[column].dtype),
                }
            )
    return issues


def validate_panel(frame: pd.DataFrame, settings, *, strict: bool = True) -> dict[str, Any]:
    """Run every panel check. Raises ``ValidationError`` on failure if strict."""
    issues = (
        check_nulls(frame, ["symbol", "observation_date", "quarter_end", "label_is_valid"])
        + check_duplicates(frame, ["symbol", "observation_date"])
        + check_ranges(frame, settings)
        + check_dtypes(frame, settings)
    )

    report = {"n_rows": len(frame), "n_issues": len(issues), "issues": issues, "passed": not issues}

    for issue in issues:
        logger.error("validation: %s", issue)

    if issues and strict:
        raise ValidationError(f"panel validation failed with {len(issues)} issue(s): {issues[:3]}")
    return report


__all__ = [
    "PanelRow",
    "PledgeStateRow",
    "check_dtypes",
    "check_duplicates",
    "check_nulls",
    "check_ranges",
    "validate_panel",
]
