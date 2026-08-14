"""Typed configuration for PledgeCast.

PLAN.md sec.4: "pydantic-settings - Typed settings from YAML + .env.
                Config as a typed object, not a dict."

Precedence (highest first):  environment vars  ->  .env  ->  config.yaml

Every value the pipeline can tune lives in ``config.yaml``; this module only
gives it types, validation and path resolution. Import the singleton::

    from config import get_settings
    settings = get_settings()
    settings.label.drawdown_threshold      # -0.15, typed float
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _absolutise(value: Path) -> Path:
    """Resolve a config path against the repo root so cwd never matters."""
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


# --------------------------------------------------------------------------- #
# Section models                                                              #
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    raw_xbrl_dir: Path
    quarantine_dir: Path
    models_dir: Path
    logs_dir: Path
    figures_dir: Path
    universe_csv: Path

    @field_validator("*", mode="after")
    @classmethod
    def _abs(cls, v: Path) -> Path:
        return _absolutise(v)

    def ensure_exist(self) -> None:
        """Create every configured directory. Idempotent."""
        for field in ("raw_xbrl_dir", "quarantine_dir", "models_dir", "logs_dir", "figures_dir"):
            getattr(self, field).mkdir(parents=True, exist_ok=True)
        self.universe_csv.parent.mkdir(parents=True, exist_ok=True)


class WindowConfig(BaseModel):
    first_quarter_end: str
    last_quarter_end: str
    expected_quarters: int = Field(gt=0)
    api_from_date: str
    api_to_date: str
    standard_quarters_only: bool = True

    def quarter_ends(self) -> list[str]:
        """Every calendar quarter end in the window, as ISO strings.

        This is the canonical spine the panel is built on: one observation date
        per quarter, shared by every company, which is what within-quarter
        evaluation requires (sec.9.6).
        """
        from datetime import date

        first, last = (
            date.fromisoformat(self.first_quarter_end),
            date.fromisoformat(self.last_quarter_end),
        )
        out: list[str] = []
        for year in range(first.year, last.year + 1):
            for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
                candidate = date(year, month, day)
                if first <= candidate <= last:
                    out.append(candidate.isoformat())
        return sorted(out)


class UniverseConfig(BaseModel):
    index_name: str
    constituents_csv_url: str
    fallback_api_path: str
    target_size: int = Field(gt=0)
    min_filings_required: int = Field(ge=0)


class PointInTimeConfig(BaseModel):
    """sec.9.3 - the single rule that prevents leakage."""

    observation_lag_days: int = Field(gt=0)
    roll_to_next_trading_day: bool


class LabelConfig(BaseModel):
    horizon_trading_days: int = Field(gt=0)
    drawdown_threshold: float
    expected_event_rate: float = Field(gt=0, lt=1)
    event_rate_tolerance: float = Field(gt=0, lt=1)

    @field_validator("drawdown_threshold")
    @classmethod
    def _must_be_negative(cls, v: float) -> float:
        if v >= 0:
            raise ValueError("drawdown_threshold must be negative (e.g. -0.15)")
        return v


class FeaturesConfig(BaseModel):
    volatility_window_days: int = Field(gt=0)
    trailing_drawdown_window_days: int = Field(gt=0)
    return_window_days: int = Field(gt=0)
    turnover_window_days: int = Field(gt=0)
    pledge_rolling_max_quarters: int = Field(gt=0)
    trading_days_per_year: int = Field(gt=0)
    max_forward_fill_quarters: int = Field(ge=0)
    min_quarters_per_company: int = Field(gt=0)
    pledge_features: list[str]
    market_features: list[str]
    pledge_static_features: list[str]

    # Reg 31 event block - an extension beyond sec.9.1's 13. Defaults keep the
    # original 13-feature study loadable from a config that predates this.
    event_window_days: int = Field(default=90, gt=0)
    event_invocation_window_days: int = Field(default=365, gt=0)
    event_disclosure_lag_days: int = Field(default=11, ge=0)
    min_event_pct_equity: float = Field(default=0.01, ge=0)
    event_features: list[str] = Field(default_factory=list)

    @property
    def core_features(self) -> list[str]:
        """The 13 of sec.9.1, pledge block first."""
        return [*self.pledge_features, *self.market_features]

    @property
    def all_features(self) -> list[str]:
        """Every feature an experiment may reference, event block last."""
        return [*self.pledge_features, *self.market_features, *self.event_features]


class ExperimentConfig(BaseModel):
    description: str
    features: list[str]


class HeadlineConfig(BaseModel):
    """sec.2.3 - HEADLINE = experiment.metric - baseline.metric."""

    experiment: str
    baseline: str
    metric: str


class WalkForwardConfig(BaseModel):
    min_train_quarters: int = Field(gt=0)
    embargo_quarters: int = Field(ge=0)
    expanding_window: bool


class TrainingConfig(BaseModel):
    random_seed: int
    search_n_points: int = Field(gt=0)
    search_fold_index: int = Field(ge=0)
    search_model: str
    # sec.9.7 rule 2 needs a width, or "ties broken toward the simpler model"
    # can never fire against floating-point AUCs.
    selection_tie_tolerance: float = Field(ge=0, lt=0.5)


class ModelSpec(BaseModel):
    estimator: str
    params: dict[str, Any]
    requires_imputation: bool
    requires_scaling: bool
    search_space: dict[str, list[Any]] = Field(default_factory=dict)


class PreprocessingConfig(BaseModel):
    winsorize_lower_quantile: float = Field(ge=0, lt=0.5)
    winsorize_upper_quantile: float = Field(gt=0.5, le=1)
    imputation_strategy: Literal["median", "mean", "most_frequent"]
    add_missing_indicator: bool


class EvaluationConfig(BaseModel):
    primary_metric: str
    precision_at_k: int = Field(gt=0)
    n_quintiles: int = Field(gt=0)
    n_deciles: int = Field(gt=0)
    min_rows_per_quarter_for_auc: int = Field(gt=0)
    shuffle_test_tolerance: float = Field(gt=0, lt=0.5)
    risk_bands: dict[str, float]

    def band_for(self, probability: float) -> str:
        """Map a probability onto its configured risk band label."""
        for name, upper in sorted(self.risk_bands.items(), key=lambda kv: kv[1]):
            if probability < upper:
                return name
        return max(self.risk_bands.items(), key=lambda kv: kv[1])[0]


class ExplainConfig(BaseModel):
    # `model_for_shap` starts with "model_", which collides with pydantic's
    # protected namespace and would otherwise emit a warning on every import.
    model_config = ConfigDict(protected_namespaces=())

    model_for_shap: str
    top_n_features: int = Field(gt=0)
    beeswarm_max_display: int = Field(gt=0)
    summary_template_enabled: bool


class IngestConfig(BaseModel):
    base_url: str
    archives_url: str
    yahoo_chart_url: str
    benchmark_symbol: str
    benchmark_name: str
    price_range: str
    price_interval: str
    bootstrap_path: str
    user_agent: str
    accept_encoding: str
    default_referer: str
    pledge_referer: str
    endpoints: dict[str, str]
    max_workers: int = Field(gt=0)
    request_delay_seconds: float = Field(ge=0)
    timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0)
    backoff_factor: float = Field(gt=1)
    session_refresh_status_codes: list[int]
    skip_existing_files: bool

    @field_validator("accept_encoding")
    @classmethod
    def _no_brotli(cls, v: str) -> str:
        # Verified 2026-08-13: requests cannot decode Brotli; advertising 'br'
        # returns binary garbage from NSE.
        if "br" in [part.strip() for part in v.split(",")]:
            raise ValueError("accept_encoding must not advertise 'br' - requests cannot decode it")
        return v


class ValidationConfig(BaseModel):
    pledge_pct_min: float
    pledge_pct_max: float
    probability_min: float
    probability_max: float
    min_volatility: float
    corporate_action_return_floor: float
    min_price_rows_per_symbol: int = Field(gt=0)


class ApiConfig(BaseModel):
    title: str
    version: str
    host: str
    predictions_page_size: int = Field(gt=0)
    predictions_max_page_size: int = Field(gt=0)


class DashboardConfig(BaseModel):
    cache_ttl_seconds: int = Field(ge=0)
    default_top_n: int = Field(gt=0)
    max_top_n: int = Field(gt=0)


class LoggingConfig(BaseModel):
    format: str
    date_format: str
    file_name: str
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=0)
    console_enabled: bool
    file_enabled: bool


# --------------------------------------------------------------------------- #
# Root settings                                                               #
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """The whole of ``config.yaml``, typed and validated."""

    model_config = SettingsConfigDict(
        yaml_file=CONFIG_PATH,
        yaml_file_encoding="utf-8",
        env_file=ENV_PATH,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # .env-overridable (sec.8 .env.example)
    db_path: Path
    api_port: int = Field(gt=0, lt=65536)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    paths: PathsConfig
    window: WindowConfig
    universe: UniverseConfig
    point_in_time: PointInTimeConfig
    label: LabelConfig
    features: FeaturesConfig
    experiments: dict[str, ExperimentConfig]
    headline: HeadlineConfig
    walkforward: WalkForwardConfig
    training: TrainingConfig
    models: dict[str, ModelSpec]
    preprocessing: PreprocessingConfig
    evaluation: EvaluationConfig
    explain: ExplainConfig
    ingest: IngestConfig
    validation: ValidationConfig
    api: ApiConfig
    dashboard: DashboardConfig
    logging: LoggingConfig

    @field_validator("db_path", mode="after")
    @classmethod
    def _abs_db(cls, v: Path) -> Path:
        return _absolutise(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v: Any) -> Any:
        return v.upper() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        """Catch config mistakes that would otherwise surface deep in training."""
        known = set(self.features.all_features)
        for name, exp in self.experiments.items():
            unknown = [f for f in exp.features if f not in known]
            if unknown:
                raise ValueError(f"experiment '{name}' references unknown features: {unknown}")

        for key in (self.headline.experiment, self.headline.baseline):
            if key not in self.experiments:
                raise ValueError(f"headline references undefined experiment '{key}'")

        if self.training.search_model not in self.models:
            raise ValueError(
                f"training.search_model '{self.training.search_model}' is not a defined model"
            )

        if self.explain.model_for_shap not in self.models:
            raise ValueError(
                f"explain.model_for_shap '{self.explain.model_for_shap}' is not a defined model"
            )

        if (
            self.preprocessing.winsorize_lower_quantile
            >= self.preprocessing.winsorize_upper_quantile
        ):
            raise ValueError("winsorize_lower_quantile must be below winsorize_upper_quantile")

        if len(self.features.all_features) != len(set(self.features.all_features)):
            raise ValueError(
                "duplicate feature names in features.pledge_features + market_features"
            )

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """env > .env > config.yaml > init kwargs."""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    # ----------------------------------------------------------------- helpers
    def experiment_features(self, experiment: str) -> list[str]:
        if experiment not in self.experiments:
            raise KeyError(f"unknown experiment '{experiment}'. Known: {sorted(self.experiments)}")
        return list(self.experiments[experiment].features)

    def snapshot(self) -> str:
        """Full JSON config for ``model_runs.config_snapshot`` (sec.10)."""
        return self.model_dump_json(exclude_none=False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide singleton. Cached so YAML is parsed exactly once."""
    return Settings()


__all__ = ["PROJECT_ROOT", "Settings", "get_settings"]
