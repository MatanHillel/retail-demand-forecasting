"""Typed configuration loader (PRD §40, §56).

Every tunable rule of the PRD lives in ``config/*.yaml``; this module parses those files into
validated Pydantic v2 models so that a typo or an impossible setting fails loudly at load time
rather than silently changing a result. No number is defined here — the YAML files are the only
source of truth, and :func:`config_snapshot` records exactly which settings produced a run.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline import paths

# Normative model ids (PRD §19). Referenced literally by later issues and by the app.
MODEL_IDS: tuple[str, ...] = (
    "B1_last_month",
    "B2_ma3",
    "B3_seasonal_naive",
    "M1_linear",
    "M2_gbm_poisson",
    "M3_gbm_squared",
    "M4_gbm_absolute",
)

_MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict, with a clear error if it is missing or not a mapping."""
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"configuration file must contain a mapping at the top level: {path}")
    return content


class _Base(BaseModel):
    """Shared behaviour: reject unknown keys (catches typos) and forbid mutation after load."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_month(value: str) -> str:
    if not _MONTH_PATTERN.match(value):
        raise ValueError(f"month must be formatted as YYYY-MM, got {value!r}")
    return value


# --------------------------------------------------------------------------
# cleaning_config.yaml
# --------------------------------------------------------------------------
class RawSpec(_Base):
    required_columns: list[str] = Field(min_length=1)
    first_month: str
    last_full_month: str
    partial_months: list[str]

    @field_validator("first_month", "last_full_month")
    @classmethod
    def _month_format(cls, value: str) -> str:
        return _validate_month(value)

    @field_validator("partial_months")
    @classmethod
    def _months_format(cls, value: list[str]) -> list[str]:
        return [_validate_month(month) for month in value]

    @model_validator(mode="after")
    def _ordered(self) -> RawSpec:
        if self.first_month >= self.last_full_month:
            raise ValueError("raw.first_month must be earlier than raw.last_full_month")
        for month in self.partial_months:
            if month <= self.last_full_month:
                raise ValueError(
                    f"partial month {month} must fall after raw.last_full_month "
                    f"{self.last_full_month}"
                )
        return self


class CleaningRules(_Base):
    cancellation_prefixes: list[str] = Field(min_length=1)
    adjustment_prefixes: list[str] = Field(min_length=1)
    drop_quantity_at_or_below: float
    drop_price_at_or_below: float
    duplicate_policy: Literal["drop_exact"]
    non_inventory_file: str
    inventory_code_pattern: str

    @field_validator("inventory_code_pattern")
    @classmethod
    def _compilable(cls, value: str) -> str:
        re.compile(value)  # raises re.error if the pattern is malformed
        return value


class CleaningWarnings(_Base):
    duplicate_max_row_share: float = Field(gt=0, le=1)
    duplicate_max_units_share: float = Field(gt=0, le=1)
    abnormal_line_quantity: int = Field(gt=0)


class CleaningConfig(_Base):
    """Rules for turning the raw extract into ``clean_transactions.parquet`` (PRD §10–§12)."""

    raw: RawSpec
    rules: CleaningRules
    warnings: CleaningWarnings


# --------------------------------------------------------------------------
# model_config.yaml
# --------------------------------------------------------------------------
class ActiveRule(_Base):
    k: int = Field(gt=0)  # §14 — active = ≥ 1 positive sale in the previous k months


class MonthRange(_Base):
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _month_format(cls, value: str) -> str:
        return _validate_month(value)

    @model_validator(mode="after")
    def _ordered(self) -> MonthRange:
        if self.start > self.end:
            raise ValueError(f"range start {self.start} must not be after end {self.end}")
        return self


class SplitConfig(_Base):
    """Temporal split (PRD §21). Shuffled splits are forbidden project-wide."""

    first_target_month: str
    train_targets: MonthRange
    validation_targets: MonthRange
    holdout_targets: MonthRange
    never_score: list[str]

    @field_validator("first_target_month")
    @classmethod
    def _month_format(cls, value: str) -> str:
        return _validate_month(value)

    @field_validator("never_score")
    @classmethod
    def _months_format(cls, value: list[str]) -> list[str]:
        return [_validate_month(month) for month in value]

    @model_validator(mode="after")
    def _temporal_order(self) -> SplitConfig:
        if self.train_targets.end >= self.holdout_targets.start:
            raise ValueError(
                f"train_targets.end ({self.train_targets.end}) must be strictly before "
                f"holdout_targets.start ({self.holdout_targets.start})"
            )
        if self.first_target_month != self.train_targets.start:
            raise ValueError("first_target_month must equal train_targets.start")
        if not (
            self.train_targets.start
            <= self.validation_targets.start
            <= self.validation_targets.end
            <= self.train_targets.end
        ):
            raise ValueError("validation_targets must lie inside train_targets")
        for month in self.never_score:
            if self.holdout_targets.start <= month <= self.holdout_targets.end:
                raise ValueError(f"never_score month {month} overlaps the hold-out window")
        return self


class BacktestConfig(_Base):
    first_origin: str
    last_origin: str

    @field_validator("first_origin", "last_origin")
    @classmethod
    def _month_format(cls, value: str) -> str:
        return _validate_month(value)

    @model_validator(mode="after")
    def _ordered(self) -> BacktestConfig:
        if self.first_origin >= self.last_origin:
            raise ValueError("backtest.first_origin must be earlier than backtest.last_origin")
        return self


class ModelSpec(_Base):
    """One forecasting candidate (PRD §19). Baselines carry no hyper-parameters."""

    kind: Literal["baseline", "linear_regression", "hist_gradient_boosting"]
    loss: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    primary: bool = False
    main_baseline: bool = False
    reference_only: bool = False
    clip_negative_for_business: bool = False


class TuningConfig(_Base):
    grid: dict[str, list[Any]]
    metric: Literal["wmape"]


class ChampionGates(_Base):
    """Gate thresholds of PRD §20 — applied by code to every candidate, baselines included."""

    max_abs_bias: float = Field(gt=0)
    monthly_abs_bias_report_threshold: float = Field(gt=0)
    tie_wmape_points: float = Field(gt=0)
    meaningful_improvement_points: float = Field(gt=0)
    similar_fill_rate_tolerance: float = Field(gt=0)
    tie_break_order: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _report_threshold_is_looser(self) -> ChampionGates:
        if self.monthly_abs_bias_report_threshold < self.max_abs_bias:
            raise ValueError(
                "monthly_abs_bias_report_threshold must not be stricter than max_abs_bias"
            )
        return self

    @field_validator("tie_break_order")
    @classmethod
    def _valid_model_ids(cls, value: list[str]) -> list[str]:
        if not set(value).issubset(MODEL_IDS):
            raise ValueError(f"tie_break_order must only contain values from {MODEL_IDS}")
        return value


class ValidationParams(_Base):
    """Sample sizes for feature validation & leakage tests (PRD §37 step 5, §55, US-14)."""

    lag_sample_rows: int = Field(gt=0)
    permutation_products: int = Field(gt=0)


class ModelConfig(_Base):
    """Seed, features, split, candidates and champion gates (PRD §14, §17, §19–§22)."""

    seed: int
    active_rule: ActiveRule
    features: list[str] = Field(min_length=1)
    split: SplitConfig
    backtest: BacktestConfig
    models: dict[str, ModelSpec]
    tuning: TuningConfig
    champion_gates: ChampionGates
    validation: ValidationParams

    @field_validator("features")
    @classmethod
    def _unique_features(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("features must not contain duplicates")
        return value

    @field_validator("models")
    @classmethod
    def _exact_model_ids(cls, value: dict[str, ModelSpec]) -> dict[str, ModelSpec]:
        if tuple(sorted(value)) != tuple(sorted(MODEL_IDS)):
            missing = sorted(set(MODEL_IDS) - set(value))
            unexpected = sorted(set(value) - set(MODEL_IDS))
            raise ValueError(
                f"models must be exactly {list(MODEL_IDS)}; missing={missing}, "
                f"unexpected={unexpected}"
            )
        return value

    @model_validator(mode="after")
    def _single_primary_and_baseline(self) -> ModelConfig:
        primaries = [name for name, spec in self.models.items() if spec.primary]
        if len(primaries) != 1:
            raise ValueError(f"exactly one model must be marked primary, got {primaries}")
        main_baselines = [name for name, spec in self.models.items() if spec.main_baseline]
        if len(main_baselines) != 1:
            raise ValueError(
                f"exactly one baseline must be marked main_baseline, got {main_baselines}"
            )
        if self.models[main_baselines[0]].kind != "baseline":
            raise ValueError("main_baseline must be a baseline model")
        return self

    @property
    def baseline_ids(self) -> tuple[str, ...]:
        return tuple(name for name, spec in self.models.items() if spec.kind == "baseline")

    @property
    def primary_model_id(self) -> str:
        return next(name for name, spec in self.models.items() if spec.primary)

    @property
    def main_baseline_id(self) -> str:
        return next(name for name, spec in self.models.items() if spec.main_baseline)


# --------------------------------------------------------------------------
# inventory_policy.yaml
# --------------------------------------------------------------------------
class SigmaConfig(_Base):
    """Robust σ settings (PRD §26, §27). σ uses out-of-sample residuals only."""

    mad_scale: float = Field(gt=0)
    min_residuals_product: int = Field(ge=1)
    fallback_levels: list[Literal["product", "abc_group", "global"]] = Field(min_length=1)

    @field_validator("fallback_levels")
    @classmethod
    def _ordered_levels(cls, value: list[str]) -> list[str]:
        if value != ["product", "abc_group", "global"]:
            raise ValueError(
                "fallback_levels must be ['product', 'abc_group', 'global'] (PRD §27)"
            )
        return value


class AbcConfig(_Base):
    a_cum_share: float = Field(gt=0, lt=1)
    b_cum_share: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _ordered(self) -> AbcConfig:
        if self.a_cum_share >= self.b_cum_share:
            raise ValueError("abc.a_cum_share must be smaller than abc.b_cum_share")
        return self


class InventoryPolicy(_Base):
    """Deterministic policy converting a forecast into Recommended Target Inventory (§24–§28)."""

    lead_time_months: int = Field(gt=0)
    z: float = Field(gt=0)
    z_options: list[float] = Field(min_length=1)
    sigma: SigmaConfig
    abc: AbcConfig
    disclaimer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _default_z_is_offered(self) -> InventoryPolicy:
        if self.z not in self.z_options:
            raise ValueError(f"default z ({self.z}) must be one of z_options ({self.z_options})")
        if any(option <= 0 for option in self.z_options):
            raise ValueError("every value in z_options must be positive")
        return self


# --------------------------------------------------------------------------
# data_sources.yaml
# --------------------------------------------------------------------------
class UciSource(_Base):
    url: str
    archive_member: str
    sheets: list[str] = Field(min_length=1)


class KaggleSource(_Base):
    dataset: str
    file: str


class DataSources(_Base):
    """Where the raw extract comes from and how its integrity is checked (PRD §42, §48)."""

    preferred: Literal["uci", "kaggle"]
    uci: UciSource
    kaggle: KaggleSource
    raw_dir: str
    expected_sha256: str | None = None
    citation: str = Field(min_length=1)

    @field_validator("expected_sha256")
    @classmethod
    def _hex_digest(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("expected_sha256 must be a lower-case 64-character hex digest")
        return value


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_cleaning_config() -> CleaningConfig:
    """Parse ``config/cleaning_config.yaml``."""
    return CleaningConfig(**_read_yaml(paths.CLEANING_CONFIG))


@lru_cache(maxsize=1)
def load_model_config() -> ModelConfig:
    """Parse ``config/model_config.yaml``."""
    return ModelConfig(**_read_yaml(paths.MODEL_CONFIG))


@lru_cache(maxsize=1)
def load_inventory_policy() -> InventoryPolicy:
    """Parse ``config/inventory_policy.yaml``."""
    return InventoryPolicy(**_read_yaml(paths.INVENTORY_POLICY))


@lru_cache(maxsize=1)
def load_data_sources() -> DataSources:
    """Parse ``config/data_sources.yaml``."""
    return DataSources(**_read_yaml(paths.DATA_SOURCES))


def load_non_inventory_codes() -> pd.DataFrame:
    """Read the excluded StockCodes as a DataFrame with columns ``stock_code, reason, status``.

    Codes are read as raw strings: ``M`` and ``m`` are distinct rows here and are merged only
    after key normalisation in the cleaning step (PRD §10 step 3, §12).
    """
    path = paths.NON_INVENTORY_STOCKCODES
    if not path.is_file():
        raise FileNotFoundError(f"configuration file not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    expected = ["stock_code", "reason", "status"]
    if list(frame.columns) != expected:
        raise ValueError(f"{path.name} must have columns {expected}, got {list(frame.columns)}")
    if frame["stock_code"].duplicated().any():
        duplicates = sorted(frame.loc[frame["stock_code"].duplicated(), "stock_code"])
        raise ValueError(f"{path.name} contains duplicate stock codes: {duplicates}")
    if (frame["stock_code"].str.strip() == "").any():
        raise ValueError(f"{path.name} contains an empty stock_code")
    return frame


def clear_config_cache() -> None:
    """Drop cached configs — used by tests that write temporary configuration files."""
    load_cleaning_config.cache_clear()
    load_model_config.cache_clear()
    load_inventory_policy.cache_clear()
    load_data_sources.cache_clear()


def config_snapshot() -> dict[str, Any]:
    """Return every configuration merged into one JSON-serialisable dict.

    Recorded in ``run_log.json`` so a run can always be traced back to the exact settings that
    produced it (PRD §40).
    """
    return {
        "cleaning_config": load_cleaning_config().model_dump(mode="json"),
        "model_config": load_model_config().model_dump(mode="json"),
        "inventory_policy": load_inventory_policy().model_dump(mode="json"),
        "data_sources": load_data_sources().model_dump(mode="json"),
        "non_inventory_stockcodes": load_non_inventory_codes().to_dict(orient="records"),
    }
