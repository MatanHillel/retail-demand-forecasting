"""Configuration loader tests (US-01, PRD §40, §56).

These tests assert structure, not results: they check that every configuration file parses into
its typed model, that the normative keys are present, and that impossible settings are rejected.
Indicative PRD numbers (row counts, wMAPE) are never asserted here.
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from pipeline import paths
from pipeline.config import (
    MODEL_IDS,
    CleaningConfig,
    InventoryPolicy,
    ModelConfig,
    SplitConfig,
    config_snapshot,
    load_cleaning_config,
    load_data_sources,
    load_inventory_policy,
    load_model_config,
    load_non_inventory_codes,
)

EXPECTED_NON_INVENTORY_COLUMNS = ["stock_code", "reason", "status"]
EXPECTED_NON_INVENTORY_ROWS = 28  # PRD §12 initial exclusion list


# --------------------------------------------------------------------------
# files exist and parse
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [
        paths.CLEANING_CONFIG,
        paths.MODEL_CONFIG,
        paths.INVENTORY_POLICY,
        paths.NON_INVENTORY_STOCKCODES,
        paths.DATA_SOURCES,
    ],
    ids=lambda path: path.name,
)
def test_configuration_file_exists(path) -> None:
    assert path.is_file(), f"missing configuration file: {path}"


def test_all_loaders_parse() -> None:
    assert isinstance(load_cleaning_config(), CleaningConfig)
    assert isinstance(load_model_config(), ModelConfig)
    assert isinstance(load_inventory_policy(), InventoryPolicy)
    assert load_data_sources().preferred in {"uci", "kaggle"}


def test_config_snapshot_is_json_serialisable() -> None:
    snapshot = config_snapshot()
    encoded = json.dumps(snapshot)
    assert json.loads(encoded) == snapshot
    assert set(snapshot) == {
        "cleaning_config",
        "model_config",
        "inventory_policy",
        "data_sources",
        "non_inventory_stockcodes",
    }


# --------------------------------------------------------------------------
# model_config.yaml — normative keys (PRD §14, §19, §21, §22, §40)
# --------------------------------------------------------------------------
def test_model_ids_are_exactly_the_seven_candidates() -> None:
    assert tuple(sorted(load_model_config().models)) == tuple(sorted(MODEL_IDS))
    assert len(MODEL_IDS) == len(set(MODEL_IDS))


def test_primary_model_and_main_baseline_are_marked() -> None:
    config = load_model_config()
    assert config.primary_model_id == "M2_gbm_poisson"
    assert config.main_baseline_id == "B2_ma3"
    assert config.models["B3_seasonal_naive"].reference_only is True


def test_split_and_backtest_windows_match_the_prd() -> None:
    split = load_model_config().split
    assert split.train_targets.start == "2010-03"
    assert split.train_targets.end == "2011-05"
    assert split.validation_targets.start == "2011-01"
    assert split.validation_targets.end == "2011-05"
    assert split.holdout_targets.start == "2011-06"
    assert split.holdout_targets.end == "2011-11"
    assert split.never_score == ["2011-12"]

    backtest = load_model_config().backtest
    assert backtest.first_origin == "2010-05"
    assert backtest.last_origin == "2011-10"


def test_never_score_month_is_outside_every_scored_window() -> None:
    config = load_model_config()
    split = config.split
    for month in split.never_score:
        assert not split.train_targets.start <= month <= split.train_targets.end
        assert not split.holdout_targets.start <= month <= split.holdout_targets.end
        assert month > config.backtest.last_origin


def test_seed_active_rule_and_features_are_present() -> None:
    config = load_model_config()
    assert config.seed == 42
    assert config.active_rule.k == 6
    assert "lag_1" in config.features
    assert "target_month_of_year" in config.features
    # Returns are EDA-only and must never appear as a feature (PRD §9, §13.2).
    assert not any("return" in feature for feature in config.features)


def test_champion_gates_are_configured() -> None:
    gates = load_model_config().champion_gates
    assert gates.max_abs_bias > 0
    assert gates.monthly_abs_bias_report_threshold >= gates.max_abs_bias
    assert gates.meaningful_improvement_points > gates.tie_wmape_points


# --------------------------------------------------------------------------
# inventory_policy.yaml (PRD §24–§27)
# --------------------------------------------------------------------------
def test_inventory_policy_keys() -> None:
    policy = load_inventory_policy()
    assert policy.lead_time_months == 1
    assert policy.z in policy.z_options
    assert policy.sigma.min_residuals_product == 6
    assert policy.sigma.fallback_levels == ["product", "abc_group", "global"]
    assert policy.abc.a_cum_share < policy.abc.b_cum_share
    assert "does not guarantee" in policy.disclaimer


def test_mad_scale_matches_the_robust_sigma_constant() -> None:
    """Robust σ = mad_scale × MAD; the scale is the normal-consistency constant (PRD §26)."""
    policy = load_inventory_policy()
    assert policy.sigma.mad_scale == pytest.approx(1.4826)


# --------------------------------------------------------------------------
# cleaning_config.yaml (PRD §10–§12)
# --------------------------------------------------------------------------
def test_cleaning_config_keys() -> None:
    config = load_cleaning_config()
    assert len(config.raw.required_columns) == 8
    assert config.raw.last_full_month == "2011-11"
    assert config.raw.partial_months == ["2011-12"]
    assert config.rules.cancellation_prefixes == ["C"]
    assert config.rules.adjustment_prefixes == ["A"]
    assert config.rules.duplicate_policy == "drop_exact"
    assert config.rules.non_inventory_file.endswith("non_inventory_stockcodes.csv")


# --------------------------------------------------------------------------
# non_inventory_stockcodes.csv (PRD §12)
# --------------------------------------------------------------------------
def test_non_inventory_codes_shape_and_columns() -> None:
    frame = load_non_inventory_codes()
    assert list(frame.columns) == EXPECTED_NON_INVENTORY_COLUMNS
    assert len(frame) == EXPECTED_NON_INVENTORY_ROWS
    assert (frame["status"] == "initial").all()
    assert not frame["reason"].str.strip().eq("").any()


def test_non_inventory_codes_include_the_documented_examples() -> None:
    codes = set(load_non_inventory_codes()["stock_code"])
    assert {"POST", "DOT", "BANK CHARGES", "AMAZONFEE", "PADS", "SP1002"} <= codes
    # `M` and `m` are kept as separate documentation rows until key normalisation (PRD §12).
    assert {"M", "m"} <= codes
    assert sum(code.startswith("gift_0001_") for code in codes) == 9


# --------------------------------------------------------------------------
# data_sources.yaml (PRD §42, §48)
# --------------------------------------------------------------------------
def test_data_sources_keys() -> None:
    sources = load_data_sources()
    assert sources.uci.url.startswith("https://")
    assert sources.uci.archive_member.endswith(".xlsx")
    assert len(sources.uci.sheets) == 2
    assert sources.kaggle.dataset == "mashlyn/online-retail-ii-uci"
    assert sources.raw_dir == "data/raw"
    # US-03 recorded the digest via `python -m pipeline.download --record-hash`.
    assert sources.expected_sha256 is not None
    assert re.fullmatch(r"[0-9a-f]{64}", sources.expected_sha256)
    assert "CC BY 4.0" in sources.citation


# --------------------------------------------------------------------------
# validation rejects impossible settings
# --------------------------------------------------------------------------
def test_split_rejects_train_window_overlapping_the_holdout() -> None:
    valid = load_model_config().split.model_dump(mode="json")
    broken = {**valid, "holdout_targets": valid["train_targets"]}
    with pytest.raises(ValidationError):
        SplitConfig(**broken)


def test_split_rejects_never_score_month_inside_the_holdout() -> None:
    valid = load_model_config().split.model_dump(mode="json")
    broken = {**valid, "never_score": [valid["holdout_targets"]["end"]]}
    with pytest.raises(ValidationError):
        SplitConfig(**broken)


def test_model_config_rejects_an_unknown_model_id() -> None:
    valid = load_model_config().model_dump(mode="json")
    broken = {**valid, "models": {**valid["models"], "M5_prophet": {"kind": "baseline"}}}
    with pytest.raises(ValidationError):
        ModelConfig(**broken)


def test_model_config_rejects_a_missing_model_id() -> None:
    valid = load_model_config().model_dump(mode="json")
    models = {key: value for key, value in valid["models"].items() if key != "M4_gbm_absolute"}
    with pytest.raises(ValidationError):
        ModelConfig(**{**valid, "models": models})


@pytest.mark.parametrize(
    "override",
    [
        {"z": 0},
        {"sigma": {"mad_scale": 0, "min_residuals_product": 6,
                   "fallback_levels": ["product", "abc_group", "global"]}},
        {"sigma": {"mad_scale": 1.5, "min_residuals_product": 0,
                   "fallback_levels": ["product", "abc_group", "global"]}},
        {"abc": {"a_cum_share": 0.95, "b_cum_share": 0.80}},
        {"lead_time_months": 0},
    ],
    ids=["non_positive_z", "non_positive_mad_scale", "zero_min_residuals",
         "abc_shares_out_of_order", "zero_lead_time"],
)
def test_inventory_policy_rejects_impossible_settings(override: dict) -> None:
    valid = load_inventory_policy().model_dump(mode="json")
    with pytest.raises(ValidationError):
        InventoryPolicy(**{**valid, **override})


def test_cleaning_config_rejects_an_unknown_key() -> None:
    valid = load_cleaning_config().model_dump(mode="json")
    with pytest.raises(ValidationError):
        CleaningConfig(**{**valid, "unexpected_section": {}})


# --------------------------------------------------------------------------
# paths.py (PRD §41)
# --------------------------------------------------------------------------
def test_required_artifact_names_are_exact() -> None:
    assert [path.name for path in paths.REQUIRED_ARTIFACTS] == [
        "clean_data.csv",
        "features.csv",
        "model.joblib",
        "eda_report.html",
        "insights.md",
        "dataset_contract.json",
        "evaluation_report.md",
        "model_card.md",
    ]


def test_paths_are_absolute_and_inside_the_project() -> None:
    for path in (*paths.REQUIRED_ARTIFACTS, paths.RUN_LOG, paths.VALIDATION_REPORT):
        assert path.is_absolute()
        assert paths.PROJECT_ROOT in path.parents
