"""Feature-validation & leakage tests (US-14, PRD §16, §17, §37 step 5, §39, §55).

Hand-built mini panels only (§40). A features frame built by the real ``build_features`` (US-13)
must pass both :func:`validate_features` and :func:`leakage_check`; each injected-leak scenario
corrupts a features frame in exactly the way its name describes and proves the matching check — and
only that kind of check — catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.config import load_model_config
from pipeline.feature_validation import LEAKAGE_FAILURE_MESSAGE, leakage_check, validate_features
from pipeline.features import build_features
from pipeline.validation import FlowValidationError, ValidationResult

K = 6
FIRST_TARGET = "2010-03"
LAST_TARGET = "2010-10"


def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _panel(sales: dict[str, dict[str, int]], end: str = LAST_TARGET) -> pd.DataFrame:
    """A US-05-shaped panel: each product runs from its first sale to ``end``, zero-filled."""
    rows = []
    for stock_code, months in sales.items():
        first = min(months)
        for month in _months(first, end):
            units = months.get(month, 0)
            rows.append(
                {
                    "month": month,
                    "stock_code": stock_code,
                    "description": "TEST PRODUCT",
                    "units_sold": units,
                    "gross_revenue": float(units) * 2.0,
                    "avg_unit_price": 2.0,
                    "invoice_count": 3 if units else 0,
                    "sale_line_count": 3 if units else 0,
                    "customer_count": 2 if units else 0,
                    "max_line_qty": units,
                    "returned_units": 0,
                    "is_partial_month": False,
                }
            )
    return pd.DataFrame(rows)


#: end=LAST_TARGET so the panel never extends past the features range being validated — otherwise
#: the active_rule coverage check would (correctly) flag active pairs the features file never had a
#: chance to cover.
SALES = {
    "P1": {
        "2009-12": 100,
        "2010-02": 50,
        "2010-03": 7,
        "2010-04": 12,
        "2010-06": 20,
        "2010-08": 15,
    },
    "P2": {"2010-06": 40, "2010-08": 20, "2010-09": 9},
    "P3": {"2010-04": 30, "2010-05": 10, "2010-07": 5},
}


@pytest.fixture()
def panel() -> pd.DataFrame:
    return _panel(SALES)


@pytest.fixture()
def features(panel: pd.DataFrame) -> pd.DataFrame:
    return build_features(panel, K, FIRST_TARGET, LAST_TARGET, load_model_config())


def _failing_checks(result: ValidationResult) -> set[str]:
    return {check["name"] for check in result.extra["checks"] if not check["passed"]}


# --------------------------------------------------------------------------
# a correctly built features frame passes both checks
# --------------------------------------------------------------------------
def test_valid_features_pass_structural_validation(features, panel) -> None:
    result = validate_features(features, panel, load_model_config(), K)
    assert result.passed, [violation.message for violation in result.violations]
    assert {check["name"] for check in result.extra["checks"]} == {
        "target_present",
        "required_features_present",
        "no_nan_in_required",
        "origin_before_target",
        "active_rule",
        "target_range",
        "key_unique",
    }


def test_valid_features_pass_the_leakage_check(features, panel) -> None:
    cfg = load_model_config()
    result = leakage_check(features, panel, cfg, cfg.seed)
    assert result.passed, [violation.message for violation in result.violations]
    assert {check["name"] for check in result.extra["checks"]} == {
        "lag_recomputation",
        "permutation_future_months",
        "calendar_only_future_info",
    }


def test_leakage_check_never_modifies_its_inputs(features, panel) -> None:
    features_before = features.copy(deep=True)
    panel_before = panel.copy(deep=True)
    cfg = load_model_config()
    leakage_check(features, panel, cfg, cfg.seed)
    pd.testing.assert_frame_equal(features, features_before)
    pd.testing.assert_frame_equal(panel, panel_before)


def test_validate_features_is_deterministic_across_runs(features, panel) -> None:
    cfg = load_model_config()
    first = leakage_check(features, panel, cfg, cfg.seed)
    second = leakage_check(features, panel, cfg, cfg.seed)
    assert first.passed == second.passed
    assert first.extra["checks"] == second.extra["checks"]


# --------------------------------------------------------------------------
# (a) a shift(0) lag_1 fails both lag_recomputation and permutation_future_months
# --------------------------------------------------------------------------
def test_shift_zero_lag_fails_lag_recomputation_and_permutation(features, panel) -> None:
    cfg = load_model_config()
    leaky = features.copy()
    # lag_1 reads the target month itself (month t) instead of t-1 — the archetypal leak.
    leaky["lag_1"] = leaky["y"]

    result = leakage_check(leaky, panel, cfg, cfg.seed)
    assert not result.passed
    failing = _failing_checks(result)
    assert "lag_recomputation" in failing
    assert "permutation_future_months" in failing


# --------------------------------------------------------------------------
# (b) a future-looking rolling_mean_6 fails the permutation test
# --------------------------------------------------------------------------
def test_future_looking_rolling_mean_fails_permutation(features, panel) -> None:
    cfg = load_model_config()
    leaky = features.copy()
    # A window "centred" on t would read the target's own outcome — simulate that dependency.
    leaky["rolling_mean_6"] = leaky["y"].astype(float) + 1000.0

    result = leakage_check(leaky, panel, cfg, cfg.seed)
    assert not result.passed
    assert "permutation_future_months" in _failing_checks(result)


# --------------------------------------------------------------------------
# (c) a 2011-12 target fails target_range
# --------------------------------------------------------------------------
def test_never_scored_target_fails_target_range(features, panel) -> None:
    cfg = load_model_config()
    extra_row = features.iloc[[0]].copy()
    extra_row["target_month"] = "2011-12"
    extra_row["forecast_origin"] = "2011-11"
    leaky = pd.concat([features, extra_row], ignore_index=True)

    result = validate_features(leaky, panel, cfg, K)
    assert not result.passed
    assert "target_range" in _failing_checks(result)


# --------------------------------------------------------------------------
# (d) an inactive row fails active_rule
# --------------------------------------------------------------------------
def test_inactive_row_fails_active_rule(features, panel) -> None:
    cfg = load_model_config()
    extra_row = features.iloc[[0]].copy()
    extra_row["stock_code"] = "P3"
    # P3's last sale is 2010-04; k=6 months later (2010-11) it is no longer active, and the panel
    # (end=LAST_TARGET=2010-10) does not even reach that month.
    extra_row["target_month"] = "2010-11"
    extra_row["forecast_origin"] = "2010-10"
    leaky = pd.concat([features, extra_row], ignore_index=True)

    result = validate_features(leaky, panel, cfg, K)
    assert not result.passed
    assert "active_rule" in _failing_checks(result)


# --------------------------------------------------------------------------
# (e) an injected NaN fails no_nan_in_required
# --------------------------------------------------------------------------
def test_nan_feature_fails_no_nan_in_required(features, panel) -> None:
    cfg = load_model_config()
    leaky = features.copy()
    leaky.loc[leaky.index[0], "lag_1"] = np.nan

    result = validate_features(leaky, panel, cfg, K)
    assert not result.passed
    assert "no_nan_in_required" in _failing_checks(result)


# --------------------------------------------------------------------------
# the exact §39 wording for a leakage failure
# --------------------------------------------------------------------------
def test_flow_validation_error_uses_the_exact_leakage_wording(features, panel) -> None:
    cfg = load_model_config()
    leaky = features.copy()
    leaky["lag_1"] = leaky["y"]

    feature_result = validate_features(leaky, panel, cfg, K)
    leakage_result = leakage_check(leaky, panel, cfg, cfg.seed)
    assert not leakage_result.passed

    combined = ValidationResult(
        step="feature_validation",
        passed=feature_result.passed and leakage_result.passed,
        violations=feature_result.violations + leakage_result.violations,
    )
    error = FlowValidationError(combined, LEAKAGE_FAILURE_MESSAGE)
    assert str(error) == f"FLOW STOPPED: {LEAKAGE_FAILURE_MESSAGE}"
