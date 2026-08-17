"""``feature_validation.json`` schema tests (US-14, PRD §37 step 5, §39, §55).

:func:`write_feature_validation` is the ``ctx``-taking writer added by AI-19's §8 interface
correction; these tests prove its output has the schema the app and the Flow depend on, and that it
goes through ``ctx.out()`` like every other run artifact (``docs/interfaces.md`` §6 rule 1).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.config import load_model_config
from pipeline.feature_validation import leakage_check, validate_features, write_feature_validation
from pipeline.features import build_features
from pipeline.run_context import RunContext, close_log_handlers

K = 6
FIRST_TARGET = "2010-03"
LAST_TARGET = "2010-10"

REQUIRED_CHECK_NAMES = {
    "target_present",
    "required_features_present",
    "no_nan_in_required",
    "origin_before_target",
    "active_rule",
    "target_range",
    "key_unique",
    "lag_recomputation",
    "permutation_future_months",
    "calendar_only_future_info",
}


def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _panel(sales: dict[str, dict[str, int]], end: str = LAST_TARGET) -> pd.DataFrame:
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
def ctx(tmp_path):
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


def _passing_results() -> tuple:
    cfg = load_model_config()
    panel = _panel(SALES)
    features = build_features(panel, K, FIRST_TARGET, LAST_TARGET, cfg)
    feature_result = validate_features(features, panel, cfg, K)
    leakage_result = leakage_check(features, panel, cfg, cfg.seed)
    return feature_result, leakage_result


# --------------------------------------------------------------------------
# schema (§6 acceptance criteria)
# --------------------------------------------------------------------------
def test_feature_validation_json_has_the_required_keys(ctx) -> None:
    feature_result, leakage_result = _passing_results()
    path = write_feature_validation([feature_result, leakage_result], ctx)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {"passed", "run_id", "timestamp", "checks", "sample_sizes", "seed"} <= set(report)
    assert report["run_id"] == ctx.run_id
    assert report["passed"] is True


def test_feature_validation_json_lists_every_check_name(ctx) -> None:
    feature_result, leakage_result = _passing_results()
    path = write_feature_validation([feature_result, leakage_result], ctx)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert {check["name"] for check in report["checks"]} == REQUIRED_CHECK_NAMES
    for check in report["checks"]:
        assert {"name", "passed", "count", "examples"} <= set(check)


def test_feature_validation_json_records_sample_sizes_and_seed(ctx) -> None:
    cfg = load_model_config()
    feature_result, leakage_result = _passing_results()
    path = write_feature_validation([feature_result, leakage_result], ctx)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["seed"] == cfg.seed
    assert set(report["sample_sizes"]) == {"lag_sample_rows", "permutation_products"}


def test_a_failing_result_still_reports_every_check(ctx) -> None:
    cfg = load_model_config()
    panel = _panel(SALES)
    features = build_features(panel, K, FIRST_TARGET, LAST_TARGET, cfg)
    leaky = features.copy()
    leaky.loc[leaky.index[0], "lag_1"] = None

    feature_result = validate_features(leaky, panel, cfg, K)
    leakage_result = leakage_check(features, panel, cfg, cfg.seed)
    path = write_feature_validation([feature_result, leakage_result], ctx)
    report = json.loads(path.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert {check["name"] for check in report["checks"]} == REQUIRED_CHECK_NAMES


def test_report_is_valid_json_serialisable_content(ctx) -> None:
    """Every example/count must already be a plain JSON type — no stray numpy scalars."""
    feature_result, leakage_result = _passing_results()
    path = write_feature_validation([feature_result, leakage_result], ctx)
    # write_feature_validation itself calls json.dumps; re-parsing proves it round-trips cleanly.
    json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# staging (``docs/interfaces.md`` §6 rule 1 — every artifact write goes through ctx.out())
# --------------------------------------------------------------------------
def test_write_feature_validation_goes_through_ctx_out(tmp_path) -> None:
    context = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        feature_result, leakage_result = _passing_results()
        staged_path = write_feature_validation([feature_result, leakage_result], context)
        final_path = context.base_dir / "artifacts" / "reports" / "feature_validation.json"
        assert staged_path != final_path
        assert not final_path.exists()
        context.promote()
        assert final_path.is_file()
    finally:
        close_log_handlers(context.run_id)


def test_write_feature_validation_registers_the_artifact(ctx) -> None:
    feature_result, leakage_result = _passing_results()
    write_feature_validation([feature_result, leakage_result], ctx)
    assert ctx.artifacts["feature_validation"] == "artifacts/reports/feature_validation.json"
