"""``evaluation_report.md`` & ``model_card.md`` generator tests (US-25, PRD §36, §38, §49).

A small fixture tree under ``tmp_path`` supplies schema-correct, minimal versions of every input
US-19 through US-24 already produce — the seven-candidate evaluation tables, the champion decision
trace, inventory KPIs, sigma summary, quarterly metrics and the model/contract/quality JSON files —
so the fixture tests never touch real project data and stay fast. One test also runs both
generators against the real, committed artifacts already produced by the upstream stories, skipped
if this checkout does not have them (mirrors ``tests/test_champion.py``'s end-to-end convention).

Real configuration (``load_model_config()`` / ``load_inventory_policy()``) is used throughout,
never a hand-rolled threshold, so a config change is felt here rather than silently diverging.
"""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import MODEL_IDS, load_inventory_policy, load_model_config
from pipeline.narrative import numbers_in_tables
from pipeline.reports import (
    MODEL_CARD_HEADINGS,
    write_all_reports,
    write_evaluation_report,
    write_model_card,
)
from pipeline.run_context import RunContext, close_log_handlers

CFG = load_model_config()
POLICY_CFG = load_inventory_policy()
GATES = CFG.champion_gates
CHAMPION = "M2_gbm_poisson"
BEST_BASELINE = "B1_last_month"

# Small, deterministic, non-zero bias/wmape values per candidate — negative and positive on
# purpose, so the guard's percentage-matching is exercised in both directions.
_WMAPE = {
    "B1_last_month": 0.557,
    "B2_ma3": 0.547,
    "B3_seasonal_naive": 0.896,
    "M1_linear": 0.560,
    "M2_gbm_poisson": 0.526,
    "M3_gbm_squared": 0.543,
    "M4_gbm_absolute": 0.502,
}
_BIAS = {
    "B1_last_month": -0.085,
    "B2_ma3": -0.174,
    "B3_seasonal_naive": 0.350,
    "M1_linear": -0.181,
    "M2_gbm_poisson": 0.007,
    "M3_gbm_squared": 0.015,
    "M4_gbm_absolute": -0.258,
}


def _write_csv(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8")


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _overall_table() -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        coverage = 0.83 if model_id == "B3_seasonal_naive" else 1.0
        rows.append(
            {
                "model": model_id,
                "wmape": _WMAPE[model_id],
                "bias": _BIAS[model_id],
                "mae": 80.0 + hash(model_id) % 10,
                "rmse": 230.0 + hash(model_id) % 20,
                "n_rows": 19968 if coverage == 1.0 else 16529,
                "sum_actual": 3_049_123.0,
                "sum_forecast": 2_800_000.0,
                "negative_share_raw": 0.0,
                "coverage_share": coverage,
                "note": (
                    "B3 (seasonal naive) has no observed month t-12 for every product."
                    if model_id == "B3_seasonal_naive"
                    else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def _by_month_table() -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        for month in ("2011-06", "2011-11"):
            rows.append(
                {
                    "model": model_id,
                    "target_month": month,
                    "wmape": _WMAPE[model_id],
                    "bias": _BIAS[model_id],
                    "mae": 80.0,
                    "rmse": 230.0,
                    "n_rows": 3284,
                }
            )
    return pd.DataFrame(rows)


def _by_abc_table() -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        for abc_class in ("A", "B", "C"):
            rows.append(
                {
                    "model": model_id,
                    "abc_class": abc_class,
                    "wmape": _WMAPE[model_id],
                    "bias": _BIAS[model_id],
                    "mae": 80.0,
                    "rmse": 230.0,
                    "n_rows": 5000,
                }
            )
    return pd.DataFrame(rows)


def _improvement_table(*, meaningful_improvement: bool) -> pd.DataFrame:
    wmape_b2 = _WMAPE["B2_ma3"]
    rows = []
    for model_id in MODEL_IDS:
        wmape_model = _WMAPE[model_id]
        points = wmape_b2 - wmape_model
        if model_id == CHAMPION:
            points = 0.02 if meaningful_improvement else 0.015
            wmape_model = wmape_b2 - points
        rows.append(
            {
                "model": model_id,
                "wmape": wmape_model,
                "wmape_b2": wmape_b2,
                "wmape_points_vs_b2": points,
                "relative_pct": points / wmape_b2 if wmape_b2 else float("nan"),
                "meaningful": points >= GATES.meaningful_improvement_points,
            }
        )
    return pd.DataFrame(rows)


def _consistency_table() -> pd.DataFrame:
    rows = []
    for model_id in MODEL_IDS:
        for origin, target in (("2011-05", "2011-06"), ("2011-10", "2011-11")):
            rows.append(
                {
                    "model": model_id,
                    "forecast_origin": origin,
                    "target_month": target,
                    "n_rows": 3284,
                    "wmape": _WMAPE[model_id],
                    "bias": _BIAS[model_id],
                    "wmape_mean": _WMAPE[model_id],
                    "wmape_std": 0.021,
                    "wmape_min": _WMAPE[model_id] - 0.05,
                    "wmape_max": _WMAPE[model_id] + 0.05,
                    "bias_mean": _BIAS[model_id],
                    "months_abs_bias_gt_threshold": 1 if _BIAS[model_id] < -0.2 else 0,
                }
            )
    return pd.DataFrame(rows)


def _excess_table() -> pd.DataFrame:
    rows = []
    for model_id in (CHAMPION, "B2_ma3"):
        for policy in ("forecast_only", "forecast_plus_ss"):
            rows.append(
                {
                    "model": model_id,
                    "policy": policy,
                    "n_rows": 16529,
                    "total_excess": 2_295_561.0,
                    "top_1pct_share": 0.176,
                    "top_5pct_share": 0.422,
                }
            )
    return pd.DataFrame(rows)


def _sigma_summary_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": CHAMPION,
                "target_month": "2011-06",
                "n_products": 3284,
                "share_product": 0.93,
                "share_abc_group": 0.07,
                "share_global": 0.0,
                "median_sigma": 27.0,
                "share_zero_mad": 0.0,
            },
            {
                "model": CHAMPION,
                "target_month": "2011-11",
                "n_products": 3388,
                "share_product": 0.86,
                "share_abc_group": 0.14,
                "share_global": 0.0,
                "median_sigma": 22.3,
                "share_zero_mad": 0.0,
            },
        ]
    )


def _kpis_table() -> pd.DataFrame:
    rows = []
    for model_id in (CHAMPION, "B2_ma3"):
        for policy, fill_rate in (("forecast_only", 0.73), ("forecast_plus_ss", 0.90)):
            for z in POLICY_CFG.z_options:
                rows.append(
                    {
                        "model": model_id,
                        "policy": policy,
                        "z": z,
                        "scope": "overall",
                        "group": "all",
                        "fill_rate": fill_rate,
                        "stockout_units": 220000.0,
                        "excess_units": 2295561.0,
                        "stockout_skumonth_rate": 0.07,
                        "excess_per_unit_shortage": 10.3,
                        "n_rows": 16529,
                    }
                )
    return pd.DataFrame(rows)


def _quarterly_metrics_table() -> pd.DataFrame:
    rows = []
    for model_id in (CHAMPION, "B2_ma3"):
        rows.append(
            {
                "model": model_id,
                "scope": "overall",
                "quarter": "all",
                "wmape": _WMAPE[model_id],
                "bias": _BIAS[model_id],
                "mae": 150.0,
                "rmse": 470.0,
                "n_rows": 16306,
                "sum_actual": 6_122_313.0,
                "sum_forecast": 6_300_000.0,
                "negative_share": 0.0,
            }
        )
        rows.append(
            {
                "model": model_id,
                "scope": "quarter",
                "quarter": "2011-Q3",
                "wmape": _WMAPE[model_id],
                "bias": _BIAS[model_id],
                "mae": 150.0,
                "rmse": 470.0,
                "n_rows": 3190,
                "sum_actual": 1_296_545.0,
                "sum_forecast": 1_144_558.0,
                "negative_share": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _champion_decision(*, meaningful_improvement: bool) -> dict:
    improvement_points = 3.0 if meaningful_improvement else 1.5
    candidates = []
    for model_id in MODEL_IDS:
        gate1_pass = model_id in (CHAMPION, BEST_BASELINE, "M3_gbm_squared")
        candidates.append(
            {
                "model": model_id,
                "wmape": _WMAPE[model_id],
                "bias": _BIAS[model_id],
                "gate1_pass": gate1_pass,
                "months_over_bias_threshold": ["2011-09"] if _BIAS[model_id] < -0.2 else [],
                "gate2_rank": 1 if model_id == CHAMPION else None,
                "gate3_compared_with": None,
                "gate3_fill_rate": None,
                "gate3_excess_units": None,
                "gate3_decision": None,
                "excluded_reason": (
                    "reference_only model, partial hold-out coverage (share=0.830000)"
                    if model_id == "B3_seasonal_naive"
                    else None
                ),
            }
        )
    return {
        "champion": CHAMPION,
        "champion_kind": "ml",
        "best_baseline": BEST_BASELINE,
        "meaningful_improvement": meaningful_improvement,
        "improvement_points": improvement_points,
        "gate1_all_failed": False,
        "candidates": candidates,
        "rules_version": "PRD-v1.3-section20",
        "config_gates": GATES.model_dump(mode="json"),
        "generated_at": "2026-01-01T00:00:00+00:00",
        "run_id": "fixture-run",
    }


def _write_fixture_artifacts(base_dir, *, meaningful_improvement: bool = True) -> None:
    """Populate ``base_dir`` with a schema-correct, minimal copy of every US-25 input."""
    eval_dir = base_dir / "artifacts" / "reports" / "evaluation_tables"
    _write_csv(eval_dir / "holdout_metrics_overall.csv", _overall_table())
    _write_csv(eval_dir / "holdout_metrics_by_month.csv", _by_month_table())
    _write_csv(eval_dir / "holdout_metrics_by_abc.csv", _by_abc_table())
    _write_csv(
        eval_dir / "improvement_vs_b2.csv",
        _improvement_table(meaningful_improvement=meaningful_improvement),
    )
    _write_csv(eval_dir / "backtest_consistency.csv", _consistency_table())
    _write_csv(eval_dir / "excess_concentration.csv", _excess_table())
    _write_csv(eval_dir / "sigma_summary.csv", _sigma_summary_table())
    _write_csv(eval_dir / "quarterly_metrics.csv", _quarterly_metrics_table())
    (eval_dir / "quarterly_limitation.md").write_text(
        "# Quarterly forecast — methodology limitation\n\n"
        "This project trains and evaluates a single one-step-ahead monthly model, so it "
        "cannot forecast all three months of a quarter at the start of the quarter.\n",
        encoding="utf-8",
    )

    _write_csv(base_dir / "artifacts" / "forecasts" / "inventory_kpis.csv", _kpis_table())

    _write_json(
        base_dir / "artifacts" / "reports" / "champion_decision.json",
        _champion_decision(meaningful_improvement=meaningful_improvement),
    )
    _write_json(
        base_dir / "artifacts" / "reports" / "data_quality_findings.json",
        {
            "waterfall": [
                {
                    "step_no": 1,
                    "step": "dataset_intake",
                    "rows_before": 1_067_371,
                    "rows_after": 1_067_371,
                },
                {
                    "step_no": 9,
                    "step": "final",
                    "rows_before": 1_003_400,
                    "rows_after": 1_003_338,
                },
            ]
        },
    )
    _write_json(
        base_dir / "artifacts" / "models" / "model_meta.json",
        {
            "champion": CHAMPION,
            "train_targets": {"start": "2010-03", "end": "2011-11"},
            "target_month": "2011-12",
        },
    )
    _write_json(
        base_dir / "artifacts" / "models" / "candidates_meta.json",
        {"models": [{"model_id": model_id, "n_train_rows": 52214} for model_id in MODEL_IDS]},
    )
    _write_json(
        base_dir / "artifacts" / "contracts" / "dataset_contract.json",
        {
            "source": (
                "UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci"
            ),
            "data_sha256": None,
            "row_counts": {
                "rows": 100717,
                "products": 4723,
                "zero_rows": 34853,
                "partial_rows": 4723,
            },
            "date_range": {
                "first_month": "2009-12",
                "last_full_month": "2011-11",
                "partial_months": ["2011-12"],
            },
        },
    )


def _run_ctx(tmp_path) -> RunContext:
    return RunContext.start(mode="no-llm", base_dir=tmp_path)


# --------------------------------------------------------------------------
# fixture-backed tests (issue §6 acceptance criteria)
# --------------------------------------------------------------------------
def test_evaluation_report_contains_required_sections_and_candidates(tmp_path) -> None:
    _write_fixture_artifacts(tmp_path)
    ctx = _run_ctx(tmp_path)
    try:
        destination = write_evaluation_report(ctx)
        text = destination.read_text(encoding="utf-8")

        for model_id in MODEL_IDS:
            assert model_id in text
        assert "Champion decision trace" in text
        assert "Back-test consistency" in text
        assert "Inventory KPIs" in text
        assert "Quarterly" in text
        assert "ABC computed on the training window" in text
        assert "dominated by a few very large orders" in text
        assert "Order Quantity" not in text
        assert CHAMPION in text
    finally:
        close_log_handlers(ctx.run_id)


def test_model_card_has_exactly_the_five_headings_plus_configuration_and_version(tmp_path) -> None:
    _write_fixture_artifacts(tmp_path)
    ctx = _run_ctx(tmp_path)
    try:
        destination = write_model_card(ctx)
        text = destination.read_text(encoding="utf-8")

        headings = [line for line in text.splitlines() if line.startswith("## ")]
        assert headings == list(MODEL_CARD_HEADINGS)

        for required in ("CC BY 4.0", "Chen", "Recommended Target Inventory", "does not guarantee"):
            assert required in text
        assert "Order Quantity" not in text
    finally:
        close_log_handlers(ctx.run_id)


def test_numbers_in_tables_passes_on_both_generated_files(tmp_path) -> None:
    _write_fixture_artifacts(tmp_path)
    ctx = _run_ctx(tmp_path)
    try:
        eval_path = write_evaluation_report(ctx)
        card_path = write_model_card(ctx)

        # write_* already raises on a failing guard; re-running it here proves the *published*
        # file, not just the in-memory string, is guard-clean once more numbers are on disk.
        broad_tables = {
            "overall": _overall_table(),
            "by_month": _by_month_table(),
            "by_abc": _by_abc_table(),
            "improvement": _improvement_table(meaningful_improvement=True),
            "consistency": _consistency_table(),
            "excess": _excess_table(),
            "sigma_summary": _sigma_summary_table(),
            "kpis": _kpis_table(),
            "quarterly_metrics": _quarterly_metrics_table(),
            "champion_decision": _champion_decision(meaningful_improvement=True),
            "config_gates": GATES.model_dump(mode="json"),
        }
        assert numbers_in_tables(eval_path.read_text(encoding="utf-8"), broad_tables).checked >= 0
        assert numbers_in_tables(card_path.read_text(encoding="utf-8"), broad_tables).checked >= 0
    finally:
        close_log_handlers(ctx.run_id)


def test_write_all_reports_writes_both_and_registers_artifacts(tmp_path) -> None:
    _write_fixture_artifacts(tmp_path)
    ctx = _run_ctx(tmp_path)
    try:
        paths_written = write_all_reports(ctx)
        assert paths_written["evaluation_report"].is_file()
        assert paths_written["model_card"].is_file()
        assert ctx.artifacts["evaluation_report"] == "artifacts/reports/evaluation_report.md"
        assert ctx.artifacts["model_card"] == "artifacts/reports/model_card.md"
    finally:
        close_log_handlers(ctx.run_id)


def test_meaningful_improvement_false_produces_simple_methods_competitive(tmp_path) -> None:
    _write_fixture_artifacts(tmp_path, meaningful_improvement=False)
    ctx = _run_ctx(tmp_path)
    try:
        eval_text = write_evaluation_report(ctx).read_text(encoding="utf-8")
        card_text = write_model_card(ctx).read_text(encoding="utf-8")
        assert "Simple methods are competitive" in eval_text
        assert "Simple methods are competitive" in card_text
    finally:
        close_log_handlers(ctx.run_id)


def test_templates_exist() -> None:
    template_dir = paths.PROJECT_ROOT / "src" / "pipeline" / "templates"
    assert (template_dir / "evaluation_report.md.j2").is_file()
    assert (template_dir / "model_card.md.j2").is_file()


# --------------------------------------------------------------------------
# end-to-end against the real, committed US-19..US-24 artifacts
# --------------------------------------------------------------------------
#: Every canonical input path :func:`pipeline.reports._load_inputs` reads.
_REAL_REQUIRED_INPUTS = (
    paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv",
    paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv",
    paths.EVAL_TABLES_DIR / "holdout_metrics_by_abc.csv",
    paths.EVAL_TABLES_DIR / "improvement_vs_b2.csv",
    paths.EVAL_TABLES_DIR / "backtest_consistency.csv",
    paths.EXCESS_CONCENTRATION,
    paths.EVAL_TABLES_DIR / "sigma_summary.csv",
    paths.INVENTORY_KPIS,
    paths.QUARTERLY_METRICS,
    paths.QUARTERLY_LIMITATION,
    paths.CHAMPION_DECISION,
    paths.MODEL_META,
    paths.CANDIDATES_META,
    paths.DATASET_CONTRACT,
    paths.DATA_QUALITY_FINDINGS,
)


def test_write_all_reports_end_to_end_on_real_artifacts(tmp_path) -> None:
    if not all(path.is_file() for path in _REAL_REQUIRED_INPUTS):
        pytest.skip("US-19..US-24 artifacts not present in this checkout")

    # Copy the real inputs into an isolated tmp_path tree rather than pointing ctx at the real
    # repo — write_evaluation_report/write_model_card read and write through the same base_dir
    # (module docstring), so running against paths.PROJECT_ROOT directly would overwrite this
    # checkout's committed reports and run_log.json as a side effect of running the test suite.
    for source in _REAL_REQUIRED_INPUTS:
        relative = source.relative_to(paths.PROJECT_ROOT)
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        result = write_all_reports(ctx)
        assert result["evaluation_report"].is_file()
        assert result["model_card"].is_file()
        card_text = result["model_card"].read_text(encoding="utf-8")
        headings = [line for line in card_text.splitlines() if line.startswith("## ")]
        assert headings == list(MODEL_CARD_HEADINGS)
    finally:
        close_log_handlers(ctx.run_id)
