"""Hold-out & back-test evaluation table tests (US-19, PRD §21, §22, §23, §55).

Synthetic predictions, built at the shape :func:`pipeline.evaluate.evaluate` actually consumes:
``holdout_predictions.csv`` (M1-M4, fixed model) and ``backtest_predictions.csv`` (all seven
candidates, rolling origins) — never full features/panel frames, since ``evaluate()`` takes neither
(issue §8: baselines are sliced out of the back-test, not recomputed). Real hold-out target months
from ``model_config.yaml`` are used throughout (§21 is the project's one fixed temporal design), so
these tests fail if the split configuration itself changes shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths
from pipeline.baselines import B1, B2, B3, BASELINE_MODEL_IDS
from pipeline.config import MODEL_IDS, load_model_config
from pipeline.evaluate import (
    BACKTEST_CONSISTENCY_COLUMNS,
    BY_ABC_COLUMNS,
    BY_MONTH_COLUMNS,
    IMPROVEMENT_VS_B2_COLUMNS,
    OVERALL_COLUMNS,
    WIDE_COLUMNS,
    evaluate,
)
from pipeline.models import HOLDOUT_PREDICTIONS_COLUMNS, TRAINABLE_MODEL_IDS
from pipeline.run_context import RunContext, close_log_handlers

CFG = load_model_config()
HOLDOUT_MONTHS = [
    str(period)
    for period in pd.period_range(
        CFG.split.holdout_targets.start, CFG.split.holdout_targets.end, freq="M"
    )
]
assert len(HOLDOUT_MONTHS) == 6

_PRODUCTS = {
    "P1": {"actual": 10.0, "abc_class": "A"},
    "P2": {"actual": 20.0, "abc_class": "C"},
}

# Fixed prediction offsets per trainable model, applied to each product's actual (deterministic,
# so wMAPE/Bias are hand-verifiable).
_M_OFFSETS = {
    "M1_linear": 2.0,
    "M2_gbm_poisson": -1.0,
    "M3_gbm_squared": 0.0,
    "M4_gbm_absolute": -2.0,
}
_BASELINE_OFFSETS = {B1: 1.0, B2: 0.5, B3: -1.0}


def _origin(target_month: str) -> str:
    return str(pd.Period(target_month, freq="M") - 1)


def _holdout_predictions_df() -> pd.DataFrame:
    """Synthetic ``holdout_predictions.csv`` shape: M1-M4 only, one row per (product, month)."""
    rows = []
    for stock_code, spec in _PRODUCTS.items():
        for month in HOLDOUT_MONTHS:
            for model_id in TRAINABLE_MODEL_IDS:
                prediction = spec["actual"] + _M_OFFSETS[model_id]
                rows.append(
                    {
                        "stock_code": stock_code,
                        "forecast_origin": _origin(month),
                        "target_month": month,
                        "model": model_id,
                        "actual": spec["actual"],
                        "prediction_raw": prediction,
                        "prediction": prediction,
                    }
                )
    return pd.DataFrame(rows)[HOLDOUT_PREDICTIONS_COLUMNS]


def _backtest_predictions_df() -> pd.DataFrame:
    """Synthetic ``backtest_predictions.csv`` shape covering all seven candidates.

    Baselines get one row per (product, holdout target month) — enough for the hold-out slice
    :func:`pipeline.evaluate._combined_holdout_predictions` performs. ``B3`` is left ``NaN`` for
    ``P2``'s first hold-out month only, so the coverage/scored-filtering path is exercised. Two
    extra pre-hold-out origins are added for ``B2`` with a large bias, so
    ``months_abs_bias_gt_threshold`` has a non-trivial answer.
    """
    rows = []
    for stock_code, spec in _PRODUCTS.items():
        for month in HOLDOUT_MONTHS:
            for model_id in (*TRAINABLE_MODEL_IDS, *BASELINE_MODEL_IDS):
                if model_id in BASELINE_MODEL_IDS:
                    if model_id == B3 and stock_code == "P2" and month == HOLDOUT_MONTHS[0]:
                        prediction = float("nan")
                    else:
                        prediction = spec["actual"] + _BASELINE_OFFSETS[model_id]
                else:
                    prediction = spec["actual"] + _M_OFFSETS[model_id]
                actual = spec["actual"]
                is_nan = prediction != prediction  # noqa: PLR0124 - NaN check without numpy/pandas
                rows.append(
                    {
                        "forecast_origin": _origin(month),
                        "target_month": month,
                        "stock_code": stock_code,
                        "model": model_id,
                        "actual": actual,
                        "prediction_raw": prediction,
                        "prediction": prediction,
                        "residual": float("nan") if is_nan else actual - prediction,
                    }
                )

    # Two extra pre-hold-out origins for B2 with a deliberately large bias, so
    # months_abs_bias_gt_threshold (default threshold 0.25) has at least one hit.
    extra_months = ["2011-04", "2011-05"]
    for stock_code, spec in _PRODUCTS.items():
        for month in extra_months:
            prediction = spec["actual"] * 2.0  # +100% -> bias way past any sane threshold
            rows.append(
                {
                    "forecast_origin": _origin(month),
                    "target_month": month,
                    "stock_code": stock_code,
                    "model": B2,
                    "actual": spec["actual"],
                    "prediction_raw": prediction,
                    "prediction": prediction,
                    "residual": spec["actual"] - prediction,
                }
            )

    return pd.DataFrame(rows)


def _abc_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_code": code,
                "abc_class": spec["abc_class"],
                "revenue_share": 0.1,
                "cum_share": 0.1,
            }
            for code, spec in _PRODUCTS.items()
        ]
    )


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict[str, pd.DataFrame]:
    base_dir = tmp_path_factory.mktemp("evaluate")
    ctx = RunContext.start(mode="no-llm", base_dir=base_dir)
    try:
        out = evaluate(
            _holdout_predictions_df(), _backtest_predictions_df(), _abc_train_df(), CFG, ctx
        )
        out["_ctx"] = ctx
    finally:
        close_log_handlers(ctx.run_id)
    return out


# --------------------------------------------------------------------------
# holdout_metrics_overall — 7 rows, wMAPE and Bias always both present (§23)
# --------------------------------------------------------------------------
def test_overall_has_seven_rows_with_wmape_and_bias(result: dict[str, pd.DataFrame]) -> None:
    overall = result["holdout_metrics_overall"]
    assert list(overall.columns) == OVERALL_COLUMNS
    assert len(overall) == 7
    assert set(overall["model"]) == set(MODEL_IDS)
    assert overall[["wmape", "bias"]].notna().all().all()


def test_overall_model_order_is_canonical(result: dict[str, pd.DataFrame]) -> None:
    overall = result["holdout_metrics_overall"]
    assert list(overall["model"]) == list(MODEL_IDS)


def test_overall_b3_coverage_share_reflects_the_missing_row(
    result: dict[str, pd.DataFrame],
) -> None:
    overall = result["holdout_metrics_overall"].set_index("model")
    # 2 products x 6 months = 12 total rows for B3, one dropped (NaN) -> 11/12 coverage.
    assert overall.loc[B3, "n_rows"] == 11
    assert overall.loc[B3, "coverage_share"] == pytest.approx(11 / 12)
    assert overall.loc[B3, "note"] != ""
    for model_id in set(MODEL_IDS) - {B3}:
        assert overall.loc[model_id, "coverage_share"] == pytest.approx(1.0)
        assert overall.loc[model_id, "note"] == ""


# --------------------------------------------------------------------------
# holdout_metrics_by_month — 6 hold-out months x 7 models, no 2011-12 (§21)
# --------------------------------------------------------------------------
def test_by_month_has_six_months_times_seven_models(result: dict[str, pd.DataFrame]) -> None:
    by_month = result["holdout_metrics_by_month"]
    assert list(by_month.columns) == BY_MONTH_COLUMNS
    assert len(by_month) == 6 * 7  # B3 misses one *row*, not a whole (model, month) group
    assert set(by_month["target_month"]) == set(HOLDOUT_MONTHS)
    assert "2011-12" not in set(by_month["target_month"])


# --------------------------------------------------------------------------
# holdout_metrics_by_abc — training-window classes only, trusted as given (§18.2, §23, §27)
# --------------------------------------------------------------------------
def test_by_abc_uses_exactly_the_supplied_training_window_classes(
    result: dict[str, pd.DataFrame],
) -> None:
    by_abc = result["holdout_metrics_by_abc"]
    assert list(by_abc.columns) == BY_ABC_COLUMNS
    assert set(by_abc["abc_class"]) == {"A", "C"}  # exactly what _abc_train_df() supplied
    assert set(by_abc["model"]) == set(MODEL_IDS)


def test_by_abc_raises_when_a_product_is_missing_from_abc_train(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        incomplete_abc = _abc_train_df().iloc[[0]]  # drops P2
        with pytest.raises(ValueError, match="absent from abc_train_df"):
            evaluate(
                _holdout_predictions_df(), _backtest_predictions_df(), incomplete_abc, CFG, ctx
            )
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# improvement_vs_b2 — B2 vs itself is exactly 0 (issue §2 test list)
# --------------------------------------------------------------------------
def test_improvement_vs_b2_for_b2_itself_is_zero(result: dict[str, pd.DataFrame]) -> None:
    improvement = result["improvement_vs_b2"]
    assert list(improvement.columns) == IMPROVEMENT_VS_B2_COLUMNS
    b2_row = improvement.set_index("model").loc[B2]
    assert b2_row["wmape_points_vs_b2"] == pytest.approx(0.0)
    assert b2_row["relative_pct"] == pytest.approx(0.0)
    assert bool(b2_row["meaningful"]) is False


def test_improvement_vs_b2_meaningful_flag_matches_config_threshold(
    result: dict[str, pd.DataFrame],
) -> None:
    improvement = result["improvement_vs_b2"].set_index("model")
    threshold = CFG.champion_gates.meaningful_improvement_points
    for _model_id, row in improvement.iterrows():
        assert bool(row["meaningful"]) == (row["wmape_points_vs_b2"] >= threshold)


# --------------------------------------------------------------------------
# backtest_consistency — counts months with |bias| > threshold correctly (issue §2 test list)
# --------------------------------------------------------------------------
def test_backtest_consistency_counts_months_over_the_bias_threshold(
    result: dict[str, pd.DataFrame],
) -> None:
    consistency = result["backtest_consistency"]
    assert list(consistency.columns) == BACKTEST_CONSISTENCY_COLUMNS

    threshold = CFG.champion_gates.monthly_abs_bias_report_threshold
    b2_rows = consistency.loc[consistency["model"] == B2]
    recomputed = int((b2_rows["bias"].abs() > threshold).sum())
    assert b2_rows["months_abs_bias_gt_threshold"].iloc[0] == recomputed
    # The two extra +100%-bias origins injected for B2 must be among the flagged months.
    assert recomputed >= 2

    for model_id in set(MODEL_IDS) - {B2}:
        rows = consistency.loc[consistency["model"] == model_id]
        recomputed_model = int((rows["bias"].abs() > threshold).sum())
        assert rows["months_abs_bias_gt_threshold"].iloc[0] == recomputed_model


# --------------------------------------------------------------------------
# holdout_rows_all_models — row count = active hold-out rows (issue §2 test list)
# --------------------------------------------------------------------------
def test_wide_table_row_count_equals_active_holdout_rows(result: dict[str, pd.DataFrame]) -> None:
    wide = result["holdout_rows_all_models"]
    assert list(wide.columns) == WIDE_COLUMNS
    assert len(wide) == len(_PRODUCTS) * len(HOLDOUT_MONTHS)  # 2 products x 6 months = 12
    assert set(wide["abc_class"]) == {"A", "C"}
    for model_id in MODEL_IDS:
        assert f"pred_{model_id}" in wide.columns
    # B3's one NaN row survives into the wide table rather than being dropped (module docstring).
    assert wide["pred_B3_seasonal_naive"].isna().sum() == 1


# --------------------------------------------------------------------------
# evaluation_summary.json content
# --------------------------------------------------------------------------
def test_evaluation_summary_fields(result: dict[str, pd.DataFrame]) -> None:
    summary = result["evaluation_summary"]
    assert summary["holdout_months"] == HOLDOUT_MONTHS
    assert summary["n_products"] == len(_PRODUCTS)
    assert summary["abc_source"] == f"training window through {CFG.split.train_targets.end}"
    assert summary["run_id"] == result["_ctx"].run_id
    assert len(summary["overall"]) == 7


# --------------------------------------------------------------------------
# row-set enforcement (technical note, issue §3)
# --------------------------------------------------------------------------
def test_mismatched_row_sets_between_sources_raise(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        holdout = _holdout_predictions_df()
        drop_mask = (holdout["stock_code"] == "P2") & (holdout["model"] == "M1_linear")
        holdout = holdout.loc[~drop_mask]
        with pytest.raises(AssertionError, match="do not cover the same"):
            evaluate(holdout, _backtest_predictions_df(), _abc_train_df(), CFG, ctx)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# artifacts written through ctx.out() at the correct names (issue §8)
# --------------------------------------------------------------------------
def test_all_seven_artifacts_are_written(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        evaluate(_holdout_predictions_df(), _backtest_predictions_df(), _abc_train_df(), CFG, ctx)
        eval_tables_dir = tmp_path / "artifacts" / "reports" / "evaluation_tables"
        expected = {
            "holdout_metrics_overall.csv",
            "holdout_metrics_by_month.csv",
            "holdout_metrics_by_abc.csv",
            "improvement_vs_b2.csv",
            "backtest_consistency.csv",
            "holdout_rows_all_models.csv",
            "evaluation_summary.json",
        }
        for name in expected:
            assert (eval_tables_dir / name).is_file(), name
        assert not (tmp_path / "artifacts" / "reports" / "evaluation_report.md").exists()
    finally:
        close_log_handlers(ctx.run_id)


def test_run_log_metrics_carries_wmape_and_bias_per_model(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        evaluate(_holdout_predictions_df(), _backtest_predictions_df(), _abc_train_df(), CFG, ctx)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    assert set(ctx.metrics["holdout"]) == set(MODEL_IDS)
    for model_id in MODEL_IDS:
        assert "wmape" in ctx.metrics["holdout"][model_id]
        assert "bias" in ctx.metrics["holdout"][model_id]
        assert isinstance(ctx.metrics["holdout"][model_id]["wmape"], float)


# --------------------------------------------------------------------------
# guard: forbidden literals never appear under src/ (CLAUDE.md §2 rule 4, issue §6)
# --------------------------------------------------------------------------
def test_forbidden_literals_never_appear_in_evaluate_module() -> None:
    text = (paths.PROJECT_ROOT / "src" / "pipeline" / "evaluate.py").read_text(encoding="utf-8")
    for token in ("0.25", "2.0", " 55", "train_test_split", "shuffle=True"):
        assert token not in text, token
