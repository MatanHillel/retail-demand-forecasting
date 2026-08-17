"""Inventory-policy tests (US-20, PRD §22, §25, §26, §27, §47, §55).

Currently covers only the robust-sigma part (:mod:`pipeline.sigma`, US-20). US-21 adds the
safety-stock / simulation tests to this same file, per the issue's own naming convention.

Fixtures build ``backtest_predictions.csv``-shaped frames directly — only the four columns
:mod:`pipeline.sigma` actually reads (``model, stock_code, target_month, residual``) — rather than
the full real back-test, so each test isolates exactly one part of the eligibility / fallback logic.
Real config (``load_inventory_policy()``) is used throughout, never hand-rolled thresholds, so a
config change is felt here rather than silently diverging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import load_inventory_policy
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.sigma import (
    SIGMA_SUMMARY_COLUMNS,
    SIGMA_TABLE_COLUMNS,
    robust_sigma_from_residuals,
    sigma_summary,
    sigma_table,
    write_sigma_outputs,
)

POLICY = load_inventory_policy()
LEVEL_PRODUCT, LEVEL_ABC_GROUP, LEVEL_GLOBAL = POLICY.sigma.fallback_levels
MIN_RESIDUALS = POLICY.sigma.min_residuals_product
MODEL = "M2_gbm_poisson"


def _row(stock_code: str, target_month: str, residual: float, model: str = MODEL) -> dict:
    return {
        "model": model,
        "stock_code": stock_code,
        "target_month": target_month,
        "residual": residual,
    }


def _abc(mapping: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame({"stock_code": list(mapping), "abc_class": list(mapping.values())})


# --------------------------------------------------------------------------
# robust_sigma_from_residuals — the formula itself (issue §2 worked example)
# --------------------------------------------------------------------------
def test_robust_sigma_worked_example_extreme_value_does_not_inflate_it() -> None:
    # median 3, |e-3| = [2,1,0,1,97], MAD = 1 -> sigma = 1.4826
    sigma = robust_sigma_from_residuals(np.array([1, 2, 3, 4, 100]), 1.4826)
    assert sigma == pytest.approx(1.4826)


def test_robust_sigma_empty_array_is_nan() -> None:
    assert np.isnan(robust_sigma_from_residuals(np.array([]), 1.4826))


def test_robust_sigma_never_negative_for_random_inputs() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=50)
    assert robust_sigma_from_residuals(values, 1.4826) >= 0.0


# --------------------------------------------------------------------------
# Level 1 -> Level 2: too few product residuals falls back to the pooled ABC group
# --------------------------------------------------------------------------
def test_fewer_than_min_residuals_falls_back_to_abc_group() -> None:
    t = "2011-06"
    # P1: 3 eligible residuals (< MIN_RESIDUALS). P3, same class, contributes 3 more so the pooled
    # group reaches exactly MIN_RESIDUALS.
    rows = [
        _row("P1", "2011-03", 1.0),
        _row("P1", "2011-04", 2.0),
        _row("P1", "2011-05", 3.0),
        _row("P1", t, 999.0),  # the row being forecast itself -> marks P1 "active" at t
        _row("P3", "2011-03", 4.0),
        _row("P3", "2011-04", 5.0),
        _row("P3", "2011-05", 6.0),
    ]
    assert len(rows) - 1 == 6  # 3 + 3 pooled residuals == MIN_RESIDUALS exactly
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A", "P3": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    p1 = table.set_index("stock_code").loc["P1"]

    assert p1["n_residuals_product"] == 3
    assert p1["sigma_source"] == LEVEL_ABC_GROUP


# --------------------------------------------------------------------------
# The target_month == t row must never count as a usable residual
# --------------------------------------------------------------------------
def test_sixth_residual_at_target_month_equal_t_is_excluded() -> None:
    t = "2011-07"
    rows = [
        _row("P1", "2011-01", 1.0),
        _row("P1", "2011-02", 2.0),
        _row("P1", "2011-03", 3.0),
        _row("P1", "2011-04", 4.0),
        _row("P1", "2011-05", 5.0),
        _row("P1", t, 6.0),  # target_month == t: NOT eligible, even though it "looks like" a 6th
    ]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    p1 = table.set_index("stock_code").loc["P1"]

    assert p1["n_residuals_product"] == 5
    assert p1["sigma_source"] != LEVEL_PRODUCT


# --------------------------------------------------------------------------
# Pooled residuals != mean of per-product sigmas (issue §2, plan worked example)
# --------------------------------------------------------------------------
def test_abc_group_pools_raw_residuals_not_the_mean_of_per_product_sigmas() -> None:
    t = "2011-06"
    rows = [
        _row("A1", "2011-03", 1.0),
        _row("A1", "2011-04", 2.0),
        _row("A1", "2011-05", 3.0),
        _row("A1", t, 999.0),
        _row("A2", "2011-03", 10.0),
        _row("A2", "2011-04", 12.0),
        _row("A2", "2011-05", 14.0),
        _row("A2", t, 999.0),
    ]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"A1": "A", "A2": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    sigmas = table.set_index("stock_code")["sigma"]

    # pooled = [1,2,3,10,12,14], median=6.5, |e-6.5|=[5.5,4.5,3.5,3.5,5.5,7.5], MAD=5.0
    pooled_expected = 1.4826 * 5.0
    mean_of_sigmas = (1.4826 * 1.0 + 1.4826 * 2.0) / 2  # what pooling must NOT equal (~2.22)

    assert sigmas["A1"] == pytest.approx(pooled_expected)
    assert sigmas["A2"] == pytest.approx(pooled_expected)
    assert sigmas["A1"] == pytest.approx(sigmas["A2"])  # same pooled group -> same sigma
    assert sigmas["A1"] != pytest.approx(mean_of_sigmas)


# --------------------------------------------------------------------------
# ABC classes are trusted exactly as given (training-window ABC), never recomputed
# --------------------------------------------------------------------------
def test_abc_class_column_matches_the_supplied_abc_train_df_exactly() -> None:
    t = "2011-06"
    rows = [_row("P1", "2011-05", 1.0), _row("P1", t, 999.0)]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "B"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    assert table.set_index("stock_code").loc["P1", "abc_class"] == "B"


# --------------------------------------------------------------------------
# Level 2 -> Level 3: ABC group itself too small falls back to the global pool
# --------------------------------------------------------------------------
def test_global_fallback_when_the_abc_group_is_also_too_small() -> None:
    t = "2011-06"
    rows = [
        # Class A: only 2 eligible residuals total (P1's own) -> below MIN_RESIDUALS at both
        # the product AND the abc_group level.
        _row("P1", "2011-04", 1.0),
        _row("P1", "2011-05", 2.0),
        _row("P1", t, 999.0),
        # Classes B/C supply enough residuals for a non-trivial global pool.
        _row("B1", "2011-01", 10.0),
        _row("B1", "2011-02", 11.0),
        _row("B1", "2011-03", 12.0),
        _row("C1", "2011-01", 20.0),
        _row("C1", "2011-02", 21.0),
        _row("C1", "2011-03", 22.0),
    ]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A", "B1": "B", "C1": "C"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    p1 = table.set_index("stock_code").loc["P1"]

    assert p1["sigma_source"] == LEVEL_GLOBAL


# --------------------------------------------------------------------------
# Residual sign convention preserved (never recomputed), sigma never negative
# --------------------------------------------------------------------------
def test_sigma_is_never_negative_and_residual_values_are_used_as_given() -> None:
    t = "2011-06"
    rows = [
        _row("P1", "2011-01", -5.0),
        _row("P1", "2011-02", 3.0),
        _row("P1", "2011-03", -1.0),
        _row("P1", "2011-04", 8.0),
        _row("P1", "2011-05", -2.0),
        _row("P1", "2011-05", 0.0),  # a second row same month is fine, just more history
        _row("P1", t, 999.0),
    ]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    assert (table["sigma"].dropna() >= 0).all()

    expected = robust_sigma_from_residuals(
        np.array([-5.0, 3.0, -1.0, 8.0, -2.0, 0.0]), POLICY.sigma.mad_scale
    )
    assert table.set_index("stock_code").loc["P1", "sigma"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# zero_mad: constant-ish residuals -> MAD == 0 -> sigma == 0, flagged and counted
# --------------------------------------------------------------------------
def test_zero_mad_is_flagged_and_reflected_in_the_summary() -> None:
    t = "2011-07"
    rows = [
        _row("P1", "2011-01", 1.0),
        _row("P1", "2011-02", 2.0),
        _row("P1", "2011-03", 3.0),
        _row("P1", "2011-04", 3.0),
        _row("P1", "2011-05", 3.0),
        _row("P1", "2011-06", 3.0),  # 6 residuals, median 3, MAD == 0
        _row("P1", t, 999.0),
    ]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    p1 = table.set_index("stock_code").loc["P1"]
    assert p1["n_residuals_product"] == MIN_RESIDUALS
    assert p1["sigma"] == pytest.approx(0.0)
    assert bool(p1["zero_mad"]) is True

    summary = sigma_summary(table, POLICY)
    assert summary.loc[0, "share_zero_mad"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Leakage test (mandatory, issue §6 AC) — future residuals must never move sigma at t
# --------------------------------------------------------------------------
def test_altering_residuals_at_or_after_t_leaves_sigma_for_t_unchanged() -> None:
    t = "2011-07"
    base_rows = [
        _row("P1", "2011-01", 1.0),
        _row("P1", "2011-02", 2.0),
        _row("P1", "2011-03", 3.0),
        _row("P1", "2011-04", 4.0),
        _row("P1", "2011-05", 5.0),
        _row("P1", "2011-06", 6.0),
        _row("P1", t, 50.0),  # target_month == t
        _row("P1", "2011-08", -30.0),  # target_month > t
    ]
    abc_train_df = _abc({"P1": "A"})

    baseline_df = pd.DataFrame(base_rows)
    mutated_rows = [
        {**row, "residual": row["residual"] * 1000.0}
        if row["target_month"] >= t
        else row
        for row in base_rows
    ]
    mutated_df = pd.DataFrame(mutated_rows)

    # Sanity: the mutation actually changed something at/after t, and nothing before t.
    assert not baseline_df.equals(mutated_df)
    before_t = baseline_df["target_month"] < t
    pd.testing.assert_frame_equal(baseline_df.loc[before_t], mutated_df.loc[before_t])

    baseline = sigma_table(baseline_df, abc_train_df, [t], MODEL, POLICY)
    mutated = sigma_table(mutated_df, abc_train_df, [t], MODEL, POLICY)

    pd.testing.assert_frame_equal(baseline, mutated)


# --------------------------------------------------------------------------
# guard: robust sigma only, never np.std (CLAUDE.md §2 rule 8)
# --------------------------------------------------------------------------
def test_forbidden_np_std_never_appears_in_sigma_module() -> None:
    text = (paths.PROJECT_ROOT / "src" / "pipeline" / "sigma.py").read_text(encoding="utf-8")
    for token in ("np.std", ".std(", "1.4826"):
        assert token not in text, token


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
def test_sigma_table_and_summary_column_order() -> None:
    t = "2011-06"
    rows = [_row("P1", "2011-05", 1.0), _row("P1", t, 999.0)]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    assert list(table.columns) == SIGMA_TABLE_COLUMNS
    assert set(table["sigma_source"]).issubset({LEVEL_PRODUCT, LEVEL_ABC_GROUP, LEVEL_GLOBAL})

    summary = sigma_summary(table, POLICY)
    assert list(summary.columns) == SIGMA_SUMMARY_COLUMNS


# --------------------------------------------------------------------------
# artifacts written through ctx.out() (issue §8), tmp_path isolated
# --------------------------------------------------------------------------
def test_write_sigma_outputs_writes_through_ctx_out(tmp_path) -> None:
    t = "2011-06"
    rows = [_row("P1", "2011-05", 1.0), _row("P1", t, 999.0)]
    backtest_df = pd.DataFrame(rows)
    abc_train_df = _abc({"P1": "A"})

    table = sigma_table(backtest_df, abc_train_df, [t], MODEL, POLICY)
    summary = sigma_summary(table, POLICY)

    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        table_path, summary_path = write_sigma_outputs(table, summary, ctx)
        assert table_path == tmp_path / "artifacts" / "forecasts" / "sigma_table.csv"
        assert summary_path == (
            tmp_path / "artifacts" / "reports" / "evaluation_tables" / "sigma_summary.csv"
        )
        assert table_path.is_file()
        assert summary_path.is_file()
        assert ctx.artifacts["sigma_table"] == "artifacts/forecasts/sigma_table.csv"
        assert (
            ctx.artifacts["sigma_summary"]
            == "artifacts/reports/evaluation_tables/sigma_summary.csv"
        )
    finally:
        close_log_handlers(ctx.run_id)
