"""Inventory-policy tests (US-20 / US-21, PRD §22, §25, §26, §27, §28, §29, §30, §47, §55).

Covers the robust-sigma part (:mod:`pipeline.sigma`, US-20) and the safety-stock / simulation part
(:mod:`pipeline.inventory`, US-21), in one file per the issue's own naming convention.

Fixtures build minimal frames directly — only the columns each function actually reads — rather
than the full real artifacts, so each test isolates exactly one part of the eligibility / fallback
/ simulation logic. One US-21 test is the exception (marked ``slow``): it runs
``run_inventory_simulation`` against the real, committed ``holdout_rows_all_models.csv`` /
``sigma_table.csv``, because that function reads those two files from their canonical
:mod:`pipeline.paths` locations itself (issue §2's own signature), so there is no smaller fixture
to substitute on the read side — only the write side is redirected, via ``ctx`` ``base_dir``.

Real config (``load_inventory_policy()``) is used throughout, never hand-rolled thresholds, so a
config change is felt here rather than silently diverging.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import MODEL_IDS, load_inventory_policy, load_model_config
from pipeline.inventory import (
    EXCESS_CONCENTRATION_COLUMNS,
    KPI_COLUMNS,
    POLICIES,
    POLICY_FORECAST_ONLY,
    ROWS_INPUT_COLUMNS,
    SCOPE_ABC,
    SCOPE_MONTH,
    SCOPE_OVERALL,
    SIM_ROWS_COLUMNS,
    STEP_NAME,
    build_simulation_rows,
    excess_concentration,
    inventory_kpis,
    run_inventory_simulation,
    safety_stock,
    simulate_inventory,
    target_inventory,
    validate_simulation_inputs,
)
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


def _sim_row(
    stock_code: str,
    target_month: str,
    actual: float,
    forecast: float,
    sigma: float,
    model: str = MODEL,
    abc_class: str = "A",
    sigma_source: str = "product",
) -> dict:
    return {
        "stock_code": stock_code,
        "target_month": target_month,
        "abc_class": abc_class,
        "actual": actual,
        "model": model,
        "forecast": forecast,
        "sigma": sigma,
        "sigma_source": sigma_source,
    }


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


# ============================================================================
# US-21 — safety_stock, target_inventory, simulate_inventory, inventory_kpis,
# build_simulation_rows, validate_simulation_inputs, run_inventory_simulation
# ============================================================================


# --------------------------------------------------------------------------
# safety_stock / target_inventory — the two formulas of §25 / §28 (issue §2 worked example)
# --------------------------------------------------------------------------
def test_safety_stock_is_z_times_sigma() -> None:
    assert safety_stock(70, 1.645) == pytest.approx(70 * 1.645)


def test_target_inventory_worked_example() -> None:
    # 820 + 1.645 x 70 = 935.15 -> ceil -> 936 (issue §2)
    assert target_inventory(820, 70, 1.645) == 936


def test_target_inventory_is_a_nonnegative_integer() -> None:
    result = target_inventory(5, 100, 1.645)
    assert isinstance(result, int)
    assert result >= 0


def test_target_inventory_never_negative_for_a_very_negative_forecast() -> None:
    assert target_inventory(-500, 0, 1.645) == 0


def test_target_inventory_array_input_returns_int_array() -> None:
    result = target_inventory(np.array([820.0, -10.0]), np.array([70.0, 0.0]), 1.645)
    assert result.dtype.kind in "iu"
    assert list(result) == [936, 0]


# --------------------------------------------------------------------------
# simulate_inventory — forecast_only ignores sigma entirely (§29)
# --------------------------------------------------------------------------
def test_forecast_only_policy_target_equals_ceil_max_zero_forecast() -> None:
    rows_df = pd.DataFrame(
        [
            _sim_row("P1", "2011-06", actual=10.0, forecast=-3.5, sigma=50.0),
            _sim_row("P2", "2011-06", actual=20.0, forecast=12.4, sigma=5.0),
        ]
    )
    sim = simulate_inventory(rows_df, POLICY.z, POLICY)
    only = sim.loc[sim["policy"] == POLICY_FORECAST_ONLY]
    for _, row in only.iterrows():
        expected = int(np.ceil(max(0.0, row["forecast"])))
        assert row["target_inventory"] == expected


# --------------------------------------------------------------------------
# simulate_inventory — shortage/excess/fulfilled identities (issue §6 AC), any forecast/sigma
# --------------------------------------------------------------------------
def test_fulfilled_shortage_excess_identities_hold_for_random_rows() -> None:
    rng = np.random.default_rng(7)
    n = 25
    rows_df = pd.DataFrame(
        [
            _sim_row(
                f"P{i}",
                "2011-06",
                actual=float(rng.uniform(0, 100)),
                forecast=float(rng.uniform(-10, 100)),
                sigma=float(rng.uniform(0, 30)),
            )
            for i in range(n)
        ]
    )
    sim = simulate_inventory(rows_df, POLICY.z, POLICY)
    np.testing.assert_allclose(sim["fulfilled"] + sim["shortage"], sim["actual"])
    np.testing.assert_allclose(
        sim["fulfilled"] + sim["excess"], sim["target_inventory"].astype(float)
    )


def test_simulate_inventory_rejects_nonpositive_z() -> None:
    rows_df = pd.DataFrame([_sim_row("P1", "2011-06", actual=1.0, forecast=1.0, sigma=1.0)])
    with pytest.raises(ValueError):
        simulate_inventory(rows_df, 0.0, POLICY)


def test_simulate_inventory_column_order() -> None:
    rows_df = pd.DataFrame([_sim_row("P1", "2011-06", actual=1.0, forecast=1.0, sigma=1.0)])
    sim = simulate_inventory(rows_df, POLICY.z, POLICY)
    assert list(sim.columns) == SIM_ROWS_COLUMNS
    assert set(sim["policy"]) == set(POLICIES)


def test_simulate_inventory_z_option_rows_exist_for_each_configured_z() -> None:
    rows_df = pd.DataFrame([_sim_row("P1", "2011-06", actual=10.0, forecast=8.0, sigma=2.0)])
    combined = pd.concat(
        [simulate_inventory(rows_df, z, POLICY) for z in POLICY.z_options], ignore_index=True
    )
    for z in POLICY.z_options:
        assert np.isclose(combined["z"], z).any(), z


# --------------------------------------------------------------------------
# inventory_kpis — 3-row hand-built example (issue §6 AC), matches hand computation
# --------------------------------------------------------------------------
def test_inventory_kpis_matches_hand_computation_on_a_three_row_example() -> None:
    # Row 1: actual 10, target 8 -> shortage 2, excess 0, fulfilled 8
    # Row 2: actual 5,  target 5 -> shortage 0, excess 0, fulfilled 5
    # Row 3: actual 3,  target 6 -> shortage 0, excess 3, fulfilled 3
    sim_df = pd.DataFrame(
        {
            "model": [MODEL] * 3,
            "policy": [POLICY_FORECAST_ONLY] * 3,
            "z": [POLICY.z] * 3,
            "actual": [10.0, 5.0, 3.0],
            "target_inventory": [8, 5, 6],
            "shortage": [2.0, 0.0, 0.0],
            "excess": [0.0, 0.0, 3.0],
            "fulfilled": [8.0, 5.0, 3.0],
        }
    )
    kpis = inventory_kpis(sim_df, [])
    assert len(kpis) == 1
    row = kpis.iloc[0]

    assert row["fill_rate"] == pytest.approx((8.0 + 5.0 + 3.0) / (10.0 + 5.0 + 3.0))
    assert row["stockout_units"] == pytest.approx(2.0)
    assert row["excess_units"] == pytest.approx(3.0)
    assert row["stockout_skumonth_rate"] == pytest.approx(1 / 3)
    assert row["excess_per_unit_shortage"] == pytest.approx(3.0 / 2.0)
    assert row["n_rows"] == 3


def test_inventory_kpis_excess_per_unit_shortage_is_nan_when_no_shortage() -> None:
    sim_df = pd.DataFrame(
        {
            "model": [MODEL],
            "policy": [POLICY_FORECAST_ONLY],
            "z": [POLICY.z],
            "actual": [5.0],
            "target_inventory": [5],
            "shortage": [0.0],
            "excess": [0.0],
            "fulfilled": [5.0],
        }
    )
    kpis = inventory_kpis(sim_df, [])
    assert np.isnan(kpis.iloc[0]["excess_per_unit_shortage"])


# --------------------------------------------------------------------------
# excess_concentration — top slice share never exceeds 1, requires at least one large outlier
# --------------------------------------------------------------------------
def test_excess_concentration_top_slice_dominates_a_skewed_distribution() -> None:
    n = 200
    excess = np.zeros(n)
    excess[0] = 10_000.0  # one very large excess order among many zeros
    sim_df = pd.DataFrame(
        {
            "model": [MODEL] * n,
            "policy": [POLICY_FORECAST_ONLY] * n,
            "excess": excess,
        }
    )
    table = excess_concentration(sim_df)
    assert list(table.columns) == EXCESS_CONCENTRATION_COLUMNS
    row = table.iloc[0]
    assert row["top_1pct_share"] == pytest.approx(1.0)
    assert row["top_5pct_share"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# build_simulation_rows — drop-and-report rows any model lacks a forecast for (module docstring)
# --------------------------------------------------------------------------
def test_build_simulation_rows_drops_rows_any_model_lacks_a_forecast() -> None:
    wide_df = pd.DataFrame(
        {
            "stock_code": ["P1", "P2"],
            "target_month": ["2011-06", "2011-06"],
            "abc_class": ["A", "A"],
            "actual": [10.0, 5.0],
            **{f"pred_{model_id}": [1.0, 2.0] for model_id in MODEL_IDS},
        }
    )
    wide_df.loc[1, "pred_B3_seasonal_naive"] = np.nan  # P2: B3 has no forecast

    sigma_df = pd.DataFrame(
        {
            "stock_code": ["P1"] * len(MODEL_IDS),
            "target_month": ["2011-06"] * len(MODEL_IDS),
            "model": list(MODEL_IDS),
            "sigma": [1.0] * len(MODEL_IDS),
            "sigma_source": ["product"] * len(MODEL_IDS),
        }
    )

    rows_df, dropped = build_simulation_rows(wide_df, sigma_df, MODEL_IDS)

    assert list(rows_df.columns) == ROWS_INPUT_COLUMNS
    assert dropped["stock_code"].tolist() == ["P2"]
    assert set(rows_df["stock_code"]) == {"P1"}
    assert len(rows_df) == len(MODEL_IDS)  # every model kept for the one surviving row


# --------------------------------------------------------------------------
# validate_simulation_inputs — a genuine ValidationResult, not a bare assert (issue §8)
# --------------------------------------------------------------------------
def test_validate_simulation_inputs_passes_on_identical_nan_free_rows() -> None:
    rows_df = pd.DataFrame(
        [
            _sim_row("P1", "2011-06", actual=10.0, forecast=8.0, sigma=1.0, model=model_id)
            for model_id in MODEL_IDS
        ]
    )
    result = validate_simulation_inputs(rows_df, MODEL_IDS)
    assert result.passed is True
    assert result.violations == []
    assert result.step == STEP_NAME


def test_validate_simulation_inputs_fires_when_a_model_lacks_rows() -> None:
    rows_df = pd.DataFrame(
        [
            _sim_row("P1", "2011-06", actual=10.0, forecast=8.0, sigma=1.0, model="B1_last_month"),
            _sim_row("P2", "2011-06", actual=5.0, forecast=4.0, sigma=1.0, model="B1_last_month"),
            _sim_row("P1", "2011-06", actual=10.0, forecast=9.0, sigma=1.0, model="B2_ma3"),
            # B2_ma3 is missing the P2 row that B1_last_month has.
        ]
    )
    result = validate_simulation_inputs(rows_df, model_ids=("B1_last_month", "B2_ma3"))
    assert result.passed is False
    assert any(v.rule == "identical_rows" for v in result.violations)


def test_validate_simulation_inputs_catches_nan_forecast() -> None:
    rows_df = pd.DataFrame(
        [
            _sim_row("P1", "2011-06", actual=10.0, forecast=8.0, sigma=1.0, model=model_id)
            for model_id in MODEL_IDS
        ]
    )
    rows_df.loc[0, "forecast"] = np.nan
    result = validate_simulation_inputs(rows_df, MODEL_IDS)
    assert result.passed is False
    assert any(v.rule == "no_nan_inputs" for v in result.violations)


# --------------------------------------------------------------------------
# guard: no literal z / mad_scale, and never "order quantity" (CLAUDE.md §2 rules 4, 10)
# --------------------------------------------------------------------------
def test_forbidden_literals_and_wording_never_appear_in_inventory_module() -> None:
    text = (paths.PROJECT_ROOT / "src" / "pipeline" / "inventory.py").read_text(encoding="utf-8")
    for token in ("1.645", "1.4826", "order quantity", "Order Quantity"):
        assert token not in text, token


# --------------------------------------------------------------------------
# run_inventory_simulation — end to end against the real, committed US-19 / US-20 artifacts
# (module docstring: the read side cannot be substituted with a smaller fixture; only the write
# side is redirected, via ctx base_dir). Slow: processes the full hold-out x every candidate.
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_run_inventory_simulation_end_to_end_writes_three_artifacts(tmp_path) -> None:
    cfg = load_model_config()
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        with ctx.step(STEP_NAME):
            result = run_inventory_simulation(cfg, ctx)

        rows_path = tmp_path / "artifacts" / "forecasts" / "holdout_simulation_rows.csv"
        kpis_path = tmp_path / "artifacts" / "forecasts" / "inventory_kpis.csv"
        concentration_path = (
            tmp_path / "artifacts" / "reports" / "evaluation_tables" / "excess_concentration.csv"
        )
        assert rows_path.is_file()
        assert kpis_path.is_file()
        assert concentration_path.is_file()
        assert (
            ctx.artifacts["holdout_simulation_rows"]
            == "artifacts/forecasts/holdout_simulation_rows.csv"
        )
        assert ctx.artifacts["inventory_kpis"] == "artifacts/forecasts/inventory_kpis.csv"
        assert (
            ctx.artifacts["excess_concentration"]
            == "artifacts/reports/evaluation_tables/excess_concentration.csv"
        )

        kpis = result["inventory_kpis"]
        assert list(kpis.columns) == KPI_COLUMNS
        assert set(kpis["policy"]) == set(POLICIES)
        assert set(kpis["scope"]) == {SCOPE_OVERALL, SCOPE_MONTH, SCOPE_ABC}
        for z in POLICY.z_options:
            assert np.isclose(kpis["z"], z).any(), z

        rows = result["holdout_simulation_rows"]
        assert list(rows.columns) == SIM_ROWS_COLUMNS
        assert (rows["target_inventory"] >= 0).all()
        row_sets = {
            model_id: frozenset(zip(group["stock_code"], group["target_month"], strict=True))
            for model_id, group in rows.groupby("model", sort=False)
        }
        reference = next(iter(row_sets.values()))
        assert all(row_set == reference for row_set in row_sets.values())
    finally:
        close_log_handlers(ctx.run_id)
