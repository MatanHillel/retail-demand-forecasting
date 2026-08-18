"""Quarterly aggregation tests (US-24, PRD §1, §6.1, §6.2, §31, §32, §50, §55).

Small hand-built frames isolate one rule each — the calendar-quarter mapping, the sum-of-three
formula, the completeness rule, the wMAPE/Bias arithmetic and the rolling operational estimate —
so the file stays fast and each failure points at exactly one thing. One test also runs
:func:`aggregate_quarterly` against the real, committed ``backtest_predictions.csv`` to prove the
documented complete-quarter set (2010-Q3 ... 2011-Q3) actually holds on real data, not just on a
fixture built to produce it.

Real configuration is used throughout (``load_model_config()`` / ``load_cleaning_config()``),
never a hand-rolled threshold, so a config change (e.g. the back-test window) is felt here rather
than silently diverging.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths
from pipeline.baselines import B2
from pipeline.config import load_cleaning_config, load_model_config
from pipeline.panel import PANEL_COLUMNS
from pipeline.quarterly import (
    LIMITATION_TEXT,
    QUARTERLY_FORECAST_COLUMNS,
    QUARTERLY_METRICS_COLUMNS,
    ROLLING_ESTIMATE_TYPE,
    SCOPE_OVERALL,
    SCOPE_QUARTER,
    aggregate_quarterly,
    default_models,
    quarter_label,
    quarter_months,
    quarterly_metrics,
    rolling_quarter_estimate,
    run_quarterly_aggregation,
)
from pipeline.run_context import RunContext, close_log_handlers

CFG = load_model_config()
CLEANING = load_cleaning_config()
CHAMPION = CFG.primary_model_id                # a fitted model, distinct from B2 (M2_gbm_poisson)
ORIGIN = CLEANING.raw.last_full_month           # 2011-11
TARGET = str(pd.Period(ORIGIN, freq="M") + 1)   # 2011-12


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _backtest_row(
    stock_code: str, target_month: str, model: str, actual: float, prediction: float
) -> dict:
    """One ``backtest_predictions.csv``-shaped row (only the columns this module reads)."""
    origin = str(pd.Period(target_month, freq="M") - 1)
    return {
        "forecast_origin": origin,
        "target_month": target_month,
        "stock_code": stock_code,
        "model": model,
        "actual": actual,
        "prediction_raw": prediction,
        "prediction": prediction,
        "residual": actual - prediction,
    }


def _panel_row(stock_code: str, month: str, units: float) -> dict:
    """One US-05 panel row. Only the columns this module reads carry real numbers."""
    return {
        "month": month,
        "stock_code": stock_code,
        "description": f"{stock_code} DESCRIPTION",
        "units_sold": units,
        "gross_revenue": units * 2.0,
        "avg_unit_price": 2.0,
        "invoice_count": 1 if units else 0,
        "sale_line_count": 1 if units else 0,
        "customer_count": 1 if units else 0,
        "max_line_qty": units,
        "returned_units": 0,
        "is_partial_month": month in CLEANING.raw.partial_months,
    }


@pytest.fixture
def ctx(tmp_path):
    """A run whose artifacts land in ``tmp_path`` instead of the repository."""
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


# --------------------------------------------------------------------------
# calendar-quarter mapping (issue §2 "quarter mapping")
# --------------------------------------------------------------------------
def test_quarter_label_maps_calendar_quarters():
    """2011-07/08/09 -> 2011-Q3, exactly as the issue's example specifies."""
    assert quarter_label("2011-07") == "2011-Q3"
    assert quarter_label("2011-08") == "2011-Q3"
    assert quarter_label("2011-09") == "2011-Q3"
    assert quarter_label("2011-10") == "2011-Q4"
    assert quarter_label("2011-01") == "2011-Q1"
    assert quarter_label("2010-12") == "2010-Q4"


def test_quarter_months_round_trips_with_quarter_label():
    assert quarter_months("2011-Q3") == ["2011-07", "2011-08", "2011-09"]
    for month in quarter_months("2011-Q3"):
        assert quarter_label(month) == "2011-Q3"


def test_default_models_dedupes_when_the_champion_is_b2():
    """The champion and B2 (issue §2) — deduplicated rather than a repeated column."""
    assert default_models(B2) == [B2]
    assert default_models(CHAMPION) == [CHAMPION, B2]


# --------------------------------------------------------------------------
# the sum-of-three-months formula & the completeness rule
# --------------------------------------------------------------------------
def test_quarterly_sum_equals_the_sum_of_the_three_monthly_rows():
    """A product with all three months present: forecast_sum/actual_sum are plain sums."""
    rows = [
        _backtest_row("A", "2011-07", B2, actual=10, prediction=12),
        _backtest_row("A", "2011-08", B2, actual=20, prediction=18),
        _backtest_row("A", "2011-09", B2, actual=30, prediction=33),
    ]
    qdf = aggregate_quarterly(pd.DataFrame(rows), [B2], CFG)

    assert len(qdf) == 1
    row = qdf.iloc[0]
    assert row["quarter"] == "2011-Q3"
    assert row["forecast_sum"] == pytest.approx(12 + 18 + 33)
    assert row["actual_sum"] == pytest.approx(10 + 20 + 30)
    assert row["n_months"] == 3
    assert bool(row["complete"]) is True
    assert row["months_included"] == "2011-07;2011-08;2011-09"
    assert list(qdf.columns) == QUARTERLY_FORECAST_COLUMNS


def test_a_quarter_with_two_months_present_is_incomplete_and_excluded_from_metrics():
    """§14/§21: a product missing one month of an otherwise-complete quarter is not scoreable."""
    rows = [
        _backtest_row("A", "2011-07", B2, actual=10, prediction=12),
        _backtest_row("A", "2011-08", B2, actual=20, prediction=18),
        _backtest_row("A", "2011-09", B2, actual=30, prediction=33),
        _backtest_row("B", "2011-07", B2, actual=5, prediction=6),
        _backtest_row("B", "2011-08", B2, actual=5, prediction=4),
    ]
    qdf = aggregate_quarterly(pd.DataFrame(rows), [B2], CFG)

    row_b = qdf.loc[qdf["stock_code"] == "B"].iloc[0]
    assert row_b["n_months"] == 2
    assert bool(row_b["complete"]) is False

    metrics = quarterly_metrics(qdf)
    overall = metrics.loc[(metrics["model"] == B2) & (metrics["scope"] == SCOPE_OVERALL)].iloc[0]
    # Only A's 60 actual units enter the metric; B's incomplete 10 units must not.
    assert overall["sum_actual"] == pytest.approx(60)


def test_aggregate_quarterly_ignores_rows_outside_the_backtest_window():
    """A defensive guard: a stray row before/after the genuine back-test range never contaminates
    a quarterly sum, even if a caller hands one in (§21: December 2011 must never be scored)."""
    rows = [
        _backtest_row("A", "2011-07", B2, actual=10, prediction=12),
        _backtest_row("A", "2011-08", B2, actual=20, prediction=18),
        _backtest_row("A", "2011-09", B2, actual=30, prediction=33),
        _backtest_row("A", "2011-12", B2, actual=999, prediction=999),  # partial month — never
        _backtest_row("A", "2010-01", B2, actual=999, prediction=999),  # before first_origin
    ]
    qdf = aggregate_quarterly(pd.DataFrame(rows), [B2], CFG)
    assert set(qdf["quarter"]) == {"2011-Q3"}
    assert "2011-12" not in ";".join(qdf["months_included"])


# --------------------------------------------------------------------------
# quarterly wMAPE / Bias, recomputed by hand
# --------------------------------------------------------------------------
def test_quarterly_wmape_and_bias_recomputed_by_hand():
    """A tiny two-quarter example, checked against the wMAPE/Bias formulas directly (§23)."""
    rows = [
        _backtest_row("C", "2011-04", B2, actual=30, prediction=27),
        _backtest_row("C", "2011-05", B2, actual=30, prediction=27),
        _backtest_row("C", "2011-06", B2, actual=30, prediction=27),  # Q2: actual 90, forecast 81
        _backtest_row("C", "2011-07", B2, actual=20, prediction=21),
        _backtest_row("C", "2011-08", B2, actual=20, prediction=21),
        _backtest_row("C", "2011-09", B2, actual=20, prediction=21),  # Q3: actual 60, forecast 63
    ]
    qdf = aggregate_quarterly(pd.DataFrame(rows), [B2], CFG)
    metrics = quarterly_metrics(qdf)

    overall = metrics.loc[(metrics["model"] == B2) & (metrics["scope"] == SCOPE_OVERALL)].iloc[0]
    expected_wmape = (abs(63 - 60) + abs(81 - 90)) / (60 + 90)
    expected_bias = ((63 - 60) + (81 - 90)) / (60 + 90)
    assert overall["wmape"] == pytest.approx(expected_wmape)
    assert overall["bias"] == pytest.approx(expected_bias)
    assert overall["quarter"] == "all"

    is_q3 = (
        (metrics["model"] == B2)
        & (metrics["scope"] == SCOPE_QUARTER)
        & (metrics["quarter"] == "2011-Q3")
    )
    q3 = metrics.loc[is_q3].iloc[0]
    assert q3["wmape"] == pytest.approx(abs(63 - 60) / 60)
    assert q3["bias"] == pytest.approx((63 - 60) / 60)
    assert list(metrics.columns) == QUARTERLY_METRICS_COLUMNS


def test_quarterly_metrics_is_empty_but_well_formed_with_no_complete_quarters():
    rows = [_backtest_row("B", "2011-07", B2, actual=5, prediction=6)]
    qdf = aggregate_quarterly(pd.DataFrame(rows), [B2], CFG)
    metrics = quarterly_metrics(qdf)
    assert metrics.empty
    assert list(metrics.columns) == QUARTERLY_METRICS_COLUMNS


# --------------------------------------------------------------------------
# the rolling 2011-Q4 operational estimate (§32)
# --------------------------------------------------------------------------
def test_rolling_estimate_uses_actual_oct_and_nov_and_the_dec_forecast_only():
    panel = pd.DataFrame(
        [
            _panel_row("P1", "2011-10", 40),
            _panel_row("P1", "2011-11", 50),
            _panel_row("P1", "2011-12", 999_999),  # must never be read — it is the partial month
        ]
    )[PANEL_COLUMNS]
    latest = pd.DataFrame(
        [{"stock_code": "P1", "status": "Forecast", "prediction": 25.0}]
    )

    result = rolling_quarter_estimate(latest, panel, CHAMPION, CLEANING)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["quarter"] == quarter_label(TARGET)
    assert row["model"] == CHAMPION
    assert row["forecast_sum"] == pytest.approx(40 + 50 + 25)
    assert pd.isna(row["actual_sum"])
    assert bool(row["complete"]) is False
    assert row["months_included"] == "2011-10;2011-11;2011-12"
    assert row["estimate_type"] == ROLLING_ESTIMATE_TYPE
    assert list(result.columns) == QUARTERLY_FORECAST_COLUMNS


def test_rolling_estimate_defaults_a_missing_panel_month_to_zero():
    """A product with no panel row for a prior month (should not happen for an active product, but
    the estimate must not raise or silently drop the product) contributes zero for that month."""
    panel = pd.DataFrame([_panel_row("P2", "2011-11", 10)])[PANEL_COLUMNS]
    latest = pd.DataFrame([{"stock_code": "P2", "status": "Forecast", "prediction": 5.0}])

    row = rolling_quarter_estimate(latest, panel, CHAMPION, CLEANING).iloc[0]
    assert row["forecast_sum"] == pytest.approx(0 + 10 + 5)


# --------------------------------------------------------------------------
# the documented limitation text (issue §6 acceptance criterion)
# --------------------------------------------------------------------------
def test_limitation_text_contains_the_required_phrase():
    phrase = "cannot forecast all three months of a quarter at the start of the quarter"
    assert phrase in LIMITATION_TEXT


# --------------------------------------------------------------------------
# run_quarterly_aggregation — the full chain, writing through ctx.out()
# --------------------------------------------------------------------------
def test_run_quarterly_aggregation_writes_three_artifacts(ctx):
    backtest = pd.DataFrame(
        [
            _backtest_row("A", "2011-07", B2, actual=10, prediction=12),
            _backtest_row("A", "2011-08", B2, actual=20, prediction=18),
            _backtest_row("A", "2011-09", B2, actual=30, prediction=33),
            _backtest_row("A", "2011-07", CHAMPION, actual=10, prediction=11),
            _backtest_row("A", "2011-08", CHAMPION, actual=20, prediction=19),
            _backtest_row("A", "2011-09", CHAMPION, actual=30, prediction=32),
        ]
    )
    panel = pd.DataFrame(
        [_panel_row("A", "2011-10", 40), _panel_row("A", "2011-11", 50)]
    )[PANEL_COLUMNS]
    latest = pd.DataFrame([{"stock_code": "A", "status": "Forecast", "prediction": 45.0}])

    result = run_quarterly_aggregation(
        CFG,
        ctx,
        backtest_df=backtest,
        latest_df=latest,
        panel_df=panel,
        cleaning_cfg=CLEANING,
        champion=CHAMPION,
    )

    forecast_path = ctx.base_dir / paths.QUARTERLY_FORECAST.relative_to(paths.PROJECT_ROOT)
    metrics_path = ctx.base_dir / paths.QUARTERLY_METRICS.relative_to(paths.PROJECT_ROOT)
    limitation_path = ctx.base_dir / paths.QUARTERLY_LIMITATION.relative_to(paths.PROJECT_ROOT)
    assert forecast_path.is_file()
    assert metrics_path.is_file()
    assert limitation_path.is_file()

    written_forecast = pd.read_csv(forecast_path)
    assert list(written_forecast.columns) == QUARTERLY_FORECAST_COLUMNS
    rolling_rows = written_forecast.loc[written_forecast["estimate_type"] == ROLLING_ESTIMATE_TYPE]
    assert len(rolling_rows) == 1
    assert rolling_rows.iloc[0]["quarter"] == "2011-Q4"
    assert bool(rolling_rows.iloc[0]["complete"]) is False

    written_metrics = pd.read_csv(metrics_path)
    assert list(written_metrics.columns) == QUARTERLY_METRICS_COLUMNS
    assert set(written_metrics["model"]) == {B2, CHAMPION}

    assert "cannot forecast all three months" in limitation_path.read_text(encoding="utf-8")
    assert len(result["quarterly_forecast"]) == len(written_forecast)


# --------------------------------------------------------------------------
# real data: the documented complete-quarter set actually holds
# --------------------------------------------------------------------------
def test_complete_quarters_on_real_backtest_are_exactly_the_documented_set():
    """Issue §6: 'complete quarters are exactly 2010-Q3 ... 2011-Q3'; 2011-Q4 is never complete."""
    if not paths.BACKTEST_PREDICTIONS.is_file():
        pytest.skip("backtest_predictions.csv not present")

    backtest = pd.read_csv(
        paths.BACKTEST_PREDICTIONS,
        dtype={"stock_code": "string", "target_month": "string", "model": "string"},
    )
    qdf = aggregate_quarterly(backtest, [B2], CFG)

    complete_quarters = set(qdf.loc[qdf["complete"], "quarter"])
    assert complete_quarters == {"2010-Q3", "2010-Q4", "2011-Q1", "2011-Q2", "2011-Q3"}
    assert "2011-Q4" not in complete_quarters
    assert not any(qdf.loc[qdf["quarter"] == q, "months_included"].str.contains("2011-12").any()
                   for q in qdf["quarter"].unique())
