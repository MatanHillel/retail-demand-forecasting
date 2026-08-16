"""Forecast-accuracy metric tests (US-15, PRD §23, §20, §55).

Hand-computed arrays only — no fixtures, no config, no I/O. :mod:`pipeline.metrics` is pure
NumPy/pandas, so every test works directly against small lists and DataFrames.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from pipeline.metrics import (
    METRIC_COLUMNS,
    bias,
    format_pct,
    mae,
    metrics_records,
    metrics_table,
    negative_share,
    relative_improvement,
    rmse,
    wmape,
)


# --------------------------------------------------------------------------
# wmape / bias — the §23 formulas
# --------------------------------------------------------------------------
def test_wmape_and_bias_on_the_worked_example() -> None:
    actual = [10, 0, 20]
    forecast = [8, 2, 30]
    # wMAPE = (|10-8| + |0-2| + |20-30|) / (10+0+20) = (2+2+10)/30
    assert wmape(actual, forecast) == pytest.approx((2 + 2 + 10) / 30)
    # Bias = ((8-10) + (2-0) + (30-20)) / 30 = (-2+2+10)/30
    assert bias(actual, forecast) == pytest.approx((-2 + 2 + 10) / 30)


def test_perfect_forecast_has_zero_wmape_and_zero_bias() -> None:
    actual = [10, 20, 30]
    assert wmape(actual, actual) == pytest.approx(0.0)
    assert bias(actual, actual) == pytest.approx(0.0)


def test_constant_20_percent_underforecast_has_bias_minus_020() -> None:
    actual = [10, 20, 30, 40]
    forecast = [8, 16, 24, 32]  # exactly 80% of actual, i.e. 20% under
    assert bias(actual, forecast) == pytest.approx(-0.20)
    assert wmape(actual, forecast) == pytest.approx(0.20)


# pipeline.run_context.get_logger() sets ``propagate = False`` on the "pipeline" logger (by
# design — its own console/file handlers are the only sinks it wants), so caplog's default
# root-attached handler never sees its records via propagation. Attaching caplog's own handler
# directly to the "pipeline" logger sidesteps that without touching production code.
def test_wmape_all_zero_actuals_returns_nan_without_raising_and_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("pipeline")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="pipeline"):
            result = wmape([0, 0, 0], [1, 2, 3])
    finally:
        logger.removeHandler(caplog.handler)
    assert np.isnan(result)
    assert any("sum(actual)" in record.getMessage() for record in caplog.records)


def test_bias_all_zero_actuals_returns_nan_without_raising_and_logs_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("pipeline")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="pipeline"):
            result = bias([0, 0, 0], [1, 2, 3])
    finally:
        logger.removeHandler(caplog.handler)
    assert np.isnan(result)
    assert any("sum(actual)" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------
# mae / rmse
# --------------------------------------------------------------------------
def test_mae_matches_hand_computed_value() -> None:
    actual = [10, 0, 20]
    forecast = [8, 2, 30]
    # |2| + |-2| + |-10| = 14 -> mean 14/3
    assert mae(actual, forecast) == pytest.approx(14 / 3)


def test_rmse_matches_hand_computed_value() -> None:
    actual = [10, 0, 20]
    forecast = [8, 2, 30]
    # errors: -2, 2, 10 -> squares 4, 4, 100 -> mean 36 -> sqrt 6.0
    assert rmse(actual, forecast) == pytest.approx(6.0)


# --------------------------------------------------------------------------
# relative_improvement — §20 gate 4
# --------------------------------------------------------------------------
def test_relative_improvement_points_and_relative_pct() -> None:
    result = relative_improvement(0.53, 0.55)
    assert result.points == pytest.approx(0.02)
    assert result.relative_pct == pytest.approx(0.02 / 0.55)


def test_relative_improvement_negative_when_model_is_worse() -> None:
    result = relative_improvement(0.60, 0.55)
    assert result.points == pytest.approx(-0.05)
    assert result.relative_pct < 0


def test_relative_improvement_zero_baseline_gives_nan_relative_pct(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="pipeline"):
        result = relative_improvement(0.0, 0.0)
    assert result.points == pytest.approx(0.0)
    assert np.isnan(result.relative_pct)


# --------------------------------------------------------------------------
# negative_share
# --------------------------------------------------------------------------
def test_negative_share_quarter_negative() -> None:
    assert negative_share([-1, 0, 2, 3]) == pytest.approx(0.25)


def test_negative_share_none_negative() -> None:
    assert negative_share([0, 1, 2, 3]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# index misalignment — positional, not index-aligned
# --------------------------------------------------------------------------
def test_series_with_different_indices_align_positionally() -> None:
    actual_list = [10, 0, 20]
    forecast_list = [8, 2, 30]
    actual_series = pd.Series(actual_list, index=[100, 200, 300])
    forecast_series = pd.Series(forecast_list, index=[9, 8, 7])

    assert wmape(actual_series, forecast_series) == pytest.approx(wmape(actual_list, forecast_list))
    assert bias(actual_series, forecast_series) == pytest.approx(bias(actual_list, forecast_list))


def test_mismatched_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError):
        wmape([1, 2, 3], [1, 2])


# --------------------------------------------------------------------------
# metrics_table
# --------------------------------------------------------------------------
def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["M2", "M2", "B2", "B2"],
            "target_month": ["2011-06", "2011-07", "2011-06", "2011-07"],
            "actual": [10, 0, 20, 5],
            "prediction": [8, 2, 30, 5],
        }
    )


def test_metrics_table_no_group_cols_returns_one_row() -> None:
    table = metrics_table(_frame(), group_cols=None)
    assert len(table) == 1
    assert {"wmape", "bias", "mae", "rmse", "n_rows", "sum_actual", "sum_forecast",
            "negative_share"} <= set(table.columns)


def test_metrics_table_grouped_by_month_has_one_row_per_month_with_wmape_and_bias() -> None:
    table = metrics_table(_frame(), group_cols=["target_month"])
    assert sorted(table["target_month"]) == ["2011-06", "2011-07"]
    assert "wmape" in table.columns and "bias" in table.columns
    assert len(table) == 2


def test_metrics_table_grouped_by_model_preserves_input_order() -> None:
    # "M2" appears before "B2" in _frame(); groupby(sort=False) must preserve that order.
    table = metrics_table(_frame(), group_cols=["model"])
    assert list(table["model"]) == ["M2", "B2"]


def test_metrics_table_overall_matches_direct_formula_call() -> None:
    frame = _frame()
    table = metrics_table(frame, group_cols=None)
    row = table.to_dict(orient="records")[0]
    assert row["wmape"] == pytest.approx(wmape(frame["actual"], frame["prediction"]))
    assert row["bias"] == pytest.approx(bias(frame["actual"], frame["prediction"]))
    assert row["n_rows"] == len(frame)


# --------------------------------------------------------------------------
# scalar types — interface correction 2: no numpy scalars leak into ctx.record_metrics
# --------------------------------------------------------------------------
def test_metrics_table_records_are_plain_python_scalars_and_json_serialisable() -> None:
    table = metrics_table(_frame(), group_cols=["model"])
    records = table.to_dict(orient="records")
    for record in records:
        assert type(record["n_rows"]) is int
        for column in ("wmape", "bias", "mae", "rmse", "sum_actual", "sum_forecast",
                       "negative_share"):
            assert type(record[column]) is float, column
        # A direct regression test for the PydanticSerializationError this guards against.
        json.dumps({k: v for k, v in record.items() if k != "model"})


def test_metrics_table_overall_row_scalars_are_plain_python_too() -> None:
    table = metrics_table(_frame(), group_cols=None)
    record = table.to_dict(orient="records")[0]
    assert type(record["n_rows"]) is int
    assert type(record["wmape"]) is float
    json.dumps(record)


# --------------------------------------------------------------------------
# metrics_records — the explicit-cast, serialisation-safe accessor
# --------------------------------------------------------------------------
_FLOAT_COLUMNS = ("wmape", "bias", "mae", "rmse", "sum_actual", "sum_forecast", "negative_share")


def test_metrics_records_ungrouped_scalars_are_plain_python_and_json_safe() -> None:
    records = metrics_records(_frame(), group_cols=None)
    assert len(records) == 1
    record = records[0]
    assert type(record["n_rows"]) is int
    for column in _FLOAT_COLUMNS:
        assert type(record[column]) is float, column
    json.dumps(record)


def test_metrics_records_grouped_scalars_are_plain_python_and_json_safe() -> None:
    records = metrics_records(_frame(), group_cols=["model"])
    assert len(records) == 2
    for record in records:
        assert type(record["n_rows"]) is int
        assert type(record["model"]) is str
        for column in _FLOAT_COLUMNS:
            assert type(record[column]) is float, column
        json.dumps(record)


def test_metrics_records_grouped_preserves_input_order() -> None:
    # "M2" appears before "B2" in _frame() — same ordering guarantee as metrics_table.
    records = metrics_records(_frame(), group_cols=["model"])
    assert [record["model"] for record in records] == ["M2", "B2"]


def test_metrics_records_values_match_metrics_table() -> None:
    frame = _frame()
    table = metrics_table(frame, group_cols=["model"])
    records = metrics_records(frame, group_cols=["model"])
    table_records = table.to_dict(orient="records")
    assert len(records) == len(table_records)
    for record, table_record in zip(records, table_records, strict=True):
        assert record["model"] == table_record["model"]
        assert record["n_rows"] == table_record["n_rows"]
        for column in _FLOAT_COLUMNS:
            assert record[column] == pytest.approx(table_record[column], nan_ok=True)


def test_metrics_records_multi_group_cols_are_all_present() -> None:
    records = metrics_records(_frame(), group_cols=["model", "target_month"])
    assert len(records) == 4
    for record in records:
        assert set(("model", "target_month", *METRIC_COLUMNS)) <= set(record)
        json.dumps(record)


# --------------------------------------------------------------------------
# format_pct
# --------------------------------------------------------------------------
def test_format_pct_default_digits() -> None:
    assert format_pct(0.532) == "53.2 %"


def test_format_pct_custom_digits() -> None:
    assert format_pct(0.53219, digits=2) == "53.22 %"


def test_format_pct_nan_is_handled_gracefully() -> None:
    assert format_pct(float("nan")) == "NaN"
