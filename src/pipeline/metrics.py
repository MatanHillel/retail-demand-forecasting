"""Deterministic forecast-accuracy metrics (PRD §23, §20).

Every wMAPE and Bias number the champion gates (§20) and the evaluation report compare comes from
this module, computed the same way everywhere, so a "wMAPE" printed by the back-test, the
evaluation report and the Streamlit app can never mean three different formulas. This module is
pure NumPy/pandas: no file I/O, no config reads, and no threshold, gate or model-selection logic —
that belongs to the callers (§22 champion gates, §26 model comparison), never here
(CLAUDE.md §2 rule 4: no number is hard-coded, and no threshold lives beside a metric formula).

Formulas, copied verbatim from CLAUDE.md / PRD §23::

    wMAPE = Σ|Actual − Forecast| / Σ Actual
    Bias  = Σ(Forecast − Actual) / Σ Actual

Both use the **same** denominator, Σ Actual, computed once over the whole group — never a
per-row ratio averaged afterwards, which is why this module has no "MAPE" function. wMAPE and Bias
are always reported together (§23: never one without the other) — :func:`metrics_table` always
returns both columns.

Every public function accepts ``actual`` / ``forecast`` as anything array-like (a list, a NumPy
array or a pandas Series) and aligns them **by position**, never by pandas index — callers merge
on their own keys (``stock_code``, ``month``) before calling, so two Series with different
indices must still compare element-by-element in the order given.

**Two ways to read the result, for two different purposes.** :func:`metrics_table` returns a
``pandas.DataFrame`` — the right shape for reports and on-screen display (evaluation tables, the
Streamlit app) — with column dtypes chosen the normal pandas way (``float64`` for the metric
columns, ``int64`` for ``n_rows``). :func:`metrics_records` is the serialisation-safe form: a
``list[dict]`` with every value **explicitly cast** (``int(...)`` for ``n_rows``, ``float(...)``
for every metric column), the form US-19 / US-22 should feed to ``ctx.record_metrics(...)``.

The explicit cast in :func:`metrics_records` exists because of a sharp edge in
:func:`metrics_table`: a DataFrame with columns of two different dtypes (``int64`` next to
``float64``) upcasts the whole row to a common dtype when a *row* is extracted across those
columns — ``table["n_rows"].iloc[0]`` and ``table.iloc[0]["n_rows"]`` both come back as
``numpy.float64``, not ``int``, because ``int64`` can be losslessly widened to ``float64`` and
pandas picks the common type for the row. (``table.to_dict(orient="records")`` does not have this
problem — it preserves each column's own dtype — but that consumption pattern still leaves every
caller responsible for remembering it, which is exactly what :func:`metrics_records` removes: a
bare ``numpy.int64``/``numpy.float64`` raises ``PydanticSerializationError`` inside
``ctx.finish()``, and ``run_log.json``'s ``model_dump(mode="json")`` needs plain Python scalars;
see docs/interfaces.md §3.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pipeline.run_context import get_logger

__all__ = [
    "wmape",
    "bias",
    "mae",
    "rmse",
    "relative_improvement",
    "negative_share",
    "metrics_table",
    "metrics_records",
    "format_pct",
]

#: Columns of :func:`metrics_table` that are cast to ``float`` by :func:`metrics_records`.
_FLOAT_METRIC_COLUMNS: tuple[str, ...] = (
    "wmape",
    "bias",
    "mae",
    "rmse",
    "sum_actual",
    "sum_forecast",
    "negative_share",
)

#: Columns of the frame :func:`metrics_table` returns, before any ``group_cols`` are prepended.
METRIC_COLUMNS: list[str] = [
    "wmape",
    "bias",
    "mae",
    "rmse",
    "n_rows",
    "sum_actual",
    "sum_forecast",
    "negative_share",
]


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------
def _as_float_array(values: Any, name: str) -> np.ndarray:
    """Positional NumPy view of a list / array / Series — never index-aligned."""
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return array


def _align(actual: Any, forecast: Any) -> tuple[np.ndarray, np.ndarray]:
    """Positional ``(actual, forecast)`` NumPy arrays; raises on a length mismatch."""
    actual_array = _as_float_array(actual, "actual")
    forecast_array = _as_float_array(forecast, "forecast")
    if len(actual_array) != len(forecast_array):
        raise ValueError(
            "actual and forecast must be the same length, got "
            f"{len(actual_array)} and {len(forecast_array)}"
        )
    return actual_array, forecast_array


# --------------------------------------------------------------------------
# metric formulas
# --------------------------------------------------------------------------
def wmape(actual: Any, forecast: Any) -> float:
    """Weighted MAPE: Σ|Actual − Forecast| / Σ Actual (§23) — never a row-wise MAPE.

    Returns ``float("nan")`` (and logs a warning) when Σ Actual is zero, rather than raising or
    silently returning zero or infinity — an all-zero-actual group has no denominator to weight by.
    """
    actual_array, forecast_array = _align(actual, forecast)
    denominator = actual_array.sum()
    if denominator == 0:
        get_logger().warning("wmape: sum(actual) is 0 — returning NaN")
        return float("nan")
    return float(np.abs(actual_array - forecast_array).sum() / denominator)


def bias(actual: Any, forecast: Any) -> float:
    """Bias: Σ(Forecast − Actual) / Σ Actual (§23), as a fraction (``0.3333``, not ``"33.3 %"``).

    Positive means the forecast is systematically too high, negative too low. Same
    zero-denominator rule as :func:`wmape`.
    """
    actual_array, forecast_array = _align(actual, forecast)
    denominator = actual_array.sum()
    if denominator == 0:
        get_logger().warning("bias: sum(actual) is 0 — returning NaN")
        return float("nan")
    return float((forecast_array - actual_array).sum() / denominator)


def mae(actual: Any, forecast: Any) -> float:
    """Mean absolute error, in the target's own units (units sold)."""
    actual_array, forecast_array = _align(actual, forecast)
    return float(np.abs(actual_array - forecast_array).mean())


def rmse(actual: Any, forecast: Any) -> float:
    """Root mean squared error, in the target's own units (units sold)."""
    actual_array, forecast_array = _align(actual, forecast)
    return float(np.sqrt(np.mean((actual_array - forecast_array) ** 2)))


@dataclass(frozen=True)
class RelativeImprovement:
    """Two ways to read "the model beat the baseline", both plain Python ``float``.

    ``points`` is what §20 gate 4 reads literally ("beats the baseline by >= 2 wMAPE points"): an
    absolute difference in wMAPE units, e.g. ``(wmape_model=0.53, wmape_baseline=0.55) -> 0.02``,
    i.e. 2.0 wMAPE points. ``relative_pct`` is the same improvement expressed as a share of the
    baseline's own error.
    """

    points: float
    relative_pct: float


def relative_improvement(wmape_model: float, wmape_baseline: float) -> RelativeImprovement:
    """How much lower ``wmape_model`` is than ``wmape_baseline`` — the input to §20 gate 4.

    ``points = wmape_baseline - wmape_model`` (positive means the model is better);
    ``relative_pct = points / wmape_baseline``. ``relative_pct`` is ``NaN`` (with a warning) when
    ``wmape_baseline`` is zero — there is no error to express a share of.
    """
    points = float(wmape_baseline) - float(wmape_model)
    if wmape_baseline == 0:
        get_logger().warning("relative_improvement: wmape_baseline is 0 — relative_pct is NaN")
        relative_pct = float("nan")
    else:
        relative_pct = points / float(wmape_baseline)
    return RelativeImprovement(points=points, relative_pct=relative_pct)


def negative_share(forecast: Any) -> float:
    """Share of ``forecast`` values below zero, e.g. ``[-1, 0, 2, 3] -> 0.25``.

    A forecast should never be negative (units cannot be sold in negative quantity); this is the
    diagnostic that catches a model that predicts one anyway. ``NaN`` for an empty input.
    """
    array = _as_float_array(forecast, "forecast")
    if len(array) == 0:
        return float("nan")
    return float((array < 0).sum() / len(array))


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _metrics_row(actual: np.ndarray, forecast: np.ndarray) -> dict[str, Any]:
    """One row of :data:`METRIC_COLUMNS` for a single (already positional) actual/forecast pair."""
    return {
        "wmape": wmape(actual, forecast),
        "bias": bias(actual, forecast),
        "mae": mae(actual, forecast),
        "rmse": rmse(actual, forecast),
        "n_rows": int(len(actual)),
        "sum_actual": float(actual.sum()),
        "sum_forecast": float(forecast.sum()),
        "negative_share": negative_share(forecast),
    }


def metrics_table(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One row of metrics per group, or a single overall row when ``group_cols`` is ``None``.

    Columns: ``wmape, bias, mae, rmse, n_rows, sum_actual, sum_forecast, negative_share`` — wMAPE
    and Bias are always both present (§23), never one without the other. When ``group_cols`` is
    given (e.g. ``["model"]``, ``["model", "target_month"]``, ``["model", "abc_class"]``), those
    columns are prepended to the result.

    Group order follows the order groups first appear in ``df`` — ``groupby(group_cols,
    sort=False)`` — so grouping by ``["model"]`` preserves the caller's model order rather than
    sorting alphabetically.

    See the module docstring for why callers should read this frame with
    ``df.to_dict(orient="records")`` rather than ``df.iloc[0]``.
    """
    if not group_cols:
        actual = df[actual_col].to_numpy(dtype=float)
        forecast = df[forecast_col].to_numpy(dtype=float)
        return pd.DataFrame([_metrics_row(actual, forecast)], columns=METRIC_COLUMNS)

    keys: list[tuple[Any, ...]] = []
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, sort=False):
        keys.append(key if isinstance(key, tuple) else (key,))
        actual = group[actual_col].to_numpy(dtype=float)
        forecast = group[forecast_col].to_numpy(dtype=float)
        rows.append(_metrics_row(actual, forecast))

    key_frame = pd.DataFrame(keys, columns=group_cols)
    metric_frame = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    return pd.concat([key_frame, metric_frame], axis=1)


def metrics_records(
    df: pd.DataFrame,
    actual_col: str = "actual",
    forecast_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """:func:`metrics_table`, as a list of dicts with every value explicitly cast to plain Python.

    This is the serialisation-safe form — the one US-19 / US-22 should hand to
    ``ctx.record_metrics(...)`` — because it does not rely on a caller remembering the
    ``to_dict(orient="records")`` vs. ``.iloc[0]`` distinction documented in the module docstring:
    ``n_rows`` is cast with ``int(...)``, and ``wmape, bias, mae, rmse, sum_actual, sum_forecast,
    negative_share`` are each cast with ``float(...)``. Group-column values (e.g. ``model``,
    ``target_month``) are passed through unchanged — they are strings already.
    """
    table = metrics_table(df, actual_col, forecast_col, group_cols)
    records: list[dict[str, Any]] = []
    for row in table.to_dict(orient="records"):
        record: dict[str, Any] = {
            column: row[column] for column in (group_cols or []) if column in row
        }
        record["n_rows"] = int(row["n_rows"])
        for column in _FLOAT_METRIC_COLUMNS:
            record[column] = float(row[column])
        records.append(record)
    return records


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------
def format_pct(x: float, digits: int = 1) -> str:
    """Format a fraction as a percentage string, e.g. ``0.532 -> "53.2 %"`` (space before ``%``).

    ``NaN`` (and ``None``) format as ``"NaN"`` rather than raising or printing ``"nan %"``.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NaN"
    return f"{float(x) * 100:.{digits}f} %"
