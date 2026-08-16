"""Feature engineering — ``features.csv`` (PRD §14, §16, §17, §21, §47).

One row per ``(stock_code, target_month)`` for every product that is **active** at that target
month (§14: at least one positive sale in the ``k`` months before it), carrying the fifteen §17
features. The grain changes here: the panel of US-05 is indexed by the month a sale *happened*,
this table is indexed by the month being *predicted*.

**The no-leakage boundary (§16) is structural, not a filter.** The forecast origin is ``t−1``, and
every feature is built from a series that has already been shifted one month back inside each
product, so month ``t`` is not reachable by any window — it is not computed and then removed. The
only values taken from ``t`` itself are ``target_month_of_year`` and ``target_quarter``: calendar
attributes, known in advance, the single exception §17 allows.

Window conventions (§17), all falling out of one construction — reindex every product onto the full
month grid of the observed dataset, fill absent months with zero units, then ``shift(1)`` and roll
with ``min_periods=1``:

* **Months before a product's first sale are observed zeros.** The dataset saw those months; the
  product simply had not launched. They count as zero-demand months in every rolling statistic, so
  a product launched in 2010-06 has ``rolling_mean_6`` divided by six at target 2010-09.
* **Months before the dataset begins are unobserved.** They never existed and are excluded, so
  windows are *truncated* rather than padded: at the first target month (2010-03) only three months
  are available and ``rolling_mean_6`` is divided by three. In the code this is the ``NaN`` that
  ``shift(1)`` leaves in each product's first grid row — rolling aggregations skip it.
* **``product_age_months`` is a lower bound** for products already selling in the first observed
  month (left-censoring, §47): they may be far older than the data can show.

Two counting conventions, chosen to agree with each other: ``months_since_last_sale`` is 1 when the
last sale was in ``t−1`` (as §17 pins), and ``product_age_months`` is likewise 1 when the first
sale was in ``t−1`` — both count *months of history through the origin*, never a raw difference
that would read 0 against the other's 1. ``rolling_std_3`` is the population standard deviation
(``ddof=0``), which also makes a single-observation window ``0.0`` instead of ``NaN``.

``stock_code`` is an identifier and never a feature (§17); ``returned_units``, ``customer_count``
and country are never used (§9, §13.2). The optional v2 features (``lag_12``, ``trend_3``, ABC as a
feature) are deliberately absent (§17, §18).

``build_features`` is **pure** — it returns a frame and touches no disk; :func:`write_features`
performs the write through ``ctx.out()`` (``docs/interfaces.md`` §6 rules 1 and 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.active import active_mask
from pipeline.config import ModelConfig, load_cleaning_config, load_model_config
from pipeline.contract import read_panel
from pipeline.run_context import RunContext

#: The fifteen §17 features, in the order ``model_config.yaml -> features`` lists them.
FEATURE_COLUMNS: list[str] = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_median_6",
    "rolling_std_3",
    "rolling_max_6",
    "nonzero_months_6",
    "months_since_last_sale",
    "product_age_months",
    "invoice_count_lag_1",
    "avg_unit_price_lag_1",
    "target_month_of_year",
    "target_quarter",
]

#: Column order of ``features.csv`` — a published schema, extend but never rename.
FEATURES_COLUMNS: list[str] = [
    "stock_code",
    "forecast_origin",
    "target_month",
    *FEATURE_COLUMNS,
    "y",
    "is_active",
]

# Window lengths are part of the §17 *feature definitions* — the names ``rolling_mean_3`` and
# ``rolling_mean_6`` encode them, so changing one without renaming the feature would be a defect.
# They are deliberately not configuration; the active-rule ``k`` is, and is never a literal here.
_SHORT_WINDOW = 3
_LONG_WINDOW = 6

_INTEGER_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_max_6",
    "nonzero_months_6",
    "months_since_last_sale",
    "product_age_months",
    "invoice_count_lag_1",
    "target_month_of_year",
    "target_quarter",
]
_FLOAT_FEATURES = [
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_median_6",
    "rolling_std_3",
    "avg_unit_price_lag_1",
]


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _ordinals(months: pd.Series) -> pd.Series:
    """Months as consecutive integers, so spans and gaps are plain arithmetic."""
    periods = pd.PeriodIndex(months.astype(str), freq="M")
    return pd.Series(periods.year * 12 + periods.month, index=months.index, dtype="int64")


def _shift_month(month: str, offset: int) -> str:
    """``month`` moved by ``offset`` calendar months, as ``YYYY-MM``."""
    return str(pd.Period(month, freq="M") + offset)


def _grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Every product × every observed month, zero-filled — the frame the windows are built on.

    The panel starts each product at its *first observed sale* (US-05), so the months a product
    existed but did not sell are missing from it. Rolling statistics must see them as zeros, which
    means reindexing onto the full month range first. Months before the dataset's own first month
    are not added: they were never observed, and their absence is what truncates the early windows.
    """
    frame = panel[["stock_code", "month", "units_sold", "invoice_count", "avg_unit_price"]].copy()
    frame["stock_code"] = frame["stock_code"].astype(str)
    frame["month"] = frame["month"].astype(str)

    products = np.sort(frame["stock_code"].unique())
    months = [str(period) for period in pd.period_range(
        frame["month"].min(), frame["month"].max(), freq="M"
    )]
    grid = pd.MultiIndex.from_product(
        [products, months], names=["stock_code", "month"]
    ).to_frame(index=False)

    grid = (
        grid.merge(frame, on=["stock_code", "month"], how="left")
        .sort_values(["stock_code", "month"], kind="mergesort")
        .reset_index(drop=True)
    )
    grid["units_sold"] = grid["units_sold"].fillna(0.0)
    grid["invoice_count"] = grid["invoice_count"].fillna(0.0)
    # Price carries forward through zero months inside the panel already; the months added here sit
    # before the product's first sale, where there is no price to carry and no active row either.
    grid["avg_unit_price"] = (
        grid["avg_unit_price"].groupby(grid["stock_code"], sort=False).ffill().fillna(0.0)
    )
    return grid


def _feature_frame(panel: pd.DataFrame, k: int, targets: list[str]) -> pd.DataFrame:
    """The §17 features for every ``(product, target month)`` in ``targets`` that is active.

    Each row is a *target* month ``t``; every value is drawn from the series shifted one month
    back, so the window ``t−n … t−1`` is the only thing any feature can see (§16).
    """
    if k <= 0:
        raise ValueError(f"k must be a positive number of months, got {k}")

    grid = _grid(panel)
    codes = grid["stock_code"]
    units = grid["units_sold"]

    # ``prior`` is units at t−1 aligned to the row of t. Every window below is built from it, so
    # month t is structurally unreachable. Its NaN in each product's first row marks the months
    # before the dataset began — rolling skips it, which is exactly the truncated-window rule.
    prior = units.groupby(codes, sort=False).shift(1)
    prior_by_product = prior.groupby(codes, sort=False)

    def _roll(window: int, how: str, **kwargs: object) -> pd.Series:
        rolled = getattr(prior_by_product.rolling(window, min_periods=1), how)(**kwargs)
        return rolled.reset_index(level=0, drop=True).sort_index()

    features = pd.DataFrame(index=grid.index)
    features["lag_1"] = prior
    features["lag_2"] = units.groupby(codes, sort=False).shift(2)
    features["lag_3"] = units.groupby(codes, sort=False).shift(3)
    features["rolling_mean_3"] = _roll(_SHORT_WINDOW, "mean")
    features["rolling_mean_6"] = _roll(_LONG_WINDOW, "mean")
    features["rolling_median_6"] = _roll(_LONG_WINDOW, "median")
    features["rolling_std_3"] = _roll(_SHORT_WINDOW, "std", ddof=0)
    features["rolling_max_6"] = _roll(_LONG_WINDOW, "max")

    # An unobserved month is not a month with sales, so it contributes 0 to the count either way.
    sold = (prior > 0).astype("float64")
    features["nonzero_months_6"] = (
        sold.groupby(codes, sort=False)
        .rolling(_LONG_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
    )

    ordinal = _ordinals(grid["month"])
    sale_ordinal = ordinal.where(units > 0)
    # Carried forward so that a month without a sale still knows when the last one was; months
    # before a product's first sale stay NaN, and no active row can land on one.
    last_sale_by_month = sale_ordinal.groupby(codes, sort=False).ffill()
    # Running minimum of that: the earliest sale seen so far. Both are then shifted, so each reads
    # the state *as known at t−1* — a whole-series ``min`` would give the same number on every
    # active row (the first sale always precedes the origin) but would read months ≥ t to do it,
    # which US-14's permutation test rightly rejects.
    first_sale_by_month = last_sale_by_month.groupby(codes, sort=False).cummin()

    # ordinal(t) − ordinal(last sale as of t−1): a sale in t−1 gives 1 (§17).
    features["months_since_last_sale"] = ordinal - last_sale_by_month.groupby(
        codes, sort=False
    ).shift(1)
    # ordinal(t) − ordinal(first observed sale as of t−1): months of history, minimum 1.
    features["product_age_months"] = ordinal - first_sale_by_month.groupby(
        codes, sort=False
    ).shift(1)

    features["invoice_count_lag_1"] = grid["invoice_count"].groupby(codes, sort=False).shift(1)
    features["avg_unit_price_lag_1"] = grid["avg_unit_price"].groupby(codes, sort=False).shift(1)

    target_periods = pd.PeriodIndex(grid["month"], freq="M")
    features["target_month_of_year"] = pd.Series(target_periods.month, index=grid.index)
    features["target_quarter"] = pd.Series(target_periods.quarter, index=grid.index)

    frame = pd.concat(
        [grid[["stock_code", "month", "units_sold"]].rename(columns={"month": "target_month"}),
         features],
        axis=1,
    )
    frame["forecast_origin"] = pd.Series(
        (target_periods - 1).astype(str), index=grid.index
    )

    # The §14 rule, from its single definition — a product is never active before its first sale,
    # so the grid months added ahead of that have no active_mask row and drop out here.
    activity = active_mask(panel, k).rename(columns={"month": "target_month"})
    activity["stock_code"] = activity["stock_code"].astype(str)
    activity["target_month"] = activity["target_month"].astype(str)
    frame = frame.merge(activity, on=["stock_code", "target_month"], how="inner")

    frame = frame[frame["is_active"] & frame["target_month"].isin(targets)]
    return frame.sort_values(["stock_code", "target_month"], kind="mergesort").reset_index(
        drop=True
    )


def _finalise(frame: pd.DataFrame, cfg: ModelConfig, include_target: bool) -> pd.DataFrame:
    """Apply dtypes and column order, then assert the invariants this module promises."""
    if list(cfg.features) != FEATURE_COLUMNS:
        raise ValueError(
            "model_config.yaml -> features must match the §17 feature set in order; "
            f"config has {list(cfg.features)}"
        )
    # Checked before the integer casts, which would otherwise fail with pandas' opaque
    # "Cannot convert non-finite values" instead of naming the feature that went wrong.
    missing = frame[FEATURE_COLUMNS].isna()
    if missing.any().any():
        offenders = sorted(missing.columns[missing.any()])
        raise ValueError(f"features must never be NaN on an active row; offending: {offenders}")

    typed = frame.copy()
    typed["stock_code"] = typed["stock_code"].astype(str)
    for column in _INTEGER_FEATURES:
        typed[column] = pd.to_numeric(typed[column]).round().astype("int64")
    for column in _FLOAT_FEATURES:
        typed[column] = pd.to_numeric(typed[column]).astype("float64")
    typed["y"] = pd.to_numeric(typed["units_sold"]).round().astype("int64")
    typed["is_active"] = typed["is_active"].astype(bool)

    columns = [column for column in FEATURES_COLUMNS if include_target or column != "y"]
    typed = typed[columns]

    misaligned = int(
        (_ordinals(typed["target_month"]) - _ordinals(typed["forecast_origin"]) != 1).sum()
    )
    if misaligned:
        raise ValueError(f"{misaligned} row(s) have a forecast_origin that is not target_month − 1")
    return typed.reset_index(drop=True)


def build_features(
    panel_df: pd.DataFrame,
    k: int,
    first_target: str,
    last_target: str,
    cfg: ModelConfig,
    include_target: bool = True,
) -> pd.DataFrame:
    """The §17 features for every active product and every target month in the closed range.

    Pure: returns a frame and writes nothing — :func:`write_features` performs the write. The range
    covers fully observed targets only (2010-03 … 2011-11 in this project); December 2011 is a
    partial month and is never a target here (§16, §21), it is served by
    :func:`build_features_for_origin`.
    """
    if first_target > last_target:
        raise ValueError(f"first_target {first_target} must not be after last_target {last_target}")
    targets = [str(period) for period in pd.period_range(first_target, last_target, freq="M")]
    frame = _feature_frame(panel_df, k, targets)
    return _finalise(frame, cfg, include_target)


def build_features_for_origin(
    panel_df: pd.DataFrame, origin: str, k: int, cfg: ModelConfig
) -> pd.DataFrame:
    """Features for the single target ``origin + 1 month``, without ``y`` — the US-23 forecast.

    Same code path as :func:`build_features`, so the operational features cannot drift from the
    ones the model was trained on. The target's actuals are unknown (or partial), which is why the
    frame carries no ``y``.
    """
    target = _shift_month(origin, 1)
    frame = _feature_frame(panel_df, k, [target])
    return _finalise(frame, cfg, include_target=False)


def write_features(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``features.csv`` through ``ctx.out()`` and register it on the run.

    Deterministic bytes (§40, §42): fixed column order, fixed sort, fixed float format, no index.
    """
    destination = ctx.out(_repo_relative(paths.FEATURES))
    frame.to_csv(
        destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("features", _repo_relative(paths.FEATURES))
    return destination


def run() -> int:
    """Standalone entry point: build the features from ``clean_data.csv`` and write them (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("build_features"):
            if not paths.CLEAN_DATA.is_file():
                raise FileNotFoundError(
                    f"{paths.CLEAN_DATA} not found. Run `python -m pipeline.panel` first."
                )
            model_cfg = load_model_config()
            cleaning_cfg = load_cleaning_config()
            panel = read_panel(paths.CLEAN_DATA)
            k = model_cfg.active_rule.k
            first_target = model_cfg.split.first_target_month
            last_target = cleaning_cfg.raw.last_full_month

            targets = [
                str(period)
                for period in pd.period_range(first_target, last_target, freq="M")
            ]
            candidates = int(panel["stock_code"].nunique() * len(targets))
            frame = build_features(panel, k, first_target, last_target, model_cfg)
            ctx.log_rows(
                "features_active_filter",
                before=candidates,
                removed=candidates - len(frame),
                after=int(len(frame)),
            )
            write_features(frame, ctx)

            zero_share = float((frame["y"] == 0).mean()) if len(frame) else 0.0
            ctx.record_metrics(
                {
                    "features_rows": int(len(frame)),
                    "features_products": int(frame["stock_code"].nunique()),
                    "features_zero_target_share": round(zero_share, 6),
                    "features_first_target_month": str(frame["target_month"].min()),
                    "features_last_target_month": str(frame["target_month"].max()),
                }
            )
        print(
            f"rows: {ctx.metrics['features_rows']}\n"
            f"products: {ctx.metrics['features_products']}\n"
            f"zero-target share: {ctx.metrics['features_zero_target_share']:.2%}\n"
            f"first target month: {ctx.metrics['features_first_target_month']}\n"
            f"last target month: {ctx.metrics['features_last_target_month']}"
        )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
