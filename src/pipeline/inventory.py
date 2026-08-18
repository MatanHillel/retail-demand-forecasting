"""Inventory simulation & KPIs (US-21, PRD §24-§30, §36, §44, §52, §55).

Converts a demand forecast into a **Recommended Target Inventory** (§7) — the dataset has no
on-hand / on-order data, so a replenishment quantity cannot be computed, only a target level. This
is a deterministic policy layer applied on top of a forecast, not a model (§24); it never predicts
anything itself.

Two policies are simulated for every candidate model (§29), on the same hold-out rows US-19
(:mod:`pipeline.evaluate`) and US-20 (:mod:`pipeline.sigma`) already produced:

* ``forecast_only`` — target = forecast, no safety stock.
* ``forecast_plus_ss`` — target = forecast + z x robust sigma (§25-§28).

**Identical rows across every model, by construction.** ``holdout_rows_all_models.csv`` has a
forecast for every candidate at every row except B3 (seasonal naive), which has no observed
month t-12 for some products. CLAUDE.md §2 rule 11 forbids scoring candidates on different row
sets, so the row a model lacks a forecast for is dropped for **every** model, not just the one
missing it — the surviving rows are the same (stock_code, target_month) set for all seven
candidates, which is what lets gate 3 (US-22) compare any pair fairly, B3 included. The drop is
counted and reported (:meth:`RunContext.log_rows` / :meth:`RunContext.warn`); it is routine, not a
validation failure. :func:`validate_simulation_inputs` then checks the *result* of that
construction really is identical and NaN-free — a violation there means the join itself is broken,
which is a genuine ``FLOW STOPPED`` condition (PRD §39).

**z never multiplies raw data twice.** ``holdout_simulation_rows.csv`` — the per-row file — is
written once, at the policy's configured default z, so its row count matches issue §3's estimate
(hold-out months x active products x candidate models x two policies, no z multiplier); Screen 5
recomputes a different z from the row's own ``forecast``/``sigma`` rather than needing every z
pre-multiplied. ``inventory_kpis.csv`` is the one artifact swept over every ``z_options`` value
(issue §2), because the KPI aggregates — unlike a raw row — cannot be recomputed from a single z's
numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.baselines import B2
from pipeline.config import (
    MODEL_IDS,
    InventoryPolicy,
    ModelConfig,
    load_inventory_policy,
    load_model_config,
)
from pipeline.metrics import format_pct
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: Step name on every ``Violation`` / the validation report / the CLI's ``ctx.step(...)``.
STEP_NAME = "inventory_simulation"

#: The two policies of §29, in canonical order. Names, not numbers — no config field owns them.
POLICY_FORECAST_ONLY = "forecast_only"
POLICY_FORECAST_PLUS_SS = "forecast_plus_ss"
POLICIES: tuple[str, str] = (POLICY_FORECAST_ONLY, POLICY_FORECAST_PLUS_SS)

#: KPI scopes (§30), in canonical report order.
SCOPE_OVERALL = "overall"
SCOPE_MONTH = "month"
SCOPE_ABC = "abc"
_SCOPE_RANK = {SCOPE_OVERALL: 0, SCOPE_MONTH: 1, SCOPE_ABC: 2}

#: Column order of the pure input rows :func:`simulate_inventory` consumes (issue §2).
ROWS_INPUT_COLUMNS: list[str] = [
    "stock_code",
    "target_month",
    "abc_class",
    "actual",
    "model",
    "forecast",
    "sigma",
    "sigma_source",
]

#: Column order of ``artifacts/forecasts/holdout_simulation_rows.csv``.
SIM_ROWS_COLUMNS: list[str] = [
    "stock_code",
    "target_month",
    "abc_class",
    "model",
    "policy",
    "z",
    "actual",
    "forecast",
    "sigma",
    "sigma_source",
    "target_inventory",
    "shortage",
    "excess",
    "fulfilled",
]

#: Column order of ``artifacts/forecasts/inventory_kpis.csv``.
KPI_COLUMNS: list[str] = [
    "model",
    "policy",
    "z",
    "scope",
    "group",
    "fill_rate",
    "stockout_units",
    "excess_units",
    "stockout_skumonth_rate",
    "excess_per_unit_shortage",
    "n_rows",
]

#: Extra columns :func:`inventory_kpis` returns beyond the ``KPI_COLUMNS`` written to disk (issue
#: §2 names ``sum_actual, sum_target`` alongside the five KPI ratios).
_KPI_COMPUTED_COLUMNS: list[str] = [
    "fill_rate",
    "stockout_units",
    "excess_units",
    "stockout_skumonth_rate",
    "excess_per_unit_shortage",
    "n_rows",
    "sum_actual",
    "sum_target",
]

#: Column order of ``artifacts/reports/evaluation_tables/excess_concentration.csv``.
EXCESS_CONCENTRATION_COLUMNS: list[str] = [
    "model",
    "policy",
    "n_rows",
    "total_excess",
    "top_1pct_share",
    "top_5pct_share",
]

_PRED_PREFIX = "pred_"


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _order_by_model(frame: pd.DataFrame, secondary: list[str] | None = None) -> pd.DataFrame:
    """Sort rows in the canonical candidate order B1, B2, B3, M1, M2, M3, M4 (mirrors
    :mod:`pipeline.evaluate`'s ``_order_by_model`` — same convention, same reason)."""
    ordered = frame.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=list(MODEL_IDS), ordered=True)
    ordered = ordered.sort_values(["model", *(secondary or [])], kind="mergesort")
    ordered["model"] = ordered["model"].astype(str)
    return ordered.reset_index(drop=True)


# --------------------------------------------------------------------------
# safety_stock, target_inventory — the two formulas of §25 / §28, verbatim
# --------------------------------------------------------------------------
def safety_stock(sigma: Any, z: float) -> Any:
    """Safety stock = z x sigma (§25) — extra units held on top of the forecast to cover the
    chance that demand in a month is higher than expected."""
    return z * sigma


def target_inventory(forecast: Any, sigma: Any, z: float) -> Any:
    """Recommended Target Inventory = ceil(max(0, forecast + z x sigma)) (§28).

    Accepts scalars or array-likes; a scalar call returns a plain ``int``, an array-like call
    returns an ``int`` array — never negative, because a forecast or a safety-stock draw below
    zero is floored at zero before rounding up.
    """
    forecast_array = np.asarray(forecast, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    raw = forecast_array + safety_stock(sigma_array, z)
    rounded = np.ceil(np.maximum(raw, 0.0))
    if rounded.ndim == 0:
        return int(rounded)
    return rounded.astype(int)


# --------------------------------------------------------------------------
# simulate_inventory — pure, no ctx, no disk (issue §2)
# --------------------------------------------------------------------------
def simulate_inventory(
    rows_df: pd.DataFrame, z: float, policy_cfg: InventoryPolicy
) -> pd.DataFrame:
    """Apply both policies (§29) to every row of ``rows_df`` at one z, stacked into one frame.

    ``rows_df`` is the long per-(stock_code, target_month, model) frame of
    :data:`ROWS_INPUT_COLUMNS` — one row per candidate, every candidate covering the identical
    row set (module docstring). ``policy_cfg`` is accepted per the issue's own signature; the
    policy set itself is fixed by §29 and carries no config field of its own, so nothing is read
    off it here — it exists so a future caller can validate ``z`` against ``policy_cfg.z_options``
    without changing this function's shape.
    """
    del policy_cfg
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")

    forecast = rows_df["forecast"].to_numpy(dtype=float)
    sigma = rows_df["sigma"].to_numpy(dtype=float)
    actual = rows_df["actual"].to_numpy(dtype=float)
    zero_sigma = np.zeros_like(sigma)

    frames = []
    for policy in POLICIES:
        effective_sigma = sigma if policy == POLICY_FORECAST_PLUS_SS else zero_sigma
        target = target_inventory(forecast, effective_sigma, z)
        target_float = target.astype(float)
        shortage = np.maximum(actual - target_float, 0.0)
        excess = np.maximum(target_float - actual, 0.0)
        fulfilled = np.minimum(target_float, actual)

        frame = rows_df.copy()
        frame["policy"] = policy
        frame["z"] = float(z)
        frame["target_inventory"] = target
        frame["shortage"] = shortage
        frame["excess"] = excess
        frame["fulfilled"] = fulfilled
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    return combined[SIM_ROWS_COLUMNS]


# --------------------------------------------------------------------------
# inventory_kpis — pure aggregation, no ctx, no disk (issue §2 / §30)
# --------------------------------------------------------------------------
def inventory_kpis(sim_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """KPIs (§30) grouped by ``model, policy, z`` plus ``group_cols`` (empty for the overall view,
    ``["target_month"]`` for the by-month view, ``["abc_class"]`` for the by-ABC view).

    ``fill_rate = sum(fulfilled) / sum(actual)``, ``stockout_units = sum(shortage)``,
    ``excess_units = sum(excess)``, ``stockout_skumonth_rate`` = share of rows where
    ``actual > target`` (equivalently ``shortage > 0``), ``excess_per_unit_shortage =
    excess_units / stockout_units`` (``NaN`` when there is no shortage — there is nothing to
    divide by). ``sum_actual`` / ``sum_target`` are carried alongside for callers that need the
    raw totals (issue §2).
    """
    keys = ["model", "policy", "z", *group_cols]
    rows: list[dict[str, Any]] = []
    for key, group in sim_df.groupby(keys, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        sum_actual = float(group["actual"].sum())
        sum_target = float(group["target_inventory"].sum())
        stockout_units = float(group["shortage"].sum())
        excess_units = float(group["excess"].sum())
        fulfilled_units = float(group["fulfilled"].sum())
        n_rows = int(len(group))
        record = dict(zip(keys, key_tuple, strict=True))
        record.update(
            fill_rate=(fulfilled_units / sum_actual) if sum_actual else float("nan"),
            stockout_units=stockout_units,
            excess_units=excess_units,
            stockout_skumonth_rate=(
                float((group["shortage"] > 0).mean()) if n_rows else float("nan")
            ),
            excess_per_unit_shortage=(
                (excess_units / stockout_units) if stockout_units else float("nan")
            ),
            n_rows=n_rows,
            sum_actual=sum_actual,
            sum_target=sum_target,
        )
        rows.append(record)
    return pd.DataFrame(rows, columns=[*keys, *_KPI_COMPUTED_COLUMNS])


def _scoped_kpis(sim_df: pd.DataFrame, scope: str, group_col: str | None) -> pd.DataFrame:
    """One scope's slice of :func:`inventory_kpis`, relabelled to the ``scope`` / ``group``
    columns the written CSV uses (§30: overall, by month, by ABC group)."""
    group_cols = [group_col] if group_col else []
    table = inventory_kpis(sim_df, group_cols)
    if group_col:
        table = table.rename(columns={group_col: "group"})
    else:
        table["group"] = "all"
    table["scope"] = scope
    return table[KPI_COLUMNS]


def _order_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row order (CLAUDE.md §40): candidate order, then policy, then z, then the
    canonical scope order, then group."""
    ordered = frame.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=list(MODEL_IDS), ordered=True)
    ordered["policy"] = pd.Categorical(ordered["policy"], categories=list(POLICIES), ordered=True)
    ordered["_scope_rank"] = ordered["scope"].map(_SCOPE_RANK)
    ordered = ordered.sort_values(
        ["model", "policy", "z", "_scope_rank", "group"], kind="mergesort"
    )
    ordered["model"] = ordered["model"].astype(str)
    ordered["policy"] = ordered["policy"].astype(str)
    return ordered.drop(columns="_scope_rank").reset_index(drop=True)[KPI_COLUMNS]


# --------------------------------------------------------------------------
# excess_concentration — pure, no ctx, no disk (§30: "excess is dominated by a few very large
# orders — the report must say so")
# --------------------------------------------------------------------------
def excess_concentration(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Share of total excess coming from the top 1% / 5% of SKU-months, per (model, policy).

    Ranks each (model, policy) group's rows by ``excess`` descending and compares the sum of the
    top slice to the group's total excess. ``NaN`` when a group has zero total excess — there is
    no concentration to report when nothing is held in excess.
    """
    rows: list[dict[str, Any]] = []
    for (model_id, policy), group in sim_df.groupby(["model", "policy"], sort=False):
        n_rows = len(group)
        total_excess = float(group["excess"].sum())
        sorted_excess = np.sort(group["excess"].to_numpy(dtype=float))[::-1]
        top_1pct_n = max(1, int(np.ceil(n_rows * 0.01)))
        top_5pct_n = max(1, int(np.ceil(n_rows * 0.05)))
        top_1pct_sum = float(sorted_excess[:top_1pct_n].sum())
        top_5pct_sum = float(sorted_excess[:top_5pct_n].sum())
        rows.append(
            {
                "model": model_id,
                "policy": policy,
                "n_rows": n_rows,
                "total_excess": total_excess,
                "top_1pct_share": (top_1pct_sum / total_excess) if total_excess else float("nan"),
                "top_5pct_share": (top_5pct_sum / total_excess) if total_excess else float("nan"),
            }
        )
    table = pd.DataFrame(rows, columns=EXCESS_CONCENTRATION_COLUMNS)
    return _order_by_model(table, secondary=["policy"])


# --------------------------------------------------------------------------
# row construction — split the wide US-19 table, melt to long, attach US-20 sigma
# --------------------------------------------------------------------------
def _covered_rows(
    wide_df: pd.DataFrame, model_ids: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``holdout_rows_all_models.csv`` into rows every model has a forecast for, and rows
    at least one model (typically B3) lacks — module docstring's "identical rows across every
    model, by construction"."""
    pred_cols = [f"{_PRED_PREFIX}{model_id}" for model_id in model_ids]
    covered_mask = wide_df[pred_cols].notna().all(axis=1)
    covered = wide_df.loc[covered_mask].reset_index(drop=True)
    dropped = wide_df.loc[~covered_mask].reset_index(drop=True)
    return covered, dropped


def _melt_to_long(covered_df: pd.DataFrame, model_ids: tuple[str, ...]) -> pd.DataFrame:
    """Wide (one column per model) -> long (one row per stock_code x target_month x model)."""
    pred_cols = [f"{_PRED_PREFIX}{model_id}" for model_id in model_ids]
    long = covered_df.melt(
        id_vars=["stock_code", "target_month", "abc_class", "actual"],
        value_vars=pred_cols,
        var_name="model",
        value_name="forecast",
    )
    long["model"] = long["model"].str.replace(_PRED_PREFIX, "", regex=False)
    return long


def build_simulation_rows(
    wide_df: pd.DataFrame, sigma_df: pd.DataFrame, model_ids: tuple[str, ...] = MODEL_IDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure core of the row-construction step: ``holdout_rows_all_models.csv`` (wide) +
    ``sigma_table.csv`` (long, per model) -> the :data:`ROWS_INPUT_COLUMNS` frame
    :func:`simulate_inventory` consumes, plus the rows dropped along the way.

    No ``ctx`` and no disk — the caller (:func:`run_inventory_simulation`) is the one that logs
    the drop and writes anything, matching the pure-core / thin-writer split every other US-19 /
    US-20 module in this package uses.
    """
    covered, dropped = _covered_rows(wide_df, model_ids)
    long_df = _melt_to_long(covered, model_ids)

    sigma_lookup = sigma_df[["stock_code", "target_month", "model", "sigma", "sigma_source"]].copy()
    merged = long_df.merge(sigma_lookup, on=["stock_code", "target_month", "model"], how="left")

    merged["stock_code"] = merged["stock_code"].astype(str)
    merged["target_month"] = merged["target_month"].astype(str)
    merged["model"] = merged["model"].astype(str)

    rows_df = (
        merged.sort_values(["model", "target_month", "stock_code"], kind="mergesort")
        .reset_index(drop=True)[ROWS_INPUT_COLUMNS]
    )
    return rows_df, dropped


# --------------------------------------------------------------------------
# validate_simulation_inputs — pure check, returns ValidationResult (issue §8, not a bare assert)
# --------------------------------------------------------------------------
def validate_simulation_inputs(
    rows_df: pd.DataFrame, model_ids: tuple[str, ...] = MODEL_IDS
) -> ValidationResult:
    """Two invariants that must hold on ``rows_df`` before it is simulated: every candidate model
    covers the identical ``(stock_code, target_month)`` set, and no row is missing a forecast,
    sigma or actual. Both should hold by construction after :func:`build_simulation_rows` drops
    any row a model lacks a forecast for — a violation here means the upstream join is broken, not
    that a product happens to be thin on history, which is why (unlike the routine drop-and-report)
    a failure here is a genuine ``FLOW STOPPED`` condition (PRD §39, issue §8).
    """
    violations: list[Violation] = []

    row_sets = {
        model_id: frozenset(zip(group["stock_code"], group["target_month"], strict=True))
        for model_id, group in rows_df.groupby("model", sort=False)
    }
    missing_models = sorted(set(model_ids) - set(row_sets))
    if missing_models:
        violations.append(
            Violation(
                step=STEP_NAME,
                rule="identical_rows",
                message=f"model(s) missing from the simulation input entirely: {missing_models}",
                count=len(missing_models),
                examples=missing_models,
            )
        )
    if row_sets:
        reference_set = next(iter(row_sets.values()))
        mismatched = sorted(
            model_id for model_id, row_set in row_sets.items() if row_set != reference_set
        )
        if mismatched:
            violations.append(
                Violation(
                    step=STEP_NAME,
                    rule="identical_rows",
                    message=(
                        "model(s) do not cover the identical (stock_code, target_month) set as "
                        f"the rest: {mismatched}"
                    ),
                    count=len(mismatched),
                    examples=mismatched,
                )
            )

    nan_mask = rows_df[["forecast", "sigma", "actual"]].isna().any(axis=1)
    if nan_mask.any():
        examples = (
            rows_df.loc[nan_mask, ["stock_code", "target_month", "model"]]
            .head(5)
            .to_dict(orient="records")
        )
        violations.append(
            Violation(
                step=STEP_NAME,
                rule="no_nan_inputs",
                message=f"{int(nan_mask.sum())} row(s) have a NaN forecast, sigma or actual",
                count=int(nan_mask.sum()),
                examples=examples,
            )
        )

    return ValidationResult(
        step=STEP_NAME,
        passed=not violations,
        violations=violations,
        checked_rows=int(len(rows_df)),
    )


# --------------------------------------------------------------------------
# run_inventory_simulation — the public entry point (issue §2 / §8)
# --------------------------------------------------------------------------
def run_inventory_simulation(
    cfg: ModelConfig,
    ctx: RunContext,
    *,
    wide_df: pd.DataFrame | None = None,
    sigma_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Simulate both policies for every candidate model on the hold-out and write the three US-21
    artifacts.

    ``cfg`` is accepted for the project's standard ``run_<step>(cfg, ctx)`` shape; every setting
    this step reads — z, z_options, the sigma hierarchy — lives in ``inventory_policy.yaml``
    (issue §3), so no field of ``cfg`` (:class:`~pipeline.config.ModelConfig`) is read.

    Standalone (both keyword frames ``None``) it reads ``holdout_rows_all_models.csv`` (US-19) and
    ``sigma_table.csv`` (US-20) from their canonical locations — safe there, where
    ``staging=False`` and both files are final. The Flow (US-31 step 7) must instead pass in the
    DataFrames the producing steps returned via ``wide_df`` / ``sigma_df`` (both together — mixing
    is a ``ValueError``): under ``staging=True`` those files are still sitting in
    ``artifacts/_staging/<run_id>/`` while the final paths hold the *previous* run's copies
    (``docs/interfaces.md`` §6 rule 7, issue §8).
    """
    del cfg
    policy_cfg = load_inventory_policy()

    if (wide_df is None) != (sigma_df is None):
        raise ValueError("pass wide_df and sigma_df together, or neither")

    if wide_df is None or sigma_df is None:
        wide_path = paths.HOLDOUT_ROWS_ALL_MODELS
        sigma_path = paths.SIGMA_TABLE
        for required in (wide_path, sigma_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} not found. Run `python -m pipeline.evaluate` and "
                    "`python -m pipeline.sigma` first."
                )

        wide_df = pd.read_csv(
            wide_path,
            dtype={"stock_code": "string", "target_month": "string", "abc_class": "string"},
        )
        sigma_df = pd.read_csv(
            sigma_path,
            dtype={
                "stock_code": "string",
                "target_month": "string",
                "model": "string",
                "sigma_source": "string",
            },
        )
    else:
        # Normalise the injected frames to the same key dtypes the CSV reads above produce, so the
        # merges inside build_simulation_rows behave identically in both call modes.
        wide_df = wide_df.copy()
        for column in ("stock_code", "target_month", "abc_class"):
            wide_df[column] = wide_df[column].astype("string")
        sigma_df = sigma_df.copy()
        for column in ("stock_code", "target_month", "model", "sigma_source"):
            sigma_df[column] = sigma_df[column].astype("string")

    rows_df, dropped = build_simulation_rows(wide_df, sigma_df, MODEL_IDS)
    ctx.log_rows(
        "inventory_simulation_rows",
        before=int(len(wide_df)),
        removed=int(len(dropped)),
        after=int(len(wide_df) - len(dropped)),
    )
    if len(dropped):
        ctx.warn(
            f"{len(dropped)} (stock_code, target_month) row(s) dropped from the inventory "
            "simulation because at least one candidate model (typically B3, seasonal naive) has "
            f"no forecast there; the remaining rows have a forecast from all {len(MODEL_IDS)} "
            "candidate models"
        )

    validation = validate_simulation_inputs(rows_df, MODEL_IDS)
    write_validation_report(
        validation,
        ctx.base_dir / _repo_relative(paths.VALIDATION_REPORT),
        run_id=ctx.run_id,
    )
    if not validation.passed:
        raise FlowValidationError(validation)

    sim_by_z = pd.concat(
        [simulate_inventory(rows_df, z, policy_cfg) for z in policy_cfg.z_options],
        ignore_index=True,
    )
    sim_default = (
        sim_by_z.loc[np.isclose(sim_by_z["z"], policy_cfg.z)]
        .sort_values(["model", "policy", "target_month", "stock_code"], kind="mergesort")
        .reset_index(drop=True)[SIM_ROWS_COLUMNS]
    )

    kpis = _order_kpis(
        pd.concat(
            [
                _scoped_kpis(sim_by_z, SCOPE_OVERALL, None),
                _scoped_kpis(sim_by_z, SCOPE_MONTH, "target_month"),
                _scoped_kpis(sim_by_z, SCOPE_ABC, "abc_class"),
            ],
            ignore_index=True,
        )
    )

    concentration = excess_concentration(sim_default)

    rows_destination = ctx.out(_repo_relative(paths.HOLDOUT_SIMULATION_ROWS))
    sim_default.to_csv(
        rows_destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("holdout_simulation_rows", _repo_relative(paths.HOLDOUT_SIMULATION_ROWS))

    kpis_destination = ctx.out(_repo_relative(paths.INVENTORY_KPIS))
    kpis.to_csv(
        kpis_destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("inventory_kpis", _repo_relative(paths.INVENTORY_KPIS))

    concentration_destination = ctx.out(_repo_relative(paths.EXCESS_CONCENTRATION))
    concentration.to_csv(
        concentration_destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("excess_concentration", _repo_relative(paths.EXCESS_CONCENTRATION))

    headline = kpis.loc[(kpis["scope"] == SCOPE_OVERALL) & np.isclose(kpis["z"], policy_cfg.z)]
    metrics = {
        f"{row['model']}__{row['policy']}": {
            "fill_rate": float(row["fill_rate"]),
            "excess_units": float(row["excess_units"]),
            "stockout_units": float(row["stockout_units"]),
        }
        for row in headline.to_dict(orient="records")
    }
    ctx.record_metrics({"inventory": metrics})

    return {
        "holdout_simulation_rows": sim_default,
        "inventory_kpis": kpis,
        "excess_concentration": concentration,
    }


# --------------------------------------------------------------------------
# CLI: python -m pipeline.inventory
# --------------------------------------------------------------------------
def run() -> int:
    """Standalone entry point: ``python -m pipeline.inventory`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        cfg = load_model_config()
        with ctx.step(STEP_NAME):
            result = run_inventory_simulation(cfg, ctx)
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()

    policy_cfg = load_inventory_policy()
    kpis = result["inventory_kpis"]
    focus_models = (cfg.primary_model_id, B2)
    display = kpis.loc[
        (kpis["scope"] == SCOPE_OVERALL)
        & np.isclose(kpis["z"], policy_cfg.z)
        & kpis["model"].isin(focus_models)
    ].copy()
    display["fill_rate"] = display["fill_rate"].map(format_pct)
    display["stockout_skumonth_rate"] = display["stockout_skumonth_rate"].map(format_pct)
    print(display.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
