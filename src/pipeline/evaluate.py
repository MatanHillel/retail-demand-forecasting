"""Hold-out & back-test evaluation tables (US-19, PRD §21, §22, §23, §36, §46, §49).

Every candidate — the three naive baselines (B1, B2, B3) and the four fitted models (M1-M4) — is
scored on the same June-November 2011 hold-out, overall, per hold-out month, per training-window
ABC group and per model, plus back-test consistency across the eighteen rolling origins. All of it
goes through :func:`pipeline.metrics.metrics_table`, never a re-implemented formula (CLAUDE.md §2
rule 3-4), so the champion gates (US-22), the evaluation report (US-25) and the Streamlit app read
identical numbers.

**Two hold-out views, not one.** M1-M4's hold-out predictions come from the *fixed* model of US-17
(``holdout_predictions.csv`` — one fit through 2011-05, scored once against the whole hold-out).
Baselines need no fitting at all, so their hold-out rows are pulled out of
``backtest_predictions.csv`` (US-18) by filtering to the six hold-out target months — the
rolling-origin back-test already covers every target back to 2010-06, hold-out included, and a
baseline's prediction does not depend on which origin produced it. This is also why
:func:`evaluate` takes ``backtest_predictions_df`` and never ``features_df``/``panel_df``: it never
recomputes a baseline, it only slices one out. Reusing ``backtest_predictions_df`` for the
*consistency* table (§22 purpose 1: "consistency of performance across origins") re-scores every
model at every one of the eighteen origins, not just the hold-out six.

**Every write goes through** ``ctx.out(paths.EVAL_TABLES_DIR / "<name>")`` **(issue §8).** There is
no per-table constant in :mod:`pipeline.paths`, and ``paths.EVALUATION_REPORT`` is a different
artifact (the US-25 narrative) that this module must never touch. ``ctx.record_metrics(...)`` is
called exactly once with the full per-model mapping, because the merge is only one level deep
(``docs/interfaces.md`` §3) — a second call with the same top-level key would replace, not merge.

No champion decision, gate or threshold is *decided* here (that is US-22) — the config values this
module reads (``champion_gates.meaningful_improvement_points``,
``champion_gates.monthly_abs_bias_report_threshold``) are only used to label rows for a later
decision-maker to read, never to accept or reject a candidate.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.baselines import B2, B3, BASELINE_MODEL_IDS
from pipeline.config import MODEL_IDS, ModelConfig, load_model_config
from pipeline.metrics import format_pct, metrics_table, negative_share, relative_improvement
from pipeline.models import HOLDOUT_PREDICTIONS_COLUMNS
from pipeline.run_context import RunContext

#: Column order of ``holdout_metrics_overall.csv``.
OVERALL_COLUMNS: list[str] = [
    "model",
    "wmape",
    "bias",
    "mae",
    "rmse",
    "n_rows",
    "sum_actual",
    "sum_forecast",
    "negative_share_raw",
    "coverage_share",
    "note",
]

#: Column order of ``holdout_metrics_by_month.csv``.
BY_MONTH_COLUMNS: list[str] = ["model", "target_month", "wmape", "bias", "mae", "rmse", "n_rows"]

#: Column order of ``holdout_metrics_by_abc.csv``.
BY_ABC_COLUMNS: list[str] = [
    "model",
    "abc_class",
    "wmape",
    "bias",
    "mae",
    "rmse",
    "n_rows",
    "sum_actual",
]

#: Column order of ``improvement_vs_b2.csv``.
IMPROVEMENT_VS_B2_COLUMNS: list[str] = [
    "model",
    "wmape",
    "wmape_b2",
    "wmape_points_vs_b2",
    "relative_pct",
    "meaningful",
]

#: Column order of ``backtest_consistency.csv`` — per-origin rows carrying a per-model summary.
BACKTEST_CONSISTENCY_COLUMNS: list[str] = [
    "model",
    "forecast_origin",
    "target_month",
    "n_rows",
    "wmape",
    "bias",
    "wmape_mean",
    "wmape_std",
    "wmape_min",
    "wmape_max",
    "bias_mean",
    "months_abs_bias_gt_threshold",
]

#: Column order of ``holdout_rows_all_models.csv``.
WIDE_COLUMNS: list[str] = [
    "stock_code",
    "target_month",
    "abc_class",
    "actual",
    *(f"pred_{model_id}" for model_id in MODEL_IDS),
]

_B3_COVERAGE_NOTE = (
    "B3 (seasonal naive) has no observed month t-12 for every product; metrics computed only "
    "on rows where a seasonal forecast exists (see coverage_share)."
)


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _order_by_model(frame: pd.DataFrame, secondary: list[str] | None = None) -> pd.DataFrame:
    """Sort rows in the canonical candidate order B1, B2, B3, M1, M2, M3, M4 (issue §2)."""
    ordered = frame.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=list(MODEL_IDS), ordered=True)
    ordered = ordered.sort_values(["model", *(secondary or [])], kind="mergesort")
    ordered["model"] = ordered["model"].astype(str)
    return ordered.reset_index(drop=True)


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with a defined prediction — excludes B3's unobserved-t-12 rows.

    Those rows are never dropped upstream (module docstring); they are excluded only here, at
    metric time, the same convention :func:`pipeline.backtest.backtest_summary` uses.
    """
    return frame.loc[frame["prediction"].notna()]


# --------------------------------------------------------------------------
# combine the two hold-out sources: fixed M1-M4 + baselines sliced from the back-test
# --------------------------------------------------------------------------
def _assert_identical_row_sets(predictions: pd.DataFrame) -> None:
    """Raise unless every model covers the identical ``(stock_code, target_month)`` set."""
    row_sets = {
        model_id: frozenset(zip(group["stock_code"], group["target_month"], strict=True))
        for model_id, group in predictions.groupby("model", sort=False)
    }
    reference_model, reference_set = next(iter(row_sets.items()))
    mismatched = [model_id for model_id, rows in row_sets.items() if rows != reference_set]
    if mismatched:
        raise AssertionError(
            f"model(s) {mismatched} do not cover the same (stock_code, target_month) set as "
            f"{reference_model!r} in the combined hold-out frame"
        )


def _combined_holdout_predictions(
    holdout_predictions_df: pd.DataFrame,
    backtest_predictions_df: pd.DataFrame,
    cfg: ModelConfig,
) -> pd.DataFrame:
    """M1-M4's fixed-model hold-out rows plus B1/B2/B3 sliced out of the back-test.

    See the module docstring for why baselines come from ``backtest_predictions_df`` rather than
    being recomputed. The two sources are combined on ``(stock_code, target_month)``; the join is
    enforced implicitly by :func:`_assert_identical_row_sets` — every candidate must cover the
    identical row set, or the run raises rather than silently comparing candidates on different
    rows (issue §3 technical note).
    """
    holdout_start = cfg.split.holdout_targets.start
    holdout_end = cfg.split.holdout_targets.end

    baseline_mask = backtest_predictions_df["model"].isin(BASELINE_MODEL_IDS) & (
        backtest_predictions_df["target_month"].between(holdout_start, holdout_end)
    )
    baseline_holdout = backtest_predictions_df.loc[
        baseline_mask, HOLDOUT_PREDICTIONS_COLUMNS
    ].copy()

    combined = pd.concat(
        [holdout_predictions_df[HOLDOUT_PREDICTIONS_COLUMNS], baseline_holdout],
        ignore_index=True,
    )
    combined["stock_code"] = combined["stock_code"].astype(str)
    combined["target_month"] = combined["target_month"].astype(str)

    _assert_identical_row_sets(combined)
    return combined


# --------------------------------------------------------------------------
# holdout_metrics_overall.csv
# --------------------------------------------------------------------------
def _holdout_metrics_overall(combined: pd.DataFrame) -> pd.DataFrame:
    scored = _scored(combined)
    table = metrics_table(
        scored, actual_col="actual", forecast_col="prediction", group_cols=["model"]
    )
    table = table[["model", "wmape", "bias", "mae", "rmse", "n_rows", "sum_actual", "sum_forecast"]]

    raw_negative = (
        scored.groupby("model", sort=False)["prediction_raw"]
        .apply(negative_share)
        .rename("negative_share_raw")
        .reset_index()
    )
    table = table.merge(raw_negative, on="model", how="left")

    total_rows = (
        combined.groupby("model", sort=False).size().rename("total_rows").reset_index()
    )
    table = table.merge(total_rows, on="model", how="left")
    table["coverage_share"] = table["n_rows"] / table["total_rows"]
    table["note"] = np.where(table["model"] == B3, _B3_COVERAGE_NOTE, "")
    table = table.drop(columns=["total_rows"])

    return _order_by_model(table[OVERALL_COLUMNS])


# --------------------------------------------------------------------------
# holdout_metrics_by_month.csv
# --------------------------------------------------------------------------
def _holdout_metrics_by_month(combined: pd.DataFrame) -> pd.DataFrame:
    scored = _scored(combined)
    table = metrics_table(
        scored, actual_col="actual", forecast_col="prediction", group_cols=["model", "target_month"]
    )
    table = table[["model", "target_month", "wmape", "bias", "mae", "rmse", "n_rows"]]
    return _order_by_model(table, secondary=["target_month"])


# --------------------------------------------------------------------------
# holdout_metrics_by_abc.csv — training-window ABC only, never recomputed here
# --------------------------------------------------------------------------
def _holdout_metrics_by_abc(combined: pd.DataFrame, abc_train_df: pd.DataFrame) -> pd.DataFrame:
    scored = _scored(combined)
    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)

    merged = scored.merge(abc_lookup, on="stock_code", how="left")
    if merged["abc_class"].isna().any():
        missing = sorted(merged.loc[merged["abc_class"].isna(), "stock_code"].unique())
        raise ValueError(f"stock_code(s) absent from abc_train_df (US-16 join key): {missing}")

    table = metrics_table(
        merged, actual_col="actual", forecast_col="prediction", group_cols=["model", "abc_class"]
    )
    table = table[["model", "abc_class", "wmape", "bias", "mae", "rmse", "n_rows", "sum_actual"]]
    return _order_by_model(table, secondary=["abc_class"])


# --------------------------------------------------------------------------
# improvement_vs_b2.csv
# --------------------------------------------------------------------------
def _improvement_vs_b2(overall: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    wmape_by_model = overall.set_index("model")["wmape"]
    wmape_b2 = float(wmape_by_model[B2])

    rows = []
    for model_id in MODEL_IDS:
        wmape_model = float(wmape_by_model[model_id])
        improvement = relative_improvement(wmape_model, wmape_b2)
        rows.append(
            {
                "model": model_id,
                "wmape": wmape_model,
                "wmape_b2": wmape_b2,
                "wmape_points_vs_b2": improvement.points,
                "relative_pct": improvement.relative_pct,
                "meaningful": (
                    improvement.points >= cfg.champion_gates.meaningful_improvement_points
                ),
            }
        )
    return pd.DataFrame(rows, columns=IMPROVEMENT_VS_B2_COLUMNS)


# --------------------------------------------------------------------------
# backtest_consistency.csv — every rolling origin, not just the hold-out six (§22 purpose 1)
# --------------------------------------------------------------------------
def _backtest_consistency(backtest_predictions_df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    scored = backtest_predictions_df.loc[backtest_predictions_df["prediction"].notna()]
    per_origin = metrics_table(
        scored,
        actual_col="actual",
        forecast_col="prediction",
        group_cols=["model", "forecast_origin", "target_month"],
    )[["model", "forecast_origin", "target_month", "n_rows", "wmape", "bias"]]

    threshold = cfg.champion_gates.monthly_abs_bias_report_threshold
    summary_rows = []
    for model_id, group in per_origin.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": model_id,
                "wmape_mean": float(group["wmape"].mean()),
                "wmape_std": float(group["wmape"].std()),
                "wmape_min": float(group["wmape"].min()),
                "wmape_max": float(group["wmape"].max()),
                "bias_mean": float(group["bias"].mean()),
                "months_abs_bias_gt_threshold": int((group["bias"].abs() > threshold).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    merged = per_origin.merge(summary, on="model", how="left")
    return _order_by_model(merged[BACKTEST_CONSISTENCY_COLUMNS], secondary=["forecast_origin"])


# --------------------------------------------------------------------------
# holdout_rows_all_models.csv
# --------------------------------------------------------------------------
def _holdout_rows_all_models(combined: pd.DataFrame, abc_train_df: pd.DataFrame) -> pd.DataFrame:
    pivoted = combined.pivot(
        index=["stock_code", "target_month"], columns="model", values="prediction"
    )
    pivoted = pivoted.reindex(columns=list(MODEL_IDS))
    pivoted.columns = [f"pred_{model_id}" for model_id in pivoted.columns]

    actual = combined.groupby(["stock_code", "target_month"], sort=False)["actual"].first()
    wide = pivoted.join(actual).reset_index()

    abc_lookup = abc_train_df[["stock_code", "abc_class"]].copy()
    abc_lookup["stock_code"] = abc_lookup["stock_code"].astype(str)
    wide = wide.merge(abc_lookup, on="stock_code", how="left")

    return (
        wide.sort_values(["stock_code", "target_month"], kind="mergesort")
        .reset_index(drop=True)[WIDE_COLUMNS]
    )


# --------------------------------------------------------------------------
# evaluation_summary.json
# --------------------------------------------------------------------------
def _evaluation_summary(
    overall: pd.DataFrame, combined: pd.DataFrame, cfg: ModelConfig, ctx: RunContext
) -> dict[str, Any]:
    holdout_months = [
        str(period)
        for period in pd.period_range(
            cfg.split.holdout_targets.start, cfg.split.holdout_targets.end, freq="M"
        )
    ]
    return {
        "overall": overall.to_dict(orient="records"),
        "holdout_months": holdout_months,
        "n_products": int(combined["stock_code"].nunique()),
        "abc_source": f"training window through {cfg.split.train_targets.end}",
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": ctx.run_id,
    }


# --------------------------------------------------------------------------
# writers — every one of the seven outputs through ctx.out() (issue §8)
# --------------------------------------------------------------------------
def _write_table(frame: pd.DataFrame, name: str, key: str, ctx: RunContext) -> Path:
    canonical = paths.EVAL_TABLES_DIR / name
    destination = ctx.out(_repo_relative(canonical))
    frame.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact(key, _repo_relative(canonical))
    return destination


def _write_summary(summary: dict[str, Any], ctx: RunContext) -> Path:
    canonical = paths.EVAL_TABLES_DIR / "evaluation_summary.json"
    destination = ctx.out(_repo_relative(canonical))
    destination.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ctx.record_artifact("evaluation_summary", _repo_relative(canonical))
    return destination


# --------------------------------------------------------------------------
# evaluate() — the public entry point (issue §2/§8)
# --------------------------------------------------------------------------
def evaluate(
    holdout_predictions_df: pd.DataFrame,
    backtest_predictions_df: pd.DataFrame,
    abc_train_df: pd.DataFrame,
    cfg: ModelConfig,
    ctx: RunContext,
) -> dict[str, pd.DataFrame]:
    """Score every candidate on the hold-out and write the seven US-19 evaluation artifacts.

    Never re-reads ``paths.*`` mid-run: every input is a DataFrame the caller already has, which is
    what keeps this function safe to call from inside the orchestrated Flow, where the upstream
    files are still sitting in ``artifacts/_staging/<run_id>/`` (``docs/interfaces.md`` §8).
    """
    with ctx.step("evaluate"):
        combined = _combined_holdout_predictions(
            holdout_predictions_df, backtest_predictions_df, cfg
        )

        overall = _holdout_metrics_overall(combined)
        by_month = _holdout_metrics_by_month(combined)
        by_abc = _holdout_metrics_by_abc(combined, abc_train_df)
        improvement = _improvement_vs_b2(overall, cfg)
        consistency = _backtest_consistency(backtest_predictions_df, cfg)
        wide = _holdout_rows_all_models(combined, abc_train_df)
        summary = _evaluation_summary(overall, combined, cfg, ctx)

        ctx.log_rows(
            "evaluate_holdout_rows",
            before=int(len(combined)),
            removed=0,
            after=int(len(combined)),
        )

        _write_table(overall, "holdout_metrics_overall.csv", "holdout_metrics_overall", ctx)
        _write_table(by_month, "holdout_metrics_by_month.csv", "holdout_metrics_by_month", ctx)
        _write_table(by_abc, "holdout_metrics_by_abc.csv", "holdout_metrics_by_abc", ctx)
        _write_table(improvement, "improvement_vs_b2.csv", "improvement_vs_b2", ctx)
        _write_table(consistency, "backtest_consistency.csv", "backtest_consistency", ctx)
        _write_table(wide, "holdout_rows_all_models.csv", "holdout_rows_all_models", ctx)
        _write_summary(summary, ctx)

        holdout_metrics = {
            str(row["model"]): {"wmape": float(row["wmape"]), "bias": float(row["bias"])}
            for row in overall.to_dict(orient="records")
        }
        ctx.record_metrics({"holdout": holdout_metrics})

    return {
        "holdout_metrics_overall": overall,
        "holdout_metrics_by_month": by_month,
        "holdout_metrics_by_abc": by_abc,
        "improvement_vs_b2": improvement,
        "backtest_consistency": consistency,
        "holdout_rows_all_models": wide,
        "evaluation_summary": summary,
    }


# --------------------------------------------------------------------------
# CLI: python -m pipeline.evaluate
# --------------------------------------------------------------------------
def _read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "stock_code": "string",
            "forecast_origin": "string",
            "target_month": "string",
            "model": "string",
        },
    )


def run() -> int:
    """Standalone entry point: ``python -m pipeline.evaluate`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        holdout_path = paths.FORECASTS_DIR / "holdout_predictions.csv"
        abc_path = paths.EVAL_TABLES_DIR / "abc_train.csv"
        for required in (holdout_path, paths.BACKTEST_PREDICTIONS, abc_path):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} not found. Run `python -m pipeline.models train`, "
                    "`python -m pipeline.backtest` and `python -m pipeline.split abc` first."
                )

        cfg = load_model_config()
        holdout_predictions = _read_predictions(holdout_path)
        backtest_predictions = _read_predictions(paths.BACKTEST_PREDICTIONS)
        abc_train_df = pd.read_csv(abc_path, dtype={"stock_code": "string"})

        result = evaluate(holdout_predictions, backtest_predictions, abc_train_df, cfg, ctx)
        overall = result["holdout_metrics_overall"]
        display = overall.copy()
        display["wmape"] = display["wmape"].map(format_pct)
        display["bias"] = display["bias"].map(format_pct)
        print(display.to_string(index=False))
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
