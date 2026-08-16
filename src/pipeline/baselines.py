"""Naive forecasting baselines B1, B2, B3 (US-16, PRD §18.1, §19, §49).

Three rules of thumb every real model must beat: B1 "next month equals last month", B2 "next month
equals the average of the last three months" (the **main baseline**, §19) and B3 "next month equals
the same month last year" (seasonal naive, **reference only** — never a feature, never the baseline
a model is compared against for the champion gates, §20).

No model fitting happens here (that is US-17) and no metric is computed here (that is
:mod:`pipeline.metrics`, US-15, already merged) — this module only turns the feature rows of
``features.csv`` into one prediction per ``(stock_code, target_month, model)``, in the same long
format the back-test and evaluation later use for *every* candidate, models included, so all of
them are compared on identical rows (§20 — "gates apply to every candidate, baselines included").

Scikit-learn's shuffled train/test splitter is never imported under ``src/`` (§21) — nothing here
or anywhere else in this project shuffles months; the split lives in :mod:`pipeline.split`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.contract import read_panel
from pipeline.run_context import RunContext

#: "Seasonal naive" *means* "twelve months back" for monthly retail data — a definition, like the
#: 3/6-month windows in :mod:`pipeline.features`, not a tunable. Every actual date still comes from
#: config; this integer is what the word "seasonal" means.
_SEASONAL_LAG_MONTHS = 12

B1 = "B1_last_month"
B2 = "B2_ma3"
B3 = "B3_seasonal_naive"

#: The three baseline model ids, in the canonical order of ``model_config.yaml -> models``.
BASELINE_MODEL_IDS: tuple[str, str, str] = (B1, B2, B3)

#: Column order of ``baseline_predictions.csv`` — the long format every candidate uses.
PREDICTIONS_COLUMNS: list[str] = [
    "stock_code",
    "forecast_origin",
    "target_month",
    "model",
    "prediction",
]


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _seasonal_lookup(panel_df: pd.DataFrame) -> pd.Series:
    """``units_sold`` indexed by ``(stock_code, month)`` for an exact panel look-up."""
    panel = panel_df.copy()
    panel["stock_code"] = panel["stock_code"].astype(str)
    panel["month"] = panel["month"].astype(str)
    return panel.set_index(["stock_code", "month"])["units_sold"]


def _b3_predictions(features_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.Series:
    """The panel's ``units_sold`` exactly twelve months before ``target_month``, or ``NaN``.

    A missing lookup means the product had no observed row that month — a short lifecycle, not a
    zero-sales month — so it is left ``NaN`` rather than filled with 0 (§18.1). NaN rows are
    excluded from B3 metrics at metric time, never here.
    """
    lookup = _seasonal_lookup(panel_df)
    stock_code = features_df["stock_code"].astype(str).to_numpy()
    seasonal_month = (
        (pd.PeriodIndex(features_df["target_month"].astype(str), freq="M") - _SEASONAL_LAG_MONTHS)
        .astype(str)
    )
    values = [
        lookup.get((code, month), np.nan)
        for code, month in zip(stock_code, seasonal_month, strict=True)
    ]
    return pd.Series(values, index=features_df.index, dtype="float64")


def predict_baselines(features_df: pd.DataFrame, panel_df: pd.DataFrame) -> pd.DataFrame:
    """B1/B2/B3 predictions for every row of ``features_df`` (long format, §19).

    B1 = ``lag_1`` (last month's units). B2 = ``rolling_mean_3`` (mean of the last three months —
    main baseline). B3 = the panel's units twelve months earlier, ``NaN`` where unobserved
    (reference only). All three baselines emit a row for *every* feature row — the same active set
    every model is scored on — so B3's NaNs are the only thing that distinguishes it, not a
    different row set.
    """
    base = features_df[["stock_code", "forecast_origin", "target_month"]].copy()
    base["stock_code"] = base["stock_code"].astype(str)

    predictions_by_model = {
        B1: features_df["lag_1"].astype("float64"),
        B2: features_df["rolling_mean_3"].astype("float64"),
        B3: _b3_predictions(features_df, panel_df),
    }

    frames = []
    for model_id in BASELINE_MODEL_IDS:
        part = base.copy()
        part["model"] = model_id
        part["prediction"] = predictions_by_model[model_id].to_numpy()
        frames.append(part)

    predictions = pd.concat(frames, ignore_index=True)
    return (
        predictions.sort_values(["stock_code", "target_month", "model"], kind="mergesort")
        .reset_index(drop=True)[PREDICTIONS_COLUMNS]
    )


def write_baseline_predictions(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``artifacts/forecasts/baseline_predictions.csv`` through ``ctx.out()``."""
    canonical = paths.FORECASTS_DIR / "baseline_predictions.csv"
    destination = ctx.out(_repo_relative(canonical))
    frame.to_csv(
        destination,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("baseline_predictions", _repo_relative(canonical))
    return destination


def run() -> int:
    """Standalone entry point: build and write the three baselines from ``features.csv`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("baselines"):
            if not paths.FEATURES.is_file():
                raise FileNotFoundError(
                    f"{paths.FEATURES} not found. Run `python -m pipeline.features` first."
                )
            if not paths.CLEAN_DATA.is_file():
                raise FileNotFoundError(
                    f"{paths.CLEAN_DATA} not found. Run `python -m pipeline.panel` first."
                )
            features = pd.read_csv(
                paths.FEATURES,
                dtype={
                    "stock_code": "string",
                    "forecast_origin": "string",
                    "target_month": "string",
                },
            )
            panel = read_panel(paths.CLEAN_DATA)

            predictions = predict_baselines(features, panel)
            write_baseline_predictions(predictions, ctx)

            counts = predictions.groupby("model").size()
            for model_id in BASELINE_MODEL_IDS:
                rows = predictions.loc[predictions["model"] == model_id, "prediction"]
                nan_rows = int(rows.isna().sum())
                ctx.log_rows(
                    f"baseline_{model_id}",
                    before=int(len(rows)),
                    removed=nan_rows,
                    after=int(len(rows)) - nan_rows,
                )

            b3_rows = predictions.loc[predictions["model"] == B3, "prediction"]
            b3_nan_share = float(b3_rows.isna().mean()) if len(b3_rows) else 0.0
            ctx.warn(
                f"{B3}: {b3_nan_share:.2%} of rows have no observed month t-12 "
                "(NaN, reference only — excluded from B3 metrics)"
            )
            ctx.record_metrics(
                {
                    "baseline_rows": int(len(predictions)),
                    "baseline_b3_nan_share": round(b3_nan_share, 6),
                }
            )
        print(
            f"rows: {ctx.metrics['baseline_rows']}\n"
            f"{B1}: {int(counts.get(B1, 0))}\n"
            f"{B2}: {int(counts.get(B2, 0))}\n"
            f"{B3}: {int(counts.get(B3, 0))}\n"
            f"{B3} coverage (non-NaN) share: {1 - ctx.metrics['baseline_b3_nan_share']:.2%}"
        )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
