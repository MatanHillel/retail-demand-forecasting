"""Rolling-origin back-test (US-18, PRD §18, §21, §22, §26, §40, §43).

A **rolling-origin back-test** means pretending to stand at the end of each past month (an
"origin"), training only on what was actually known then, forecasting the single next month, and
repeating that for every origin in turn. :func:`pipeline.split.rolling_origins` fixes the eighteen
origins this project replays — 2010-05 … 2011-10, whose targets are 2010-06 … 2011-11 — the same
origins for every candidate, so wMAPE/Bias comparisons across models are comparisons on identical
rows (§20 "gates apply to every candidate, baselines included").

This module does not fit anything itself. For the four trainable candidates (M1-M4) it replays
:func:`pipeline.models.fit_predict_one_origin`, **unmodified**, at every origin — that function is
the single place the no-leakage boundary is enforced (fit on ``target_month <= origin``, predict
``target_month == origin + 1``), and reusing it here means the rolling back-test and US-17's
hyper-parameter tuning can never drift apart on what "no leakage" means. Because the fit always
includes every row known at the origin with no lower bound, the window is **expanding**, never
sliding, automatically. Hyper-parameters come from ``model_config.yaml`` and are identical at every
origin (§22) — nothing here ever refits or reselects them.

For the three baselines (B1/B2/B3) nothing is fit at all:
:func:`pipeline.baselines.predict_baselines` is called **once**, before the origin loop, and each
origin's baseline rows are the subset with ``target_month == origin + 1``. Because that function
emits a row for every feature row, the baseline row set at each origin is identical by construction
to the trainable models' row set.

## Columns of ``backtest_predictions.csv`` (:data:`BACKTEST_COLUMNS`)

``forecast_origin, target_month, stock_code, model, actual, prediction_raw, prediction, residual``

* ``actual`` is ``y`` from the feature row for that ``(stock_code, target_month)`` — the units
  actually sold.
* ``prediction`` is the forecast actually used downstream (negative model output clipped to zero);
  ``prediction_raw`` is the unclipped value. The column is **always populated**, for every
  candidate — not just M1 — because M4 also produced negative raw output in US-17. For a baseline,
  clipping never applies, so ``prediction_raw`` simply equals ``prediction``.
* ``residual = actual - prediction`` (§26's ``e = Actual - Forecast``), computed from the
  **clipped** ``prediction`` because that is the number the inventory policy (US-20 σ onward)
  actually consumes. Everything downstream inherits this sign convention.
* B3 (seasonal naive) has no observed month ``t-12`` for a short-lived product: those rows keep
  ``prediction = NaN`` and therefore ``residual = NaN``. **The rows still exist** in this file —
  they are excluded only at metric time (:func:`backtest_summary`), never dropped here.

## Columns of ``backtest_by_origin.csv`` (:data:`BACKTEST_BY_ORIGIN_COLUMNS`)

``model, forecast_origin, target_month, n_rows, wmape, bias`` — one row per ``(model, origin)``,
computed by :func:`backtest_summary` with B3's ``NaN`` rows excluded from the metric, never from
the predictions file itself.

## Assertions

Three invariants are checked, and any violation **raises** rather than warns — a silent failure
here would corrupt every σ and inventory number downstream:

1. ``forecast_origin`` is exactly one month before ``target_month`` on every row.
2. Every model covers the identical ``(stock_code, target_month)`` set.
3. Every ``target_month`` falls inside ``[first_origin + 1, last_origin + 1]``, and the configured
   ``never_score`` month(s) (2011-12, the partial month) never appear.

The stronger guarantee — "the training frame used at an origin never contains a target after that
origin" — lives in :func:`pipeline.models.fit_predict_one_origin` and is unit-tested there; it is
re-verified at this engine's level with a monkeypatch spy in ``tests/test_backtest.py``.

## Two scope decisions

* **No caching.** Caching per ``(model, origin)`` was considered and rejected: the full back-test
  runs in a couple of minutes, so a hash key, a git-ignored cache tree and its own tests would cost
  more than they save.
* **No gzip.** The predictions file is well under the 50 MiB artifact-size threshold (measured, not
  assumed — see the run's own report). If a future change ever pushes it over that line, the fix is
  to change the *value* of :data:`pipeline.paths.BACKTEST_PREDICTIONS`, never to add a second path
  constant.

No metric threshold, gate or champion decision is made in this module — that is US-19/US-22.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

from pipeline import paths
from pipeline.baselines import BASELINE_MODEL_IDS, predict_baselines
from pipeline.config import MODEL_IDS, ModelConfig, load_model_config
from pipeline.contract import read_panel
from pipeline.metrics import metrics_table
from pipeline.models import ORIGIN_PREDICTIONS_COLUMNS, TRAINABLE_MODEL_IDS, fit_predict_one_origin
from pipeline.run_context import RunContext
from pipeline.split import SplitSpec, rolling_origins

#: Column order of ``artifacts/forecasts/backtest_predictions.csv``.
BACKTEST_COLUMNS: list[str] = [
    "forecast_origin",
    "target_month",
    "stock_code",
    "model",
    "actual",
    "prediction_raw",
    "prediction",
    "residual",
]

#: Column order of ``artifacts/reports/evaluation_tables/backtest_by_origin.csv``.
BACKTEST_BY_ORIGIN_COLUMNS: list[str] = [
    "model",
    "forecast_origin",
    "target_month",
    "n_rows",
    "wmape",
    "bias",
]


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _assert_origin_precedes_target(predictions: pd.DataFrame) -> None:
    """Raise unless ``forecast_origin`` is exactly one month before ``target_month`` everywhere."""
    origins = pd.PeriodIndex(predictions["forecast_origin"], freq="M")
    expected_target = (origins + 1).astype(str)
    mismatched = expected_target != predictions["target_month"].to_numpy()
    if mismatched.any():
        raise AssertionError(
            f"{int(mismatched.sum())} row(s) have forecast_origin not exactly one month before "
            "target_month"
        )


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
            f"{reference_model!r}"
        )


def _assert_target_range(predictions: pd.DataFrame, split: SplitSpec) -> None:
    """Raise unless every target is inside ``[first_origin+1, last_origin+1]`` and never-scored."""
    first_target = str(pd.Period(split.backtest.first_origin, freq="M") + 1)
    last_target = str(pd.Period(split.backtest.last_origin, freq="M") + 1)
    out_of_range = ~predictions["target_month"].between(first_target, last_target)
    if out_of_range.any():
        bad = sorted(predictions.loc[out_of_range, "target_month"].unique())
        raise AssertionError(f"target_month outside [{first_target}, {last_target}]: {bad}")

    never_score = set(split.split.never_score)
    leaked = predictions["target_month"].isin(never_score)
    if leaked.any():
        raise AssertionError(
            f"never-scored month(s) {sorted(never_score)} appear in the back-test "
            f"({int(leaked.sum())} row(s))"
        )


def backtest(
    features_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: ModelConfig,
    ctx: RunContext,
    split: SplitSpec,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Replay every candidate at every rolling origin and write ``backtest_predictions.csv``.

    ``models=None`` (the default) runs all seven candidates in :data:`pipeline.config.MODEL_IDS`; a
    supplied list filters to those ids and raises ``ValueError`` on an id that is not one of the
    seven normative ids. Must be called with an open ``ctx`` — the write and the per-origin
    ``ctx.log_rows`` waterfall both need it.
    """
    if models is None:
        selected = list(MODEL_IDS)
    else:
        unknown = sorted(set(models) - set(MODEL_IDS))
        if unknown:
            raise ValueError(f"unknown model id(s): {unknown}; expected one of {list(MODEL_IDS)}")
        selected = list(models)

    selected_baselines = [model_id for model_id in BASELINE_MODEL_IDS if model_id in selected]
    selected_trainable = [model_id for model_id in TRAINABLE_MODEL_IDS if model_id in selected]

    origins = rolling_origins(split)

    # B1/B2/B3 need no fitting: compute every baseline prediction once, up front, then slice each
    # origin's target rows out of it below (module docstring).
    baseline_predictions: pd.DataFrame | None = None
    if selected_baselines:
        all_baselines = predict_baselines(features_df, panel_df)
        baseline_predictions = all_baselines.loc[
            all_baselines["model"].isin(selected_baselines)
        ].copy()
        # Baselines are never clipped, so prediction_raw is defined as equal to prediction rather
        # than left mostly empty (module docstring).
        baseline_predictions["prediction_raw"] = baseline_predictions["prediction"]

    actual = features_df[["stock_code", "target_month", "y"]].rename(columns={"y": "actual"})

    with ctx.step("backtest"):
        frames: list[pd.DataFrame] = []
        seconds_by_origin: dict[str, float] = {}

        for origin in origins:
            started = time.perf_counter()
            target = str(pd.Period(origin, freq="M") + 1)
            origin_frames: list[pd.DataFrame] = []

            for model_id in selected_trainable:
                origin_frames.append(
                    fit_predict_one_origin(features_df, model_id, origin, cfg, ctx.seed)
                )

            if baseline_predictions is not None:
                origin_baseline = baseline_predictions.loc[
                    baseline_predictions["target_month"] == target, ORIGIN_PREDICTIONS_COLUMNS
                ]
                origin_frames.append(origin_baseline)

            origin_frame = pd.concat(origin_frames, ignore_index=True)
            frames.append(origin_frame)

            duration = time.perf_counter() - started
            seconds_by_origin[origin] = round(float(duration), 6)
            ctx.log_rows(
                f"origin_{origin}", before=len(origin_frame), removed=0, after=len(origin_frame)
            )
            ctx.logger.info(f"origin {origin}: {len(origin_frame)} rows in {duration:.3f}s")

        predictions = pd.concat(frames, ignore_index=True)
        predictions = predictions.merge(actual, on=["stock_code", "target_month"], how="left")
        predictions["residual"] = predictions["actual"] - predictions["prediction"]

        _assert_origin_precedes_target(predictions)
        _assert_identical_row_sets(predictions)
        _assert_target_range(predictions, split)

        predictions = (
            predictions.sort_values(["model", "target_month", "stock_code"], kind="mergesort")
            .reset_index(drop=True)[BACKTEST_COLUMNS]
        )

        destination = ctx.out(_repo_relative(paths.BACKTEST_PREDICTIONS))
        predictions.to_csv(
            destination, index=False, float_format="%.4f", lineterminator="\n", encoding="utf-8"
        )
        ctx.record_artifact("backtest_predictions", _repo_relative(paths.BACKTEST_PREDICTIONS))
        ctx.record_metrics({"backtest_seconds_by_origin": seconds_by_origin})

    return predictions


def backtest_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per ``(model, origin)`` wMAPE/Bias table — pure, no ``ctx``, no disk (§22, §23).

    B3's unobservable rows (``prediction`` is ``NaN``) are excluded here, at metric time — never
    upstream in :func:`backtest`, which keeps every row that exists.
    """
    scored = predictions.loc[predictions["prediction"].notna()]
    table = metrics_table(
        scored,
        actual_col="actual",
        forecast_col="prediction",
        group_cols=["model", "forecast_origin", "target_month"],
    )
    return (
        table.loc[:, BACKTEST_BY_ORIGIN_COLUMNS]
        .sort_values(["model", "target_month"], kind="mergesort")
        .reset_index(drop=True)
    )


def write_backtest_summary(frame: pd.DataFrame, ctx: RunContext) -> Path:
    """Write ``artifacts/reports/evaluation_tables/backtest_by_origin.csv`` through ``ctx.out()``.
    """
    canonical = paths.EVAL_TABLES_DIR / "backtest_by_origin.csv"
    destination = ctx.out(_repo_relative(canonical))
    frame.to_csv(
        destination, index=False, float_format="%.4f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact("backtest_by_origin", _repo_relative(canonical))
    return destination


# --------------------------------------------------------------------------
# CLI: python -m pipeline.backtest [--models MODEL_ID [MODEL_ID ...]]
# --------------------------------------------------------------------------
def _read_features() -> pd.DataFrame:
    if not paths.FEATURES.is_file():
        raise FileNotFoundError(
            f"{paths.FEATURES} not found. Run `python -m pipeline.features` first."
        )
    return pd.read_csv(
        paths.FEATURES,
        dtype={"stock_code": "string", "forecast_origin": "string", "target_month": "string"},
    )


def run(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``python -m pipeline.backtest [--models MODEL_ID ...]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    models: list[str] | None = None
    if args:
        if args[0] != "--models" or len(args) < 2:
            print(
                "usage: python -m pipeline.backtest [--models MODEL_ID [MODEL_ID ...]]",
                file=sys.stderr,
            )
            return 2
        models = args[1:]

    ctx = RunContext.start(mode="no-llm")
    try:
        if not paths.CLEAN_DATA.is_file():
            raise FileNotFoundError(
                f"{paths.CLEAN_DATA} not found. Run `python -m pipeline.panel` first."
            )
        cfg = load_model_config()
        spec = SplitSpec.load()
        features = _read_features()
        panel = read_panel(paths.CLEAN_DATA)

        predictions = backtest(features, panel, cfg, ctx, spec, models=models)

        summary = backtest_summary(predictions)
        write_backtest_summary(summary, ctx)

        scored = predictions.loc[predictions["prediction"].notna()]
        overall = metrics_table(
            scored, actual_col="actual", forecast_col="prediction", group_cols=["model"]
        )
        for record in overall.to_dict(orient="records"):
            print(
                f"{record['model']}: wMAPE={record['wmape']:.4f} Bias={record['bias']:.4f} "
                f"(n={int(record['n_rows'])})"
            )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
