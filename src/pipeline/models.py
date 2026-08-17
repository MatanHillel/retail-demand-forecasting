"""Forecasting model candidates M1-M4 (US-17, PRD §17, §19, §20, §36, §40, §43).

One **global** model per candidate (§17) — never a model per SKU. Four estimators are built from
the §17 features: Linear Regression (M1, a linear baseline challenger) and three
``HistGradientBoostingRegressor`` variants that differ only in loss function — Poisson (M2, the
primary model: a loss built for counting things like units sold, and it never predicts a negative
number), squared-error (M3) and absolute-error (M4, the accuracy challenger).

:func:`fit_predict_one_origin` is the single primitive that enforces the no-leakage boundary: fit
on every row whose ``target_month`` is at or before the origin, predict the single row one month
ahead. It is reused, unmodified, by :func:`tune` here and by the rolling back-test of US-18, so the
leakage rule lives in exactly one place. :func:`train_models` is different on purpose — it fits
*one* model through the last training month and scores it once against the whole hold-out window
(§21's "final hold-out"), which is additional to, not a replacement for, the rolling-origin
back-test.

Hyper-parameter selection (:func:`tune`) never touches the hold-out: it only replays
:func:`fit_predict_one_origin` over the rolling origins whose target falls inside the internal
validation window (§21), which is itself inside the training period. Nothing here corrects a
model's systematic lean after the fact — that idea proved unstable in early exploration and is
explicitly out of scope for this project (§20) — and no metric threshold or champion decision is
made in this module (that is US-22/US-23).
"""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import paths
from pipeline.config import ModelConfig, clear_config_cache, load_model_config
from pipeline.metrics import bias, metrics_table, wmape
from pipeline.run_context import RunContext
from pipeline.split import SplitSpec, holdout_mask, train_mask, validation_origins

#: Model ids this module trains — the seven normative ids of ``pipeline.config.MODEL_IDS`` minus
#: the three baselines (§19). Derived from the config rather than hand-copied so a future model
#: added to ``model_config.yaml`` cannot silently drift from what this module actually trains.
#: Which model ids exist does not change between a ``tune`` run and a ``train`` run (tuning only
#: edits a model's ``params:``), so resolving this once at import time is safe.
TRAINABLE_MODEL_IDS: tuple[str, ...] = tuple(
    name for name, spec in load_model_config().models.items() if spec.kind != "baseline"
)

#: Column order of the frame :func:`fit_predict_one_origin` returns.
ORIGIN_PREDICTIONS_COLUMNS: list[str] = [
    "stock_code",
    "forecast_origin",
    "target_month",
    "model",
    "prediction_raw",
    "prediction",
]

#: Column order of ``artifacts/forecasts/holdout_predictions.csv``.
HOLDOUT_PREDICTIONS_COLUMNS: list[str] = [
    "stock_code",
    "forecast_origin",
    "target_month",
    "model",
    "actual",
    "prediction_raw",
    "prediction",
]

#: Column order of ``artifacts/reports/evaluation_tables/tuning_results.csv``.
TUNING_RESULTS_COLUMNS: list[str] = [
    "model",
    "learning_rate",
    "max_leaf_nodes",
    "max_iter",
    "wmape",
    "bias",
    "n_rows",
]

# A model header inside ``models:`` is indented by exactly two spaces and has nothing after the
# colon but an optional comment; every deeper key (``kind:``, ``params:``, ...) is indented by at
# least four, so it can never match this pattern and reset ``current_model`` by accident.
_MODEL_HEADER_RE = re.compile(r"^  (\S+):\s*(#.*)?$")
_PARAMS_LINE_RE = re.compile(r"^(\s*params:\s*)\{[^}]*\}(\s*(#.*)?)$")


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _resolve_read(ctx: RunContext, relative: Path) -> Path:
    """Locate an artifact *this run* wrote: the staged copy first, the final location second.

    ``ctx.out()`` must never be used to *find* a file — it registers the path for promotion, so
    using it to read would make ``promote()`` warn about an artifact that was never written. And
    reading the final path directly is worse: under ``staging=True`` it still holds the *previous*
    run's file until ``promote()``, so a summary computed from it would silently describe stale
    predictions (``docs/interfaces.md`` §6 rule 7, §7).
    """
    staged = ctx.staging_dir / relative
    if staged.is_file():
        return staged
    return ctx.base_dir / relative


def _feature_matrix(rows: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Exactly ``model_config.features``, in order — ``stock_code`` is never a feature (§17)."""
    return rows[list(cfg.features)]


def _assert_non_negative_y(model_id: str, cfg: ModelConfig, y: pd.Series) -> None:
    """A Poisson-loss model requires ``y >= 0`` — checked, not merely hoped for."""
    if cfg.models[model_id].loss == "poisson" and (y < 0).any():
        raise ValueError(
            f"{model_id}: target must be non-negative for a Poisson-loss model "
            "(units sold cannot be negative)"
        )


# --------------------------------------------------------------------------
# estimator construction (§17, §19)
# --------------------------------------------------------------------------
def make_model(model_id: str, cfg: ModelConfig, seed: int) -> Any:
    """Build the (un-fit) scikit-learn estimator for ``model_id`` from its config parameters.

    ``M1_linear`` is a ``Pipeline`` of ``StandardScaler`` -> ``LinearRegression`` (all fifteen §17
    features are numeric, so no ``ColumnTransformer`` is needed). The three
    ``HistGradientBoostingRegressor`` variants (M2 Poisson - the primary model, M3 squared-error,
    M4 absolute-error) differ only in ``loss``; ``early_stopping`` is fixed to ``False`` so a fit is
    reproducible from the seed alone. ``seed`` is the caller's ``ctx.seed`` (from
    ``model_config.yaml``, PRD §40) and is used only as ``random_state`` — this function never seeds
    global randomness itself.
    """
    spec = cfg.models[model_id]
    if spec.kind == "linear_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linreg", LinearRegression(**spec.params)),
            ]
        )
    if spec.kind == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            loss=spec.loss,
            random_state=seed,
            early_stopping=False,
            **spec.params,
        )
    raise ValueError(f"{model_id}: unsupported model kind {spec.kind!r} for pipeline.models")


# --------------------------------------------------------------------------
# the one-origin primitive (§16, §21, §22) — shared by tune() and the US-18 back-test
# --------------------------------------------------------------------------
def fit_predict_one_origin(
    features_df: pd.DataFrame, model_id: str, origin: str, cfg: ModelConfig, seed: int
) -> pd.DataFrame:
    """Fit on every target known at ``origin``, predict the single next target month.

    Fits on rows with ``target_month <= origin`` (everything the model could legitimately know at
    that origin) and predicts the rows with ``target_month == origin + 1``. This is the one place
    the leakage boundary is enforced for hyper-parameter selection and the rolling back-test alike.

    Returns ``stock_code, forecast_origin, target_month, model, prediction_raw, prediction`` with
    ``prediction = max(prediction_raw, 0)`` — negatives clipped for business use (only M1 can
    realistically produce one; the GBM Poisson model is non-negative by construction).
    """
    target = str(pd.Period(origin, freq="M") + 1)
    months = features_df["target_month"].astype(str)
    train_rows = features_df.loc[months <= origin]
    predict_rows = features_df.loc[months == target]

    y_train = train_rows["y"].astype(float)
    _assert_non_negative_y(model_id, cfg, y_train)

    model = make_model(model_id, cfg, seed)
    model.fit(_feature_matrix(train_rows, cfg), y_train)

    if len(predict_rows) == 0:
        raw = np.array([], dtype=float)
    else:
        raw = np.asarray(model.predict(_feature_matrix(predict_rows, cfg)), dtype=float)
    prediction = np.clip(raw, 0, None)

    result = predict_rows[["stock_code", "forecast_origin", "target_month"]].copy()
    result["model"] = model_id
    result["prediction_raw"] = raw
    result["prediction"] = prediction
    return result.reset_index(drop=True)[ORIGIN_PREDICTIONS_COLUMNS]


# --------------------------------------------------------------------------
# training-window config override, used only inside the tuning search
# --------------------------------------------------------------------------
def _cfg_with_params(cfg: ModelConfig, model_id: str, params: dict[str, Any]) -> ModelConfig:
    """A copy of ``cfg`` with one model's ``params`` replaced — used only inside the grid search.

    ``model_copy(update=...)`` does not re-run validators and works on a frozen model, so this never
    mutates ``cfg`` itself and never touches disk.
    """
    updated_spec = cfg.models[model_id].model_copy(update={"params": dict(params)})
    updated_models = {**cfg.models, model_id: updated_spec}
    return cfg.model_copy(update={"models": updated_models})


# --------------------------------------------------------------------------
# tune() — small grid, validation origins only, writes params back to YAML (§19, §21)
# --------------------------------------------------------------------------
def tune(
    features_df: pd.DataFrame,
    cfg: ModelConfig,
    ctx: RunContext,
    split: SplitSpec,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Grid-search the three GBM variants on rolling folds strictly inside the training period.

    For every combination of ``model_config.yaml -> tuning.grid`` (learning rate x max leaf nodes x
    max iterations) and every model, replays :func:`fit_predict_one_origin` over
    :func:`pipeline.split.validation_origins` (targets 2011-01...2011-05 — all inside the training
    window) and scores the pooled ``wmape`` (never the hold-out). The best combination per model
    (tie -> smaller ``max_iter``, then smaller ``max_leaf_nodes``) is written back into
    ``config/model_config.yaml`` as a targeted text edit (comments and every other key survive
    byte-for-byte) and the cached config loader is cleared so the new params are visible
    immediately.

    ``config_path`` defaults to the real ``config/model_config.yaml``; tests point it at a
    temporary copy so the repository file is never touched by the test suite.
    """
    target_path = paths.MODEL_CONFIG if config_path is None else config_path
    origins = validation_origins(split)
    grid = cfg.tuning.grid
    gbm_ids = [
        model_id
        for model_id in TRAINABLE_MODEL_IDS
        if cfg.models[model_id].kind == "hist_gradient_boosting"
    ]
    combos = list(
        itertools.product(grid["learning_rate"], grid["max_leaf_nodes"], grid["max_iter"])
    )

    rows: list[dict[str, Any]] = []
    best_params: dict[str, dict[str, Any]] = {}
    best_scores: dict[str, dict[str, Any]] = {}

    for model_id in gbm_ids:
        best_key: tuple[float, Any, Any] | None = None
        best_trial: dict[str, Any] | None = None
        best_row: dict[str, Any] | None = None

        for learning_rate, max_leaf_nodes, max_iter in combos:
            trial_params = {
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "max_iter": max_iter,
            }
            trial_cfg = _cfg_with_params(cfg, model_id, trial_params)

            frames = []
            for origin in origins:
                predicted = fit_predict_one_origin(
                    features_df, model_id, origin, trial_cfg, ctx.seed
                )
                target = str(pd.Period(origin, freq="M") + 1)
                actual = features_df.loc[
                    features_df["target_month"].astype(str) == target,
                    ["stock_code", "target_month", "y"],
                ]
                frames.append(
                    predicted.merge(actual, on=["stock_code", "target_month"], how="left")
                )
            pooled = pd.concat(frames, ignore_index=True)

            trial_wmape = wmape(pooled["y"], pooled["prediction"])
            trial_bias = bias(pooled["y"], pooled["prediction"])
            row = {
                "model": model_id,
                "learning_rate": learning_rate,
                "max_leaf_nodes": max_leaf_nodes,
                "max_iter": max_iter,
                "wmape": trial_wmape,
                "bias": trial_bias,
                "n_rows": int(len(pooled)),
            }
            rows.append(row)

            candidate_key = (trial_wmape, max_iter, max_leaf_nodes)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_trial = trial_params
                best_row = row

        assert best_trial is not None and best_row is not None
        best_params[model_id] = best_trial
        best_scores[model_id] = best_row

    results = pd.DataFrame(rows, columns=TUNING_RESULTS_COLUMNS)
    destination = ctx.out(_repo_relative(paths.EVAL_TABLES_DIR / "tuning_results.csv"))
    results.to_csv(
        destination, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact(
        "tuning_results", _repo_relative(paths.EVAL_TABLES_DIR / "tuning_results.csv")
    )

    _write_tuned_params(target_path, best_params)
    clear_config_cache()

    return {"best_params": best_params, "best_scores": best_scores, "tuning_results": results}


def _format_params_flow(params: dict[str, Any]) -> str:
    """``{learning_rate: 0.05, max_leaf_nodes: 31, max_iter: 300}`` — the file's own flow style."""
    body = ", ".join(f"{key}: {value}" for key, value in params.items())
    return "{" + body + "}"


def _write_tuned_params(config_path: Path, best_params: dict[str, dict[str, Any]]) -> None:
    """Rewrite each tuned model's ``params:`` line in place, leaving everything else untouched.

    A targeted text edit rather than a YAML load-mutate-dump: ``yaml.safe_dump`` would reorder keys
    and destroy every comment in the file, and ``ModelConfig`` is frozen with ``extra="forbid"`` so
    it cannot be safely re-serialised as a partial update either. Each model's params live on one
    flow-style line directly under its ``  <model_id>:`` header; that header is the only line at
    exactly two-space indent with nothing but a name and a colon, so tracking the most recently seen
    header is enough to know which model a ``params:`` line belongs to.
    """
    lines = config_path.read_text(encoding="utf-8").split("\n")
    current_model: str | None = None
    updated: set[str] = set()
    for index, line in enumerate(lines):
        header = _MODEL_HEADER_RE.match(line)
        if header:
            current_model = header.group(1)
            continue
        if current_model in best_params:
            match = _PARAMS_LINE_RE.match(line)
            if match:
                lines[index] = (
                    f"{match.group(1)}{_format_params_flow(best_params[current_model])}"
                    f"{match.group(2)}"
                )
                updated.add(current_model)

    missing = set(best_params) - updated
    if missing:
        raise ValueError(
            f"could not find a params: line for model id(s) {sorted(missing)} in {config_path}"
        )
    config_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# train_models() — the fixed final hold-out model, one per candidate (§21, §43)
# --------------------------------------------------------------------------
def train_models(
    features_df: pd.DataFrame, cfg: ModelConfig, ctx: RunContext, split: SplitSpec
) -> dict[str, Any]:
    """Fit each of the four candidates on the training targets, score them once on the hold-out.

    This is the "final hold-out" reading of §21: one model per candidate, trained through
    2011-05 and never refit per origin, scored against every hold-out target month
    (2011-06...2011-11) at once. The rolling-origin results of §22 are additional and are produced
    by US-18 reusing :func:`fit_predict_one_origin`, not by this function.

    Writes ``artifacts/models/<model_id>.joblib`` for each candidate, one
    ``artifacts/models/candidates_meta.json`` and ``artifacts/forecasts/holdout_predictions.csv``.
    Never writes ``artifacts/models/model.joblib`` — that is the champion, chosen and written by
    US-23. Returns the fitted estimators, keyed by model id.
    """
    train_rows = features_df.loc[train_mask(features_df, split)]
    holdout_rows = features_df.loc[holdout_mask(features_df, split)]
    feature_cols = list(cfg.features)
    sklearn_version = importlib.metadata.version("scikit-learn")

    estimators: dict[str, Any] = {}
    meta_models: list[dict[str, Any]] = []
    holdout_frames: list[pd.DataFrame] = []

    for model_id in TRAINABLE_MODEL_IDS:
        y_train = train_rows["y"].astype(float)
        _assert_non_negative_y(model_id, cfg, y_train)

        model = make_model(model_id, cfg, ctx.seed)
        started = time.perf_counter()
        model.fit(train_rows[feature_cols], y_train)
        fit_seconds = time.perf_counter() - started

        train_raw = np.asarray(model.predict(train_rows[feature_cols]), dtype=float)
        negative_share_train = float((train_raw < 0).mean()) if len(train_raw) else 0.0

        if len(holdout_rows):
            holdout_raw = np.asarray(model.predict(holdout_rows[feature_cols]), dtype=float)
        else:
            holdout_raw = np.array([], dtype=float)
        holdout_pred = np.clip(holdout_raw, 0, None)

        one = holdout_rows[["stock_code", "forecast_origin", "target_month"]].copy()
        one["model"] = model_id
        one["actual"] = holdout_rows["y"].astype(float).to_numpy()
        one["prediction_raw"] = holdout_raw
        one["prediction"] = holdout_pred
        holdout_frames.append(one)

        model_path = ctx.out(_repo_relative(paths.candidate_model(model_id)))
        joblib.dump(model, model_path)
        ctx.record_artifact(model_id, _repo_relative(paths.candidate_model(model_id)))

        estimators[model_id] = model
        spec = cfg.models[model_id]
        meta_models.append(
            {
                "model_id": model_id,
                "kind": spec.kind,
                "loss": spec.loss,
                "params": spec.params,
                "n_train_rows": int(len(train_rows)),
                "fit_seconds": round(fit_seconds, 6),
                "negative_prediction_share_train": round(negative_share_train, 6),
            }
        )

    holdout_predictions = (
        pd.concat(holdout_frames, ignore_index=True)
        .sort_values(["stock_code", "target_month", "model"], kind="mergesort")
        .reset_index(drop=True)[HOLDOUT_PREDICTIONS_COLUMNS]
    )
    holdout_path = ctx.out(_repo_relative(paths.FORECASTS_DIR / "holdout_predictions.csv"))
    holdout_predictions.to_csv(
        holdout_path, index=False, float_format="%.6f", lineterminator="\n", encoding="utf-8"
    )
    ctx.record_artifact(
        "holdout_predictions", _repo_relative(paths.FORECASTS_DIR / "holdout_predictions.csv")
    )

    meta = {
        "seed": ctx.seed,
        "sklearn_version": sklearn_version,
        "features": feature_cols,
        "train_target_start": split.split.train_targets.start,
        "train_target_end": split.split.train_targets.end,
        "holdout_target_start": split.split.holdout_targets.start,
        "holdout_target_end": split.split.holdout_targets.end,
        "models": meta_models,
    }
    meta_path = ctx.out(_repo_relative(paths.MODELS_DIR / "candidates_meta.json"))
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ctx.record_artifact(
        "candidates_meta", _repo_relative(paths.MODELS_DIR / "candidates_meta.json")
    )

    return estimators


# --------------------------------------------------------------------------
# CLI: python -m pipeline.models tune | train
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


def _cli_tune() -> int:
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("tune"):
            features = _read_features()
            cfg = load_model_config()
            spec = SplitSpec.load()
            result = tune(features, cfg, ctx, spec)
            metrics: dict[str, Any] = {}
            for model_id, row in result["best_scores"].items():
                metrics[f"tuning_best_wmape_{model_id}"] = float(row["wmape"])
                metrics[f"tuning_best_bias_{model_id}"] = float(row["bias"])
            ctx.record_metrics(metrics)
        for model_id, params in result["best_params"].items():
            score = result["best_scores"][model_id]
            print(
                f"{model_id}: {params} -> wMAPE={score['wmape']:.4f} Bias={score['bias']:.4f} "
                f"(n={score['n_rows']})"
            )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


def _cli_train() -> int:
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("train_models"):
            features = _read_features()
            cfg = load_model_config()
            spec = SplitSpec.load()
            train_models(features, cfg, ctx, spec)
            holdout = pd.read_csv(
                _resolve_read(ctx, _repo_relative(paths.FORECASTS_DIR / "holdout_predictions.csv"))
            )
            summary = metrics_table(
                holdout, actual_col="actual", forecast_col="prediction", group_cols=["model"]
            )
            metrics: dict[str, Any] = {}
            for record in summary.to_dict(orient="records"):
                metrics[f"holdout_wmape_{record['model']}"] = float(record["wmape"])
                metrics[f"holdout_bias_{record['model']}"] = float(record["bias"])
            ctx.record_metrics(metrics)
        for record in summary.to_dict(orient="records"):
            print(
                f"{record['model']}: wMAPE={record['wmape']:.4f} Bias={record['bias']:.4f} "
                f"(n={int(record['n_rows'])})"
            )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


def run(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``python -m pipeline.models tune|train``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["tune"]:
        return _cli_tune()
    if args == ["train"]:
        return _cli_train()
    print("usage: python -m pipeline.models tune|train", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(run())
