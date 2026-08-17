"""Model candidate tests (US-17, PRD §17, §19, §20, §21, §22, §40, §43, §55).

Uses the real ``model_config.yaml`` dates and grid on purpose for the split/tuning-origin tests —
the temporal design is the project's one fixed thing (§21) — but keeps every features frame tiny
and synthetic so the suite stays fast. The tuning test overrides ``cfg.tuning.grid`` down to one
combination per model so the search itself is exercised without the full grid's runtime.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import paths
from pipeline.config import clear_config_cache, load_model_config
from pipeline.features import FEATURES_COLUMNS
from pipeline.models import (
    HOLDOUT_PREDICTIONS_COLUMNS,
    TRAINABLE_MODEL_IDS,
    fit_predict_one_origin,
    make_model,
    train_models,
    tune,
)
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.split import SplitSpec, validation_origins

CFG = load_model_config()
SPEC = SplitSpec.load()


# --------------------------------------------------------------------------
# synthetic features frames
# --------------------------------------------------------------------------
def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _synthetic_features(stock_codes: list[str], months: list[str]) -> pd.DataFrame:
    """A tiny, fully-populated features frame with plausible non-negative values.

    Every column of ``pipeline.features.FEATURES_COLUMNS`` is present so the frame is a legitimate
    stand-in for ``features.csv`` without needing the real 72k-row file.
    """
    rows = []
    for code_index, code in enumerate(stock_codes):
        for month_index, month in enumerate(months):
            origin = str(pd.Period(month, freq="M") - 1)
            base = float((code_index + 1) * 3 + (month_index % 5))
            period = pd.Period(month, freq="M")
            rows.append(
                {
                    "stock_code": code,
                    "forecast_origin": origin,
                    "target_month": month,
                    "lag_1": base,
                    "lag_2": max(base - 1.0, 0.0),
                    "lag_3": max(base - 2.0, 0.0),
                    "rolling_mean_3": base,
                    "rolling_mean_6": base,
                    "rolling_median_6": base,
                    "rolling_std_3": float(month_index % 3),
                    "rolling_max_6": base + 2.0,
                    "nonzero_months_6": float(min(month_index + 1, 6)),
                    "months_since_last_sale": 1,
                    "product_age_months": month_index + 1,
                    "invoice_count_lag_1": 1,
                    "avg_unit_price_lag_1": 5.0,
                    "target_month_of_year": period.month,
                    "target_quarter": period.quarter,
                    "y": base,
                    "is_active": True,
                }
            )
    frame = pd.DataFrame(rows)
    return frame[FEATURES_COLUMNS]


@pytest.fixture
def ctx(tmp_path):
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


# --------------------------------------------------------------------------
# make_model (§17, §19, §40)
# --------------------------------------------------------------------------
def test_make_model_m1_is_a_scaler_then_linear_regression_pipeline() -> None:
    model = make_model("M1_linear", CFG, seed=123)
    assert isinstance(model, Pipeline)
    names = [name for name, _ in model.steps]
    assert names == ["scaler", "linreg"]
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["linreg"], LinearRegression)


@pytest.mark.parametrize(
    ("model_id", "expected_loss"),
    [
        ("M2_gbm_poisson", "poisson"),
        ("M3_gbm_squared", "squared_error"),
        ("M4_gbm_absolute", "absolute_error"),
    ],
)
def test_make_model_gbm_variants_use_the_right_loss_and_seed(
    model_id: str, expected_loss: str
) -> None:
    model = make_model(model_id, CFG, seed=777)
    assert isinstance(model, HistGradientBoostingRegressor)
    assert model.loss == expected_loss
    assert model.random_state == 777
    assert model.early_stopping is False


def test_trainable_model_ids_excludes_baselines() -> None:
    assert set(TRAINABLE_MODEL_IDS) == {
        "M1_linear",
        "M2_gbm_poisson",
        "M3_gbm_squared",
        "M4_gbm_absolute",
    }


# --------------------------------------------------------------------------
# fit_predict_one_origin — the leakage boundary (§16, §21, §22)
# --------------------------------------------------------------------------
def test_fit_predict_one_origin_never_fits_on_rows_after_the_origin(monkeypatch) -> None:
    features = _synthetic_features(["P1", "P2"], _months("2010-06", "2011-02"))
    origin = "2010-12"

    captured: dict[str, pd.DataFrame] = {}
    original_fit = HistGradientBoostingRegressor.fit

    def spy_fit(self, X, y=None, **kwargs):
        captured["X"] = X
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy_fit)

    fit_predict_one_origin(features, "M2_gbm_poisson", origin, CFG, seed=1)

    assert "X" in captured
    used_index = captured["X"].index
    max_target_used = features.loc[used_index, "target_month"].max()
    assert max_target_used == origin


def test_fit_predict_one_origin_predicts_only_the_next_month() -> None:
    features = _synthetic_features(["P1", "P2"], _months("2010-06", "2011-02"))
    origin = "2010-12"
    target = "2011-01"

    result = fit_predict_one_origin(features, "M2_gbm_poisson", origin, CFG, seed=1)

    assert set(result["target_month"]) == {target}
    assert len(result) == (features["target_month"] == target).sum()
    assert (result["prediction"] >= 0).all()


def test_fit_predict_one_origin_m1_raw_may_be_negative_but_prediction_is_clipped() -> None:
    # A clean downward trend, with the target left un-clipped, lets a linear model reproduce it
    # almost exactly (every feature is proportional to the same trend value) and so extrapolate
    # below zero at the final step, exactly like a real over-shooting linear forecast would.
    months = _months("2010-01", "2011-06")
    rows = []
    for month_index, month in enumerate(months):
        origin = str(pd.Period(month, freq="M") - 1)
        trend = 50.0 - month_index * 4.0  # goes negative well before the series ends
        period = pd.Period(month, freq="M")
        rows.append(
            {
                "stock_code": "P1",
                "forecast_origin": origin,
                "target_month": month,
                "lag_1": trend,
                "lag_2": trend,
                "lag_3": trend,
                "rolling_mean_3": trend,
                "rolling_mean_6": trend,
                "rolling_median_6": trend,
                "rolling_std_3": 0.0,
                "rolling_max_6": trend,
                "nonzero_months_6": 6.0,
                "months_since_last_sale": 1,
                "product_age_months": month_index + 1,
                "invoice_count_lag_1": 1,
                "avg_unit_price_lag_1": 5.0,
                "target_month_of_year": period.month,
                "target_quarter": period.quarter,
                "y": trend,  # left un-clipped on purpose, see comment above
                "is_active": True,
            }
        )
    features = pd.DataFrame(rows)[FEATURES_COLUMNS]
    origin = months[-2]

    result = fit_predict_one_origin(features, "M1_linear", origin, CFG, seed=1)

    assert (result["prediction"] >= 0).all()
    assert (result["prediction_raw"] < 0).any()
    assert (result.loc[result["prediction_raw"] < 0, "prediction"] == 0).all()


def test_fit_predict_one_origin_column_order() -> None:
    features = _synthetic_features(["P1"], _months("2010-06", "2011-02"))
    result = fit_predict_one_origin(features, "M2_gbm_poisson", "2010-12", CFG, seed=1)
    assert list(result.columns) == [
        "stock_code",
        "forecast_origin",
        "target_month",
        "model",
        "prediction_raw",
        "prediction",
    ]


def test_poisson_model_rejects_a_negative_target(monkeypatch) -> None:
    features = _synthetic_features(["P1"], _months("2010-06", "2011-02"))
    features = features.copy()
    features.loc[features.index[0], "y"] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        fit_predict_one_origin(features, "M2_gbm_poisson", "2010-12", CFG, seed=1)


# --------------------------------------------------------------------------
# tune() — validation origins only, never the hold-out (§21, §22)
# --------------------------------------------------------------------------
@pytest.fixture
def tiny_grid_cfg():
    """The real config with the tuning grid collapsed to one combination per GBM model."""
    tiny_grid = {"learning_rate": [0.1], "max_leaf_nodes": [15], "max_iter": [30]}
    tuning = CFG.tuning.model_copy(update={"grid": tiny_grid})
    return CFG.model_copy(update={"tuning": tuning})


def test_tune_only_evaluates_validation_origins(monkeypatch, tmp_path, ctx, tiny_grid_cfg) -> None:
    features = _synthetic_features(["P1", "P2"], _months("2010-01", "2011-11"))

    seen_origins: list[str] = []
    import pipeline.models as models_module

    original = models_module.fit_predict_one_origin

    def spy(features_df, model_id, origin, cfg, seed):
        seen_origins.append(origin)
        return original(features_df, model_id, origin, cfg, seed)

    monkeypatch.setattr(models_module, "fit_predict_one_origin", spy)

    config_copy = tmp_path / "model_config.yaml"
    config_copy.write_text(paths.MODEL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with ctx.step("tune"):
        result = models_module.tune(
            features, tiny_grid_cfg, ctx, SPEC, config_path=config_copy
        )

    expected_origins = set(validation_origins(SPEC))
    assert set(seen_origins) == expected_origins
    for origin in seen_origins:
        target = str(pd.Period(origin, freq="M") + 1)
        assert target <= SPEC.split.train_targets.end

    assert set(result["best_params"]) == {"M2_gbm_poisson", "M3_gbm_squared", "M4_gbm_absolute"}


def test_tuning_results_written_and_has_no_month_past_train_end(
    ctx, tiny_grid_cfg, tmp_path
) -> None:
    features = _synthetic_features(["P1", "P2"], _months("2010-01", "2011-11"))
    config_copy = tmp_path / "model_config.yaml"
    config_copy.write_text(paths.MODEL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with ctx.step("tune"):
        tune(features, tiny_grid_cfg, ctx, SPEC, config_path=config_copy)

    table_path = ctx.base_dir / "artifacts" / "reports" / "evaluation_tables" / "tuning_results.csv"
    assert table_path.is_file()
    table = pd.read_csv(table_path)
    assert list(table.columns) == [
        "model",
        "learning_rate",
        "max_leaf_nodes",
        "max_iter",
        "wmape",
        "bias",
        "n_rows",
    ]
    assert set(table["model"]) == {"M2_gbm_poisson", "M3_gbm_squared", "M4_gbm_absolute"}


def test_tune_writes_params_back_and_reloadable_by_load_model_config(
    ctx, tiny_grid_cfg, tmp_path, monkeypatch
) -> None:
    features = _synthetic_features(["P1", "P2"], _months("2010-01", "2011-11"))
    config_copy = tmp_path / "model_config.yaml"
    config_copy.write_text(paths.MODEL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")

    with ctx.step("tune"):
        result = tune(features, tiny_grid_cfg, ctx, SPEC, config_path=config_copy)

    # Comments must survive the targeted text edit.
    new_text = config_copy.read_text(encoding="utf-8")
    assert "# Modelling configuration" in new_text
    assert "§40" in new_text or "40" in new_text

    monkeypatch.setattr(paths, "MODEL_CONFIG", config_copy)
    clear_config_cache()
    try:
        reloaded = load_model_config()
        for model_id, expected_params in result["best_params"].items():
            assert reloaded.models[model_id].params == expected_params
    finally:
        clear_config_cache()  # restore the real config for every test after this one


# --------------------------------------------------------------------------
# train_models() — the fixed final hold-out model (§21, §43)
# --------------------------------------------------------------------------
def _full_range_features() -> pd.DataFrame:
    months = _months(CFG.split.train_targets.start, CFG.split.holdout_targets.end)
    return _synthetic_features(["P1", "P2", "P3"], months)


def test_train_models_writes_four_joblib_files_and_never_the_champion(tmp_path) -> None:
    features = _full_range_features()
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        with ctx.step("train_models"):
            train_models(features, CFG, ctx, SPEC)
        for model_id in TRAINABLE_MODEL_IDS:
            assert (tmp_path / "artifacts" / "models" / f"{model_id}.joblib").is_file()
        assert not (tmp_path / "artifacts" / "models" / "model.joblib").exists()
    finally:
        close_log_handlers(ctx.run_id)


def test_holdout_predictions_schema_and_coverage(tmp_path) -> None:
    features = _full_range_features()
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        with ctx.step("train_models"):
            train_models(features, CFG, ctx, SPEC)
        holdout_path = tmp_path / "artifacts" / "forecasts" / "holdout_predictions.csv"
        table = pd.read_csv(holdout_path)
        assert list(table.columns) == HOLDOUT_PREDICTIONS_COLUMNS
        assert set(table["target_month"]) == set(
            _months(CFG.split.holdout_targets.start, CFG.split.holdout_targets.end)
        )
        assert set(table["model"]) == set(TRAINABLE_MODEL_IDS)
        assert (table["prediction"] >= 0).all()
    finally:
        close_log_handlers(ctx.run_id)


def test_candidates_meta_records_seed_features_and_sklearn_version(tmp_path) -> None:
    features = _full_range_features()
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        with ctx.step("train_models"):
            train_models(features, CFG, ctx, SPEC)
        meta_path = tmp_path / "artifacts" / "models" / "candidates_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["seed"] == load_model_config().seed
        assert meta["features"] == list(CFG.features)
        assert meta["sklearn_version"]
        assert {m["model_id"] for m in meta["models"]} == set(TRAINABLE_MODEL_IDS)
        m1 = next(m for m in meta["models"] if m["model_id"] == "M1_linear")
        assert "negative_prediction_share_train" in m1
    finally:
        close_log_handlers(ctx.run_id)


def test_train_models_is_deterministic_across_two_runs(tmp_path) -> None:
    features = _full_range_features()

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    ctx_a = RunContext.start(mode="no-llm", base_dir=dir_a)
    ctx_b = RunContext.start(mode="no-llm", base_dir=dir_b)
    try:
        with ctx_a.step("train_models"):
            train_models(features, CFG, ctx_a, SPEC)
        with ctx_b.step("train_models"):
            train_models(features, CFG, ctx_b, SPEC)

        text_a = (dir_a / "artifacts" / "forecasts" / "holdout_predictions.csv").read_bytes()
        text_b = (dir_b / "artifacts" / "forecasts" / "holdout_predictions.csv").read_bytes()
        assert text_a == text_b
    finally:
        close_log_handlers(ctx_a.run_id)
        close_log_handlers(ctx_b.run_id)


# --------------------------------------------------------------------------
# guard: forbidden literals never appear under src/ (CLAUDE.md §2 rule 2, §20)
# --------------------------------------------------------------------------
def test_forbidden_literals_never_appear_under_src() -> None:
    src = paths.PROJECT_ROOT / "src"
    forbidden = ("bias_correction", "shuffle=True", "train_test_split")
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append((str(path.relative_to(paths.PROJECT_ROOT)), token))
    assert not offenders, f"forbidden literal(s) found: {offenders}"
