"""Rolling-origin back-test engine tests (US-18, PRD §18, §21, §22, §26, §55).

Small synthetic features/panel frames exercise the engine cheaply against the *real*
``model_config.yaml`` split (the temporal design is the project's one fixed thing, §21, so a test
frame should not be free to reinvent it) — one module-scoped run of
:func:`pipeline.backtest.backtest` is shared by every structural assertion so the eighteen-origin
loop only pays its cost once.

The final test in this file is the deliberate cross-check tying US-17 and US-18 together: the
back-test's origin ``2011-05`` fits on exactly the same rows as US-17's ``train_models``
(everything with ``target_month <= 2011-05``), so its predictions for target ``2011-06`` must
equal the committed ``holdout_predictions.csv``, model by model. It reads the real
``features.csv`` and ``holdout_predictions.csv`` and is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from pipeline import paths
from pipeline.backtest import (
    BACKTEST_BY_ORIGIN_COLUMNS,
    BACKTEST_COLUMNS,
    backtest,
    backtest_summary,
)
from pipeline.config import MODEL_IDS, load_model_config
from pipeline.features import FEATURES_COLUMNS
from pipeline.models import TRAINABLE_MODEL_IDS, fit_predict_one_origin
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.split import SplitSpec, rolling_origins

CFG = load_model_config()
SPEC = SplitSpec.load()


# --------------------------------------------------------------------------
# synthetic features/panel frames (mirrors tests/test_models.py's builder)
# --------------------------------------------------------------------------
def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _synthetic_features(stock_codes: list[str], months: list[str]) -> pd.DataFrame:
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


def _synthetic_panel(stock_codes: list[str], months: list[str]) -> pd.DataFrame:
    """``units_sold`` matching the ``base`` used for ``lag_1`` above, for a B3 lookup."""
    rows = []
    for code_index, code in enumerate(stock_codes):
        for month_index, month in enumerate(months):
            base = float((code_index + 1) * 3 + (month_index % 5))
            rows.append({"stock_code": code, "month": month, "units_sold": base})
    return pd.DataFrame(rows)


_STOCK_CODES = ["P1", "P2"]
_MONTHS = _months("2010-01", "2011-11")
_FEATURES = _synthetic_features(_STOCK_CODES, _MONTHS)
_PANEL = _synthetic_panel(_STOCK_CODES, _MONTHS)


@pytest.fixture(scope="module")
def backtest_result(tmp_path_factory) -> pd.DataFrame:
    """Run the full seven-candidate back-test once and share it across the tests below."""
    base_dir = tmp_path_factory.mktemp("backtest_engine")
    ctx = RunContext.start(mode="no-llm", base_dir=base_dir)
    try:
        with_all = backtest(_FEATURES, _PANEL, CFG, ctx, SPEC)
    finally:
        close_log_handlers(ctx.run_id)
    return with_all


# --------------------------------------------------------------------------
# structure (§22)
# --------------------------------------------------------------------------
def test_backtest_column_order(backtest_result: pd.DataFrame) -> None:
    assert list(backtest_result.columns) == BACKTEST_COLUMNS


def test_backtest_covers_all_seven_candidate_ids(backtest_result: pd.DataFrame) -> None:
    assert set(backtest_result["model"]) == set(MODEL_IDS)


def test_backtest_origin_count_matches_config(backtest_result: pd.DataFrame) -> None:
    origins = rolling_origins(SPEC)
    assert len(origins) == 18
    assert set(backtest_result["forecast_origin"]) == set(origins)


def test_backtest_forecast_origin_is_exactly_one_month_before_target(
    backtest_result: pd.DataFrame,
) -> None:
    origins = pd.PeriodIndex(backtest_result["forecast_origin"], freq="M")
    expected_target = (origins + 1).astype(str)
    assert (expected_target == backtest_result["target_month"].to_numpy()).all()


def test_backtest_never_scores_december_2011(backtest_result: pd.DataFrame) -> None:
    assert "2011-12" not in set(backtest_result["target_month"])
    first_target = str(pd.Period(SPEC.backtest.first_origin, freq="M") + 1)
    last_target = str(pd.Period(SPEC.backtest.last_origin, freq="M") + 1)
    assert backtest_result["target_month"].between(first_target, last_target).all()


def test_backtest_every_model_shares_the_identical_row_set(backtest_result: pd.DataFrame) -> None:
    row_sets = {
        model_id: frozenset(zip(group["stock_code"], group["target_month"], strict=True))
        for model_id, group in backtest_result.groupby("model")
    }
    reference = next(iter(row_sets.values()))
    for model_id, rows in row_sets.items():
        assert rows == reference, f"{model_id} does not cover the same rows as the rest"


# --------------------------------------------------------------------------
# residual and raw/clipped prediction conventions (§26)
# --------------------------------------------------------------------------
def test_residual_equals_actual_minus_prediction(backtest_result: pd.DataFrame) -> None:
    recomputed = backtest_result["actual"] - backtest_result["prediction"]
    pd.testing.assert_series_equal(
        backtest_result["residual"], recomputed, check_names=False, atol=1e-9
    )


def test_baseline_prediction_raw_equals_prediction(backtest_result: pd.DataFrame) -> None:
    from pipeline.baselines import BASELINE_MODEL_IDS

    baseline_rows = backtest_result.loc[backtest_result["model"].isin(BASELINE_MODEL_IDS)]
    non_nan = baseline_rows.dropna(subset=["prediction"])
    assert (non_nan["prediction_raw"] == non_nan["prediction"]).all()


def test_prediction_is_always_populated_for_every_trainable_model(
    backtest_result: pd.DataFrame,
) -> None:
    trainable_rows = backtest_result.loc[backtest_result["model"].isin(TRAINABLE_MODEL_IDS)]
    assert trainable_rows["prediction_raw"].notna().all()
    assert trainable_rows["prediction"].notna().all()
    assert (trainable_rows["prediction"] >= 0).all()


# --------------------------------------------------------------------------
# baseline definitions, replayed inside the engine (§18.1, §19)
# --------------------------------------------------------------------------
def test_b2_predictions_equal_rolling_mean_3(backtest_result: pd.DataFrame) -> None:
    b2 = backtest_result.loc[backtest_result["model"] == "B2_ma3"]
    merged = b2.merge(
        _FEATURES[["stock_code", "target_month", "rolling_mean_3"]],
        on=["stock_code", "target_month"],
        how="left",
    )
    assert np.allclose(merged["prediction"], merged["rolling_mean_3"])


def test_b3_predictions_equal_panel_twelve_months_back_or_nan(
    backtest_result: pd.DataFrame,
) -> None:
    b3 = backtest_result.loc[backtest_result["model"] == "B3_seasonal_naive"]
    lookup = _PANEL.set_index(["stock_code", "month"])["units_sold"]

    for row in b3.itertuples():
        seasonal_month = str(pd.Period(row.target_month, freq="M") - 12)
        expected = lookup.get((row.stock_code, seasonal_month), np.nan)
        if np.isnan(expected):
            assert np.isnan(row.prediction)
        else:
            assert row.prediction == pytest.approx(expected)


# --------------------------------------------------------------------------
# unknown model ids (§2 assertion contract)
# --------------------------------------------------------------------------
def test_unknown_model_id_raises(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        with pytest.raises(ValueError, match="unknown model id"):
            backtest(_FEATURES, _PANEL, CFG, ctx, SPEC, models=["not_a_model"])
    finally:
        close_log_handlers(ctx.run_id)


def test_models_filter_restricts_the_output(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        result = backtest(_FEATURES, _PANEL, CFG, ctx, SPEC, models=["B1_last_month", "M1_linear"])
    finally:
        close_log_handlers(ctx.run_id)
    assert set(result["model"]) == {"B1_last_month", "M1_linear"}


# --------------------------------------------------------------------------
# leakage guard, re-verified at engine level with a spy (§2, §16, §21)
# --------------------------------------------------------------------------
def test_engine_never_fits_on_rows_after_the_origin(monkeypatch, tmp_path) -> None:
    """The engine calls the shared primitive with the right origin at every step, and the
    primitive's fit never sees a row whose target is after that origin — checked here by spying
    on both the primitive (to know which origin was active) and the estimator's ``fit`` (to see
    exactly which rows it trained on).
    """
    import pipeline.backtest as backtest_module

    origins_seen: list[str] = []
    fit_indexes: list[pd.Index] = []

    original_fit = HistGradientBoostingRegressor.fit

    def spy_fit(self, X, y=None, **kwargs):
        fit_indexes.append(X.index)
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy_fit)

    original_primitive = backtest_module.fit_predict_one_origin

    def spy_primitive(features_df, model_id, origin, cfg, seed):
        origins_seen.append(origin)
        return original_primitive(features_df, model_id, origin, cfg, seed)

    monkeypatch.setattr(backtest_module, "fit_predict_one_origin", spy_primitive)

    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        backtest_module.backtest(_FEATURES, _PANEL, CFG, ctx, SPEC, models=["M2_gbm_poisson"])
    finally:
        close_log_handlers(ctx.run_id)

    assert origins_seen == rolling_origins(SPEC)
    assert len(fit_indexes) == len(origins_seen)
    for origin, index in zip(origins_seen, fit_indexes, strict=True):
        max_target_used = _FEATURES.loc[index, "target_month"].max()
        assert max_target_used <= origin


# --------------------------------------------------------------------------
# backtest_summary — pure, no ctx, no disk (§22, §23)
# --------------------------------------------------------------------------
def test_backtest_summary_columns_and_no_nan_metrics(backtest_result: pd.DataFrame) -> None:
    summary = backtest_summary(backtest_result)
    assert list(summary.columns) == BACKTEST_BY_ORIGIN_COLUMNS
    assert summary[["wmape", "bias"]].notna().all().all()


def test_backtest_summary_has_one_row_per_origin_for_non_b3_models(
    backtest_result: pd.DataFrame,
) -> None:
    summary = backtest_summary(backtest_result)
    origins = rolling_origins(SPEC)
    for model_id in set(MODEL_IDS) - {"B3_seasonal_naive"}:
        rows = summary.loc[summary["model"] == model_id]
        assert len(rows) == len(origins)


def test_backtest_summary_b3_rows_are_a_subset_bounded_by_origin_count(
    backtest_result: pd.DataFrame,
) -> None:
    summary = backtest_summary(backtest_result)
    b3_rows = summary.loc[summary["model"] == "B3_seasonal_naive"]
    assert len(b3_rows) <= len(rolling_origins(SPEC))


# --------------------------------------------------------------------------
# the US-17 / US-18 cross-check: origin 2011-05 == train_models' fit (§21, §22)
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_origin_2011_05_matches_committed_holdout_predictions_for_target_2011_06() -> None:
    holdout_path = paths.FORECASTS_DIR / "holdout_predictions.csv"
    if not paths.FEATURES.is_file() or not holdout_path.is_file():
        pytest.skip("real features.csv / holdout_predictions.csv are not present")

    features = pd.read_csv(
        paths.FEATURES,
        dtype={"stock_code": "string", "forecast_origin": "string", "target_month": "string"},
    )
    holdout = pd.read_csv(
        holdout_path,
        dtype={"stock_code": "string", "forecast_origin": "string", "target_month": "string"},
    )

    origin = "2011-05"
    target = "2011-06"
    assert origin == CFG.split.train_targets.end  # the exact boundary the cross-check relies on

    holdout_target = holdout.loc[holdout["target_month"] == target]

    for model_id in TRAINABLE_MODEL_IDS:
        engine = fit_predict_one_origin(features, model_id, origin, CFG, CFG.seed)
        expected = (
            holdout_target.loc[holdout_target["model"] == model_id]
            .set_index("stock_code")["prediction"]
            .sort_index()
        )
        actual = engine.set_index("stock_code")["prediction"].sort_index()
        pd.testing.assert_series_equal(
            actual, expected, check_names=False, check_exact=False, atol=1e-6, rtol=0
        )


# --------------------------------------------------------------------------
# guard: forbidden literals never appear under src/ (CLAUDE.md §2 rule 2, §20)
# --------------------------------------------------------------------------
def test_forbidden_literals_never_appear_in_backtest_module() -> None:
    text = (paths.PROJECT_ROOT / "src" / "pipeline" / "backtest.py").read_text(encoding="utf-8")
    for token in ("bias_correction", "shuffle=True", "train_test_split"):
        assert token not in text
