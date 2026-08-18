"""Latest operational forecast & inventory plan tests (US-23, PRD §4, §7, §14-§16, §24-§28, §55).

Everything runs on a small synthetic panel rather than the real 100k-row one, so each test isolates
exactly one rule — the origin boundary, the partial-month guard, the §28 formula, the cold-start
statuses, the naming rule — and the file stays fast enough to run on every commit. The panel is
built to the US-05 shape (each product starts at its first sale and is contiguous to the panel end)
because :func:`pipeline.features.build_features_for_origin` and
:func:`pipeline.active.active_mask` both rely on that guarantee.

Real configuration is used throughout (``load_model_config()`` / ``load_inventory_policy()`` /
``load_cleaning_config()``), never a hand-rolled threshold, so a config change is felt here rather
than silently diverging. The one deliberate exception is the panel's own months: the synthetic panel
covers 2010-12 … 2011-12 rather than the full 2009-12 … 2011-12, which keeps the fixture small while
the *origin* (2011-11) and the partial month (2011-12) stay exactly the configured ones.
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.baselines import B2, B3
from pipeline.config import (
    load_cleaning_config,
    load_inventory_policy,
    load_model_config,
)
from pipeline.features import build_features
from pipeline.inventory import target_inventory
from pipeline.latest_forecast import (
    INVENTORY_OUTPUT_NAME,
    INVENTORY_PLAN_COLUMNS,
    LATEST_FORECAST_COLUMNS,
    STATUS_FORECAST,
    STATUS_NEW_PRODUCT,
    STEP_NAME,
    BaselineForecaster,
    build_inventory_plan,
    build_latest_forecast,
    champion_id,
    inactive_status,
    operational_features,
    operational_origin,
    operational_sigma,
    refit_champion,
    resolve_champion,
    sanity_report,
    validate_operational_inputs,
)
from pipeline.panel import PANEL_COLUMNS
from pipeline.run_context import RunContext, close_log_handlers

CFG = load_model_config()
CLEANING = load_cleaning_config()
POLICY = load_inventory_policy()
K = CFG.active_rule.k
ORIGIN = CLEANING.raw.last_full_month          # 2011-11
TARGET = str(pd.Period(ORIGIN, freq="M") + 1)  # 2011-12

#: Every month the synthetic panel covers, first to last.
MONTHS = [str(period) for period in pd.period_range("2010-12", TARGET, freq="M")]

#: The four products, each exercising one branch of the §14 / §15 status logic.
ACTIVE_A = "ACT-A"
ACTIVE_B = "ACT-B"
STOPPED = "STOPPED"
NEWBIE = "NEWBIE"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _panel_row(stock_code: str, month: str, units: int) -> dict:
    """One US-05 panel row. Only the columns the feature build reads carry real numbers."""
    return {
        "month": month,
        "stock_code": stock_code,
        "description": f"{stock_code} DESCRIPTION",
        "units_sold": units,
        "gross_revenue": float(units) * 2.0,
        "avg_unit_price": 2.0,
        "invoice_count": 1 if units else 0,
        "sale_line_count": 1 if units else 0,
        "customer_count": 1 if units else 0,
        "max_line_qty": units,
        "returned_units": 0,
        "is_partial_month": month in CLEANING.raw.partial_months,
    }


@pytest.fixture
def panel() -> pd.DataFrame:
    """A four-product panel: two products still selling, one that stopped, one born in the
    partial month.

    ``ACT-A`` and ``ACT-B`` sell every month, so both are active at the target. ``STOPPED`` sells
    through 2011-04 and then reports zero months, which puts its last sale outside the ``k``-month
    window and makes it inactive. ``NEWBIE`` has a single row, in the partial month itself — the
    §15 cold-start case: no observed history at or before the forecast origin.

    The partial month carries deliberately extreme sales (10,000 units) for the active products: if
    any feature read month ``t``, the operational numbers would be visibly wrong rather than subtly
    so.
    """
    rows: list[dict] = []
    for index, month in enumerate(MONTHS):
        partial = month in CLEANING.raw.partial_months
        rows.append(_panel_row(ACTIVE_A, month, 10_000 if partial else 100 + index))
        rows.append(_panel_row(ACTIVE_B, month, 10_000 if partial else 40))
        if month <= "2011-04":
            rows.append(_panel_row(STOPPED, month, 70))
        else:
            rows.append(_panel_row(STOPPED, month, 0))
    rows.append(_panel_row(NEWBIE, TARGET, 500))

    frame = pd.DataFrame(rows)[PANEL_COLUMNS]
    return frame.sort_values(["stock_code", "month"], kind="mergesort").reset_index(drop=True)


@pytest.fixture
def train_features(panel: pd.DataFrame) -> pd.DataFrame:
    """``features.csv``-shaped training rows: every fully observed target through the origin."""
    return build_features(panel, K, "2011-03", ORIGIN, CFG)


@pytest.fixture
def abc_train() -> pd.DataFrame:
    """A minimal ``abc_train.csv``: only the two columns this module reads."""
    return pd.DataFrame(
        {
            "stock_code": [ACTIVE_A, ACTIVE_B, STOPPED, NEWBIE],
            "abc_class": ["A", "B", "C", "C"],
        }
    )


@pytest.fixture
def backtest() -> pd.DataFrame:
    """Back-test residuals for the champion, ending at target 2011-11 as the real one does.

    ``ACT-A`` gets more than ``min_residuals_product`` residuals so it earns a product-level σ;
    ``ACT-B`` gets two, so it falls back a level (§27).
    """
    months = [str(period) for period in pd.period_range("2011-01", ORIGIN, freq="M")]
    rows = [
        {"stock_code": ACTIVE_A, "target_month": month, "model": B2, "residual": value}
        for month, value in zip(months, [10.0, -8.0, 12.0, -6.0, 9.0, -11.0, 7.0, -5.0, 13.0,
                                         -9.0, 6.0], strict=True)
    ]
    rows += [
        {"stock_code": ACTIVE_B, "target_month": month, "model": B2, "residual": 4.0}
        for month in months[-2:]
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def ctx(tmp_path):
    """A run whose artifacts land in ``tmp_path`` instead of the repository."""
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


def _plan(ctx, panel, train_features, abc_train, backtest, champion=B2):
    """Run the whole US-23 chain on the fixtures and return everything it produced."""
    features = operational_features(panel, CFG, ORIGIN)
    model = refit_champion(train_features, champion, CFG, ctx, origin=ORIGIN)
    latest = build_latest_forecast(
        panel,
        model,
        CFG,
        ctx,
        champion=champion,
        abc_train_df=abc_train,
        origin=ORIGIN,
        features_df=features,
    )
    sigma_df = operational_sigma(backtest, abc_train, latest, champion, POLICY)
    plan = build_inventory_plan(
        latest,
        sigma_df,
        POLICY,
        ctx,
        panel_df=panel,
        abc_train_df=abc_train,
        features_df=features,
        cfg=CFG,
        origin=ORIGIN,
    )
    return {
        "features": features,
        "model": model,
        "latest": latest,
        "sigma": sigma_df,
        "plan": plan,
    }


# --------------------------------------------------------------------------
# the origin boundary (§16)
# --------------------------------------------------------------------------
def test_origin_is_the_last_full_month(panel):
    """The operational origin is configuration, never a literal — and the target is origin + 1."""
    assert operational_origin(CLEANING) == ORIGIN
    frame = operational_features(panel, CFG, ORIGIN)
    assert set(frame["forecast_origin"]) == {ORIGIN}
    assert set(frame["target_month"]) == {TARGET}


def test_operational_features_cover_only_active_products(panel):
    """§14: a product with no sale in the previous ``k`` months gets no operational feature row."""
    frame = operational_features(panel, CFG, ORIGIN)
    assert sorted(frame["stock_code"]) == [ACTIVE_A, ACTIVE_B]
    assert frame["is_active"].all()


def test_partial_month_never_reaches_a_feature(panel, train_features):
    """The §16 guard passes on a correct build — corrupting December 2011 changes nothing.

    The active products sell 10,000 units in the partial month; every feature is built from the
    series shifted one month back, so none of them can see it.
    """
    frame = operational_features(panel, CFG, ORIGIN)
    result = validate_operational_inputs(frame, train_features, panel, ORIGIN, CFG, CLEANING)
    assert result.passed, result.summary()
    assert result.step == STEP_NAME
    assert result.extra["target_month"] == TARGET
    assert result.extra["partial_months"] == [TARGET]
    # lag_1 is last month's units, not this month's 10,000.
    assert frame.loc[frame["stock_code"] == ACTIVE_B, "lag_1"].tolist() == [40]


def test_guard_detects_a_feature_that_moved_with_the_partial_month(panel, train_features):
    """A feature value that does not survive the rebuild is reported, not silently accepted."""
    frame = operational_features(panel, CFG, ORIGIN)
    contaminated = frame.copy()
    contaminated.loc[contaminated.index[0], "lag_1"] += 1

    result = validate_operational_inputs(
        contaminated, train_features, panel, ORIGIN, CFG, CLEANING
    )
    assert not result.passed
    rules = {violation.rule for violation in result.violations}
    assert rules == {"partial_month_not_used"}
    assert "lag_1" in result.violations[0].examples


def test_guard_rejects_a_refit_row_after_the_origin(panel, train_features):
    """§16: the champion may only be refit on targets that were known at the forecast origin."""
    frame = operational_features(panel, CFG, ORIGIN)
    leaking = pd.concat([train_features, train_features.head(1).assign(target_month=TARGET)])

    result = validate_operational_inputs(frame, leaking, panel, ORIGIN, CFG, CLEANING)
    assert not result.passed
    assert {violation.rule for violation in result.violations} == {"refit_window"}
    assert result.violations[0].examples == [TARGET]


def test_guard_rejects_a_wrong_target_month(panel, train_features):
    """A frame built for the wrong month is a stop condition, not something to forecast from."""
    frame = operational_features(panel, CFG, ORIGIN)
    wrong = frame.assign(target_month="2012-01")

    result = validate_operational_inputs(wrong, train_features, panel, ORIGIN, CFG, CLEANING)
    assert not result.passed
    assert "target_month" in {violation.rule for violation in result.violations}


# --------------------------------------------------------------------------
# refit & model.joblib (§41)
# --------------------------------------------------------------------------
def test_refit_writes_a_loadable_model_and_meta(ctx, panel, train_features):
    """``model.joblib`` loads and predicts, and its metadata stops at the forecast origin."""
    model = refit_champion(train_features, CFG.primary_model_id, CFG, ctx, origin=ORIGIN)

    model_path = ctx.base_dir / paths.MODEL.relative_to(paths.PROJECT_ROOT)
    meta_path = ctx.base_dir / paths.MODEL_META.relative_to(paths.PROJECT_ROOT)
    assert model_path.stat().st_size > 0

    loaded = joblib.load(model_path)
    predictions = loaded.predict(train_features[list(CFG.features)])
    assert len(predictions) == len(train_features)
    np.testing.assert_allclose(predictions, model.predict(train_features[list(CFG.features)]))

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["champion"] == CFG.primary_model_id
    assert meta["train_targets"]["end"] == ORIGIN
    assert meta["train_targets"]["start"] == train_features["target_month"].min()
    assert meta["n_rows"] == len(train_features)
    assert meta["features"] == list(CFG.features)
    assert meta["seed"] == ctx.seed
    assert meta["run_id"] == ctx.run_id
    assert set(meta["holdout_metrics_reference"]) == {"wmape", "bias"}


def test_refit_never_sees_the_partial_month(ctx, panel, train_features):
    """The rows handed to the refit stop at the origin — December 2011 is not among them."""
    assert TARGET not in set(train_features["target_month"])
    refit_champion(train_features, CFG.primary_model_id, CFG, ctx, origin=ORIGIN)

    meta_path = ctx.base_dir / paths.MODEL_META.relative_to(paths.PROJECT_ROOT)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["train_targets"]["end"] <= ORIGIN
    assert meta["target_month"] == TARGET


def test_a_baseline_champion_is_still_a_loadable_model(ctx, train_features):
    """A baseline winning the gates is a legitimate outcome, so ``model.joblib`` must exist
    for it too."""
    model = refit_champion(train_features, B2, CFG, ctx, origin=ORIGIN)
    assert isinstance(model, BaselineForecaster)

    model_path = ctx.base_dir / paths.MODEL.relative_to(paths.PROJECT_ROOT)
    loaded = joblib.load(model_path)
    np.testing.assert_allclose(
        loaded.predict(train_features[list(CFG.features)]),
        train_features["rolling_mean_3"].to_numpy(dtype=float),
    )

    meta_path = ctx.base_dir / paths.MODEL_META.relative_to(paths.PROJECT_ROOT)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "baseline"
    assert meta["n_rows"] == 0


def test_baseline_forecaster_refuses_the_reference_only_baseline():
    """B3 reads the panel's month t-12, not a feature column, and is reference only (§19)."""
    with pytest.raises(ValueError, match="reference only"):
        BaselineForecaster(B3)


# --------------------------------------------------------------------------
# latest_forecast.csv
# --------------------------------------------------------------------------
def test_latest_forecast_shape_and_content(ctx, panel, train_features, abc_train, backtest):
    """Every row is made at the origin for the target, by the champion, and is non-negative."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    latest = result["latest"]

    assert list(latest.columns) == LATEST_FORECAST_COLUMNS
    assert set(latest["forecast_origin"]) == {ORIGIN}
    assert set(latest["target_month"]) == {TARGET}
    assert set(latest["model"]) == {B2}
    assert (latest["prediction"] >= 0).all()
    assert set(latest["status"]) == {STATUS_FORECAST}
    assert latest["is_active"].all()
    assert sorted(latest["stock_code"]) == [ACTIVE_A, ACTIVE_B]
    assert latest.loc[latest["stock_code"] == ACTIVE_B, "description"].iloc[0] == (
        f"{ACTIVE_B} DESCRIPTION"
    )

    written = ctx.base_dir / paths.LATEST_FORECAST.relative_to(paths.PROJECT_ROOT)
    assert written.is_file()
    assert list(pd.read_csv(written).columns) == LATEST_FORECAST_COLUMNS


def test_the_baseline_champion_forecast_is_its_own_feature(ctx, panel, train_features, abc_train,
                                                           backtest):
    """B2's operational number is ``rolling_mean_3`` — the same rule the back-test scored."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    merged = result["latest"].merge(
        result["features"][["stock_code", "rolling_mean_3"]],
        on="stock_code",
        suffixes=("_latest", "_features"),
    )
    np.testing.assert_allclose(merged["prediction"], merged["rolling_mean_3_features"])


# --------------------------------------------------------------------------
# the §28 formula
# --------------------------------------------------------------------------
def test_the_prd_worked_example():
    """PRD §4: a forecast of 820 with σ = 70 at z = 1.645 gives a target inventory of 936."""
    assert target_inventory(820, 70, 1.645) == 936


def test_target_inventory_is_recomputable_row_by_row(ctx, panel, train_features, abc_train,
                                                     backtest):
    """Every forecast row satisfies ``ceil(max(0, forecast + z*sigma))``, recomputed from the
    written CSV rather than from the frame in memory."""
    _plan(ctx, panel, train_features, abc_train, backtest)
    written = ctx.base_dir / paths.INVENTORY_PLAN.relative_to(paths.PROJECT_ROOT)
    plan = pd.read_csv(written)

    rows = plan.loc[plan["status"] == STATUS_FORECAST]
    assert len(rows) == 2
    for row in rows.to_dict(orient="records"):
        expected = math.ceil(max(0.0, row["forecast"] + row["z"] * row["sigma"]))
        assert row["target_inventory"] == expected
        assert row["safety_stock"] == pytest.approx(row["z"] * row["sigma"])
        assert row["z"] == POLICY.z


def test_sigma_source_is_one_of_the_three_levels(ctx, panel, train_features, abc_train, backtest):
    """§27's hierarchy: ``ACT-A`` has enough residuals of its own, ``ACT-B`` falls back a level."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    rows = result["plan"].loc[result["plan"]["status"] == STATUS_FORECAST].set_index("stock_code")

    assert set(rows["sigma_source"]) <= set(POLICY.sigma.fallback_levels)
    assert rows.loc[ACTIVE_A, "sigma_source"] == "product"
    assert rows.loc[ACTIVE_A, "n_residuals_product"] >= POLICY.sigma.min_residuals_product
    assert rows.loc[ACTIVE_B, "sigma_source"] != "product"
    assert rows.loc[ACTIVE_B, "n_residuals_product"] < POLICY.sigma.min_residuals_product


def test_operational_sigma_uses_only_residuals_before_the_target(abc_train, backtest):
    """A residual dated at the target month is the row being forecast and must not price itself."""
    latest = pd.DataFrame({"stock_code": [ACTIVE_A], "target_month": [TARGET]})
    clean = operational_sigma(backtest, abc_train, latest, B2, POLICY)

    poisoned = pd.concat(
        [
            backtest,
            pd.DataFrame(
                [{"stock_code": ACTIVE_A, "target_month": TARGET, "model": B2, "residual": 9e6}]
            ),
        ],
        ignore_index=True,
    )
    assert operational_sigma(poisoned, abc_train, latest, B2, POLICY)["sigma"].iloc[0] == (
        clean["sigma"].iloc[0]
    )


def test_uncertainty_ratio(ctx, panel, train_features, abc_train, backtest):
    """§4's extra column: σ relative to the forecast, floored at one unit so it never divides
    by zero."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    rows = result["plan"].loc[result["plan"]["status"] == STATUS_FORECAST]
    expected = rows["sigma"] / np.maximum(rows["forecast"], 1.0)
    np.testing.assert_allclose(rows["uncertainty_ratio"], expected)


# --------------------------------------------------------------------------
# the whole product universe & the §15 cold-start statuses
# --------------------------------------------------------------------------
def test_plan_covers_every_panel_product_with_an_explicit_status(ctx, panel, train_features,
                                                                 abc_train, backtest):
    """Nothing is silently dropped: an inactive or brand-new product gets a row and a reason."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    plan = result["plan"].set_index("stock_code")

    assert list(result["plan"].columns) == INVENTORY_PLAN_COLUMNS
    assert sorted(plan.index) == sorted(panel["stock_code"].unique())
    assert set(plan["status"]) == {
        STATUS_FORECAST,
        inactive_status(K),
        STATUS_NEW_PRODUCT,
    }
    assert plan.loc[ACTIVE_A, "status"] == STATUS_FORECAST
    assert plan.loc[STOPPED, "status"] == inactive_status(K)
    assert plan.loc[NEWBIE, "status"] == STATUS_NEW_PRODUCT
    assert set(plan["run_id"]) == {ctx.run_id}
    assert set(plan["forecast_origin"]) == {ORIGIN}
    assert set(plan["target_month"]) == {TARGET}


def test_products_without_a_forecast_carry_no_numbers(ctx, panel, train_features, abc_train,
                                                      backtest):
    """§15: no cold-start forecast, no artificial history, and therefore no target inventory."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    plan = result["plan"].set_index("stock_code")

    for code in (STOPPED, NEWBIE):
        row = plan.loc[code]
        assert pd.isna(row["forecast"])
        assert pd.isna(row["target_inventory"])
        assert pd.isna(row["safety_stock"])
        assert pd.isna(row["sigma"])
        assert pd.isna(row["sigma_source"])
        assert pd.isna(row["z"])
        assert pd.isna(row["months_since_last_sale"])
        assert pd.isna(row["product_age_months"])

    # NEWBIE has no observed month at or before the origin at all, so not even last month's units.
    assert pd.isna(plan.loc[NEWBIE, "last_month_units"])
    assert plan.loc[STOPPED, "last_month_units"] == 0


def test_last_month_units_matches_the_lag_1_feature(ctx, panel, train_features, abc_train,
                                                    backtest):
    """The plan's history column and the model's ``lag_1`` feature are the same number."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    merged = result["plan"].merge(result["latest"], on="stock_code", suffixes=("", "_latest"))
    rows = merged.loc[merged["status"] == STATUS_FORECAST]
    assert rows["last_month_units"].tolist() == rows["lag_1"].tolist()


def test_written_plan_keeps_whole_numbers_whole(ctx, panel, train_features, abc_train, backtest):
    """``target_inventory`` is a count of units: ``936``, never ``936.000000`` and never a
    phantom ``0``."""
    _plan(ctx, panel, train_features, abc_train, backtest)
    written = ctx.base_dir / paths.INVENTORY_PLAN.relative_to(paths.PROJECT_ROOT)
    text = written.read_text(encoding="utf-8")

    header, *lines = text.strip().split("\n")
    position = header.split(",").index("target_inventory")
    cells = [line.split(",")[position] for line in lines]
    for cell in cells:
        assert cell == "" or cell.isdigit()
    # The products with no forecast leave the cell empty rather than claiming a target of zero.
    assert sum(cell == "" for cell in cells) == 2


# --------------------------------------------------------------------------
# §7 naming
# --------------------------------------------------------------------------
def test_the_output_is_named_recommended_target_inventory(ctx, panel, train_features, abc_train,
                                                          backtest):
    """§7: the dataset has no on-hand position, so this is a target level, never an order size."""
    _plan(ctx, panel, train_features, abc_train, backtest)

    meta_path = ctx.base_dir / paths.MODEL_META.relative_to(paths.PROJECT_ROOT)
    meta_text = meta_path.read_text(encoding="utf-8")
    assert INVENTORY_OUTPUT_NAME == "Recommended Target Inventory"
    assert INVENTORY_OUTPUT_NAME in meta_text

    written = [
        ctx.base_dir / paths.LATEST_FORECAST.relative_to(paths.PROJECT_ROOT),
        ctx.base_dir / paths.INVENTORY_PLAN.relative_to(paths.PROJECT_ROOT),
    ]
    for path in [*written, meta_path]:
        lowered = path.read_text(encoding="utf-8").lower()
        assert "order quantity" not in lowered


def test_no_source_file_says_order_quantity():
    """The forbidden phrase never enters the code base either (§7, issue §6)."""
    offenders = [
        path.relative_to(paths.PROJECT_ROOT).as_posix()
        for path in sorted((paths.PROJECT_ROOT / "src").rglob("*.py"))
        if "order quantity" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# champion resolution & the sanity report
# --------------------------------------------------------------------------
def test_champion_comes_from_the_context_before_the_file(ctx):
    """US-22 puts the decision on the run; re-reading the JSON under staging would be stale."""
    ctx.champion = {"champion": B2, "holdout_metrics_reference": {"wmape": 0.5, "bias": 0.01}}
    decision = resolve_champion(ctx)
    assert champion_id(decision) == B2


def test_a_missing_champion_decision_is_a_hard_stop(ctx, monkeypatch, tmp_path):
    """No champion, no forecast — this module never picks one itself (§20, CLAUDE.md §6)."""
    monkeypatch.setattr(paths, "CHAMPION_DECISION", tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="US-22"):
        resolve_champion(ctx)


def test_sanity_report_mentions_every_status(ctx, panel, train_features, abc_train, backtest):
    """The printed summary is the operator's first read of a run — it must be complete."""
    result = _plan(ctx, panel, train_features, abc_train, backtest)
    report = sanity_report(result["plan"], result["latest"], POLICY)

    assert ORIGIN in report and TARGET in report
    assert INVENTORY_OUTPUT_NAME in report
    assert inactive_status(K) in report
    assert STATUS_NEW_PRODUCT in report
    for level in POLICY.sigma.fallback_levels:
        assert level in report
