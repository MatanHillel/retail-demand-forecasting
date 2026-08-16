"""Split, training-window ABC and baseline tests (US-16, PRD §18.1, §18.2, §19, §21, §22, §23,
§27, §55).

Uses the real ``model_config.yaml`` dates on purpose — the temporal split is the project's one
fixed design (§21), not something a test panel should be free to reinvent. Hand-built mini panels
cover the ABC leakage boundary and the three baseline definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import paths
from pipeline.baselines import (
    B1,
    B2,
    B3,
    BASELINE_MODEL_IDS,
    PREDICTIONS_COLUMNS,
    predict_baselines,
)
from pipeline.config import load_inventory_policy
from pipeline.split import (
    ABC_SOURCE_DEFAULT_NO_HISTORY,
    ABC_SOURCE_TRAINING_WINDOW,
    ABC_TRAIN_COLUMNS,
    SplitSpec,
    abc_train,
    holdout_mask,
    rolling_origins,
    train_mask,
    training_window_end,
    validation_mask,
    validation_origins,
)


def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


@pytest.fixture(scope="module")
def spec() -> SplitSpec:
    return SplitSpec.load()


# --------------------------------------------------------------------------
# masks (§21) — train 2010-03…2011-05, hold-out 2011-06…2011-11, 2011-12 never scored
# --------------------------------------------------------------------------
def test_train_mask_covers_exactly_the_train_targets(spec: SplitSpec) -> None:
    frame = pd.DataFrame({"target_month": _months("2010-01", "2012-01")})
    covered = sorted(frame.loc[train_mask(frame, spec), "target_month"])
    assert covered == _months("2010-03", "2011-05")


def test_holdout_mask_covers_exactly_the_holdout_targets_and_never_december(
    spec: SplitSpec,
) -> None:
    frame = pd.DataFrame({"target_month": _months("2010-01", "2012-01")})
    covered = sorted(frame.loc[holdout_mask(frame, spec), "target_month"])
    assert covered == _months("2011-06", "2011-11")
    assert "2011-12" not in covered


def test_train_and_holdout_masks_do_not_overlap(spec: SplitSpec) -> None:
    frame = pd.DataFrame({"target_month": _months("2010-01", "2012-01")})
    train = set(frame.loc[train_mask(frame, spec), "target_month"])
    holdout = set(frame.loc[holdout_mask(frame, spec), "target_month"])
    assert train.isdisjoint(holdout)


def test_validation_mask_is_a_subset_of_train_mask(spec: SplitSpec) -> None:
    frame = pd.DataFrame({"target_month": _months("2010-01", "2012-01")})
    train = set(frame.loc[train_mask(frame, spec), "target_month"])
    validation = set(frame.loc[validation_mask(frame, spec), "target_month"])
    assert validation.issubset(train)
    assert sorted(validation) == _months("2011-01", "2011-05")


# --------------------------------------------------------------------------
# rolling origins (§22)
# --------------------------------------------------------------------------
def test_rolling_origins_has_eighteen_entries_from_config(spec: SplitSpec) -> None:
    origins = rolling_origins(spec)
    assert len(origins) == 18
    assert origins[0] == spec.backtest.first_origin
    assert origins[-1] == spec.backtest.last_origin
    assert origins == sorted(origins)


def test_validation_origins_target_exactly_the_validation_months(spec: SplitSpec) -> None:
    origins = validation_origins(spec)
    targets = sorted(str(pd.Period(origin, freq="M") + 1) for origin in origins)
    assert targets == _months("2011-01", "2011-05")
    assert set(origins).issubset(set(rolling_origins(spec)))


def test_training_window_end_matches_train_targets_end(spec: SplitSpec) -> None:
    assert training_window_end(spec) == spec.split.train_targets.end == "2011-05"


def test_split_spec_load_is_a_thin_wrapper_over_model_config() -> None:
    from pipeline.config import load_model_config

    cfg = load_model_config()
    spec = SplitSpec.load()
    assert spec.split == cfg.split
    assert spec.backtest == cfg.backtest


# --------------------------------------------------------------------------
# abc_train — training-window only, leakage guard (§18.2, §23, §27)
# --------------------------------------------------------------------------
def _panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


_BASE_ABC_ROWS = [
    {"stock_code": "A1", "month": "2010-03", "gross_revenue": 1000.0, "units_sold": 10},
    {"stock_code": "A1", "month": "2011-05", "gross_revenue": 500.0, "units_sold": 5},
    {"stock_code": "B1", "month": "2010-03", "gross_revenue": 10.0, "units_sold": 1},
    {"stock_code": "B1", "month": "2011-05", "gross_revenue": 10.0, "units_sold": 1},
]


def test_abc_train_ignores_revenue_after_the_training_cutoff(spec: SplitSpec) -> None:
    """Injecting a huge post-cut-off sale for a *different* product must not move A1's class."""
    policy = load_inventory_policy()
    before = abc_train(_panel(_BASE_ABC_ROWS), spec, policy)
    class_before = before.set_index("stock_code").loc["A1", "abc_class"]

    tampered = [
        *_BASE_ABC_ROWS,
        {"stock_code": "B1", "month": "2011-06", "gross_revenue": 1_000_000.0, "units_sold": 1},
    ]
    after = abc_train(_panel(tampered), spec, policy)
    class_after = after.set_index("stock_code").loc["A1", "abc_class"]

    assert class_before == class_after


def test_abc_train_covers_every_panel_product_exactly_once(spec: SplitSpec) -> None:
    policy = load_inventory_policy()
    rows = [
        {"stock_code": "A1", "month": "2010-03", "gross_revenue": 100.0, "units_sold": 10},
        {"stock_code": "LATE", "month": "2011-08", "gross_revenue": 100.0, "units_sold": 10},
    ]
    frame = abc_train(_panel(rows), spec, policy)
    assert sorted(frame["stock_code"]) == ["A1", "LATE"]
    assert not frame["stock_code"].duplicated().any()

    late_row = frame.set_index("stock_code").loc["LATE"]
    assert late_row["abc_class"] == "C"
    assert late_row["abc_source"] == ABC_SOURCE_DEFAULT_NO_HISTORY
    assert late_row["revenue_share"] == 0.0
    assert late_row["cum_share"] == 0.0

    a1_row = frame.set_index("stock_code").loc["A1"]
    assert a1_row["abc_source"] == ABC_SOURCE_TRAINING_WINDOW


def test_abc_train_column_order_and_valid_classes(spec: SplitSpec) -> None:
    policy = load_inventory_policy()
    rows = [{"stock_code": "A1", "month": "2010-03", "gross_revenue": 100.0, "units_sold": 10}]
    frame = abc_train(_panel(rows), spec, policy)
    assert list(frame.columns) == ABC_TRAIN_COLUMNS
    assert set(frame["abc_class"]).issubset({"A", "B", "C"})


# --------------------------------------------------------------------------
# baselines (§18.1, §19)
# --------------------------------------------------------------------------
def _features_row(
    stock_code: str, target_month: str, lag_1: float, rolling_mean_3: float
) -> dict[str, object]:
    origin = str(pd.Period(target_month, freq="M") - 1)
    return {
        "stock_code": stock_code,
        "forecast_origin": origin,
        "target_month": target_month,
        "lag_1": lag_1,
        "rolling_mean_3": rolling_mean_3,
    }


def test_b1_equals_lag_1_and_b2_equals_rolling_mean_3() -> None:
    features = pd.DataFrame(
        [
            _features_row("P1", "2011-01", lag_1=7, rolling_mean_3=5.5),
            _features_row("P2", "2011-02", lag_1=0, rolling_mean_3=1.5),
        ]
    )
    panel = pd.DataFrame({"stock_code": [], "month": [], "units_sold": []})
    predictions = predict_baselines(features, panel)

    b1 = predictions.loc[predictions["model"] == B1].set_index("stock_code")["prediction"]
    b2 = predictions.loc[predictions["model"] == B2].set_index("stock_code")["prediction"]
    assert b1["P1"] == 7
    assert b1["P2"] == 0
    assert b2["P1"] == pytest.approx(5.5)
    assert b2["P2"] == pytest.approx(1.5)


def test_b3_equals_panel_units_twelve_months_earlier_or_nan_when_unobserved() -> None:
    features = pd.DataFrame(
        [
            _features_row("P1", "2011-06", lag_1=1, rolling_mean_3=1.0),  # panel has 2010-06
            _features_row("P2", "2011-06", lag_1=1, rolling_mean_3=1.0),  # panel lacks 2010-06
        ]
    )
    panel = pd.DataFrame(
        [
            {"stock_code": "P1", "month": "2010-06", "units_sold": 42},
            {"stock_code": "P2", "month": "2011-01", "units_sold": 9},
        ]
    )
    predictions = predict_baselines(features, panel)
    b3 = predictions.loc[predictions["model"] == B3].set_index("stock_code")["prediction"]
    assert b3["P1"] == 42
    assert np.isnan(b3["P2"])


def test_predict_baselines_output_schema_and_row_count() -> None:
    features = pd.DataFrame(
        [
            _features_row("P1", "2011-01", lag_1=7, rolling_mean_3=5.5),
            _features_row("P2", "2011-02", lag_1=0, rolling_mean_3=1.5),
        ]
    )
    panel = pd.DataFrame({"stock_code": [], "month": [], "units_sold": []})
    predictions = predict_baselines(features, panel)

    assert list(predictions.columns) == PREDICTIONS_COLUMNS
    assert len(predictions) == 3 * len(features)
    assert set(predictions["model"]) == set(BASELINE_MODEL_IDS)
    assert not predictions.loc[predictions["model"] == B1, "prediction"].isna().any()
    assert not predictions.loc[predictions["model"] == B2, "prediction"].isna().any()


def test_baselines_use_the_same_rows_for_every_model() -> None:
    """Same active feature rows, all three models — only B3's values may be missing (§19)."""
    features = pd.DataFrame(
        [
            _features_row("P1", "2011-01", lag_1=7, rolling_mean_3=5.5),
            _features_row("P2", "2011-02", lag_1=0, rolling_mean_3=1.5),
            _features_row("P3", "2011-03", lag_1=2, rolling_mean_3=2.0),
        ]
    )
    panel = pd.DataFrame({"stock_code": [], "month": [], "units_sold": []})
    predictions = predict_baselines(features, panel)
    counts = predictions.groupby("model").size()
    assert counts[B1] == counts[B2] == counts[B3] == len(features)


# --------------------------------------------------------------------------
# guard: shuffled splits are forbidden everywhere (§2 rule 2, §21)
# --------------------------------------------------------------------------
def test_train_test_split_is_never_referenced_under_src() -> None:
    src = paths.PROJECT_ROOT / "src"
    offenders = [
        str(path.relative_to(paths.PROJECT_ROOT))
        for path in src.rglob("*.py")
        if "train_test_split" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"train_test_split referenced in: {offenders}"
