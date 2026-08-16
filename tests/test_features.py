"""Feature engineering tests (US-13, PRD §14, §16, §17, §21, §47, §55).

Hand-built mini panels only — the PRD's indicative sizes (≈ 70,700 rows, ≈ 4,850 products) are
expectations to log, never to assert (§40). The three products below are the three cases the
window conventions differ on, and their expected numbers are computed by hand in the docstrings:

* ``P1`` sells from the dataset's first month — its early windows are **truncated**, because the
  months before the data begins were never observed.
* ``P2`` launches in 2010-06 — the months before that **were** observed and count as zeros.
* ``P3`` sells once, in 2010-04, then stops — it exercises the active rule's exit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from pipeline import paths
from pipeline.config import clear_config_cache, load_model_config
from pipeline.features import (
    FEATURE_COLUMNS,
    FEATURES_COLUMNS,
    build_features,
    build_features_for_origin,
)

K = 6
FIRST_TARGET = "2010-03"
LAST_TARGET = "2010-10"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _months(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _panel(sales: dict[str, dict[str, int]], end: str = "2010-12") -> pd.DataFrame:
    """A US-05-shaped panel: each product runs from its first sale to ``end``, zero-filled.

    ``sales`` maps ``stock_code -> {month: units}``; months in between that are absent are the
    zero-sales months the panel spells out explicitly.
    """
    rows = []
    for stock_code, months in sales.items():
        first = min(months)
        for month in _months(first, end):
            units = months.get(month, 0)
            rows.append(
                {
                    "month": month,
                    "stock_code": stock_code,
                    "description": "TEST PRODUCT",
                    "units_sold": units,
                    "gross_revenue": float(units) * 2.0,
                    "avg_unit_price": 2.0,
                    "invoice_count": 3 if units else 0,
                    "sale_line_count": 3 if units else 0,
                    "customer_count": 2 if units else 0,
                    "max_line_qty": units,
                    "returned_units": 0,
                    "is_partial_month": False,
                }
            )
    return pd.DataFrame(rows)


#: P1 from the first observed month, P2 launched later, P3 a single sale then silence.
SALES = {
    "P1": {"2009-12": 100, "2010-02": 50, "2010-03": 7, "2010-04": 12},
    "P2": {"2010-06": 40, "2010-08": 20, "2010-09": 9},
    "P3": {"2010-04": 30},
}


@pytest.fixture()
def features() -> pd.DataFrame:
    return build_features(
        _panel(SALES), K, FIRST_TARGET, LAST_TARGET, load_model_config()
    )


def _row(frame: pd.DataFrame, stock_code: str, target: str) -> pd.Series:
    match = frame[(frame["stock_code"] == stock_code) & (frame["target_month"] == target)]
    assert len(match) == 1, f"expected exactly one row for {stock_code} / {target}"
    return match.iloc[0]


# --------------------------------------------------------------------------
# P1 — truncated windows at the start of the dataset (§17, §47)
# --------------------------------------------------------------------------
def test_p1_first_target_uses_truncated_windows(features: pd.DataFrame) -> None:
    """Target 2010-03, origin 2010-02. Units: 2009-12 → 100, 2010-01 → 0, 2010-02 → 50.

    The window t−6…t−1 is 2009-09…2010-02, but only three of those months were ever observed —
    the data begins 2009-12 — so the six-month statistics divide by three, not by six.
    """
    row = _row(features, "P1", FIRST_TARGET)
    assert row["forecast_origin"] == "2010-02"
    assert (row["lag_1"], row["lag_2"], row["lag_3"]) == (50, 0, 100)
    assert row["rolling_mean_3"] == pytest.approx((100 + 0 + 50) / 3)
    assert row["rolling_mean_6"] == pytest.approx((100 + 0 + 50) / 3)  # divided by 3, not 6
    assert row["rolling_median_6"] == pytest.approx(50.0)
    assert row["rolling_std_3"] == pytest.approx(np.std([100.0, 0.0, 50.0]))  # ddof=0
    assert row["rolling_max_6"] == 100
    assert row["nonzero_months_6"] == 2
    assert row["months_since_last_sale"] == 1
    assert row["product_age_months"] == 3  # Dec, Jan, Feb — a lower bound (§47)
    assert row["invoice_count_lag_1"] == 3
    assert row["avg_unit_price_lag_1"] == pytest.approx(2.0)
    assert (row["target_month_of_year"], row["target_quarter"]) == (3, 1)
    assert row["y"] == 7


# --------------------------------------------------------------------------
# P2 — observed zeros before the first sale (§17)
# --------------------------------------------------------------------------
def test_p2_months_before_first_sale_are_observed_zeros(features: pd.DataFrame) -> None:
    """Target 2010-09, origin 2010-08. Units: 2010-06 → 40, 2010-07 → 0, 2010-08 → 20.

    The whole window 2010-03…2010-08 was observed; P2 had simply not launched. Those months are
    real zero-demand months, so ``rolling_mean_6`` is 60/6 = 10.0 — the divisor is what separates
    this case from P1's truncation, and the two must not be confused.
    """
    row = _row(features, "P2", "2010-09")
    assert (row["lag_1"], row["lag_2"], row["lag_3"]) == (20, 0, 40)
    assert row["rolling_mean_3"] == pytest.approx(60 / 3)
    assert row["rolling_mean_6"] == pytest.approx(60 / 6)  # divided by 6, not 3
    assert row["rolling_median_6"] == pytest.approx(0.0)  # median of 0,0,0,0,20,40
    assert row["rolling_max_6"] == 40
    assert row["nonzero_months_6"] == 2
    assert row["months_since_last_sale"] == 1
    assert row["product_age_months"] == 3


def test_truncated_and_zero_filled_windows_differ(features: pd.DataFrame) -> None:
    """The same arithmetic sum, a different divisor — the distinction this module exists to keep."""
    p1 = _row(features, "P1", FIRST_TARGET)
    p2 = _row(features, "P2", "2010-09")
    assert p1["rolling_mean_3"] == pytest.approx(p1["rolling_mean_6"])  # only 3 months observed
    assert p2["rolling_mean_6"] == pytest.approx(p2["rolling_mean_3"] / 2)  # all 6 observed


def test_product_launched_later_has_no_row_before_its_first_sale(features: pd.DataFrame) -> None:
    """P2's first sale is 2010-06, so its first target is 2010-07 — at 2010-06 it was not active."""
    targets = sorted(features.loc[features["stock_code"] == "P2", "target_month"])
    assert targets == _months("2010-07", LAST_TARGET)


# --------------------------------------------------------------------------
# P3 — the active rule's exit (§14)
# --------------------------------------------------------------------------
def test_intermittent_product_has_rows_only_for_k_months_after_its_sale(
    features: pd.DataFrame,
) -> None:
    """A single sale in month m yields targets m+1 … m+k and nothing after."""
    targets = sorted(features.loc[features["stock_code"] == "P3", "target_month"])
    assert targets == _months("2010-05", "2010-10")  # 2010-04 + 1 … 2010-04 + k


def test_p3_last_active_target(features: pd.DataFrame) -> None:
    """Target 2010-10: the sale of 30 units in 2010-04 is exactly k months back."""
    row = _row(features, "P3", "2010-10")
    assert (row["lag_1"], row["lag_2"], row["lag_3"]) == (0, 0, 0)
    assert row["months_since_last_sale"] == K
    assert row["nonzero_months_6"] == 1
    assert row["rolling_mean_6"] == pytest.approx(30 / 6)
    assert row["rolling_max_6"] == 30
    assert row["y"] == 0


# --------------------------------------------------------------------------
# lag correctness against the panel it came from (§17)
# --------------------------------------------------------------------------
def test_lags_equal_the_panel_units_of_the_preceding_months(features: pd.DataFrame) -> None:
    panel = _panel(SALES)
    units = {(row.stock_code, row.month): row.units_sold for row in panel.itertuples()}
    for row in features.itertuples():
        for offset, column in ((1, "lag_1"), (2, "lag_2"), (3, "lag_3")):
            month = str(pd.Period(row.target_month, freq="M") - offset)
            # Months before the product's first sale are absent from the panel and read as 0.
            assert getattr(row, column) == units.get((row.stock_code, month), 0)


# --------------------------------------------------------------------------
# invariants that must hold on every row
# --------------------------------------------------------------------------
def test_no_nan_in_any_feature(features: pd.DataFrame) -> None:
    assert not features[FEATURE_COLUMNS].isna().any().any()


def test_forecast_origin_is_exactly_one_month_before_the_target(features: pd.DataFrame) -> None:
    origins = pd.PeriodIndex(features["forecast_origin"], freq="M")
    targets = pd.PeriodIndex(features["target_month"], freq="M")
    assert ((targets - origins).map(lambda offset: offset.n) == 1).all()


def test_months_since_last_sale_never_exceeds_k(features: pd.DataFrame) -> None:
    """A larger value would mean no sale in the previous k months — the row would not be active."""
    assert features["months_since_last_sale"].between(1, K).all()


def test_product_age_is_at_least_one_month(features: pd.DataFrame) -> None:
    assert (features["product_age_months"] >= 1).all()


def test_calendar_features_are_in_range(features: pd.DataFrame) -> None:
    assert features["target_month_of_year"].between(1, 12).all()
    assert features["target_quarter"].between(1, 4).all()
    for row in features.itertuples():
        period = pd.Period(row.target_month, freq="M")
        assert (row.target_month_of_year, row.target_quarter) == (period.month, period.quarter)


def test_primary_key_is_unique_and_is_active_is_always_true(features: pd.DataFrame) -> None:
    assert not features.duplicated(subset=["stock_code", "target_month"]).any()
    assert features["is_active"].all()


def test_targets_stay_inside_the_requested_window(features: pd.DataFrame) -> None:
    assert features["target_month"].min() == FIRST_TARGET
    assert features["target_month"].max() <= LAST_TARGET


def test_rows_are_sorted_by_stock_code_then_target_month(features: pd.DataFrame) -> None:
    expected = features.sort_values(["stock_code", "target_month"], kind="mergesort")
    assert features["stock_code"].tolist() == expected["stock_code"].tolist()
    assert features["target_month"].tolist() == expected["target_month"].tolist()


# --------------------------------------------------------------------------
# schema (§41 acceptance criteria)
# --------------------------------------------------------------------------
def test_column_order_is_the_published_schema(features: pd.DataFrame) -> None:
    assert list(features.columns) == FEATURES_COLUMNS


def test_feature_list_equals_model_config_features_in_order() -> None:
    assert FEATURE_COLUMNS == list(load_model_config().features)
    assert FEATURES_COLUMNS[3:-2] == FEATURE_COLUMNS


def test_stock_code_is_not_a_feature() -> None:
    """§17: the identifier is never an input — no per-SKU model."""
    assert "stock_code" not in FEATURE_COLUMNS


def test_diagnostic_panel_columns_are_never_features() -> None:
    """§9, §13.2: returns and customer counts are EDA-only."""
    for forbidden in ("returned_units", "customer_count", "country", "description"):
        assert not any(forbidden in feature for feature in FEATURE_COLUMNS)


# --------------------------------------------------------------------------
# no leakage (§16) — a smoke test; US-14 runs the full permutation check
# --------------------------------------------------------------------------
def test_changing_the_target_month_and_later_leaves_features_unchanged() -> None:
    """Only ``y`` may move when month t and everything after it changes."""
    baseline = build_features(_panel(SALES), K, FIRST_TARGET, LAST_TARGET, load_model_config())

    tampered_sales = {code: dict(months) for code, months in SALES.items()}
    tampered_sales["P1"]["2010-03"] = 9_999  # the target month of the first row
    tampered_sales["P1"]["2010-04"] = 8_888  # and a later one
    tampered = build_features(
        _panel(tampered_sales), K, FIRST_TARGET, LAST_TARGET, load_model_config()
    )

    first = tampered["target_month"] == FIRST_TARGET
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["target_month"] == FIRST_TARGET, FEATURE_COLUMNS].reset_index(
            drop=True
        ),
        tampered.loc[first, FEATURE_COLUMNS].reset_index(drop=True),
    )
    assert _row(tampered, "P1", FIRST_TARGET)["y"] == 9_999


# --------------------------------------------------------------------------
# build_features_for_origin (US-23)
# --------------------------------------------------------------------------
def test_build_features_for_origin_targets_the_next_month_and_omits_y() -> None:
    frame = build_features_for_origin(_panel(SALES), "2010-09", K, load_model_config())
    assert set(frame["target_month"]) == {"2010-10"}
    assert set(frame["forecast_origin"]) == {"2010-09"}
    assert "y" not in frame.columns
    assert list(frame.columns) == [column for column in FEATURES_COLUMNS if column != "y"]


def test_build_features_for_origin_matches_the_training_features() -> None:
    """The operational path must not drift from the trained one — same code, same numbers."""
    training = build_features(_panel(SALES), K, FIRST_TARGET, LAST_TARGET, load_model_config())
    operational = build_features_for_origin(_panel(SALES), "2010-09", K, load_model_config())
    expected = training[training["target_month"] == "2010-10"].drop(columns="y")
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), operational.reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def test_k_changes_how_long_a_product_stays_active() -> None:
    """``k`` is a parameter, never a literal: a shorter window drops P3 out sooner."""
    frame = build_features(_panel(SALES), 2, FIRST_TARGET, LAST_TARGET, load_model_config())
    targets = sorted(frame.loc[frame["stock_code"] == "P3", "target_month"])
    assert targets == _months("2010-05", "2010-06")  # 2010-04 + 1 … 2010-04 + 2


def test_a_config_whose_feature_list_disagrees_with_the_module_is_rejected(
    tmp_path, monkeypatch
) -> None:
    """``load_model_config`` is cached and frozen, so the override goes through a temp YAML (§8)."""
    content = yaml.safe_load(paths.MODEL_CONFIG.read_text(encoding="utf-8"))
    content["features"] = list(reversed(content["features"]))
    temporary = tmp_path / "model_config.yaml"
    temporary.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(paths, "MODEL_CONFIG", temporary)
    clear_config_cache()
    try:
        with pytest.raises(ValueError, match="features must match"):
            build_features(_panel(SALES), K, FIRST_TARGET, LAST_TARGET, load_model_config())
    finally:
        monkeypatch.undo()
        clear_config_cache()
