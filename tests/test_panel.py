"""Product × Month panel tests (US-05, PRD §9, §10 steps 11-12, §13.2, §14, §55).

Structure and rules only — the PRD's indicative sizes (≈ 99 k rows, ≈ 4,890 products) are never
asserted. Tests use tiny hand-built frames plus one build over the committed CI fixture, always
against a ``tmp_path`` base directory so nothing real is overwritten.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.active import active_mask
from pipeline.cleaning import clean_transactions
from pipeline.config import load_cleaning_config, load_model_config
from pipeline.download import load_raw
from pipeline.panel import PANEL_COLUMNS, build_panel, validate_panel
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

EXPECTED_COLUMNS = [
    "month",
    "stock_code",
    "description",
    "units_sold",
    "gross_revenue",
    "avg_unit_price",
    "invoice_count",
    "sale_line_count",
    "customer_count",
    "max_line_qty",
    "returned_units",
    "is_partial_month",
]

RETURNS_COLUMNS = ["stock_code", "month", "quantity_abs", "invoice"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _panel_end() -> pd.Period:
    """Last month the panel must reach: the latest configured partial month."""
    raw = load_cleaning_config().raw
    return pd.Period(max([raw.last_full_month, *raw.partial_months]), freq="M")


def _clean(rows: list[dict]) -> pd.DataFrame:
    """Cleaned-transaction frame (the US-04 output shape); each dict overrides one row."""
    defaults = {
        "invoice": "500001",
        "stock_code": "10001",
        "description": "RED MUG",
        "quantity": 10,
        "month": "2010-03",
        "price": 2.5,
        "customer_id": 13085,
        "country": "United Kingdom",
        "source_sheet": "Year 2009-2010",
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["description_raw"] = frame["description"]
    frame["invoice_date"] = pd.PeriodIndex(frame["month"], freq="M").to_timestamp()
    frame["line_revenue"] = frame["quantity"] * frame["price"]
    frame["is_partial_month"] = frame["month"].isin(load_cleaning_config().raw.partial_months)
    frame["is_abnormal_qty"] = False
    frame["customer_id"] = frame["customer_id"].astype("Int64")
    return frame


def _returns(rows: list[dict]) -> pd.DataFrame:
    """Cancellation side table (the US-04 ``returns_lines`` shape)."""
    if not rows:
        return pd.DataFrame({name: [] for name in RETURNS_COLUMNS}).astype(
            {"stock_code": str, "month": str, "quantity_abs": "int64", "invoice": str}
        )
    defaults = {"stock_code": "10001", "month": "2010-03", "quantity_abs": 3, "invoice": "C900"}
    return pd.DataFrame([{**defaults, **row} for row in rows])[RETURNS_COLUMNS]


def _build(clean: pd.DataFrame, returns: pd.DataFrame, base_dir: Path):
    """Run ``build_panel`` inside a step against an isolated base directory."""
    ctx = RunContext.start(mode="no-llm", base_dir=base_dir)
    try:
        with ctx.step("build_panel"):
            panel = build_panel(clean, returns, load_cleaning_config(), ctx)
    finally:
        close_log_handlers(ctx.run_id)
    return panel, ctx


def _months(panel: pd.DataFrame, stock_code: str) -> list[str]:
    return list(panel.loc[panel["stock_code"] == stock_code, "month"])


@pytest.fixture(scope="module")
def sample_panel(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One panel built from the committed CI fixture, shared by the structural tests."""
    base = tmp_path_factory.mktemp("panel_sample")
    raw, _ = load_raw(RAW_SAMPLE)
    ctx = RunContext.start(mode="no-llm", base_dir=base)
    try:
        with ctx.step("cleaning"):
            clean, _ = clean_transactions(raw, load_cleaning_config(), ctx)
        returns = pd.read_parquet(base / "data" / "processed" / "returns_lines.parquet")
        with ctx.step("build_panel"):
            panel = build_panel(clean, returns, load_cleaning_config(), ctx)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    return SimpleNamespace(clean=clean, returns=returns, panel=panel, ctx=ctx, base=base)


# --------------------------------------------------------------------------
# zero-fill (§10 step 12)
# --------------------------------------------------------------------------
def test_zero_fill_covers_every_month_from_first_sale_to_the_partial_month(
    tmp_path: Path,
) -> None:
    clean = _clean([{"month": "2010-03"}, {"month": "2010-06", "invoice": "500002"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    expected = [
        str(month) for month in pd.period_range("2010-03", _panel_end(), freq="M")
    ]
    assert _months(panel, "10001") == expected
    assert panel["month"].min() == "2010-03"  # nothing before the first sale
    assert panel["month"].max() == str(_panel_end())


def test_no_rows_before_first_sale_and_first_row_is_a_sale(tmp_path: Path) -> None:
    clean = _clean(
        [
            {"stock_code": "10001", "month": "2010-03"},
            {"stock_code": "20002", "month": "2011-02", "invoice": "500002"},
        ]
    )
    panel, _ = _build(clean, _returns([]), tmp_path)
    first_rows = panel.sort_values(["stock_code", "month"]).groupby("stock_code").first()
    assert (first_rows["units_sold"] > 0).all()
    assert _months(panel, "10001")[0] == "2010-03"
    assert _months(panel, "20002")[0] == "2011-02"


def test_product_first_selling_in_the_partial_month_gets_only_that_row(
    tmp_path: Path,
) -> None:
    partial = load_cleaning_config().raw.partial_months[0]
    clean = _clean([{"stock_code": "30003", "month": partial}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    assert _months(panel, "30003") == [partial]
    assert panel["is_partial_month"].all()


def test_partial_flag_is_true_only_for_configured_partial_months(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    partial_months = set(load_cleaning_config().raw.partial_months)
    assert set(panel.loc[panel["is_partial_month"], "month"]) == partial_months
    assert not panel.loc[~panel["is_partial_month"], "month"].isin(partial_months).any()


# --------------------------------------------------------------------------
# aggregation (§10 step 11, §13.2)
# --------------------------------------------------------------------------
def test_aggregation_of_a_single_month(tmp_path: Path) -> None:
    clean = _clean(
        [
            {"month": "2010-03", "invoice": "A1", "quantity": 10, "price": 2.0},
            {"month": "2010-03", "invoice": "A1", "quantity": 5, "price": 4.0},
            {"month": "2010-03", "invoice": "A2", "quantity": 5, "price": 2.0, "customer_id": 99},
        ]
    )
    panel, _ = _build(clean, _returns([]), tmp_path)
    row = panel[panel["month"] == "2010-03"].iloc[0]
    assert row["units_sold"] == 20
    assert row["gross_revenue"] == pytest.approx(50.0)  # 20 + 20 + 10
    assert row["avg_unit_price"] == pytest.approx(2.5)  # revenue-weighted, not the mean price
    assert row["invoice_count"] == 2
    assert row["sale_line_count"] == 3
    assert row["customer_count"] == 2
    assert row["max_line_qty"] == 10


def test_customer_count_ignores_missing_customer_ids(tmp_path: Path) -> None:
    clean = _clean(
        [
            {"month": "2010-03", "customer_id": 13085},
            {"month": "2010-03", "invoice": "500002", "customer_id": pd.NA},
        ]
    )
    panel, _ = _build(clean, _returns([]), tmp_path)
    row = panel[panel["month"] == "2010-03"].iloc[0]
    assert row["customer_count"] == 1
    assert row["sale_line_count"] == 2  # the row itself is kept (§9)


def test_avg_unit_price_forward_fills_through_zero_months(tmp_path: Path) -> None:
    clean = _clean(
        [
            {"month": "2010-03", "quantity": 4, "price": 2.5},
            {"month": "2010-06", "invoice": "500002", "quantity": 4, "price": 9.0},
        ]
    )
    panel, _ = _build(clean, _returns([]), tmp_path)
    by_month = panel.set_index("month")
    assert by_month.loc["2010-04", "avg_unit_price"] == pytest.approx(2.5)
    assert by_month.loc["2010-05", "avg_unit_price"] == pytest.approx(2.5)
    assert by_month.loc["2010-07", "avg_unit_price"] == pytest.approx(9.0)
    assert by_month.loc["2010-04", "units_sold"] == 0
    assert by_month.loc["2010-04", "gross_revenue"] == 0.0


def test_returned_units_come_from_cancellation_lines(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    returns = _returns(
        [
            {"month": "2010-05", "quantity_abs": 7},
            {"month": "2010-05", "quantity_abs": 2, "invoice": "C901"},
            {"month": "2010-03", "quantity_abs": 1},
        ]
    )
    panel, _ = _build(clean, returns, tmp_path)
    by_month = panel.set_index("month")
    assert by_month.loc["2010-05", "returned_units"] == 9  # summed within the month
    assert by_month.loc["2010-05", "units_sold"] == 0  # returns never become demand (§9)
    assert by_month.loc["2010-03", "returned_units"] == 1
    assert by_month.loc["2010-04", "returned_units"] == 0


def test_canonical_description_is_carried_onto_zero_filled_rows(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03", "description": "RED MUG"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    assert set(panel["description"]) == {"RED MUG"}


# --------------------------------------------------------------------------
# schema, key and the written artifact
# --------------------------------------------------------------------------
def test_schema_dtypes_and_primary_key(sample_panel: SimpleNamespace) -> None:
    panel = sample_panel.panel
    assert PANEL_COLUMNS == EXPECTED_COLUMNS
    assert list(panel.columns) == EXPECTED_COLUMNS
    assert panel.duplicated(subset=["stock_code", "month"]).sum() == 0
    for column in (
        "units_sold",
        "invoice_count",
        "sale_line_count",
        "customer_count",
        "max_line_qty",
        "returned_units",
    ):
        assert panel[column].dtype == "int64", column
        assert (panel[column] >= 0).all(), column
    assert panel["gross_revenue"].dtype == "float64"
    assert (panel["gross_revenue"] >= 0).all()
    assert (panel["avg_unit_price"] >= 0).all()
    assert panel["is_partial_month"].dtype == "bool"
    assert list(panel["stock_code"]) == sorted(panel["stock_code"])  # sorted by key


def test_clean_data_csv_written_and_registered(sample_panel: SimpleNamespace) -> None:
    written = sample_panel.base / "data" / "processed" / "clean_data.csv"
    assert written.is_file()
    header = written.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(EXPECTED_COLUMNS)
    assert sample_panel.ctx.artifacts["clean_data"] == "data/processed/clean_data.csv"


def test_panel_build_is_deterministic(sample_panel: SimpleNamespace, tmp_path: Path) -> None:
    _build(sample_panel.clean, sample_panel.returns, tmp_path)
    first = (sample_panel.base / "data" / "processed" / "clean_data.csv").read_bytes()
    second = (tmp_path / "data" / "processed" / "clean_data.csv").read_bytes()
    assert first == second


def test_run_context_records_the_shape_change(sample_panel: SimpleNamespace) -> None:
    step = next(record for record in sample_panel.ctx.steps if record.name == "build_panel")
    assert "panel_zero_fill" in step.row_counts
    metrics = sample_panel.ctx.metrics
    assert metrics["panel_rows"] == len(sample_panel.panel)
    assert metrics["panel_products"] == sample_panel.panel["stock_code"].nunique()
    assert metrics["panel_nonzero_rows"] + metrics["panel_zero_filled_rows"] == metrics[
        "panel_rows"
    ]


# --------------------------------------------------------------------------
# validate_panel — pure, returns a ValidationResult (never raises)
# --------------------------------------------------------------------------
def test_validate_panel_passes_on_a_good_panel(sample_panel: SimpleNamespace) -> None:
    result = validate_panel(sample_panel.panel, load_cleaning_config())
    assert result.passed, result.summary()
    assert result.violations == []
    assert result.checked_rows == len(sample_panel.panel)


def test_validate_panel_detects_duplicate_keys(sample_panel: SimpleNamespace) -> None:
    broken = pd.concat([sample_panel.panel, sample_panel.panel.head(1)], ignore_index=True)
    result = validate_panel(broken, load_cleaning_config())
    assert not result.passed
    assert any(violation.rule == "primary_key" for violation in result.violations)


def test_validate_panel_detects_a_row_before_the_first_sale(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    early = panel.iloc[[0]].copy()
    early["month"] = "2010-02"
    early["units_sold"] = 0
    broken = pd.concat([early, panel], ignore_index=True)
    result = validate_panel(broken, load_cleaning_config())
    assert not result.passed
    assert any(violation.rule == "first_row_is_a_sale" for violation in result.violations)


def test_validate_panel_detects_a_month_gap(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    broken = panel[panel["month"] != "2010-05"]
    result = validate_panel(broken, load_cleaning_config())
    assert not result.passed
    assert any(violation.rule == "contiguous_months" for violation in result.violations)


def test_validate_panel_detects_a_wrong_partial_flag(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    broken = panel.copy()
    broken.loc[broken["month"] == "2010-04", "is_partial_month"] = True
    result = validate_panel(broken, load_cleaning_config())
    assert not result.passed
    assert any(violation.rule == "is_partial_month" for violation in result.violations)


def test_validate_panel_detects_negative_units(tmp_path: Path) -> None:
    clean = _clean([{"month": "2010-03"}])
    panel, _ = _build(clean, _returns([]), tmp_path)
    broken = panel.copy()
    broken.loc[broken.index[1], "units_sold"] = -1
    result = validate_panel(broken, load_cleaning_config())
    assert not result.passed
    assert any(violation.rule == "non_negative" for violation in result.violations)


# --------------------------------------------------------------------------
# active_mask (§14) — only months strictly before t
# --------------------------------------------------------------------------
def _hand_built_panel(units: dict[str, int]) -> pd.DataFrame:
    """Twelve consecutive months for one product, with the given non-zero months."""
    months = [str(month) for month in pd.period_range("2010-01", "2010-12", freq="M")]
    return pd.DataFrame(
        {
            "stock_code": ["10001"] * len(months),
            "month": months,
            "units_sold": [units.get(month, 0) for month in months],
        }
    )


def test_active_mask_k6_sales_in_months_1_and_3() -> None:
    panel = _hand_built_panel({"2010-01": 5, "2010-03": 7})
    mask = active_mask(panel, k=6).set_index("month")["is_active"]
    assert list(mask.index) == list(panel["month"])
    assert not mask["2010-01"]  # nothing sold before the first month
    for month in ("2010-02", "2010-03", "2010-04", "2010-05", "2010-06", "2010-07", "2010-08"):
        assert mask[month], month
    assert mask["2010-09"]  # month 3 is still inside the six-month window
    for month in ("2010-10", "2010-11", "2010-12"):
        assert not mask[month], month


def test_active_mask_ignores_the_target_month_itself() -> None:
    """Changing the sales of month t may only change months *after* t (§14, §16)."""
    target = "2010-10"
    panel = _hand_built_panel({"2010-01": 5, "2010-03": 7})
    before = active_mask(panel, k=6).set_index("month")["is_active"]
    changed = panel.copy()
    changed.loc[changed["month"] == target, "units_sold"] = 9_999
    after = active_mask(changed, k=6).set_index("month")["is_active"]

    assert not after[target], "a sale in month t must not make month t active"
    up_to_t = [month for month in panel["month"] if month <= target]
    assert before.loc[up_to_t].equals(after.loc[up_to_t]), "no month at or before t may change"
    # The sale is visible from t+1 onwards, for the whole k-month window.
    assert after["2010-11"] and after["2010-12"]
    assert not before["2010-11"] and not before["2010-12"]


def test_active_mask_returns_the_declared_columns_and_is_per_product() -> None:
    panel = pd.concat(
        [
            _hand_built_panel({"2010-01": 5}),
            _hand_built_panel({"2010-07": 4}).assign(stock_code="20002"),
        ],
        ignore_index=True,
    )
    mask = active_mask(panel, k=6)
    assert list(mask.columns) == ["stock_code", "month", "is_active"]
    assert len(mask) == len(panel)
    first = mask[mask["stock_code"] == "10001"].set_index("month")["is_active"]
    second = mask[mask["stock_code"] == "20002"].set_index("month")["is_active"]
    assert first["2010-02"] and not second["2010-02"]  # products never leak into each other
    assert second["2010-08"]


def test_active_mask_k_defaults_to_the_configured_value() -> None:
    panel = _hand_built_panel({"2010-01": 5})
    configured = load_model_config().active_rule.k
    assert active_mask(panel).equals(active_mask(panel, k=configured))
