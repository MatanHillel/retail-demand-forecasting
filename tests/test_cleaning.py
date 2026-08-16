"""Cleaning-pipeline tests (US-04, PRD §9, §10 steps 3-10, §11, §12, §55).

Structure and rules only — the PRD's indicative full-file counts (C −19,494 etc.) are never
asserted. Tests run on tiny hand-built frames plus the committed CI fixture, always against a
``tmp_path`` base directory so nothing real is overwritten.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.cleaning import clean_transactions
from pipeline.config import load_cleaning_config, load_non_inventory_codes
from pipeline.download import load_raw
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

WATERFALL_COLUMNS = [
    "step_no",
    "step",
    "rule",
    "rows_before",
    "rows_removed",
    "rows_after",
    "units_before",
    "units_removed",
    "units_after",
    "revenue_after",
]

CLEAN_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "description_raw",
    "quantity",
    "invoice_date",
    "month",
    "price",
    "customer_id",
    "country",
    "line_revenue",
    "is_partial_month",
    "is_abnormal_qty",
    "source_sheet",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(rows: list[dict]) -> pd.DataFrame:
    """Tiny raw-shaped frame; each dict overrides the defaults of one row."""
    defaults = {
        "Invoice": "500001",
        "StockCode": "10001",
        "Description": "RED MUG",
        "Quantity": 2,
        "InvoiceDate": pd.Timestamp("2010-03-05 10:00:00"),
        "Price": 1.5,
        "Customer ID": 13085.0,
        "Country": "United Kingdom",
        "source_sheet": "Year 2009-2010",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _clean(frame: pd.DataFrame, base_dir: Path):
    """Run the cleaning inside a step against an isolated base directory."""
    ctx = RunContext.start(mode="no-llm", base_dir=base_dir)
    try:
        with ctx.step("cleaning"):
            clean_df, waterfall = clean_transactions(frame, load_cleaning_config(), ctx)
    finally:
        close_log_handlers(ctx.run_id)
    return clean_df, waterfall, ctx


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One cleaning run over the committed CI fixture, shared by the structural tests."""
    base = tmp_path_factory.mktemp("cleaning_sample")
    raw, _ = load_raw(RAW_SAMPLE)
    ctx = RunContext.start(mode="no-llm", base_dir=base)
    try:
        with ctx.step("cleaning"):
            clean_df, waterfall = clean_transactions(raw, load_cleaning_config(), ctx)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    return SimpleNamespace(raw=raw, clean=clean_df, waterfall=waterfall, ctx=ctx, base=base)


# --------------------------------------------------------------------------
# waterfall structure (§10)
# --------------------------------------------------------------------------
def test_waterfall_has_ten_consistent_rows(sample_run: SimpleNamespace) -> None:
    waterfall = sample_run.waterfall
    assert list(waterfall.columns) == WATERFALL_COLUMNS
    assert list(waterfall["step_no"]) == list(range(1, 11))
    # steps 1-2 belong to US-03 and remove nothing
    assert list(waterfall["rows_removed"].iloc[:2]) == [0, 0]
    # chain consistency: what one step leaves is what the next step starts from
    for i in range(len(waterfall) - 1):
        assert waterfall["rows_after"].iloc[i] == waterfall["rows_before"].iloc[i + 1]
        assert waterfall["units_after"].iloc[i] == waterfall["units_before"].iloc[i + 1]
    removed = waterfall["rows_before"] - waterfall["rows_after"]
    assert (waterfall["rows_removed"] == removed).all()
    units_removed = waterfall["units_before"] - waterfall["units_after"]
    assert (waterfall["units_removed"] == units_removed).all()


def test_waterfall_csv_written_and_deterministic(
    sample_run: SimpleNamespace, tmp_path: Path
) -> None:
    relative = (paths.EDA_TABLES_DIR / "E01_cleaning_waterfall.csv").relative_to(
        paths.PROJECT_ROOT
    )
    first = sample_run.base / relative
    assert first.is_file()
    _clean(sample_run.raw, tmp_path)
    second = tmp_path / relative
    assert second.read_bytes() == first.read_bytes()


def test_run_log_row_counts_reach_the_run_context(sample_run: SimpleNamespace) -> None:
    row_counts = sample_run.ctx.steps[0].row_counts
    assert len(row_counts) >= 10, "one log_rows entry per cleaning rule (plus C/A sub-entries)"


# --------------------------------------------------------------------------
# output schema and §9 invariants
# --------------------------------------------------------------------------
def test_clean_output_schema_and_invariants(sample_run: SimpleNamespace) -> None:
    clean = sample_run.clean
    cfg = load_cleaning_config()
    assert list(clean.columns) == CLEAN_COLUMNS
    assert (clean["quantity"] > cfg.rules.drop_quantity_at_or_below).all()
    assert (clean["price"] > cfg.rules.drop_price_at_or_below).all()
    prefixes = tuple([*cfg.rules.cancellation_prefixes, *cfg.rules.adjustment_prefixes])
    assert not clean["invoice"].str.startswith(prefixes).any()
    excluded = set(load_non_inventory_codes()["stock_code"].str.strip().str.upper())
    assert not set(clean["stock_code"]) & excluded
    original_equivalent = [
        "invoice",
        "stock_code",
        "description_raw",
        "quantity",
        "invoice_date",
        "price",
        "customer_id",
        "country",
    ]
    assert clean.duplicated(subset=original_equivalent).sum() == 0
    assert clean["customer_id"].isna().any(), "rows without Customer ID are kept (§9)"
    assert str(clean["customer_id"].dtype) == "Int64"
    partial = set(cfg.raw.partial_months)
    assert (clean["is_partial_month"] == clean["month"].isin(partial)).all()
    assert (clean["line_revenue"] == clean["quantity"] * clean["price"]).all()
    parquet = sample_run.base / "data" / "processed" / "clean_transactions.parquet"
    assert parquet.is_file()


def test_returns_lines_side_table(sample_run: SimpleNamespace) -> None:
    returns_path = sample_run.base / "data" / "processed" / "returns_lines.parquet"
    assert returns_path.is_file()
    returns = pd.read_parquet(returns_path)
    assert list(returns.columns) == ["stock_code", "month", "quantity_abs", "invoice"]
    cancel = tuple(load_cleaning_config().rules.cancellation_prefixes)
    assert returns["invoice"].str.startswith(cancel).all()
    assert (returns["quantity_abs"] >= 0).all()


# --------------------------------------------------------------------------
# individual rules on tiny frames
# --------------------------------------------------------------------------
def test_lowercase_stock_code_merges(tmp_path: Path) -> None:
    frame = _frame(
        [
            {"StockCode": "85123A", "Description": "HANGING HEART"},
            {"StockCode": " 85123a ", "Description": "HANGING HEART", "Invoice": "500002"},
        ]
    )
    clean, _, _ = _clean(frame, tmp_path)
    assert len(clean) == 2
    assert set(clean["stock_code"]) == {"85123A"}


def test_cancellation_and_adjustment_removed_and_returns_kept(tmp_path: Path) -> None:
    rules = load_cleaning_config().rules
    cancel = rules.cancellation_prefixes[0]
    adjust = rules.adjustment_prefixes[0]
    frame = _frame(
        [
            {},
            {"Invoice": f"{cancel}500009", "Quantity": -4},
            {"Invoice": f"{adjust}500010", "Quantity": 1},
        ]
    )
    clean, waterfall, _ = _clean(frame, tmp_path)
    assert len(clean) == 1
    assert waterfall.loc[waterfall["step_no"] == 5, "rows_removed"].item() == 2
    returns = pd.read_parquet(tmp_path / "data" / "processed" / "returns_lines.parquet")
    assert list(returns["invoice"]) == [f"{cancel}500009"]
    assert list(returns["quantity_abs"]) == [4]


def test_non_positive_quantity_and_price_removed(tmp_path: Path) -> None:
    rules = load_cleaning_config().rules
    frame = _frame(
        [
            {},
            {"Quantity": rules.drop_quantity_at_or_below, "Invoice": "500003"},
            {"Price": rules.drop_price_at_or_below, "Invoice": "500004"},
        ]
    )
    clean, waterfall, _ = _clean(frame, tmp_path)
    assert len(clean) == 1
    assert waterfall.loc[waterfall["step_no"] == 6, "rows_removed"].item() == 1
    assert waterfall.loc[waterfall["step_no"] == 7, "rows_removed"].item() == 1


def test_exclusion_list_codes_removed_case_insensitively(tmp_path: Path) -> None:
    code = load_non_inventory_codes()["stock_code"].iloc[0]  # e.g. POST
    frame = _frame(
        [
            {},
            {"StockCode": code, "Invoice": "500005"},
            {"StockCode": code.lower(), "Invoice": "500006"},
        ]
    )
    clean, waterfall, _ = _clean(frame, tmp_path)
    assert len(clean) == 1
    assert waterfall.loc[waterfall["step_no"] == 8, "rows_removed"].item() == 2


def test_exact_duplicate_pair_collapses_to_one_row(tmp_path: Path) -> None:
    frame = _frame([{}, {}, {"Invoice": "500007"}])
    clean, waterfall, ctx = _clean(frame, tmp_path)
    assert len(clean) == 2
    assert waterfall.loc[waterfall["step_no"] == 9, "rows_removed"].item() == 1
    assert any("duplicate" in warning for warning in ctx.warnings)  # 1/3 ≫ threshold


def test_canonical_description_most_frequent_then_alphabetical(tmp_path: Path) -> None:
    frame = _frame(
        [
            {"Description": "ZEBRA MUG"},
            {"Description": "APPLE MUG", "Invoice": "500011"},
            {"Description": "APPLE MUG", "Invoice": "500012"},
            {"StockCode": "20002", "Description": "BLUE", "Invoice": "500013"},
            {"StockCode": "20002", "Description": "AMBER", "Invoice": "500014"},
        ]
    )
    clean, _, _ = _clean(frame, tmp_path)
    first = clean[clean["stock_code"] == "10001"]
    assert set(first["description"]) == {"APPLE MUG"}  # most frequent wins
    assert set(first["description_raw"]) == {"ZEBRA MUG", "APPLE MUG"}
    tie = clean[clean["stock_code"] == "20002"]
    assert set(tie["description"]) == {"AMBER"}  # tie broken alphabetically


def test_abnormal_quantity_flagged_never_removed(tmp_path: Path) -> None:
    threshold = load_cleaning_config().warnings.abnormal_line_quantity
    frame = _frame(
        [
            {},
            {"Quantity": threshold + 1, "Invoice": "500008"},
        ]
    )
    clean, _, _ = _clean(frame, tmp_path)
    assert len(clean) == 2
    flagged = clean[clean["quantity"] == threshold + 1]
    assert flagged["is_abnormal_qty"].all()
    assert not clean[clean["quantity"] != threshold + 1]["is_abnormal_qty"].any()
