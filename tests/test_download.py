"""Raw-data acquisition tests (US-03, PRD §10 steps 1-2, §39, §42).

These tests run without network access and without the full raw file: they exercise the hashing,
schema-validation, loading and sampling logic on small synthetic frames, plus structural checks
on the committed CI fixture ``tests/fixtures/raw_sample.csv``. Indicative PRD numbers (full-file
row counts) are never asserted here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import (
    load_cleaning_config,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.download import (
    SAMPLE_MAX_BYTES,
    SAMPLE_MIN_BAD_PRICE_ROWS,
    SAMPLE_MIN_BAD_QUANTITY_ROWS,
    SAMPLE_MIN_CANCELLATION_ROWS,
    SAMPLE_MIN_DUPLICATE_ROWS,
    SAMPLE_STOCK_CODES,
    compute_sha256,
    load_raw,
    make_sample,
    validate_raw_schema,
    verify_hash,
)

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _month_span() -> pd.PeriodIndex:
    """Every month of the raw period, first_month .. last partial month, from config."""
    raw = load_cleaning_config().raw
    last = max([raw.last_full_month, *raw.partial_months])
    return pd.period_range(raw.first_month, last, freq="M")


def _raw_frame() -> pd.DataFrame:
    """Synthetic raw extract exercising every rule ``make_sample`` must satisfy."""
    rules = load_cleaning_config().rules
    cancel = rules.cancellation_prefixes[0]
    adjust = rules.adjustment_prefixes[0]
    non_inventory_code = load_non_inventory_codes()["stock_code"].iloc[0]

    rows: list[dict] = []
    invoice = 400000

    def add(stock_code: str, month: pd.Period, **overrides) -> None:
        nonlocal invoice
        invoice += 1
        row = {
            "Invoice": str(invoice),
            "StockCode": stock_code,
            "Description": f"PRODUCT {stock_code}",
            "Quantity": 6,
            "InvoiceDate": month.to_timestamp() + pd.Timedelta(hours=9),
            "Price": 2.55,
            "Customer ID": 13085.0,
            "Country": "United Kingdom",
        }
        row.update(overrides)
        rows.append(row)

    months = _month_span()
    for month in months:
        for code in ("10001", "10002", "10003", "10004", "10005", "10006"):
            add(code, month)
    mid = months[len(months) // 2]
    for _ in range(6):  # cancellations
        add("10001", mid)
        rows[-1]["Invoice"] = f"{cancel}{rows[-1]['Invoice']}"
        rows[-1]["Quantity"] = -3
    for _ in range(2):  # bad-debt adjustments
        add("B", mid)
        rows[-1]["Invoice"] = f"{adjust}{rows[-1]['Invoice']}"
    for _ in range(6):  # non-positive price
        add("10002", mid, Price=0.0)
    for _ in range(6):  # non-positive quantity on a regular invoice
        add("10003", mid, Quantity=0)
    for _ in range(3):  # non-inventory code
        add(non_inventory_code, mid, Price=18.0)
    frame = pd.DataFrame(rows)
    # exact duplicates: one row repeated 4x and one repeated 3x -> 5 redundant copies
    frame = pd.concat(
        [frame, frame.iloc[[0, 0, 0, 1, 1]]],
        ignore_index=True,
    )
    return frame


# --------------------------------------------------------------------------
# compute_sha256 / verify_hash
# --------------------------------------------------------------------------
def test_compute_sha256_matches_hashlib_and_is_stable(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"online retail ii")
    expected = hashlib.sha256(b"online retail ii").hexdigest()
    assert compute_sha256(target) == expected
    assert compute_sha256(target) == expected  # stable across calls


def test_verify_hash_passes_on_matching_digest(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"payload")
    result = verify_hash(target, expected=compute_sha256(target))
    assert result.passed
    assert result.violations == []


def test_verify_hash_mismatch_returns_failed_result(tmp_path: Path) -> None:
    target = tmp_path / "blob.bin"
    target.write_bytes(b"payload")
    actual = compute_sha256(target)
    wrong = "0" * 64
    result = verify_hash(target, expected=wrong)
    assert not result.passed
    violation = result.violations[0]
    assert violation.rule == "expected_sha256"
    assert "raw data hash mismatch (expected" in violation.message
    assert wrong in violation.message
    assert actual in violation.message


# --------------------------------------------------------------------------
# validate_raw_schema
# --------------------------------------------------------------------------
def test_validate_raw_schema_passes_on_valid_frame() -> None:
    result = validate_raw_schema(_raw_frame())
    assert result.passed
    assert result.violations == []
    assert result.checked_rows == len(_raw_frame())


def test_validate_raw_schema_missing_column() -> None:
    frame = _raw_frame().drop(columns=["Quantity"])
    result = validate_raw_schema(frame)
    assert not result.passed
    missing = [v for v in result.violations if v.rule == "required_columns"]
    assert missing, "expected a required_columns violation"
    assert "Missing required column Quantity" in missing[0].message


def test_validate_raw_schema_rejects_non_numeric_quantity() -> None:
    frame = _raw_frame()
    frame["Quantity"] = frame["Quantity"].astype(object)
    frame.loc[frame.index[:3], "Quantity"] = "not-a-number"
    result = validate_raw_schema(frame)
    assert not result.passed
    assert any("Quantity" in violation.message for violation in result.violations)


def test_validate_raw_schema_rejects_null_invoice() -> None:
    frame = _raw_frame()
    frame.loc[frame.index[:2], "Invoice"] = None
    result = validate_raw_schema(frame)
    assert not result.passed
    assert any("Invoice" in violation.message for violation in result.violations)


# --------------------------------------------------------------------------
# load_raw
# --------------------------------------------------------------------------
def test_load_raw_csv_returns_canonical_columns(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    _raw_frame().to_csv(source, index=False)
    frame, meta = load_raw(source)
    required = load_cleaning_config().raw.required_columns
    assert list(frame.columns) == [*required, "source_sheet"]
    assert pd.api.types.is_datetime64_any_dtype(frame["InvoiceDate"])
    assert frame["source_sheet"].notna().all()
    assert meta["rows"] == len(frame)
    assert meta["sha256"] == compute_sha256(source)
    assert set(meta) == {"file", "sha256", "rows", "columns", "rows_per_sheet"}


def test_load_raw_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.csv")


# --------------------------------------------------------------------------
# make_sample
# --------------------------------------------------------------------------
def test_make_sample_is_deterministic_and_meets_targets(tmp_path: Path) -> None:
    frame = _raw_frame()
    rules = load_cleaning_config().rules
    out = tmp_path / "sample.csv"
    kwargs = dict(
        out=out,
        n_stock_codes=3,
        min_cancellation_rows=4,
        min_bad_price_rows=4,
        min_bad_quantity_rows=4,
        min_duplicate_rows=3,
    )
    make_sample(frame, **kwargs)
    first = out.read_bytes()
    make_sample(frame, **kwargs)
    assert out.read_bytes() == first, "make_sample must be byte-identical across runs"

    sample = pd.read_csv(out, dtype={"Invoice": str, "StockCode": str})
    sample["InvoiceDate"] = pd.to_datetime(sample["InvoiceDate"], format="ISO8601")

    months = sample["InvoiceDate"].dt.to_period("M")
    assert set(months) == set(_month_span()), "every raw month must appear in the sample"

    cancel = tuple(rules.cancellation_prefixes)
    adjust = tuple(rules.adjustment_prefixes)
    assert sample["Invoice"].str.startswith(cancel).sum() >= 4
    raw_adjust = frame["Invoice"].astype(str).str.startswith(adjust).sum()
    assert sample["Invoice"].str.startswith(adjust).sum() == raw_adjust  # all A rows kept
    assert (sample["Price"] <= rules.drop_price_at_or_below).sum() >= 4
    assert (sample["Quantity"] <= rules.drop_quantity_at_or_below).sum() >= 4

    required = load_cleaning_config().raw.required_columns
    assert sample.duplicated(subset=required, keep="first").sum() >= 3

    non_inventory = set(load_non_inventory_codes()["stock_code"])
    assert non_inventory & set(sample["StockCode"]), "non-inventory codes must be sampled"


def test_make_sample_uses_config_seed_by_default(tmp_path: Path) -> None:
    frame = _raw_frame()
    out_default = tmp_path / "default.csv"
    out_explicit = tmp_path / "explicit.csv"
    make_sample(frame, out=out_default, n_stock_codes=3)
    make_sample(frame, out=out_explicit, seed=load_model_config().seed, n_stock_codes=3)
    assert out_default.read_bytes() == out_explicit.read_bytes()


# --------------------------------------------------------------------------
# the committed CI fixture (created by `python -m pipeline.download --make-sample`)
# --------------------------------------------------------------------------
def test_fixture_exists_within_size_budget() -> None:
    assert RAW_SAMPLE.is_file(), "run `python -m pipeline.download --make-sample` and commit it"
    assert RAW_SAMPLE.stat().st_size <= SAMPLE_MAX_BYTES


def test_fixture_passes_schema_and_canonical_columns() -> None:
    frame, meta = load_raw(RAW_SAMPLE)
    required = load_cleaning_config().raw.required_columns
    assert list(frame.columns) == [*required, "source_sheet"]
    assert meta["rows"] == len(frame)
    result = validate_raw_schema(frame)
    assert result.passed, f"fixture must satisfy the raw schema: {result.summary()}"


def test_fixture_covers_every_month_and_row_class() -> None:
    frame, _ = load_raw(RAW_SAMPLE)
    rules = load_cleaning_config().rules

    months = frame["InvoiceDate"].dt.to_period("M")
    assert set(months) == set(_month_span()), "fixture must cover every raw month"

    invoices = frame["Invoice"].astype(str)
    assert invoices.str.startswith(tuple(rules.cancellation_prefixes)).sum() >= (
        SAMPLE_MIN_CANCELLATION_ROWS
    )
    assert invoices.str.startswith(tuple(rules.adjustment_prefixes)).sum() >= 1
    assert (frame["Price"] <= rules.drop_price_at_or_below).sum() >= SAMPLE_MIN_BAD_PRICE_ROWS
    assert (frame["Quantity"] <= rules.drop_quantity_at_or_below).sum() >= (
        SAMPLE_MIN_BAD_QUANTITY_ROWS
    )

    required = load_cleaning_config().raw.required_columns
    assert frame.duplicated(subset=required, keep="first").sum() >= SAMPLE_MIN_DUPLICATE_ROWS

    non_inventory = set(load_non_inventory_codes()["stock_code"])
    assert non_inventory & set(frame["StockCode"]), "fixture must include non-inventory codes"

    assert frame["StockCode"].nunique() >= SAMPLE_STOCK_CODES, (
        "fixture must include the sampled stock codes plus the special-row classes"
    )
