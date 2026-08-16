"""Cleaning pipeline — PRD §10 steps 3-10, in that exact order (US-04).

Implements the gross-demand definition of §9: cancellations (``C`` invoices) and bad-debt
adjustments (``A``) are removed from the sales table but **never subtracted** from demand; rows
without ``Customer ID`` are **kept**. Every rule and threshold comes from
``config/cleaning_config.yaml`` — no number lives in this file (§40).

Each step appends one row to a row-count waterfall (rows / units before, removed, after and
revenue after), which is written to ``artifacts/reports/eda_tables/E01_cleaning_waterfall.csv``
and mirrored into the run log via ``ctx.log_rows`` (§10 "logged as a row-count waterfall").
The removed cancellation lines are kept as ``data/processed/returns_lines.parquet`` because
§13.2 needs ``returned_units`` per product-month for EDA — never as a model feature.

All writes go through ``ctx.out(...)`` so the §39 staging guarantee holds when the Flow drives
this step; output paths are resolved against ``ctx.base_dir`` so tests can redirect them.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd

from pipeline import paths
from pipeline.config import CleaningConfig, load_cleaning_config, load_non_inventory_codes
from pipeline.download import load_raw
from pipeline.run_context import RunContext

WATERFALL_FILENAME = "E01_cleaning_waterfall.csv"
RETURNS_FILENAME = "returns_lines.parquet"

_MONTH_FORMAT = "%Y-%m"


class _Stats(NamedTuple):
    """Snapshot of a frame between steps: row count, total units, total revenue."""

    rows: int
    units: int
    revenue: float


def _stats(df: pd.DataFrame) -> _Stats:
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    price = pd.to_numeric(df["Price"], errors="coerce")
    return _Stats(
        rows=int(len(df)),
        units=int(quantity.sum()) if len(df) else 0,
        revenue=float((quantity * price).sum()) if len(df) else 0.0,
    )


class Waterfall:
    """Accumulates the §10 row-count waterfall, one row per cleaning step."""

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def add(self, step_no: int, step: str, rule: str, before: _Stats, after: _Stats) -> None:
        self._rows.append(
            {
                "step_no": step_no,
                "step": step,
                "rule": rule,
                "rows_before": before.rows,
                "rows_removed": before.rows - after.rows,
                "rows_after": after.rows,
                "units_before": before.units,
                "units_removed": before.units - after.units,
                "units_after": after.units,
                "revenue_after": round(after.revenue, 2),
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)


def _resolve_out(ctx: RunContext, canonical: Path) -> Path:
    """Register a canonical ``paths.*`` location with ``ctx.out``, honouring ``base_dir``.

    In production ``base_dir == PROJECT_ROOT`` and this is exactly ``ctx.out(canonical)``;
    under a test ``base_dir`` (or staging) the write lands inside that directory instead of
    on the real artifacts.
    """
    return ctx.out(ctx.base_dir / canonical.relative_to(paths.PROJECT_ROOT))


def clean_transactions(
    raw_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply §10 steps 3-10 to the raw extract; return ``(clean_df, waterfall_df)``.

    Must be called inside an open ``ctx.step(...)`` block — every step is recorded through
    ``ctx.log_rows``, which requires one. Writes ``clean_transactions.parquet``,
    ``returns_lines.parquet`` and the waterfall CSV through ``ctx.out(...)``.
    """
    original_columns = list(cfg.raw.required_columns)
    waterfall = Waterfall()

    def record(step_no: int, step: str, rule: str, before: _Stats, after: _Stats) -> None:
        waterfall.add(step_no, step, rule, before, after)
        ctx.log_rows(f"{step_no:02d}_{step}", before.rows, before.rows - after.rows, after.rows)

    df = raw_df.copy()

    # Steps 1-2 (dataset intake, raw schema validation) are US-03's; they remove nothing and
    # appear here so the waterfall documents the whole §10 chain.
    intake = _stats(df)
    record(1, "dataset_intake", "load_raw", intake, intake)
    record(2, "raw_schema_validation", "required_columns", intake, intake)

    # Step 3 — normalise keys.
    before = _stats(df)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string").str.strip().replace("", pd.NA)
    record(3, "normalise_keys", "strip_upper_stockcode", before, _stats(df))

    # Step 4 — convert dates, derive month and the partial-month flag.
    before = _stats(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["month"] = df["InvoiceDate"].dt.strftime(_MONTH_FORMAT)
    df["is_partial_month"] = df["month"].isin(cfg.raw.partial_months)
    record(4, "convert_dates", "month_and_partial_flag", before, _stats(df))

    # Step 5 — remove cancellations (C) and adjustments (A); C and A logged separately.
    before = _stats(df)
    cancel_mask = df["Invoice"].str.startswith(tuple(cfg.rules.cancellation_prefixes))
    adjust_mask = df["Invoice"].str.startswith(tuple(cfg.rules.adjustment_prefixes))
    cancellations = int(cancel_mask.sum())
    adjustments = int(adjust_mask.sum())
    returns_lines = df.loc[cancel_mask, ["StockCode", "month", "Quantity", "Invoice"]]
    df = df[~(cancel_mask | adjust_mask)]
    after = _stats(df)
    waterfall.add(
        5,
        "remove_cancellations_adjustments",
        "cancellation_prefixes,adjustment_prefixes",
        before,
        after,
    )
    ctx.log_rows("05a_cancellations", before.rows, cancellations, before.rows - cancellations)
    ctx.log_rows("05b_adjustments", before.rows - cancellations, adjustments, after.rows)
    ctx.record_metrics(
        {"cancellation_rows_removed": cancellations, "adjustment_rows_removed": adjustments}
    )

    # Step 6 — remove non-positive quantities (those remaining after step 5).
    before = _stats(df)
    df = df[pd.to_numeric(df["Quantity"]) > cfg.rules.drop_quantity_at_or_below]
    record(6, "remove_nonpositive_quantity", "drop_quantity_at_or_below", before, _stats(df))

    # Step 7 — remove non-positive prices (§9: such lines are not demand).
    before = _stats(df)
    df = df[pd.to_numeric(df["Price"]) > cfg.rules.drop_price_at_or_below]
    record(7, "remove_nonpositive_price", "drop_price_at_or_below", before, _stats(df))

    # Step 8 — remove non-inventory stock codes, compared after the same normalisation.
    before = _stats(df)
    excluded = {
        code.strip().upper() for code in load_non_inventory_codes()["stock_code"]
    }
    df = df[~df["StockCode"].isin(excluded)]
    record(8, "remove_non_inventory_codes", "non_inventory_file", before, _stats(df))

    # Step 9 — remove exact duplicates (all 8 original columns identical, §11).
    before = _stats(df)
    df = df[~df.duplicated(subset=original_columns, keep="first")]
    after = _stats(df)
    record(9, "remove_exact_duplicates", "duplicate_policy", before, after)
    if before.rows:
        row_share = (before.rows - after.rows) / before.rows
        units_share = (before.units - after.units) / before.units if before.units else 0.0
        ctx.record_metrics(
            {"dup_row_share": round(row_share, 6), "dup_units_share": round(units_share, 6)}
        )
        if row_share > cfg.warnings.duplicate_max_row_share:
            ctx.warn(
                f"exact duplicate rows are {row_share:.4%} of rows, above the "
                f"{cfg.warnings.duplicate_max_row_share:.4%} threshold (§11)"
            )
        if units_share > cfg.warnings.duplicate_max_units_share:
            ctx.warn(
                f"exact duplicate units are {units_share:.4%} of units, above the "
                f"{cfg.warnings.duplicate_max_units_share:.4%} threshold (§11)"
            )

    # Step 10 — canonical description per StockCode: most frequent non-empty, ties → first
    # alphabetically. The original (stripped) text is kept as description_raw.
    before = _stats(df)
    counts = (
        df.dropna(subset=["Description"])
        .groupby(["StockCode", "Description"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(
            ["StockCode", "n", "Description"],
            ascending=[True, False, True],
            kind="mergesort",
        )
    )
    canonical = counts.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    df = df.assign(description=df["StockCode"].map(canonical))
    record(10, "canonical_descriptions", "most_frequent_nonempty", before, _stats(df))

    # Abnormal quantities are flagged, never removed (§10).
    abnormal = pd.to_numeric(df["Quantity"]) > cfg.warnings.abnormal_line_quantity
    abnormal_count = int(abnormal.sum())
    ctx.record_metrics({"abnormal_qty_rows": abnormal_count})
    ctx.logger.info(
        f"abnormal-quantity lines flagged (never removed): {abnormal_count} "
        f"above {cfg.warnings.abnormal_line_quantity}"
    )

    quantity = pd.to_numeric(df["Quantity"])
    price = pd.to_numeric(df["Price"])
    clean = pd.DataFrame(
        {
            "invoice": df["Invoice"],
            "stock_code": df["StockCode"],
            "description": df["description"].astype("string"),
            "description_raw": df["Description"].astype("string"),
            "quantity": quantity,
            "invoice_date": df["InvoiceDate"],
            "month": df["month"],
            "price": price,
            "customer_id": df["Customer ID"].astype("Int64"),
            "country": df["Country"],
            "line_revenue": quantity * price,
            "is_partial_month": df["is_partial_month"],
            "is_abnormal_qty": abnormal,
            "source_sheet": df["source_sheet"],
        }
    )
    clean = clean.sort_values(
        ["invoice_date", "invoice", "stock_code", "quantity", "price"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    returns = pd.DataFrame(
        {
            "stock_code": returns_lines["StockCode"],
            "month": returns_lines["month"],
            "quantity_abs": pd.to_numeric(returns_lines["Quantity"]).abs(),
            "invoice": returns_lines["Invoice"],
        }
    )
    returns = returns.sort_values(
        ["stock_code", "month", "invoice", "quantity_abs"], kind="mergesort"
    ).reset_index(drop=True)

    waterfall_df = waterfall.to_frame()

    clean.to_parquet(_resolve_out(ctx, paths.CLEAN_TRANSACTIONS), compression="snappy")
    returns.to_parquet(
        _resolve_out(ctx, paths.PROCESSED_DIR / RETURNS_FILENAME), compression="snappy"
    )
    waterfall_df.to_csv(
        _resolve_out(ctx, paths.EDA_TABLES_DIR / WATERFALL_FILENAME),
        index=False,
        float_format="%.2f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("clean_transactions", paths.CLEAN_TRANSACTIONS)
    ctx.record_artifact("returns_lines", paths.PROCESSED_DIR / RETURNS_FILENAME)
    ctx.record_artifact("cleaning_waterfall", paths.EDA_TABLES_DIR / WATERFALL_FILENAME)
    ctx.logger.info(
        f"cleaning finished: {waterfall_df['rows_after'].iloc[-1]} sales rows, "
        f"revenue {waterfall_df['revenue_after'].iloc[-1]:,.2f}"
    )
    return clean, waterfall_df


def run() -> int:
    """Standalone entry point: load the raw file, clean it, print the waterfall (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("cleaning"):
            raw_df, meta = load_raw(ctx=ctx)
            clean_df, waterfall_df = clean_transactions(raw_df, load_cleaning_config(), ctx)
        print(waterfall_df.to_string(index=False))
        print(f"clean rows: {len(clean_df)} (from {meta['rows']} raw rows)")
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
