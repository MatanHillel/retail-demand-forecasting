"""Product × Month panel — the hand-off table ``clean_data.csv`` (PRD §10 steps 11-12, §13.2).

Aggregates the cleaned sales lines of US-04 to ``(stock_code, month)`` grain and zero-fills the
months in which a product sold nothing, so that every product has one contiguous row per month
from its first observed sale to the partial month (2011-12, flagged). ``units_sold`` is gross
demand (§9): returns are reported alongside as ``returned_units`` and never subtracted, and they
— like ``customer_count`` — are EDA diagnostics, never model features (§13.2, §17).

``clean_data.csv`` is one of the eight required artifacts (§41) and is committed, so the write is
deterministic: fixed column order, fixed sort, fixed float format.

Panel invariants live in the pure :func:`validate_panel`, which returns a
:class:`~pipeline.validation.ValidationResult`; the caller writes the report and decides whether
to stop (``docs/interfaces.md`` §5). Every month boundary comes from ``cleaning_config.yaml`` —
no month string is written here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME
from pipeline.config import CleaningConfig, load_cleaning_config
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: Column order of ``clean_data.csv`` (PRD §13.2) — a published schema, extend but never rename.
PANEL_COLUMNS: list[str] = [
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

_COUNT_COLUMNS = [
    "units_sold",
    "invoice_count",
    "sale_line_count",
    "customer_count",
    "max_line_qty",
    "returned_units",
]
_NON_NEGATIVE_COLUMNS = [*_COUNT_COLUMNS, "gross_revenue", "avg_unit_price"]
_EXAMPLE_LIMIT = 5


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant.

    ``ctx.out()`` rebases a *relative* path onto the run's base directory, and
    ``ctx.record_artifact()`` stores a repo-relative string — so passing the relative form is
    correct in production, under a test ``base_dir`` and under staging alike.
    """
    return path.relative_to(paths.PROJECT_ROOT)


def _month_ordinals(months: pd.Series) -> pd.Series:
    """Months as consecutive integers, so gaps and spans are plain arithmetic."""
    periods = pd.PeriodIndex(months.astype(str), freq="M")
    return pd.Series(periods.year * 12 + periods.month, index=months.index)


def build_panel(
    clean_df: pd.DataFrame,
    returns_lines: pd.DataFrame,
    cfg: CleaningConfig,
    ctx: RunContext,
) -> pd.DataFrame:
    """Aggregate and zero-fill the cleaned sales lines into the §13.2 panel.

    Must be called inside an open ``ctx.step(...)`` block (``ctx.log_rows`` requires one).
    Writes ``data/processed/clean_data.csv`` through ``ctx.out(...)`` and registers it on the run.
    """
    sales = clean_df.copy()
    sales["month"] = sales["month"].astype(str)
    sales["stock_code"] = sales["stock_code"].astype(str)
    sales["_revenue"] = pd.to_numeric(sales["quantity"]) * pd.to_numeric(sales["price"])

    # -- step 11: aggregate the positive sales lines ------------------------
    aggregated = (
        sales.groupby(["stock_code", "month"], sort=True)
        .agg(
            units_sold=("quantity", "sum"),
            gross_revenue=("_revenue", "sum"),
            invoice_count=("invoice", "nunique"),
            sale_line_count=("invoice", "size"),
            customer_count=("customer_id", "nunique"),  # missing ids are skipped, rows are kept
            max_line_qty=("quantity", "max"),
        )
        .reset_index()
    )
    nonzero_rows = len(aggregated)

    if nonzero_rows == 0:
        panel = pd.DataFrame({column: [] for column in PANEL_COLUMNS})
    else:
        panel = _zero_fill(aggregated, cfg)
        panel = _attach_returns(panel, returns_lines, ctx)
        canonical = (
            sales.dropna(subset=["description"])
            .groupby("stock_code")["description"]
            .first()
        )
        panel["description"] = panel["stock_code"].map(canonical).astype("string")
        panel["is_partial_month"] = panel["month"].isin(cfg.raw.partial_months)

    panel = _finalise_types(panel)
    panel = panel[PANEL_COLUMNS].sort_values(
        ["stock_code", "month"], kind="mergesort"
    ).reset_index(drop=True)

    destination = ctx.out(_repo_relative(paths.CLEAN_DATA))
    panel.to_csv(
        destination,
        index=False,
        float_format="%.4f",
        lineterminator="\n",
        encoding="utf-8",
    )
    ctx.record_artifact("clean_data", _repo_relative(paths.CLEAN_DATA))

    zero_filled = int(len(panel) - nonzero_rows)
    partial_rows = int(panel["is_partial_month"].sum()) if len(panel) else 0
    ctx.log_rows("panel_zero_fill", before=nonzero_rows, removed=0, after=int(len(panel)))
    ctx.record_metrics(
        {
            "panel_rows": int(len(panel)),
            "panel_products": int(panel["stock_code"].nunique()),
            "panel_nonzero_rows": nonzero_rows,
            "panel_zero_filled_rows": zero_filled,
            "panel_partial_rows": partial_rows,
            "panel_zero_share": round(zero_filled / len(panel), 6) if len(panel) else 0.0,
        }
    )
    ctx.logger.info(
        f"panel built: {len(panel)} rows, {panel['stock_code'].nunique()} products, "
        f"{zero_filled} zero-filled ({zero_filled / len(panel):.1%} of rows) "
        f"and {partial_rows} partial-month rows"
        if len(panel)
        else "panel built: no sales rows"
    )
    return panel


def _zero_fill(aggregated: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Expand to one row per product per month, first observed sale → last configured month."""
    first_sale = aggregated.groupby("stock_code")["month"].min()
    configured_end = max([cfg.raw.last_full_month, *cfg.raw.partial_months])
    # Never shrink the range below what the data holds: a month after the configured end is a
    # configuration problem, reported by validate_panel — it must not silently drop sales.
    end = max(configured_end, aggregated["month"].max())
    months = [str(period) for period in pd.period_range(first_sale.min(), end, freq="M")]

    grid = pd.MultiIndex.from_product(
        [first_sale.index, months], names=["stock_code", "month"]
    ).to_frame(index=False)
    grid = grid[grid["month"] >= grid["stock_code"].map(first_sale)]

    panel = grid.merge(aggregated, on=["stock_code", "month"], how="left")
    for column in ("units_sold", "gross_revenue", "invoice_count", "sale_line_count",
                   "customer_count", "max_line_qty"):
        panel[column] = panel[column].fillna(0)

    # avg_unit_price is revenue-weighted where there were sales, and the last known price in a
    # zero month (§13.2). The first row of a product is a sale by construction, so the
    # forward fill always has a value to carry.
    panel = panel.sort_values(["stock_code", "month"], kind="mergesort").reset_index(drop=True)
    priced = np.where(
        panel["units_sold"] > 0,
        panel["gross_revenue"] / panel["units_sold"].where(panel["units_sold"] > 0),
        np.nan,
    )
    panel["avg_unit_price"] = (
        pd.Series(priced, index=panel.index)
        .groupby(panel["stock_code"], sort=False)
        .ffill()
        .fillna(0.0)
    )
    return panel


def _attach_returns(
    panel: pd.DataFrame, returns_lines: pd.DataFrame, ctx: RunContext
) -> pd.DataFrame:
    """Join ``returned_units`` (EDA only, §13.2) onto the panel, filling months with none."""
    if returns_lines is None or returns_lines.empty:
        panel["returned_units"] = 0
        return panel
    returns = (
        returns_lines.assign(
            stock_code=returns_lines["stock_code"].astype(str),
            month=returns_lines["month"].astype(str),
        )
        .groupby(["stock_code", "month"], sort=True)["quantity_abs"]
        .sum()
        .reset_index(name="returned_units")
    )
    merged = panel.merge(returns, on=["stock_code", "month"], how="left")
    matched = merged["returned_units"].notna().sum()
    merged["returned_units"] = merged["returned_units"].fillna(0)

    orphans = len(returns) - int(matched)
    if orphans > 0:
        # Cancellations for a product-month with no panel row: a return before the product's
        # first sale, or for a code that never sold. Diagnostics only — never demand (§9).
        ctx.warn(
            f"{orphans} product-month(s) of returns have no panel row (returns before the "
            f"first sale, or products that never sold) and are not represented in clean_data.csv"
        )
    ctx.record_metrics({"returns_without_panel_row": orphans})
    return merged


def _finalise_types(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the §13.2 dtypes so the committed CSV is stable across runs and machines."""
    if panel.empty:
        typed = panel.copy()
        for column in _COUNT_COLUMNS:
            typed[column] = typed[column].astype("int64")
        for column in ("gross_revenue", "avg_unit_price"):
            typed[column] = typed[column].astype("float64")
        typed["is_partial_month"] = typed["is_partial_month"].astype(bool)
        typed["month"] = typed["month"].astype(str)
        typed["stock_code"] = typed["stock_code"].astype(str)
        typed["description"] = typed["description"].astype("string")
        return typed
    typed = panel.copy()
    for column in _COUNT_COLUMNS:
        typed[column] = pd.to_numeric(typed[column]).round().astype("int64")
    for column in ("gross_revenue", "avg_unit_price"):
        typed[column] = pd.to_numeric(typed[column]).astype("float64")
    typed["is_partial_month"] = typed["is_partial_month"].astype(bool)
    typed["month"] = typed["month"].astype(str)
    typed["stock_code"] = typed["stock_code"].astype(str)
    typed["description"] = typed["description"].astype("string")
    return typed


def validate_panel(panel: pd.DataFrame, cfg: CleaningConfig) -> ValidationResult:
    """Check the §13.2/§6 panel invariants — pure: no ``ctx``, no disk, never raises."""
    step = "panel"
    violations: list[Violation] = []

    missing = [column for column in PANEL_COLUMNS if column not in panel.columns]
    if missing:
        violations.append(
            Violation(
                step=step,
                rule="schema",
                message=f"clean_data.csv is missing column(s) {', '.join(missing)}",
                count=len(missing),
            )
        )
        return ValidationResult(
            step=step, passed=False, violations=violations, checked_rows=int(len(panel))
        )

    duplicates = panel.duplicated(subset=["stock_code", "month"])
    if duplicates.any():
        examples = (
            panel.loc[duplicates, ["stock_code", "month"]]
            .head(_EXAMPLE_LIMIT)
            .to_dict("records")
        )
        violations.append(
            Violation(
                step=step,
                rule="primary_key",
                message="(stock_code, month) must be unique",
                count=int(duplicates.sum()),
                examples=examples,
            )
        )

    for column in _NON_NEGATIVE_COLUMNS:
        negative = pd.to_numeric(panel[column], errors="coerce") < 0
        if negative.any():
            violations.append(
                Violation(
                    step=step,
                    rule="non_negative",
                    message=f"{column} must never be negative",
                    count=int(negative.sum()),
                    examples=panel.loc[negative, "stock_code"].head(_EXAMPLE_LIMIT).tolist(),
                )
            )

    expected_partial = panel["month"].isin(cfg.raw.partial_months)
    wrong_flag = panel["is_partial_month"].astype(bool) != expected_partial
    if wrong_flag.any():
        violations.append(
            Violation(
                step=step,
                rule="is_partial_month",
                message=(
                    "is_partial_month must be true exactly for the configured partial months "
                    f"({', '.join(cfg.raw.partial_months)})"
                ),
                count=int(wrong_flag.sum()),
                examples=panel.loc[wrong_flag, "month"].head(_EXAMPLE_LIMIT).unique().tolist(),
            )
        )

    configured_end = max([cfg.raw.last_full_month, *cfg.raw.partial_months])
    beyond = panel["month"] > configured_end
    if beyond.any():
        violations.append(
            Violation(
                step=step,
                rule="month_range",
                message=f"months after the configured last month {configured_end} are present",
                count=int(beyond.sum()),
                examples=sorted(panel.loc[beyond, "month"].unique())[:_EXAMPLE_LIMIT],
            )
        )

    if not panel.empty:
        ordered = panel.sort_values(["stock_code", "month"], kind="mergesort")
        first_rows = ordered.groupby("stock_code", sort=False).head(1)
        not_a_sale = pd.to_numeric(first_rows["units_sold"], errors="coerce") <= 0
        if not_a_sale.any():
            violations.append(
                Violation(
                    step=step,
                    rule="first_row_is_a_sale",
                    message="a product's first month must be its first observed sale",
                    count=int(not_a_sale.sum()),
                    examples=first_rows.loc[not_a_sale, "stock_code"]
                    .head(_EXAMPLE_LIMIT)
                    .tolist(),
                )
            )

        ordinals = _month_ordinals(panel["month"])
        spans = (
            pd.DataFrame({"stock_code": panel["stock_code"], "ordinal": ordinals})
            .groupby("stock_code")["ordinal"]
            .agg(["min", "max", "nunique"])
        )
        gaps = spans[(spans["max"] - spans["min"] + 1) != spans["nunique"]]
        if not gaps.empty:
            violations.append(
                Violation(
                    step=step,
                    rule="contiguous_months",
                    message="each product must have one row per month with no gap",
                    count=int(len(gaps)),
                    examples=gaps.index[:_EXAMPLE_LIMIT].tolist(),
                )
            )
        short = spans[spans["max"] != spans["max"].max()]
        if not short.empty:
            violations.append(
                Violation(
                    step=step,
                    rule="panel_end",
                    message="every product must be zero-filled through the last panel month",
                    count=int(len(short)),
                    examples=short.index[:_EXAMPLE_LIMIT].tolist(),
                )
            )

    return ValidationResult(
        step=step,
        passed=not violations,
        violations=violations,
        checked_rows=int(len(panel)),
    )


def run() -> int:
    """Standalone entry point: read the US-04 outputs, build and validate the panel (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("build_panel"):
            transactions = paths.CLEAN_TRANSACTIONS
            returns_path = paths.PROCESSED_DIR / RETURNS_FILENAME
            for required in (transactions, returns_path):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"{required} not found. Run `python -m pipeline.cleaning` first."
                    )
            cfg = load_cleaning_config()
            panel = build_panel(
                pd.read_parquet(transactions), pd.read_parquet(returns_path), cfg, ctx
            )
            result = validate_panel(panel, cfg)
            write_validation_report(result, run_id=ctx.run_id)
            if not result.passed:
                raise FlowValidationError(result)
        zero_filled = ctx.metrics["panel_zero_filled_rows"]
        print(
            f"panel rows: {ctx.metrics['panel_rows']}\n"
            f"products: {ctx.metrics['panel_products']}\n"
            f"non-zero product-months: {ctx.metrics['panel_nonzero_rows']}\n"
            f"zero-filled rows: {zero_filled} ({ctx.metrics['panel_zero_share']:.2%})\n"
            f"partial-month rows: {ctx.metrics['panel_partial_rows']}"
        )
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
