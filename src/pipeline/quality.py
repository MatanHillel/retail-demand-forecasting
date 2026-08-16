"""Data-quality profiling — the Data Quality Analyst's tool set (US-07, PRD §35).

Everything here **describes** the data; nothing removes a row. Removal is US-04's job
(:mod:`pipeline.cleaning`), and keeping the two apart is the point: the profile is what
justifies each cleaning rule, so it has to be measured on the raw extract *before* the rule ran.

Three profiling tools (§2 of the issue), all pure functions of a DataFrame:

* :func:`profile_raw` — shape, dtypes, missing values, cancellations and adjustments, non-positive
  quantities and prices, lower-case stock-code variants, rows per month, the largest single line
  and the weekday distribution.
* :func:`detect_duplicates` — exact duplicates over the eight original columns, against the §11
  warning thresholds.
* :func:`list_nonproduct_codes` — every stock code that does not look like a product code.

Plus two functions that touch disk and therefore take ``ctx`` (``docs/interfaces.md`` §6 rule 5):
:func:`confirm_exclusion_list`, which writes the reviewed ``config/non_inventory_stockcodes.csv``,
and :func:`build_data_quality_findings`, which assembles ``data_quality_findings.json``.

No number from the PRD is written here (§40): every threshold comes from ``cleaning_config.yaml``
and every count is computed. The indicative figures in PRD §8 are expectations to compare the
output against, never literals in code and never assertions in tests.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import paths
from pipeline.config import CleaningConfig, load_cleaning_config, load_non_inventory_codes
from pipeline.download import load_raw
from pipeline.run_context import RunContext

#: How many of the most-duplicated lines are listed in the duplicates summary (§2).
TOP_DUPLICATED_LINES = 10

#: How many abnormal lines are echoed into the findings JSON. The full list always goes to
#: ``E01_abnormal_lines.csv`` — the JSON is a summary the app and the LLM reviewer read.
TOP_ABNORMAL_LINES = 20

#: The three ``status`` values ``config/non_inventory_stockcodes.csv`` may carry.
STATUS_INITIAL = "initial"
STATUS_CONFIRMED = "confirmed"
STATUS_REVIEW_NEEDED = "review_needed"

#: Column contract of the exclusion list. ``load_non_inventory_codes()`` rejects anything else.
EXCLUSION_COLUMNS = ["stock_code", "reason", "status"]

#: Reason recorded for a code the review step discovered in the data but not on the list. It is
#: flagged for a human, never auto-excluded (§12).
REVIEW_REASON = "does not match inventory_code_pattern; found by the US-07 review — not excluded"

_MONTH_FORMAT = "%Y-%m"

#: Weekday order used for the distribution table: Monday first, as ``dt.dayofweek`` numbers them.
_WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cfg(cfg: CleaningConfig | None) -> CleaningConfig:
    """Resolve the cleaning configuration, so callers may omit it (as ``§2`` writes the calls)."""
    return load_cleaning_config() if cfg is None else cfg


def _normalise_code(codes: pd.Series) -> pd.Series:
    """Key normalisation of PRD §10 step 3 — the same one :mod:`pipeline.cleaning` applies."""
    return codes.astype(str).str.strip().str.upper()


def _invoices(raw_df: pd.DataFrame) -> pd.Series:
    return raw_df["Invoice"].astype(str).str.strip()


def _quantity(raw_df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(raw_df["Quantity"], errors="coerce")


def _price(raw_df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(raw_df["Price"], errors="coerce")


def _share(part: float, whole: float) -> float:
    """Share of ``whole``, or 0.0 when there is nothing to divide by."""
    return float(part) / float(whole) if whole else 0.0


def _json_ready(value: Any) -> Any:
    """Convert NumPy and pandas scalars into plain Python so ``json.dumps`` accepts them.

    Every aggregate pandas returns is a ``np.int64`` / ``np.float64`` / ``np.bool_``, none of
    which the JSON encoder handles — without this the whole step fails on its very last line,
    after all the work is done.
    """
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating | float):
        return None if pd.isna(value) else float(value)
    if isinstance(value, pd.Timestamp | datetime):
        return value.isoformat()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, str | int):
        return value
    if pd.isna(value):
        return None
    return str(value)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records of a frame, preserving its row and column order."""
    return [_json_ready(record) for record in frame.to_dict(orient="records")]


def _source_sheet_profile(
    raw_df: pd.DataFrame, months: pd.Series, invoices: pd.Series
) -> dict[str, Any]:
    """Describe the source sheets and quantify any overlap between them.

    The UCI workbook ships two sheets whose coverage is **not** disjoint, so concatenating them
    double-counts the shared period. That is invisible in a whole-file duplicate count — it looks
    like ordinary duplication — which is why it is measured separately here.
    """
    if "source_sheet" not in raw_df.columns:
        return {}
    sheet = raw_df["source_sheet"].astype(str)

    per_sheet = (
        pd.DataFrame({"source_sheet": sheet, "month": months})
        .groupby("source_sheet", sort=True)
        .agg(rows=("month", "size"), first_month=("month", "min"), last_month=("month", "max"))
        .reset_index()
    )

    by_month = pd.crosstab(months, sheet)
    overlapping = by_month.index[(by_month > 0).sum(axis=1) > 1].tolist()
    overlap_rows = (
        by_month.loc[overlapping].reset_index().rename(columns={by_month.index.name or 0: "month"})
        if overlapping
        else pd.DataFrame()
    )

    shared_invoices = 0
    if overlapping:
        pairs = pd.DataFrame({"invoice": invoices, "source_sheet": sheet}).drop_duplicates()
        shared_invoices = int((pairs.groupby("invoice")["source_sheet"].nunique() > 1).sum())

    return {
        "sheets": _records(per_sheet),
        "overlap": {
            "months": [str(month) for month in overlapping],
            "rows_per_sheet": _records(overlap_rows) if overlapping else [],
            "invoices_in_more_than_one_sheet": shared_invoices,
        },
    }


# --------------------------------------------------------------------------
# profiling tools (pure)
# --------------------------------------------------------------------------
def profile_raw(raw_df: pd.DataFrame, cfg: CleaningConfig | None = None) -> dict[str, Any]:
    """Describe the raw extract without changing it (§2, §8).

    Returns four of the ``data_quality_findings.json`` sections, already shaped:
    ``raw_profile``, ``missing_values``, ``cancellations_adjustments`` and ``partial_month``.
    Splitting them here rather than in the assembler keeps every count in one place.
    """
    cfg = _cfg(cfg)
    rows = int(len(raw_df))
    invoices = _invoices(raw_df)
    quantity = _quantity(raw_df)
    price = _price(raw_df)
    revenue = quantity * price
    total_revenue = float(revenue.sum())

    cancel_mask = invoices.str.startswith(tuple(cfg.rules.cancellation_prefixes))
    adjust_mask = invoices.str.startswith(tuple(cfg.rules.adjustment_prefixes))

    # -- missing values, per column and for the customer id specifically ----
    missing_values = {
        str(column): {
            "missing": int(raw_df[column].isna().sum()),
            "share": _share(int(raw_df[column].isna().sum()), rows),
        }
        for column in raw_df.columns
    }
    anonymous = raw_df["Customer ID"].isna()
    anonymous_rows = int(anonymous.sum())

    # -- rows per month, with the partial month flagged (§8) ----------------
    dates = pd.to_datetime(raw_df["InvoiceDate"], errors="coerce")
    months = dates.dt.strftime(_MONTH_FORMAT)
    per_month = (
        pd.DataFrame({"month": months, "rows": 1, "units": quantity})
        .groupby("month", dropna=True, sort=True)
        .agg(rows=("rows", "sum"), units=("units", "sum"))
        .reset_index()
    )
    per_month["is_partial"] = per_month["month"].isin(cfg.raw.partial_months)

    # -- weekday distribution: documents the "almost no Saturday trading" fact
    weekday = (
        pd.DataFrame({"weekday": dates.dt.dayofweek, "rows": 1, "units": quantity})
        .dropna(subset=["weekday"])
        .groupby("weekday", sort=True)
        .agg(rows=("rows", "sum"), units=("units", "sum"))
        .reset_index()
    )
    weekday["weekday"] = weekday["weekday"].astype(int)
    weekday["weekday_name"] = weekday["weekday"].map(dict(enumerate(_WEEKDAY_NAMES)))

    # -- lower-case stock-code variants merged by upper-casing (§8) ---------
    raw_codes = raw_df["StockCode"].astype(str)
    normalised = _normalise_code(raw_df["StockCode"])
    variant_mask = raw_codes.str.strip() != normalised

    largest_index = quantity.idxmax() if rows and quantity.notna().any() else None
    largest_line = (
        {
            "quantity": int(quantity.loc[largest_index]),
            "invoice": str(invoices.loc[largest_index]),
            "stock_code": str(normalised.loc[largest_index]),
            "description": _json_ready(raw_df["Description"].loc[largest_index]),
            "month": _json_ready(months.loc[largest_index]),
        }
        if largest_index is not None
        else {}
    )

    adjustment_lines = pd.DataFrame(
        {
            "invoice": invoices[adjust_mask],
            "stock_code": normalised[adjust_mask],
            "description": raw_df["Description"][adjust_mask],
            "quantity": quantity[adjust_mask],
            "price": price[adjust_mask],
            "price_sign": np.sign(price[adjust_mask]).astype("Int64"),
            "month": months[adjust_mask],
        }
    ).sort_values(["invoice", "stock_code"], kind="mergesort")

    nonpositive_quantity = quantity <= cfg.rules.drop_quantity_at_or_below
    nonpositive_price = price <= cfg.rules.drop_price_at_or_below
    partial_mask = months.isin(cfg.raw.partial_months)

    return {
        "raw_profile": {
            "rows": rows,
            "columns": [str(column) for column in raw_df.columns],
            "dtypes": {str(column): str(dtype) for column, dtype in raw_df.dtypes.items()},
            "total_units": int(quantity.sum()) if rows else 0,
            "total_revenue": total_revenue,
            "distinct_invoices": int(invoices.nunique()),
            "distinct_stock_codes": int(normalised.nunique()),
            "distinct_customers": int(raw_df["Customer ID"].nunique(dropna=True)),
            "date_range": {
                "first": _json_ready(dates.min()),
                "last": _json_ready(dates.max()),
            },
            "rows_per_month": _records(per_month),
            "source_sheets": _source_sheet_profile(raw_df, months, invoices),
            "weekday_distribution": _records(
                weekday[["weekday", "weekday_name", "rows", "units"]]
            ),
            "largest_line": largest_line,
            "lowercase_stock_code_variants": {
                "rows": int(variant_mask.sum()),
                "codes": int(raw_codes[variant_mask].str.strip().nunique()),
            },
            "nonpositive_quantity_without_cancellation": {
                "rows": int((nonpositive_quantity & ~cancel_mask).sum()),
                "share": _share(int((nonpositive_quantity & ~cancel_mask).sum()), rows),
                "threshold": float(cfg.rules.drop_quantity_at_or_below),
            },
            "nonpositive_price": {
                "rows": int(nonpositive_price.sum()),
                "share": _share(int(nonpositive_price.sum()), rows),
                "threshold": float(cfg.rules.drop_price_at_or_below),
            },
        },
        "missing_values": {
            "per_column": missing_values,
            "customer_id": {
                "rows": anonymous_rows,
                "row_share": _share(anonymous_rows, rows),
                "revenue": float(revenue[anonymous].sum()),
                "revenue_share": _share(float(revenue[anonymous].sum()), total_revenue),
            },
        },
        "cancellations_adjustments": {
            "cancellation_prefixes": list(cfg.rules.cancellation_prefixes),
            "adjustment_prefixes": list(cfg.rules.adjustment_prefixes),
            "cancellations": {
                "invoices": int(invoices[cancel_mask].nunique()),
                "invoice_share": _share(
                    int(invoices[cancel_mask].nunique()), int(invoices.nunique())
                ),
                "rows": int(cancel_mask.sum()),
                "row_share": _share(int(cancel_mask.sum()), rows),
                "units_abs": int(quantity[cancel_mask].abs().sum()),
            },
            "adjustments": {
                "invoices": int(invoices[adjust_mask].nunique()),
                "rows": int(adjust_mask.sum()),
                "lines": _records(adjustment_lines),
            },
            "note": (
                "Gross demand (§9): cancellations and adjustments are removed from the sales "
                "table but never subtracted from demand."
            ),
        },
        "partial_month": {
            "months": list(cfg.raw.partial_months),
            "rows": int(partial_mask.sum()),
            "row_share": _share(int(partial_mask.sum()), rows),
            "units": int(quantity[partial_mask].sum()),
            "last_full_month": cfg.raw.last_full_month,
            "note": "Never scored (§8, §16, §21); may be displayed as a hatched partial actual.",
        },
    }


def detect_duplicates(raw_df: pd.DataFrame, cfg: CleaningConfig | None = None) -> dict[str, Any]:
    """Count exact duplicates over the eight original columns and apply the §11 thresholds.

    The subset is ``cfg.raw.required_columns``, **not** ``raw_df.columns``: ``load_raw`` appends a
    ``source_sheet`` column, and de-duplicating on nine columns would silently report a different
    number from the one :mod:`pipeline.cleaning` acts on.
    """
    cfg = _cfg(cfg)
    subset = [column for column in cfg.raw.required_columns if column in raw_df.columns]
    rows = int(len(raw_df))
    quantity = _quantity(raw_df)
    total_units = int(quantity.sum()) if rows else 0

    extra_copies = raw_df.duplicated(subset=subset, keep="first")
    duplicate_rows = int(extra_copies.sum())
    duplicate_units = int(quantity[extra_copies].sum()) if duplicate_rows else 0
    row_share = _share(duplicate_rows, rows)
    units_share = _share(duplicate_units, total_units)

    top = (
        raw_df.loc[raw_df.duplicated(subset=subset, keep=False), subset]
        .groupby(subset, dropna=False, sort=False)
        .size()
        .reset_index(name="occurrences")
        .sort_values(["occurrences", *subset], ascending=False, kind="mergesort")
        .head(TOP_DUPLICATED_LINES)
        if duplicate_rows
        else pd.DataFrame(columns=[*subset, "occurrences"])
    )

    # Split the count by source sheet. The two are very different problems: duplication *within*
    # a sheet is data entry repeated at the till, duplication *across* sheets is the same
    # transaction shipped twice because the sheets' coverage overlaps (see
    # :func:`_source_sheet_profile`). PRD §8's indicative figure is the within-sheet one.
    within_source: int | None = None
    if "source_sheet" in raw_df.columns:
        within_source = int(
            sum(
                part.duplicated(subset=subset, keep="first").sum()
                for _, part in raw_df.groupby("source_sheet", sort=True)
            )
        )

    warning = (
        row_share > cfg.warnings.duplicate_max_row_share
        or units_share > cfg.warnings.duplicate_max_units_share
    )
    return {
        "subset": subset,
        "rows": rows,
        "duplicate_rows": duplicate_rows,
        "within_source_duplicate_rows": within_source,
        "cross_source_duplicate_rows": (
            None if within_source is None else duplicate_rows - within_source
        ),
        "duplicate_row_share": row_share,
        "duplicate_units": duplicate_units,
        "duplicate_units_share": units_share,
        "row_share_threshold": float(cfg.warnings.duplicate_max_row_share),
        "units_share_threshold": float(cfg.warnings.duplicate_max_units_share),
        "warning": bool(warning),
        "top_duplicated_lines": _records(top),
    }


def list_nonproduct_codes(
    raw_df: pd.DataFrame, cfg: CleaningConfig | None = None
) -> pd.DataFrame:
    """Every normalised stock code that does not match ``rules.inventory_code_pattern`` (§12).

    Pure: it computes the review table but never writes it. :func:`confirm_exclusion_list`
    performs the write, because a function that touches disk takes ``ctx``
    (``docs/interfaces.md`` §6 rule 5).
    """
    cfg = _cfg(cfg)
    pattern = re.compile(cfg.rules.inventory_code_pattern)
    codes = _normalise_code(raw_df["StockCode"])
    quantity = _quantity(raw_df)

    frame = pd.DataFrame(
        {
            "stock_code": codes,
            "description": raw_df["Description"].astype("string"),
            "units": quantity,
            "revenue": quantity * _price(raw_df),
        }
    )
    frame = frame[~frame["stock_code"].map(lambda code: bool(pattern.match(code)))]

    if frame.empty:
        return pd.DataFrame(
            columns=["stock_code", "rows", "units", "revenue", "sample_description",
                     "in_exclusion_list"]
        )

    aggregated = (
        frame.groupby("stock_code", sort=True)
        .agg(rows=("stock_code", "size"), units=("units", "sum"), revenue=("revenue", "sum"))
        .reset_index()
    )
    # Sample description: the most frequent non-empty one, ties broken alphabetically — the same
    # rule :mod:`pipeline.cleaning` uses for the canonical description, so the two agree.
    described = (
        frame.dropna(subset=["description"])
        .groupby(["stock_code", "description"], sort=False)
        .size()
        .reset_index(name="n")
        .sort_values(["stock_code", "n", "description"], ascending=[True, False, True],
                     kind="mergesort")
        .drop_duplicates("stock_code")
        .set_index("stock_code")["description"]
    )
    aggregated["sample_description"] = aggregated["stock_code"].map(described)

    excluded = {code.strip().upper() for code in load_non_inventory_codes()["stock_code"]}
    aggregated["in_exclusion_list"] = aggregated["stock_code"].isin(excluded)

    aggregated = aggregated.sort_values(
        ["rows", "stock_code"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    return aggregated[
        ["stock_code", "rows", "units", "revenue", "sample_description", "in_exclusion_list"]
    ]


def abnormal_quantities(
    clean_df: pd.DataFrame, threshold: int | None = None, cfg: CleaningConfig | None = None
) -> pd.DataFrame:
    """Sales lines above ``warnings.abnormal_line_quantity`` — flagged, never removed (§10).

    Reuses the ``is_abnormal_qty`` column :mod:`pipeline.cleaning` already computed against the
    same configured threshold whenever it is present, so the two never drift apart; the explicit
    comparison is the fallback for a frame that does not carry it.
    """
    cfg = _cfg(cfg)
    limit = cfg.warnings.abnormal_line_quantity if threshold is None else threshold
    if "is_abnormal_qty" in clean_df.columns and threshold is None:
        mask = clean_df["is_abnormal_qty"].astype(bool)
    else:
        mask = pd.to_numeric(clean_df["quantity"]) > limit

    lines = clean_df.loc[mask].copy()
    result = pd.DataFrame(
        {
            "invoice": lines["invoice"].astype(str),
            "stock_code": lines["stock_code"].astype(str),
            "description": lines["description"].astype("string"),
            "quantity": pd.to_numeric(lines["quantity"]),
            "price": pd.to_numeric(lines["price"]),
            "customer_identified": lines["customer_id"].notna(),
            "month": lines["month"].astype(str),
        }
    )
    return result.sort_values(
        ["quantity", "invoice", "stock_code"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)


def panel_preview(panel_df: pd.DataFrame, cfg: CleaningConfig | None = None) -> dict[str, Any]:
    """Facts about the hand-off panel (``clean_data.csv``) for the findings JSON."""
    cfg = _cfg(cfg)
    rows = int(len(panel_df))
    zero_rows = int((panel_df["units_sold"] == 0).sum())
    partial_rows = int(panel_df["is_partial_month"].astype(bool).sum())
    return {
        "rows": rows,
        "products": int(panel_df["stock_code"].nunique()),
        "months": int(panel_df["month"].nunique()),
        "first_month": str(panel_df["month"].min()) if rows else None,
        "last_month": str(panel_df["month"].max()) if rows else None,
        "zero_rows": zero_rows,
        "zero_share": _share(zero_rows, rows),
        "partial_rows": partial_rows,
        "last_full_month": cfg.raw.last_full_month,
    }


# --------------------------------------------------------------------------
# writers (take ctx — docs/interfaces.md §6 rule 5)
# --------------------------------------------------------------------------
def _exclusion_path(ctx: RunContext) -> Path:
    """Where the reviewed exclusion list is written.

    Deliberately **not** routed through ``ctx.out()``: ``config/`` is an *input* to the run, and a
    staged copy under ``artifacts/_staging/<run_id>/config/`` would be invisible to
    ``load_non_inventory_codes()``, which always reads the real location. Resolving against
    ``ctx.base_dir`` keeps a test run from rewriting the repository's own configuration.
    """
    return ctx.base_dir / paths.NON_INVENTORY_STOCKCODES.relative_to(paths.PROJECT_ROOT)


def confirm_exclusion_list(codes_df: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
    """Confirm the non-inventory exclusion list against the data and write it back (§12).

    * A listed code that appears in the data becomes ``confirmed``.
    * A listed code absent from the data keeps its current status — being unused is not a reason
      to promote it to ``confirmed``.
    * A non-matching code found in the data but *not* on the list is appended as
      ``review_needed``. It is **not** excluded: that decision belongs to a human (§12).

    Stock codes are compared after normalisation but written back **verbatim**. The list holds
    both ``M`` and ``m`` on purpose, and ``load_non_inventory_codes()`` raises on a duplicated
    ``stock_code`` — normalising on the way out would merge them and break ``config_snapshot()``,
    and with it every later ``RunContext.start()``.
    """
    current = load_non_inventory_codes()
    present = set(codes_df["stock_code"]) if len(codes_df) else set()

    updated = current.copy()
    seen = _normalise_code(updated["stock_code"])
    updated["status"] = np.where(seen.isin(present), STATUS_CONFIRMED, updated["status"])

    new_codes = (
        codes_df.loc[~codes_df["in_exclusion_list"], "stock_code"].tolist()
        if len(codes_df)
        else []
    )
    if new_codes:
        additions = pd.DataFrame(
            {
                "stock_code": sorted(new_codes),
                "reason": REVIEW_REASON,
                "status": STATUS_REVIEW_NEEDED,
            }
        )
        updated = pd.concat([updated, additions], ignore_index=True)

    updated = updated[EXCLUSION_COLUMNS]
    destination = _exclusion_path(ctx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(destination, index=False, lineterminator="\n", encoding="utf-8")

    confirmed = int((updated["status"] == STATUS_CONFIRMED).sum())
    ctx.logger.info(
        f"exclusion list reviewed: {confirmed} confirmed, {len(new_codes)} flagged "
        f"{STATUS_REVIEW_NEEDED}, {len(updated)} rows total"
    )
    if new_codes:
        ctx.warn(
            f"{len(new_codes)} non-product stock code(s) are not on the exclusion list and were "
            f"flagged {STATUS_REVIEW_NEEDED} (not excluded): {', '.join(sorted(new_codes))}"
        )
    return updated


def build_data_quality_findings(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    waterfall_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: CleaningConfig,
    ctx: RunContext,
    *,
    profile: dict[str, Any] | None = None,
    duplicates: dict[str, Any] | None = None,
    codes_df: pd.DataFrame | None = None,
    abnormal_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assemble and write ``artifacts/reports/data_quality_findings.json`` (§10 step 13, §35).

    The keyword arguments let a caller that has already computed a piece — E1 builds the same
    tables — hand it in instead of paying for a second pass over a million rows. Omitted pieces
    are computed here, so the documented six-argument call works on its own.

    Written through ``ctx.out()`` so a failed run cannot overwrite the previous findings (§39).
    The file is not an ``E<nn>_`` table, so it does not go through :mod:`pipeline.eda.io`.
    """
    profile = profile_raw(raw_df, cfg) if profile is None else profile
    duplicates = detect_duplicates(raw_df, cfg) if duplicates is None else duplicates
    codes_df = list_nonproduct_codes(raw_df, cfg) if codes_df is None else codes_df
    abnormal_df = abnormal_quantities(clean_df, cfg=cfg) if abnormal_df is None else abnormal_df

    overlap = profile["raw_profile"].get("source_sheets", {}).get("overlap", {})
    if overlap.get("months"):
        ctx.warn(
            f"source sheets overlap on {', '.join(overlap['months'])}: "
            f"{overlap['invoices_in_more_than_one_sheet']} invoice number(s) appear in more than "
            f"one sheet and {duplicates['cross_source_duplicate_rows']} rows are cross-sheet "
            "exact duplicates — the shared period is double-counted before de-duplication"
        )

    if duplicates["warning"]:
        ctx.warn(
            f"raw exact duplicates are {duplicates['duplicate_row_share']:.4%} of rows and "
            f"{duplicates['duplicate_units_share']:.4%} of units, above the "
            f"{duplicates['row_share_threshold']:.4%} / "
            f"{duplicates['units_share_threshold']:.4%} thresholds (§11)"
        )

    findings: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": ctx.run_id,
        "data_sha256": ctx.data.sha256,
        "raw_profile": profile["raw_profile"],
        "missing_values": profile["missing_values"],
        "cancellations_adjustments": profile["cancellations_adjustments"],
        "duplicates": duplicates,
        "nonproduct_codes": {
            "pattern": cfg.rules.inventory_code_pattern,
            "codes": int(len(codes_df)),
            "rows": int(codes_df["rows"].sum()) if len(codes_df) else 0,
            "units": int(codes_df["units"].sum()) if len(codes_df) else 0,
            "revenue": float(codes_df["revenue"].sum()) if len(codes_df) else 0.0,
            "on_exclusion_list": int(codes_df["in_exclusion_list"].sum()) if len(codes_df) else 0,
            "review_needed": int((~codes_df["in_exclusion_list"]).sum()) if len(codes_df) else 0,
            "table": _records(codes_df),
        },
        "abnormal_quantities": {
            "threshold": int(cfg.warnings.abnormal_line_quantity),
            "lines": int(len(abnormal_df)),
            "max_quantity": int(abnormal_df["quantity"].max()) if len(abnormal_df) else 0,
            "units": int(abnormal_df["quantity"].sum()) if len(abnormal_df) else 0,
            "top_lines": _records(abnormal_df.head(TOP_ABNORMAL_LINES)),
            "note": "Flagged, never removed (§10); motivates the robust σ of §26.",
        },
        "partial_month": profile["partial_month"],
        "waterfall": _records(waterfall_df),
        "panel_preview": panel_preview(panel_df, cfg),
        "warnings": list(ctx.warnings),
    }

    relative = paths.DATA_QUALITY_FINDINGS.relative_to(paths.PROJECT_ROOT)
    destination = ctx.out(relative)
    # ``newline="\n"`` matters: in text mode Windows would translate every "\n" to "\r\n", so the
    # same run on Windows and in CI would produce different bytes for a committed artifact (§40).
    destination.write_text(
        json.dumps(_json_ready(findings), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ctx.record_artifact("data_quality_findings", relative)
    ctx.logger.info(f"data-quality findings written: {relative.as_posix()}")
    return findings


# --------------------------------------------------------------------------
# standalone entry point
# --------------------------------------------------------------------------
def run() -> int:
    """``python -m pipeline.quality`` — profile the data, review the exclusion list, build E1."""
    # Imported here rather than at module scope: E1 imports these tools, so a top-level import
    # would close the cycle.
    from pipeline.contract import read_panel
    from pipeline.eda import e01_data_quality
    from pipeline.eda.io import load_table

    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("data_quality_profiling"):
            for required in (paths.CLEAN_TRANSACTIONS, paths.CLEAN_DATA):
                if not required.is_file():
                    raise FileNotFoundError(
                        f"{required} not found. Run `python -m pipeline.cleaning` and "
                        "`python -m pipeline.panel` first."
                    )
            cfg = load_cleaning_config()
            raw_df, _ = load_raw(ctx=ctx)  # also records file/sha256/rows on the run
            clean_df = pd.read_parquet(paths.CLEAN_TRANSACTIONS)
            panel_df = read_panel(paths.CLEAN_DATA)
            waterfall_df = load_table("E01_cleaning_waterfall", ctx)

            codes_df = list_nonproduct_codes(raw_df, cfg)
            confirm_exclusion_list(codes_df, ctx)
            result = e01_data_quality.run(
                raw_df, clean_df, waterfall_df, panel_df, cfg, ctx, codes_df=codes_df
            )
        findings = result["findings"]
        duplicates = findings["duplicates"]
        print(
            f"duplicate rows: {duplicates['duplicate_rows']} "
            f"({duplicates['duplicate_row_share']:.4%} of rows, "
            f"warning={duplicates['warning']})\n"
            f"non-product codes: {findings['nonproduct_codes']['codes']} "
            f"({findings['nonproduct_codes']['rows']} rows, "
            f"{findings['nonproduct_codes']['review_needed']} need review)\n"
            f"abnormal lines: {findings['abnormal_quantities']['lines']} "
            f"above {findings['abnormal_quantities']['threshold']} units\n"
            f"tables: {', '.join(sorted(result['tables']))}\n"
            f"figures: {', '.join(sorted(result['figures']))}"
        )
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
