"""E10 — Order structure (US-10, PRD §35A E10, §47).

What one order looks like: how much it is worth, how many product lines it carries, and how many
units sit on a line. Together they say whether this is a shop or a wholesaler — and the answer,
"mostly wholesale", is the context for the extreme values E9 measures.

**The wholesale cut is an EDA convention, not a rule.** The PRD names no threshold, so
:data:`WHOLESALE_LINE_QUANTITY` is stated openly, reported as a share, and used nowhere else in
the project — nothing downstream branches on it (§4 of the issue).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import LOG_SCALE_SUFFIX, PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E10"

#: A line of this many units or more is counted as a wholesale-sized line. Descriptive only: the
#: PRD fixes no threshold, and no model, metric or policy reads this number.
WHOLESALE_LINE_QUANTITY = 12

#: Quantiles reported for each distribution.
_QUANTILES = {"p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p90": 0.90, "p99": 0.99}

_LOG_BINS = 40


def _invoices(clean_df: pd.DataFrame) -> pd.DataFrame:
    """One row per invoice: its value, its line count and its units."""
    return (
        clean_df.groupby(clean_df["invoice"].astype(str), sort=True)
        .agg(
            invoice_value=("line_revenue", "sum"),
            lines=("line_revenue", "size"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )


def _describe(values: pd.Series, metric: str, unit: str) -> list[dict[str, Any]]:
    """Long-format count/mean/quantiles/max for one distribution.

    Long format rather than one wide row per metric: the distributions have different units, and
    a scalar like the wholesale share has no quantiles to pad with blanks.
    """
    numeric = pd.to_numeric(values)
    rows = [
        {"metric": metric, "unit": unit, "statistic": "count", "value": float(len(numeric))},
        {"metric": metric, "unit": unit, "statistic": "mean", "value": float(numeric.mean())},
        {"metric": metric, "unit": unit, "statistic": "min", "value": float(numeric.min())},
    ]
    rows += [
        {"metric": metric, "unit": unit, "statistic": name, "value": float(numeric.quantile(q))}
        for name, q in _QUANTILES.items()
    ]
    rows.append(
        {"metric": metric, "unit": unit, "statistic": "max", "value": float(numeric.max())}
    )
    return rows


def _summary(clean_df: pd.DataFrame, invoices: pd.DataFrame) -> pd.DataFrame:
    """Invoice value, lines per invoice, units per line, and the wholesale-line share."""
    quantity = pd.to_numeric(clean_df["quantity"])
    rows = (
        _describe(invoices["invoice_value"], "invoice_value", "GBP")
        + _describe(invoices["lines"], "lines_per_invoice", "lines")
        + _describe(quantity, "quantity_per_line", "units")
    )
    wholesale = int((quantity >= WHOLESALE_LINE_QUANTITY).sum())
    rows += [
        {
            "metric": "wholesale_lines",
            "unit": f"lines with quantity >= {WHOLESALE_LINE_QUANTITY}",
            "statistic": "count",
            "value": float(wholesale),
        },
        {
            "metric": "wholesale_lines",
            "unit": f"lines with quantity >= {WHOLESALE_LINE_QUANTITY}",
            "statistic": "share",
            "value": wholesale / len(quantity) if len(quantity) else 0.0,
        },
    ]
    return pd.DataFrame(rows)


def _figure(invoices: pd.DataFrame, ctx: RunContext) -> Path:
    """Invoice value and lines per invoice, both log-scaled — the range spans four decades."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    value = pd.to_numeric(invoices["invoice_value"])
    positive = value.loc[value > 0]
    left.hist(
        positive,
        bins=np.logspace(0, np.log10(max(float(positive.max()), 10.0)), _LOG_BINS),
        color=PALETTE[0],
    )
    left.set_xscale("log")
    left.set_xlabel(f"Invoice value (GBP){LOG_SCALE_SUFFIX}")
    left.set_ylabel("Invoices (count)")

    lines = pd.to_numeric(invoices["lines"])
    right.hist(
        lines,
        bins=np.logspace(0, np.log10(max(float(lines.max()), 10.0)), _LOG_BINS),
        color=PALETTE[1],
    )
    right.set_xscale("log")
    right.set_xlabel(f"Product lines per invoice (lines){LOG_SCALE_SUFFIX}")
    right.set_ylabel("Invoices (count)")

    finalize(
        fig,
        title="Order structure: what one invoice is worth and how many lines it carries",
        xlabel=f"Invoice value (GBP){LOG_SCALE_SUFFIX}",
        ylabel="Invoices (count)",
    )
    save_figure(fig, "E10_invoice_hist", ctx)
    return figure_path("E10_invoice_hist")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E10 and write its table and figure."""
    invoices = _invoices(clean_df)
    tables = {"E10_invoice_summary": _summary(clean_df, invoices)}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E10_invoice_hist": _figure(invoices, ctx)}
    return {"tables": tables, "figures": figures}
