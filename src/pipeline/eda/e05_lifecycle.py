"""E5 — Product lifecycle (US-09, PRD §35A E5).

For each product: how many months it actually sold in, when it first and last sold, and how long
its span was. The distribution matters for modelling — a catalogue where most products sell in
only a handful of months cannot be forecast with long lag windows, which is why the feature set
leans on short lags and the active-product rule (§18.1, §14).

Two terms used in the outputs:

* **selling months** — months in which the product sold at least one unit. Zero-filled months in
  the panel are months the product existed but sold nothing; they do not count.
* **span months** — calendar distance from the first selling month to the last, inclusive. A
  product with a long span but few selling months is intermittent rather than long-lived.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.periods import month_to_period
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E5"

#: Bucket boundaries the summary reports, from §35A E5. Analysis definitions, not tunable
#: thresholds: "short-lived" is at most six selling months, "long-lived" at least twenty-four.
SHORT_LIFECYCLE_MONTHS = 6
LONG_LIFECYCLE_MONTHS = 24


def _lifecycle(panel_df: pd.DataFrame) -> pd.DataFrame:
    """One row per product: selling months, first and last selling month, span."""
    selling = panel_df.loc[panel_df["units_sold"] > 0, ["stock_code", "month", "units_sold"]]
    if selling.empty:
        return pd.DataFrame(
            columns=["stock_code", "selling_months", "first_month", "last_month",
                     "span_months", "units"]
        )

    frame = (
        selling.groupby(selling["stock_code"].astype(str), sort=True)
        .agg(
            selling_months=("month", "nunique"),
            first_month=("month", "min"),
            last_month=("month", "max"),
            units=("units_sold", "sum"),
        )
        .reset_index()
    )
    first = month_to_period(frame["first_month"])
    last = month_to_period(frame["last_month"])
    frame["span_months"] = (last - first).map(lambda offset: offset.n) + 1
    return frame[
        ["stock_code", "selling_months", "first_month", "last_month", "span_months", "units"]
    ]


def _summary(lifecycle: pd.DataFrame) -> pd.DataFrame:
    """Share of the catalogue that is short-lived, long-lived, and the median selling months."""
    products = int(len(lifecycle))
    if not products:
        return pd.DataFrame()
    short = int((lifecycle["selling_months"] <= SHORT_LIFECYCLE_MONTHS).sum())
    long_lived = int((lifecycle["selling_months"] >= LONG_LIFECYCLE_MONTHS).sum())
    return pd.DataFrame(
        [
            {
                "products": products,
                "short_lifecycle_months": SHORT_LIFECYCLE_MONTHS,
                "short_lifecycle_products": short,
                "short_lifecycle_share": short / products,
                "long_lifecycle_months": LONG_LIFECYCLE_MONTHS,
                "long_lifecycle_products": long_lived,
                "long_lifecycle_share": long_lived / products,
                "median_selling_months": float(lifecycle["selling_months"].median()),
                "median_span_months": float(lifecycle["span_months"].median()),
            }
        ]
    )


def _figure(lifecycle: pd.DataFrame, ctx: RunContext) -> Path:
    """Histogram of selling months per product, one bar per possible count."""
    apply_style()
    counts = (
        lifecycle["selling_months"].value_counts().sort_index()
        if len(lifecycle)
        else pd.Series(dtype=int)
    )
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.to_numpy(), color=PALETTE[0])
    ax.axvline(
        SHORT_LIFECYCLE_MONTHS,
        color=PALETTE[3],
        linestyle="--",
        linewidth=1.2,
        label=f"{SHORT_LIFECYCLE_MONTHS} months or fewer",
    )
    ax.legend(loc="upper right")
    finalize(
        fig,
        title="How many months each product actually sold in",
        xlabel="Selling months per product (months with at least one unit sold)",
        ylabel="Products (count)",
    )
    save_figure(fig, "E05_lifecycle_hist", ctx)
    return figure_path("E05_lifecycle_hist")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E5 and write its two tables and one figure."""
    lifecycle = _lifecycle(panel_df)
    summary = _summary(lifecycle)

    tables = {"E05_lifecycle": lifecycle, "E05_lifecycle_summary": summary}
    save_table(lifecycle, "E05_lifecycle", ctx)
    save_table(summary, "E05_lifecycle_summary", ctx, fmt="json")

    figures = {"E05_lifecycle_hist": _figure(lifecycle, ctx)}
    return {"tables": tables, "figures": figures}
