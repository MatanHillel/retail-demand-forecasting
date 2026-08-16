"""E7 — The twenty biggest products, by units and by revenue (US-09, PRD §35A E7).

Two different rankings on purpose. The best-selling product by *volume* is usually a cheap
high-turnover item; the best by *revenue* is usually not the same product. Showing both stops the
report claiming one "top product" and keeps the reader from reading a units chart as a money
chart.

Totals exclude the partial December 2011 (§8), so a share here is a share of the complete
Dec 2009 – Nov 2011 period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E7"

#: Length of each ranking (§35A E7).
TOP_N = 20

#: Longest product description shown on a bar label before it is trimmed.
_LABEL_CHARS = 38


def _totals(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Units, revenue and the canonical description per product, partial month excluded."""
    window = panel_df.loc[~panel_df["month"].astype(str).isin(cfg.raw.partial_months)]
    grouped = (
        window.groupby(window["stock_code"].astype(str), sort=True)
        .agg(units=("units_sold", "sum"), revenue=("gross_revenue", "sum"))
        .reset_index()
    )
    # The panel repeats one canonical description on every month of a product, so the first
    # non-null occurrence is that description. The panel is sorted, so "first" is deterministic.
    described = window.dropna(subset=["description"]).drop_duplicates("stock_code", keep="first")
    descriptions = pd.Series(
        described["description"].to_numpy(),
        index=described["stock_code"].astype(str).to_numpy(),
    )
    grouped["description"] = grouped["stock_code"].map(descriptions)
    return grouped


def _ranking(totals: pd.DataFrame, column: str) -> pd.DataFrame:
    """Top ``TOP_N`` products by ``column``, with each one's share of the period total."""
    total = float(totals[column].sum())
    ranked = totals.sort_values(
        [column, "stock_code"], ascending=[False, True], kind="mergesort"
    ).head(TOP_N).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked["share"] = ranked[column] / total if total else 0.0
    return ranked[["rank", "stock_code", "description", "units", "revenue", "share"]]


def _figure(ranked: pd.DataFrame, column: str, name: str, title: str, xlabel: str,
            ctx: RunContext) -> Path:
    """Horizontal bars, biggest at the top."""
    apply_style()
    labels = [
        f"{row.stock_code} {str(row.description)[:_LABEL_CHARS]}"
        for row in ranked.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    positions = list(range(len(ranked)))[::-1]  # rank 1 at the top
    ax.barh(positions, ranked[column], color=PALETTE[0])
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    finalize(fig, title=title, xlabel=xlabel, ylabel="Product (stock code and description)")
    save_figure(fig, name, ctx)
    return figure_path(name)


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E7 and write its two tables and two figures."""
    totals = _totals(panel_df, cfg)
    by_units = _ranking(totals, "units")
    by_revenue = _ranking(totals, "revenue")

    tables = {"E07_top20_units": by_units, "E07_top20_revenue": by_revenue}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {
        "E07_top20_units": _figure(
            by_units, "units", "E07_top20_units",
            f"Top {TOP_N} products by units sold (Dec 2009 - Nov 2011)",
            "Units sold (units)", ctx,
        ),
        "E07_top20_revenue": _figure(
            by_revenue, "revenue", "E07_top20_revenue",
            f"Top {TOP_N} products by gross revenue (Dec 2009 - Nov 2011)",
            "Gross revenue (GBP)", ctx,
        ),
    }
    return {"tables": tables, "figures": figures}
