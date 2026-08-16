"""E11 — Customers and countries (US-10, PRD §35A E11, §9, §6.2).

Who buys and from where. Both are **descriptive only**: neither customer nor country is a
modelling input, and no table written here is ever read by a feature, a metric or a policy
(§9, §6.2). Two reasons they still matter:

* roughly a fifth of rows carry no ``Customer ID`` at all, and those rows are **kept** (§9) — the
  units were still sold and still had to be in stock, so dropping them would understate demand;
* the business is overwhelmingly one country, which is precisely why the model has no country
  dimension: a split that leaves 8 % of the data spread over 40 countries would learn noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E11"

#: Length of the country ranking (§35A E11).
TOP_N = 10


def _customer_identification(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Identified against anonymous rows: how many, how much revenue, and their shares."""
    identified = clean_df["customer_id"].notna()
    rows = int(len(clean_df))
    revenue = pd.to_numeric(clean_df["line_revenue"])
    units = pd.to_numeric(clean_df["quantity"])
    total_revenue = float(revenue.sum())
    total_units = float(units.sum())

    records = []
    for label, mask in (("identified", identified), ("anonymous", ~identified)):
        records.append(
            {
                "customer": label,
                "rows": int(mask.sum()),
                "row_share": float(mask.sum() / rows) if rows else 0.0,
                "units": int(units[mask].sum()),
                "units_share": float(units[mask].sum() / total_units) if total_units else 0.0,
                "revenue": float(revenue[mask].sum()),
                "revenue_share": float(revenue[mask].sum() / total_revenue)
                if total_revenue
                else 0.0,
            }
        )
    return pd.DataFrame(records)


def _top_countries(clean_df: pd.DataFrame) -> pd.DataFrame:
    """The ten biggest countries by units, with each one's share of the whole business."""
    grouped = (
        clean_df.groupby(clean_df["country"].astype(str), sort=True)
        .agg(
            rows=("quantity", "size"),
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    total_rows = float(grouped["rows"].sum())
    total_units = float(grouped["units"].sum())
    total_revenue = float(grouped["revenue"].sum())
    grouped["row_share"] = grouped["rows"] / total_rows if total_rows else 0.0
    grouped["units_share"] = grouped["units"] / total_units if total_units else 0.0
    grouped["revenue_share"] = grouped["revenue"] / total_revenue if total_revenue else 0.0

    ranked = grouped.sort_values(
        ["units", "country"], ascending=[False, True], kind="mergesort"
    ).head(TOP_N).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[
        ["rank", "country", "rows", "row_share", "units", "units_share", "revenue",
         "revenue_share", "customers"]
    ]


def _figure(countries: pd.DataFrame, ctx: RunContext) -> Path:
    """Horizontal bars, biggest country at the top."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    positions = list(range(len(countries)))[::-1]
    ax.barh(positions, countries["units"], color=PALETTE[0])
    ax.set_yticks(positions)
    ax.set_yticklabels(countries["country"])
    for position, row in zip(positions, countries.itertuples(), strict=True):
        ax.annotate(
            f"{row.units_share:.1%}",
            xy=(row.units, position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.margins(x=0.12)
    finalize(
        fig,
        title=f"Top {TOP_N} countries by units sold (descriptive only - not a model input)",
        xlabel="Units sold (units)",
        ylabel="Country",
    )
    save_figure(fig, "E11_top_countries", ctx)
    return figure_path("E11_top_countries")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E11 and write its two tables and one figure."""
    identification = _customer_identification(clean_df)
    countries = _top_countries(clean_df)

    tables = {
        "E11_customer_identification": identification,
        "E11_top10_countries": countries,
    }
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E11_top_countries": _figure(countries, ctx)}
    return {"tables": tables, "figures": figures}
