"""E6 — ABC concentration and the Pareto curve (US-09, PRD §35A E6, §18.2).

How concentrated revenue is: what share of the catalogue produces the first 80 % of revenue (class
A), the next 15 % (B) and the long tail (C). Concentration is why the inventory policy treats
classes differently and why σ falls back to the ABC group when a product has too few residuals
(§27).

**This is the full-period ABC and every output says so.** Modelling, evaluation and the σ fallback
use ABC computed on the *training window only*, because a class derived from revenue the model has
not seen yet leaks the future (§18.2, §23, §27). The two are different tables on purpose; the
label on this figure is what stops them being confused.

The cut-off is the last **full** month: December 2011 is partial and excluded from every total
(§8). Thresholds come from ``inventory_policy.yaml → abc``; nothing is written in code (§40).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.abc import ABC_CLASSES, compute_abc
from pipeline.config import CleaningConfig, load_inventory_policy
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import ABC_COLORS, PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E6"

#: Stated on every E6 output so the descriptive classes are never read as the modelling ones.
WINDOW_LABEL = "full period"


def _abc_table(abc: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """One row per class: products, and their share of the catalogue, revenue and units."""
    frame = abc.assign(units=abc["stock_code"].map(units).fillna(0))
    products = int(len(frame))
    total_revenue = float(frame["revenue"].sum())
    total_units = float(frame["units"].sum())

    rows = []
    for abc_class in ABC_CLASSES:
        part = frame.loc[frame["abc_class"] == abc_class]
        rows.append(
            {
                "abc_class": abc_class,
                "window": WINDOW_LABEL,
                "products": int(len(part)),
                "product_share": len(part) / products if products else 0.0,
                "revenue": float(part["revenue"].sum()),
                "revenue_share": float(part["revenue"].sum() / total_revenue)
                if total_revenue
                else 0.0,
                "units": int(part["units"].sum()),
                "units_share": float(part["units"].sum() / total_units) if total_units else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _pareto(abc: pd.DataFrame) -> pd.DataFrame:
    """Cumulative product share against cumulative revenue share, best product first."""
    frame = abc.loc[:, ["stock_code", "revenue", "revenue_share", "cum_share", "abc_class"]].copy()
    products = len(frame)
    frame.insert(0, "rank", range(1, products + 1))
    frame["cum_product_share"] = frame["rank"] / products if products else 0.0
    frame = frame.rename(columns={"cum_share": "cum_revenue_share"})
    return frame[
        ["rank", "stock_code", "cum_product_share", "revenue", "revenue_share",
         "cum_revenue_share", "abc_class"]
    ]


def _figure(pareto: pd.DataFrame, ctx: RunContext) -> Path:
    """Pareto curve with the A and B cut-offs marked and the three classes coloured."""
    apply_style()
    thresholds = load_inventory_policy().abc
    fig, ax = plt.subplots()

    for abc_class in ABC_CLASSES:
        part = pareto.loc[pareto["abc_class"] == abc_class]
        if part.empty:
            continue
        ax.fill_between(
            part["cum_product_share"],
            0,
            part["cum_revenue_share"],
            color=ABC_COLORS[abc_class],
            alpha=0.35,
            label=f"class {abc_class}",
        )
    ax.plot(
        pareto["cum_product_share"], pareto["cum_revenue_share"], color=PALETTE[7], linewidth=1.6
    )

    for share, name in ((thresholds.a_cum_share, "A"), (thresholds.b_cum_share, "B")):
        ax.axhline(share, color=PALETTE[7], linestyle="--", linewidth=1.0)
        ax.annotate(
            f"{name} cut-off {share:.0%}",
            xy=(0.02, share),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    finalize(
        fig,
        title=f"Revenue concentration: Pareto curve and ABC classes ({WINDOW_LABEL})",
        xlabel="Cumulative share of products (ranked by revenue, best first)",
        ylabel="Cumulative share of revenue",
    )
    save_figure(fig, "E06_pareto", ctx)
    return figure_path("E06_pareto")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E6 and write its two tables and one figure."""
    through_month = cfg.raw.last_full_month
    abc = compute_abc(panel_df, through_month=through_month)

    window = panel_df.loc[panel_df["month"].astype(str) <= through_month]
    units = window.groupby(window["stock_code"].astype(str), sort=False)["units_sold"].sum()

    table = _abc_table(abc, units)
    pareto = _pareto(abc)

    tables = {"E06_abc_table": table, "E06_pareto": pareto}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E06_pareto": _figure(pareto, ctx)}
    return {"tables": tables, "figures": figures}
