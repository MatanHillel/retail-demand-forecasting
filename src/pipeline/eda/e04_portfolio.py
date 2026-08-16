"""E4 — Product portfolio churn (US-09, PRD §35A E4).

How many products the business actually sells in a month, and how fast the catalogue turns over.
This is the descriptive counterpart to the active-product rule (§14): if products appear and
disappear constantly, forecasting every code in the catalogue every month is meaningless, and the
"at least one sale in the last k months" filter is what keeps the model on live products.

A product counts as **new** in the quarter of its first sale and as **disappearing** in the
quarter of its last sale. Both ends of the window are censored, and each is marked:

* the **last** quarter is right-censored — every product still selling then looks like it is
  disappearing, purely because the data stops — so its disappearing count is left empty rather
  than reported as a cliff (§35A E4);
* the **first** quarter is left-censored — every product already selling in Dec 2009 counts as
  "new" there although it may be years old (the left-censoring of the glossary and §18.1). Its
  new-product count is kept in the table, flagged, and clipped in the figure, because otherwise
  one bar of ~2,700 flattens every real quarter of churn into invisibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.periods import month_to_period, partial_positions
from pipeline.eda.style import PALETTE, apply_style, finalize, hatch_partial, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E4"


def _selling(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Panel rows that represent an actual sale — zero-filled months are not "selling"."""
    return panel_df.loc[panel_df["units_sold"] > 0, ["month", "stock_code"]].copy()


def _products_per_month(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Distinct products with at least one sale in each month."""
    selling = _selling(panel_df)
    grouped = (
        selling.groupby(selling["month"].astype(str), sort=True)["stock_code"]
        .nunique()
        .reset_index(name="products")
        .rename(columns={"index": "month"})
    )
    grouped["is_partial"] = grouped["month"].isin(cfg.raw.partial_months)
    return grouped


def _new_and_disappearing(panel_df: pd.DataFrame) -> pd.DataFrame:
    """New and disappearing products per calendar quarter, with both censored ends flagged.

    The last quarter is right-censored: a product whose final sale lands there may simply still
    be selling, so its ``disappearing`` cell is left empty (``<NA>``) rather than filled with a
    number that would be read as churn. The first quarter is left-censored: its ``new_products``
    count is real but means "already selling", not "launched".
    """
    selling = _selling(panel_df)
    if selling.empty:
        return pd.DataFrame(
            columns=["quarter", "new_products", "disappearing_products",
                     "is_left_censored", "is_right_censored"]
        )

    periods = month_to_period(selling["month"])
    selling = selling.assign(quarter=periods.asfreq("Q").astype(str))
    span = selling.groupby("stock_code")["quarter"].agg(["min", "max"])

    quarters = sorted(selling["quarter"].unique())

    new_counts = span["min"].value_counts()
    gone_counts = span["max"].value_counts()

    frame = pd.DataFrame({"quarter": quarters})
    frame["new_products"] = frame["quarter"].map(new_counts).fillna(0).astype(int)
    frame["disappearing_products"] = (
        frame["quarter"].map(gone_counts).fillna(0).astype("Int64")
    )
    frame["is_left_censored"] = frame["quarter"] == quarters[0]
    frame["is_right_censored"] = frame["quarter"] == quarters[-1]
    frame.loc[frame["is_right_censored"], "disappearing_products"] = pd.NA
    return frame


def _figure(
    per_month: pd.DataFrame, churn: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> Path:
    """Products per month as a line, with quarterly new/disappearing bars beneath it."""
    apply_style()
    months = per_month["month"].tolist()
    positions = list(range(len(months)))

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.0, 7.5))
    top.plot(positions, per_month["products"], marker="o", color=PALETTE[0], linewidth=2)
    partial = partial_positions(months, cfg)
    if partial:
        hatch_partial(top, partial)
    top.set_xticks(positions)
    top.set_xticklabels(months, rotation=90)
    top.set_ylabel("Products sold (count)")

    quarters = churn["quarter"].tolist()
    quarter_positions = list(range(len(quarters)))
    disappearing = churn["disappearing_products"].fillna(0).astype(int)
    bottom.bar(quarter_positions, churn["new_products"], color=PALETTE[2], label="new")
    bottom.bar(
        quarter_positions,
        disappearing,
        bottom=churn["new_products"],
        color=PALETTE[3],
        label="disappearing",
    )

    # Clip the y-axis to the quarters that mean "launched". The left-censored first quarter is
    # every product that already existed, so at full scale it is an order of magnitude taller
    # than real churn and hides it; the bar stays, annotated with its true value.
    comparable = churn.loc[~churn["is_left_censored"]]
    if len(comparable):
        headroom = int((comparable["new_products"] + disappearing.loc[comparable.index]).max())
        bottom.set_ylim(0, headroom * 1.25)
    for position, row in zip(quarter_positions, churn.itertuples(), strict=True):
        if row.is_left_censored:
            bottom.annotate(
                f"{int(row.new_products):,}\n(already selling)",
                xy=(position, bottom.get_ylim()[1]),
                xytext=(0, -6),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9,
            )

    bottom.set_xticks(quarter_positions)
    bottom.set_xticklabels(quarters, rotation=45)
    bottom.set_xlabel(
        "Quarter (first quarter left-censored and clipped; final quarter right-censored)"
    )
    bottom.set_ylabel("Products (count)")
    bottom.legend(loc="upper right")

    finalize(
        fig,
        title="Product portfolio: products sold per month and quarterly churn",
        xlabel="Month",
        ylabel="Products sold (count)",
    )
    save_figure(fig, "E04_products_per_month", ctx)
    return figure_path("E04_products_per_month")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E4 and write its two tables and one figure."""
    per_month = _products_per_month(panel_df, cfg)
    churn = _new_and_disappearing(panel_df)

    tables = {
        "E04_products_per_month": per_month,
        "E04_new_disappearing_by_quarter": churn,
    }
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E04_products_per_month": _figure(per_month, churn, cfg, ctx)}
    return {"tables": tables, "figures": figures}
