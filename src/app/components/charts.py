"""Matplotlib/Seaborn chart helpers for the Streamlit shell (US-27, PRD §33, §35A.3).

Every figure reuses ``pipeline.eda.style`` — the palette, the ABC colours, the partial-month
hatching, the title/footnote furniture — so the app reads as the same visual system as the EDA
report. Figures mirror ``pipeline/eda/e02_demand_over_time.py``'s ``_monthly_figure`` /
``_seasonal_figure`` but return the live figure for ``st.pyplot`` instead of saving to disk.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from pipeline.eda.style import PALETTE, apply_style, finalize, hatch_partial


def monthly_demand_line(monthly_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Two stacked panels (units, revenue) per month, with partial months hatched."""
    apply_style()
    months = monthly_df["month"].tolist()
    positions = list(range(len(months)))
    partial = [index for index, flag in enumerate(monthly_df["is_partial"]) if flag]

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10.0, 7.0))
    top.plot(positions, monthly_df["units"], marker="o", color=PALETTE[0], linewidth=2)
    bottom.plot(positions, monthly_df["revenue"], marker="o", color=PALETTE[1], linewidth=2)
    for axes in (top, bottom):
        if partial:
            hatch_partial(axes, partial)
    bottom.set_xticks(positions)
    bottom.set_xticklabels(months, rotation=90)
    bottom.set_xlabel("Month")
    bottom.set_ylabel("Gross revenue (GBP)")
    top.set_ylabel("Units sold (units)")

    finalize(
        fig,
        title="Monthly demand and revenue, Dec 2009 - Dec 2011",
        xlabel="",
        ylabel="Units sold (units)",
    )
    return fig


def seasonal_index_bar(seasonal_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """Bar chart of the seasonal index with the average-month reference line at 1.0."""
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(seasonal_df["month_of_year"], seasonal_df["seasonal_index"], color=PALETTE[0])
    ax.axhline(1.0, color=PALETTE[7], linestyle="--", linewidth=1.2, label="average month = 1.0")
    ax.set_xticks(seasonal_df["month_of_year"])
    ax.legend(loc="upper left")
    finalize(
        fig,
        title="Seasonal index by calendar month (mean of two Dec-Nov years)",
        xlabel="Calendar month (1 = January)",
        ylabel="Seasonal index (1.0 = average month)",
    )
    return fig


def kpi_tile(label: str, value: object, help: str | None = None) -> None:
    st.metric(label=label, value=value, help=help)
