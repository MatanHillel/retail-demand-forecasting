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


def product_history_chart(
    history_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    *,
    stock_code: str,
    description: str,
    abc_class: str | None,
    next_month: str | None,
    next_forecast: float | None,
    next_sigma: float | None,
    z: float,
) -> matplotlib.figure.Figure:
    """Screen 3 product chart (§33.3, §35A.2): actual units, back-test forecasts of the champion,
    the next-month forecast with a ``± z*sigma`` band, and the partial month hatched.

    ``history_df`` is the product's full panel slice (``month, units_sold, is_partial_month``,
    ascending). ``backtest_df`` is the champion's back-test predictions for this product
    (``target_month, prediction``) — may be empty when the product has too little history to have
    been back-tested. The next-month forecast is only drawn when ``next_forecast`` is not ``None``.
    """
    apply_style()
    months = history_df["month"].tolist()
    positions = list(range(len(months)))
    month_index = {month: position for position, month in enumerate(months)}
    partial_positions = [
        position
        for position, is_partial in zip(positions, history_df["is_partial_month"], strict=False)
        if is_partial
    ]

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.bar(
        positions,
        history_df["units_sold"],
        color=PALETTE[0],
        label="Actual units sold",
        zorder=2,
    )
    if partial_positions:
        hatch_partial(ax, partial_positions)

    backtest_in_range = backtest_df.loc[backtest_df["target_month"].isin(month_index)]
    if not backtest_in_range.empty:
        backtest_positions = [month_index[month] for month in backtest_in_range["target_month"]]
        ax.plot(
            backtest_positions,
            backtest_in_range["prediction"],
            marker="o",
            linestyle="--",
            color=PALETTE[1],
            linewidth=1.5,
            markersize=5,
            label="Back-test forecast (champion)",
            zorder=3,
        )

    if next_forecast is not None and next_month in month_index:
        position = month_index[next_month]
        error = z * next_sigma if next_sigma is not None else 0.0
        ax.errorbar(
            [position],
            [next_forecast],
            yerr=[[error], [error]],
            fmt="D",
            color=PALETTE[3],
            markersize=8,
            capsize=6,
            linewidth=1.8,
            label=f"Next-month forecast (± {z:g}·σ)",
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(months, rotation=90)
    ax.legend(loc="upper left")

    badge = f" (ABC {abc_class})" if abc_class else ""
    finalize(
        fig,
        title=f"{stock_code} — {description}{badge}: monthly units sold",
        xlabel="Month",
        ylabel="Units sold (units)",
    )
    return fig
