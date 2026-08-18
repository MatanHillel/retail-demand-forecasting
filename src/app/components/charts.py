"""Matplotlib/Seaborn chart helpers for the Streamlit shell (US-27, PRD §33, §35A.3).

Every figure reuses ``pipeline.eda.style`` — the palette, the ABC colours, the partial-month
hatching, the title/footnote furniture — so the app reads as the same visual system as the EDA
report. Figures mirror ``pipeline/eda/e02_demand_over_time.py``'s ``_monthly_figure`` /
``_seasonal_figure`` but return the live figure for ``st.pyplot`` instead of saving to disk.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline.config import MODEL_IDS
from pipeline.eda.style import (
    ABC_COLORS,
    BASE_FONT_SIZE,
    DEFAULT_FOOTNOTE,
    PALETTE,
    apply_style,
    finalize,
    hatch_partial,
)

#: Same colour per model id everywhere a chart needs one, in :data:`MODEL_IDS` order (§35A.2 — a
#: fixed mapping, same reasoning as :data:`pipeline.eda.style.ABC_COLORS`).
MODEL_COLORS: dict[str, str] = dict(zip(MODEL_IDS, PALETTE, strict=False))


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


def _footnote(fig: matplotlib.figure.Figure) -> None:
    """Source footnote for multi-axes figures :func:`finalize` cannot furnish (it only touches the
    figure's first axes)."""
    fig.text(
        0.01,
        0.01,
        DEFAULT_FOOTNOTE,
        ha="left",
        va="bottom",
        fontsize=BASE_FONT_SIZE - 3,
        color="#444444",
    )


def model_line_chart(
    df: pd.DataFrame,
    value_col: str,
    *,
    x_col: str,
    ylabel: str,
    title: str,
    models: tuple[str, ...] = MODEL_IDS,
) -> matplotlib.figure.Figure:
    """One line per candidate model, coloured by :data:`MODEL_COLORS` (Screen 4 by-month /
    back-test-consistency tabs, PRD §33.4)."""
    apply_style()
    fig, ax = plt.subplots()
    for model_id in models:
        sub = df.loc[df["model"] == model_id].sort_values(x_col)
        if sub.empty:
            continue
        ax.plot(
            sub[x_col],
            sub[value_col],
            marker="o",
            label=model_id,
            color=MODEL_COLORS.get(model_id),
        )
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="best", fontsize=8)
    finalize(fig, title=title, xlabel="Month", ylabel=ylabel)
    return fig


def model_abc_grouped_bars(
    df: pd.DataFrame,
    value_col: str,
    *,
    ylabel: str,
    title: str,
    models: tuple[str, ...] = MODEL_IDS,
    abc_classes: tuple[str, ...] = ("A", "B", "C"),
) -> matplotlib.figure.Figure:
    """Grouped bars, one group per model, bars coloured by ABC class (Screen 4 by-ABC tab, §23,
    §35A.2 — ABC classes always the same three colours)."""
    apply_style()
    fig, ax = plt.subplots()
    positions = np.arange(len(models))
    width = 0.8 / len(abc_classes)
    indexed = df.set_index(["model", "abc_class"])[value_col]
    for offset, abc_class in enumerate(abc_classes):
        values = [indexed.get((model_id, abc_class), float("nan")) for model_id in models]
        ax.bar(
            positions + (offset - (len(abc_classes) - 1) / 2) * width,
            values,
            width=width,
            label=f"ABC {abc_class}",
            color=ABC_COLORS.get(abc_class),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="best")
    finalize(fig, title=title, xlabel="Model", ylabel=ylabel)
    return fig


def actual_vs_forecast_monthly(
    rows_df: pd.DataFrame, *, model_id: str
) -> matplotlib.figure.Figure:
    """Monthly totals, actual vs forecast, for one model (Screen 4 actual-vs-forecast tab)."""
    apply_style()
    pred_col = f"pred_{model_id}"
    monthly = (
        rows_df.groupby("target_month", as_index=False)[["actual", pred_col]]
        .sum()
        .sort_values("target_month")
    )
    fig, ax = plt.subplots()
    ax.plot(
        monthly["target_month"], monthly["actual"], marker="o", color=PALETTE[0], label="Actual"
    )
    ax.plot(
        monthly["target_month"], monthly[pred_col], marker="o", color=PALETTE[1], label="Forecast"
    )
    ax.tick_params(axis="x", rotation=90)
    ax.legend(loc="best")
    finalize(
        fig,
        title=f"Monthly totals, actual vs forecast — {model_id}",
        xlabel="Month",
        ylabel="Units sold (units)",
    )
    return fig


def actual_vs_forecast_scatter(
    rows_df: pd.DataFrame, *, model_id: str, sample_size: int, seed: int
) -> matplotlib.figure.Figure:
    """Log-log scatter of a product-month sample, actual vs forecast (Screen 4)."""
    apply_style()
    pred_col = f"pred_{model_id}"
    sample = rows_df.loc[(rows_df["actual"] > 0) & (rows_df[pred_col] > 0), ["actual", pred_col]]
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed)

    fig, ax = plt.subplots()
    ax.scatter(sample["actual"], sample[pred_col], alpha=0.35, s=12, color=PALETTE[0])
    if not sample.empty:
        lower = min(sample["actual"].min(), sample[pred_col].min())
        upper = max(sample["actual"].max(), sample[pred_col].max())
        ax.plot(
            [lower, upper],
            [lower, upper],
            color=PALETTE[7],
            linestyle="--",
            linewidth=1,
            label="y = x",
        )
        ax.legend(loc="best")
    ax.set_xscale("log")
    finalize(
        fig,
        title=f"Product-month sample, actual vs forecast — {model_id} (n={len(sample):,})",
        xlabel="Actual units (units, log scale)",
        ylabel="Forecast units (units)",
        log_y=True,
    )
    return fig


def inventory_kpi_bar_pairs(
    rows: pd.DataFrame, *, series_col: str, metrics: tuple[tuple[str, str], ...], title: str
) -> matplotlib.figure.Figure:
    """One subplot per KPI, bars are ``rows`` grouped by ``series_col`` (Screen 5 KPI bar pairs,
    §30). ``rows`` has one row per series with the KPI columns already computed."""
    apply_style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    labels = rows[series_col].tolist()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    for ax, (col, label) in zip(axes, metrics, strict=True):
        ax.bar(labels, rows[col], color=colors)
        ax.set_title(label, fontsize=BASE_FONT_SIZE - 1)
        ax.tick_params(axis="x", rotation=30, labelsize=BASE_FONT_SIZE - 3)
    fig.suptitle(title, fontsize=BASE_FONT_SIZE + 1)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.93))
    _footnote(fig)
    return fig


def fill_rate_vs_excess_curve(
    df: pd.DataFrame, *, series_col: str, series_values: tuple[str, ...]
) -> matplotlib.figure.Figure:
    """Fill rate vs excess units across the z options, one line per series (Screen 5, §25, §30)."""
    apply_style()
    fig, ax = plt.subplots()
    for series in series_values:
        sub = df.loc[df[series_col] == series].sort_values("z")
        if sub.empty:
            continue
        color = MODEL_COLORS.get(series)
        ax.plot(sub["fill_rate"], sub["excess_units"], marker="o", label=series, color=color)
        for _, row in sub.iterrows():
            ax.annotate(
                f"z={row['z']:g}",
                (row["fill_rate"], row["excess_units"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(4, 4),
            )
    ax.legend(loc="best")
    finalize(
        fig,
        title="Fill rate vs excess units across z options",
        xlabel="Fill rate (share of demand fulfilled)",
        ylabel="Excess units (units)",
    )
    return fig


def inventory_group_bars(
    panels: tuple[tuple[str, pd.DataFrame], ...],
    *,
    metric_col: str,
    group_col: str,
    group_order: list[str],
    ylabel: str,
    title: str,
    color_map: dict[str, str] | None = None,
) -> matplotlib.figure.Figure:
    """One subplot per ``panels`` entry (e.g. per model), bars across ``group_order`` (by-month /
    by-ABC breakdowns, Screen 5, §30). ``color_map`` gives per-group bar colours (ABC colours for
    the by-ABC breakdown); omitted for the by-month breakdown, which uses the default palette."""
    apply_style()
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (label, sub) in zip(axes, panels, strict=True):
        indexed = sub.set_index(group_col)[metric_col]
        values = [indexed.get(group, float("nan")) for group in group_order]
        if color_map:
            colors = [color_map.get(group, PALETTE[0]) for group in group_order]
        else:
            colors = PALETTE[0]
        ax.bar(group_order, values, color=colors)
        ax.set_title(label, fontsize=BASE_FONT_SIZE - 1)
        ax.tick_params(axis="x", rotation=45, labelsize=BASE_FONT_SIZE - 3)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=BASE_FONT_SIZE + 1)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.9))
    _footnote(fig)
    return fig
