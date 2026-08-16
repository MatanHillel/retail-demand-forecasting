"""E2 — Demand over time, seasonality and year-on-year growth (US-09, PRD §35A E2).

Answers the first half of SQ1, "what has happened in the business": how much was sold each month,
which calendar months are peaks, whether the business grew, and how much of the year lands in the
September–November run-up. That last number is the documented reason the hold-out window includes
the peak instead of stopping before it (§21) — a model validated only on quiet months would be
scored on data that looks nothing like the months that matter.

December 2011 is shown but never counted: it is a partial month (§8), so it is excluded from every
total, share and growth figure and drawn hatched in the monthly series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.periods import (
    MONTHS_PER_YEAR,
    full_years,
    in_window,
    month_to_period,
    partial_positions,
    year_label,
)
from pipeline.eda.style import PALETTE, apply_style, finalize, hatch_partial, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E2"

#: The autumn run-up (§35A E2, §21). September, October, November as month-of-year numbers —
#: the window whose share of the year justifies the hold-out covering the peak.
PEAK_MONTHS: tuple[int, ...] = (9, 10, 11)
PEAK_LABEL = "Sep-Nov"


def _monthly(clean_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Units, revenue and distinct invoices per month.

    Invoices are counted from the transaction table, not summed from the panel's
    ``invoice_count``: that column counts invoices *per product*, so adding it up would count one
    basket once for every line it contains.
    """
    grouped = (
        clean_df.groupby(clean_df["month"].astype(str), sort=True)
        .agg(
            units=("quantity", "sum"),
            revenue=("line_revenue", "sum"),
            invoices=("invoice", "nunique"),
        )
        .reset_index()
        .rename(columns={"index": "month"})
    )
    grouped["is_partial"] = grouped["month"].isin(cfg.raw.partial_months)
    return grouped


def _seasonal_index(monthly: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Mean share of the annual units falling in each calendar month, scaled so average = 1.0.

    Averaging the *shares* of the two complete years rather than pooling raw units keeps a bigger
    year from dominating the shape. Index 1.5 means "this month sells half as much again as an
    average month".
    """
    windows = full_years(cfg)
    shares: list[pd.DataFrame] = []
    for window in windows:
        year = monthly.loc[in_window(monthly["month"], window)].copy()
        total = float(year["units"].sum())
        year["month_of_year"] = month_to_period(year["month"]).month
        year["share"] = year["units"] / total if total else 0.0
        shares.append(year[["month_of_year", "share"]])

    if not shares:
        return pd.DataFrame(columns=["month_of_year", "mean_share", "seasonal_index", "years"])

    stacked = pd.concat(shares, ignore_index=True)
    result = (
        stacked.groupby("month_of_year", sort=True)["share"]
        .mean()
        .reset_index(name="mean_share")
    )
    result["seasonal_index"] = result["mean_share"] * MONTHS_PER_YEAR
    result["years"] = len(windows)
    return result


def _yoy(monthly: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Units and revenue per complete Dec→Nov year, with growth against the previous one."""
    rows = []
    for window in full_years(cfg):
        year = monthly.loc[in_window(monthly["month"], window)]
        rows.append(
            {
                "year": year_label(window),
                "first_month": window[0],
                "last_month": window[1],
                "units": int(year["units"].sum()),
                "revenue": float(year["revenue"].sum()),
                "invoices": int(year["invoices"].sum()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["units_growth"] = frame["units"].pct_change()
    frame["revenue_growth"] = frame["revenue"].pct_change()
    return frame


def _peak_share(monthly: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Share of each year's units and revenue that falls in the September-November window."""
    rows = []
    for window in full_years(cfg):
        year = monthly.loc[in_window(monthly["month"], window)].copy()
        year["month_of_year"] = month_to_period(year["month"]).month
        peak = year.loc[year["month_of_year"].isin(PEAK_MONTHS)]
        rows.append(
            {
                "year": year_label(window),
                "window": PEAK_LABEL,
                "units": int(peak["units"].sum()),
                "units_share": float(peak["units"].sum() / year["units"].sum())
                if year["units"].sum()
                else 0.0,
                "revenue": float(peak["revenue"].sum()),
                "revenue_share": float(peak["revenue"].sum() / year["revenue"].sum())
                if year["revenue"].sum()
                else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    pooled = {
        "year": "pooled",
        "window": PEAK_LABEL,
        "units": int(frame["units"].sum()),
        "units_share": float(frame["units"].sum() / _full_total(monthly, cfg, "units")),
        "revenue": float(frame["revenue"].sum()),
        "revenue_share": float(frame["revenue"].sum() / _full_total(monthly, cfg, "revenue")),
    }
    return pd.concat([frame, pd.DataFrame([pooled])], ignore_index=True)


def _full_total(monthly: pd.DataFrame, cfg: CleaningConfig, column: str) -> float:
    """Total of ``column`` across the complete years only — the denominator for a pooled share."""
    mask = pd.Series(False, index=monthly.index)
    for window in full_years(cfg):
        mask |= in_window(monthly["month"], window)
    total = float(monthly.loc[mask, column].sum())
    return total if total else 1.0


def _monthly_figure(monthly: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext) -> Path:
    """Two stacked panels — units and revenue per month — with the partial month hatched."""
    apply_style()
    months = monthly["month"].tolist()
    positions = list(range(len(months)))
    partial = partial_positions(months, cfg)

    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(10.0, 7.0))
    top.plot(positions, monthly["units"], marker="o", color=PALETTE[0], linewidth=2)
    bottom.plot(positions, monthly["revenue"], marker="o", color=PALETTE[1], linewidth=2)
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
    save_figure(fig, "E02_monthly_units", ctx)
    return figure_path("E02_monthly_units")


def _seasonal_figure(seasonal: pd.DataFrame, ctx: RunContext) -> Path:
    """Bar chart of the seasonal index with the average-month reference line at 1.0."""
    apply_style()
    fig, ax = plt.subplots()
    ax.bar(seasonal["month_of_year"], seasonal["seasonal_index"], color=PALETTE[0])
    ax.axhline(1.0, color=PALETTE[7], linestyle="--", linewidth=1.2, label="average month = 1.0")
    ax.set_xticks(seasonal["month_of_year"])
    ax.legend(loc="upper left")
    finalize(
        fig,
        title="Seasonal index by calendar month (mean of two Dec-Nov years)",
        xlabel="Calendar month (1 = January)",
        ylabel="Seasonal index (1.0 = average month)",
    )
    save_figure(fig, "E02_seasonal_index", ctx)
    return figure_path("E02_seasonal_index")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E2 and write its four tables and two figures."""
    monthly = _monthly(clean_df, cfg)
    seasonal = _seasonal_index(monthly, cfg)
    yoy = _yoy(monthly, cfg)
    peak = _peak_share(monthly, cfg)

    tables = {
        "E02_monthly_units_revenue": monthly,
        "E02_seasonal_index": seasonal,
        "E02_yoy": yoy,
        "E02_sep_nov_share": peak,
    }
    for name in ("E02_monthly_units_revenue", "E02_seasonal_index", "E02_yoy"):
        save_table(tables[name], name, ctx)
    save_table(peak, "E02_sep_nov_share", ctx, fmt="json")

    figures = {
        "E02_monthly_units": _monthly_figure(monthly, cfg, ctx),
        "E02_seasonal_index": _seasonal_figure(seasonal, ctx),
    }
    return {"tables": tables, "figures": figures}
