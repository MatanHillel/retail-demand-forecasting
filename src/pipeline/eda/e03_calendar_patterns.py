"""E3 — Weekday and hour-of-day trading patterns (US-09, PRD §35A E3).

Pure context for the reader: it shows *when* orders are placed, which explains two things a
monthly model would otherwise make look strange — trading is a weekday office-hours business, and
Saturday is essentially closed.

**Never a feature.** The model forecasts a month at a time, so a weekday or an hour cannot enter
it; these tables exist to describe the business, not to predict it (§35A E3, §9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E3"

#: ``dt.dayofweek`` numbers Monday 0 … Sunday 6.
WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _by_weekday(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Units, sales lines and distinct invoices by day of the week, Monday first."""
    dates = pd.to_datetime(clean_df["invoice_date"])
    frame = pd.DataFrame(
        {
            "weekday": dates.dt.dayofweek,
            "quantity": clean_df["quantity"],
            "invoice": clean_df["invoice"],
        }
    )
    grouped = (
        frame.groupby("weekday", sort=True)
        .agg(units=("quantity", "sum"), lines=("quantity", "size"),
             invoices=("invoice", "nunique"))
        .reset_index()
    )
    grouped["weekday_name"] = grouped["weekday"].map(dict(enumerate(WEEKDAY_NAMES)))
    return grouped[["weekday", "weekday_name", "units", "lines", "invoices"]]


def _by_hour(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Units, sales lines and distinct invoices by hour of the day."""
    dates = pd.to_datetime(clean_df["invoice_date"])
    frame = pd.DataFrame(
        {
            "hour": dates.dt.hour,
            "quantity": clean_df["quantity"],
            "invoice": clean_df["invoice"],
        }
    )
    return (
        frame.groupby("hour", sort=True)
        .agg(units=("quantity", "sum"), lines=("quantity", "size"),
             invoices=("invoice", "nunique"))
        .reset_index()
    )


def _bar_figure(
    frame: pd.DataFrame, labels: list[str], name: str, title: str, xlabel: str, ctx: RunContext
) -> Path:
    apply_style()
    fig, ax = plt.subplots()
    positions = list(range(len(frame)))
    ax.bar(positions, frame["units"], color=PALETTE[0])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45 if len(labels) <= len(WEEKDAY_NAMES) else 0)
    finalize(fig, title=title, xlabel=xlabel, ylabel="Units sold (units)")
    save_figure(fig, name, ctx)
    return figure_path(name)


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E3 and write its two tables and two figures."""
    weekday = _by_weekday(clean_df)
    hour = _by_hour(clean_df)

    tables = {"E03_weekday": weekday, "E03_hour": hour}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {
        "E03_weekday": _bar_figure(
            weekday,
            weekday["weekday_name"].tolist(),
            "E03_weekday",
            "Units sold by day of the week",
            "Day of the week",
            ctx,
        ),
        "E03_hour": _bar_figure(
            hour,
            [str(value) for value in hour["hour"]],
            "E03_hour",
            "Units sold by hour of the day",
            "Hour of the day (24-hour clock)",
            ctx,
        ),
    }
    return {"tables": tables, "figures": figures}
