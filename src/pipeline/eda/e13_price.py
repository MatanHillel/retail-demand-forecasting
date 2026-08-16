"""E13 — Unit price: level and stability (US-10, PRD §35A E13).

Supports one feature, ``avg_unit_price_lag_1``. A lagged price is only worth carrying if price
actually moves and if last month's price says something about this month's — so this analysis
measures both the level and the stability.

**Coefficient of variation (CV)** is the standard deviation divided by the mean: a unit-free
measure of how much a product's price bounces around. CV 0.0 means the price never changed; 0.5
means the typical swing is half the average price. A catalogue of near-zero CVs would mean the
lagged price is nearly a constant per product, and the model would get most of its value from the
product identity instead.

Prices are read from the panel's ``avg_unit_price``, restricted to months the product actually
sold in: a zero-sales month carries the last known price forward (US-05), which is the right
behaviour for a feature but would understate volatility if counted as a real observation here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import LOG_SCALE_SUFFIX, PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E13"

#: Fewest selling months a product needs before its price variation means anything.
MIN_MONTHS_FOR_CV = 2

_QUANTILES = {"p01": 0.01, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75,
              "p90": 0.90, "p99": 0.99}
_LOG_BINS = 40


def _describe(values: pd.Series, metric: str, unit: str) -> list[dict[str, Any]]:
    numeric = pd.to_numeric(values).dropna()
    rows = [
        {"metric": metric, "unit": unit, "statistic": "count", "value": float(len(numeric))},
        {"metric": metric, "unit": unit, "statistic": "mean", "value": float(numeric.mean())},
        {"metric": metric, "unit": unit, "statistic": "min", "value": float(numeric.min())},
    ]
    rows += [
        {"metric": metric, "unit": unit, "statistic": name, "value": float(numeric.quantile(q))}
        for name, q in _QUANTILES.items()
    ]
    rows.append(
        {"metric": metric, "unit": unit, "statistic": "max", "value": float(numeric.max())}
    )
    return rows


def _stability(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Per product: mean price across its selling months and the coefficient of variation."""
    selling = panel_df.loc[panel_df["units_sold"] > 0, ["stock_code", "avg_unit_price"]].copy()
    selling["stock_code"] = selling["stock_code"].astype(str)
    grouped = (
        selling.groupby("stock_code", sort=True)["avg_unit_price"]
        .agg(selling_months="size", mean_price="mean", std_price="std")
        .reset_index()
    )
    grouped["std_price"] = grouped["std_price"].fillna(0.0)
    grouped["price_cv"] = np.where(
        (grouped["selling_months"] >= MIN_MONTHS_FOR_CV) & (grouped["mean_price"] > 0),
        grouped["std_price"] / grouped["mean_price"],
        np.nan,
    )
    return grouped


def _distribution(clean_df: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of the line-level unit price and of the per-product CV, in one long table."""
    rows = _describe(clean_df["price"], "unit_price", "GBP per unit")
    rows += _describe(stability["price_cv"], "price_cv", "ratio (std / mean)")
    rows.append(
        {
            "metric": "price_cv",
            "unit": "ratio (std / mean)",
            "statistic": "products_with_too_few_months",
            "value": float(stability["price_cv"].isna().sum()),
        }
    )
    return pd.DataFrame(rows)


def _figure(clean_df: pd.DataFrame, stability: pd.DataFrame, ctx: RunContext) -> Path:
    """Price level (log-scaled) beside how much each product's price moves."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    price = pd.to_numeric(clean_df["price"])
    positive = price.loc[price > 0]
    bins = np.logspace(
        np.log10(max(float(positive.min()), 0.01)),
        np.log10(max(float(positive.max()), 1.0)),
        _LOG_BINS,
    )
    left.hist(positive, bins=bins, color=PALETTE[0])
    left.set_xscale("log")
    left.set_xlabel(f"Unit price (GBP){LOG_SCALE_SUFFIX}")
    left.set_ylabel("Sales lines (count)")

    cv = stability["price_cv"].dropna()
    right.hist(cv, bins=40, color=PALETTE[1])
    right.set_xlabel("Price coefficient of variation (std / mean)")
    right.set_ylabel("Products (count)")

    finalize(
        fig,
        title="Unit price: what things cost, and how much each product's price moves",
        xlabel=f"Unit price (GBP){LOG_SCALE_SUFFIX}",
        ylabel="Sales lines (count)",
    )
    save_figure(fig, "E13_price_hist", ctx)
    return figure_path("E13_price_hist")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E13 and write its two tables and one figure."""
    stability = _stability(panel_df)
    distribution = _distribution(clean_df, stability)

    tables = {
        "E13_price_distribution": distribution,
        "E13_price_stability": stability,
    }
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E13_price_hist": _figure(clean_df, stability, ctx)}
    return {"tables": tables, "figures": figures}
