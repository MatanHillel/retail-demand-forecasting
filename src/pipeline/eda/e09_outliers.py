"""E9 — Demand magnitude and outliers (US-10, PRD §35A E9, §23, §26, §47).

The size distribution of this business is extreme: most product-months are a handful of units, a
few are tens of thousands. Two modelling decisions follow directly, and this analysis is the
evidence for both:

* **wMAPE, not MAPE** (§23). A percentage error averaged per product would let a product that
  sold three units dominate one that sold thirty thousand. Weighting by volume does not.
* **Robust σ, not a standard deviation** (§26). A standard deviation is defined by the squares of
  the deviations, so a handful of 80,000-unit wholesale lines would inflate the safety stock of
  every product that shares their ABC group. The MAD-based estimate ignores them.

A **log scale** is an axis where each step multiplies rather than adds, which is the only way a
range from 1 to 80,000 fits on one chart. Both scaled axes say so in their label (§35A.2).
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

ANALYSIS_ID = "E9"

#: Length of each "largest" ranking (§35A E9).
TOP_N = 20

#: Bars in the log-spaced histogram.
_LOG_BINS = 40


def _largest_lines(clean_df: pd.DataFrame) -> pd.DataFrame:
    """The twenty biggest single sales lines, with everything needed to judge them."""
    ranked = clean_df.sort_values(
        ["quantity", "invoice", "stock_code"], ascending=[False, True, True], kind="mergesort"
    ).head(TOP_N)
    frame = pd.DataFrame(
        {
            "invoice": ranked["invoice"].astype(str),
            "stock_code": ranked["stock_code"].astype(str),
            "description": ranked["description"].astype("string"),
            "quantity": pd.to_numeric(ranked["quantity"]),
            "price": pd.to_numeric(ranked["price"]),
            "line_revenue": pd.to_numeric(ranked["line_revenue"]),
            "customer_identified": ranked["customer_id"].notna(),
            "country": ranked["country"].astype(str),
            "invoice_date": pd.to_datetime(ranked["invoice_date"]).dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "month": ranked["month"].astype(str),
        }
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def _largest_product_months(panel_df: pd.DataFrame) -> pd.DataFrame:
    """The twenty biggest product-months — the extreme values of the model's actual target."""
    ranked = panel_df.sort_values(
        ["units_sold", "stock_code", "month"], ascending=[False, True, True], kind="mergesort"
    ).head(TOP_N)
    frame = pd.DataFrame(
        {
            "month": ranked["month"].astype(str),
            "stock_code": ranked["stock_code"].astype(str),
            "description": ranked["description"].astype("string"),
            "units_sold": pd.to_numeric(ranked["units_sold"]),
            "gross_revenue": pd.to_numeric(ranked["gross_revenue"]),
            "max_line_qty": pd.to_numeric(ranked["max_line_qty"]),
            "invoice_count": pd.to_numeric(ranked["invoice_count"]),
            "is_partial_month": ranked["is_partial_month"].astype(bool),
        }
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame


def _magnitude_summary(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Quantiles of the non-zero product-month target, plus the abnormal-line threshold."""
    selling = pd.to_numeric(panel_df.loc[panel_df["units_sold"] > 0, "units_sold"])
    quantiles = {
        "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75,
        "p90": 0.90, "p99": 0.99, "p999": 0.999,
    }
    rows = [
        {"metric": "units_sold_nonzero", "statistic": "count", "value": float(len(selling))},
        {"metric": "units_sold_nonzero", "statistic": "mean", "value": float(selling.mean())},
        {"metric": "units_sold_nonzero", "statistic": "max", "value": float(selling.max())},
    ]
    rows += [
        {
            "metric": "units_sold_nonzero",
            "statistic": name,
            "value": float(selling.quantile(share)),
        }
        for name, share in quantiles.items()
    ]
    rows.append(
        {
            "metric": "abnormal_line_quantity_threshold",
            "statistic": "value",
            "value": float(cfg.warnings.abnormal_line_quantity),
        }
    )
    return pd.DataFrame(rows)


def _hist_figure(panel_df: pd.DataFrame, ctx: RunContext) -> Path:
    """Log-scaled histogram of the non-zero product-month target."""
    apply_style()
    selling = pd.to_numeric(panel_df.loc[panel_df["units_sold"] > 0, "units_sold"])
    bins = np.logspace(0, np.log10(max(float(selling.max()), 10.0)), _LOG_BINS)

    fig, ax = plt.subplots()
    ax.hist(selling, bins=bins, color=PALETTE[0])
    ax.set_xscale("log")
    finalize(
        fig,
        title="Most product-months are small; a few are enormous",
        xlabel=f"Units sold in a product-month (units){LOG_SCALE_SUFFIX}",
        ylabel="Product-months (count)",
        log_y=True,
    )
    save_figure(fig, "E09_units_hist_log", ctx)
    return figure_path("E09_units_hist_log")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E9 and write its three tables and one figure."""
    tables = {
        "E09_largest_lines": _largest_lines(clean_df),
        "E09_largest_product_months": _largest_product_months(panel_df),
        "E09_magnitude_summary": _magnitude_summary(panel_df, cfg),
    }
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E09_units_hist_log": _hist_figure(panel_df, ctx)}
    return {"tables": tables, "figures": figures}
