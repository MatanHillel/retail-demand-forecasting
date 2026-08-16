"""E8 — Intermittency and the zero-target share by k (US-10, PRD §35A E8, §14).

**Intermittent demand** means a product sells in some months and not in others. This catalogue is
full of it, and that single fact drives three design decisions: the active-product rule exists at
all, `k` is set to six months, and the error metric is wMAPE rather than a percentage error that
would divide by zero.

The k-sweep is the documented justification for the configured `k` (§14). Widening the window
admits more product-months into the model — but the extra ones are the marginal products, so the
share of rows whose target is zero rises with it. Both effects are measured here rather than
argued:

* small `k` → fewer rows, cleaner targets, but products that sell every other month get dropped;
* large `k` → more rows, more zeros, and a model that spends its capacity predicting nothing.

The active rule itself is **not** re-implemented: :func:`pipeline.active.active_mask` is the one
definition (§14), shared with feature engineering (US-13), so the EDA curve and the model can
never disagree about what "active" means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.active import active_mask
from pipeline.config import CleaningConfig, load_model_config
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E8"

#: Window lengths swept for the justification table (§35A E8). Not a tunable threshold — the
#: configured `k` lives in ``model_config.yaml → active_rule.k`` and is flagged in the output.
K_GRID: tuple[int, ...] = (1, 3, 6, 12)


def _full_months(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Panel rows outside the partial month — a partial month is never scored (§8)."""
    return panel_df.loc[~panel_df["month"].astype(str).isin(cfg.raw.partial_months), :]


def _zero_share_per_product(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Per product: months present in the panel, how many sold nothing, and the share."""
    full = _full_months(panel_df, cfg)
    frame = (
        full.groupby(full["stock_code"].astype(str), sort=True)
        .agg(
            panel_months=("units_sold", "size"),
            zero_months=("units_sold", lambda values: int((values == 0).sum())),
            units=("units_sold", "sum"),
        )
        .reset_index()
    )
    frame["zero_share"] = frame["zero_months"] / frame["panel_months"]
    return frame


def _zero_share_by_k(panel_df: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Active rows and zero-target share for each candidate window length.

    Restricted to the months the model actually targets: ``split.first_target_month`` through the
    last full month. Earlier months cannot be targets (there is no history to look back on) and
    the partial month is never scored.
    """
    model_cfg = load_model_config()
    first_target = model_cfg.split.first_target_month
    last_full = cfg.raw.last_full_month

    months = panel_df["month"].astype(str)
    in_window = (months >= first_target) & (months <= last_full)
    targets = panel_df.loc[in_window, ["stock_code", "month", "units_sold"]].copy()
    targets["stock_code"] = targets["stock_code"].astype(str)
    targets["month"] = targets["month"].astype(str)

    rows = []
    for k in K_GRID:
        mask = active_mask(panel_df, k)
        mask = mask.assign(
            stock_code=mask["stock_code"].astype(str), month=mask["month"].astype(str)
        )
        merged = targets.merge(mask, on=["stock_code", "month"], how="left")
        active = merged.loc[merged["is_active"].fillna(False)]
        zero_rows = int((active["units_sold"] == 0).sum())
        rows.append(
            {
                "k": k,
                "rows": int(len(active)),
                "products": int(active["stock_code"].nunique()),
                "zero_rows": zero_rows,
                "zero_target_share": zero_rows / len(active) if len(active) else 0.0,
                "first_target_month": first_target,
                "last_target_month": last_full,
                "is_configured_k": k == model_cfg.active_rule.k,
            }
        )
    return pd.DataFrame(rows)


def _hist_figure(per_product: pd.DataFrame, ctx: RunContext) -> Path:
    """How the zero share is distributed across the catalogue."""
    apply_style()
    fig, ax = plt.subplots()
    ax.hist(per_product["zero_share"], bins=20, range=(0.0, 1.0), color=PALETTE[0])
    finalize(
        fig,
        title="How intermittent each product is",
        xlabel="Share of the product's panel months with no sales",
        ylabel="Products (count)",
    )
    save_figure(fig, "E08_zero_share_hist", ctx)
    return figure_path("E08_zero_share_hist")


def _by_k_figure(by_k: pd.DataFrame, ctx: RunContext) -> Path:
    """Rows admitted and zero-target share against k, on twin axes (§35A E8)."""
    apply_style()
    fig, ax = plt.subplots()
    positions = list(range(len(by_k)))
    ax.bar(positions, by_k["rows"], color=PALETTE[0], label="active product-months")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(value) for value in by_k["k"]])

    twin = ax.twinx()
    twin.plot(
        positions,
        by_k["zero_target_share"],
        marker="o",
        color=PALETTE[3],
        linewidth=2,
        label="zero-target share",
    )
    twin.set_ylabel("Share of rows whose target is zero")
    twin.grid(False)

    configured = by_k.loc[by_k["is_configured_k"]]
    for position, row in zip(positions, by_k.itertuples(), strict=True):
        if row.is_configured_k:
            ax.axvline(position, color=PALETTE[7], linestyle="--", linewidth=1.2)
            ax.annotate(
                f"configured k = {int(row.k)}",
                xy=(position, 0),
                xytext=(6, 8),
                textcoords="offset points",
                fontsize=9,
            )
    if configured.empty:  # pragma: no cover - configuration always names one of the grid values
        ctx.warn("the configured k is not one of the swept values, so the figure marks none")

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left")

    finalize(
        fig,
        title="Widening the active window admits more rows and more zeros",
        xlabel="k (months of history the active rule looks back over)",
        ylabel="Active product-months (rows)",
    )
    save_figure(fig, "E08_zero_share_by_k", ctx)
    return figure_path("E08_zero_share_by_k")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E8 and write its two tables and two figures."""
    per_product = _zero_share_per_product(panel_df, cfg)
    by_k = _zero_share_by_k(panel_df, cfg)

    tables = {"E08_zero_share_per_product": per_product, "E08_zero_share_by_k": by_k}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {
        "E08_zero_share_hist": _hist_figure(per_product, ctx),
        "E08_zero_share_by_k": _by_k_figure(by_k, ctx),
    }
    return {"tables": tables, "figures": figures}
