"""E12 — Returns and cancellations (US-10, PRD §35A E12, §9, §13.2).

This analysis exists to justify one definition. The target is **gross demand**: units on positive
sales lines, with cancellations *not* subtracted (§9). That looks wrong at first glance — the
goods came back, so surely they were not demanded? — and the reasoning is inventory, not
accounting: the stock had to be on the shelf to fulfil the original order. Netting returns off
would systematically under-forecast exactly the products that get returned most.

So ``returned_units`` is measured here, reported here, and never becomes a feature (§13.2).

The returns side table is read from ``returns_lines.parquet``, the lines :mod:`pipeline.cleaning`
removed at step 5. Some of those stock codes never appear in the panel — a product can be
returned in a month it never sold, or be a code the exclusion list removed — so the join is
explicitly left-sided and the unmatched volume is reported rather than silently dropped (§3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME
from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_path, save_figure, save_table
from pipeline.eda.periods import partial_positions
from pipeline.eda.style import PALETTE, apply_style, finalize, hatch_partial, plt
from pipeline.run_context import RunContext

ANALYSIS_ID = "E12"

#: Length of the returned-products ranking (§35A E12).
TOP_N = 20


def load_returns(ctx: RunContext) -> pd.DataFrame:
    """Read the cancellation lines US-04 set aside, this run's staged copy first.

    Resolved the way :func:`pipeline.eda.io.load_table` resolves a table and deliberately not
    through ``ctx.out()``, which would register the path for promotion and make ``promote()``
    warn about a file this run only read.
    """
    relative = (paths.PROCESSED_DIR / RETURNS_FILENAME).relative_to(paths.PROJECT_ROOT)
    for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
        if candidate.is_file():
            return pd.read_parquet(candidate)
    raise FileNotFoundError(
        f"{relative.as_posix()} not found in staging or its final location — it is written by "
        "pipeline.cleaning (US-04)"
    )


def _monthly(
    returns: pd.DataFrame, clean_df: pd.DataFrame, cfg: CleaningConfig
) -> pd.DataFrame:
    """Cancellation rows and units per month against the sales of the same month."""
    sales = (
        clean_df.groupby(clean_df["month"].astype(str), sort=True)
        .agg(sales_rows=("quantity", "size"), sold_units=("quantity", "sum"))
        .reset_index()
    )
    returned = (
        returns.groupby(returns["month"].astype(str), sort=True)
        .agg(
            cancellation_rows=("quantity_abs", "size"),
            returned_units=("quantity_abs", "sum"),
            cancellation_invoices=("invoice", "nunique"),
        )
        .reset_index()
    )
    frame = sales.merge(returned, on="month", how="outer").sort_values("month")
    for column in (
        "sales_rows", "sold_units", "cancellation_rows", "returned_units",
        "cancellation_invoices",
    ):
        frame[column] = frame[column].fillna(0).astype(int)

    frame["returned_units_share"] = frame.apply(
        lambda row: row["returned_units"] / row["sold_units"] if row["sold_units"] else 0.0,
        axis=1,
    )
    frame["cancellation_row_share"] = frame.apply(
        lambda row: row["cancellation_rows"] / (row["sales_rows"] + row["cancellation_rows"])
        if (row["sales_rows"] + row["cancellation_rows"])
        else 0.0,
        axis=1,
    )
    frame["is_partial"] = frame["month"].isin(cfg.raw.partial_months)
    return frame.reset_index(drop=True)


def _top_returned(returns: pd.DataFrame, panel_df: pd.DataFrame, ctx: RunContext) -> pd.DataFrame:
    """The twenty most-returned products, with each one's returned-to-sold ratio."""
    returned = (
        returns.groupby(returns["stock_code"].astype(str), sort=True)["quantity_abs"]
        .sum()
        .reset_index(name="returned_units")
    )

    sold = (
        panel_df.groupby(panel_df["stock_code"].astype(str), sort=False)
        .agg(sold_units=("units_sold", "sum"))
        .reset_index()
    )
    descriptions = (
        panel_df.dropna(subset=["description"])
        .drop_duplicates("stock_code", keep="first")
    )
    description_by_code = pd.Series(
        descriptions["description"].to_numpy(),
        index=descriptions["stock_code"].astype(str).to_numpy(),
    )

    merged = returned.merge(sold, on="stock_code", how="left")
    unmatched = merged["sold_units"].isna()
    if bool(unmatched.any()):
        # §3: a returned stock code need not exist in the panel. Report it, never drop it silently.
        ctx.warn(
            f"{int(unmatched.sum())} returned stock code(s) covering "
            f"{int(merged.loc[unmatched, 'returned_units'].sum())} returned units have no row in "
            "the panel (returned in a month they never sold, or excluded as non-inventory)"
        )
    merged["in_panel"] = ~unmatched
    merged["sold_units"] = merged["sold_units"].fillna(0).astype(int)
    merged["description"] = merged["stock_code"].map(description_by_code)
    merged["returned_to_sold_ratio"] = merged.apply(
        lambda row: row["returned_units"] / row["sold_units"] if row["sold_units"] else pd.NA,
        axis=1,
    )

    ranked = merged.sort_values(
        ["returned_units", "stock_code"], ascending=[False, True], kind="mergesort"
    ).head(TOP_N).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked[
        ["rank", "stock_code", "description", "returned_units", "sold_units",
         "returned_to_sold_ratio", "in_panel"]
    ]


def _figure(monthly: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext) -> Path:
    """Returned units as a share of units sold, month by month."""
    apply_style()
    months = monthly["month"].tolist()
    positions = list(range(len(months)))

    fig, ax = plt.subplots()
    ax.bar(positions, monthly["returned_units_share"], color=PALETTE[3])
    partial = partial_positions(months, cfg)
    if partial:
        hatch_partial(ax, partial)
    ax.set_xticks(positions)
    ax.set_xticklabels(months, rotation=90)
    finalize(
        fig,
        title="Returned units as a share of units sold (never subtracted from demand)",
        xlabel="Month",
        ylabel="Returned units / units sold",
    )
    save_figure(fig, "E12_cancellation_rate", ctx)
    return figure_path("E12_cancellation_rate")


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E12 and write its two tables and one figure."""
    returns = load_returns(ctx)
    monthly = _monthly(returns, clean_df, cfg)
    top_returned = _top_returned(returns, panel_df, ctx)

    tables = {
        "E12_cancellation_rate_monthly": monthly,
        "E12_top_returned": top_returned,
    }
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    figures = {"E12_cancellation_rate": _figure(monthly, cfg, ctx)}
    return {"tables": tables, "figures": figures}
