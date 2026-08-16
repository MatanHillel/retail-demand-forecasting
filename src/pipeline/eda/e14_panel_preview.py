"""E14 — Panel preview, read back from the hand-off file (US-10, PRD §35A E14).

The last check before the dataset contract is written. Every other analysis works on the panel
held in memory; this one deliberately re-reads ``clean_data.csv`` **from disk** and describes what
it finds, so the file that is actually handed to modelling — and committed to the repository — is
the thing being validated, not the DataFrame that was supposed to produce it.

That distinction has caught real problems elsewhere in this project: a plain ``pd.read_csv``
turns the stock code ``01234`` into the integer ``1234``. The file is therefore read through
:func:`pipeline.contract.read_panel`, which keeps the key columns as strings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.config import CleaningConfig
from pipeline.contract import read_panel
from pipeline.eda.io import save_table
from pipeline.run_context import RunContext

ANALYSIS_ID = "E14"


def load_panel_file(ctx: RunContext) -> tuple[pd.DataFrame, Path]:
    """Read ``clean_data.csv`` from this run's view of the tree: staged copy first.

    Under the Flow the panel this run produced is still in the staging tree; the final location
    holds the *previous* run's file until ``promote()``. Reading the final path there would
    describe the wrong dataset — the exact failure this analysis exists to catch.
    """
    relative = paths.CLEAN_DATA.relative_to(paths.PROJECT_ROOT)
    for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
        if candidate.is_file():
            return read_panel(candidate), candidate
    raise FileNotFoundError(
        f"{relative.as_posix()} not found in staging or its final location — it is written by "
        "pipeline.panel (US-05)"
    )


def _preview(panel: pd.DataFrame, cfg: CleaningConfig) -> pd.DataFrame:
    """Rows, zero rows, products and the partial flag, month by month."""
    months = panel["month"].astype(str)
    frame = (
        pd.DataFrame(
            {
                "month": months,
                "stock_code": panel["stock_code"].astype(str),
                "units_sold": pd.to_numeric(panel["units_sold"]),
                "is_partial_month": panel["is_partial_month"].astype(bool),
            }
        )
        .groupby("month", sort=True)
        .agg(
            rows=("units_sold", "size"),
            zero_rows=("units_sold", lambda values: int((values == 0).sum())),
            products=("stock_code", "nunique"),
            units=("units_sold", "sum"),
            partial_rows=("is_partial_month", "sum"),
        )
        .reset_index()
    )
    frame["zero_share"] = frame["zero_rows"] / frame["rows"]
    frame["partial_rows"] = frame["partial_rows"].astype(int)
    frame["is_partial"] = frame["month"].isin(cfg.raw.partial_months)
    return frame[
        ["month", "rows", "zero_rows", "zero_share", "products", "units", "partial_rows",
         "is_partial"]
    ]


def run(
    clean_df: pd.DataFrame, panel_df: pd.DataFrame, cfg: CleaningConfig, ctx: RunContext
) -> dict[str, Any]:
    """Compute E14 from the file on disk and write its table. No figure — this is a check."""
    from_disk, source = load_panel_file(ctx)
    ctx.logger.info(f"E14 read the hand-off panel from {source}")

    if len(from_disk) != len(panel_df):
        # Not fatal here: the contract validator (US-08) is what stops the Flow. E14's job is to
        # make the discrepancy visible in the report and the run log.
        ctx.warn(
            f"clean_data.csv on disk has {len(from_disk)} rows but the in-memory panel has "
            f"{len(panel_df)} — the hand-off file is stale or was written by another run"
        )

    tables = {"E14_panel_preview": _preview(from_disk, cfg)}
    for name, frame in tables.items():
        save_table(frame, name, ctx)

    return {"tables": tables, "figures": {}}
