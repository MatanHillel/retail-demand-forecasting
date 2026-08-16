"""E1 — Data quality and the cleaning waterfall (US-07, PRD §35A E1).

The first of the fourteen mandatory analyses. It turns the profiling tools of
:mod:`pipeline.quality` into the artifacts the report, the app and the LLM reviewer read:

* ``E01_cleaning_waterfall.csv`` — written by :mod:`pipeline.cleaning`; verified here, not rebuilt
* ``E01_missing_values.csv`` — missing values per raw column
* ``E01_nonproduct_codes.csv`` — codes that do not look like products, with their review status
* ``E01_abnormal_lines.csv`` — every sales line above the abnormal-quantity threshold
* ``E01_duplicates_summary.json`` — exact-duplicate counts against the §11 thresholds
* ``E01_waterfall.png`` — rows removed by each cleaning step

Every write goes through :mod:`pipeline.eda.io`, so all six land under ``ctx.out()`` and a failed
run cannot overwrite the previous run's good figures (§39). This module imports
:mod:`pipeline.quality` and never the reverse — the dependency runs one way.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipeline.config import CleaningConfig
from pipeline.eda.index import build_entry, merge_entries
from pipeline.eda.io import load_table, save_figure, save_table
from pipeline.eda.style import PALETTE, apply_style, finalize, plt
from pipeline.quality import (
    abnormal_quantities,
    build_data_quality_findings,
    detect_duplicates,
    list_nonproduct_codes,
    profile_raw,
)
from pipeline.run_context import RunContext

ANALYSIS_ID = "E1"
ANALYSIS_TITLE = "Data quality and the cleaning waterfall"

#: Name of the waterfall table US-04 writes; E1 verifies it rather than recomputing it.
WATERFALL_TABLE = "E01_cleaning_waterfall"

#: PRD §10 numbers the cleaning chain 1-10; steps 5-9 are the ones that remove rows, and the
#: figure shows exactly those. These are step *identifiers*, not tunable thresholds.
FIRST_REMOVAL_STEP = 5
LAST_REMOVAL_STEP = 9


def _missing_values_table(profile: dict[str, Any]) -> pd.DataFrame:
    """Missing values per raw column, most-missing first."""
    rows = [
        {"column": column, "missing": stats["missing"], "share": stats["share"]}
        for column, stats in profile["missing_values"]["per_column"].items()
    ]
    frame = pd.DataFrame(rows, columns=["column", "missing", "share"])
    return frame.sort_values(
        ["missing", "column"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _duplicates_summary_table(duplicates: dict[str, Any]) -> pd.DataFrame:
    """The duplicates summary as a one-row frame.

    ``save_table(..., fmt="json")`` serialises a DataFrame with ``orient="records"``, so the file
    is a JSON **array holding one object** rather than a bare object. That is the price of going
    through the one artifact choke point, and it keeps the table readable by ``load_table`` —
    which US-11 uses to check every narrative number against a computed table.
    """
    return pd.DataFrame(
        [
            {
                "duplicate_rows": duplicates["duplicate_rows"],
                "duplicate_row_share": duplicates["duplicate_row_share"],
                "duplicate_units": duplicates["duplicate_units"],
                "duplicate_units_share": duplicates["duplicate_units_share"],
                "row_share_threshold": duplicates["row_share_threshold"],
                "units_share_threshold": duplicates["units_share_threshold"],
                "warning": duplicates["warning"],
            }
        ]
    )


def _waterfall_figure(waterfall_df: pd.DataFrame, ctx: RunContext) -> str:
    """Horizontal bar chart of the rows each cleaning step removed (§35A E1)."""
    apply_style()
    steps = waterfall_df[
        waterfall_df["step_no"].between(FIRST_REMOVAL_STEP, LAST_REMOVAL_STEP)
    ].copy()
    # Highest step number at the bottom, so the chart reads top-down in execution order.
    steps = steps.sort_values("step_no", ascending=False, kind="mergesort")
    labels = [f"{int(row.step_no)}. {row.step}" for row in steps.itertuples()]

    fig, ax = plt.subplots()
    ax.barh(labels, steps["rows_removed"], color=PALETTE[0])
    for label, value in zip(labels, steps["rows_removed"], strict=True):
        ax.annotate(
            f"{int(value):,}",
            xy=(value, label),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )
    ax.margins(x=0.12)
    finalize(
        fig,
        title="Rows removed by cleaning step",
        xlabel="Rows removed (rows)",
        ylabel="Cleaning step (PRD §10)",
    )
    save_figure(fig, "E01_waterfall", ctx)
    return "E01_waterfall"


def run(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    waterfall_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: CleaningConfig,
    ctx: RunContext,
    *,
    codes_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute E1, write its tables, figure and ``data_quality_findings.json``.

    Returns ``{"tables": {...}, "figures": {...}, "findings": {...}}`` — the same shape as the
    E2-E14 analyses, plus the findings dict the caller prints and the Flow records.

    ``codes_df`` may be passed in when the caller already built the non-product code table (the
    CLI does, because the exclusion-list review needs it first); otherwise it is computed here.
    """
    # The waterfall itself is US-04's output. Verify it first — before profiling a million rows —
    # and verify it for *this* run: under staging the final artifacts/ tree still holds the
    # previous run's copy, and ``load_table`` resolves the staged location first.
    try:
        waterfall_written = load_table(WATERFALL_TABLE, ctx)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{WATERFALL_TABLE}.csv is missing — it is written by pipeline.cleaning (US-04) "
            "and E1 only verifies it"
        ) from error

    profile = profile_raw(raw_df, cfg)
    duplicates = detect_duplicates(raw_df, cfg)
    codes_df = list_nonproduct_codes(raw_df, cfg) if codes_df is None else codes_df
    abnormal_df = abnormal_quantities(clean_df, cfg=cfg)

    tables = {
        WATERFALL_TABLE: waterfall_written,
        "E01_missing_values": _missing_values_table(profile),
        "E01_nonproduct_codes": codes_df,
        "E01_abnormal_lines": abnormal_df,
        "E01_duplicates_summary": _duplicates_summary_table(duplicates),
    }
    for name in ("E01_missing_values", "E01_nonproduct_codes", "E01_abnormal_lines"):
        save_table(tables[name], name, ctx)
    save_table(tables["E01_duplicates_summary"], "E01_duplicates_summary", ctx, fmt="json")

    figures = {"E01_waterfall": _waterfall_figure(waterfall_df, ctx)}

    # E1 is part of the report's contents page too. The index merges by analysis id, so this
    # entry survives a later E2-E14 run and US-11 sees all fourteen (US-10 acceptance criteria).
    merge_entries([build_entry(ANALYSIS_ID, ANALYSIS_TITLE, tables, figures)], ctx)

    findings = build_data_quality_findings(
        raw_df,
        clean_df,
        waterfall_df,
        panel_df,
        cfg,
        ctx,
        profile=profile,
        duplicates=duplicates,
        codes_df=codes_df,
        abnormal_df=abnormal_df,
    )
    return {"tables": tables, "figures": figures, "findings": findings}
