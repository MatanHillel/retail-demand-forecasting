"""``run_eda()`` — the Crew-1 tool that produces the whole EDA (US-11, PRD §35, §41).

One call runs all fourteen analyses and assembles the two artifacts the course requires by name:
``eda_report.html`` and ``insights.md`` (§41). It is deliberately LLM-free — CI runs
``--no-llm`` and both files must exist there (§37). In LLM mode the Business & EDA Analyst agent
is kicked off *around* this tool to rewrite prose; it never computes a number, and the
``numbers_in_tables`` guard validates whatever it writes (§38).

Order matters. E1 needs the raw extract, E2–E14 need the cleaned transactions and the panel, and
the report needs all fourteen to have registered themselves in ``index.json`` first. If any
analysis is missing when the report is assembled the run stops gracefully, naming the E-id, via
the project's standard ``ValidationResult`` → ``FlowValidationError`` path rather than rendering
a report with a hole in it.
"""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.config import CleaningConfig, load_cleaning_config, load_non_inventory_codes
from pipeline.contract import read_panel
from pipeline.download import load_raw
from pipeline.eda import e01_data_quality
from pipeline.eda.index import read_index
from pipeline.eda.insights import generate_insights
from pipeline.eda.io import load_table, table_path
from pipeline.eda.report import build_eda_report
from pipeline.eda.run_analyses import ANALYSES, run_analyses
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

RUN_EDA_STEP = "run_eda"

#: Every analysis the report expects: E1 from :mod:`pipeline.quality`, E2-E14 from the registry.
EXPECTED_IDS: tuple[str, ...] = ("E1", *(analysis.id for analysis in ANALYSES))


def _load_frame(name: str, ctx: RunContext) -> pd.DataFrame | None:
    """Read a table by name in whichever format it was written, or ``None`` if it is missing."""
    for fmt in ("csv", "json"):
        relative = table_path(name, fmt)
        for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
            if candidate.is_file():
                return load_table(name, ctx, fmt=fmt)
    return None


def validate_index(index: list[dict[str, Any]], ctx: RunContext) -> ValidationResult:
    """Check that every expected analysis ran and every table it registered is readable.

    Pure: it computes the result and touches nothing. The caller writes the report and raises —
    the same division of labour as every other validation in the project.
    """
    present = {entry["id"]: entry for entry in index}
    violations = [
        Violation(
            step=RUN_EDA_STEP,
            rule="missing_analysis",
            message=(
                f"{analysis_id} is missing from eda_tables/index.json — the report cannot be "
                "assembled without it"
            ),
        )
        for analysis_id in EXPECTED_IDS
        if analysis_id not in present
    ]

    missing_tables: list[str] = []
    for entry in index:
        for name in entry.get("table_names", []):
            if _load_frame(name, ctx) is None:
                missing_tables.append(f"{entry['id']}:{name}")
    if missing_tables:
        violations.append(
            Violation(
                step=RUN_EDA_STEP,
                rule="missing_table",
                message=(
                    f"{len(missing_tables)} table(s) named in the index are not on disk: "
                    f"{', '.join(missing_tables[:5])}"
                ),
                count=len(missing_tables),
                examples=missing_tables[:5],
            )
        )

    return ValidationResult(
        step=RUN_EDA_STEP,
        passed=not violations,
        violations=violations,
        extra={"expected": list(EXPECTED_IDS), "present": sorted(present)},
    )


def collect_tables(index: list[dict[str, Any]], ctx: RunContext) -> dict[str, pd.DataFrame]:
    """Every table the analyses wrote, keyed by name — the only permitted source of numbers."""
    tables: dict[str, pd.DataFrame] = {}
    for entry in index:
        for name in entry.get("table_names", []):
            frame = _load_frame(name, ctx)
            if frame is not None:
                tables[name] = frame
    return tables


def run_eda(
    clean_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    waterfall_df: pd.DataFrame,
    cfg: CleaningConfig,
    ctx: RunContext,
    *,
    raw_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run E1-E14 and assemble ``eda_report.html`` and ``insights.md``.

    ``raw_df`` is keyword-only and optional: E1 profiles the *raw* extract, which the five
    documented positional arguments do not carry, so it is loaded here when the caller has not
    already got it in memory. Loading through ``load_raw(ctx=ctx)`` also records the dataset hash
    on the run, which is what the report header shows.

    Returns ``{"index", "tables", "insights", "report_path", "insights_path"}``.

    Raises:
        FlowValidationError: when an analysis or one of its tables is missing. The message names
            the E-id and ``validation_report.json`` is written first, so a failed run still
            reports why (PRD §39).
    """
    if raw_df is None:
        raw_df, _ = load_raw(ctx=ctx)

    ctx.logger.info("E1: data quality and the cleaning waterfall")
    e01_data_quality.run(raw_df, clean_df, waterfall_df, panel_df, cfg, ctx)
    run_analyses([analysis.id for analysis in ANALYSES], clean_df, panel_df, cfg, ctx)

    index = sorted(read_index(ctx).values(), key=lambda entry: int(str(entry["id"]).lstrip("E")))
    result = validate_index(index, ctx)
    write_validation_report(result, run_id=ctx.run_id)
    if not result.passed:
        raise FlowValidationError(result)

    tables = collect_tables(index, ctx)
    figures = [name for entry in index for name in entry.get("figure_names", [])]

    report_path = build_eda_report(
        index, cfg, ctx, exclusion_df=load_non_inventory_codes(),
        contract_path=paths.DATASET_CONTRACT if paths.DATASET_CONTRACT.is_file() else None,
    )
    insights = generate_insights(tables, cfg, ctx, figures=figures)

    ctx.record_metrics(
        {
            "eda_analyses": len(index),
            "eda_tables": len(tables),
            "eda_figures": len(figures),
            "eda_insights": len(insights),
        }
    )
    return {
        "index": index,
        "tables": tables,
        "insights": insights,
        "report_path": report_path,
        "insights_path": ctx.out(paths.INSIGHTS.relative_to(paths.PROJECT_ROOT)),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, CleaningConfig]:
    """Read the US-04 and US-05 outputs the analyses run on."""
    for required in (paths.CLEAN_TRANSACTIONS, paths.CLEAN_DATA):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} not found. Run `python -m pipeline.cleaning` and "
                "`python -m pipeline.panel` first."
            )
    return (
        pd.read_parquet(paths.CLEAN_TRANSACTIONS),
        read_panel(paths.CLEAN_DATA),
        load_cleaning_config(),
    )


def main(argv: list[str] | None = None) -> int:
    """``python -m pipeline.eda.run_eda`` — the whole EDA, end to end."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step(RUN_EDA_STEP):
            clean_df, panel_df, cfg = load_inputs()
            waterfall_df = load_table(e01_data_quality.WATERFALL_TABLE, ctx)
            result = run_eda(clean_df, panel_df, waterfall_df, cfg, ctx)
        print(
            f"analyses: {len(result['index'])}\n"
            f"tables:   {len(result['tables'])}\n"
            f"insights: {len(result['insights'])}\n"
            f"report:   {paths.EDA_REPORT.relative_to(paths.PROJECT_ROOT).as_posix()}\n"
            f"insights: {paths.INSIGHTS.relative_to(paths.PROJECT_ROOT).as_posix()}"
        )
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
