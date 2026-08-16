"""Runner for the EDA analyses and the table index (US-09, PRD §35A).

One entry point for every analysis:

```
python -m pipeline.eda.run_analyses --ids E2 E3 E4 E5 E6 E7
python -m pipeline.eda.run_analyses --all
```

Each analysis module exposes ``run(clean_df, panel_df, cfg, ctx) -> {"tables": …, "figures": …}``
and is listed in :data:`ANALYSES`. Adding E8-E14 (US-10) means adding rows to that registry and
nothing else.

**The index is merged, never overwritten.** ``artifacts/reports/eda_tables/index.json`` is what
US-11 walks to assemble ``eda_report.html``, and analyses are runnable one at a time, so a run of
E2 alone must not delete the entries E3-E7 wrote earlier. The existing index is read from *this
run's* view of the artifacts tree (staged copy first, final second) and merged by analysis id.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.config import CleaningConfig, load_cleaning_config
from pipeline.contract import read_panel
from pipeline.eda import (
    e02_demand_over_time,
    e03_calendar_patterns,
    e04_portfolio,
    e05_lifecycle,
    e06_abc,
    e07_top_products,
    e08_intermittency,
    e09_outliers,
    e10_order_structure,
    e11_customers_countries,
    e12_returns,
    e13_price,
    e14_panel_preview,
)
from pipeline.eda.index import INDEX_FILENAME, build_entry, index_path, merge_entries
from pipeline.eda.io import TABLE_FORMATS, table_path
from pipeline.run_context import RunContext

__all__ = [
    "ANALYSES",
    "ANALYSES_BY_ID",
    "Analysis",
    "INDEX_FILENAME",
    "index_path",
    "load_inputs",
    "main",
    "run_analyses",
    "written_table_path",
]

AnalysisRun = Callable[[pd.DataFrame, pd.DataFrame, CleaningConfig, RunContext], dict[str, Any]]


@dataclass(frozen=True)
class Analysis:
    """One registered EDA analysis."""

    id: str
    title: str
    run: AnalysisRun


#: Every analysis this runner knows about, in report order. E1 is produced by
#: :mod:`pipeline.quality` (US-07), which needs the *raw* extract as well and therefore has its
#: own entry point; it registers its own index entry through :mod:`pipeline.eda.index`.
ANALYSES: tuple[Analysis, ...] = (
    Analysis("E2", "Demand over time, seasonality and year-on-year growth",
             e02_demand_over_time.run),
    Analysis("E3", "Weekday and hour-of-day trading patterns", e03_calendar_patterns.run),
    Analysis("E4", "Product portfolio churn", e04_portfolio.run),
    Analysis("E5", "Product lifecycle", e05_lifecycle.run),
    Analysis("E6", "ABC concentration and the Pareto curve", e06_abc.run),
    Analysis("E7", "Top twenty products by units and by revenue", e07_top_products.run),
    Analysis("E8", "Intermittency and the zero-target share by k", e08_intermittency.run),
    Analysis("E9", "Demand magnitude and outliers", e09_outliers.run),
    Analysis("E10", "Order structure: invoice value, lines and wholesale sizing",
             e10_order_structure.run),
    Analysis("E11", "Customers and countries (descriptive only)", e11_customers_countries.run),
    Analysis("E12", "Returns and cancellations behind the gross-demand definition",
             e12_returns.run),
    Analysis("E13", "Unit price level and stability", e13_price.run),
    Analysis("E14", "Panel preview read back from clean_data.csv", e14_panel_preview.run),
)

ANALYSES_BY_ID: dict[str, Analysis] = {analysis.id: analysis for analysis in ANALYSES}


def written_table_path(name: str, ctx: RunContext) -> Path:
    """Repo-relative path of a table this run wrote, whichever format it used.

    ``save_table`` picks the extension, so the runner resolves it by looking for the file the
    same way a reader would: this run's staged copy first, the final location second.
    """
    for fmt in TABLE_FORMATS:
        relative = table_path(name, fmt)
        for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
            if candidate.is_file():
                return relative
    return table_path(name)


def run_analyses(
    ids: list[str],
    clean_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    cfg: CleaningConfig,
    ctx: RunContext,
) -> dict[str, dict[str, Any]]:
    """Run the named analyses in registry order and merge their entries into the index."""
    unknown = [identifier for identifier in ids if identifier not in ANALYSES_BY_ID]
    if unknown:
        raise ValueError(
            f"unknown analysis id(s): {', '.join(unknown)}. "
            f"Known: {', '.join(ANALYSES_BY_ID)}"
        )

    selected = [analysis for analysis in ANALYSES if analysis.id in set(ids)]
    results: dict[str, dict[str, Any]] = {}
    entries = []
    for analysis in selected:
        ctx.logger.info(f"{analysis.id}: {analysis.title}")
        result = analysis.run(clean_df, panel_df, cfg, ctx)
        results[analysis.id] = result
        entries.append(
            build_entry(analysis.id, analysis.title, result["tables"], result["figures"])
        )
    merge_entries(entries, ctx)
    return results


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, CleaningConfig]:
    """Read the US-04 and US-05 outputs the analyses run on."""
    for required in (paths.CLEAN_TRANSACTIONS, paths.CLEAN_DATA):
        if not required.is_file():
            raise FileNotFoundError(
                f"{required} not found. Run `python -m pipeline.cleaning` and "
                "`python -m pipeline.panel` first."
            )
    # ``read_panel`` rather than ``pd.read_csv``: plain reading infers ``stock_code`` as an
    # integer and silently drops the leading zero from a code like ``01234``.
    return (
        pd.read_parquet(paths.CLEAN_TRANSACTIONS),
        read_panel(paths.CLEAN_DATA),
        load_cleaning_config(),
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eda.run_analyses",
        description="Run the descriptive EDA analyses and refresh the table index.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", nargs="+", metavar="ID", help="analysis ids, e.g. E2 E3 E6")
    group.add_argument("--all", action="store_true", help="run every registered analysis")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """``python -m pipeline.eda.run_analyses --ids E2 E3 E4 E5 E6 E7``."""
    args = _parse_args(argv)
    ids = [analysis.id for analysis in ANALYSES] if args.all else [
        identifier.upper() for identifier in args.ids
    ]

    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("eda_e02_e07"):
            clean_df, panel_df, cfg = load_inputs()
            results = run_analyses(ids, clean_df, panel_df, cfg, ctx)
        for identifier, result in results.items():
            print(f"{identifier}:")
            for name in sorted(result["tables"]):
                print(f"  table  {written_table_path(name, ctx).as_posix()}")
            for _, figure in sorted(result["figures"].items()):
                print(f"  figure {Path(figure).as_posix()}")
        print(f"index    {index_path().as_posix()}")
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
