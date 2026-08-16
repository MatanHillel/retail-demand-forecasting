"""The EDA table index — one contents page for all fourteen analyses (US-09/US-10, PRD §35A).

``artifacts/reports/eda_tables/index.json`` is what US-11 walks to assemble ``eda_report.html``.
Three different issues write into it — E1 from :mod:`pipeline.quality` (US-07), E2–E7 and E8–E14
from :mod:`pipeline.eda.run_analyses` — and each of them is runnable on its own, so the index is
**merged by analysis id, never overwritten**. Running E6 alone must not delete the other thirteen
entries.

It lives here rather than in ``run_analyses`` because E1 needs the raw extract and therefore has
its own entry point; a shared module is what lets both writers use the same merge without either
importing the other.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.run_context import RunContext

#: File name of the index inside ``paths.EDA_TABLES_DIR``. ``pipeline.paths`` has no constant for
#: the file itself, so it is built from the directory constant rather than typed as a path.
INDEX_FILENAME = "index.json"


def index_path() -> Path:
    """Repo-relative location of the index file."""
    return (paths.EDA_TABLES_DIR / INDEX_FILENAME).relative_to(paths.PROJECT_ROOT)


def _sort_key(entry: dict[str, Any]) -> int:
    """Analyses sort by their number, so ``E10`` follows ``E9`` instead of ``E1``."""
    return int(str(entry["id"]).lstrip("E"))


def read_index(ctx: RunContext) -> dict[str, dict[str, Any]]:
    """Load the current index, this run's staged copy first, keyed by analysis id.

    Resolved the way :func:`pipeline.eda.io.load_table` resolves a table, and deliberately *not*
    through ``ctx.out()``: ``out()`` registers a path for promotion, so using it to locate a file
    would make ``promote()`` warn about an artifact this run never wrote. Reading the staged copy
    first is what stops an E8–E14 append from merging into the *previous* run's index.
    """
    relative = index_path()
    for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
        if candidate.is_file():
            entries = json.loads(candidate.read_text(encoding="utf-8"))
            return {entry["id"]: entry for entry in entries}
    return {}


def build_entry(
    analysis_id: str,
    title: str,
    tables: dict[str, pd.DataFrame] | list[str],
    figures: dict[str, Any] | list[str],
) -> dict[str, Any]:
    """One index record.

    ``one_line_summary_placeholder`` is exactly that — a slot US-11 fills from the computed
    tables. No number is written here: a narrative number may only come from a table (§35A.2).
    """
    return {
        "id": analysis_id,
        "title": title,
        "table_names": sorted(tables),
        "figure_names": sorted(figures),
        "one_line_summary_placeholder": "",
    }


def write_index(entries: dict[str, dict[str, Any]], ctx: RunContext) -> Path:
    """Write the index, sorted by analysis number so the file is stable across runs (§40)."""
    ordered = sorted(entries.values(), key=_sort_key)
    relative = index_path()
    destination = ctx.out(relative)
    destination.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ctx.record_artifact("eda_tables_index", relative)
    ctx.logger.info(f"index written: {relative.as_posix()} ({len(ordered)} analyses)")
    return destination


def merge_entries(new_entries: list[dict[str, Any]], ctx: RunContext) -> Path:
    """Merge records into the existing index and write it back."""
    entries = read_index(ctx)
    for entry in new_entries:
        entries[entry["id"]] = entry
    return write_index(entries, ctx)
