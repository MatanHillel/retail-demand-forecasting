"""``eda_report.html`` — one self-contained page for the whole EDA (US-11, PRD §35A.2, §41, §49).

**Self-contained** means a single file with every picture packed inside it, so it opens from a
USB stick on a laptop with no network. That is a hard requirement (§35A.2), and it is enforced
rather than intended: after rendering, :func:`build_eda_report` asserts that the HTML contains no
``http://``, no ``https://`` and no ``src="file`` reference, and that the eight figures §35A.1
marks mandatory are actually embedded. A report that quietly lost an image fails the build
instead of shipping.

Every image is a **base64** data URI — the PNG written as text inside the ``<img>`` tag. That is
what makes the file portable, and it is why the report is a few megabytes rather than a few
kilobytes.

The page is assembled from ``index.json`` (the contents page all fourteen analyses register into)
plus the tables and figures themselves, read through :mod:`pipeline.eda.io` so that under the
Flow they resolve to *this* run's staged copies rather than the previous run's leftovers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from pipeline import paths
from pipeline.config import CleaningConfig
from pipeline.eda.io import figure_to_base64, load_table, table_path
from pipeline.run_context import RunContext

#: Where the Jinja2 template lives. Shipped as package data (see ``pyproject.toml``).
TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "eda_report.html.j2"

#: Figures §35A.1 requires the report to show. Missing any one of them is a build failure (§49).
MANDATORY_FIGURES: tuple[str, ...] = (
    "E01_waterfall",
    "E02_monthly_units",
    "E05_lifecycle_hist",
    "E06_pareto",
    "E07_top20_units",
    "E08_zero_share_by_k",
    "E09_units_hist_log",
    "E11_top_countries",
)

#: Substrings that would mean the page reaches outside itself for an asset.
EXTERNAL_MARKERS: tuple[str, ...] = ("http://", "https://", 'src="file')

#: Long tables are shown truncated; the full file is always on disk and named in the note.
MAX_TABLE_ROWS = 20

DATASET_NAME = "Online Retail II (UCI), Chen 2019, CC BY 4.0"


def _table_formats(name: str, ctx: RunContext) -> str:
    """Which serialisation this table was written in, so the note can name the real file."""
    for fmt in ("csv", "json"):
        relative = table_path(name, fmt)
        for candidate in (ctx.staging_dir / relative, ctx.base_dir / relative):
            if candidate.is_file():
                return fmt
    return "csv"


def _render_table(name: str, ctx: RunContext) -> dict[str, Any]:
    """One table as HTML, truncated for display but never for the file on disk."""
    fmt = _table_formats(name, ctx)
    frame = load_table(name, ctx, fmt=fmt)
    shown = frame.head(MAX_TABLE_ROWS)
    return {
        "name": name,
        "rows": int(len(frame)),
        "shown": int(len(shown)),
        "truncated": len(frame) > len(shown),
        "path": table_path(name, fmt).as_posix(),
        "html": shown.to_html(index=False, border=0, na_rep="", justify="right"),
    }


def _sections(index: list[dict[str, Any]], ctx: RunContext) -> list[dict[str, Any]]:
    """One section per analysis, in the index's order, with figures embedded and tables rendered."""
    sections = []
    for entry in index:
        figures = [
            {"name": name, "base64": figure_to_base64(name, ctx)}
            for name in entry.get("figure_names", [])
        ]
        tables = [_render_table(name, ctx) for name in entry.get("table_names", [])]
        sections.append(
            {
                "id": entry["id"],
                "title": entry.get("title", entry["id"]),
                "figures": figures,
                "tables": tables,
            }
        )
    return sections


def _appendix(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"analysis": section["id"], "name": table["name"], "rows": table["rows"]}
        for section in sections
        for table in section["tables"]
    ]


def _verify(html: str) -> None:
    """Fail the build rather than publish a report that is not self-contained (§35A.2, §49)."""
    found = [marker for marker in EXTERNAL_MARKERS if marker in html]
    if found:
        raise ValueError(
            f"eda_report.html must be self-contained but references {found} — every image has "
            "to be embedded as a data: URI and no stylesheet may be linked (PRD §35A.2)"
        )
    missing = [name for name in MANDATORY_FIGURES if name not in html]
    if missing:
        raise ValueError(
            f"eda_report.html is missing the mandatory figure(s) {missing} required by "
            "PRD §35A.1 / §49"
        )
    if "data:image/png;base64," not in html:
        raise ValueError("eda_report.html embeds no images at all")


def build_eda_report(
    index: list[dict[str, Any]],
    cfg: CleaningConfig,
    ctx: RunContext,
    *,
    exclusion_df: pd.DataFrame | None = None,
    contract_path: Path | None = None,
    out: Path | None = None,
) -> Path:
    """Render the one-file EDA report and return the path written.

    Everything after ``ctx`` is keyword-only: the issue's original signature put a defaulted
    parameter before a required one, which Python rejects at import time.

    The tables and figures directories are **not** parameters. :mod:`pipeline.eda.io` already
    resolves both, staged copy first, which is the behaviour the Flow needs — a caller handing in
    a raw directory would read the previous run's artifacts under ``staging=True``.

    Args:
        index: the entries of ``eda_tables/index.json``, in report order.
        cfg: cleaning configuration, for the period shown in the header.
        ctx: the run — supplies the run id, the data hash and the staging redirect.
        exclusion_df: the reviewed non-inventory list, rendered with its reasons under E1 (§12).
        contract_path: optional dataset contract to name in the footer.
        out: override the destination; defaults to ``paths.EDA_REPORT``.
    """
    sections = _sections(index, ctx)
    environment = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    partial = ", ".join(cfg.raw.partial_months)
    html = environment.get_template(TEMPLATE_NAME).render(
        title="Exploratory Data Analysis — Retail Demand Forecasting",
        subtitle=(
            "Fourteen mandatory analyses (E1–E14) behind the forecasting and inventory decisions."
        ),
        dataset=DATASET_NAME,
        period=(
            f"{cfg.raw.first_month} to {cfg.raw.last_full_month} complete, "
            f"plus the partial month {partial} (shown, never scored)"
        ),
        run_id=ctx.run_id,
        data_sha256=ctx.data.sha256,
        generated_at=_now(),
        source=(
            "Computed by pipeline.eda; every figure is embedded, so this file renders offline."
        ),
        sections=sections,
        exclusion_rows=(
            [] if exclusion_df is None else exclusion_df.to_dict(orient="records")
        ),
        appendix=_appendix(sections),
        footer=(
            "Generated deterministically by pipeline.eda.run_eda. "
            + (
                f"Dataset contract: {Path(contract_path).name}. "
                if contract_path is not None
                else ""
            )
            + "In LLM mode the narrative may be rewritten by the Business & EDA Analyst agent, "
            "but every number still has to come from these tables (PRD §38)."
        ),
    )
    _verify(html)

    relative = (paths.EDA_REPORT if out is None else Path(out)).relative_to(paths.PROJECT_ROOT)
    destination = ctx.out(relative)
    destination.write_text(html, encoding="utf-8", newline="\n")
    ctx.record_artifact("eda_report", relative)
    ctx.logger.info(
        f"eda_report.html written: {relative.as_posix()} "
        f"({len(sections)} sections, {sum(len(s['figures']) for s in sections)} figures, "
        f"{len(html) / 1_000_000:.1f} MB)"
    )
    return destination


def _now() -> str:
    """ISO-8601 UTC — the project's only timestamp format."""
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
