"""Readers and writers for EDA figures and tables (US-06, PRD §35A.2, §39).

Every EDA figure and every EDA table in the project passes through this module, which makes it the
single place where three rules are enforced:

* **Stable names.** ``E<nn>_<topic>`` — ``E02_monthly_units.png``, ``E06_pareto.csv``. The report
  builder (US-11) and the Streamlit app look figures up by that name, so a rename breaks them.
* **Staging.** Writes go through ``ctx.out(...)``, so a failed run cannot overwrite the previous
  run's good figures (§39). Costs nothing outside staging: with ``staging=False`` ``ctx.out()``
  returns the path unchanged and creates the parent directory.
* **Read-your-own-writes.** Under staging the final ``artifacts/reports/`` still holds the
  *previous* run's files until ``promote()`` is called, so a mid-run reader that looked there
  would silently embed stale numbers and stale images. :func:`load_table` and
  :func:`figure_to_base64` therefore resolve the staged location first and fall back to the final
  one. They do **not** call ``ctx.out()`` to do it: ``out()`` registers a path for promotion, and
  ``promote()` would then warn "staged artifact was never written" for every file this run only
  read.

Tables are the only permitted source of numbers for ``insights.md`` (§35A.2) — the
``numbers_in_tables`` guard checks each narrative number against what was written here.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths

# Importing the style module first is what selects the headless ``Agg`` backend.
from pipeline.eda.style import FIGURE_DPI, plt
from pipeline.run_context import RunContext

#: Artifact naming convention (normative, §3 of the issue): ``E`` + two digits + a topic.
NAME_PATTERN = re.compile(r"^E\d{2}_[A-Za-z0-9_]+$")

#: Serialisation formats accepted by :func:`save_table` / :func:`load_table`.
TABLE_FORMATS: tuple[str, ...] = ("csv", "json")

#: Fixed float formatting, so two runs on the same data produce byte-identical tables (§40).
FLOAT_FORMAT = "%.4f"
JSON_PRECISION = 4


def _check_name(name: str) -> str:
    """Reject anything that is not an ``E<nn>_<topic>`` artifact name."""
    if not NAME_PATTERN.match(name):
        raise ValueError(
            f"invalid artifact name {name!r}: expected the E<nn>_<topic> convention, "
            "e.g. 'E02_monthly_units' (PRD §35A.2)"
        )
    return name


def _check_format(fmt: str) -> str:
    if fmt not in TABLE_FORMATS:
        raise ValueError(f"unsupported table format {fmt!r}: expected one of {TABLE_FORMATS}")
    return fmt


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical ``paths.*`` location.

    ``ctx.out()`` rebases a *relative* path onto the run's base directory but returns an
    *absolute* one unchanged — so the absolute form would escape a test ``base_dir`` and write
    into the real repository. The relative form is correct in production, under a test
    ``base_dir`` and under staging alike (``docs/interfaces.md`` §6 rule 12).
    """
    return path.relative_to(paths.PROJECT_ROOT)


def _resolve_read(ctx: RunContext, relative: Path) -> Path:
    """Locate an artifact for *reading*: this run's staged copy first, the final one second.

    Deliberately does not call ``ctx.out()`` — that would register the path for promotion and make
    ``promote()`` warn about a file this run never wrote.
    """
    candidates = [ctx.staging_dir / relative, ctx.base_dir / relative]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"artifact not found in staging or final location: {searched}")


def figure_path(name: str) -> Path:
    """Repo-relative location of a figure, e.g. ``artifacts/reports/figures/E06_pareto.png``."""
    return _repo_relative(paths.FIGURES_DIR / f"{_check_name(name)}.png")


def table_path(name: str, fmt: str = "csv") -> Path:
    """Repo-relative location of table ``name`` in ``fmt``."""
    return _repo_relative(paths.EDA_TABLES_DIR / f"{_check_name(name)}.{_check_format(fmt)}")


def save_figure(fig: Any, name: str, ctx: RunContext) -> Path:
    """Write ``fig`` to ``artifacts/reports/figures/<name>.png`` and close it; return the path.

    ``dpi`` is passed explicitly rather than taken from ``rcParams`` so the ≥ 150 dpi requirement
    holds even when the caller never invoked :func:`pipeline.eda.style.apply_style`. Closing the
    figure matters: a long EDA run draws seventeen of them, and Matplotlib keeps every unclosed
    figure alive.
    """
    relative = figure_path(name)
    destination = ctx.out(relative)
    fig.savefig(destination, dpi=FIGURE_DPI, format="png", bbox_inches="tight")
    plt.close(fig)
    ctx.logger.info(f"figure written: {relative.as_posix()}")
    return destination


def save_table(df: pd.DataFrame, name: str, ctx: RunContext, fmt: str = "csv") -> Path:
    """Write ``df`` to ``artifacts/reports/eda_tables/<name>.<fmt>``; return the path.

    Deterministic on purpose (§40): the caller's column and row order is preserved verbatim, the
    index is never written, floats use ``%.4f`` and the line terminator is ``\\n`` on every
    platform — so two runs on the same data produce byte-identical files.
    """
    relative = table_path(name, fmt)
    destination = ctx.out(relative)
    if fmt == "csv":
        df.to_csv(
            destination,
            index=False,
            float_format=FLOAT_FORMAT,
            lineterminator="\n",
            encoding="utf-8",
        )
    else:
        payload = df.to_json(
            orient="records",
            indent=2,
            double_precision=JSON_PRECISION,
            date_format="iso",
            force_ascii=False,
        )
        destination.write_text(f"{payload}\n", encoding="utf-8")
    ctx.logger.info(f"table written: {relative.as_posix()} ({len(df)} rows)")
    return destination


def load_table(name: str, ctx: RunContext, fmt: str = "csv") -> pd.DataFrame:
    """Read table ``name`` back — this run's staged copy if there is one, else the final one.

    Used by the report builder (US-11), the ``numbers_in_tables`` guard and the app, all of which
    must see the numbers *this* run computed rather than the previous run's leftovers.
    """
    source = _resolve_read(ctx, table_path(name, fmt))
    if fmt == "csv":
        return pd.read_csv(source)
    return pd.read_json(source, orient="records")


def figure_to_base64(figure: str | Path, ctx: RunContext | None = None) -> str:
    """Return a figure PNG as a base64 string, for embedding in the self-contained HTML report.

    Accepts either an ``E<nn>_<topic>`` name — resolved staged-first, which needs ``ctx`` — or a
    path, read as given. The returned value carries no ``data:`` prefix; US-11 embeds it as
    ``<img src="data:image/png;base64,{value}">``, which is what keeps ``eda_report.html`` free of
    any external reference (§35A.2).
    """
    if isinstance(figure, Path):
        source = figure
    elif ctx is None:
        raise ValueError("figure_to_base64() needs a ctx to resolve a figure name")
    else:
        source = _resolve_read(ctx, figure_path(figure))
    return base64.b64encode(source.read_bytes()).decode("ascii")
