"""One plotting style for every figure in the project (PRD §35A.2).

The course requires Matplotlib/Seaborn only, and the PRD requires that all fourteen EDA figures,
the Streamlit charts and the evaluation figures read as one system: the same colour-blind-safe
palette, the same three ABC colours, a title, axis labels *with units*, a source footnote and one
message per figure. Implementing that once here is what keeps them consistent.

**Global state.** :func:`apply_style` mutates process-global Matplotlib ``rcParams`` and installs
the Seaborn theme; it affects every figure created afterwards in the same process, including ones
drawn by code that never called it. Nothing in this module *depends* on it having been called —
in particular :func:`pipeline.eda.io.save_figure` passes ``dpi`` explicitly at save time, so the
≥ 150 dpi guarantee holds even when a caller forgot the call.

The backend is forced to ``Agg`` (headless, file-only) on import: the pipeline never opens a
window, and CI has no display. Streamlit renders these figures through ``st.pyplot``, which works
with ``Agg``.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend selection)
import seaborn as sns  # noqa: E402

#: Okabe–Ito — the reference colour-blind-safe qualitative palette (Okabe & Ito, 2008). Ordered
#: for series assignment: the first colours stay distinguishable under every common form of colour
#: vision deficiency and in greyscale print. Seaborn's default palette is *not* colour-blind safe,
#: so it is always overridden explicitly.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#000000",  # black
]

#: ABC classes always use the same three colours, everywhere, forever (§35A.2). Fixing them here
#: is the whole point: a reader who learns "blue = A" on the Pareto chart reads every later figure
#: without re-checking the legend. Never reorder or re-map these.
ABC_COLORS: dict[str, str] = {
    "A": "#0072B2",  # blue
    "B": "#E69F00",  # orange
    "C": "#009E73",  # bluish green
}

#: Every figure is saved at this resolution or better (§35A.2 requires ≥ 150 dpi PNGs).
FIGURE_DPI: int = 150

#: Default figure size in inches (width, height).
FIGURE_SIZE: tuple[float, float] = (9.0, 5.5)

#: Smallest font size used anywhere; §35A.2 requires legible labels (≥ 10 pt).
BASE_FONT_SIZE: int = 11

#: Provenance line printed under every figure. The date range is the cleaned dataset's coverage.
DEFAULT_FOOTNOTE: str = "Source: Online Retail II (UCI), cleaned, Dec 2009–Nov 2011"

#: Appended to an axis label whenever that axis is drawn on a log scale — §35A.2 requires the
#: scale to be stated in the label, not merely visible in the tick spacing.
LOG_SCALE_SUFFIX: str = " (log scale)"

#: Hatch pattern marking a month that is not a full month (Dec 2011 runs only to the 9th, §8).
PARTIAL_HATCH: str = "///"

#: Label drawn over a hatched partial region.
PARTIAL_LABEL: str = "partial"


def apply_style() -> None:
    """Install the project palette, Seaborn theme and Matplotlib defaults for this process.

    Call once before drawing. Sets font sizes (≥ ``BASE_FONT_SIZE``), a light grid, the default
    figure size and ``FIGURE_DPI``. Mutates global ``rcParams`` — see the module docstring.
    """
    sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.0)
    sns.set_palette(PALETTE)
    plt.rcParams.update(
        {
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE + 3,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        }
    )


def finalize(
    fig: Any,
    title: str,
    xlabel: str,
    ylabel: str,
    footnote: str = DEFAULT_FOOTNOTE,
    log_y: bool = False,
) -> Any:
    """Apply the mandatory furniture to a finished figure and return it.

    Sets the title, both axis labels — the caller supplies the **units** in them, e.g.
    ``"Units sold (units)"`` — the source footnote, and the layout. When ``log_y`` is true the
    y-axis is switched to a log scale *and* ``"(log scale)"`` is appended to ``ylabel``, because
    §35A.2 requires the scale to be stated in the label.

    Acts on the figure's first axes; a figure with several axes gets the title and footnote once
    and the labels on that first axes.
    """
    axes = fig.axes
    if not axes:
        raise ValueError("finalize() needs a figure with at least one axes")
    ax = axes[0]

    if log_y:
        ax.set_yscale("log")
        if not ylabel.endswith(LOG_SCALE_SUFFIX):
            ylabel = f"{ylabel}{LOG_SCALE_SUFFIX}"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # tight_layout first, then reserve the strip the footnote sits in, so the two never fight.
    fig.tight_layout()
    if footnote:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.18))
        fig.text(
            0.01,
            0.01,
            footnote,
            ha="left",
            va="bottom",
            fontsize=BASE_FONT_SIZE - 2,
            color="#444444",
        )
    return fig


def hatch_partial(ax: Any, x_positions: list[Any]) -> list[Any]:
    """Hatch and label the partial months at ``x_positions`` and return the drawn patches.

    A partial month (Dec 2011, which ends on the 9th — §8) must never be read as a collapse in
    demand, so it is drawn hatched and labelled ``"partial"`` rather than hidden. It is shown, but
    it is never scored.

    ``x_positions`` are x-axis coordinates of the bars/points to mark: categorical index positions
    (``0, 1, 2 …``) for a bar chart, or numeric/date coordinates for a line chart. Each is covered
    by a half-unit-wide band, which lines up with Matplotlib's default categorical bar spacing.
    """
    patches = []
    for index, position in enumerate(x_positions):
        span = ax.axvspan(
            position - 0.5,
            position + 0.5,
            facecolor="none",
            edgecolor="#666666",
            hatch=PARTIAL_HATCH,
            linewidth=0.0,
            zorder=0,
            label=PARTIAL_LABEL if index == 0 else None,
        )
        patches.append(span)
        if index == 0:
            ax.annotate(
                PARTIAL_LABEL,
                xy=(position, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -6),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=BASE_FONT_SIZE - 2,
                color="#444444",
            )
    return patches
