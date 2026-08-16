"""EDA foundations tests (US-06, PRD §35A.2, §18.2, §23, §27, §39, §40).

Covers the three modules every later EDA issue builds on: the shared plotting style, the
figure/table savers and the ABC classification. Nothing here asserts a business number from the
PRD — only structure, the staging guarantee and the leakage rule. Every write goes to a
``tmp_path`` base directory, so no test can touch the real ``artifacts/`` tree.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from pipeline.abc import ABC_COLUMNS, compute_abc
from pipeline.config import load_inventory_policy
from pipeline.eda.io import (
    figure_path,
    figure_to_base64,
    load_table,
    save_figure,
    save_table,
    table_path,
)
from pipeline.eda.style import (
    ABC_COLORS,
    BASE_FONT_SIZE,
    DEFAULT_FOOTNOTE,
    FIGURE_DPI,
    LOG_SCALE_SUFFIX,
    PALETTE,
    PARTIAL_LABEL,
    apply_style,
    finalize,
    hatch_partial,
)
from pipeline.run_context import RunContext, close_log_handlers

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def ctx(tmp_path: Path):
    """A run whose artifacts land under ``tmp_path`` instead of the repository."""
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


@pytest.fixture
def staged_ctx(tmp_path: Path):
    """A staging run: writes go to ``artifacts/_staging/<run_id>/`` until ``promote()``."""
    context = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


@pytest.fixture
def clean_rc():
    """Restore global Matplotlib settings, which :func:`apply_style` mutates process-wide."""
    original = plt.rcParams.copy()
    yield
    plt.rcParams.update(original)


@pytest.fixture
def figure():
    """A trivial drawn figure; closed afterwards in case the test did not save it."""
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [3, 1, 4])
    yield fig
    plt.close(fig)


def _panel(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Minimal panel frame from ``(month, stock_code, gross_revenue)`` triples."""
    return pd.DataFrame(rows, columns=["month", "stock_code", "gross_revenue"])


#: Revenue chosen so both class boundaries fall exactly on a product:
#: cumulative share is 0.80 after P1 (last A) and 0.95 after P3 (last B).
CUT_OFF = "2011-05"
BASE_ROWS = [
    (CUT_OFF, "P1", 80.0),
    (CUT_OFF, "P2", 10.0),
    (CUT_OFF, "P3", 5.0),
    (CUT_OFF, "P4", 4.0),
    (CUT_OFF, "P5", 1.0),
    (CUT_OFF, "P6", 0.0),
]
EXPECTED_CLASSES = {"P1": "A", "P2": "B", "P3": "B", "P4": "C", "P5": "C", "P6": "C"}


# --------------------------------------------------------------------------
# style — palette and theme
# --------------------------------------------------------------------------
def test_palette_is_an_ordered_list_of_distinct_hex_colours():
    assert len(PALETTE) >= 5
    assert len(set(PALETTE)) == len(PALETTE)
    assert all(re.fullmatch(r"#[0-9A-Fa-f]{6}", colour) for colour in PALETTE)


def test_abc_colours_are_the_three_fixed_classes():
    assert set(ABC_COLORS) == {"A", "B", "C"}
    assert len(set(ABC_COLORS.values())) == 3
    assert all(colour in PALETTE for colour in ABC_COLORS.values())


def test_apply_style_sets_readable_fonts_and_the_required_dpi(clean_rc):
    plt.rcParams.update({"font.size": 6.0, "figure.dpi": 72.0})
    apply_style()
    assert plt.rcParams["font.size"] >= 10
    assert plt.rcParams["figure.dpi"] >= FIGURE_DPI
    assert plt.rcParams["savefig.dpi"] >= FIGURE_DPI
    assert plt.rcParams["axes.grid"] is True
    cycle_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle_colours[: len(PALETTE)] == PALETTE


# --------------------------------------------------------------------------
# style — finalize / hatch_partial
# --------------------------------------------------------------------------
def test_finalize_adds_title_axis_labels_and_the_source_footnote(figure):
    finalize(figure, "Monthly units", "Month", "Units sold (units)")
    ax = figure.axes[0]
    assert ax.get_title() == "Monthly units"
    assert ax.get_xlabel() == "Month"
    assert ax.get_ylabel() == "Units sold (units)"
    footnotes = [text.get_text() for text in figure.texts]
    assert DEFAULT_FOOTNOTE in footnotes


def test_finalize_accepts_a_custom_footnote(figure):
    finalize(figure, "T", "x", "y", footnote="Source: training window only")
    assert "Source: training window only" in [text.get_text() for text in figure.texts]


def test_finalize_states_the_log_scale_in_the_axis_label(figure):
    finalize(figure, "Units", "Product", "Units sold (units)", log_y=True)
    ax = figure.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_ylabel().endswith(LOG_SCALE_SUFFIX)


def test_finalize_does_not_repeat_the_log_suffix(figure):
    finalize(figure, "Units", "Product", f"Units sold (units){LOG_SCALE_SUFFIX}", log_y=True)
    assert figure.axes[0].get_ylabel().count(LOG_SCALE_SUFFIX) == 1


def test_finalize_rejects_a_figure_without_axes():
    empty = plt.figure()
    try:
        with pytest.raises(ValueError):
            finalize(empty, "T", "x", "y")
    finally:
        plt.close(empty)


def test_hatch_partial_marks_every_position_and_labels_the_first(figure):
    ax = figure.axes[0]
    patches = hatch_partial(ax, [1, 2])
    assert len(patches) == 2
    assert all(patch.get_hatch() for patch in patches)
    assert PARTIAL_LABEL in [annotation.get_text() for annotation in ax.texts]


def test_base_font_size_is_legible():
    assert BASE_FONT_SIZE >= 10


# --------------------------------------------------------------------------
# io — figures
# --------------------------------------------------------------------------
def test_save_figure_writes_a_png_of_at_least_150_dpi(ctx, figure, tmp_path):
    written = save_figure(figure, "E02_monthly_units", ctx)

    assert written == tmp_path / "artifacts" / "reports" / "figures" / "E02_monthly_units.png"
    assert written.read_bytes()[:8] == PNG_MAGIC
    with Image.open(written) as image:
        horizontal, vertical = image.info["dpi"]
    # The PNG pHYs chunk stores whole pixels-per-metre, so 150 dpi round-trips as 149.99…
    assert round(horizontal) >= FIGURE_DPI
    assert round(vertical) >= FIGURE_DPI


def test_save_figure_closes_the_figure(ctx, figure):
    save_figure(figure, "E06_pareto", ctx)
    assert not plt.fignum_exists(figure.number)


def test_save_figure_guarantees_the_dpi_without_apply_style(ctx, figure, clean_rc):
    plt.rcParams.update({"savefig.dpi": 72.0, "figure.dpi": 72.0})
    written = save_figure(figure, "E06_pareto", ctx)
    with Image.open(written) as image:
        assert round(image.info["dpi"][0]) >= FIGURE_DPI


@pytest.mark.parametrize("name", ["monthly_units", "e02_monthly_units", "E2_units", "E02 units"])
def test_save_figure_rejects_names_outside_the_convention(ctx, figure, name):
    with pytest.raises(ValueError, match="E<nn>_<topic>"):
        save_figure(figure, name, ctx)


def test_figure_path_is_repo_relative():
    assert figure_path("E06_pareto") == Path("artifacts/reports/figures/E06_pareto.png")


# --------------------------------------------------------------------------
# io — tables
# --------------------------------------------------------------------------
def test_save_table_writes_csv_and_load_table_reads_it_back(ctx, tmp_path):
    table = pd.DataFrame({"stock_code": ["B", "A"], "share": [0.123456, 0.5]})

    written = save_table(table, "E06_abc_summary", ctx)

    assert written == tmp_path / "artifacts" / "reports" / "eda_tables" / "E06_abc_summary.csv"
    text = written.read_text(encoding="utf-8")
    assert text.startswith("stock_code,share\n")
    assert "0.1235" in text  # float_format="%.4f"
    assert "\r\n" not in text  # deterministic line terminator on every platform
    # Row order is the caller's, never re-sorted.
    assert list(load_table("E06_abc_summary", ctx)["stock_code"]) == ["B", "A"]


def test_save_table_writes_json_when_asked(ctx):
    table = pd.DataFrame({"stock_code": ["A"], "revenue": [12.5]})
    written = save_table(table, "E06_abc_summary", ctx, fmt="json")
    assert written.suffix == ".json"
    assert list(load_table("E06_abc_summary", ctx, fmt="json")["stock_code"]) == ["A"]


def test_save_table_is_byte_identical_across_runs(tmp_path):
    table = pd.DataFrame({"stock_code": ["A", "B"], "revenue": [1 / 3, 2 / 7]})
    written = []
    for index in range(2):
        context = RunContext.start(mode="no-llm", base_dir=tmp_path / f"run{index}")
        written.append(save_table(table, "E06_abc_summary", context).read_bytes())
        close_log_handlers(context.run_id)
    assert written[0] == written[1]


def test_save_table_rejects_an_unknown_format(ctx):
    with pytest.raises(ValueError, match="unsupported table format"):
        save_table(pd.DataFrame({"a": [1]}), "E06_abc_summary", ctx, fmt="xlsx")


def test_table_path_is_repo_relative():
    assert table_path("E06_pareto") == Path("artifacts/reports/eda_tables/E06_pareto.csv")


def test_load_table_reports_a_missing_artifact(ctx):
    with pytest.raises(FileNotFoundError):
        load_table("E99_absent", ctx)


# --------------------------------------------------------------------------
# io — staging (PRD §39)
# --------------------------------------------------------------------------
def test_writes_are_staged_and_only_reach_the_final_path_on_promote(staged_ctx, figure, tmp_path):
    final = tmp_path / "artifacts" / "reports" / "figures" / "E06_pareto.png"

    staged = save_figure(figure, "E06_pareto", staged_ctx)

    assert staged.is_relative_to(staged_ctx.staging_dir)
    assert not final.exists()  # a failed run would leave the previous figure untouched
    staged_ctx.promote()
    assert final.exists()


def test_load_table_prefers_this_runs_staged_copy_over_a_stale_final_one(staged_ctx, tmp_path):
    stale = tmp_path / "artifacts" / "reports" / "eda_tables" / "E06_abc_summary.csv"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("stock_code,revenue\nOLD,1\n", encoding="utf-8")

    save_table(pd.DataFrame({"stock_code": ["NEW"], "revenue": [2]}), "E06_abc_summary", staged_ctx)

    assert list(load_table("E06_abc_summary", staged_ctx)["stock_code"]) == ["NEW"]


def test_reading_an_artifact_does_not_register_it_for_promotion(staged_ctx, figure):
    save_figure(figure, "E06_pareto", staged_ctx)
    with pytest.raises(FileNotFoundError):
        load_table("E99_absent", staged_ctx)

    staged_ctx.promote()

    assert not any("never written" in warning for warning in staged_ctx.warnings)


# --------------------------------------------------------------------------
# io — base64 embedding (US-11)
# --------------------------------------------------------------------------
def test_figure_to_base64_round_trips_the_png_bytes(ctx, figure):
    written = save_figure(figure, "E06_pareto", ctx)

    by_name = figure_to_base64("E06_pareto", ctx)
    by_path = figure_to_base64(written)

    assert by_name == by_path
    assert base64.b64decode(by_name) == written.read_bytes()
    assert base64.b64decode(by_name)[:8] == PNG_MAGIC


def test_figure_to_base64_needs_a_context_to_resolve_a_name():
    with pytest.raises(ValueError, match="ctx"):
        figure_to_base64("E06_pareto")


# --------------------------------------------------------------------------
# abc — classification
# --------------------------------------------------------------------------
def test_compute_abc_classifies_the_hand_built_panel():
    result = compute_abc(_panel(BASE_ROWS), through_month=CUT_OFF)

    assert list(result.columns) == ABC_COLUMNS
    assert dict(zip(result["stock_code"], result["abc_class"], strict=True)) == EXPECTED_CLASSES


def test_compute_abc_ranks_by_revenue_and_accumulates_shares():
    result = compute_abc(_panel(BASE_ROWS), through_month=CUT_OFF)

    assert list(result["stock_code"]) == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert result["revenue_share"].iloc[0] == pytest.approx(0.80)
    assert result["cum_share"].iloc[2] == pytest.approx(0.95)
    assert result["cum_share"].iloc[-1] == pytest.approx(1.0)


def test_compute_abc_ignores_revenue_after_the_cut_off():
    """The leakage rule: revenue the forecast origin cannot see must not change a class."""
    before = compute_abc(_panel(BASE_ROWS), through_month=CUT_OFF)
    after_cut_off = BASE_ROWS + [
        ("2011-06", "P5", 10_000.0),  # would make P5 class A if the window leaked
        ("2011-11", "P6", 5_000.0),
    ]

    with_future = compute_abc(_panel(after_cut_off), through_month=CUT_OFF)

    pd.testing.assert_frame_equal(before, with_future)
    assert dict(zip(with_future["stock_code"], with_future["abc_class"], strict=True)) == (
        EXPECTED_CLASSES
    )


def test_compute_abc_sees_the_extra_revenue_when_the_window_is_widened():
    rows = BASE_ROWS + [("2011-06", "P5", 6.0)]
    widened = compute_abc(_panel(rows), through_month="2011-06").set_index("stock_code")
    assert widened.loc["P5", "revenue"] == pytest.approx(7.0)
    assert widened.loc["P5", "abc_class"] == "B"  # class C while the window stopped at the cut-off


def test_compute_abc_excludes_products_first_seen_after_the_cut_off():
    rows = BASE_ROWS + [("2011-06", "P7", 50.0)]
    result = compute_abc(_panel(rows), through_month=CUT_OFF)
    assert "P7" not in set(result["stock_code"])


def test_compute_abc_puts_zero_revenue_products_in_class_c():
    result = compute_abc(_panel([(CUT_OFF, "P1", 0.0), (CUT_OFF, "P2", 0.0)]), CUT_OFF)
    assert set(result["abc_class"]) == {"C"}


def test_compute_abc_breaks_ties_by_stock_code():
    rows = [(CUT_OFF, "ZZZ", 10.0), (CUT_OFF, "AAA", 10.0), (CUT_OFF, "MMM", 10.0)]
    result = compute_abc(_panel(rows), CUT_OFF)
    assert list(result["stock_code"]) == ["AAA", "MMM", "ZZZ"]


def test_compute_abc_sums_a_product_across_months():
    rows = [("2011-04", "P1", 30.0), (CUT_OFF, "P1", 50.0), (CUT_OFF, "P2", 20.0)]
    result = compute_abc(_panel(rows), CUT_OFF).set_index("stock_code")
    assert result.loc["P1", "revenue"] == pytest.approx(80.0)


def test_compute_abc_defaults_to_the_configured_thresholds():
    policy = load_inventory_policy().abc
    explicit = compute_abc(
        _panel(BASE_ROWS), CUT_OFF, a_cum_share=policy.a_cum_share, b_cum_share=policy.b_cum_share
    )
    pd.testing.assert_frame_equal(compute_abc(_panel(BASE_ROWS), CUT_OFF), explicit)


def test_compute_abc_honours_caller_supplied_thresholds():
    result = compute_abc(_panel(BASE_ROWS), CUT_OFF, a_cum_share=0.5, b_cum_share=0.9)
    classes = dict(zip(result["stock_code"], result["abc_class"], strict=True))
    assert classes["P1"] == "B"  # 0.80 cumulative is now past the tightened A boundary
    assert classes["P2"] == "B"  # lands exactly on the B boundary at 0.90
    assert classes["P3"] == "C"  # 0.95 is past it


def test_compute_abc_rejects_inverted_thresholds():
    with pytest.raises(ValueError, match="a_cum_share"):
        compute_abc(_panel(BASE_ROWS), CUT_OFF, a_cum_share=0.9, b_cum_share=0.5)


def test_compute_abc_handles_an_empty_window():
    result = compute_abc(_panel(BASE_ROWS), through_month="2009-12")
    assert result.empty
    assert list(result.columns) == ABC_COLUMNS


def test_abc_module_hard_codes_no_thresholds():
    """PRD §40 / §6 acceptance criterion: 0.80 and 0.95 live in config, never in the source."""
    source = (Path(__file__).resolve().parents[1] / "src" / "pipeline" / "abc.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert not re.search(r"\b0\.8\d*\b|\b0\.9[0-9]*\b", code)
