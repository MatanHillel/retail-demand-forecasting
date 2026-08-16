"""EDA report, insights and the numbers-in-tables guard (US-11, PRD §35A.2, §38, §41, §49).

The guard is the important part of this file. PRD §38 says no model ever computes a number, and
:func:`pipeline.narrative.numbers_in_tables` is the mechanism that enforces it — so it is tested
for what it must *reject* as much as for what it accepts.

Everything writes to a ``tmp_path`` base directory; no test touches the real ``artifacts/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import load_cleaning_config
from pipeline.download import compute_sha256, load_raw
from pipeline.eda.insights import MAX_INSIGHTS, MIN_INSIGHTS, build_insights, generate_insights
from pipeline.eda.report import EXTERNAL_MARKERS, MANDATORY_FIGURES, build_eda_report
from pipeline.eda.run_eda import EXPECTED_IDS, run_eda, validate_index
from pipeline.narrative import (
    NarrativeCheck,
    collect_table_values,
    extract_numbers,
    numbers_in_tables,
)
from pipeline.panel import build_panel
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

MANDATORY_IN_HTML = ("E01", "E02", "E05", "E06", "E07", "E08", "E09", "E11")
MIN_EMBEDDED_IMAGES = 8


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eda_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One full ``run_eda`` over the committed CI fixture."""
    base = tmp_path_factory.mktemp("run_eda")
    cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    ctx = RunContext.start(mode="no-llm", base_dir=base)
    # Stand in for US-03: in a real run ``load_raw(ctx=ctx)`` records the dataset identity, and
    # the report header shows that hash. The fixture hands ``raw_df`` in directly so the test
    # never touches the full extract, so it records the identity itself.
    ctx.record_data(
        file=RAW_SAMPLE, sha256=compute_sha256(RAW_SAMPLE), rows=len(raw), columns=raw.shape[1]
    )
    try:
        with ctx.step("run_eda"):
            clean_df, waterfall_df = clean_transactions(raw, cfg, ctx)
            returns = pd.read_parquet(base / "data/processed" / RETURNS_FILENAME)
            panel_df = build_panel(clean_df, returns, cfg, ctx)
            result = run_eda(clean_df, panel_df, waterfall_df, cfg, ctx, raw_df=raw)
        ctx.finish()
    finally:
        close_log_handlers(ctx.run_id)
    report = base / paths.EDA_REPORT.relative_to(paths.PROJECT_ROOT)
    insights = base / paths.INSIGHTS.relative_to(paths.PROJECT_ROOT)
    return SimpleNamespace(
        base=base, ctx=ctx, cfg=cfg, result=result,
        html=report.read_text(encoding="utf-8"),
        markdown=insights.read_text(encoding="utf-8"),
        report_path=report, insights_path=insights,
    )


@pytest.fixture
def ctx(tmp_path: Path):
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


# --------------------------------------------------------------------------
# extract_numbers
# --------------------------------------------------------------------------
def test_extract_numbers_handles_percent_and_thousands_separators() -> None:
    assert extract_numbers("Sep-Nov ~ 35 % of units; 1,024,951 rows") == [35.0, 1024951.0]


def test_extract_numbers_applies_a_magnitude_suffix() -> None:
    assert extract_numbers("revenue of £20.05 M") == [20_050_000.0]
    assert extract_numbers("about 4.7 k rows") == [4700.0]


def test_extract_numbers_ignores_identifiers() -> None:
    """``E08`` and ``I01`` are names, not measurements."""
    assert extract_numbers("E08 and I01 and E02_seasonal_index") == []


def test_extract_numbers_reads_plain_decimals() -> None:
    assert extract_numbers("a ratio of 0.27 and a count of 6") == [0.27, 6.0]


# --------------------------------------------------------------------------
# numbers_in_tables — what it must reject
# --------------------------------------------------------------------------
def test_guard_rejects_a_made_up_number() -> None:
    tables = {"T": pd.DataFrame({"units": [1024951], "share": [0.355]})}
    check = numbers_in_tables("The business sold 999,111 units.", tables)
    assert isinstance(check, NarrativeCheck)
    assert not check.passed
    assert "999,111" in check.unmatched


def test_guard_accepts_a_number_that_is_in_a_table() -> None:
    tables = {"T": pd.DataFrame({"units": [1024951], "share": [0.355]})}
    assert numbers_in_tables("The business sold 1,024,951 units.", tables).passed


def test_guard_accepts_a_share_written_as_a_percentage() -> None:
    """0.35512 in a table may legitimately be written "35.5 %"."""
    tables = {"T": pd.DataFrame({"share": [0.35512]})}
    assert numbers_in_tables("Sep-Nov is 35.5 % of units.", tables).passed


def test_guard_rejects_a_percentage_that_rounds_from_nothing() -> None:
    tables = {"T": pd.DataFrame({"share": [0.35512]})}
    assert not numbers_in_tables("Sep-Nov is 48.2 % of units.", tables).passed


def test_guard_is_not_fooled_by_a_near_miss() -> None:
    """4,999 is a different number from 4,998 — the whole point of the guard."""
    tables = {"T": pd.DataFrame({"rows": [4998]})}
    assert not numbers_in_tables("There are 4,999 rows.", tables).passed


def test_guard_whitelists_years_and_small_integers() -> None:
    """A sentence may say "Dec 2009 to Nov 2011" or "k = 6" without citing a table."""
    check = numbers_in_tables("From 2009 to 2011, with k = 6 over 12 months.", {})
    assert check.passed
    assert check.checked == 0


def test_guard_reads_dicts_as_well_as_frames() -> None:
    """``data_quality_findings.json`` is a nested dict, not a table."""
    tables = {"findings": {"duplicates": {"duplicate_rows": 34335, "warning": True}}}
    assert numbers_in_tables("34,335 duplicated rows.", tables).passed
    assert not numbers_in_tables("34,336 duplicated rows.", tables).passed


def test_collect_table_values_ignores_booleans() -> None:
    """``True`` is not the number 1; treating it as one would weaken the guard."""
    assert collect_table_values({"t": {"flag": True, "count": 3}}) == [3.0]


# --------------------------------------------------------------------------
# the HTML report
# --------------------------------------------------------------------------
def test_report_is_self_contained(eda_run: SimpleNamespace) -> None:
    """§35A.2: one file, no network. This is the criterion the build asserts on itself."""
    for marker in EXTERNAL_MARKERS:
        assert marker not in eda_run.html, f"report references {marker!r}"


def test_report_embeds_the_images(eda_run: SimpleNamespace) -> None:
    assert eda_run.html.count("data:image/png;base64,") >= MIN_EMBEDDED_IMAGES


def test_report_contains_every_mandatory_analysis(eda_run: SimpleNamespace) -> None:
    for name in MANDATORY_IN_HTML:
        assert name in eda_run.html, f"missing {name}"
    for figure in MANDATORY_FIGURES:
        assert figure in eda_run.html, f"missing mandatory figure {figure} (§35A.1, §49)"


def test_report_shows_the_waterfall_and_the_exclusion_list(eda_run: SimpleNamespace) -> None:
    assert "E01_cleaning_waterfall" in eda_run.html
    assert "remove_exact_duplicates" in eda_run.html
    # The exclusion list must show reasons, not just codes (§12).
    assert "Non-inventory exclusion list" in eda_run.html
    assert "postage" in eda_run.html


def test_report_header_carries_the_run_identity(eda_run: SimpleNamespace) -> None:
    assert eda_run.ctx.run_id in eda_run.html
    assert eda_run.ctx.data.sha256 in eda_run.html


def test_report_build_fails_when_a_mandatory_figure_is_absent(
    eda_run: SimpleNamespace,
) -> None:
    """A report that quietly lost a required figure must not be published (§49).

    Uses the fixture's own context so the remaining figures still resolve — the build has to fail
    on the *missing* one, not on being unable to find anything at all. ``_verify`` raises before
    the file is written, so the good report on disk is untouched.
    """
    trimmed = [entry for entry in eda_run.result["index"] if entry["id"] != "E6"]
    with pytest.raises(ValueError, match="mandatory figure"):
        build_eda_report(trimmed, eda_run.cfg, eda_run.ctx)


def test_report_is_registered_as_an_artifact(eda_run: SimpleNamespace) -> None:
    relative = paths.EDA_REPORT.relative_to(paths.PROJECT_ROOT).as_posix()
    assert eda_run.ctx.artifacts["eda_report"] == relative


# --------------------------------------------------------------------------
# insights.md
# --------------------------------------------------------------------------
def test_insights_count_is_within_the_required_range(eda_run: SimpleNamespace) -> None:
    numbered = re.findall(r"^\d+\. ", eda_run.markdown, flags=re.MULTILINE)
    assert MIN_INSIGHTS <= len(numbered) <= MAX_INSIGHTS
    assert len(numbered) == len(eda_run.result["insights"])


def test_every_insight_cites_a_table_and_states_a_number(eda_run: SimpleNamespace) -> None:
    for line in eda_run.markdown.splitlines():
        if not re.match(r"^\d+\. ", line):
            continue
        assert re.search(r"\(E\d+, table `E\d{2}_[A-Za-z0-9_]+`\)$", line), line
        assert extract_numbers(line), f"insight states no number: {line}"


def test_insights_numbers_are_all_backed_by_tables(eda_run: SimpleNamespace) -> None:
    """The acceptance criterion: running the guard on insights.md returns passed=True."""
    check = numbers_in_tables(eda_run.markdown, eda_run.result["tables"])
    assert check.passed, check.summary()
    assert check.checked > 0


def test_insight_ids_are_stable(eda_run: SimpleNamespace) -> None:
    """US-12 diffs the LLM rewrite against this list, so the ids may not drift.

    Stability binds an id to a *topic*, not to a position: ``I08`` is always intermittency, even
    though the narrative orders the findings by importance and drops a topic when its table is
    missing. That is what keeps the two versions diffable line for line (§3).
    """
    insights = eda_run.result["insights"]
    ids = [insight.id for insight in insights]
    assert len(set(ids)) == len(ids), "an id may not be reused"
    assert all(re.fullmatch(r"I\d{2}", value) for value in ids)

    topics = {insight.id: insight.e_ref for insight in insights}
    assert topics["I01"] == "E2"
    assert topics["I06"] == "E6"
    assert topics["I08"] == "E8"
    assert topics["I12"] == "E12"


def test_insights_generation_refuses_an_unbacked_number(
    eda_run: SimpleNamespace, ctx: RunContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a template ever states a number the tables do not contain, generation must fail."""
    from pipeline.eda import insights as insights_module

    def _liar(tables, cfg):  # noqa: ANN001, ANN202 - test double
        real = build_insights(tables, cfg)
        real[0] = real[0].model_copy(update={"text": "The business sold 987,654,321 units."})
        return real

    monkeypatch.setattr(insights_module, "build_insights", _liar)
    with pytest.raises(ValueError, match="no computed table"):
        with ctx.step("insights"):
            generate_insights(eda_run.result["tables"], eda_run.cfg, ctx)


def test_insights_are_registered_as_an_artifact(eda_run: SimpleNamespace) -> None:
    relative = paths.INSIGHTS.relative_to(paths.PROJECT_ROOT).as_posix()
    assert eda_run.ctx.artifacts["insights"] == relative


# --------------------------------------------------------------------------
# run_eda
# --------------------------------------------------------------------------
def test_run_eda_produces_all_fourteen_analyses(eda_run: SimpleNamespace) -> None:
    ids = [entry["id"] for entry in eda_run.result["index"]]
    assert ids == list(EXPECTED_IDS)
    assert len(ids) == 14


def test_run_eda_writes_both_required_artifacts(eda_run: SimpleNamespace) -> None:
    """``eda_report.html`` and ``insights.md`` are two of the eight required names (§41)."""
    assert eda_run.report_path.is_file()
    assert eda_run.insights_path.is_file()
    assert eda_run.report_path.name == "eda_report.html"
    assert eda_run.insights_path.name == "insights.md"


def test_run_eda_records_metrics(eda_run: SimpleNamespace) -> None:
    metrics = eda_run.ctx.metrics
    assert metrics["eda_analyses"] == 14
    assert metrics["eda_insights"] == len(eda_run.result["insights"])
    assert metrics["eda_figures"] > 0


def test_missing_analysis_names_the_e_id(eda_run: SimpleNamespace, ctx: RunContext) -> None:
    """§3: run_eda must fail with a clear message naming the missing E-id."""
    trimmed = [entry for entry in eda_run.result["index"] if entry["id"] != "E9"]
    result = validate_index(trimmed, ctx)
    assert not result.passed
    assert any("E9" in violation.message for violation in result.violations)
    error = FlowValidationError(result)
    assert str(error).startswith("FLOW STOPPED:")


def test_no_crewai_import_anywhere_in_the_eda_path() -> None:
    """Rule 10: ``--no-llm`` mode must stay free of any LLM import."""
    for module in (
        paths.PROJECT_ROOT / "src/pipeline/narrative.py",
        paths.PROJECT_ROOT / "src/pipeline/eda/run_eda.py",
        paths.PROJECT_ROOT / "src/pipeline/eda/report.py",
        paths.PROJECT_ROOT / "src/pipeline/eda/insights.py",
    ):
        source = module.read_text(encoding="utf-8")
        assert "crewai" not in source.lower()


def test_written_artifacts_use_unix_line_endings(eda_run: SimpleNamespace) -> None:
    for path in (eda_run.report_path, eda_run.insights_path):
        assert b"\r\n" not in path.read_bytes(), f"CRLF line endings in {path.name}"
