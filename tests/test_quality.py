"""Data-quality profiling tests (US-07, PRD §8, §11, §12, §35A E1, §55).

Structure and rules only. The PRD's indicative counts (12,133 duplicated rows, 22.8 % anonymous
rows, …) are expectations to compare a real run against, never assertions — a test that pinned
them would fail the moment the dataset is re-downloaded. Everything writes to a ``tmp_path`` base
directory, so no test can touch the real ``artifacts/`` tree or the repository's own
``config/non_inventory_stockcodes.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipeline import paths
from pipeline.cleaning import clean_transactions
from pipeline.config import load_cleaning_config, load_non_inventory_codes
from pipeline.download import load_raw
from pipeline.eda import e01_data_quality
from pipeline.panel import build_panel
from pipeline.quality import (
    EXCLUSION_COLUMNS,
    STATUS_CONFIRMED,
    STATUS_INITIAL,
    STATUS_REVIEW_NEEDED,
    abnormal_quantities,
    build_data_quality_findings,
    confirm_exclusion_list,
    detect_duplicates,
    list_nonproduct_codes,
    profile_raw,
)
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

FINDINGS_SECTIONS = [
    "raw_profile",
    "missing_values",
    "cancellations_adjustments",
    "duplicates",
    "nonproduct_codes",
    "abnormal_quantities",
    "partial_month",
    "waterfall",
    "panel_preview",
    "warnings",
]

E1_TABLES = [
    "E01_cleaning_waterfall.csv",
    "E01_missing_values.csv",
    "E01_nonproduct_codes.csv",
    "E01_abnormal_lines.csv",
    "E01_duplicates_summary.json",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(rows: list[dict]) -> pd.DataFrame:
    """Tiny raw-shaped frame; each dict overrides the defaults of one row."""
    defaults = {
        "Invoice": "500001",
        "StockCode": "10001",
        "Description": "RED MUG",
        "Quantity": 2,
        "InvoiceDate": pd.Timestamp("2010-03-05 10:00:00"),
        "Price": 1.5,
        "Customer ID": 13085.0,
        "Country": "United Kingdom",
        "source_sheet": "Year 2009-2010",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


@pytest.fixture
def ctx(tmp_path: Path):
    """A run whose artifacts and config writes land under ``tmp_path``."""
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One full E1 run over the committed CI fixture, shared by the structural tests."""
    base = tmp_path_factory.mktemp("quality_sample")
    cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    context = RunContext.start(mode="no-llm", base_dir=base)
    try:
        with context.step("data_quality_profiling"):
            clean_df, waterfall_df = clean_transactions(raw, cfg, context)
            returns = pd.read_parquet(base / "data/processed/returns_lines.parquet")
            panel_df = build_panel(clean_df, returns, cfg, context)
            result = e01_data_quality.run(raw, clean_df, waterfall_df, panel_df, cfg, context)
        context.finish()
    finally:
        close_log_handlers(context.run_id)
    return SimpleNamespace(
        base=base, ctx=context, raw=raw, clean=clean_df, panel=panel_df,
        waterfall=waterfall_df, result=result, cfg=cfg,
    )


# --------------------------------------------------------------------------
# detect_duplicates (§11)
# --------------------------------------------------------------------------
def test_detect_duplicates_warns_above_the_configured_share() -> None:
    """3 duplicated rows out of 100 is 3 %, above the 1.5 % threshold — warning must be set."""
    cfg = load_cleaning_config()
    unique = [{"Invoice": f"5{index:05d}", "StockCode": f"1{index:04d}"} for index in range(96)]
    duplicated = [{"Invoice": "600001", "StockCode": "20001"}] * 4  # 1 original + 3 extra copies
    frame = _frame([*unique, *duplicated])
    assert len(frame) == 100

    result = detect_duplicates(frame, cfg)

    assert result["duplicate_rows"] == 3
    assert result["duplicate_row_share"] == pytest.approx(0.03)
    assert result["duplicate_row_share"] > cfg.warnings.duplicate_max_row_share
    assert result["warning"] is True
    assert result["top_duplicated_lines"][0]["occurrences"] == 4


def test_detect_duplicates_stays_quiet_below_the_threshold() -> None:
    frame = _frame([{"Invoice": f"5{index:05d}", "StockCode": f"1{index:04d}"} for index in
                    range(200)])
    result = detect_duplicates(frame)
    assert result["duplicate_rows"] == 0
    assert result["warning"] is False
    assert result["top_duplicated_lines"] == []


def test_detect_duplicates_ignores_the_source_sheet_column() -> None:
    """The subset is the eight original columns; ``source_sheet`` must not split a duplicate."""
    frame = _frame(
        [
            {"Invoice": "500009", "source_sheet": "Year 2009-2010"},
            {"Invoice": "500009", "source_sheet": "Year 2010-2011"},
        ]
    )
    result = detect_duplicates(frame)
    assert "source_sheet" not in result["subset"]
    assert result["duplicate_rows"] == 1


def test_detect_duplicates_splits_within_and_cross_sheet_counts() -> None:
    """Same line in two sheets is a cross-sheet duplicate; twice in one sheet is within-sheet."""
    frame = _frame(
        [
            {"Invoice": "500010", "source_sheet": "Year 2009-2010"},
            {"Invoice": "500010", "source_sheet": "Year 2010-2011"},  # cross-sheet copy
            {"Invoice": "500011", "source_sheet": "Year 2010-2011"},
            {"Invoice": "500011", "source_sheet": "Year 2010-2011"},  # within-sheet copy
        ]
    )
    result = detect_duplicates(frame)
    assert result["duplicate_rows"] == 2
    assert result["within_source_duplicate_rows"] == 1
    assert result["cross_source_duplicate_rows"] == 1


def test_profile_raw_reports_overlapping_source_sheets() -> None:
    """The two UCI sheets share a month; concatenating them double-counts it (§8)."""
    frame = _frame(
        [
            {"Invoice": "500020", "InvoiceDate": pd.Timestamp("2010-12-05 10:00:00"),
             "source_sheet": "Year 2009-2010"},
            {"Invoice": "500020", "InvoiceDate": pd.Timestamp("2010-12-05 10:00:00"),
             "source_sheet": "Year 2010-2011"},
            {"Invoice": "500021", "InvoiceDate": pd.Timestamp("2011-05-05 10:00:00"),
             "source_sheet": "Year 2010-2011"},
        ]
    )
    overlap = profile_raw(frame)["raw_profile"]["source_sheets"]["overlap"]
    assert overlap["months"] == ["2010-12"]
    assert overlap["invoices_in_more_than_one_sheet"] == 1


def test_profile_raw_reports_no_overlap_for_disjoint_sheets() -> None:
    frame = _frame(
        [
            {"InvoiceDate": pd.Timestamp("2010-03-05 10:00:00"),
             "source_sheet": "Year 2009-2010"},
            {"InvoiceDate": pd.Timestamp("2011-05-05 10:00:00"),
             "source_sheet": "Year 2010-2011"},
        ]
    )
    overlap = profile_raw(frame)["raw_profile"]["source_sheets"]["overlap"]
    assert overlap["months"] == []
    assert overlap["invoices_in_more_than_one_sheet"] == 0


# --------------------------------------------------------------------------
# list_nonproduct_codes (§12)
# --------------------------------------------------------------------------
def test_list_nonproduct_codes_flags_only_the_non_product_codes() -> None:
    frame = _frame(
        [
            {"StockCode": "85123A"},
            {"StockCode": "DCGS0070"},
            {"StockCode": "POST"},
            {"StockCode": "BANK CHARGES"},
        ]
    )
    codes = list_nonproduct_codes(frame)
    flagged = set(codes["stock_code"])
    assert flagged == {"POST", "BANK CHARGES"}
    assert codes["in_exclusion_list"].all()


def test_list_nonproduct_codes_normalises_before_matching() -> None:
    """A lower-case variant is the same code after §10 step 3 normalisation."""
    codes = list_nonproduct_codes(_frame([{"StockCode": " post "}, {"StockCode": "85123a"}]))
    assert list(codes["stock_code"]) == ["POST"]


def test_list_nonproduct_codes_reports_rows_units_and_a_description() -> None:
    frame = _frame(
        [
            {"StockCode": "POST", "Quantity": 3, "Price": 2.0, "Description": "POSTAGE"},
            {"StockCode": "POST", "Quantity": 5, "Price": 2.0, "Description": "POSTAGE"},
            {"StockCode": "POST", "Quantity": 1, "Price": 2.0, "Description": "CARRIAGE"},
        ]
    )
    row = list_nonproduct_codes(frame).iloc[0]
    assert row["rows"] == 3
    assert row["units"] == 9
    assert row["revenue"] == pytest.approx(18.0)
    assert row["sample_description"] == "POSTAGE"  # most frequent, ties alphabetical


# --------------------------------------------------------------------------
# confirm_exclusion_list (§12) — the write-back contract
# --------------------------------------------------------------------------
def test_confirm_exclusion_list_marks_present_codes_confirmed(ctx: RunContext) -> None:
    before = load_non_inventory_codes().set_index("stock_code")["status"]
    codes = list_nonproduct_codes(_frame([{"StockCode": "POST"}]))

    updated = confirm_exclusion_list(codes, ctx)

    assert list(updated.columns) == EXCLUSION_COLUMNS
    assert updated.loc[updated["stock_code"] == "POST", "status"].item() == STATUS_CONFIRMED
    # A listed code that does not appear in the data keeps whatever status it already had —
    # being unused is not a reason to promote it. Compared against the file's own prior state,
    # never a fixed value: a real run rewrites this column.
    assert updated.loc[updated["stock_code"] == "DOT", "status"].item() == before["DOT"]


def test_confirm_exclusion_list_appends_unknown_codes_as_review_needed(ctx: RunContext) -> None:
    codes = list_nonproduct_codes(_frame([{"StockCode": "MYSTERY9"}]))
    assert not codes["in_exclusion_list"].any()

    updated = confirm_exclusion_list(codes, ctx)

    row = updated.loc[updated["stock_code"] == "MYSTERY9"]
    assert row["status"].item() == STATUS_REVIEW_NEEDED
    assert row["reason"].item()  # a non-empty reason is part of the column contract
    assert any("MYSTERY9" in warning for warning in ctx.warnings)


def test_confirm_exclusion_list_keeps_the_case_variant_rows_distinct(ctx: RunContext) -> None:
    """``M`` and ``m`` are separate rows on purpose; merging them would break every later run.

    ``load_non_inventory_codes()`` raises on a duplicated ``stock_code``, and
    ``config_snapshot()`` calls it inside ``RunContext.start()`` — so an upper-cased write-back
    would make the *next* run fail rather than this one.
    """
    codes = list_nonproduct_codes(_frame([{"StockCode": "m"}]))
    updated = confirm_exclusion_list(codes, ctx)

    assert list(updated["stock_code"]).count("M") == 1
    assert list(updated["stock_code"]).count("m") == 1
    assert not updated["stock_code"].duplicated().any()
    # Both spellings normalise to the code found in the data, so both are confirmed.
    assert set(updated.loc[updated["stock_code"].isin(["M", "m"]), "status"]) == {
        STATUS_CONFIRMED
    }


def test_confirm_exclusion_list_writes_a_readable_csv(ctx: RunContext, tmp_path: Path) -> None:
    """The written file must satisfy the loader's column contract, or every later run breaks."""
    repository_before = paths.NON_INVENTORY_STOCKCODES.read_bytes()
    confirm_exclusion_list(list_nonproduct_codes(_frame([{"StockCode": "POST"}])), ctx)

    written = tmp_path / paths.NON_INVENTORY_STOCKCODES.relative_to(paths.PROJECT_ROOT)
    assert written.is_file()
    assert b"\r\n" not in written.read_bytes(), "config must stay LF on every platform (§40)"
    frame = pd.read_csv(written, dtype=str, keep_default_na=False, na_values=[])
    assert list(frame.columns) == EXCLUSION_COLUMNS
    assert not frame["stock_code"].duplicated().any()
    assert set(frame["status"]) <= {STATUS_INITIAL, STATUS_CONFIRMED, STATUS_REVIEW_NEEDED}
    # A test run writes under its own base_dir and never touches the repository's copy.
    assert paths.NON_INVENTORY_STOCKCODES.read_bytes() == repository_before


# --------------------------------------------------------------------------
# profile_raw (§8) and abnormal_quantities (§10)
# --------------------------------------------------------------------------
def test_profile_raw_counts_cancellations_adjustments_and_anonymous_rows() -> None:
    frame = _frame(
        [
            {"Invoice": "C500002", "Quantity": -2},
            {"Invoice": "A500003", "Price": -11062.06},
            {"Invoice": "500004", "Customer ID": None},
            {"Invoice": "500005", "Quantity": -1},  # negative without a C invoice
            {"Invoice": "500006", "Price": 0.0},
        ]
    )
    profile = profile_raw(frame)

    assert profile["cancellations_adjustments"]["cancellations"]["rows"] == 1
    assert profile["cancellations_adjustments"]["adjustments"]["rows"] == 1
    assert profile["cancellations_adjustments"]["adjustments"]["lines"][0]["price_sign"] == -1
    assert profile["missing_values"]["customer_id"]["rows"] == 1
    assert profile["missing_values"]["customer_id"]["row_share"] == pytest.approx(0.2)
    assert profile["raw_profile"]["nonpositive_quantity_without_cancellation"]["rows"] == 1
    # The zero-price line and the negative-price adjustment line — the profile counts every
    # non-positive price, whatever invoice it sits on.
    assert profile["raw_profile"]["nonpositive_price"]["rows"] == 2


def test_profile_raw_flags_the_partial_month_and_lower_case_variants() -> None:
    cfg = load_cleaning_config()
    partial = pd.Timestamp(f"{cfg.raw.partial_months[0]}-05 10:00:00")
    frame = _frame(
        [
            {"InvoiceDate": partial, "StockCode": "85123a"},
            {"InvoiceDate": pd.Timestamp("2010-03-05 10:00:00")},
        ]
    )
    profile = profile_raw(frame, cfg)

    assert profile["partial_month"]["rows"] == 1
    assert profile["raw_profile"]["lowercase_stock_code_variants"]["rows"] == 1
    flags = {row["month"]: row["is_partial"] for row in profile["raw_profile"]["rows_per_month"]}
    assert flags[cfg.raw.partial_months[0]] is True
    assert flags["2010-03"] is False


def test_profile_raw_reports_the_weekday_distribution() -> None:
    """2010-03-06 is a Saturday — the fact §8 wants documented."""
    frame = _frame(
        [
            {"InvoiceDate": pd.Timestamp("2010-03-05 10:00:00")},  # Friday
            {"InvoiceDate": pd.Timestamp("2010-03-06 10:00:00")},  # Saturday
        ]
    )
    distribution = {
        row["weekday_name"]: row["rows"]
        for row in profile_raw(frame)["raw_profile"]["weekday_distribution"]
    }
    assert distribution == {"Friday": 1, "Saturday": 1}


def test_abnormal_quantities_uses_the_configured_threshold(sample_run: SimpleNamespace) -> None:
    threshold = sample_run.cfg.warnings.abnormal_line_quantity
    lines = abnormal_quantities(sample_run.clean, cfg=sample_run.cfg)

    assert list(lines.columns) == [
        "invoice", "stock_code", "description", "quantity", "price",
        "customer_identified", "month",
    ]
    assert (lines["quantity"] > threshold).all()
    assert lines["quantity"].is_monotonic_decreasing
    # Agrees with the flag cleaning already computed — one definition, not two.
    assert len(lines) == int(sample_run.clean["is_abnormal_qty"].sum())


# --------------------------------------------------------------------------
# findings JSON and the E1 artifacts
# --------------------------------------------------------------------------
def test_findings_json_has_every_required_section(sample_run: SimpleNamespace) -> None:
    written = sample_run.base / paths.DATA_QUALITY_FINDINGS.relative_to(paths.PROJECT_ROOT)
    assert written.is_file()

    findings = json.loads(written.read_text(encoding="utf-8"))
    for section in FINDINGS_SECTIONS:
        assert section in findings, f"missing section: {section}"
    assert findings["run_id"] == sample_run.ctx.run_id
    assert "data_sha256" in findings
    assert "generated_at" in findings
    assert findings["waterfall"], "the waterfall section must carry the US-04 rows"


def test_findings_are_registered_as_an_artifact(sample_run: SimpleNamespace) -> None:
    recorded = sample_run.ctx.artifacts["data_quality_findings"]
    assert recorded == paths.DATA_QUALITY_FINDINGS.relative_to(paths.PROJECT_ROOT).as_posix()


def test_findings_panel_preview_matches_the_panel(sample_run: SimpleNamespace) -> None:
    preview = sample_run.result["findings"]["panel_preview"]
    assert preview["rows"] == len(sample_run.panel)
    assert preview["products"] == sample_run.panel["stock_code"].nunique()
    assert preview["zero_rows"] == int((sample_run.panel["units_sold"] == 0).sum())


def test_e1_writes_every_table_and_the_waterfall_figure(sample_run: SimpleNamespace) -> None:
    tables_dir = sample_run.base / paths.EDA_TABLES_DIR.relative_to(paths.PROJECT_ROOT)
    for name in E1_TABLES:
        assert (tables_dir / name).is_file(), f"missing table: {name}"

    figure = (
        sample_run.base
        / paths.FIGURES_DIR.relative_to(paths.PROJECT_ROOT)
        / "E01_waterfall.png"
    )
    assert figure.is_file()
    assert figure.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_e1_duplicates_summary_is_a_one_row_records_array(sample_run: SimpleNamespace) -> None:
    written = (
        sample_run.base
        / paths.EDA_TABLES_DIR.relative_to(paths.PROJECT_ROOT)
        / "E01_duplicates_summary.json"
    )
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) == 1
    assert {"duplicate_rows", "duplicate_row_share", "warning"} <= set(payload[0])


def test_written_artifacts_use_unix_line_endings(sample_run: SimpleNamespace) -> None:
    """No CRLF anywhere: the same run must produce identical bytes on Windows and in CI (§40)."""
    written = [sample_run.base / paths.DATA_QUALITY_FINDINGS.relative_to(paths.PROJECT_ROOT)]
    tables_dir = sample_run.base / paths.EDA_TABLES_DIR.relative_to(paths.PROJECT_ROOT)
    written += [tables_dir / name for name in E1_TABLES]

    for path in written:
        assert b"\r\n" not in path.read_bytes(), f"CRLF line endings in {path.name}"


def test_e1_fails_clearly_when_the_us04_waterfall_is_missing(
    ctx: RunContext, tmp_path: Path
) -> None:
    """E1 verifies the waterfall table; it never silently regenerates US-04's output."""
    cfg = load_cleaning_config()
    frame = _frame([{"Invoice": "500001"}])
    with pytest.raises(FileNotFoundError, match="E01_cleaning_waterfall"):
        with ctx.step("e1"):
            e01_data_quality.run(
                frame, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), cfg, ctx
            )


def test_findings_write_is_staged_until_promotion(tmp_path: Path) -> None:
    """Under staging the findings file must not appear at its final path before ``promote()``."""
    cfg = load_cleaning_config()
    context = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        with context.step("data_quality_profiling"):
            raw, _ = load_raw(RAW_SAMPLE)
            clean_df, waterfall_df = clean_transactions(raw, cfg, context)
            returns = pd.read_parquet(
                context.staging_dir / "data/processed/returns_lines.parquet"
            )
            panel_df = build_panel(clean_df, returns, cfg, context)
            build_data_quality_findings(raw, clean_df, waterfall_df, panel_df, cfg, context)

        relative = paths.DATA_QUALITY_FINDINGS.relative_to(paths.PROJECT_ROOT)
        assert (context.staging_dir / relative).is_file()
        assert not (tmp_path / relative).exists()

        context.promote()
        assert (tmp_path / relative).is_file()
    finally:
        close_log_handlers(context.run_id)
