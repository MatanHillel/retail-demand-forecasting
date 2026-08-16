"""Dataset contract tests (US-08, PRD Appendix A, §12, §13.2, §14, §21, §37 step 3, §39, §55).

The contract is the machine-checked promise about ``clean_data.csv``, so these tests prove two
things: the document is *generated* from the panel and the configuration rather than typed, and the
validator catches each way the panel can break it — with the right rule name and the exact §39
wording, which the Flow reuses verbatim.

No PRD business number is asserted (row counts, product counts). Every write goes to a ``tmp_path``
base directory.
"""

from __future__ import annotations

import json
import re
import tokenize
from pathlib import Path

import pandas as pd
import pytest

from pipeline import paths
from pipeline.config import (
    load_cleaning_config,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.contract import (
    CLEANING_ASSUMPTIONS,
    CONTRACT_STEP,
    FEATURE_CONVENTIONS,
    LEAKAGE_RULES,
    contract_failure_message,
    validate_contract,
    validate_contract_files,
    write_contract,
)
from pipeline.panel import PANEL_COLUMNS
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError, ValidationResult

#: Keys the §6 acceptance criterion requires, plus the run-identity fields of §2.
REQUIRED_KEYS = [
    "dataset",
    "version",
    "source",
    "generated_at",
    "run_id",
    "data_sha256",
    "grain",
    "primary_key",
    "date_range",
    "columns",
    "cleaning_assumptions",
    "exclusion_list",
    "active_rule",
    "partial_month_rule",
    "leakage_rules",
    "modeling_split",
    "row_counts",
    "feature_conventions",
]


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def ctx(tmp_path: Path):
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


def _panel_months() -> list[str]:
    """Every month the panel spans, from the configured first month to the partial one."""
    raw = load_cleaning_config().raw
    end = max([raw.last_full_month, *raw.partial_months])
    return [str(period) for period in pd.period_range(raw.first_month, end, freq="M")]


def _panel() -> pd.DataFrame:
    """A small, fully valid panel: three products with different first-sale months.

    ``DCGSSGIRL`` starts in the partial month and therefore has exactly one row — the §4 edge case
    the grain check must accept.
    """
    months = _panel_months()
    partial = set(load_cleaning_config().raw.partial_months)
    first_sale_offsets = {"10002": 0, "21232A": 5, "DCGSSGIRL": len(months) - 1}

    rows = []
    for stock_code, start in first_sale_offsets.items():
        for offset, month in enumerate(months[start:]):
            # Offset 0 is the first observed sale, so it must be positive; the rest alternate
            # between sales and zero-filled months.
            units = 0 if offset % 3 == 1 else 10 + offset
            rows.append(
                {
                    "month": month,
                    "stock_code": stock_code,
                    "description": "TEST ITEM",
                    "units_sold": units,
                    "gross_revenue": round(units * 2.5, 4),
                    "avg_unit_price": 2.5,
                    "invoice_count": 1 if units else 0,
                    "sale_line_count": 1 if units else 0,
                    "customer_count": 1 if units else 0,
                    "max_line_qty": units,
                    "returned_units": 0,
                    "is_partial_month": month in partial,
                }
            )
    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    return panel.sort_values(["stock_code", "month"], kind="mergesort").reset_index(drop=True)


def _contract(ctx: RunContext, panel: pd.DataFrame | None = None) -> dict:
    return write_contract(
        _panel() if panel is None else panel,
        load_cleaning_config(),
        load_model_config(),
        load_non_inventory_codes(),
        ctx,
    )


# --------------------------------------------------------------------------
# writer — Appendix A content
# --------------------------------------------------------------------------
def test_contract_has_every_required_key(ctx):
    contract = _contract(ctx)
    assert [key for key in REQUIRED_KEYS if key not in contract] == []


def test_columns_block_is_exactly_the_twelve_panel_fields(ctx):
    contract = _contract(ctx)
    assert list(contract["columns"]) == PANEL_COLUMNS


def test_every_column_declares_a_type_and_nullability(ctx):
    columns = _contract(ctx)["columns"]
    assert all("type" in spec and "nullable" in spec for spec in columns.values())
    assert columns["description"]["nullable"] is True
    assert columns["month"]["format"] == "YYYY-MM"
    assert columns["returned_units"]["note"] == "EDA only - never a feature"


def test_active_rule_k_comes_from_model_config(ctx):
    contract = _contract(ctx)
    k = load_model_config().active_rule.k
    assert contract["active_rule"]["k"] == k
    assert str(k) in contract["active_rule"]["definition"]


def test_exclusion_list_has_one_entry_per_csv_row(ctx):
    exclusions = load_non_inventory_codes()
    contract = _contract(ctx)
    assert len(contract["exclusion_list"]) == len(exclusions)
    assert set(contract["exclusion_list"][0]) == {"stock_code", "reason", "status"}


def test_stock_code_pattern_is_the_configured_one_and_json_escaped(ctx, tmp_path):
    contract = _contract(ctx)
    assert (
        contract["columns"]["stock_code"]["pattern"]
        == load_cleaning_config().rules.inventory_code_pattern
    )
    written = (tmp_path / paths.DATASET_CONTRACT.relative_to(paths.PROJECT_ROOT)).read_text(
        encoding="utf-8"
    )
    assert '"^\\\\d{5}[A-Z]{0,2}$|^DCGS"' in written


def test_modeling_split_uses_the_appendix_a_key_names(ctx):
    split = load_model_config().split
    modeling_split = _contract(ctx)["modeling_split"]
    assert set(modeling_split) == {"train_targets", "test_targets"}
    assert modeling_split["train_targets"] == (
        f"{split.train_targets.start}..{split.train_targets.end}"
    )
    assert modeling_split["test_targets"] == (
        f"{split.holdout_targets.start}..{split.holdout_targets.end}"
    )


def test_date_range_is_computed_from_the_panel(ctx):
    raw = load_cleaning_config().raw
    panel = _panel()
    date_range = _contract(ctx, panel)["date_range"]
    assert date_range["first_month"] == panel["month"].min()
    assert date_range["last_full_month"] == raw.last_full_month
    assert date_range["partial_months"] == list(raw.partial_months)


def test_row_counts_are_computed_from_the_panel(ctx):
    panel = _panel()
    counts = _contract(ctx, panel)["row_counts"]
    assert counts["rows"] == len(panel)
    assert counts["products"] == panel["stock_code"].nunique()
    assert counts["zero_rows"] == int((panel["units_sold"] == 0).sum())
    assert counts["partial_rows"] == int(panel["is_partial_month"].sum())


def test_fixed_appendix_a_prose_is_carried_verbatim(ctx):
    contract = _contract(ctx)
    assert contract["cleaning_assumptions"] == CLEANING_ASSUMPTIONS
    assert contract["leakage_rules"] == LEAKAGE_RULES
    assert contract["feature_conventions"] == FEATURE_CONVENTIONS


def test_contract_is_written_under_the_run_base_dir_and_registered(ctx, tmp_path):
    _contract(ctx)
    relative = paths.DATASET_CONTRACT.relative_to(paths.PROJECT_ROOT)
    written = tmp_path / relative
    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8"))["dataset"] == "clean_data"
    assert ctx.artifacts["dataset_contract"] == relative.as_posix()
    # The file is committed, so it must be byte-identical on Windows and on CI (§40).
    assert b"\r\n" not in written.read_bytes()


def test_data_sha256_is_null_until_the_raw_file_is_recorded(ctx):
    assert _contract(ctx)["data_sha256"] is None
    ctx.record_data(file="raw.csv", sha256="abc123", rows=1, columns=8)
    assert _contract(ctx)["data_sha256"] == "abc123"


def test_a_panel_starting_late_is_recorded_and_warned_about(ctx):
    panel = _panel()
    late = panel[panel["month"] > panel["month"].min()].reset_index(drop=True)
    contract = _contract(ctx, late)
    assert contract["date_range"]["first_month"] == late["month"].min()
    assert any("cleaning_config.raw.first_month" in warning for warning in ctx.warnings)


# --------------------------------------------------------------------------
# validator — the happy path
# --------------------------------------------------------------------------
def test_validation_passes_on_the_fixture_panel(ctx):
    panel = _panel()
    result = validate_contract(panel, _contract(ctx, panel))
    assert isinstance(result, ValidationResult)
    assert result.passed, [violation.model_dump() for violation in result.violations]
    assert result.step == CONTRACT_STEP
    assert result.checked_rows == len(panel)


def test_a_product_selling_only_in_the_partial_month_is_valid(ctx):
    panel = _panel()
    assert (panel["stock_code"] == "DCGSSGIRL").sum() == 1  # the §4 edge case
    assert validate_contract(panel, _contract(ctx, panel)).passed


def test_validation_does_not_modify_the_panel(ctx):
    panel = _panel()
    before = panel.copy(deep=True)
    validate_contract(panel, _contract(ctx, panel))
    pd.testing.assert_frame_equal(panel, before)


# --------------------------------------------------------------------------
# validator — one corruption per rule
# --------------------------------------------------------------------------
def _corrupt_drop_column(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.drop(columns=["units_sold"])


def _corrupt_duplicate_key(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([panel, panel.head(1)], ignore_index=True)


def _corrupt_negative_units(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    corrupted.loc[corrupted.index[-1], "units_sold"] = -5
    return corrupted


def _corrupt_row_before_first_sale(panel: pd.DataFrame) -> pd.DataFrame:
    """Zero out a product's first month, so the panel claims history before its first sale."""
    corrupted = panel.copy()
    first = corrupted.index[corrupted["stock_code"] == "10002"][0]
    corrupted.loc[first, ["units_sold", "gross_revenue", "max_line_qty"]] = 0
    return corrupted


def _corrupt_partial_flag(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    holdout_start = load_model_config().split.holdout_targets.start
    corrupted.loc[corrupted["month"] == holdout_start, "is_partial_month"] = True
    return corrupted


def _corrupt_month_out_of_range(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    beyond = str(pd.Period(corrupted["month"].max(), freq="M") + 1)
    corrupted.loc[corrupted.index[-1], "month"] = beyond
    return corrupted


def _corrupt_null_units(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    corrupted["units_sold"] = corrupted["units_sold"].astype("float64")
    corrupted.loc[corrupted.index[-1], "units_sold"] = None
    return corrupted


def _corrupt_stock_code(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    corrupted["stock_code"] = corrupted["stock_code"].replace({"10002": "POST"})
    return corrupted


def _corrupt_month_format(panel: pd.DataFrame) -> pd.DataFrame:
    corrupted = panel.copy()
    corrupted.loc[corrupted.index[-1], "month"] = "2011/12"
    return corrupted


def _corrupt_extra_column(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(forecast=1)


def _corrupt_month_gap(panel: pd.DataFrame) -> pd.DataFrame:
    """Delete a middle month of the oldest product, leaving a hole in its history."""
    corrupted = panel[panel["stock_code"] == "10002"]
    drop = corrupted.index[len(corrupted) // 2]
    return panel.drop(index=drop).reset_index(drop=True)


CORRUPTIONS = [
    ("columns", _corrupt_drop_column),
    ("unexpected_columns", _corrupt_extra_column),
    ("primary_key", _corrupt_duplicate_key),
    ("non_negative", _corrupt_negative_units),
    ("first_row_is_a_sale", _corrupt_row_before_first_sale),
    ("is_partial_month", _corrupt_partial_flag),
    ("month_range", _corrupt_month_out_of_range),
    ("month_format", _corrupt_month_format),
    ("nullable", _corrupt_null_units),
    ("stock_code_pattern", _corrupt_stock_code),
    ("contiguous_months", _corrupt_month_gap),
]


@pytest.mark.parametrize("rule,corrupt", CORRUPTIONS, ids=[rule for rule, _ in CORRUPTIONS])
def test_each_corruption_fails_with_its_own_rule(ctx, rule, corrupt):
    panel = _panel()
    contract = _contract(ctx, panel)

    result = validate_contract(corrupt(panel), contract)

    assert not result.passed
    assert rule in {violation.rule for violation in result.violations}
    assert all(violation.step == CONTRACT_STEP for violation in result.violations)


def test_a_missing_column_short_circuits_to_a_single_violation(ctx):
    """AC 2 expects exactly '(1 violations)' after deleting units_sold."""
    panel = _panel()
    result = validate_contract(_corrupt_drop_column(panel), _contract(ctx, panel))
    assert len(result.violations) == 1
    assert result.violations[0].rule == "columns"


def test_row_count_differences_are_a_warning_not_a_violation(ctx):
    panel = _panel()
    contract = _contract(ctx, panel)
    shorter = panel[panel["stock_code"] != "DCGSSGIRL"].reset_index(drop=True)

    result = validate_contract(shorter, contract)

    assert "row_counts" in result.extra
    assert "row_counts" not in {violation.rule for violation in result.violations}


# --------------------------------------------------------------------------
# failure semantics (§39) — the wording the Flow reuses
# --------------------------------------------------------------------------
def test_failure_message_matches_the_prd_wording(ctx):
    panel = _panel()
    result = validate_contract(_corrupt_drop_column(panel), _contract(ctx, panel))

    message = contract_failure_message(result)

    assert message == "clean_data does not match dataset_contract.json (1 violations)"
    assert str(FlowValidationError(result, message)) == f"FLOW STOPPED: {message}"


def test_failure_message_counts_every_violation(ctx):
    panel = _panel()
    result = validate_contract(_corrupt_negative_units(_corrupt_duplicate_key(panel)),
                               _contract(ctx, panel))
    assert len(result.violations) >= 2
    assert contract_failure_message(result).endswith(f"({len(result.violations)} violations)")


# --------------------------------------------------------------------------
# file-level entry point (CLI / CI)
# --------------------------------------------------------------------------
def test_validate_contract_files_reads_both_files(ctx, tmp_path):
    panel = _panel()
    _contract(ctx, panel)
    clean_data = tmp_path / "clean_data.csv"
    panel.to_csv(clean_data, index=False, float_format="%.4f", lineterminator="\n")
    contract_path = tmp_path / paths.DATASET_CONTRACT.relative_to(paths.PROJECT_ROOT)

    result = validate_contract_files(clean_data, contract_path)

    assert result.passed, [violation.model_dump() for violation in result.violations]
    assert result.checked_rows == len(panel)


def test_validate_contract_files_keeps_stock_code_a_string(ctx, tmp_path):
    """Read back naively, '10002' becomes an integer and fails a pattern it actually matches."""
    panel = _panel()
    contract = _contract(ctx, panel)
    clean_data = tmp_path / "clean_data.csv"
    panel.to_csv(clean_data, index=False, lineterminator="\n")
    contract_path = tmp_path / paths.DATASET_CONTRACT.relative_to(paths.PROJECT_ROOT)

    result = validate_contract_files(clean_data, contract_path)

    assert "stock_code_pattern" not in {violation.rule for violation in result.violations}
    assert contract["columns"]["stock_code"]["type"] == "string"


# --------------------------------------------------------------------------
# §40 / AC 4 — nothing is typed that belongs in config
# --------------------------------------------------------------------------
def _split_tokens(path: Path) -> tuple[str, str]:
    """Return ``(code, text)``: executable tokens and string literals, comments discarded.

    Split with the tokenizer rather than a regex because the two halves need opposite checks — a
    hard-coded month would hide inside a string, a hard-coded ``k`` inside the code — and a
    docstring that merely cites "§36" must not be mistaken for either.
    """
    code, text = [], []
    with path.open(encoding="utf-8") as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.COMMENT:
                continue
            (text if token.type == tokenize.STRING else code).append(token.string)
    return " ".join(code), " ".join(text)


def test_contract_module_hard_codes_no_config_values():
    code, text = _split_tokens(paths.PROJECT_ROOT / "src" / "pipeline" / "contract.py")
    k = load_model_config().active_rule.k

    assert not re.search(r"\b20\d\d-\d\d\b", text), "a month literal belongs in cleaning_config"
    assert not re.search(rf"\b{k}\b", code), "k belongs in model_config.active_rule"
