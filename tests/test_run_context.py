"""Run-context, logging, staging and validation-report tests (US-02, PRD §37, §39, §40).

Every test runs against ``tmp_path`` via ``base_dir``, so the suite never writes into the real
``artifacts/`` or ``logs/`` directories.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from pipeline import paths
from pipeline.config import load_model_config
from pipeline.run_context import (
    RUN_ID_PATTERN,
    RunContext,
    close_log_handlers,
    new_run_id,
    redact,
    set_global_seed,
)
from pipeline.validation import (
    FLOW_STOPPED_PREFIX,
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

REQUIRED_RUN_LOG_KEYS = {
    "run_id",
    "started_at",
    "mode",
    "status",
    "seed",
    "data",
    "config_snapshot",
    "versions",
    "steps",
    "warnings",
    "metrics",
    "champion",
    "errors",
    "artifacts",
}
REQUIRED_VERSION_KEYS = {"python", "pandas", "numpy", "sklearn"}


@pytest.fixture
def ctx(tmp_path):
    """A run context isolated in a temporary directory; its log handler is closed afterwards."""
    context = RunContext.start(mode="no-llm", base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


@pytest.fixture
def staged_ctx(tmp_path):
    context = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    yield context
    close_log_handlers(context.run_id)


def read_run_log(context: RunContext) -> dict:
    return json.loads((context.base_dir / "artifacts" / "run_log.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# run id
# --------------------------------------------------------------------------
def test_run_id_format_and_uniqueness() -> None:
    first, second = new_run_id(), new_run_id()
    assert RUN_ID_PATTERN.match(first), first
    assert RUN_ID_PATTERN.match(second)
    assert first != second


def test_context_run_id_matches_the_format(ctx: RunContext) -> None:
    assert RUN_ID_PATTERN.match(ctx.run_id)


# --------------------------------------------------------------------------
# run_log.json schema (PRD §40)
# --------------------------------------------------------------------------
def test_write_run_log_contains_every_required_key(ctx: RunContext) -> None:
    ctx.write_run_log()
    log = read_run_log(ctx)
    assert REQUIRED_RUN_LOG_KEYS <= set(log)
    assert REQUIRED_VERSION_KEYS <= set(log["versions"])
    assert log["versions"]["python"].startswith("3.11")
    assert log["mode"] == "no-llm"
    assert set(log["data"]) == {"file", "sha256", "rows", "columns"}


def test_run_log_records_the_configuration_snapshot(ctx: RunContext) -> None:
    ctx.write_run_log()
    log = read_run_log(ctx)
    assert "model_config" in log["config_snapshot"]
    assert log["seed"] == load_model_config().seed


def test_run_log_is_archived_next_to_the_text_log(ctx: RunContext) -> None:
    ctx.write_run_log()
    archive = ctx.base_dir / "logs" / f"run_{ctx.run_id}.json"
    assert archive.is_file()
    assert json.loads(archive.read_text(encoding="utf-8"))["run_id"] == ctx.run_id


def test_context_is_json_serialisable_at_any_moment(ctx: RunContext) -> None:
    with ctx.step("mid-flight"):
        json.dumps(ctx.model_dump(mode="json"))


def test_finish_stamps_status_and_finished_at(ctx: RunContext) -> None:
    ctx.finish()
    log = read_run_log(ctx)
    assert log["status"] == "success"
    assert log["finished_at"] is not None


# --------------------------------------------------------------------------
# steps, row counts, warnings, metrics, artifacts
# --------------------------------------------------------------------------
def test_step_records_duration_and_success(ctx: RunContext) -> None:
    with ctx.step("cleaning", inputs={"file": "raw.csv"}) as step:
        step.outputs["rows"] = 10
    record = ctx.steps[0]
    assert record.name == "cleaning"
    assert record.status == "success"
    assert record.duration_s is not None and record.duration_s >= 0
    assert record.inputs == {"file": "raw.csv"}
    assert record.outputs == {"rows": 10}


def test_log_rows_records_a_waterfall_line(ctx: RunContext) -> None:
    with ctx.step("cleaning"):
        ctx.log_rows("drop_zero_price", before=100, removed=4, after=96)
    assert ctx.steps[0].row_counts["drop_zero_price"] == {
        "before": 100,
        "removed": 4,
        "after": 96,
    }


def test_log_rows_outside_a_step_is_a_programming_error(ctx: RunContext) -> None:
    with pytest.raises(RuntimeError, match="inside a ctx.step"):
        ctx.log_rows("orphan", before=1, removed=0, after=1)


def test_warn_lands_on_both_the_step_and_the_run(ctx: RunContext) -> None:
    with ctx.step("cleaning"):
        ctx.warn("duplicate share above the configured threshold")
    assert ctx.warnings == ["duplicate share above the configured threshold"]
    assert ctx.steps[0].warnings == ctx.warnings


def test_record_metrics_artifacts_and_data(ctx: RunContext) -> None:
    ctx.record_metrics({"wmape": 0.5})
    ctx.record_metrics({"bias": -0.01})
    ctx.record_artifact("clean_data", ctx.base_dir / "data" / "processed" / "clean_data.csv")
    ctx.record_data(file="online_retail_II.xlsx", sha256="a" * 64, rows=7, columns=8)
    ctx.write_run_log()
    log = read_run_log(ctx)
    assert log["metrics"] == {"wmape": 0.5, "bias": -0.01}
    assert log["artifacts"]["clean_data"] == "data/processed/clean_data.csv"
    assert log["data"]["rows"] == 7


# --------------------------------------------------------------------------
# artifact checksums (US-34, PRD §40)
# --------------------------------------------------------------------------
def test_record_artifact_checksums_fingerprints_every_recorded_artifact(ctx: RunContext) -> None:
    import hashlib

    final = ctx.base_dir / "data" / "processed" / "clean_data.csv"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("stock_code,month\nA,2011-01\n", encoding="utf-8")
    ctx.record_artifact("clean_data", final)

    ctx.record_artifact_checksums()

    entry = ctx.artifact_checksums["clean_data"]
    assert entry["path"] == "data/processed/clean_data.csv"
    assert entry["bytes"] == final.stat().st_size
    assert entry["sha256"] == hashlib.sha256(final.read_bytes()).hexdigest()


def test_record_artifact_checksums_skips_a_missing_file(ctx: RunContext) -> None:
    ctx.record_artifact("never_written", ctx.base_dir / "artifacts" / "forecasts" / "ghost.csv")
    ctx.record_artifact_checksums()
    assert "never_written" not in ctx.artifact_checksums


def test_artifact_checksums_reach_run_log_json(ctx: RunContext) -> None:
    final = ctx.base_dir / "artifacts" / "forecasts" / "inventory_kpis.csv"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("stock_code,kpi\nA,1\n", encoding="utf-8")
    ctx.record_artifact("inventory_kpis", final)
    ctx.record_artifact_checksums()
    ctx.write_run_log()

    log = read_run_log(ctx)
    checksum = log["artifact_checksums"]["inventory_kpis"]
    assert checksum["path"] == "artifacts/forecasts/inventory_kpis.csv"


# --------------------------------------------------------------------------
# failure path (PRD §39)
# --------------------------------------------------------------------------
def test_failed_step_marks_the_run_failed_and_re_raises(ctx: RunContext) -> None:
    with pytest.raises(ValueError, match="boom"), ctx.step("modelling"):
        raise ValueError("boom")

    assert ctx.status == "failed"
    assert ctx.steps[0].status == "failed"
    assert len(ctx.errors) == 1
    assert ctx.errors[0]["step"] == "modelling"
    assert ctx.errors[0]["type"] == "ValueError"
    assert "boom" in ctx.errors[0]["traceback"]


def test_run_log_is_valid_json_after_a_failure(ctx: RunContext) -> None:
    with pytest.raises(ValueError), ctx.step("modelling"):
        raise ValueError("boom")
    ctx.finish()

    log = read_run_log(ctx)
    assert log["status"] == "failed"
    assert log["errors"][0]["type"] == "ValueError"


def test_finish_never_upgrades_a_failed_run_to_success(ctx: RunContext) -> None:
    with pytest.raises(ValueError), ctx.step("modelling"):
        raise ValueError("boom")
    ctx.finish(status="success")
    assert read_run_log(ctx)["status"] == "failed"


# --------------------------------------------------------------------------
# staging and promotion (PRD §39)
# --------------------------------------------------------------------------
def test_out_returns_the_path_unchanged_without_staging(ctx: RunContext) -> None:
    final = ctx.base_dir / "artifacts" / "forecasts" / "latest_forecast.csv"
    assert ctx.out(final) == final
    assert final.parent.is_dir()


def test_out_redirects_into_the_staging_directory(staged_ctx: RunContext) -> None:
    final = staged_ctx.base_dir / "artifacts" / "forecasts" / "latest_forecast.csv"
    staged = staged_ctx.out(final)
    assert staged != final
    assert staged.parent.is_dir()
    assert staged.relative_to(staged_ctx.staging_dir).as_posix() == (
        "artifacts/forecasts/latest_forecast.csv"
    )


def test_promote_moves_staged_files_to_their_final_locations(staged_ctx: RunContext) -> None:
    final = staged_ctx.base_dir / "artifacts" / "forecasts" / "latest_forecast.csv"
    staged = staged_ctx.out(final)
    staged.write_text("stock_code,forecast\n", encoding="utf-8")
    assert not final.exists()

    promoted = staged_ctx.promote()

    assert promoted == [final]
    assert final.read_text(encoding="utf-8") == "stock_code,forecast\n"
    assert not staged.exists()
    assert not list(staged_ctx.staging_dir.rglob("*.csv"))
    assert not list(final.parent.glob("*.tmp-*"))


def test_promote_replaces_a_previous_artifact(staged_ctx: RunContext) -> None:
    final = staged_ctx.base_dir / "artifacts" / "forecasts" / "latest_forecast.csv"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("previous good run\n", encoding="utf-8")

    staged = staged_ctx.out(final)
    staged.write_text("new run\n", encoding="utf-8")
    staged_ctx.promote()

    assert final.read_text(encoding="utf-8") == "new run\n"


def test_a_failed_run_promotes_nothing_and_leaves_good_output_untouched(
    staged_ctx: RunContext,
) -> None:
    final = staged_ctx.base_dir / "artifacts" / "forecasts" / "latest_forecast.csv"
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("previous good run\n", encoding="utf-8")

    staged = staged_ctx.out(final)
    staged.write_text("half-finished run\n", encoding="utf-8")

    with pytest.raises(ValueError), staged_ctx.step("modelling"):
        raise ValueError("boom")

    with pytest.raises(RuntimeError, match="refusing to promote"):
        staged_ctx.promote()

    # PRD §39: the previous forecast survives a failed run untouched.
    assert final.read_text(encoding="utf-8") == "previous good run\n"
    staged_ctx.finish()
    assert read_run_log(staged_ctx)["status"] == "failed"


def test_promote_warns_about_a_staged_path_that_was_never_written(staged_ctx: RunContext) -> None:
    staged_ctx.out(staged_ctx.base_dir / "artifacts" / "forecasts" / "inventory_plan.csv")
    assert staged_ctx.promote() == []
    assert any("never written" in warning for warning in staged_ctx.warnings)


def test_out_refuses_a_path_outside_the_project(staged_ctx: RunContext, tmp_path) -> None:
    with pytest.raises(ValueError, match="must live under"):
        staged_ctx.out(tmp_path.parent / "elsewhere.csv")


def test_discard_staging_removes_the_run_directory(staged_ctx: RunContext) -> None:
    staged = staged_ctx.out(staged_ctx.base_dir / "artifacts" / "forecasts" / "sigma_table.csv")
    staged.write_text("x\n", encoding="utf-8")
    staged_ctx.discard_staging()
    assert not staged_ctx.staging_dir.exists()


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
def read_log_lines(context: RunContext) -> list[dict]:
    log_file = context.base_dir / "logs" / f"run_{context.run_id}.log"
    for handler in logging.getLogger("pipeline").handlers:
        handler.flush()
    return [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]


def test_log_file_is_json_lines_attributed_to_the_step(ctx: RunContext) -> None:
    with ctx.step("cleaning"):
        ctx.log_rows("drop_zero_price", before=10, removed=1, after=9)

    entries = read_log_lines(ctx)
    assert entries, "the run log file should not be empty"
    assert all({"time", "level", "step", "message"} <= set(entry) for entry in entries)
    assert any(entry["step"] == "cleaning" for entry in entries)


def test_get_logger_does_not_duplicate_handlers(ctx: RunContext) -> None:
    before = len(logging.getLogger("pipeline").handlers)
    ctx.logger.info("one")
    ctx.logger.info("two")
    assert len(logging.getLogger("pipeline").handlers) == before
    assert sum(entry["message"] in {"one", "two"} for entry in read_log_lines(ctx)) == 2


def test_secrets_never_reach_the_log(ctx: RunContext, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value-not-for-logs")
    ctx.logger.info("calling the API with sk-ant-abcdefghijklmnopqrstuvwxyz012345")
    ctx.warn("key is super-secret-value-not-for-logs")

    text = (ctx.base_dir / "logs" / f"run_{ctx.run_id}.log").read_text(encoding="utf-8")
    assert "sk-ant-abcdefghijklmnopqrstuvwxyz012345" not in text
    assert "super-secret-value-not-for-logs" not in text
    assert "***REDACTED***" in text
    assert "super-secret-value-not-for-logs" not in " ".join(ctx.warnings)


def test_redact_leaves_ordinary_text_alone() -> None:
    assert redact("wMAPE computed for 3 products") == "wMAPE computed for 3 products"


# --------------------------------------------------------------------------
# seeding (PRD §40)
# --------------------------------------------------------------------------
def test_set_global_seed_defaults_to_the_configured_seed() -> None:
    assert set_global_seed() == load_model_config().seed


def test_set_global_seed_makes_draws_repeatable() -> None:
    set_global_seed()
    first = np.random.rand(5).tolist()
    set_global_seed()
    assert np.random.rand(5).tolist() == first


def test_seed_is_not_written_into_the_code() -> None:
    """PRD §40 / CLAUDE.md §2.4: the seed lives in model_config.yaml, never as a literal."""
    source = (paths.PROJECT_ROOT / "src" / "pipeline" / "run_context.py").read_text(
        encoding="utf-8"
    )
    assert str(load_model_config().seed) not in source


# --------------------------------------------------------------------------
# validation report (PRD §39)
# --------------------------------------------------------------------------
def test_write_validation_report_has_the_required_keys(tmp_path) -> None:
    result = ValidationResult(
        step="contract_validation",
        passed=False,
        checked_rows=100,
        violations=[
            Violation(
                step="contract_validation",
                rule="required_column",
                message="Missing required column Quantity",
                count=1,
                examples=["Quantity"],
            )
        ],
    )
    path = write_validation_report(result, path=tmp_path / "validation_report.json", run_id="r-1")
    report = json.loads(path.read_text(encoding="utf-8"))

    assert {"run_id", "timestamp", "step", "passed", "checked_rows", "violations", "extra"} <= set(
        report
    )
    assert report["run_id"] == "r-1"
    assert report["passed"] is False
    assert set(report["violations"][0]) == {"step", "rule", "message", "count", "examples"}


def test_validation_report_is_written_for_a_passing_step_too(tmp_path) -> None:
    result = ValidationResult(step="feature_validation", passed=True, checked_rows=10)
    path = write_validation_report(
        result, path=tmp_path / "validation_report.json", run_id="r-2"
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["run_id"] == "r-2"


def test_validation_report_cannot_be_written_without_a_run_id(tmp_path) -> None:
    """The run id is mandatory by signature, not by discipline.

    The report is never cleared between runs, so a reader must match its run id against
    ``run_log.json``. A report with no id can never match, and a failed run would show the user a
    banner with no reason. Making the argument keyword-only and required turns that mistake from
    invisible into impossible.
    """
    result = ValidationResult(step="feature_validation", passed=True, checked_rows=10)
    with pytest.raises(TypeError):
        write_validation_report(result, path=tmp_path / "validation_report.json")


def test_flow_validation_error_message_starts_with_flow_stopped() -> None:
    result = ValidationResult(
        step="contract_validation",
        passed=False,
        violations=[
            Violation(
                step="contract_validation",
                rule="required_column",
                message="Missing required column Quantity",
            )
        ],
    )
    error = FlowValidationError(result)
    assert str(error).startswith(FLOW_STOPPED_PREFIX)
    assert str(error) == "FLOW STOPPED: Missing required column Quantity"
    assert error.result is result


def test_flow_validation_error_summarises_multiple_violations() -> None:
    violations = [
        Violation(step="contract_validation", rule=f"rule_{index}", message=f"problem {index}")
        for index in range(3)
    ]
    result = ValidationResult(step="contract_validation", passed=False, violations=violations)
    assert str(FlowValidationError(result)) == (
        "FLOW STOPPED: contract_validation failed with 3 violations"
    )


def test_flow_validation_error_accepts_an_explicit_message() -> None:
    result = ValidationResult(step="artifact_validation", passed=False)
    error = FlowValidationError(result, "model.joblib was not generated")
    assert str(error) == "FLOW STOPPED: model.joblib was not generated"
