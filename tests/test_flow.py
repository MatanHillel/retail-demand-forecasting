"""US-32 — graceful failure handling (PRD §39, §55 ``test_flow.py``).

``tests/test_flow_no_llm.py`` (US-31) already proves the Flow succeeds end to end and that one
failing step drives ``kickoff()`` into the failure handler. This file proves the §39 *contract*
around that handler in depth, for each failure category the issue names:

* exit code is never 0 on a failure (§39, PRD §42 CI exit codes);
* ``artifacts/validation_report.json`` carries this run's id, the exact ``FLOW STOPPED: …``
  wording and the right ``rule``;
* ``artifacts/run_log.json`` flips to ``status: "failed"``;
* the *previous* run's ``inventory_plan.csv``, ``latest_forecast.csv`` and ``model.joblib`` are
  byte-identical after the failure (§39 "does not produce or overwrite forecasts") — proved with
  sentinel files written before the run, not a real prior pipeline run;
* ``artifacts/_staging/`` ends up empty and ``logs/failed_runs/<run_id>/`` exists (US-32,
  ``docs/interfaces.md`` §13 interface corrections);
* the LLM crews are never invoked on the failure path (issue constraint).

Every fault case stubs out the steps that are irrelevant to the failure under test (their real
behaviour is proved by their own dedicated test files — ``test_contract.py``, ``test_leakage.py``,
etc.) so each test fails fast instead of paying for a full pipeline run. The one exception is
``test_missing_required_column_stops_gracefully``, which runs the real ``dataset_intake`` step
against a small fixture with the ``Quantity`` column removed — cheap, and it is also the fixture
named in the issue's manual acceptance-criteria run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flow import failure as flow_failure
from flow import steps as flow_steps
from flow.main import run_flow
from flow.state import FlowState
from pipeline import paths
from pipeline.contract import CONTRACT_MISMATCH_TEMPLATE
from pipeline.download import MISSING_COLUMN
from pipeline.feature_validation import LEAKAGE_FAILURE_MESSAGE
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError, ValidationResult, Violation

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"
RAW_SAMPLE_MISSING_QUANTITY = paths.FIXTURES_DIR / "raw_sample_missing_quantity.csv"

SENTINEL_INVENTORY_PLAN = b"stock_code,forecast\nSENTINEL-INVENTORY-PLAN,1\n"
SENTINEL_LATEST_FORECAST = b"stock_code,forecast\nSENTINEL-LATEST-FORECAST,1\n"
SENTINEL_MODEL = b"sentinel-model-bytes-from-the-previous-good-run"


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def _relative(path: Path) -> Path:
    return path.relative_to(paths.PROJECT_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_sentinels(base: Path) -> dict[Path, bytes]:
    """Fake a previous good run's output at the FINAL (non-staged) artifact locations."""
    sentinels = {
        base / _relative(paths.INVENTORY_PLAN): SENTINEL_INVENTORY_PLAN,
        base / _relative(paths.LATEST_FORECAST): SENTINEL_LATEST_FORECAST,
        base / _relative(paths.MODEL): SENTINEL_MODEL,
    }
    for path, content in sentinels.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return sentinels


def _assert_sentinels_untouched(sentinels: dict[Path, bytes]) -> None:
    for path, content in sentinels.items():
        assert path.is_file(), f"{path} was removed by a failed run (§39)"
        assert path.read_bytes() == content, f"{path} was overwritten by a failed run (§39)"


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure path must never reach for an LLM, no matter which step failed."""
    crews_common = pytest.importorskip("crews.common")

    def _forbidden(**kwargs):  # pragma: no cover - would fail the test
        raise AssertionError("make_llm must never be called on the failure path (§39)")

    monkeypatch.setattr(crews_common, "make_llm", _forbidden)


def _noop_step(state: FlowState, ctx: RunContext, data) -> FlowState:
    """Stand-in for a step whose real behaviour is out of scope for the failure under test."""
    return state


def _assert_graceful_failure(
    tmp_path: Path,
    state: FlowState,
    ctx: RunContext,
    *,
    expected_message: str,
    expected_rule: str,
) -> dict:
    """The §39 shape every graceful-stop test shares: report, run log, staging archived."""
    assert state.status == "failed"
    assert state.errors, "no error recorded on the failed state"
    assert state.errors[-1]["type"] == "FlowValidationError"
    assert state.errors[-1]["message"] == expected_message

    run_log = _read_json(tmp_path / _relative(paths.RUN_LOG))
    assert run_log["run_id"] == ctx.run_id
    assert run_log["status"] == "failed"

    report = _read_json(tmp_path / _relative(paths.VALIDATION_REPORT))
    assert report["run_id"] == ctx.run_id
    assert report["passed"] is False
    assert report["violations"][0]["rule"] == expected_rule

    _assert_staging_archived(tmp_path, ctx)
    return report


def _assert_staging_archived(tmp_path: Path, ctx: RunContext) -> None:
    staging_root = tmp_path / "artifacts" / "_staging"
    leftover = [p for p in staging_root.rglob("*") if p.is_file()] if staging_root.exists() else []
    assert leftover == [], "a failed run left files behind under artifacts/_staging/"

    archived = tmp_path / "logs" / "failed_runs" / ctx.run_id
    assert archived.is_dir(), "the failed run's staging tree was not archived (US-32)"


def _cli_exit_code(monkeypatch: pytest.MonkeyPatch, state: FlowState, ctx: RunContext) -> int:
    """Route this run's already-computed outcome through the real CLI exit-code mapping."""
    import flow.main
    from pipeline.__main__ import main

    monkeypatch.setattr(flow.main, "run_flow", lambda **kwargs: (state, ctx))
    return main(["--no-llm", "--sample"])


# --------------------------------------------------------------------------
# (a) raw sample without the Quantity column
# --------------------------------------------------------------------------
def test_missing_required_column_stops_gracefully(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    state, ctx = run_flow(
        mode="no-llm", raw_path=RAW_SAMPLE_MISSING_QUANTITY, skip_tuning=True, base_dir=tmp_path
    )
    try:
        report = _assert_graceful_failure(
            tmp_path,
            state,
            ctx,
            expected_message=f"FLOW STOPPED: {MISSING_COLUMN.format(column='Quantity')}",
            expected_rule="required_columns",
        )
        assert report["step"] == "raw_schema"
        _assert_sentinels_untouched(sentinels)
        assert _cli_exit_code(monkeypatch, state, ctx) == 2
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# (b) contract corrupted (e.g. primary_key changed)
# --------------------------------------------------------------------------
def test_contract_mismatch_stops_gracefully(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    def failing_contract_validation(state, ctx, data):
        violations = [
            Violation(
                step="contract_validation",
                rule="primary_key",
                message="(stock_code, month) is not unique in clean_data.csv",
                count=3,
            )
        ]
        result = ValidationResult(step="contract_validation", passed=False, violations=violations)
        with ctx.step("contract_validation"):
            raise FlowValidationError(result, CONTRACT_MISMATCH_TEMPLATE.format(count=3))

    monkeypatch.setattr(flow_steps, "data_analyst_work", _noop_step)
    monkeypatch.setattr(flow_steps, "contract_validation", failing_contract_validation)

    state, ctx = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path)
    try:
        report = _assert_graceful_failure(
            tmp_path,
            state,
            ctx,
            expected_message=f"FLOW STOPPED: {CONTRACT_MISMATCH_TEMPLATE.format(count=3)}",
            expected_rule="primary_key",
        )
        assert report["step"] == "contract_validation"
        _assert_sentinels_untouched(sentinels)
        assert _cli_exit_code(monkeypatch, state, ctx) == 2
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# (c) leaky lag_1 — feature uses information after the forecast origin
# --------------------------------------------------------------------------
def test_leaky_feature_stops_gracefully(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    def failing_feature_validation(state, ctx, data):
        step = flow_steps.FEATURE_VALIDATION_STEP
        violations = [
            Violation(step=step, rule="lag_1_leakage", message=LEAKAGE_FAILURE_MESSAGE)
        ]
        result = ValidationResult(step=step, passed=False, violations=violations)
        with ctx.step(step):
            raise FlowValidationError(result, LEAKAGE_FAILURE_MESSAGE)

    monkeypatch.setattr(flow_steps, "data_analyst_work", _noop_step)
    monkeypatch.setattr(flow_steps, "contract_validation", _noop_step)
    monkeypatch.setattr(flow_steps, "data_scientist_work", _noop_step)
    monkeypatch.setattr(flow_steps, "feature_validation", failing_feature_validation)

    state, ctx = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path)
    try:
        report = _assert_graceful_failure(
            tmp_path,
            state,
            ctx,
            expected_message=f"FLOW STOPPED: {LEAKAGE_FAILURE_MESSAGE}",
            expected_rule="lag_1_leakage",
        )
        assert report["step"] == flow_steps.FEATURE_VALIDATION_STEP
        _assert_sentinels_untouched(sentinels)
        assert _cli_exit_code(monkeypatch, state, ctx) == 2
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# (d) refit_champion writes nothing -> model.joblib was not generated
# --------------------------------------------------------------------------
def test_missing_model_artifact_stops_gracefully(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    def stage_everything_but_model(state, ctx, data):
        with ctx.step("data_analyst_work"):
            for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
                if canonical == paths.MODEL:
                    continue
                target = ctx.out(_relative(canonical))
                target.write_text("placeholder", encoding="utf-8")
        return state

    monkeypatch.setattr(flow_steps, "data_analyst_work", stage_everything_but_model)
    for name in (
        "contract_validation",
        "data_scientist_work",
        "feature_validation",
        "training_and_backtest",
        "evaluation_and_champion",
        "inventory_policy_calibration",
    ):
        monkeypatch.setattr(flow_steps, name, _noop_step)

    state, ctx = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path)
    try:
        report = _assert_graceful_failure(
            tmp_path,
            state,
            ctx,
            expected_message="FLOW STOPPED: model.joblib was not generated",
            expected_rule="required_artifact",
        )
        assert report["step"] == flow_steps.ARTIFACT_VALIDATION_STEP
        _assert_sentinels_untouched(sentinels)
        assert _cli_exit_code(monkeypatch, state, ctx) == 2
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# (e) unexpected exception (e.g. KeyError) inside a step
# --------------------------------------------------------------------------
def test_unexpected_exception_stops_gracefully_with_exit_code_one(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    def boom(state, ctx, data):
        with ctx.step("training_and_backtest"):
            raise KeyError("M2_gbm_poisson missing from candidates")

    monkeypatch.setattr(flow_steps, "data_analyst_work", _noop_step)
    monkeypatch.setattr(flow_steps, "contract_validation", _noop_step)
    monkeypatch.setattr(flow_steps, "data_scientist_work", _noop_step)
    monkeypatch.setattr(flow_steps, "feature_validation", _noop_step)
    monkeypatch.setattr(flow_steps, "training_and_backtest", boom)

    state, ctx = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path)
    try:
        assert state.status == "failed"
        assert state.errors[-1]["type"] == "KeyError"

        run_log = _read_json(tmp_path / _relative(paths.RUN_LOG))
        assert run_log["status"] == "failed"

        report = _read_json(tmp_path / _relative(paths.VALIDATION_REPORT))
        assert report["run_id"] == ctx.run_id
        assert report["passed"] is False
        assert report["step"] == "training_and_backtest"
        assert report["violations"][0]["rule"] == flow_failure.UNEXPECTED_EXCEPTION_RULE

        _assert_staging_archived(tmp_path, ctx)
        _assert_sentinels_untouched(sentinels)
        assert _cli_exit_code(monkeypatch, state, ctx) == 1
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# --no-keep-failed discards the staging tree instead of archiving it
# --------------------------------------------------------------------------
def test_no_keep_failed_discards_staging_instead_of_archiving(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)

    def failing_intake(state, ctx, data):
        result = ValidationResult(
            step="raw_schema",
            passed=False,
            violations=[
                Violation(
                    step="raw_schema",
                    rule="required_columns",
                    message=MISSING_COLUMN.format(column="Quantity"),
                )
            ],
        )
        with ctx.step("dataset_intake"):
            raise FlowValidationError(result)

    monkeypatch.setattr(flow_steps, "dataset_intake", failing_intake)

    state, ctx = run_flow(
        mode="no-llm",
        raw_path=RAW_SAMPLE,
        skip_tuning=True,
        base_dir=tmp_path,
        keep_failed=False,
    )
    try:
        assert state.status == "failed"
        assert not (tmp_path / "artifacts" / "_staging" / ctx.run_id).exists()
        assert not (tmp_path / "logs" / "failed_runs" / ctx.run_id).exists()
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# a successful run after a failed run promotes and overwrites normally
# --------------------------------------------------------------------------
def test_successful_run_after_a_failed_run_promotes_and_overwrites(tmp_path, monkeypatch) -> None:
    _forbid_llm(monkeypatch)
    sentinels = _write_sentinels(tmp_path)

    def failing_intake(state, ctx, data):
        result = ValidationResult(
            step="raw_schema",
            passed=False,
            violations=[
                Violation(
                    step="raw_schema",
                    rule="required_columns",
                    message=MISSING_COLUMN.format(column="Quantity"),
                )
            ],
        )
        with ctx.step("dataset_intake"):
            raise FlowValidationError(result)

    monkeypatch.setattr(flow_steps, "dataset_intake", failing_intake)
    failed_state, failed_ctx = run_flow(
        mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path
    )
    try:
        assert failed_state.status == "failed"
        _assert_sentinels_untouched(sentinels)
    finally:
        close_log_handlers(failed_ctx.run_id)
    monkeypatch.undo()

    def stage_all_required_artifacts(state, ctx, data):
        with ctx.step("data_analyst_work"):
            for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
                target = ctx.out(_relative(canonical))
                target.write_text("placeholder-from-the-successful-run", encoding="utf-8")
        return state

    for name in (
        "contract_validation",
        "data_scientist_work",
        "feature_validation",
        "training_and_backtest",
        "evaluation_and_champion",
        "inventory_policy_calibration",
    ):
        monkeypatch.setattr(flow_steps, name, _noop_step)
    monkeypatch.setattr(flow_steps, "data_analyst_work", stage_all_required_artifacts)

    good_state, good_ctx = run_flow(
        mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path
    )
    try:
        assert good_state.status == "success"

        run_log = _read_json(tmp_path / _relative(paths.RUN_LOG))
        assert run_log["run_id"] == good_ctx.run_id
        assert run_log["status"] == "success"

        for path in sentinels:
            assert path.read_bytes() != sentinels[path], f"{path} was not overwritten on success"
            assert path.read_text(encoding="utf-8") == "placeholder-from-the-successful-run"

        staging_root = tmp_path / "artifacts" / "_staging"
        leftover = (
            [p for p in staging_root.rglob("*") if p.is_file()] if staging_root.exists() else []
        )
        assert leftover == []
    finally:
        close_log_handlers(good_ctx.run_id)


# --------------------------------------------------------------------------
# flow.failure unit tests — the synthesised-result and archive-vs-discard branches directly
# --------------------------------------------------------------------------
def test_handle_failure_synthesizes_a_result_for_an_unexpected_exception(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        state = FlowState(run_id=ctx.run_id, current_step="training_and_backtest")
        archived = flow_failure.handle_failure(state, ctx, KeyError("boom"))

        assert state.status == "failed"
        assert ctx.status == "failed"

        report = _read_json(tmp_path / _relative(paths.VALIDATION_REPORT))
        assert report["step"] == "training_and_backtest"
        assert report["violations"][0]["rule"] == flow_failure.UNEXPECTED_EXCEPTION_RULE
        assert report["violations"][0]["message"] == "'boom'"  # str(KeyError("boom"))

        assert archived == tmp_path / "logs" / "failed_runs" / ctx.run_id
        assert archived.is_dir()
    finally:
        close_log_handlers(ctx.run_id)


def test_handle_failure_reuses_the_validation_result_for_a_graceful_stop(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        state = FlowState(run_id=ctx.run_id, current_step="contract_validation")
        result = ValidationResult(
            step="contract_validation",
            passed=False,
            violations=[
                Violation(step="contract_validation", rule="primary_key", message="dup keys")
            ],
        )
        error = FlowValidationError(result)
        flow_failure.handle_failure(state, ctx, error)

        report = _read_json(tmp_path / _relative(paths.VALIDATION_REPORT))
        assert report["violations"][0]["rule"] == "primary_key"
        assert report["violations"][0]["message"] == "dup keys"
    finally:
        close_log_handlers(ctx.run_id)


def test_handle_failure_discards_staging_when_keep_failed_is_false(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        staged = ctx.out(_relative(paths.CLEAN_DATA))
        staged.write_text("x", encoding="utf-8")

        state = FlowState(run_id=ctx.run_id, current_step="build_panel")
        archived = flow_failure.handle_failure(state, ctx, RuntimeError("boom"), keep_failed=False)

        assert archived is None
        assert not ctx.staging_dir.exists()
        assert not (tmp_path / "logs" / "failed_runs" / ctx.run_id).exists()
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# CLI: --keep-failed / --no-keep-failed reaches run_flow()
# --------------------------------------------------------------------------
def test_cli_threads_keep_failed_flag_through_to_run_flow(monkeypatch) -> None:
    import flow.main
    from pipeline.__main__ import main

    captured: dict = {}

    def fake_run_flow(**kwargs):
        captured.update(kwargs)
        return FlowState(status="success"), None

    monkeypatch.setattr(flow.main, "run_flow", fake_run_flow)
    assert main(["--no-llm", "--sample", "--skip-tuning"]) == 0
    assert captured["keep_failed"] is True

    captured.clear()
    assert main(["--no-llm", "--sample", "--skip-tuning", "--no-keep-failed"]) == 0
    assert captured["keep_failed"] is False
