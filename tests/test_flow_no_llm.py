"""US-31 — the CrewAI Flow in ``--no-llm`` mode (PRD §37, §39, §41).

What this file proves:

* the Flow runs end to end on the CI sample fixture: ``status == "success"``, every validation
  flag ``True``, a champion chosen, all required §41 artifacts promoted and recorded, and the
  staging area left without files;
* no LLM is ever constructed in ``--no-llm`` mode (``crews.common.make_llm`` is monkeypatched to
  raise, and the run still succeeds);
* the routers return ``"fail"`` on a failed state, and a failing step drives ``kickoff()`` into
  the failure handler: ``run_log.json`` flips to ``failed``, ``validation_report.json`` carries
  this run's id and the ``FLOW STOPPED: `` message, and nothing is promoted;
* step 9 checks the **staged** paths: a good ``model.joblib`` already sitting at the final
  location must not mask a run that produced none (issue §8 — the false green §39 exists to
  prevent);
* the CLI maps outcomes to exit codes 0 / 2 and keeps CrewAI out of ``pipeline.*`` module scope.

The end-to-end fixture is module-scoped (one full pipeline run, ~minutes on the sample) and runs
with ``skip_tuning=True`` — tuning would rewrite ``config/model_config.yaml`` in place and blow
the CI time budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flow import steps as flow_steps
from flow.main import (
    ARTIFACTS_OK,
    CONTRACT_OK,
    FAIL,
    FEATURES_OK,
    INTAKE_OK,
    RetailForecastFlow,
    run_flow,
)
from flow.state import FlowState
from pipeline import paths
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError, ValidationResult, Violation

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: Basenames of every file step 9 requires (§41 plus the three forecast files).
REQUIRED_NAMES = [path.name for path in flow_steps.REQUIRED_FLOW_ARTIFACTS]


def _relative(path: Path) -> Path:
    return path.relative_to(paths.PROJECT_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# end-to-end run on the sample fixture (module-scoped: one pipeline run)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flow_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("flow_no_llm")

    def _no_llm_allowed(**kwargs):  # pragma: no cover - would fail the run
        raise AssertionError("make_llm must never be called in --no-llm mode (§37)")

    mp = pytest.MonkeyPatch()
    crews_common = pytest.importorskip("crews.common")
    mp.setattr(crews_common, "make_llm", _no_llm_allowed)

    state, ctx = run_flow(
        mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=base
    )
    yield SimpleNamespace(state=state, ctx=ctx, base=base)
    mp.undo()
    close_log_handlers(ctx.run_id)


def test_flow_succeeds_end_to_end(flow_run) -> None:
    state = flow_run.state
    assert state.status == "success"
    assert state.run_id == flow_run.ctx.run_id
    assert state.current_step == "publish"


def test_all_validation_flags_true(flow_run) -> None:
    flags = flow_run.state.validation
    assert flags.raw_schema is True
    assert flags.contract is True
    assert flags.features is True
    assert flags.leakage is True
    assert flags.artifacts is True


def test_champion_selected(flow_run) -> None:
    champion = flow_run.state.champion
    assert champion is not None
    assert isinstance(champion.get("champion"), str) and champion["champion"]


def test_artifact_paths_cover_required_names(flow_run) -> None:
    recorded = flow_run.state.artifact_paths
    assert sorted(recorded) == sorted(REQUIRED_NAMES)
    for name, relative in recorded.items():
        promoted = flow_run.base / Path(relative)
        assert promoted.is_file(), f"{name} was not promoted to {relative}"
        assert promoted.stat().st_size > 0, f"{name} was promoted empty"


def test_run_log_reports_success_with_champion_and_artifacts(flow_run) -> None:
    run_log = _read_json(flow_run.base / _relative(paths.RUN_LOG))
    assert run_log["run_id"] == flow_run.ctx.run_id
    assert run_log["status"] == "success"
    assert run_log["champion"] is not None
    for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
        assert _relative(canonical).as_posix() in run_log["artifacts"].values()


def test_staging_contains_no_files_after_publish(flow_run) -> None:
    staging_root = flow_run.base / "artifacts" / "_staging"
    leftover = [p for p in staging_root.rglob("*") if p.is_file()] if staging_root.exists() else []
    assert leftover == []


def test_state_is_json_serialisable(flow_run) -> None:
    payload = flow_run.state.model_dump(mode="json")
    assert json.loads(json.dumps(payload)) == payload


# --------------------------------------------------------------------------
# routers return "fail" on a failed state (unit — no pipeline work)
# --------------------------------------------------------------------------
def _make_flow(tmp_path: Path) -> tuple[RetailForecastFlow, RunContext]:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    return RetailForecastFlow(ctx, raw_path=RAW_SAMPLE, skip_tuning=True), ctx


def test_routers_route_failed_state_to_fail(tmp_path) -> None:
    flow, ctx = _make_flow(tmp_path)
    try:
        flow.state.status = "failed"
        assert flow.route_after_intake() == FAIL
        assert flow.route_after_contract() == FAIL
        assert flow.route_after_features() == FAIL
        assert flow.route_after_artifacts() == FAIL
    finally:
        close_log_handlers(ctx.run_id)


def test_routers_route_running_state_to_continue(tmp_path) -> None:
    flow, ctx = _make_flow(tmp_path)
    try:
        assert flow.route_after_intake() == INTAKE_OK
        assert flow.route_after_contract() == CONTRACT_OK
        assert flow.route_after_features() == FEATURES_OK
        assert flow.route_after_artifacts() == ARTIFACTS_OK
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# graceful failure path (§39): failed run log + stamped validation report, no promote
# --------------------------------------------------------------------------
def test_failed_step_drives_kickoff_into_failure_handler(tmp_path, monkeypatch) -> None:
    def failing_intake(state, ctx, data):
        result = ValidationResult(
            step="dataset_intake",
            passed=False,
            violations=[
                Violation(
                    step="dataset_intake",
                    rule="required_columns",
                    message="Missing required column Quantity",
                )
            ],
        )
        with ctx.step("dataset_intake"):
            raise FlowValidationError(result)

    monkeypatch.setattr(flow_steps, "dataset_intake", failing_intake)

    state, ctx = run_flow(
        mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path
    )
    try:
        assert state.status == "failed"
        assert state.errors[-1]["type"] == "FlowValidationError"
        assert state.errors[-1]["message"] == "FLOW STOPPED: Missing required column Quantity"

        run_log = _read_json(tmp_path / _relative(paths.RUN_LOG))
        assert run_log["status"] == "failed"

        report = _read_json(tmp_path / _relative(paths.VALIDATION_REPORT))
        assert report["run_id"] == ctx.run_id
        assert report["passed"] is False
        assert report["step"] == "dataset_intake"
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# step 9 checks STAGED paths — a stale final model.joblib must not mask a missing one
# --------------------------------------------------------------------------
def test_artifact_validation_ignores_stale_final_artifacts(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        # A previous "good" run's champion sits at the FINAL location…
        stale_model = tmp_path / _relative(paths.MODEL)
        stale_model.parent.mkdir(parents=True, exist_ok=True)
        stale_model.write_bytes(b"previous run's champion")
        # …while THIS run staged every required artifact except model.joblib.
        for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
            if canonical == paths.MODEL:
                continue
            staged = ctx.staging_dir / _relative(canonical)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text("content", encoding="utf-8")

        state = FlowState(run_id=ctx.run_id)
        with pytest.raises(FlowValidationError) as excinfo:
            flow_steps.artifact_validation(state, ctx, flow_steps.FlowData())

        assert str(excinfo.value) == "FLOW STOPPED: model.joblib was not generated"
        assert state.validation.artifacts is False
    finally:
        close_log_handlers(ctx.run_id)


def test_artifact_validation_passes_when_everything_is_staged(tmp_path) -> None:
    ctx = RunContext.start(mode="no-llm", staging=True, base_dir=tmp_path)
    try:
        for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
            staged = ctx.staging_dir / _relative(canonical)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text("content", encoding="utf-8")

        state = FlowState(run_id=ctx.run_id)
        flow_steps.artifact_validation(state, ctx, flow_steps.FlowData())

        assert state.validation.artifacts is True
        assert sorted(state.artifact_paths) == sorted(REQUIRED_NAMES)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# CLI: exit codes and CrewAI isolation
# --------------------------------------------------------------------------
def test_cli_exit_code_zero_on_success(monkeypatch, capsys) -> None:
    import flow.main
    from pipeline.__main__ import main

    def fake_run_flow(**kwargs):
        return FlowState(status="success"), None

    monkeypatch.setattr(flow.main, "run_flow", fake_run_flow)
    assert main(["--no-llm", "--sample", "--skip-tuning"]) == 0


def test_cli_exit_code_two_on_graceful_stop(monkeypatch, capsys) -> None:
    import flow.main
    from pipeline.__main__ import main

    def fake_run_flow(**kwargs):
        state = FlowState(status="failed")
        state.errors = [
            {
                "step": "contract_validation",
                "type": "FlowValidationError",
                "message": "FLOW STOPPED: clean_data does not match dataset_contract.json",
            }
        ]
        return state, None

    monkeypatch.setattr(flow.main, "run_flow", fake_run_flow)
    assert main(["--no-llm", "--sample"]) == 2
    assert "FLOW STOPPED:" in capsys.readouterr().err


def test_cli_exit_code_one_on_unexpected_failure(monkeypatch, capsys) -> None:
    import flow.main
    from pipeline.__main__ import main

    def fake_run_flow(**kwargs):
        state = FlowState(status="failed")
        state.errors = [{"step": "backtest", "type": "MemoryError", "message": "boom"}]
        return state, None

    monkeypatch.setattr(flow.main, "run_flow", fake_run_flow)
    assert main(["--no-llm", "--sample"]) == 1


def test_flow_steps_module_is_llm_free() -> None:
    """The AC grep: no LLM class and no ``make_llm`` anywhere in the deterministic steps."""
    source = (paths.PROJECT_ROOT / "src" / "flow" / "steps.py").read_text(encoding="utf-8")
    assert "make_llm" not in source
    assert "LLM(" not in source
    assert "import crewai" not in source
    assert "from crewai" not in source


def test_pipeline_package_import_does_not_pull_crewai() -> None:
    """``src/pipeline/`` stays CrewAI-free at module scope (docs/interfaces.md §6 rule 10)."""
    import subprocess
    import sys

    code = (
        "import sys; import pipeline.__main__; "
        "sys.exit(1 if any(m.startswith('crewai') for m in sys.modules) else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0
