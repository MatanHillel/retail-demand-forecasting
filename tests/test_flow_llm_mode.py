"""US-33 — LLM mode: where the crews are kicked off, and what they are not allowed to do.

No test here makes a network call. The two functions in :mod:`flow.llm_mode` that construct a
crew — ``run_analyst_crew`` and ``run_scientist_crew`` — are the seams the whole LLM stack sits
behind, and every test replaces them with a stub that records what happened. That is the honest
way to test *wiring*: the crews themselves are proved by ``test_crew_data_analyst.py`` and
``test_crew_data_scientist.py``.

What this file proves:

* **Order.** In LLM mode crew 1 runs after step 3 (``contract_validation``) and before step 4;
  crew 2 runs after step 9 (``artifact_validation``) and before step 10 (``publish``).
* **Gating.** A router that returned ``"fail"`` reaches neither crew, and no LLM call is made —
  ``metrics`` carries no ``llm`` block and no cost. ``--no-llm`` reaches neither crew either.
* **The determinism guard.** A crew that rewrites ``features.csv`` has it restored byte for byte
  and the run records ``crew modified numeric artifact … — restored`` (§38).
* **The narrative guard.** An ``insights.md`` rewrite containing a number no table backs is
  discarded and the deterministic text restored, into *this* run's staged destination.
* **The cost cap.** Reaching ``llm.max_cost_usd`` aborts the narrative step and nothing else: the
  run still finishes ``success``, with ``narrative_accepted`` false.
* **Completeness after the rewrite.** Step 9 ran before the narrative; a crew that truncates
  ``evaluation_report.md`` is caught by the re-check and stops the run gracefully.
* **`run_log.json`** carries ``metrics.llm`` with statuses, tokens, cost and the accepted flags,
  and no credential value reaches any artifact or log.
* **The CLI mode switch**: no key falls back to ``--no-llm`` and exits 0; ``--llm`` without a key
  exits 2 before a run context is ever started.

The deterministic steps are stubbed out throughout — their real behaviour is proved end to end by
``test_flow_no_llm.py``, and a full pipeline run per assertion would cost minutes for wiring that
is decided in the first millisecond.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from flow import llm_mode as flow_llm_mode
from flow import main as flow_main
from flow import steps as flow_steps
from flow.main import run_flow
from pipeline import paths
from pipeline.config import load_model_config
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError, ValidationResult, Violation

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: The ten deterministic step functions, in execution order — every one of them is stubbed.
DETERMINISTIC_STEPS: tuple[str, ...] = (
    "dataset_intake",
    "data_analyst_work",
    "contract_validation",
    "data_scientist_work",
    "feature_validation",
    "training_and_backtest",
    "evaluation_and_champion",
    "inventory_policy_calibration",
    "artifact_validation",
)

#: Placeholder content for a staged artifact — enough to be "present and non-empty".
STAGED_CONTENT = b"deterministic-artifact-bytes\n"


def _relative(path: Path) -> Path:
    return path.relative_to(paths.PROJECT_ROOT)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_required_artifacts(ctx: RunContext) -> None:
    """Write every artifact step 9 requires into this run's staging tree.

    Registered through ``ctx.out(...)`` exactly as a real step would, so ``publish`` promotes them
    and the §39 staging contract is exercised rather than bypassed.
    """
    for canonical in flow_steps.REQUIRED_FLOW_ARTIFACTS:
        ctx.out(_relative(canonical)).write_bytes(STAGED_CONTENT)


class _Harness:
    """One stubbed LLM-mode run: which steps ran, in which order, and what the crews were told."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail_at: str | None = None) -> None:
        self.calls: list[str] = []
        self.fail_at = fail_at
        self.analyst_summary: dict = {"insights_narrative_accepted": True}
        self.scientist_summary: dict = {
            "evaluation_report_accepted": True,
            "model_card_accepted": True,
        }
        #: Extra work a crew stub performs, e.g. tampering with an artifact.
        self.analyst_action = None
        self.scientist_action = None
        self.analyst_error: Exception | None = None

        for name in DETERMINISTIC_STEPS:
            monkeypatch.setattr(flow_steps, name, self._step(name))
        monkeypatch.setattr(flow_llm_mode, "run_analyst_crew", self._analyst)
        monkeypatch.setattr(flow_llm_mode, "run_scientist_crew", self._scientist)

    def _step(self, name: str):
        def step(state, ctx, data):
            self.calls.append(name)
            if name == "dataset_intake":
                # Stage everything up front so both crews see a complete artifact set; the real
                # steps write these across steps 2-8.
                _stage_required_artifacts(ctx)
            if name == self.fail_at:
                result = ValidationResult(
                    step=name,
                    passed=False,
                    violations=[
                        Violation(step=name, rule="contract_mismatch", message="column drift")
                    ],
                )
                with ctx.step(name):
                    raise FlowValidationError(result)
            return state

        return step

    def _analyst(self, ctx: RunContext) -> dict:
        self.calls.append("crew1")
        if self.analyst_action is not None:
            self.analyst_action(ctx)
        if self.analyst_error is not None:
            raise self.analyst_error
        return self.analyst_summary

    def _scientist(self, ctx: RunContext) -> dict:
        self.calls.append("crew2")
        if self.scientist_action is not None:
            self.scientist_action(ctx)
        return self.scientist_summary


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _Harness(monkeypatch)


def _run(tmp_path: Path, *, mode: str = "llm", **kwargs) -> SimpleNamespace:
    state, ctx = run_flow(
        mode=mode, raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=tmp_path, **kwargs
    )
    close_log_handlers(ctx.run_id)
    return SimpleNamespace(state=state, ctx=ctx, base=tmp_path)


# --------------------------------------------------------------------------
# order and gating (§37)
# --------------------------------------------------------------------------
def test_both_crews_run_in_the_prescribed_order(tmp_path, harness) -> None:
    run = _run(tmp_path)

    assert run.state.status == "success"
    assert harness.calls == [
        "dataset_intake",
        "data_analyst_work",
        "contract_validation",
        "crew1",
        "data_scientist_work",
        "feature_validation",
        "training_and_backtest",
        "evaluation_and_champion",
        "inventory_policy_calibration",
        "artifact_validation",
        "crew2",
    ]
    # publish is not stubbed: it is what promoted the staged artifacts and finished the run.
    assert run.state.current_step == "publish"


def test_no_llm_mode_never_kicks_off_a_crew(tmp_path, harness) -> None:
    run = _run(tmp_path, mode="no-llm")

    assert "crew1" not in harness.calls
    assert "crew2" not in harness.calls
    assert run.state.status == "success"
    assert run.state.llm == {}
    assert "llm" not in run.ctx.metrics


def test_a_failed_validation_reaches_no_crew_and_costs_nothing(tmp_path, monkeypatch) -> None:
    harness = _Harness(monkeypatch, fail_at="contract_validation")
    run = _run(tmp_path)

    assert run.state.status == "failed"
    assert "crew1" not in harness.calls
    assert "crew2" not in harness.calls

    run_log = _read_json(run.base / _relative(paths.RUN_LOG))
    assert run_log["status"] == "failed"
    assert "llm" not in run_log["metrics"], "an LLM block on a run that never called an LLM"


def test_crew_steps_are_recorded_as_steps(tmp_path, harness) -> None:
    run = _run(tmp_path)
    names = [record.name for record in run.ctx.steps]
    assert flow_llm_mode.CREW1_STEP in names
    assert flow_llm_mode.CREW2_STEP in names
    assert names.index(flow_llm_mode.CREW1_STEP) < names.index(flow_llm_mode.CREW2_STEP)


# --------------------------------------------------------------------------
# the determinism guard (§38): a crew may not change a number
# --------------------------------------------------------------------------
def test_a_crew_that_rewrites_features_csv_has_it_restored_and_logged(tmp_path, harness) -> None:
    def tamper(ctx: RunContext) -> None:
        (ctx.staging_dir / _relative(paths.FEATURES)).write_bytes(b"stock_code,lag_1\nX,999\n")

    harness.analyst_action = tamper
    run = _run(tmp_path)

    staged_features = run.base / _relative(paths.FEATURES)
    assert run.state.status == "success"
    assert staged_features.read_bytes() == STAGED_CONTENT, "the crew's version was published"

    expected = f"crew modified numeric artifact {_relative(paths.FEATURES).as_posix()} — restored"
    assert expected in run.ctx.warnings
    assert run.state.llm["guard_restored"] == [_relative(paths.FEATURES).as_posix()]


def test_a_crew_that_deletes_a_numeric_artifact_has_it_restored(tmp_path, harness) -> None:
    def delete(ctx: RunContext) -> None:
        (ctx.staging_dir / _relative(paths.CLEAN_DATA)).unlink()

    harness.scientist_action = delete
    run = _run(tmp_path)

    assert run.state.status == "success"
    assert (run.base / _relative(paths.CLEAN_DATA)).read_bytes() == STAGED_CONTENT


def test_the_guard_leaves_no_copies_behind(tmp_path, harness) -> None:
    run = _run(tmp_path)
    assert not flow_llm_mode.guard_dir(run.ctx).exists()
    staging_root = run.base / "artifacts" / "_staging"
    leftover = [p for p in staging_root.rglob("*") if p.is_file()] if staging_root.exists() else []
    assert leftover == []


def test_a_crew_that_only_rewrites_a_narrative_triggers_no_restore(tmp_path, harness) -> None:
    def rewrite_insights(ctx: RunContext) -> None:
        ctx.out(_relative(paths.INSIGHTS)).write_text("polished prose\n", encoding="utf-8")

    harness.analyst_action = rewrite_insights
    run = _run(tmp_path)

    assert run.state.llm["guard_restored"] == []
    assert (run.base / _relative(paths.INSIGHTS)).read_text(encoding="utf-8") == "polished prose\n"


# --------------------------------------------------------------------------
# the narrative guard (§38): a number no table backs is never published
# --------------------------------------------------------------------------
def test_an_unbacked_insights_rewrite_is_discarded_and_the_deterministic_text_restored(
    tmp_path, harness
) -> None:
    """The mechanism the Flow relies on, exercised at the Flow's own staged destination.

    The crew stub does exactly what ``write_insights_narrative`` does: hands a candidate to
    :class:`crews.common.NarrativeGuard` with the tables that must back it, writing to
    ``ctx.out(...)`` — this run's staging tree, never the final path, which still holds the
    previous run's copy.
    """
    import pandas as pd

    from crews.common import NarrativeGuard

    deterministic = "Revenue was 1,024,951 units across the period.\n"
    unbacked = "Revenue was 4,999,123 units across the period.\n"
    tables = {"totals": pd.DataFrame({"units": [1024951]})}

    def publish_unbacked(ctx: RunContext) -> None:
        guard = NarrativeGuard("insights", tables, deterministic)
        decision = guard.publish(unbacked, ctx.out(_relative(paths.INSIGHTS)), ctx)
        harness.analyst_summary = {"insights_narrative_accepted": decision.accepted}

    harness.analyst_action = publish_unbacked
    run = _run(tmp_path)

    published = (run.base / _relative(paths.INSIGHTS)).read_text(encoding="utf-8")
    assert published == deterministic
    assert run.state.llm["narrative_accepted"]["insights"] is False
    assert any("insights narrative rejected" in warning for warning in run.ctx.warnings)


def test_accepted_narratives_are_reported_per_file(tmp_path, harness) -> None:
    run = _run(tmp_path)
    assert run.state.llm["narrative_accepted"] == {
        "insights": True,
        "evaluation_report": True,
        "model_card": True,
    }


# --------------------------------------------------------------------------
# cost cap (§47): aborts the narrative step, never the run
# --------------------------------------------------------------------------
def test_the_cost_cap_aborts_the_narrative_step_but_the_run_succeeds(tmp_path, harness) -> None:
    cap = load_model_config().llm.max_cost_usd

    def burn_the_budget(ctx: RunContext) -> None:
        # Priced through llm.pricing, this is far past any sane cap.
        ctx.record_metrics({"crew_data_analyst_prompt_tokens": 500_000_000})

    harness.analyst_action = burn_the_budget
    run = _run(tmp_path)

    assert run.state.status == "success", "the cap must abort the narrative, not the run"
    assert "crew2" not in harness.calls
    assert run.state.llm["crew1_status"] == flow_llm_mode.STATUS_COMPLETED
    assert run.state.llm["crew2_status"] == flow_llm_mode.STATUS_COST_CAPPED
    assert run.state.llm["cost_usd"] > cap
    assert run.state.llm["narrative_accepted"]["evaluation_report"] is False
    assert run.state.llm["narrative_accepted"]["model_card"] is False
    assert any("has reached the" in warning for warning in run.ctx.warnings)

    run_log = _read_json(run.base / _relative(paths.RUN_LOG))
    assert run_log["status"] == "success"


def test_the_cli_cap_overrides_the_configured_one(tmp_path, harness) -> None:
    run = _run(tmp_path, max_cost_usd=0.5)
    assert run.state.llm["max_cost_usd"] == 0.5


# --------------------------------------------------------------------------
# a crew's own failure is a warning, not a failed run
# --------------------------------------------------------------------------
def test_a_crew_error_warns_and_the_run_still_publishes(tmp_path, harness) -> None:
    harness.analyst_error = RuntimeError("provider returned 503")
    run = _run(tmp_path)

    assert run.state.status == "success"
    assert run.state.llm["crew1_status"] == flow_llm_mode.STATUS_FAILED
    assert any("provider returned 503" in warning for warning in run.ctx.warnings)
    # The run went on to crew 2 and published.
    assert "crew2" in harness.calls
    assert (run.base / _relative(paths.CLEAN_DATA)).is_file()


# --------------------------------------------------------------------------
# completeness is re-checked after the narrative rewrite (§8 of the issue)
# --------------------------------------------------------------------------
def test_a_truncated_report_after_the_rewrite_stops_the_run_gracefully(tmp_path, harness) -> None:
    def truncate(ctx: RunContext) -> None:
        (ctx.staging_dir / _relative(paths.EVALUATION_REPORT)).write_bytes(b"")

    harness.scientist_action = truncate
    run = _run(tmp_path)

    assert run.state.status == "failed"
    assert run.state.errors[-1]["type"] == "FlowValidationError"
    assert "evaluation_report.md was not generated" in run.state.errors[-1]["message"]

    report = _read_json(run.base / _relative(paths.VALIDATION_REPORT))
    assert report["run_id"] == run.ctx.run_id
    assert report["passed"] is False
    assert report["violations"][0]["rule"] == "required_artifact"
    # §39: nothing was promoted, so the previous run's files are untouched.
    assert not (run.base / _relative(paths.INVENTORY_PLAN)).exists()


# --------------------------------------------------------------------------
# run_log.json — the published record of an LLM run (§8 of the issue)
# --------------------------------------------------------------------------
def test_run_log_records_mode_llm_and_the_metrics_llm_block(tmp_path, harness) -> None:
    run = _run(tmp_path)
    run_log = _read_json(run.base / _relative(paths.RUN_LOG))

    assert run_log["mode"] == "llm"
    llm = run_log["metrics"]["llm"]
    assert llm["crew1_status"] == flow_llm_mode.STATUS_COMPLETED
    assert llm["crew2_status"] == flow_llm_mode.STATUS_COMPLETED
    assert set(llm["tokens"]) == {"prompt", "cached_prompt", "completion", "total"}
    assert isinstance(llm["cost_usd"], float)
    assert set(llm["narrative_accepted"]) == set(flow_llm_mode.NARRATIVE_KEYS)
    assert "llm" not in run_log, "metrics.llm is the published location, not a top-level key"


def test_no_credential_value_reaches_the_run_log(tmp_path, harness, monkeypatch) -> None:
    secret = "sk-test-DO-NOT-LEAK-0123456789abcdef"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    run = _run(tmp_path)

    payload = (run.base / _relative(paths.RUN_LOG)).read_text(encoding="utf-8")
    assert secret not in payload
    log_file = run.base / "logs" / f"run_{run.ctx.run_id}.log"
    assert secret not in log_file.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# token accounting and pricing
# --------------------------------------------------------------------------
def test_token_totals_count_cached_prompt_tokens_exactly_once(tmp_path) -> None:
    ctx = RunContext.start(mode="llm", staging=True, base_dir=tmp_path)
    try:
        ctx.record_metrics(
            {
                "crew_data_analyst_prompt_tokens": 1000,
                "crew_data_analyst_cached_prompt_tokens": 400,
                "crew_data_analyst_completion_tokens": 200,
                "crew_data_analyst_total_tokens": 1200,
                "crew_data_scientist_prompt_tokens": 500,
                "crew_data_scientist_cached_prompt_tokens": 0,
                "crew_data_scientist_completion_tokens": 100,
                "crew_data_scientist_total_tokens": 600,
                "panel_rows": 4242,  # an unrelated metric must not be counted
            }
        )
        totals = flow_llm_mode.token_totals(ctx)
    finally:
        close_log_handlers(ctx.run_id)

    assert totals == {"prompt": 1500, "cached_prompt": 400, "completion": 300, "total": 1800}


def test_cached_prompt_tokens_are_priced_below_uncached_ones() -> None:
    from crews.environment import estimate_cost_usd

    uncached = estimate_cost_usd(prompt_tokens=10_000, model="gpt-4o-mini")
    half_cached = estimate_cost_usd(
        prompt_tokens=10_000, cached_prompt_tokens=5_000, model="gpt-4o-mini"
    )
    assert 0 < half_cached < uncached


def test_an_unpriced_model_falls_back_to_the_default_rate() -> None:
    from crews.environment import estimate_cost_usd

    pricing = load_model_config().llm.pricing
    assert "default" in pricing
    unknown = estimate_cost_usd(prompt_tokens=1000, model="a-model-nobody-listed")
    assert unknown == pytest.approx(pricing["default"].prompt_usd_per_1k)


# --------------------------------------------------------------------------
# narrative-only mode: hydration + the guard, over real artifacts
# --------------------------------------------------------------------------
def _hydrated_crew(tmp_path: Path):
    """Build the narrative-only crew over a copy of the repository's committed artifacts.

    Copied into a ``base_dir`` rather than read in place so the tool writes land in a temporary
    tree; skipped when the repository has no artifacts yet (a fresh clone before the first run).
    """
    import shutil

    from crewai import LLM

    from crews.data_scientist.crew import DataScientistCrew
    from crews.data_scientist.tools import HYDRATED_TABLES

    sources = [
        *HYDRATED_TABLES.values(),
        paths.CHAMPION_DECISION,
        paths.DATASET_CONTRACT,
        paths.MODEL_META,
        paths.CANDIDATES_META,
        paths.EVALUATION_REPORT,
        paths.MODEL_CARD,
    ]
    if not all(path.is_file() for path in sources):
        pytest.skip("the repository's committed artifacts are not present")
    for canonical in sources:
        destination = tmp_path / _relative(canonical)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical, destination)

    ctx = RunContext.start(mode="llm", base_dir=tmp_path)
    stub = LLM(model="stub-model", api_key="stub-not-used-no-call-is-made")
    return DataScientistCrew(ctx, llm=stub, narrative_only=True), ctx


def test_narrative_only_hydration_finds_every_source(tmp_path) -> None:
    builder, ctx = _hydrated_crew(tmp_path)
    try:
        state = builder.toolset.state
        assert builder.missing_sources == []
        assert [task.name for task in builder.tasks] == ["narrative_rewrite"]
        assert state.reports_written is True
        assert state.decision is not None
        assert state.deterministic_evaluation_report
        assert state.deterministic_model_card
    finally:
        close_log_handlers(ctx.run_id)


def test_narrative_only_crew_carries_no_tool_that_computes_a_number(tmp_path) -> None:
    from crews.data_scientist.tools import NARRATIVE_TOOL_NAMES

    builder, ctx = _hydrated_crew(tmp_path)
    try:
        names = [tool.name for agent in builder.agents.values() for tool in agent.tools]
        assert names == list(NARRATIVE_TOOL_NAMES)
    finally:
        close_log_handlers(ctx.run_id)


def test_a_hydrated_narrative_rejects_an_unbacked_number(tmp_path) -> None:
    """The guard must work identically over hydrated tables — that is the point of hydrating."""
    builder, ctx = _hydrated_crew(tmp_path)
    try:
        deterministic = builder.toolset.state.deterministic_evaluation_report
        tampered = deterministic.replace(
            "# ", "Fill rate reached 99.87 % on 1,234,567 units.\n\n# ", 1
        )
        tool = next(
            tool
            for tool in builder.toolset.narrative_tools
            if tool.name == "write_evaluation_narrative_tool"
        )
        result = json.loads(tool.run(markdown=tampered))

        assert result["accepted"] is False
        published = ctx.base_dir / _relative(paths.EVALUATION_REPORT)
        assert published.read_text(encoding="utf-8") == deterministic
    finally:
        close_log_handlers(ctx.run_id)


def test_a_hydrated_narrative_accepts_the_deterministic_text_unchanged(tmp_path) -> None:
    builder, ctx = _hydrated_crew(tmp_path)
    try:
        deterministic = builder.toolset.state.deterministic_evaluation_report
        tool = next(
            tool
            for tool in builder.toolset.narrative_tools
            if tool.name == "write_evaluation_narrative_tool"
        )
        result = json.loads(tool.run(markdown=deterministic))
        assert result["accepted"] is True
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# the CLI mode switch (§2 of the issue)
# --------------------------------------------------------------------------
def _clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from crews.environment import API_KEY_VARIABLES

    for name in API_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_default_without_a_key_falls_back_to_no_llm(monkeypatch, capsys) -> None:
    from pipeline.__main__ import main

    _clear_credentials(monkeypatch)
    recorded: dict = {}

    def fake_run_flow(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(status="success", errors=[]), None

    monkeypatch.setattr(flow_main, "run_flow", fake_run_flow)
    assert main(["--sample"]) == 0
    assert recorded["mode"] == "no-llm"
    assert "falling back to --no-llm" in capsys.readouterr().out


def test_llm_flag_without_a_key_exits_2_without_starting_a_run(monkeypatch, capsys) -> None:
    from pipeline.__main__ import main

    _clear_credentials(monkeypatch)

    def forbidden(**kwargs):  # pragma: no cover - would fail the test
        raise AssertionError("no run may start without a credential in --llm mode")

    monkeypatch.setattr(flow_main, "run_flow", forbidden)
    assert main(["--llm", "--sample"]) == 2
    assert "No LLM credential found" in capsys.readouterr().err


def test_default_with_a_key_selects_llm_mode(monkeypatch, capsys) -> None:
    from pipeline.__main__ import main

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-DO-NOT-LEAK-0123456789abcdef")
    recorded: dict = {}

    def fake_run_flow(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(status="success", errors=[]), None

    monkeypatch.setattr(flow_main, "run_flow", fake_run_flow)
    assert main(["--sample", "--max-llm-cost-usd", "0.25"]) == 0
    assert recorded["mode"] == "llm"
    assert recorded["max_cost_usd"] == 0.25

    printed = capsys.readouterr().out
    assert "OPENAI_API_KEY" in printed, "the variable NAME is what gets printed"
    assert "sk-test" not in printed, "the variable VALUE never is"


def test_no_llm_and_llm_are_mutually_exclusive() -> None:
    from pipeline.__main__ import main

    with pytest.raises(SystemExit):
        main(["--no-llm", "--llm"])


# --------------------------------------------------------------------------
# the import boundary (docs/interfaces.md §6 rule 10)
# --------------------------------------------------------------------------
def test_the_deterministic_steps_still_import_no_crewai() -> None:
    source = Path(flow_steps.__file__).read_text(encoding="utf-8")
    assert "crewai" not in source
    assert "make_llm" not in source


def test_llm_mode_imports_the_crews_only_inside_functions() -> None:
    """Importing :mod:`flow.llm_mode` must cost nothing — the seams import the crews lazily."""
    source = Path(flow_llm_mode.__file__).read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and ("crews" in line or "crewai" in line)
    ]
    assert module_level == []


def test_pipeline_main_imports_no_crewai_at_module_scope() -> None:
    source = (Path(paths.PROJECT_ROOT) / "src" / "pipeline" / "__main__.py").read_text(
        encoding="utf-8"
    )
    module_level = [
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any("crew" in line for line in module_level)
