"""Data Scientist Crew — structure, tool boundary and the narrative guard (US-26, PRD §36, §38).

No test here makes a network call. The crew is built with a stub LLM and never kicked off (except
for one cheap "produced nothing" check); the tools are driven directly, which is the honest way to
test them — they are the part that must behave identically whether an agent or the Flow calls them.

Three things are the centre of the file, mirroring ``test_crew_data_analyst.py``:

* the tool chain — T1 (feature engineering) through T3 (evaluation, inventory, champion, reports)
  driven directly against the CI sample fixture, once, in a module-scoped fixture;
* the stop-on-validation-failure contract — a validation tool returning ``passed=false`` must stop
  the run itself, not hand the agent a JSON blob it could choose to ignore (§8 interface
  corrections);
* the narrative guard — a rewrite is published only if its headings and tables are unchanged and
  every number in it is backed by a computed table; otherwise the deterministic version is kept.

Everything writes to a ``tmp_path`` base directory; no test touches the real ``artifacts/``, except
where the crew's own tools call ``write_validation_report`` with no explicit path (the same
convention :mod:`crews.data_analyst.tools` already uses) — those write to the committed
``artifacts/validation_report.json``, exactly as the Data Analyst Crew's own tests already do.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from crewai import LLM, Process

from crews import common
from crews.data_scientist import __main__ as entry_point
from crews.data_scientist import crew as crew_module
from crews.data_scientist import tools as tools_module
from crews.data_scientist.crew import (
    AGENT_ORDER,
    AGENTS_CONFIG,
    REQUIRED_OUTPUTS,
    TASK_ORDER,
    TASKS_CONFIG,
    DataScientistCrew,
    build_crew,
    load_config,
    verify_outputs,
)
from crews.data_scientist.tools import (
    DataScientistToolset,
    make_tools,
    relative_path,
    resolve_read,
)
from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import load_cleaning_config, load_model_config, load_non_inventory_codes
from pipeline.contract import write_contract
from pipeline.download import load_raw
from pipeline.eda.run_eda import run_eda
from pipeline.evaluate import evaluate
from pipeline.features import build_features, write_features
from pipeline.panel import build_panel, validate_panel
from pipeline.reports import MODEL_CARD_HEADINGS
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import FlowValidationError, ValidationResult, Violation

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: The tools §2 of the issue assigns to each agent.
PRD_TOOLS: dict[str, set[str]] = {
    "feature_engineering_specialist": {
        "validate_contract_tool",
        "build_features_tool",
        "leakage_check_tool",
    },
    "forecasting_model_scientist": {"tune_tool", "train_models_tool", "backtest_tool"},
    "model_evaluation_inventory_scientist": {
        "evaluate_tool",
        "robust_sigma_tool",
        "simulate_inventory_tool",
        "select_champion_tool",
        "latest_forecast_tool",
        "quarterly_tool",
        "write_reports_tool",
        "read_eval_table_tool",
        "read_champion_decision_tool",
        "read_model_meta_tool",
    },
}

#: The sentence PRD §38 requires every agent to carry, in its goal and in its backstory.
NEVER_COMPUTE = (
    "You never compute numbers yourself; you call tools and quote numbers only from tool "
    "outputs / tables (PRD §38)."
)

#: A number that is in no table this project computes, at a precision the guard cannot excuse.
UNBACKED_NUMBER = "1,234,567.89"


def stub_llm() -> LLM:
    """An LLM object that is never invoked — the crew is built, never kicked off."""
    return LLM(model="stub-model-never-called")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crew_run(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """Drive the crew's tools through T1 -> T2 -> T3 on the committed CI fixture.

    This is the deterministic half of a crew run: exactly the calls the three agents are asked to
    make (skipping the optional ``tune_tool``, precisely as ``tests/test_flow_no_llm.py`` skips
    tuning — the grid search would blow the CI time budget for no change in correctness here).
    ``clean_data.csv`` and ``dataset_contract.json`` are built first, directly, the way the Data
    Analyst Crew (this crew's declared dependency) produces them.
    """
    base = tmp_path_factory.mktemp("crew_data_scientist")
    ctx = RunContext.start(mode="llm", base_dir=base)

    cleaning_cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    with ctx.step("setup:clean_transactions"):
        clean_df, waterfall_df = clean_transactions(raw, cleaning_cfg, ctx)
    returns_path = resolve_read(ctx, relative_path(paths.PROCESSED_DIR / RETURNS_FILENAME))
    with ctx.step("setup:build_panel"):
        panel = build_panel(clean_df, pd.read_parquet(returns_path), cleaning_cfg, ctx)
        panel_check = validate_panel(panel, cleaning_cfg)
        assert panel_check.passed, panel_check.summary()
    with ctx.step("setup:run_eda"):
        # write_reports_tool reads data_quality_findings.json, which only run_eda assembles.
        run_eda(clean_df, panel, waterfall_df, cleaning_cfg, ctx, raw_df=raw)
    with ctx.step("setup:write_contract"):
        write_contract(panel, cleaning_cfg, load_model_config(), load_non_inventory_codes(), ctx)

    toolset = DataScientistToolset(ctx)
    results = {
        "validate_contract_tool": toolset._validate_contract(),
        "build_features_tool": toolset._build_features(),
        "leakage_check_tool": toolset._leakage_check(),
        "train_models_tool": toolset._train_models(),
        "backtest_tool": toolset._backtest(),
        "evaluate_tool": toolset._evaluate(),
        "robust_sigma_tool": toolset._robust_sigma(),
        "simulate_inventory_tool": toolset._simulate_inventory(),
        "select_champion_tool": toolset._select_champion(),
        "latest_forecast_tool": toolset._latest_forecast(),
        "quarterly_tool": toolset._quarterly(),
        "write_reports_tool": toolset._write_reports(),
    }
    yield SimpleNamespace(ctx=ctx, toolset=toolset, base=base, results=results)
    close_log_handlers(ctx.run_id)


@pytest.fixture
def bare_ctx(tmp_path: Path) -> RunContext:
    """A fresh run against a temporary base directory, for tests that need no data."""
    ctx = RunContext.start(mode="llm", base_dir=tmp_path)
    yield ctx
    close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# structure: three agents, their tools, the task order
# --------------------------------------------------------------------------
def test_crew_builds_with_three_agents_and_runs_sequentially(bare_ctx: RunContext) -> None:
    crew = build_crew(bare_ctx, llm=stub_llm())
    assert len(crew.agents) == 3
    assert crew.process is Process.sequential


def test_agents_yaml_defines_exactly_the_three_prd_agents() -> None:
    config = load_config(AGENTS_CONFIG)
    assert set(config) == set(AGENT_ORDER) == set(PRD_TOOLS)


def test_every_agent_carries_the_never_compute_sentence() -> None:
    """PRD §38 is part of each agent's own prompt, not just of the code around it."""
    config = load_config(AGENTS_CONFIG)
    for key, agent in config.items():
        for field in ("goal", "backstory"):
            collapsed = " ".join(agent[field].split())
            assert NEVER_COMPUTE in collapsed, f"{key}.{field} is missing the §38 sentence"


def test_tools_match_the_prd_table(bare_ctx: RunContext) -> None:
    toolset = DataScientistToolset(bare_ctx)
    for key, expected in PRD_TOOLS.items():
        names = {tool.name for tool in toolset.by_agent[key]}
        assert expected <= names, f"{key} is missing {sorted(expected - names)}"
    # No agent holds a tool the PRD gives to another agent.
    for key, expected in PRD_TOOLS.items():
        others = set().union(*(value for other, value in PRD_TOOLS.items() if other != key))
        assert not ({tool.name for tool in toolset.by_agent[key]} & (others - expected))


def test_make_tools_returns_every_tool_bound_to_the_run(bare_ctx: RunContext) -> None:
    names = {tool.name for tool in make_tools(bare_ctx)}
    assert set().union(*PRD_TOOLS.values()) <= names
    assert {"write_evaluation_narrative_tool", "write_model_card_narrative_tool"} <= names


def test_task_order_is_t1_t2_t3_narrative(bare_ctx: RunContext) -> None:
    builder = DataScientistCrew(bare_ctx, llm=stub_llm())
    assert [task.name for task in builder.tasks] == list(TASK_ORDER)
    tasks_config = load_config(TASKS_CONFIG)
    for task, key in zip(builder.tasks, TASK_ORDER, strict=True):
        assert task.agent is builder.agents[tasks_config[key]["agent"]]


def test_writing_tools_accept_no_model_supplied_arguments(bare_ctx: RunContext) -> None:
    """The model chooses *when* a deterministic tool runs, never what it runs on (§8)."""
    toolset = DataScientistToolset(bare_ctx)
    by_name = {tool.name: tool for tool in toolset.tools}
    no_arg_names = (
        PRD_TOOLS["feature_engineering_specialist"] | PRD_TOOLS["forecasting_model_scientist"]
    )
    for name in no_arg_names:
        assert by_name[name].args_schema.model_fields == {}, f"{name} exposes an argument"
    for name in (
        "evaluate_tool",
        "robust_sigma_tool",
        "simulate_inventory_tool",
        "select_champion_tool",
        "latest_forecast_tool",
        "quarterly_tool",
        "write_reports_tool",
        "read_champion_decision_tool",
        "read_model_meta_tool",
    ):
        assert by_name[name].args_schema.model_fields == {}, f"{name} exposes an argument"
    assert set(by_name["read_eval_table_tool"].args_schema.model_fields) == {"name"}
    assert set(by_name["write_evaluation_narrative_tool"].args_schema.model_fields) == {"markdown"}


# --------------------------------------------------------------------------
# the tool chain: the numbers are the pipeline's, not the agent's
# --------------------------------------------------------------------------
def test_every_tool_reported_success(crew_run: SimpleNamespace) -> None:
    for name, payload in crew_run.results.items():
        decoded = json.loads(payload)
        assert "error" not in decoded, f"{name} returned {payload}"


def test_tool_results_are_json_summaries_not_frames(crew_run: SimpleNamespace) -> None:
    for name, payload in crew_run.results.items():
        decoded = json.loads(payload)
        assert isinstance(decoded, dict), name
        for key, value in decoded.items():
            if not isinstance(value, list):
                continue
            rows = [item for item in value if isinstance(item, dict)]
            assert len(rows) <= 20, f"{name}.{key} returned more than the 20-row cap"


def test_every_tool_call_is_recorded_as_a_step(crew_run: SimpleNamespace) -> None:
    recorded = {step.name for step in crew_run.ctx.steps}
    # Tools whose wrapped function opens its own named step are recorded under that name, not the
    # crew's prefix (module docstring of tools.py) — everything else is prefixed.
    directly_named = {
        "backtest",
        "evaluate",
        "sigma",
        "champion_selection",
        "quarterly_aggregation",
    }
    prefixed = {
        "validate_contract",
        "build_features",
        "feature_validation",
        "train_models",
        "backtest_summary",
        "inventory_simulation",
        "latest_forecast",
        "reports",
    }
    for name in prefixed:
        assert f"crew_data_scientist:{name}" in recorded, name
    assert directly_named <= recorded


def test_the_required_artifacts_exist(crew_run: SimpleNamespace) -> None:
    for key in REQUIRED_OUTPUTS:
        assert resolve_read(crew_run.ctx, REQUIRED_OUTPUTS[key]).is_file(), key


def test_verify_outputs_reports_nothing_missing(crew_run: SimpleNamespace) -> None:
    assert verify_outputs(crew_run.ctx) == []


def test_out_of_order_tool_call_is_reported_but_does_not_fail_the_run(
    bare_ctx: RunContext,
) -> None:
    """An agent mistake is recoverable; only a broken deterministic tool fails a run."""
    toolset = DataScientistToolset(bare_ctx)
    decoded = json.loads(toolset._build_features())
    assert "validate_contract" in decoded["error"]
    assert bare_ctx.status == "running"
    assert bare_ctx.errors == []
    assert bare_ctx.steps == []


def test_read_eval_table_rejects_a_name_that_is_not_computed(crew_run: SimpleNamespace) -> None:
    decoded = json.loads(crew_run.toolset._read_eval_table("not_a_real_table"))
    assert "error" in decoded
    assert decoded["available"], "the agent is told which tables exist"


def test_read_eval_table_caps_the_rows_it_shows(crew_run: SimpleNamespace) -> None:
    name = sorted(crew_run.toolset.state.tables)[0]
    decoded = json.loads(crew_run.toolset._read_eval_table(name))
    assert len(decoded["records"]) <= 20
    assert decoded["rows"] >= decoded["rows_shown"]


def test_read_champion_decision_reports_the_winner(crew_run: SimpleNamespace) -> None:
    decoded = json.loads(crew_run.toolset._read_champion_decision())
    assert decoded["champion"]
    assert decoded["champion_kind"] in {"ml", "baseline"}
    assert decoded["candidates"]


def test_read_model_meta_reports_the_refit_champion(crew_run: SimpleNamespace) -> None:
    decoded = json.loads(crew_run.toolset._read_model_meta())
    assert decoded["champion"] == crew_run.toolset.state.decision.champion


# --------------------------------------------------------------------------
# byte-identical outputs (§6 of the issue): the crew must not recompute anything
# --------------------------------------------------------------------------
def test_features_csv_is_byte_identical_to_a_direct_build(
    crew_run: SimpleNamespace, tmp_path: Path
) -> None:
    direct_base = tmp_path / "direct_features"
    ctx = RunContext.start(mode="no-llm", base_dir=direct_base)
    cfg = load_model_config()
    cleaning_cfg = load_cleaning_config()
    with ctx.step("direct"):
        frame = build_features(
            crew_run.toolset.state.panel_df,
            cfg.active_rule.k,
            cfg.split.first_target_month,
            cleaning_cfg.raw.last_full_month,
            cfg,
        )
        write_features(frame, ctx)
    close_log_handlers(ctx.run_id)

    from_crew = (crew_run.base / relative_path(paths.FEATURES)).read_bytes()
    from_direct = (direct_base / relative_path(paths.FEATURES)).read_bytes()
    assert from_crew == from_direct


def test_holdout_metrics_overall_is_byte_identical_to_a_direct_evaluate(
    crew_run: SimpleNamespace, tmp_path: Path
) -> None:
    direct_base = tmp_path / "direct_evaluate"
    ctx = RunContext.start(mode="no-llm", base_dir=direct_base)
    cfg = load_model_config()
    state = crew_run.toolset.state
    evaluate(state.holdout_predictions_df, state.backtest_df, state.abc_train_df, cfg, ctx)
    close_log_handlers(ctx.run_id)

    canonical = paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv"
    from_crew = (crew_run.base / relative_path(canonical)).read_bytes()
    from_direct = (direct_base / relative_path(canonical)).read_bytes()
    assert from_crew == from_direct


# --------------------------------------------------------------------------
# stop-on-validation-failure (§8 interface corrections): the wrapper stops, not the agent
# --------------------------------------------------------------------------
def test_validate_contract_stops_the_run_on_failure(
    monkeypatch: pytest.MonkeyPatch, bare_ctx: RunContext
) -> None:
    bad_result = ValidationResult(
        step="contract_validation",
        passed=False,
        violations=[
            Violation(
                step="contract_validation", rule="primary_key", message="synthetic failure for test"
            )
        ],
    )
    monkeypatch.setattr(tools_module, "validate_contract", lambda panel, contract: bad_result)
    monkeypatch.setattr(
        tools_module, "read_panel", lambda path: pd.DataFrame({"stock_code": ["A"]})
    )

    panel_path = bare_ctx.base_dir / relative_path(paths.CLEAN_DATA)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel_path.write_text("stock_code\nA\n", encoding="utf-8")
    contract_path = bare_ctx.base_dir / relative_path(paths.DATASET_CONTRACT)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text("{}", encoding="utf-8")

    toolset = DataScientistToolset(bare_ctx)
    with pytest.raises(FlowValidationError):
        toolset._validate_contract()

    assert bare_ctx.status == "failed"
    assert toolset.state.panel_df is None
    # a validation stop must not let a downstream tool proceed as if the data were good
    decoded = json.loads(toolset._build_features())
    assert "error" in decoded


def test_leakage_check_stops_the_run_on_failure(
    monkeypatch: pytest.MonkeyPatch, bare_ctx: RunContext
) -> None:
    bad_result = ValidationResult(
        step=tools_module.FEATURE_VALIDATION_STEP,
        passed=False,
        violations=[
            Violation(
                step=tools_module.FEATURE_VALIDATION_STEP,
                rule="target_present",
                message="synthetic failure for test",
            )
        ],
    )
    ok_result = ValidationResult(step=tools_module.FEATURE_VALIDATION_STEP, passed=True)
    monkeypatch.setattr(tools_module, "validate_features", lambda *a, **k: bad_result)
    monkeypatch.setattr(tools_module, "leakage_check", lambda *a, **k: ok_result)

    toolset = DataScientistToolset(bare_ctx)
    toolset.state.panel_df = pd.DataFrame({"stock_code": ["A"], "month": ["2011-01"]})
    toolset.state.features_df = pd.DataFrame({"stock_code": ["A"], "target_month": ["2011-01"]})

    with pytest.raises(FlowValidationError):
        toolset._leakage_check()
    assert bare_ctx.status == "failed"


def test_a_crew_that_produced_nothing_fails_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """promote() only warns about a missing artifact (rule 8) — this must raise instead."""
    monkeypatch.setattr(crew_module, "make_llm", stub_llm)

    class _NoOpCrew:
        token_usage = None

        def kickoff(self) -> SimpleNamespace:
            return SimpleNamespace(token_usage=None)

    monkeypatch.setattr(crew_module.DataScientistCrew, "crew", lambda self: _NoOpCrew())

    ctx = RunContext.start(mode="llm", base_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="without producing"):
            crew_module.run_data_scientist_crew(ctx)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# the narrative guard — the mechanism behind §38
# --------------------------------------------------------------------------
def test_evaluation_narrative_accepts_the_text_unchanged(crew_run: SimpleNamespace) -> None:
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_evaluation_report
    assert deterministic
    decoded = json.loads(toolset._write_evaluation_narrative(deterministic))
    assert decoded["accepted"] is True
    published = resolve_read(crew_run.ctx, relative_path(paths.EVALUATION_REPORT)).read_text(
        encoding="utf-8"
    )
    assert published == deterministic


def test_evaluation_narrative_rejects_an_unbacked_number(crew_run: SimpleNamespace) -> None:
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_evaluation_report
    assert deterministic
    invented = deterministic + f"\n\nDemand actually reached {UNBACKED_NUMBER} units.\n"

    decoded = json.loads(toolset._write_evaluation_narrative(invented))
    assert decoded["accepted"] is False
    assert UNBACKED_NUMBER in decoded["unmatched"]

    published = resolve_read(crew_run.ctx, relative_path(paths.EVALUATION_REPORT)).read_text(
        encoding="utf-8"
    )
    assert published == deterministic, "the rejected draft must not survive"

    # restore a clean state for later tests in this module
    toolset._write_evaluation_narrative(deterministic)


def test_evaluation_narrative_rejects_an_altered_table(crew_run: SimpleNamespace) -> None:
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_evaluation_report
    assert deterministic
    table_lines = [line for line in deterministic.splitlines() if line.strip().startswith("|")]
    assert table_lines, "the deterministic report must contain at least one markdown table"
    mutated = deterministic.replace(table_lines[0], table_lines[0] + " ", 1)

    decoded = json.loads(toolset._write_evaluation_narrative(mutated))
    assert "error" in decoded
    published = resolve_read(crew_run.ctx, relative_path(paths.EVALUATION_REPORT)).read_text(
        encoding="utf-8"
    )
    assert published == deterministic


def test_model_card_narrative_accepts_the_text_unchanged(crew_run: SimpleNamespace) -> None:
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_model_card
    assert deterministic
    decoded = json.loads(toolset._write_model_card_narrative(deterministic))
    assert decoded["accepted"] is True
    published = resolve_read(crew_run.ctx, relative_path(paths.MODEL_CARD)).read_text(
        encoding="utf-8"
    )
    assert published == deterministic


def test_model_card_narrative_rejects_a_missing_heading(crew_run: SimpleNamespace) -> None:
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_model_card
    assert deterministic
    mutated = deterministic.replace(MODEL_CARD_HEADINGS[4], "## Ethics", 1)
    assert mutated != deterministic

    decoded = json.loads(toolset._write_model_card_narrative(mutated))
    assert "error" in decoded
    published = resolve_read(crew_run.ctx, relative_path(paths.MODEL_CARD)).read_text(
        encoding="utf-8"
    )
    assert published == deterministic

    # restore a clean state for later tests in this module
    toolset._write_model_card_narrative(deterministic)


def test_narrative_tool_requires_write_reports_to_have_run(bare_ctx: RunContext) -> None:
    toolset = DataScientistToolset(bare_ctx)
    decoded = json.loads(toolset._write_evaluation_narrative("# anything\n"))
    assert "error" in decoded


# --------------------------------------------------------------------------
# credentials: none in the source, and a clean exit without one
# --------------------------------------------------------------------------
def test_missing_credential_exits_2_without_starting_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in common.API_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    assert entry_point.main([]) == 2
    assert not (tmp_path / "artifacts").exists()


def test_the_crew_imports_without_a_credential() -> None:
    """Importing the crew must never require a key — CI has none (§6)."""
    assert callable(build_crew)


def test_the_crew_config_holds_no_secret() -> None:
    """Rule 11: config_snapshot is serialised verbatim into run_log.json — keys stay in env."""
    for path in (AGENTS_CONFIG, TASKS_CONFIG):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        flattened = json.dumps(config).lower()
        for forbidden in ("api_key", "openai_api_key", "token", "secret", "password"):
            assert forbidden not in flattened, f"{path.name} mentions {forbidden}"
