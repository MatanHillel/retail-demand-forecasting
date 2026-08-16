"""Data Analyst Crew — structure, tool boundary and the narrative guard (US-12, PRD §35, §38).

No test here makes a network call. The crew is built with a stub LLM and never kicked off; the
tools are driven directly, which is the honest way to test them — they are the part that must
behave identically whether an agent or the Flow calls them.

The centre of the file is the guard: an agent may rewrite ``insights.md``, but a rewrite
containing a number that is in no computed table must be thrown away and the deterministic
version restored. That is PRD §38 made mechanical, so it is tested for what it *rejects* as much
as for what it accepts.

Everything writes to a ``tmp_path`` base directory; no test touches the real ``artifacts/``.
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
from crews.data_analyst import __main__ as entry_point
from crews.data_analyst import crew as crew_module
from crews.data_analyst.crew import (
    AGENT_ORDER,
    AGENTS_CONFIG,
    REQUIRED_OUTPUTS,
    TASK_ORDER,
    TASKS_CONFIG,
    DataAnalystCrew,
    build_crew,
    load_config,
    verify_outputs,
)
from crews.data_analyst.tools import (
    DataAnalystToolset,
    deterministic_review,
    make_tools,
    relative_path,
    resolve_read,
    review_path,
)
from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import load_cleaning_config
from pipeline.download import compute_sha256, load_raw
from pipeline.narrative import numbers_in_tables
from pipeline.panel import build_panel
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: The tools PRD §35 assigns to each agent. The crew adds read-only helpers and the two narrative
#: writers on top; these are the ones the PRD table names, and they must all be present.
PRD_TOOLS: dict[str, set[str]] = {
    "data_quality_analyst": {"profile_raw", "detect_duplicates", "list_nonproduct_codes"},
    "data_preparation_analyst": {"clean_transactions", "build_panel"},
    "business_eda_analyst": {"run_eda", "write_contract"},
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
    """Drive the crew's tools through T1 → T2 → T3 on the committed CI fixture.

    This is the deterministic half of a crew run: exactly the calls the three agents are asked to
    make, with no LLM in the loop. What it produces must be identical to what ``--no-llm``
    produces, which the byte-comparison test then checks.
    """
    base = tmp_path_factory.mktemp("crew_data_analyst")
    ctx = RunContext.start(mode="llm", base_dir=base)
    toolset = DataAnalystToolset(ctx)
    raw, _ = load_raw(RAW_SAMPLE)
    # Stand in for US-03: a real run loads through ``load_raw(ctx=ctx)``, which records the
    # dataset identity the report header shows. The fixture hands the frame in directly.
    toolset.state.raw_df = raw
    ctx.record_data(
        file=RAW_SAMPLE, sha256=compute_sha256(RAW_SAMPLE), rows=len(raw), columns=raw.shape[1]
    )

    results = {
        "profile_raw": toolset._profile_raw(),
        "detect_duplicates": toolset._detect_duplicates(),
        "list_nonproduct_codes": toolset._list_nonproduct_codes(),
        "clean_transactions": toolset._clean_transactions(),
        "build_panel": toolset._build_panel(),
        "run_eda": toolset._run_eda(),
        "write_contract": toolset._write_contract(),
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
    toolset = DataAnalystToolset(bare_ctx)
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
    assert {"read_table", "read_findings", "write_insights_narrative"} <= names


def test_task_order_is_t1_t2_t3(bare_ctx: RunContext) -> None:
    builder = DataAnalystCrew(bare_ctx, llm=stub_llm())
    assert [task.name for task in builder.tasks] == list(TASK_ORDER)
    # Each task is owned by the agent tasks.yaml assigns it to, in the §35 order.
    tasks_config = load_config(TASKS_CONFIG)
    for task, key in zip(builder.tasks, TASK_ORDER, strict=True):
        assert task.agent is builder.agents[tasks_config[key]["agent"]]
    assert [tasks_config[key]["agent"] for key in TASK_ORDER] == list(AGENT_ORDER)


def test_writing_tools_accept_no_model_supplied_arguments(bare_ctx: RunContext) -> None:
    """The model chooses *when* a deterministic tool runs, never what it runs on (§8)."""
    toolset = DataAnalystToolset(bare_ctx)
    by_name = {tool.name: tool for tool in toolset.tools}
    for name in ("clean_transactions", "build_panel", "run_eda", "write_contract"):
        assert by_name[name].args_schema.model_fields == {}, f"{name} exposes an argument"


# --------------------------------------------------------------------------
# the tool chain: the numbers are the pipeline's, not the agent's
# --------------------------------------------------------------------------
def test_every_tool_reported_success(crew_run: SimpleNamespace) -> None:
    for name, payload in crew_run.results.items():
        assert "error" not in json.loads(payload), f"{name} returned {payload}"


def test_tool_results_are_json_summaries_not_frames(crew_run: SimpleNamespace) -> None:
    """§3: a tool returns a compact summary — never a DataFrame, never a whole table.

    The cap applies to *rows* — lists of records. A list of table names or column names is a
    summary of what exists, which is the opposite of dumping a frame into the prompt.
    """
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
    for name in crew_run.results:
        assert f"crew_data_analyst:{name}" in recorded


def test_the_required_artifacts_exist(crew_run: SimpleNamespace) -> None:
    """§41 names these files exactly; the crew produces them under exactly those names."""
    for key in ("clean_data", "eda_report", "insights", "dataset_contract"):
        assert resolve_read(crew_run.ctx, REQUIRED_OUTPUTS[key]).is_file(), key


def test_verify_outputs_reports_only_what_is_missing(crew_run: SimpleNamespace) -> None:
    """Completeness is checked explicitly, not inferred from promote() (rule 8).

    No agent runs in this fixture, so the only output that may legitimately be absent is the
    agent-written review; the four deterministic artifacts must always be there.
    """
    assert set(verify_outputs(crew_run.ctx)) <= {"data_quality_review"}


def test_clean_data_is_byte_identical_to_a_direct_build_panel(
    crew_run: SimpleNamespace, tmp_path: Path
) -> None:
    """The crew path and the ``--no-llm`` path produce the same file, byte for byte (§40)."""
    direct_base = tmp_path / "direct"
    ctx = RunContext.start(mode="no-llm", base_dir=direct_base)
    cfg = load_cleaning_config()
    raw, _ = load_raw(RAW_SAMPLE)
    with ctx.step("direct"):
        clean_df, _ = clean_transactions(raw, cfg, ctx)
        returns = ctx.out(relative_path(paths.PROCESSED_DIR / RETURNS_FILENAME))
        build_panel(clean_df, pd.read_parquet(returns), cfg, ctx)
    close_log_handlers(ctx.run_id)

    from_crew = (crew_run.base / relative_path(paths.CLEAN_DATA)).read_bytes()
    from_direct = (direct_base / relative_path(paths.CLEAN_DATA)).read_bytes()
    assert from_crew == from_direct


def test_out_of_order_tool_call_is_reported_but_does_not_fail_the_run(
    bare_ctx: RunContext,
) -> None:
    """An agent mistake is recoverable; only a broken deterministic tool fails a run."""
    toolset = DataAnalystToolset(bare_ctx)
    decoded = json.loads(toolset._build_panel())
    assert "clean_transactions" in decoded["error"]
    assert bare_ctx.status == "running"
    assert bare_ctx.errors == []
    assert bare_ctx.steps == []


def test_read_table_rejects_a_name_that_is_not_a_table(crew_run: SimpleNamespace) -> None:
    decoded = json.loads(crew_run.toolset._read_table("top_products"))
    assert "error" in decoded
    assert decoded["available"], "the agent is told which tables exist"


def test_read_table_caps_the_rows_it_shows(crew_run: SimpleNamespace) -> None:
    name = sorted(crew_run.toolset.state.tables)[0]
    decoded = json.loads(crew_run.toolset._read_table(name))
    assert len(decoded["records"]) <= 20
    assert decoded["rows"] >= decoded["rows_shown"]


# --------------------------------------------------------------------------
# the narrative guard — the mechanism behind §38
# --------------------------------------------------------------------------
def test_guard_restores_the_deterministic_insights_on_an_unbacked_number(
    crew_run: SimpleNamespace,
) -> None:
    ctx = crew_run.ctx
    toolset = crew_run.toolset
    deterministic = toolset.state.deterministic_insights
    assert deterministic

    invented = f"# What the data says\n\n1. **Demand reached {UNBACKED_NUMBER} units** — act now.\n"
    decoded = json.loads(toolset._write_insights_narrative(invented))

    assert decoded["accepted"] is False
    assert UNBACKED_NUMBER in decoded["unmatched"]
    published = resolve_read(ctx, relative_path(paths.INSIGHTS)).read_text(encoding="utf-8")
    assert published == deterministic, "the agent's text must not survive"
    assert any(
        warning.startswith("insights narrative rejected:") for warning in ctx.warnings
    ), ctx.warnings


def test_guard_accepts_a_narrative_whose_numbers_are_all_in_tables(
    crew_run: SimpleNamespace,
) -> None:
    """The deterministic text is itself a valid agent answer — the guard is not a blanket veto."""
    toolset = crew_run.toolset
    rewritten = toolset.state.deterministic_insights.replace(
        "# What the data says", "# What the data says (reviewed)"
    )
    decoded = json.loads(toolset._write_insights_narrative(rewritten))

    assert decoded["accepted"] is True
    published = resolve_read(crew_run.ctx, relative_path(paths.INSIGHTS)).read_text(
        encoding="utf-8"
    )
    assert published == rewritten
    # Restore the deterministic file so later tests in this module see a clean state.
    toolset._write_insights_narrative(toolset.state.deterministic_insights)


def test_guard_rejects_an_unbacked_number_in_the_data_quality_review(
    crew_run: SimpleNamespace,
) -> None:
    decoded = json.loads(
        crew_run.toolset._write_data_quality_review(
            f"# Review\n\nThe extract holds {UNBACKED_NUMBER} suspicious rows.\n"
        )
    )
    assert "error" in decoded
    assert UNBACKED_NUMBER in decoded["unmatched"]
    assert not resolve_read(crew_run.ctx, REQUIRED_OUTPUTS["data_quality_review"]).is_file()


def test_a_backed_data_quality_review_is_published(crew_run: SimpleNamespace) -> None:
    profile = crew_run.toolset.state.profile
    rows = profile["raw_profile"]["rows"]
    markdown = f"# Review\n\nThe extract holds {rows} rows; nothing blocks cleaning.\n"

    decoded = json.loads(crew_run.toolset._write_data_quality_review(markdown))
    assert decoded["accepted"] is True
    written = resolve_read(crew_run.ctx, REQUIRED_OUTPUTS["data_quality_review"])
    assert written.read_text(encoding="utf-8") == markdown


def test_the_deterministic_review_passes_its_own_guard(crew_run: SimpleNamespace) -> None:
    """The fallback review must obey the rule it exists to enforce."""
    state = crew_run.toolset.state
    check = numbers_in_tables(deterministic_review(state), state.quality_tables())
    assert check.passed, check.summary()


# --------------------------------------------------------------------------
# the run wrapper: what happens around kickoff(), with no LLM in the loop
# --------------------------------------------------------------------------
class _StubCrew:
    """Stands in for the assembled crew: ``kickoff()`` calls the tools the agents are told to.

    This is the honest boundary for a test with no network. The three agents' only effect on the
    artifacts is *which tools they call in which order*, and that is exactly what this replays —
    without the narrative writers, so the fallback paths are the ones under test.
    """

    def __init__(self, builder: DataAnalystCrew, calls: bool = True) -> None:
        self.builder = builder
        self.calls = calls
        self.token_usage = None

    def kickoff(self) -> SimpleNamespace:
        if self.calls:
            toolset = self.builder.toolset
            toolset.state.raw_df = load_raw(RAW_SAMPLE)[0]
            for call in (
                toolset._profile_raw,
                toolset._detect_duplicates,
                toolset._list_nonproduct_codes,
                toolset._clean_transactions,
                toolset._build_panel,
                toolset._run_eda,
                toolset._write_contract,
            ):
                call()
        return SimpleNamespace(token_usage=None)


@pytest.fixture
def stubbed_kickoff(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace the LLM and the crew object; keep everything ``run_data_analyst_crew`` does."""
    captured: dict = {"calls": True}
    monkeypatch.setattr(crew_module, "make_llm", stub_llm)

    def fake_crew(self: DataAnalystCrew) -> _StubCrew:
        captured["builder"] = self
        return _StubCrew(self, calls=captured["calls"])

    monkeypatch.setattr(crew_module.DataAnalystCrew, "crew", fake_crew)
    return captured


def test_the_deterministic_review_is_written_when_no_agent_draft_survives(
    stubbed_kickoff: dict, tmp_path: Path
) -> None:
    """§6 expects data_quality_review.md to exist — the guard may reject every draft."""
    ctx = RunContext.start(mode="llm", base_dir=tmp_path)
    try:
        summary = crew_module.run_data_analyst_crew(ctx)
    finally:
        close_log_handlers(ctx.run_id)

    written = resolve_read(ctx, REQUIRED_OUTPUTS["data_quality_review"])
    assert written.is_file()
    assert written.read_text(encoding="utf-8") == deterministic_review(
        stubbed_kickoff["builder"].toolset.state
    )
    assert any(
        warning.startswith("data quality review rejected:") for warning in ctx.warnings
    ), ctx.warnings
    assert verify_outputs(ctx) == []
    assert set(summary["artifacts"]) == set(REQUIRED_OUTPUTS)


def test_a_crew_that_produced_nothing_fails_the_run(
    stubbed_kickoff: dict, tmp_path: Path
) -> None:
    """promote() only warns about a missing artifact (rule 8) — this must raise instead."""
    stubbed_kickoff["calls"] = False
    ctx = RunContext.start(mode="llm", base_dir=tmp_path)
    try:
        with pytest.raises(RuntimeError, match="without producing"):
            crew_module.run_data_analyst_crew(ctx)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# credentials: none in the source, and a clean exit without one
# --------------------------------------------------------------------------
def test_missing_credential_exits_2_without_starting_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§6: no key → exit 2, and no run log stranded at status 'running' (§8)."""
    for name in common.API_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    assert entry_point.main([]) == 2
    assert not (tmp_path / "artifacts").exists()


def test_make_llm_raises_without_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in common.API_KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(common.MissingAPIKeyError, match="--no-llm"):
        common.make_llm()


def test_the_crew_imports_without_a_credential() -> None:
    """Importing the crew must never require a key — CI has none (§6)."""
    assert callable(build_crew)
    assert review_path().name == "data_quality_review.md"


def test_no_credential_literal_in_the_crew_source() -> None:
    """§6 greps for the key prefix; keep the source clean of it entirely."""
    prefix = "sk" + "-"
    offenders = [
        path.relative_to(paths.PROJECT_ROOT).as_posix()
        for path in (paths.PROJECT_ROOT / "src" / "crews").rglob("*")
        if path.is_file() and prefix in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_the_crew_config_holds_no_secret() -> None:
    """Rule 11: config_snapshot is serialised verbatim into run_log.json — keys stay in env."""
    for path in (AGENTS_CONFIG, TASKS_CONFIG):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        flattened = json.dumps(config).lower()
        for forbidden in ("api_key", "openai_api_key", "token", "secret", "password"):
            assert forbidden not in flattened, f"{path.name} mentions {forbidden}"
