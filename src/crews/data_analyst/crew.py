"""The Data Analyst Crew: three agents, three tasks, one run (US-12, PRD §34, §35, §38).

The crew is assembled from ``config/agents.yaml`` and ``config/tasks.yaml`` and executed with
``Process.sequential``, so T1 → T2 → T3 is fixed rather than negotiated. Everything numeric it
produces comes from :mod:`crews.data_analyst.tools`, which call the same ``pipeline`` functions a
``--no-llm`` run calls — the agents choose when to call them and write the prose around them.

Two guarantees are enforced here rather than hoped for:

* **The narrative can only lose.** ``insights.md`` exists in a deterministic, already-guarded form
  before any agent speaks. An agent's rewrite is published only if every number in it is in a
  computed table; otherwise the deterministic text is restored into *this run's* destination
  (§8 of the issue) and the reason lands in ``run_log.json → warnings[]``. After the crew has
  finished, the published file is checked once more against the tables — if the crew somehow left
  something unbacked on disk, it is replaced before the run is allowed to succeed.
* **A missing artifact fails the run.** ``promote()`` only *warns* when a registered path was
  never written (``docs/interfaces.md`` §6 rule 8), so completeness is checked explicitly, against
  the staged paths (rule 7), and a missing output raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from crews.common import NarrativeGuard, make_llm, record_token_usage
from crews.data_analyst.tools import (
    DataAnalystToolset,
    deterministic_review,
    relative_path,
    resolve_read,
    review_path,
)
from pipeline import paths
from pipeline.run_context import RunContext

CONFIG_DIR: Path = Path(__file__).parent / "config"
AGENTS_CONFIG: Path = CONFIG_DIR / "agents.yaml"
TASKS_CONFIG: Path = CONFIG_DIR / "tasks.yaml"

#: The three agents of PRD §35, in the order they work.
AGENT_ORDER: tuple[str, ...] = (
    "data_quality_analyst",
    "data_preparation_analyst",
    "business_eda_analyst",
)

#: T1 → T2 → T3. Paired with the agent that owns each, exactly as ``tasks.yaml`` assigns them.
TASK_ORDER: tuple[str, ...] = ("quality_profiling", "cleaning_and_panel", "eda_and_contract")

#: What a completed crew run must have produced (§6 of the issue), as repo-relative paths.
REQUIRED_OUTPUTS: dict[str, Path] = {
    "data_quality_review": relative_path(review_path()),
    "clean_data": relative_path(paths.CLEAN_DATA),
    "eda_report": relative_path(paths.EDA_REPORT),
    "insights": relative_path(paths.INSIGHTS),
    "dataset_contract": relative_path(paths.DATASET_CONTRACT),
}

#: Prefix for the token counters this crew records in ``run_log.json → metrics`` (§47).
METRICS_LABEL = "crew_data_analyst"


def load_config(path: Path) -> dict[str, dict[str, Any]]:
    """Read one of the crew's YAML configuration files."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class DataAnalystCrew:
    """Builds the crew for one run, holding the toolset the tasks and the guard both need.

    ``ctx`` is not optional. The tools write artifacts, so they need the run context to route
    every write through ``ctx.out(...)`` and to record each call as a step — which means the
    builder needs it too (§8 of the issue).
    """

    def __init__(self, ctx: RunContext, llm: LLM | None = None) -> None:
        self.ctx = ctx
        self.llm = llm
        self.toolset = DataAnalystToolset(ctx)
        self.agents_config = load_config(AGENTS_CONFIG)
        self.tasks_config = load_config(TASKS_CONFIG)
        self.agents: dict[str, Agent] = {key: self._agent(key) for key in AGENT_ORDER}
        self.tasks: list[Task] = [self._task(key) for key in TASK_ORDER]

    def _agent(self, key: str) -> Agent:
        config = self.agents_config[key]
        return Agent(
            role=config["role"].strip(),
            goal=config["goal"].strip(),
            backstory=config["backstory"].strip(),
            tools=self.toolset.by_agent[key],
            llm=self.llm,
            allow_delegation=bool(config["allow_delegation"]),
            verbose=bool(config["verbose"]),
            max_iter=int(config["max_iter"]),
        )

    def _task(self, key: str) -> Task:
        config = self.tasks_config[key]
        return Task(
            name=key,
            description=config["description"].strip(),
            expected_output=config["expected_output"].strip(),
            agent=self.agents[config["agent"]],
        )

    def crew(self) -> Crew:
        """The assembled crew. Sequential by design — the tasks have a hard data dependency."""
        return Crew(
            agents=[self.agents[key] for key in AGENT_ORDER],
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # CrewAI's tool-result cache: a repeated call with identical arguments is served from
            # memory instead of re-billing the provider for the surrounding turn (§47 cost). It
            # caches tool *output*, not prompts; nothing here re-runs a deterministic tool twice.
            cache=True,
        )


def build_crew(ctx: RunContext, llm: LLM | None = None) -> Crew:
    """Assemble the Data Analyst Crew for ``ctx``.

    ``llm=None`` means "let CrewAI resolve the model from the environment", which is what the
    tests use with a stub. :func:`run_data_analyst_crew` passes an explicit
    :func:`crews.common.make_llm`.
    """
    return DataAnalystCrew(ctx, llm=llm).crew()


def verify_outputs(ctx: RunContext) -> list[str]:
    """Names of the required outputs that are not on disk for *this* run.

    Resolved staged-first: before promotion the final locations still hold the previous run's
    files, so checking them would pass on stale leftovers (``docs/interfaces.md`` §6 rule 7).
    """
    return [
        key
        for key, relative in REQUIRED_OUTPUTS.items()
        if not resolve_read(ctx, relative).is_file()
    ]


def run_data_analyst_crew(ctx: RunContext) -> dict[str, Any]:
    """Run the crew end to end and return a summary of what it produced.

    Raises:
        RuntimeError: if a required artifact is missing when the crew has finished. A crew run
            that skipped a tool must not be allowed to look successful.
    """
    builder = DataAnalystCrew(ctx, llm=make_llm())
    crew = builder.crew()
    output = crew.kickoff()

    usage = record_token_usage(ctx, METRICS_LABEL, getattr(output, "token_usage", None))

    decision = _final_narrative_check(ctx, builder)
    _ensure_review(ctx, builder)

    missing = verify_outputs(ctx)
    if missing:
        raise RuntimeError(
            "the Data Analyst Crew finished without producing: "
            + ", ".join(f"{key} ({REQUIRED_OUTPUTS[key].as_posix()})" for key in sorted(missing))
        )

    summary = {
        "run_id": ctx.run_id,
        "tasks": [task.name for task in builder.tasks],
        "insights_narrative_accepted": bool(decision.accepted) if decision else False,
        "insights_unmatched": list(decision.unmatched) if decision else [],
        "artifacts": {key: path.as_posix() for key, path in REQUIRED_OUTPUTS.items()},
        "token_usage": usage,
        "final_answer": str(output),
    }
    ctx.record_metrics({f"{METRICS_LABEL}_tasks": len(builder.tasks)})
    return summary


def _final_narrative_check(ctx: RunContext, builder: DataAnalystCrew) -> Any:
    """Re-check the published ``insights.md`` against the tables, restoring it if it fails.

    The tool already guards each rewrite, so this normally confirms the decision it recorded. It
    exists for the case the tool never ran, or ran and something later touched the file: §38 is a
    property of what is *published*, not of the path that was taken to publish it.
    """
    state = builder.toolset.state
    if state.deterministic_insights is None:
        return state.insights_decision

    published = resolve_read(ctx, relative_path(paths.INSIGHTS))
    current = published.read_text(encoding="utf-8") if published.is_file() else ""
    guard = NarrativeGuard("insights", state.tables, state.deterministic_insights)
    if current == state.deterministic_insights:
        # Nothing was rewritten, or the tool already restored it — the deterministic text has
        # been through this same check inside generate_insights(), so re-running it is noise.
        return state.insights_decision

    decision = guard.publish(current, ctx.out(relative_path(paths.INSIGHTS)), ctx)
    state.insights_decision = decision
    return decision


def _ensure_review(ctx: RunContext, builder: DataAnalystCrew) -> None:
    """Write the deterministic ``data_quality_review.md`` if the agent never published one."""
    state = builder.toolset.state
    relative = relative_path(review_path())
    if state.review_written or resolve_read(ctx, relative).is_file():
        return
    if state.profile is None:
        return  # T1 never ran at all; verify_outputs() reports the missing artifact.
    ctx.warn(
        "data quality review rejected: no agent draft passed the numbers-in-tables guard — the "
        "deterministic review was written instead"
    )
    ctx.out(relative).write_text(deterministic_review(state), encoding="utf-8", newline="\n")
    ctx.record_artifact("data_quality_review", relative)
