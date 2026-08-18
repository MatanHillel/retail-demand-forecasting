"""The Data Scientist Crew: three agents, four tasks, one run (US-26, PRD §36, §38).

The crew is assembled from ``config/agents.yaml`` and ``config/tasks.yaml`` and executed with
``Process.sequential`` — T1 (feature engineering) -> T2 (model training & back-test) -> T3
(evaluation, inventory, champion, reports) -> T3-narrative (rewrite the two reports' prose) is
fixed rather than negotiated. Everything numeric it produces comes from
:mod:`crews.data_scientist.tools`, which call the same ``pipeline`` functions a ``--no-llm`` run
calls — the agents choose when to call them and write the prose around them.

Two guarantees are enforced here rather than hoped for, mirroring
:mod:`crews.data_analyst.crew` (US-12):

* **The narrative can only lose.** ``evaluation_report.md`` and ``model_card.md`` exist in a
  deterministic, already-guarded form before any agent speaks (``write_reports_tool`` calls
  :func:`pipeline.reports.write_all_reports`, which itself runs ``numbers_in_tables`` and raises if
  it fails). An agent's rewrite is published only if its headings and tables are unchanged and
  every number in it is in a computed table; otherwise the deterministic text is restored into
  *this run's* destination (§8 of the issue) and the reason lands in ``run_log.json -> warnings[]``.
  After the crew has finished, both published files are checked once more against the tables and
  restored if either has drifted from what the tools last accepted.
* **A missing artifact fails the run.** ``promote()`` only *warns* when a registered path was never
  written (``docs/interfaces.md`` §6 rule 8), so completeness is checked explicitly, against the
  staged paths (rule 7), and a missing output raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from crewai import LLM, Agent, Crew, Process, Task

from crews.common import NarrativeGuard, make_llm, record_token_usage
from crews.data_scientist.tools import DataScientistToolset, relative_path, resolve_read
from pipeline import paths
from pipeline.run_context import RunContext

CONFIG_DIR: Path = Path(__file__).parent / "config"
AGENTS_CONFIG: Path = CONFIG_DIR / "agents.yaml"
TASKS_CONFIG: Path = CONFIG_DIR / "tasks.yaml"

#: The three agents of PRD §36, in the order they work.
AGENT_ORDER: tuple[str, ...] = (
    "feature_engineering_specialist",
    "forecasting_model_scientist",
    "model_evaluation_inventory_scientist",
)

#: T1 -> T2 -> T3 -> T3-narrative, paired with the agent that owns each (tasks.yaml assigns them).
TASK_ORDER: tuple[str, ...] = (
    "feature_engineering",
    "model_training",
    "evaluation_and_reports",
    "narrative_rewrite",
)

#: What a completed crew run must have produced (§6 of the issue), as repo-relative paths — the
#: four §41 artifacts this crew owns plus the three forecast files the acceptance criteria name.
REQUIRED_OUTPUTS: dict[str, Path] = {
    "features": relative_path(paths.FEATURES),
    "model": relative_path(paths.MODEL),
    "backtest_predictions": relative_path(paths.BACKTEST_PREDICTIONS),
    "latest_forecast": relative_path(paths.LATEST_FORECAST),
    "inventory_plan": relative_path(paths.INVENTORY_PLAN),
    "evaluation_report": relative_path(paths.EVALUATION_REPORT),
    "model_card": relative_path(paths.MODEL_CARD),
}

#: Prefix for the token counters this crew records in ``run_log.json -> metrics`` (§47).
METRICS_LABEL = "crew_data_scientist"

#: (label, canonical path, state attribute holding the accept/reject decision) for each narrative.
_NARRATIVES: tuple[tuple[str, Path, str], ...] = (
    ("evaluation_report", paths.EVALUATION_REPORT, "evaluation_decision"),
    ("model_card", paths.MODEL_CARD, "model_card_decision"),
)


def load_config(path: Path) -> dict[str, dict[str, Any]]:
    """Read one of the crew's YAML configuration files."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class DataScientistCrew:
    """Builds the crew for one run, holding the toolset the tasks and the guard both need.

    ``ctx`` is not optional — the tools write artifacts and need the run context to route every
    write through ``ctx.out(...)`` and to record each call as a step, so the builder needs it too
    (§8 of the issue).
    """

    def __init__(self, ctx: RunContext, llm: LLM | None = None) -> None:
        self.ctx = ctx
        self.llm = llm
        self.toolset = DataScientistToolset(ctx)
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
        """The assembled crew. Sequential by design - the tasks have a hard data dependency."""
        return Crew(
            agents=[self.agents[key] for key in AGENT_ORDER],
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # CrewAI's tool-result cache: a repeated call with identical arguments is served from
            # memory instead of re-billing the provider for the surrounding turn (§47 cost).
            cache=True,
        )


def build_crew(ctx: RunContext, llm: LLM | None = None) -> Crew:
    """Assemble the Data Scientist Crew for ``ctx``.

    ``llm=None`` means "let CrewAI resolve the model from the environment", which is what the
    tests use with a stub. :func:`run_data_scientist_crew` passes an explicit
    :func:`crews.common.make_llm`.
    """
    return DataScientistCrew(ctx, llm=llm).crew()


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


def run_data_scientist_crew(ctx: RunContext) -> dict[str, Any]:
    """Run the crew end to end and return a summary of what it produced.

    Raises:
        RuntimeError: if a required artifact is missing when the crew has finished. A crew run
            that skipped a tool must not be allowed to look successful.
    """
    builder = DataScientistCrew(ctx, llm=make_llm())
    crew = builder.crew()
    output = crew.kickoff()

    usage = record_token_usage(ctx, METRICS_LABEL, getattr(output, "token_usage", None))

    decisions = {
        attribute: _final_narrative_check(ctx, builder, label, canonical, attribute)
        for label, canonical, attribute in _NARRATIVES
    }

    missing = verify_outputs(ctx)
    if missing:
        raise RuntimeError(
            "the Data Scientist Crew finished without producing: "
            + ", ".join(f"{key} ({REQUIRED_OUTPUTS[key].as_posix()})" for key in sorted(missing))
        )

    champion = ctx.champion.get("champion") if ctx.champion else None
    eval_decision = decisions["evaluation_decision"]
    card_decision = decisions["model_card_decision"]
    summary = {
        "run_id": ctx.run_id,
        "tasks": [task.name for task in builder.tasks],
        "champion": champion,
        "evaluation_report_accepted": bool(eval_decision.accepted) if eval_decision else False,
        "model_card_accepted": bool(card_decision.accepted) if card_decision else False,
        "artifacts": {key: path.as_posix() for key, path in REQUIRED_OUTPUTS.items()},
        "token_usage": usage,
        "final_answer": str(output),
    }
    ctx.record_metrics({f"{METRICS_LABEL}_tasks": len(builder.tasks)})
    return summary


def _final_narrative_check(
    ctx: RunContext, builder: DataScientistCrew, label: str, canonical: Path, attribute: str
) -> Any:
    """Re-check one published narrative against the tables, restoring it if it now fails.

    The tool already guards each rewrite, so this normally confirms the decision it recorded. It
    exists for the case the tool never ran, or ran and something later touched the file: §38 is a
    property of what is *published*, not of the path that was taken to publish it.
    """
    state = builder.toolset.state
    deterministic = (
        state.deterministic_evaluation_report
        if attribute == "evaluation_decision"
        else state.deterministic_model_card
    )
    if deterministic is None:
        return getattr(state, attribute)

    relative = relative_path(canonical)
    published = resolve_read(ctx, relative)
    current = published.read_text(encoding="utf-8") if published.is_file() else ""
    if current == deterministic:
        # Nothing was rewritten, or the tool already restored it - the deterministic text has
        # already passed its own numbers_in_tables check inside write_all_reports().
        return getattr(state, attribute)

    guard = NarrativeGuard(label, state.narrative_tables(ctx), deterministic)
    decision = guard.publish(current, ctx.out(relative), ctx)
    setattr(state, attribute, decision)
    return decision
