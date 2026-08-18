"""Pydantic ``FlowState`` — the Flow's JSON-serialisable notebook (US-31, PRD §37 State).

``FlowState`` mirrors what the run has done: which files were produced, whether the validation
checkpoints passed, which model won, and whether the run is still going. It never carries a
DataFrame — the frames the steps hand to each other live in :class:`flow.steps.FlowData`, an
in-memory carrier on the Flow instance — and it is never the source of ``run_log.json``: that file
is written from the :class:`~pipeline.run_context.RunContext` alone (one source, US-02), while the
state mirrors the same facts for the Flow's own routing and for tests.

Every field has a default because ``crewai``'s ``Flow[FlowState]`` instantiates the class with no
arguments; the real values are filled in by :class:`flow.main.RetailForecastFlow` right after the
context starts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ValidationFlags(BaseModel):
    """One flag per validation checkpoint (§37): ``None`` = not reached yet, else pass/fail."""

    model_config = ConfigDict(extra="forbid")

    raw_schema: bool | None = None
    contract: bool | None = None
    features: bool | None = None
    leakage: bool | None = None
    artifacts: bool | None = None


class FlowState(BaseModel):
    """State of one Flow run (PRD §37): run id, timestamp, data hash, artifact paths, validation
    flags, metrics, champion, errors, status and the step currently executing."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    run_id: str = ""
    started_at: str = ""
    mode: Literal["no-llm", "llm"] = "no-llm"
    #: ``{file, sha256, rows, columns}`` — mirrored from ``ctx.data`` after dataset intake.
    data: dict[str, Any] = Field(default_factory=dict)
    #: Required §41 artifact name -> repo-relative path, recorded by step 9.
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    validation: ValidationFlags = Field(default_factory=ValidationFlags)
    #: Mirrored from ``ctx.metrics`` at the end of the run.
    metrics: dict[str, Any] = Field(default_factory=dict)
    #: LLM mode only (US-33): ``{crew1_status, crew2_status, model, tokens, cost_usd,
    #: max_cost_usd, narrative_accepted, guard_restored}``. Empty in ``--no-llm`` mode. The same
    #: dict reaches ``run_log.json`` as ``metrics.llm`` — ``RunContext`` is ``extra="forbid"``
    #: with a published field list, so there is no top-level ``llm`` key to write it to.
    llm: dict[str, Any] = Field(default_factory=dict)
    #: The §20 decision, mirrored from ``ctx.champion`` (set by ``select_champion``).
    champion: dict[str, Any] | None = None
    #: Mirrored from ``ctx.errors`` — ``{step, type, message, traceback}`` per failure.
    errors: list[dict[str, Any]] = Field(default_factory=list)
    status: Literal["running", "success", "failed"] = "running"
    current_step: str = ""
