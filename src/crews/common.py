"""Shared crew plumbing: the LLM factory, the narrative guard and cost logging (US-12, PRD §38).

Three things every crew in this project needs, and none of them may live under
:mod:`pipeline` — importing CrewAI there would put an LLM import into a ``--no-llm`` run
(``docs/interfaces.md`` §6 rule 10).

**The guard is the important part.** PRD §38 says no model ever computes a number, and
:class:`NarrativeGuard` is where that becomes mechanical: an agent may rewrite prose, but the
rewrite is published only if every number in it is already in a computed table. Otherwise the
deterministic version is restored and the run records *why*. The agent is never trusted, and
never asked to be trustworthy — it is checked.

The credential itself is never read into a variable that is passed around: :func:`make_llm`
confirms one is present and leaves LiteLLM to read the environment. Nothing here can therefore
leak a key into ``run_log.json`` or into a log line (§6 rule 11).
"""

from __future__ import annotations

import os
from typing import Any

from crewai import LLM
from pydantic import BaseModel, ConfigDict, Field

from pipeline.config import load_model_config
from pipeline.narrative import numbers_in_tables
from pipeline.run_context import RunContext

#: Credentials the crew accepts, in the order they are tried. The provider is chosen entirely by
#: which of these is set — no provider name is hard-coded in ``src/`` (PRD §40).
API_KEY_VARIABLES: tuple[str, ...] = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

#: Environment variable holding the LiteLLM model id CrewAI should use.
MODEL_VARIABLE = "CREWAI_LLM_MODEL"

#: Used when :data:`MODEL_VARIABLE` is unset. A model *id* is not a tuning constant — it selects
#: the provider, it does not enter a calculation — so it is not config (PRD §40 governs numbers).
DEFAULT_MODEL = "gpt-4o-mini"

#: Temperature 0 for every crew call: the numbers are already fixed by the tools, and the prose
#: should vary as little as the provider allows (§47 — LLM output is never bit-reproducible).
LLM_TEMPERATURE = 0.0

NO_API_KEY_MESSAGE = (
    "No LLM credential found. Set one of "
    + ", ".join(API_KEY_VARIABLES)
    + " in the environment (see .env.example) to run the crew.\n"
    "The whole pipeline also runs without any LLM, producing numerically identical artifacts:\n"
    "    python -m pipeline --no-llm"
)


class MissingAPIKeyError(RuntimeError):
    """No LLM credential is available, so the crew cannot run (exit code 2)."""

    def __init__(self, message: str = NO_API_KEY_MESSAGE) -> None:
        super().__init__(message)


def api_key_variable() -> str | None:
    """Name of the first credential variable that is set and non-empty, or ``None``.

    The *name* is returned, never the value — the name is safe to log, the value never is.
    """
    for name in API_KEY_VARIABLES:
        if (os.environ.get(name) or "").strip():
            return name
    return None


def require_api_key() -> str:
    """Return the name of the credential variable in use, or raise :class:`MissingAPIKeyError`.

    Call this *before* ``RunContext.start()``: a run that stops here should leave no run log
    stranded at ``status: "running"`` (§8 of the issue).
    """
    name = api_key_variable()
    if name is None:
        raise MissingAPIKeyError()
    return name


def llm_model_name() -> str:
    """The LiteLLM model id from the environment, falling back to :data:`DEFAULT_MODEL`."""
    return (os.environ.get(MODEL_VARIABLE) or "").strip() or DEFAULT_MODEL


def make_llm(*, seed: int | None = None, temperature: float = LLM_TEMPERATURE) -> LLM:
    """Build the LLM every crew agent shares.

    ``seed`` comes from ``model_config.yaml`` so it is the same single global seed the rest of
    the project uses (PRD §40). Providers that honour it become more repeatable; those that do
    not are unaffected. The credential is deliberately *not* passed as an argument — LiteLLM
    reads it from the environment, so the value never enters this process's own objects.
    """
    require_api_key()
    resolved_seed = load_model_config().seed if seed is None else seed
    return LLM(model=llm_model_name(), temperature=temperature, seed=resolved_seed)


class GuardDecision(BaseModel):
    """What the guard decided about one candidate narrative, and the text to publish."""

    model_config = ConfigDict(extra="forbid")

    label: str
    accepted: bool
    text: str
    checked: int = 0
    unmatched: list[str] = Field(default_factory=list)

    @property
    def message(self) -> str:
        """The exact wording that goes to ``ctx.warn`` / the log (§2 of the issue)."""
        if self.accepted:
            return (
                f"{self.label} narrative accepted: all {self.checked} number(s) are backed by a "
                "computed table"
            )
        return f"{self.label} narrative rejected: {', '.join(self.unmatched)}"


class NarrativeGuard:
    """Publishes an LLM narrative only when every number in it is in a computed table (§38).

    ``fallback`` is the deterministic text to fall back to — for ``insights.md`` that is the
    US-11 output, which :func:`pipeline.eda.insights.generate_insights` has *already* put through
    this same check, so the fallback is known-good rather than merely assumed to be.
    """

    def __init__(self, label: str, tables: dict[str, Any], fallback: str) -> None:
        self.label = label
        self.tables = tables
        self.fallback = fallback

    def review(self, candidate: str) -> GuardDecision:
        """Check ``candidate`` and return the decision, without writing anything."""
        check = numbers_in_tables(candidate, self.tables)
        return GuardDecision(
            label=self.label,
            accepted=check.passed,
            text=candidate if check.passed else self.fallback,
            checked=check.checked,
            unmatched=list(check.unmatched),
        )

    def publish(self, candidate: str, destination: Any, ctx: RunContext) -> GuardDecision:
        """Check ``candidate``, write whichever text won to ``destination``, record the outcome.

        ``destination`` is a path already resolved through ``ctx.out(...)`` by the caller, so
        under staging the restored deterministic text lands in *this* run's staging directory and
        never republishes the previous run's file (§8 of the issue).
        """
        decision = self.review(candidate)
        destination.write_text(decision.text, encoding="utf-8", newline="\n")
        if decision.accepted:
            ctx.logger.info(decision.message)
        else:
            ctx.warn(decision.message)
        return decision


def record_token_usage(ctx: RunContext, label: str, usage: Any) -> dict[str, int]:
    """Record a crew's token usage on the run (§47 cost logging).

    ``ctx.record_metrics`` is the only recording API — there is no token-specific one — so the
    counters are prefixed with the crew's name and merged into ``run_log.json → metrics``.
    Returns what was recorded, so a caller can print it.
    """
    counters = ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests")
    recorded = {
        f"{label}_{counter}": int(getattr(usage, counter, 0) or 0)
        for counter in counters
        if usage is not None
    }
    if recorded:
        ctx.record_metrics(recorded)
    return recorded
