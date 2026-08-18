"""Shared crew plumbing: the LLM factory, the narrative guard and cost logging (US-12, US-33).

Four things every crew in this project needs, and none of them may live under :mod:`pipeline` —
importing CrewAI there would put an LLM import into a ``--no-llm`` run (``docs/interfaces.md``
§6 rule 10).

**The guard is the important part.** PRD §38 says no model ever computes a number, and
:class:`NarrativeGuard` is where that becomes mechanical: an agent may rewrite prose, but the
rewrite is published only if every number in it is already in a computed table. Otherwise the
deterministic version is restored and the run records *why*. The agent is never trusted, and
never asked to be trustworthy — it is checked.

**Cost is measured, not assumed** (PRD §47, US-33). :func:`record_token_usage` records what a
crew actually spent — including ``cached_prompt_tokens``, the counter that evidences prompt
caching — and :func:`estimate_cost_usd` prices it from ``model_config.yaml -> llm.pricing``. No
price is written in code (PRD §40), and the estimate is what the Flow's cost cap acts on.

The credential itself is never read into a variable that is passed around: :func:`make_llm`
confirms one is present and leaves LiteLLM to read the environment. Nothing here can therefore
leak a key into ``run_log.json`` or into a log line (§6 rule 11). The credential check, the model
id and the pricing arithmetic live in :mod:`crews.environment`, which imports no CrewAI so that
the CLI's mode switch and the Flow's cost cap can use them without loading the LLM stack; all of
them are re-exported here unchanged.
"""

from __future__ import annotations

from typing import Any

from crewai import LLM
from pydantic import BaseModel, ConfigDict, Field

from crews.environment import (
    API_KEY_VARIABLES,
    DEFAULT_MODEL,
    MODEL_VARIABLE,
    NO_API_KEY_MESSAGE,
    MissingAPIKeyError,
    api_key_variable,
    estimate_cost_usd,
    llm_model_name,
    model_variable,
    require_api_key,
)
from pipeline.config import load_model_config
from pipeline.narrative import numbers_in_tables
from pipeline.run_context import RunContext

__all__ = [
    "API_KEY_VARIABLES",
    "DEFAULT_MODEL",
    "LLM_TEMPERATURE",
    "MODEL_VARIABLE",
    "NO_API_KEY_MESSAGE",
    "TOKEN_COUNTERS",
    "GuardDecision",
    "MissingAPIKeyError",
    "NarrativeGuard",
    "api_key_variable",
    "estimate_cost_usd",
    "llm_model_name",
    "make_llm",
    "model_variable",
    "record_token_usage",
    "require_api_key",
]

#: Temperature 0 for every crew call: the numbers are already fixed by the tools, and the prose
#: should vary as little as the provider allows (§47 — LLM output is never bit-reproducible).
LLM_TEMPERATURE = 0.0

#: The usage counters recorded per crew. ``cached_prompt_tokens`` is the one that evidences
#: prompt caching, and :func:`crews.environment.estimate_cost_usd` prices it at its own lower
#: rate.
TOKEN_COUNTERS: tuple[str, ...] = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "successful_requests",
)


def make_llm(*, seed: int | None = None, temperature: float = LLM_TEMPERATURE) -> LLM:
    """Build the LLM every crew agent shares (PRD §47).

    ``seed`` comes from ``model_config.yaml`` so it is the same single global seed the rest of
    the project uses (PRD §40). Providers that honour it become more repeatable; those that do
    not are unaffected. The credential is deliberately *not* passed as an argument — LiteLLM
    reads it from the environment, so the value never enters this process's own objects.

    **Prompt caching.** ``crewai.LLM`` forwards any extra keyword straight to
    ``litellm.completion``, so ``llm.extra_params`` in ``model_config.yaml`` is the hook for a
    provider that needs an explicit caching header. The default provider (OpenAI) caches long
    prompt prefixes automatically and reports the hits back as ``cached_prompt_tokens``, which
    :func:`record_token_usage` records and :func:`estimate_cost_usd` prices at the cached rate —
    so the saving is visible in ``run_log.json`` either way, with no configuration at all.
    """
    require_api_key()
    model_cfg = load_model_config()
    resolved_seed = model_cfg.seed if seed is None else seed
    return LLM(
        model=llm_model_name(),
        temperature=temperature,
        seed=resolved_seed,
        **dict(model_cfg.llm.extra_params),
    )


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
    counters are prefixed with the crew's name and merged into ``run_log.json -> metrics``.
    Returns what was recorded, so a caller can price it or print it.
    """
    recorded = {
        f"{label}_{counter}": int(getattr(usage, counter, 0) or 0)
        for counter in TOKEN_COUNTERS
        if usage is not None
    }
    if recorded:
        ctx.record_metrics(recorded)
    return recorded
