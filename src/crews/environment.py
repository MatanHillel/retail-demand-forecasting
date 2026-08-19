"""What LLM mode costs and how it identifies itself — the crew module that imports no CrewAI.

Split out of :mod:`crews.common` (which imports ``crewai`` at module scope) for two callers that
must not pay for the LLM stack:

* ``python -m pipeline`` decides between LLM mode and ``--no-llm`` by asking whether a credential
  exists. Without the split, a run with no key would have to import the whole LLM stack to
  discover that it cannot use it, and ``docs/interfaces.md`` §6 rule 10 — ``src/pipeline/`` never
  imports CrewAI — would hold only by the width of a function body.
* :mod:`flow.llm_mode` prices each crew's token usage and enforces the cost cap. Pricing is
  arithmetic over ``model_config.yaml``; it needs no LLM class, and keeping it here means the
  cap can be evaluated before a crew is ever constructed.

Everything here is re-exported unchanged from :mod:`crews.common`, so
``from crews.common import require_api_key`` keeps working exactly as before.

**A credential value never leaves the environment.** Every function here answers with the *name*
of an environment variable. The name is safe to log; the value never is, and never enters this
process's own objects — LiteLLM reads it from the environment itself (``docs/interfaces.md``
§6 rule 11).
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

from pipeline.config import load_model_config

#: Credentials the crews accept, in the order they are tried. The provider is chosen entirely by
#: which of these is set — no provider name is hard-coded in ``src/`` (PRD §40).
API_KEY_VARIABLES: tuple[str, ...] = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")

#: Fallback name of the environment variable holding the LiteLLM model id. The name actually in
#: force is ``model_config.yaml -> llm.model_env_var`` (US-33); this is what
#: :func:`model_variable` falls back to if that setting is ever empty.
MODEL_VARIABLE = "CREWAI_LLM_MODEL"

#: Used when the model variable is unset. A model *id* is not a tuning constant — it selects the
#: provider, it does not enter a calculation — so it is not config (PRD §40 governs numbers).
DEFAULT_MODEL = "gpt-4o-mini"

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


#: Set this to any non-empty value to stop :func:`load_env_file` reading a ``.env`` at all. The
#: test suite sets it (``tests/conftest.py``), so a developer's real credential can never leak into
#: a test run and make a "no credential" test pass for the wrong reason.
ENV_FILE_DISABLE_VARIABLE = "RDF_DISABLE_DOTENV"


def load_env_file() -> str | None:
    """Load a local ``.env`` into the environment; return the file read, or ``None``.

    Call this **once, at a process entry point**, before anything asks for a credential. It is not
    called at import time on purpose: importing a module must never mutate the environment out
    from under a caller (``docs/interfaces.md`` §6 rule 10 is the same instinct).

    **A variable already present in the environment always wins.** ``override=False`` is
    python-dotenv's default and is passed explicitly here because it is the invariant that
    matters, not a default worth inheriting silently. The consequence is worth stating plainly:
    an exported ``OPENAI_API_KEY`` — including a stale one left over in a shell, a launcher or a
    Windows user profile — silently takes precedence over the value in ``.env``, and the file is
    then dead weight. That failure mode looks exactly like a bad key in ``.env`` (the provider
    answers 401), which is why :func:`api_key_variable` reports the variable *name* it used.

    The value is not read into this process: :func:`load_dotenv` writes it into ``os.environ`` and
    LiteLLM reads it back from there. Nothing here returns, logs or stores a credential.
    """
    if (os.environ.get(ENV_FILE_DISABLE_VARIABLE) or "").strip():
        return None
    path = find_dotenv(usecwd=True)
    if not path:
        return None
    load_dotenv(path, override=False)
    return path


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
    stranded at ``status: "running"`` for a run that never began.
    """
    name = api_key_variable()
    if name is None:
        raise MissingAPIKeyError()
    return name


def model_variable() -> str:
    """Name of the environment variable holding the model id (``llm.model_env_var``)."""
    return load_model_config().llm.model_env_var or MODEL_VARIABLE


def llm_model_name() -> str:
    """The LiteLLM model id from the environment, falling back to :data:`DEFAULT_MODEL`."""
    return (os.environ.get(model_variable()) or "").strip() or DEFAULT_MODEL


def estimate_cost_usd(
    *,
    prompt_tokens: int = 0,
    cached_prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str | None = None,
) -> float:
    """Estimated USD cost of a crew's token usage, priced from ``llm.pricing`` (PRD §47).

    Every rate is read from ``model_config.yaml`` — no price is ever written in code (PRD §40) —
    and a model id the table does not name is priced at the mandatory ``default`` entry. Cached
    prompt tokens are billed at their own lower rate and are assumed to be *part of*
    ``prompt_tokens``, which is how crewai reports them, so they are subtracted before the full
    rate is applied.

    It is an *estimate*: the provider's invoice is the authority. It exists to drive the cost cap
    and to make an LLM run's spend visible in ``run_log.json``, not to reconcile a bill.
    """
    price = load_model_config().llm.price_of(model or llm_model_name())
    cached = max(0, int(cached_prompt_tokens))
    uncached = max(0, int(prompt_tokens) - cached)
    total = (
        uncached * price.prompt_usd_per_1k
        + cached * price.cached_prompt_usd_per_1k
        + max(0, int(completion_tokens)) * price.completion_usd_per_1k
    ) / 1000
    return round(total, 6)
