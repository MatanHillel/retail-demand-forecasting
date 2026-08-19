"""``python -m crews.data_analyst`` — run the Data Analyst Crew standalone (US-12 §6).

Exit codes match the rest of the project: ``0`` success, ``2`` a graceful stop (no credential, or
a validation failure), ``1`` an unexpected exception.

The credential is checked **before** ``RunContext.start()`` on purpose. Starting a run and then
discovering there is no key would leave ``run_log.json`` reporting a run stuck at
``status: "running"`` that never actually began (§8 of the issue).
"""

from __future__ import annotations

import json
import sys

from crews.common import MissingAPIKeyError, llm_model_name, require_api_key
from crews.data_analyst.crew import run_data_analyst_crew
from crews.environment import load_env_file
from pipeline.run_context import RunContext
from pipeline.validation import FlowValidationError


def main(argv: list[str] | None = None) -> int:
    """Run the crew, returning the process exit code."""
    # Before the credential check, so a key that lives only in .env is found. A variable already
    # set in the environment still wins (load_env_file, override=False).
    load_env_file()
    try:
        credential = require_api_key()
    except MissingAPIKeyError as error:
        print(str(error), file=sys.stderr)
        return 2

    ctx = RunContext.start(mode="llm")
    # Write the log immediately: a run that dies before finish() would otherwise leave the
    # *previous* run's log on disk, still saying "success" (docs/interfaces.md §3).
    ctx.write_run_log()
    # The variable name is safe to log; its value never is, and is never read into this process.
    ctx.logger.info(f"LLM credential from {credential}; model {llm_model_name()}")

    try:
        summary = run_data_analyst_crew(ctx)
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()

    verdict = (
        "accepted"
        if summary["insights_narrative_accepted"]
        else "rejected — the deterministic version was restored"
    )
    print(
        f"run:      {summary['run_id']}\n"
        f"tasks:    {' -> '.join(summary['tasks'])}\n"
        f"insights: narrative {verdict}\n"
        f"tokens:   {json.dumps(summary['token_usage'])}\n"
        "artifacts:\n"
        + "\n".join(f"  {key}: {path}" for key, path in sorted(summary["artifacts"].items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
