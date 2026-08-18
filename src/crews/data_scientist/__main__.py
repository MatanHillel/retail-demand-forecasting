"""``python -m crews.data_scientist`` — run the Data Scientist Crew standalone (US-26 §8).

Exit codes match the rest of the project: ``0`` success, ``2`` a graceful stop (no credential, or
a validation failure), ``1`` an unexpected exception.

The credential is checked **before** ``RunContext.start()`` on purpose. Starting a run and then
discovering there is no key would leave ``run_log.json`` reporting a run stuck at
``status: "running"`` that never actually began (§8 of the issue interface corrections). For the
same reason the context is started with ``mode="llm"`` explicitly rather than the default
``"no-llm"`` — this run is not free of an LLM, and the US-34 no-llm/llm comparison depends on the
mode being recorded honestly.
"""

from __future__ import annotations

import json
import sys

from crews.common import MissingAPIKeyError, llm_model_name, require_api_key
from crews.data_scientist.crew import run_data_scientist_crew
from pipeline.run_context import RunContext
from pipeline.validation import FlowValidationError


def main(argv: list[str] | None = None) -> int:
    """Run the crew, returning the process exit code."""
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
        summary = run_data_scientist_crew(ctx)
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()

    restored = "rejected — deterministic version restored"
    eval_verdict = "accepted" if summary["evaluation_report_accepted"] else restored
    card_verdict = "accepted" if summary["model_card_accepted"] else restored
    print(
        f"run:       {summary['run_id']}\n"
        f"tasks:     {' -> '.join(summary['tasks'])}\n"
        f"champion:  {summary['champion']}\n"
        f"evaluation_report: {eval_verdict}\n"
        f"model_card:        {card_verdict}\n"
        f"tokens:    {json.dumps(summary['token_usage'])}\n"
        "artifacts:\n"
        + "\n".join(f"  {key}: {path}" for key, path in sorted(summary["artifacts"].items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
