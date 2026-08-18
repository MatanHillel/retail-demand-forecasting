"""``python -m pipeline`` — run the whole forecasting pipeline end to end (US-31, US-33, §37).

Two modes, ten identical deterministic steps:

* ``--no-llm`` runs those ten steps with no LLM involvement at all — the mode CI uses (§42) and
  the reproducibility baseline (§40).
* the default (or ``--llm``) additionally kicks the Data Analyst Crew off after step 3 and the
  Data Scientist Crew's narrative task off after step 9 (US-33). The crews review results and
  write prose; every number still comes from the same deterministic tools, so the artifacts are
  numerically identical either way (§38).

Without a credential the default falls back to ``--no-llm`` and says so, exiting 0; ``--llm``
demands one and exits 2 without it. The check happens **before** the run context is started, so a
run that stops here leaves no run log stranded at ``status: "running"`` for a run that never
began, and the context is started with the *effective* mode — ``RunMode`` has exactly two values
and ``run_log.json`` must report what actually ran.

Exit codes match the rest of the project: ``0`` success, ``2`` a graceful validation stop (the
``FLOW STOPPED: …`` message goes to stderr), ``1`` an unexpected exception.

Every CrewAI-touching import happens inside :func:`main`, never at module scope: ``src/pipeline/``
stays free of CrewAI imports (``docs/interfaces.md`` §6 rule 10), so importing any ``pipeline.*``
module — in tests, in the app — never pulls the LLM stack in. The credential check itself costs no
CrewAI import at all: :mod:`crews.environment` is deliberately free of one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline import paths


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Run the retail demand forecasting pipeline (PRD §37, ten steps).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--no-llm",
        action="store_true",
        help="run fully deterministically, without any LLM involvement (CI mode, §37)",
    )
    mode.add_argument(
        "--llm",
        action="store_true",
        help="require LLM mode: exit 2 rather than falling back when no API key is set (§37)",
    )
    parser.add_argument(
        "--max-llm-cost-usd",
        type=float,
        default=None,
        metavar="USD",
        help="override model_config.yaml -> llm.max_cost_usd for this run; reaching the cap "
        "aborts the narrative step, not the run (§47)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="skip the hyper-parameter grid search and use the parameters in model_config.yaml",
    )
    parser.add_argument(
        "--keep-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="archive a failed run's staging tree under logs/failed_runs/<run_id>/ for debugging "
        "(default); --no-keep-failed deletes it instead (§39)",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="explicit raw CSV to run on (default: the canonical download in data/raw/)",
    )
    source.add_argument(
        "--sample",
        action="store_true",
        help="run on tests/fixtures/raw_sample.csv (the CI fixture, US-03)",
    )
    return parser.parse_args(argv)


def _resolve_mode(args: argparse.Namespace) -> str | None:
    """The mode this run will actually use, or ``None`` when ``--llm`` was asked for without a key.

    Printing happens here because the message is part of the contract: a fallback must be
    announced (§2 of US-33), and a hard failure must point at the alternative that works.
    """
    if args.no_llm:
        return "no-llm"

    # crews.environment imports no CrewAI (module docstring), so the key check is free.
    from crews.environment import NO_API_KEY_MESSAGE, api_key_variable, llm_model_name

    credential = api_key_variable()
    if credential is None:
        if args.llm:
            print(NO_API_KEY_MESSAGE, file=sys.stderr)
            return None
        print("LLM mode requires an API key — falling back to --no-llm")
        return "no-llm"
    # The variable NAME is safe to print; its value never is, and is never read into this process.
    print(f"LLM mode: credential from {credential}, model {llm_model_name()}")
    return "llm"


def main(argv: list[str] | None = None) -> int:
    """Run the Flow, returning the process exit code."""
    args = _parse_args(argv)
    mode = _resolve_mode(args)
    if mode is None:
        return 2

    raw_path = paths.FIXTURES_DIR / "raw_sample.csv" if args.sample else args.raw
    if raw_path is not None and not Path(raw_path).is_file():
        print(f"raw file not found: {raw_path}", file=sys.stderr)
        return 2

    # Deferred import: flow.main imports crewai, which must never load at pipeline.* module scope
    # (docs/interfaces.md §6 rule 10).
    from flow.main import run_flow

    state, _ = run_flow(
        mode=mode,
        raw_path=raw_path,
        skip_tuning=args.skip_tuning,
        keep_failed=args.keep_failed,
        max_cost_usd=args.max_llm_cost_usd,
    )

    if state.status == "success":
        return 0

    last_error = state.errors[-1] if state.errors else {}
    message = last_error.get("message") or "pipeline failed"
    print(message, file=sys.stderr)
    # A validation checkpoint stopping the run is a graceful exit (2); anything else crashed (1).
    return 2 if last_error.get("type") == "FlowValidationError" else 1


if __name__ == "__main__":
    raise SystemExit(main())
