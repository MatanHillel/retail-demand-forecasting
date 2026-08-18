"""``python -m pipeline`` — run the whole forecasting pipeline end to end (US-31, PRD §37).

``--no-llm`` runs the ten deterministic Flow steps with no LLM involvement at all — the mode CI
uses (§42) and the reproducibility baseline (§40). Without the flag the command *will* become the
LLM mode of US-33 (crews reviewing the results and writing narrative around the same numbers);
until that lands it prints a notice and behaves exactly like ``--no-llm``.

Exit codes match the rest of the project: ``0`` success, ``2`` a graceful validation stop (the
``FLOW STOPPED: …`` message goes to stderr), ``1`` an unexpected exception.

The CrewAI import happens inside :func:`main`, not at module scope: ``src/pipeline/`` stays free
of CrewAI imports (``docs/interfaces.md`` §6 rule 10), so importing any ``pipeline.*`` module —
in tests, in the app — never pulls the LLM stack in.
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
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="run fully deterministically, without any LLM involvement (CI mode, §37)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="skip the hyper-parameter grid search and use the parameters in model_config.yaml",
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


def main(argv: list[str] | None = None) -> int:
    """Run the Flow, returning the process exit code."""
    args = _parse_args(argv)
    if not args.no_llm:
        print(
            "LLM mode (crew review + narrative) arrives with US-33; "
            "running the deterministic pipeline (--no-llm behaviour)."
        )

    raw_path = paths.FIXTURES_DIR / "raw_sample.csv" if args.sample else args.raw
    if raw_path is not None and not Path(raw_path).is_file():
        print(f"raw file not found: {raw_path}", file=sys.stderr)
        return 2

    # Deferred import: flow.main imports crewai, which must never load at pipeline.* module scope
    # (docs/interfaces.md §6 rule 10).
    from flow.main import run_flow

    state, _ = run_flow(mode="no-llm", raw_path=raw_path, skip_tuning=args.skip_tuning)

    if state.status == "success":
        return 0

    last_error = state.errors[-1] if state.errors else {}
    message = last_error.get("message") or "pipeline failed"
    print(message, file=sys.stderr)
    # A validation checkpoint stopping the run is a graceful exit (2); anything else crashed (1).
    return 2 if last_error.get("type") == "FlowValidationError" else 1


if __name__ == "__main__":
    raise SystemExit(main())
