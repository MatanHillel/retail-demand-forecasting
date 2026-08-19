"""Assert that a deliberately broken input stopped the pipeline *gracefully* (US-35, PRD §39).

Run by the ``failure-path`` CI job after the pipeline has been pointed at
``tests/fixtures/raw_sample_missing_quantity.csv``::

    python scripts/ci_check_failure_run.py ci_out_fail <exit-code> <stderr-file>

A crash and a graceful stop both leave a non-zero exit code, so the job proves the difference:

* the exit code is **2** — the graceful-validation-stop code, not ``1`` (an unexpected exception)
  and not ``0``;
* stderr carries the exact ``FLOW STOPPED: Missing required column Quantity`` wording, which is
  where the prefixed message actually appears — ``FlowValidationError`` prepends
  ``FLOW STOPPED:`` to ``str(exc)``, so the *report's* own violation message has no prefix;
* ``validation_report.json`` says ``passed: false`` and carries the unprefixed violation message,
  with a ``run_id`` matching ``run_log.json``'s. The report is written on success *and* failure and
  is never cleared between runs, so a reader that skips the run-id comparison can report the
  previous run's reason (``docs/interfaces.md`` §6 rule 6);
* ``run_log.json`` says ``status == "failed"`` **exactly** — ``running`` is what a CI timeout
  leaves behind, and a ``!= "success"`` test would call that a pass (the issue's §8).

Exits 0 when the graceful stop is exactly as specified, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import paths

#: What a graceful validation stop exits with (US-32): 0 success, 1 crash, 2 graceful stop.
EXPECTED_EXIT_CODE = 2

#: The violation's own message, as ``pipeline.download`` writes it into the report.
EXPECTED_VIOLATION_MESSAGE = "Missing required column Quantity"

#: The full line the user sees, assembled by ``FlowValidationError`` and printed to stderr.
EXPECTED_STDERR_LINE = f"FLOW STOPPED: {EXPECTED_VIOLATION_MESSAGE}"

#: ``Violation.rule`` the raw-schema check stamps on it.
EXPECTED_RULE = "required_columns"


def _rebase(out_root: Path, canonical: Path) -> Path:
    return out_root / canonical.relative_to(paths.PROJECT_ROOT)


def check(out_root: Path, exit_code: int, stderr_text: str) -> list[str]:
    """Return a list of failures; empty means the stop was graceful and correctly reported."""
    failures: list[str] = []

    if exit_code != EXPECTED_EXIT_CODE:
        failures.append(
            f"exit code was {exit_code}, expected {EXPECTED_EXIT_CODE} "
            "(2 = graceful validation stop; 1 = unexpected exception; 0 = success)"
        )

    if EXPECTED_STDERR_LINE not in stderr_text:
        failures.append(f"stderr does not contain {EXPECTED_STDERR_LINE!r}")

    report_path = _rebase(out_root, paths.VALIDATION_REPORT)
    run_log_path = _rebase(out_root, paths.RUN_LOG)

    if not report_path.is_file():
        failures.append(f"{report_path} does not exist — a failed run must still report why")
        return failures
    if not run_log_path.is_file():
        failures.append(f"{run_log_path} does not exist")
        return failures

    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))

    status = run_log.get("status")
    if status != "failed":
        failures.append(f'run_log.json status is {status!r}, expected "failed"')

    if report.get("passed") is not False:
        failures.append(
            f"validation_report.json passed is {report.get('passed')!r}, expected False"
        )

    if report.get("run_id") != run_log.get("run_id"):
        failures.append(
            f"validation_report.json run_id {report.get('run_id')!r} does not match "
            f"run_log.json run_id {run_log.get('run_id')!r} — the report is from another run"
        )

    messages = [violation.get("message") for violation in report.get("violations", [])]
    if EXPECTED_VIOLATION_MESSAGE not in messages:
        failures.append(
            f"no violation says {EXPECTED_VIOLATION_MESSAGE!r}; violations were {messages}"
        )

    rules = [violation.get("rule") for violation in report.get("violations", [])]
    if EXPECTED_RULE not in rules:
        failures.append(f"no violation carries rule {EXPECTED_RULE!r}; rules were {rules}")

    return failures


def report_summary(out_root: Path, exit_code: int, failures: list[str]) -> None:
    """Print what was observed, so the job's log explains itself either way."""
    print(f"exit code: {exit_code} (expected {EXPECTED_EXIT_CODE})")

    report_path = _rebase(out_root, paths.VALIDATION_REPORT)
    run_log_path = _rebase(out_root, paths.RUN_LOG)
    if run_log_path.is_file():
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        print(f"run_log.json:           status={run_log.get('status')!r} "
              f"run_id={run_log.get('run_id')}")
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"validation_report.json: passed={report.get('passed')!r} "
              f"step={report.get('step')!r} run_id={report.get('run_id')}")
        for violation in report.get("violations", []):
            print(f"  violation: rule={violation.get('rule')!r} "
                  f"message={violation.get('message')!r}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(
            "usage: python scripts/ci_check_failure_run.py <out-root> <exit-code> <stderr-file>",
            file=sys.stderr,
        )
        return 2

    out_root = Path(arguments[0]).resolve()
    exit_code = int(arguments[1])
    stderr_path = Path(arguments[2])
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace") if (
        stderr_path.is_file()
    ) else ""

    failures = check(out_root, exit_code, stderr_text)
    report_summary(out_root, exit_code, failures)
    if failures:
        return 1
    print("\nOK: the run stopped gracefully, exit 2, and reported exactly why.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
