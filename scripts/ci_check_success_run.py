"""Assert that a successful ``--no-llm`` pipeline run really succeeded (US-35, PRD §42).

Run by the ``pipeline-no-llm`` CI job after ``python -m pipeline --no-llm --sample`` has exited 0,
against the ``--out-root`` directory that run wrote into::

    python scripts/ci_check_success_run.py ci_out

Three things are checked, and each of them has a specific failure it exists to catch:

* ``run_log.json`` says ``status == "success"`` **exactly**. ``status`` has three values, not two —
  ``running`` is what stays on disk when a process is killed before ``finish()``, which is exactly
  what a CI timeout does. A ``!= "failed"`` test would call a timed-out run green
  (``docs/interfaces.md`` §14 / the issue's §8).
* Every artifact in :data:`pipeline.paths.REQUIRED_ARTIFACTS` exists with a non-zero size. The
  list is read from ``pipeline.paths``, never restated here, so a rename cannot leave CI checking
  a path nobody writes any more (§8). These are the *final* locations: the process has exited, so
  promotion has already happened.
* A champion was chosen — a run that produced artifacts but selected no model is not a success.

Exits 0 when everything holds, 1 with a readable report when it does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline import paths


def _rebase(out_root: Path, canonical: Path) -> Path:
    """Where ``canonical`` landed under this run's output root."""
    return out_root / canonical.relative_to(paths.PROJECT_ROOT)


def check(out_root: Path) -> list[str]:
    """Return a list of failures; empty means the run is a genuine success."""
    failures: list[str] = []

    run_log_path = _rebase(out_root, paths.RUN_LOG)
    if not run_log_path.is_file():
        return [f"{run_log_path} does not exist — the run wrote no run log at all"]

    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))

    status = run_log.get("status")
    if status != "success":
        # "running" means the process never reached finish() — a CI timeout looks exactly like
        # this, and must never be reported as a pass.
        failures.append(f'run_log.json status is {status!r}, expected "success"')

    for canonical in paths.REQUIRED_ARTIFACTS:
        artifact = _rebase(out_root, canonical)
        if not artifact.is_file():
            failures.append(f"required artifact missing: {canonical.name} ({artifact})")
        elif artifact.stat().st_size == 0:
            failures.append(f"required artifact is empty: {canonical.name} ({artifact})")

    champion = run_log.get("champion")
    if not champion or not champion.get("champion"):
        failures.append("run_log.json records no champion")

    return failures


def report(out_root: Path, failures: list[str]) -> None:
    """Print what was checked, so a green job is still readable in the CI log."""
    run_log_path = _rebase(out_root, paths.RUN_LOG)
    if run_log_path.is_file():
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        champion = (run_log.get("champion") or {}).get("champion", "—")
        print(f"run id:   {run_log.get('run_id')}")
        print(f"mode:     {run_log.get('mode')}")
        print(f"status:   {run_log.get('status')}")
        print(f"champion: {champion}")

    print(f"required artifacts ({len(paths.REQUIRED_ARTIFACTS)}):")
    for canonical in paths.REQUIRED_ARTIFACTS:
        artifact = _rebase(out_root, canonical)
        size = artifact.stat().st_size if artifact.is_file() else 0
        mark = "ok " if size > 0 else "MISSING"
        print(f"  {mark} {canonical.name:<24} {size:>10,} bytes")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: python scripts/ci_check_success_run.py <out-root>", file=sys.stderr)
        return 2

    out_root = Path(arguments[0]).resolve()
    failures = check(out_root)
    report(out_root, failures)
    if failures:
        return 1
    print("\nOK: the run succeeded and every required artifact was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
