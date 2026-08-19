"""Compare two ``--out-root`` runs and prove the numbers did not move (US-35, PRD §40).

Run by the ``determinism`` CI job after the same pipeline has been executed twice, into two
separate output roots::

    python scripts/ci_check_determinism.py ci_out_a ci_out_b

``tests/test_determinism.py`` (US-34) already proves this *in process*. This script proves the
same property through the **command line**: two real subprocesses, two output roots, byte
comparison of the files the PRD names. It is the end-to-end half of the guarantee — a
non-determinism that only appears when the interpreter starts fresh (import order, environment,
hash seeding of a child process) would pass the unit test and fail here.

What is compared, and what deliberately is not:

* **Compared byte for byte** — ``clean_data.csv``, ``features.csv`` and
  ``holdout_metrics_overall.csv``: the three files the issue names, all pure computed output.
* **Never compared** — ``run_log.json``. Every ``steps[]`` entry carries a wall-clock
  ``duration_s`` and its own ``started_at``, and the run carries ``run_id`` / ``started_at`` /
  ``finished_at``, so two runs differ there *by design* (``docs/interfaces.md`` §8 on US-34). The
  field-wise comparison of its deterministic subtree is ``test_determinism.py``'s job; repeating
  it here with ``cmp`` would only produce a flaky red.

Exits 0 when every compared file matches, 1 with the first differing line otherwise.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from pipeline import paths

#: The three files the issue names for the CLI-level comparison. Read from ``pipeline.paths`` so a
#: rename cannot leave this script comparing a path nobody writes any more.
COMPARED: tuple[Path, ...] = (
    paths.CLEAN_DATA,
    paths.FEATURES,
    paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv",
)


def _rebase(out_root: Path, canonical: Path) -> Path:
    return out_root / canonical.relative_to(paths.PROJECT_ROOT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_difference(left: Path, right: Path) -> str:
    """A readable description of where two text files first diverge."""
    left_lines = left.read_text(encoding="utf-8", errors="replace").splitlines()
    right_lines = right.read_text(encoding="utf-8", errors="replace").splitlines()
    for number, (a, b) in enumerate(zip(left_lines, right_lines, strict=False), start=1):
        if a != b:
            return f"first difference at line {number}:\n      run A: {a!r}\n      run B: {b!r}"
    if len(left_lines) != len(right_lines):
        return f"different line counts: {len(left_lines)} vs {len(right_lines)}"
    return "files differ in trailing bytes only (line endings?)"


def check(root_a: Path, root_b: Path) -> list[str]:
    """Return a list of failures; empty means the two runs are byte-identical where it matters."""
    failures: list[str] = []
    for canonical in COMPARED:
        name = canonical.relative_to(paths.PROJECT_ROOT).as_posix()
        left, right = _rebase(root_a, canonical), _rebase(root_b, canonical)

        if not left.is_file() or not right.is_file():
            missing = [str(p) for p in (left, right) if not p.is_file()]
            failures.append(f"{name}: not produced by both runs (missing: {', '.join(missing)})")
            continue

        digest_a, digest_b = _sha256(left), _sha256(right)
        if digest_a == digest_b:
            print(f"  IDENTICAL  {name}")
            print(f"             sha256 {digest_a}")
        else:
            print(f"  DIFFERS    {name}")
            print(f"             run A {digest_a}")
            print(f"             run B {digest_b}")
            print(f"             {_first_difference(left, right)}")
            failures.append(f"{name} differs between two runs of the same pipeline")
    return failures


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print("usage: python scripts/ci_check_determinism.py <out-root-a> <out-root-b>",
              file=sys.stderr)
        return 2

    root_a, root_b = Path(arguments[0]).resolve(), Path(arguments[1]).resolve()
    print(f"comparing {len(COMPARED)} artifact(s) between two runs:")
    failures = check(root_a, root_b)

    if failures:
        print("\nFAILED — the pipeline is not deterministic:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: two independent runs produced byte-identical numeric artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
