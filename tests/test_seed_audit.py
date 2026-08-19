"""Seed audit (US-34, PRD §40, §55).

**Seed** — a fixed starting number handed to anything "random", so that a "random" choice repeats
identically every time it is run. This file proves the project's single-global-seed discipline
holds across the whole ``src/`` tree: no module hard-codes a seed or a ``random_state`` of its own,
``set_global_seed`` is called exactly once per run — the one call already inside
``RunContext.start()`` (``docs/interfaces.md`` §3, §8 interface corrections) — and nothing under
``src/pipeline`` draws from NumPy's global random state without going through that single seeded
call or an explicitly-seeded ``Generator``.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np

from pipeline import paths
from pipeline.config import load_model_config
from pipeline.run_context import RunContext, close_log_handlers, set_global_seed

SRC_DIR = paths.PROJECT_ROOT / "src"

#: A literal digit right after ``random_state=`` / ``seed=`` — a hard-coded seed. The variable form
#: (``random_state=seed``, ``random_state=resolved``) is exactly what the project requires and is
#: not matched by this pattern; only a literal number is forbidden (PRD §40, CLAUDE.md §2.4).
_LITERAL_SEED_PATTERN = re.compile(r"\b(random_state|seed)\s*=\s*[0-9]")

#: ``set_global_seed`` itself calls ``np.random.seed(...)`` — the one sanctioned, exempted spot
#: (issue §8: "Exempt that function by name").
_EXEMPT_FUNCTION = "set_global_seed"
_RUN_CONTEXT_FILE = SRC_DIR / "pipeline" / "run_context.py"

#: An explicitly-seeded Generator call — allowed anywhere (e.g. ``pipeline.download``'s sampling).
_SEEDED_GENERATOR_PATTERN = re.compile(r"np\.random\.default_rng\(")
_BARE_NP_RANDOM_PATTERN = re.compile(r"np\.random\.")


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


# --------------------------------------------------------------------------
# (a) no literal random_state= / seed= anywhere under src/
# --------------------------------------------------------------------------
def test_no_literal_seed_or_random_state_under_src() -> None:
    offenders = []
    for path in _python_files(SRC_DIR):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _LITERAL_SEED_PATTERN.search(line):
                relative = path.relative_to(paths.PROJECT_ROOT)
                offenders.append(f"{relative}:{line_number}: {line.strip()}")
    assert offenders == [], "hard-coded seed/random_state found (PRD §40):\n" + "\n".join(offenders)


# --------------------------------------------------------------------------
# (b) set_global_seed is called exactly once per run, by RunContext.start()
# --------------------------------------------------------------------------
def test_set_global_seed_is_called_exactly_once_by_run_context_start(tmp_path, monkeypatch) -> None:
    import pipeline.run_context as run_context_module

    calls: list[int | None] = []
    original = run_context_module.set_global_seed

    def counting_seed(seed=None):
        calls.append(seed)
        return original(seed)

    monkeypatch.setattr(run_context_module, "set_global_seed", counting_seed)

    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    try:
        assert len(calls) == 1, f"set_global_seed called {len(calls)} times, expected exactly 1"
        assert ctx.seed == load_model_config().seed
    finally:
        close_log_handlers(ctx.run_id)


def test_run_flow_never_passes_an_explicit_seed_to_run_context_start() -> None:
    """The CLI/Flow entry point must let ``start()`` resolve the seed, never pass one in."""
    import flow.main

    source = inspect.getsource(flow.main.run_flow)
    assert "seed=" not in source, "flow.main.run_flow must not pass seed= to RunContext.start()"


def test_set_global_seed_reads_from_config_not_a_literal() -> None:
    assert set_global_seed() == load_model_config().seed


# --------------------------------------------------------------------------
# (c) no bare np.random.* under src/pipeline without an explicit seed
# --------------------------------------------------------------------------
def _set_global_seed_line_range() -> range:
    """Line numbers spanned by ``set_global_seed`` in run_context.py — the one exempt function."""
    tree = ast.parse(_RUN_CONTEXT_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _EXEMPT_FUNCTION:
            return range(node.lineno, (node.end_lineno or node.lineno) + 1)
    raise AssertionError(f"{_EXEMPT_FUNCTION} not found in {_RUN_CONTEXT_FILE}")


def test_no_bare_np_random_under_pipeline_outside_the_seeded_entry_points() -> None:
    exempt_lines = _set_global_seed_line_range()
    offenders = []
    for path in _python_files(SRC_DIR / "pipeline"):
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not _BARE_NP_RANDOM_PATTERN.search(line):
                continue
            if _SEEDED_GENERATOR_PATTERN.search(line):
                continue  # np.random.default_rng(seed) — explicitly seeded, allowed anywhere
            if path == _RUN_CONTEXT_FILE and line_number in exempt_lines:
                continue  # the sanctioned np.random.seed(resolved) call inside set_global_seed
            relative = path.relative_to(paths.PROJECT_ROOT)
            offenders.append(f"{relative}:{line_number}: {line.strip()}")
    assert offenders == [], "unseeded np.random.* usage found under src/pipeline:\n" + "\n".join(
        offenders
    )


def test_seeded_draws_are_repeatable() -> None:
    set_global_seed()
    first = np.random.rand(5).tolist()
    set_global_seed()
    assert np.random.rand(5).tolist() == first
