"""Pinned requirements (US-34, PRD §40, §55).

Reproducibility needs every dependency nailed to one exact version — a range like ``pandas>=2.2``
could quietly resolve to a different release on a different machine or a different day and change
a result. This file checks ``requirements.txt`` uses ``==`` everywhere, and that the installed
environment has no conflicting packages (``pip check``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest

from pipeline import paths

REQUIREMENTS_FILE = paths.PROJECT_ROOT / "requirements.txt"
_PIN_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.\-]+$")


def _requirement_lines() -> list[str]:
    text = REQUIREMENTS_FILE.read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def test_requirements_file_is_not_empty() -> None:
    assert _requirement_lines(), f"{REQUIREMENTS_FILE} has no dependency lines"


def test_every_requirement_uses_double_equals_pin() -> None:
    unpinned = [line for line in _requirement_lines() if not _PIN_PATTERN.match(line)]
    assert unpinned == [], f"requirements.txt has non-== pin(s): {unpinned}"


def test_pip_check_passes() -> None:
    """The installed environment has no version conflicts.

    CI (US-35) installs with plain ``pip``, so ``python -m pip check`` is the real check there.
    Locally (CLAUDE.md §4) the venv is built with ``uv``, which does not install a ``pip`` module
    into it — ``uv pip check`` is the equivalent for that environment. Either way this asserts an
    actual conflict check ran; it never silently skips just because ``pip`` itself is absent.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 and "No module named pip" in (result.stdout + result.stderr):
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("neither pip nor uv is available to run a dependency-conflict check")
        result = subprocess.run(
            [uv, "pip", "check", "--python", sys.executable],
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
