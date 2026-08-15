"""Smoke test: the four source packages import and the normative folders exist (PRD §41)."""

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

PACKAGES = ["pipeline", "crews", "flow", "app"]

REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/forecasts",
    "artifacts/reports/figures",
    "artifacts/reports/eda_tables",
    "artifacts/reports/evaluation_tables",
    "artifacts/contracts",
    "config",
    "src/pipeline",
    "src/crews",
    "src/flow",
    "src/app",
    "tests/fixtures",
    "logs",
    "docs",
    "scripts",
]


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports(package: str) -> None:
    """`pip install -e .` makes each src-layout package importable by its top-level name."""
    assert importlib.import_module(package) is not None


@pytest.mark.parametrize("relative_path", REQUIRED_DIRS)
def test_required_directory_exists(relative_path: str) -> None:
    assert (REPO_ROOT / relative_path).is_dir(), f"missing required directory: {relative_path}"


def test_prd_is_present_and_is_v1_3() -> None:
    prd = REPO_ROOT / "docs" / "PRD.md"
    assert prd.is_file(), "docs/PRD.md is the specification of record"
    assert "Version:** 1.3" in prd.read_text(encoding="utf-8")
