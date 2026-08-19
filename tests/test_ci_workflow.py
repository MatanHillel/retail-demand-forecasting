"""The CI workflow is part of the contract, so it is tested like code (US-35, PRD §42, §54).

`.github/workflows/ci.yml` is what stands between `main` and a broken pipeline. Nothing else in
the repository fails if a job is renamed, a Python version drifts or a secret creeps in — the
workflow simply stops checking what it claims to check, quietly. This file parses the YAML and
asserts the properties the issue and the PRD actually depend on:

* the four jobs exist under the exact names branch protection requires, and the three heavy ones
  gate on the fast one;
* CI runs the pipeline in `--no-llm` mode on the committed sample fixture — never downloading the
  raw dataset (§42) and never needing a credential (§37);
* **no `secrets.` reference anywhere.** A CI that can read a secret is a CI that can leak one, and
  this pipeline has no use for one;
* Python 3.11 everywhere, matching `requires-python` in `pyproject.toml` (§43);
* the failure path asserts the exit code and the exact `FLOW STOPPED:` wording;
* `make ci-local` covers the same four jobs, so "green locally" means something.

The workflow is read as YAML rather than grepped as text, so reformatting it cannot break these
tests and cannot silently satisfy them either.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from pipeline import paths

WORKFLOW_PATH = paths.PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE_PATH = paths.PROJECT_ROOT / "Makefile"

#: The four jobs, which are also the four required status checks on protected `main`
#: (docs/branch_protection.md). Renaming one here means renaming it there.
REQUIRED_JOBS = ("lint-test", "pipeline-no-llm", "failure-path", "determinism")

#: The three that must not start until the fast gate is green.
GATED_JOBS = ("pipeline-no-llm", "failure-path", "determinism")

PYTHON_VERSION = "3.11"


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def _steps(workflow: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return workflow["jobs"][job]["steps"]


def _run_commands(workflow: dict[str, Any], job: str) -> str:
    """Every `run:` script in one job, concatenated — what that job actually executes."""
    return "\n".join(step["run"] for step in _steps(workflow, job) if "run" in step)


def _all_run_commands(workflow: dict[str, Any]) -> str:
    return "\n".join(_run_commands(workflow, job) for job in workflow["jobs"])


def _load_module(path: Path) -> ModuleType:
    """Import a standalone script by file path — ``scripts/`` is not a package."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# the four jobs exist, are named exactly, and are ordered
# --------------------------------------------------------------------------
def test_the_workflow_parses(workflow) -> None:
    assert workflow, f"{WORKFLOW_PATH} is empty or not valid YAML"


def test_the_four_required_jobs_exist(workflow) -> None:
    assert sorted(workflow["jobs"]) == sorted(REQUIRED_JOBS)


@pytest.mark.parametrize("job", REQUIRED_JOBS)
def test_the_status_check_name_matches_the_job_id(workflow, job) -> None:
    """The check name branch protection sees is the job's `name:`; keep the two identical."""
    assert workflow["jobs"][job].get("name") == job


@pytest.mark.parametrize("job", GATED_JOBS)
def test_the_heavy_jobs_wait_for_lint_test(workflow, job) -> None:
    needs = workflow["jobs"][job].get("needs")
    needs = [needs] if isinstance(needs, str) else needs
    assert needs == ["lint-test"], f"{job} must not start before the fast gate is green"


@pytest.mark.parametrize("job", REQUIRED_JOBS)
def test_every_job_has_a_timeout(workflow, job) -> None:
    """An untimed job can hang for six hours and hold the whole merge queue."""
    timeout = workflow["jobs"][job].get("timeout-minutes")
    assert isinstance(timeout, int) and 0 < timeout <= 30


def test_the_workflow_runs_on_pull_requests_and_pushes_to_main(workflow) -> None:
    # PyYAML resolves the bare key `on:` to the boolean True (the "Norway problem").
    triggers = workflow.get("on") or workflow.get(True)
    assert "pull_request" in triggers
    assert triggers["push"]["branches"] == ["main"]


def test_runs_are_cancelled_per_branch(workflow) -> None:
    concurrency = workflow["concurrency"]
    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True


# --------------------------------------------------------------------------
# Python 3.11 everywhere (PRD §43)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("job", REQUIRED_JOBS)
def test_every_job_uses_python_311(workflow, job) -> None:
    setups = [
        step
        for step in _steps(workflow, job)
        if str(step.get("uses", "")).startswith("actions/setup-python")
    ]
    assert setups, f"{job} never sets up Python"
    for step in setups:
        assert str(step["with"]["python-version"]) == PYTHON_VERSION


def test_the_workflow_python_matches_pyproject() -> None:
    pyproject = (paths.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'>={PYTHON_VERSION}' in pyproject, "pyproject no longer requires the CI Python"


# --------------------------------------------------------------------------
# no credential, no raw download (PRD §37, §42)
# --------------------------------------------------------------------------
def test_no_secret_is_referenced_anywhere(workflow_text) -> None:
    """CI needs no credential: the pipeline runs in --no-llm mode, which imports no LLM class."""
    assert "secrets." not in workflow_text
    assert "${{ secrets" not in workflow_text


def test_the_pipeline_runs_in_no_llm_mode_on_the_sample(workflow) -> None:
    commands = _run_commands(workflow, "pipeline-no-llm")
    assert "python -m pipeline --no-llm --sample" in commands


def test_no_job_downloads_the_raw_dataset(workflow) -> None:
    """§42: CI reads the committed fixture only — the real extract is never fetched here."""
    commands = _all_run_commands(workflow)
    assert "pipeline.download" not in commands
    assert "--record-hash" not in commands


def test_every_pipeline_invocation_is_isolated_with_out_root(workflow) -> None:
    """Without --out-root a run would write into the checkout and the `git diff` guards below
    could never be trusted (US-34)."""
    # Un-wrap shell line-continuations first: a command split over several lines with a trailing
    # backslash is still one command, and its --out-root may sit on a later line.
    joined = _all_run_commands(workflow).replace("\\\n", " ")
    invocations = [line for line in joined.splitlines() if "-m pipeline" in line]
    assert invocations, "no job runs the pipeline at all"
    for line in invocations:
        assert "--out-root" in line, f"pipeline invocation without --out-root: {line.strip()}"


def test_the_pipeline_job_proves_the_checkout_is_untouched(workflow) -> None:
    commands = _run_commands(workflow, "pipeline-no-llm")
    assert "git diff --exit-code" in commands


# --------------------------------------------------------------------------
# the failure path asserts the exact contract (PRD §39)
# --------------------------------------------------------------------------
def test_the_failure_job_uses_the_missing_quantity_fixture(workflow) -> None:
    commands = _run_commands(workflow, "failure-path")
    assert "tests/fixtures/raw_sample_missing_quantity.csv" in commands


def test_the_missing_quantity_fixture_is_committed() -> None:
    fixture = paths.FIXTURES_DIR / "raw_sample_missing_quantity.csv"
    assert fixture.is_file(), "the failure-path job's input is not in the repository"
    assert "Quantity" not in fixture.read_text(encoding="utf-8").splitlines()[0]


def test_the_failure_job_asserts_exit_code_two_and_the_flow_stopped_wording() -> None:
    """The two assertions live in the checker the job calls, so they are read from there."""
    checker = (paths.PROJECT_ROOT / "scripts" / "ci_check_failure_run.py").read_text(
        encoding="utf-8"
    )
    assert "EXPECTED_EXIT_CODE = 2" in checker
    assert "FLOW STOPPED: " in checker
    assert "Missing required column Quantity" in checker


def test_the_failure_checker_agrees_with_the_pipeline_message() -> None:
    """The expected wording must match the pipeline's own template, not a second copy of it.

    ``scripts/`` is not an importable package (and CI runs ``pytest`` as a bare executable, which
    does not put the working directory on ``sys.path``), so the checker is loaded from its file.
    """
    from pipeline.download import MISSING_COLUMN

    checker = _load_module(paths.PROJECT_ROOT / "scripts" / "ci_check_failure_run.py")
    assert checker.EXPECTED_VIOLATION_MESSAGE == MISSING_COLUMN.format(column="Quantity")
    assert checker.EXPECTED_STDERR_LINE == f"FLOW STOPPED: {checker.EXPECTED_VIOLATION_MESSAGE}"


# --------------------------------------------------------------------------
# artifacts are published for a human to read
# --------------------------------------------------------------------------
def test_the_pipeline_job_uploads_the_reports(workflow) -> None:
    uploads = [
        step
        for step in _steps(workflow, "pipeline-no-llm")
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads, "the pipeline job publishes nothing for a reviewer to download"
    published = uploads[0]["with"]["path"]
    for expected in ("run_log.json", "evaluation_report.md", "model_card.md", "eda_report.html"):
        assert expected in published


def test_the_upload_runs_even_when_the_job_failed(workflow) -> None:
    """A red run is exactly when its run log is worth reading."""
    uploads = [
        step
        for step in _steps(workflow, "pipeline-no-llm")
        if str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert uploads[0].get("if") == "always()"


# --------------------------------------------------------------------------
# make ci-local mirrors the workflow
# --------------------------------------------------------------------------
@pytest.mark.parametrize("job", REQUIRED_JOBS)
def test_ci_local_covers_every_job(job) -> None:
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    target = "ci-pipeline" if job == "pipeline-no-llm" else f"ci-{job}"
    assert f"{target}:" in makefile, f"make ci-local does not reproduce the {job} job"
    assert target in makefile.split("ci-local:")[1].splitlines()[0]


def test_the_documented_required_checks_match_the_workflow() -> None:
    """docs/branch_protection.md names the checks a human ticks in the GitHub UI."""
    doc = (paths.DOCS_DIR / "branch_protection.md").read_text(encoding="utf-8")
    for job in REQUIRED_JOBS:
        assert f"`{job}`" in doc, f"{job} is not listed as a required check in the doc"
