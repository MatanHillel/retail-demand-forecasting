"""The MVP acceptance audit is checked like code, because the team trusts its verdict (US-37).

`scripts/mvp_acceptance_check.py` is the program that walks PRD §49 — the checklist that says when
the MVP is "done" — and reports PASS / FAIL / MANUAL / PENDING per clause. If it silently stops
checking something, or reports green while an artifact is missing, the whole point of it is gone.
This file proves the properties that would otherwise rot quietly:

* the checklist has one entry per §49 clause, each with a `check_NN` method behind it;
* running it over the repository produces one result per clause, each with a known status **and a
  non-empty evidence sentence** — the issue's "no `OK` without evidence";
* only clause 22 is PENDING before US-38 (slides) and US-39 (video) land;
* the report and the summary JSON have the documented shape, and the script exits non-zero as soon
  as one clause FAILs — injected here by pointing `paths.REQUIRED_ARTIFACTS` at a file that does
  not exist, so nothing on disk has to be renamed;
* the run-log gate turns every run-derived clause red when the audited run did not succeed;
* the audit is read-only: `run_log.json` and `validation_report.json` are byte-identical afterwards;
* `run_pytest` attributes a failure to the file it came from, which is what every test-backed
  clause depends on.

The heavy part of the audit — running the project's own test files in a subprocess — is replaced by
a stub here. Without it this file would re-run half the suite inside the suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

import pytest

from pipeline import paths

SCRIPT = paths.PROJECT_ROOT / "scripts" / "mvp_acceptance_check.py"

#: PRD §49 has this many clauses; the report is required to have one row per clause.
EXPECTED_CRITERIA = 22


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    """Import the script by path — `scripts/` is not a package.

    The module is registered in ``sys.modules`` *before* it is executed: ``@dataclass`` resolves a
    field's annotation by looking its own module up there, and a module that is not registered
    makes that lookup return ``None``.
    """
    spec = importlib.util.spec_from_file_location("mvp_acceptance_check", SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _all_pass(test_files: Sequence[Path]) -> dict[str, bool]:
    """Stand-in for `run_pytest`: every named file passed."""
    return {path.name: True for path in test_files}


def _all_fail(test_files: Sequence[Path]) -> dict[str, bool]:
    return {path.name: False for path in test_files}


@pytest.fixture(scope="module")
def results(audit: ModuleType) -> list:
    """One full audit of the repository, with the test suite stubbed out as green."""
    return audit.Auditor(skip_slow=True, pytest_runner=_all_pass).run()


# --------------------------------------------------------------------------
# the checklist itself
# --------------------------------------------------------------------------
def test_the_checklist_covers_every_prd_clause(audit: ModuleType) -> None:
    numbers = [number for number, _, _ in audit.CRITERIA]
    assert numbers == list(range(1, EXPECTED_CRITERIA + 1))


def test_every_clause_has_a_check_behind_it(audit: ModuleType) -> None:
    missing = [
        number
        for number, _, _ in audit.CRITERIA
        if not hasattr(audit.Auditor, f"check_{number:02d}")
    ]
    assert not missing, f"criteria with no check method: {missing}"


def test_no_criterion_or_how_column_is_empty(audit: ModuleType) -> None:
    blank = [number for number, criterion, how in audit.CRITERIA if not criterion or not how]
    assert not blank, f"criteria missing a title or a 'how checked' description: {blank}"


# --------------------------------------------------------------------------
# running it over the repository
# --------------------------------------------------------------------------
def test_one_result_per_clause_with_a_known_status(audit: ModuleType, results: list) -> None:
    assert len(results) == EXPECTED_CRITERIA
    assert [result.number for result in results] == list(range(1, EXPECTED_CRITERIA + 1))
    unknown = [result.number for result in results if result.status not in audit.STATUSES]
    assert not unknown, f"criteria with an unknown status: {unknown}"


def test_every_result_cites_evidence(results: list) -> None:
    """"no `OK` without evidence" — every verdict names a file, a value or a test."""
    bare = [result.number for result in results if len(result.evidence.strip()) < 20]
    assert not bare, f"criteria whose evidence column says nothing concrete: {bare}"


def test_only_the_presentation_and_demo_are_pending(audit: ModuleType, results: list) -> None:
    pending = {result.number for result in results if result.status == audit.PENDING}
    assert pending == {EXPECTED_CRITERIA}, (
        "before US-38 and US-39 exactly one clause — the presentation and demo video — may be "
        f"PENDING, but these are: {sorted(pending)}"
    )


def test_the_required_artifacts_clause_passes_on_this_repository(
    audit: ModuleType, results: list
) -> None:
    """The eight §41 artifacts are committed, so clause 18 must be green here."""
    clause = next(result for result in results if result.number == 18)
    assert clause.status == audit.PASS, clause.evidence


def test_a_red_test_file_turns_its_clause_red(audit: ModuleType) -> None:
    auditor = audit.Auditor(skip_slow=True, pytest_runner=_all_fail)
    verdict = auditor.check_20()
    assert verdict.status == audit.FAIL
    assert "test_readme.py FAILED" in verdict.evidence


# --------------------------------------------------------------------------
# the run-log gate
# --------------------------------------------------------------------------
def test_a_run_that_did_not_succeed_fails_every_run_derived_clause(
    audit: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`status` has three values; `running` is what a killed process leaves behind."""
    fake_log = tmp_path / "run_log.json"
    fake_log.write_text(json.dumps({"run_id": "20260101T000000Z-aaaaaa", "status": "running"}))
    monkeypatch.setattr(paths, "RUN_LOG", fake_log)

    auditor = audit.Auditor(skip_slow=True, pytest_runner=_all_pass)
    assert auditor.run_log_error
    for number in (1, 12, 18):
        verdict = getattr(auditor, f"check_{number:02d}")()
        assert verdict.status == audit.FAIL, f"clause {number} should be gated by the run status"
        assert "running" in verdict.evidence


def test_a_missing_run_log_is_reported_rather_than_crashing(
    audit: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "absent.json")
    auditor = audit.Auditor(skip_slow=True, pytest_runner=_all_pass)
    assert auditor.run_id == "unknown"
    assert auditor.check_18().status == audit.FAIL


# --------------------------------------------------------------------------
# the two output files
# --------------------------------------------------------------------------
@pytest.fixture
def written(audit: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Run the CLI end to end into `tmp_path`, with the test suite stubbed green."""
    monkeypatch.setattr(audit, "run_pytest", _all_pass)
    report = tmp_path / "acceptance_report.md"
    code = audit.main(["--out", str(report), "--skip-slow"])
    summary = tmp_path / paths.ACCEPTANCE_SUMMARY.name
    return code, report, summary


def test_the_cli_writes_both_files_and_exits_zero(written) -> None:
    code, report, summary = written
    assert report.is_file() and summary.is_file()
    assert code == 0, report.read_text(encoding="utf-8")


def test_the_report_has_one_row_per_clause_and_names_the_audited_run(
    audit: ModuleType, written
) -> None:
    _, report, _ = written
    text = report.read_text(encoding="utf-8")
    assert "*Audited run:*" in text, "the report must name the run it audited"
    rows = [line for line in text.splitlines() if line.startswith("| ") and "**" in line]
    criteria_rows = [line for line in rows if line.split("|")[1].strip().isdigit()]
    assert len(criteria_rows) == EXPECTED_CRITERIA
    for status in audit.STATUSES:
        assert f"| {status} |" in text, f"the summary table is missing the {status} count"


def test_the_summary_json_has_the_documented_shape(audit: ModuleType, written) -> None:
    _, _, summary = written
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert set(payload) == {
        "generated_at",
        "run_id",
        "run_status",
        "total",
        "counts",
        "failed",
        "criteria",
    }
    assert payload["total"] == EXPECTED_CRITERIA
    assert set(payload["counts"]) == set(audit.STATUSES)
    assert sum(payload["counts"].values()) == EXPECTED_CRITERIA
    assert len(payload["criteria"]) == EXPECTED_CRITERIA


def test_one_missing_required_artifact_makes_the_script_exit_non_zero(
    audit: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Inject a failure the way §41 would break: a required artifact that is not on disk.

    `paths.REQUIRED_ARTIFACTS` is redirected rather than renaming a committed file, so the
    repository is never touched by this test.
    """
    monkeypatch.setattr(audit, "run_pytest", _all_pass)
    monkeypatch.setattr(
        paths, "REQUIRED_ARTIFACTS", (*paths.REQUIRED_ARTIFACTS, tmp_path / "model_card.md")
    )
    report = tmp_path / "acceptance_report.md"
    code = audit.main(["--out", str(report), "--skip-slow"])
    assert code == 1

    payload = json.loads((tmp_path / paths.ACCEPTANCE_SUMMARY.name).read_text(encoding="utf-8"))
    assert 18 in payload["failed"]
    assert payload["counts"][audit.FAIL] >= 1
    assert "model_card.md" in payload["criteria"][17]["evidence"]


def test_the_audit_modifies_nothing_but_its_own_reports(
    audit: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It opens no `RunContext` and never calls `write_validation_report()` (issue §8)."""
    watched = [paths.RUN_LOG, paths.VALIDATION_REPORT, paths.CLEAN_DATA, paths.CHAMPION_DECISION]
    before = {path: path.read_bytes() for path in watched if path.is_file()}
    assert before, "nothing to watch — the repository has no artifacts to audit"

    monkeypatch.setattr(audit, "run_pytest", _all_pass)
    audit.main(["--out", str(tmp_path / "acceptance_report.md"), "--skip-slow"])

    changed = [path.name for path, content in before.items() if path.read_bytes() != content]
    assert not changed, f"the audit rewrote {changed}"


# --------------------------------------------------------------------------
# the pytest runner behind every test-backed clause
# --------------------------------------------------------------------------
def test_run_pytest_attributes_a_failure_to_its_own_file(audit: ModuleType, tmp_path: Path) -> None:
    green = tmp_path / "test_green_sample.py"
    green.write_text("def test_green():\n    assert True\n", encoding="utf-8")
    red = tmp_path / "test_red_sample.py"
    red.write_text("def test_red():\n    assert False\n", encoding="utf-8")

    outcome = audit.run_pytest([green, red])
    assert outcome == {"test_green_sample.py": True, "test_red_sample.py": False}


def test_a_file_that_produced_no_test_case_counts_as_failed(
    audit: ModuleType, tmp_path: Path
) -> None:
    """A collection error yields no `testcase` element; silence must not read as a pass."""
    broken = tmp_path / "test_broken_sample.py"
    broken.write_text("import a_module_that_does_not_exist\n", encoding="utf-8")
    assert audit.run_pytest([broken]) == {"test_broken_sample.py": False}
