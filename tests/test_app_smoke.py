"""Streamlit app shell smoke tests (US-27, PRD §39).

Exercises ``src/app/Home.py`` through ``streamlit.testing.v1.AppTest`` against every run-status
state the banner (``app.components.status.run_status_banner``) must handle: missing, running,
failed (with and without a matching validation report, with a mismatched run id, and with a
matching-but-passed report) and success. Every non-success case must render no KPI tile and must
never leak a traceback into the page.

Fixtures build real ``RunContext`` / ``ValidationResult`` objects — the same construction pattern
as ``tests/test_run_context.py`` — under ``tmp_path``, then monkeypatch ``pipeline.paths`` so
``app.data_access`` reads from there instead of the real ``artifacts/`` directory. The path
constants are patched only *after* the files are written, because ``RunContext.write_run_log()``
rebases the *real*, unpatched ``paths.RUN_LOG`` onto ``base_dir`` to find where to write.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from pipeline import paths
from pipeline.run_context import RunContext, close_log_handlers
from pipeline.validation import ValidationResult, Violation, write_validation_report

HOME = str(Path(__file__).resolve().parents[1] / "src" / "app" / "Home.py")


def _make_run_context(tmp_path, *, status, errors=None) -> RunContext:
    ctx = RunContext.start(mode="no-llm", base_dir=tmp_path)
    if errors is not None:
        ctx.errors = errors
    if status == "running":
        ctx.write_run_log()
    else:
        ctx.finish(status=status)
    close_log_handlers(ctx.run_id)
    return ctx


def _write_validation_report(tmp_path, *, run_id, passed, violations=None) -> None:
    result = ValidationResult(
        step="contract_validation",
        passed=passed,
        checked_rows=100,
        violations=[Violation(**v) for v in (violations or [])],
    )
    write_validation_report(
        result,
        path=tmp_path / "artifacts" / "validation_report.json",
        run_id=run_id,
    )


def _patch_paths(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(paths, "RUN_LOG", tmp_path / "artifacts" / "run_log.json")
    monkeypatch.setattr(
        paths, "VALIDATION_REPORT", tmp_path / "artifacts" / "validation_report.json"
    )


def _text(at: AppTest) -> str:
    parts = []
    for group in (at.error, at.warning, at.info, at.caption, at.markdown, at.text):
        parts.extend(str(element.value) for element in group)
    return "\n".join(parts)


def _assert_no_kpi_tiles(at: AppTest) -> None:
    assert len(at.metric) == 0


class TestRunStatusBanner:
    def setup_method(self) -> None:
        st.cache_data.clear()

    def teardown_method(self) -> None:
        st.cache_data.clear()

    def test_home_renders_against_real_artifacts(self) -> None:
        # Generous timeout: this is the only case that draws Matplotlib figures, whose first-run
        # font-cache build can be slow.
        at = AppTest.from_file(HOME, default_timeout=30).run()
        assert not at.exception

    def test_failed_with_matching_validation_violations(self, tmp_path, monkeypatch) -> None:
        ctx = _make_run_context(tmp_path, status="failed")
        _write_validation_report(
            tmp_path,
            run_id=ctx.run_id,
            passed=False,
            violations=[
                {
                    "step": "contract_validation",
                    "rule": "no_nulls",
                    "message": "DISTINCTIVE_VIOLATION_MESSAGE",
                    "count": 3,
                    "examples": ["10080"],
                }
            ],
        )
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert f"run id {ctx.run_id}" in _text(at)
        assert "DISTINCTIVE_VIOLATION_MESSAGE" in _text(at)
        _assert_no_kpi_tiles(at)

    def test_failed_without_validation_report(self, tmp_path, monkeypatch) -> None:
        _make_run_context(
            tmp_path,
            status="failed",
            errors=[
                {
                    "step": "contract_validation",
                    "type": "ValueError",
                    "message": "DISTINCTIVE_ERROR_MESSAGE",
                    "traceback": "Traceback (most recent call last):\n  boom",
                }
            ],
        )
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert "DISTINCTIVE_ERROR_MESSAGE" in _text(at)
        assert "Traceback" not in _text(at)
        _assert_no_kpi_tiles(at)

    def test_failed_with_mismatched_run_id_falls_back_to_errors(
        self, tmp_path, monkeypatch
    ) -> None:
        _make_run_context(
            tmp_path,
            status="failed",
            errors=[
                {
                    "step": "contract_validation",
                    "type": "ValueError",
                    "message": "DISTINCTIVE_ERROR_MESSAGE",
                    "traceback": "Traceback (most recent call last):\n  boom",
                }
            ],
        )
        _write_validation_report(
            tmp_path,
            run_id="some-other-run-id",
            passed=False,
            violations=[
                {
                    "step": "contract_validation",
                    "rule": "no_nulls",
                    "message": "SHOULD_NOT_APPEAR",
                }
            ],
        )
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert "SHOULD_NOT_APPEAR" not in _text(at)
        assert "DISTINCTIVE_ERROR_MESSAGE" in _text(at)
        _assert_no_kpi_tiles(at)

    def test_failed_with_matching_report_but_passed_true_falls_back_to_errors(
        self, tmp_path, monkeypatch
    ) -> None:
        ctx = _make_run_context(
            tmp_path,
            status="failed",
            errors=[
                {
                    "step": "unexpected_exception",
                    "type": "RuntimeError",
                    "message": "DISTINCTIVE_ERROR_MESSAGE",
                    "traceback": "Traceback (most recent call last):\n  boom",
                }
            ],
        )
        _write_validation_report(tmp_path, run_id=ctx.run_id, passed=True)
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert "DISTINCTIVE_ERROR_MESSAGE" in _text(at)
        _assert_no_kpi_tiles(at)

    def test_running_status_hides_forecast_widgets(self, tmp_path, monkeypatch) -> None:
        ctx = _make_run_context(tmp_path, status="running")
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert "in progress" in _text(at)
        assert ctx.run_id in _text(at)
        _assert_no_kpi_tiles(at)

    def test_missing_run_log_shows_first_run_message(self, tmp_path, monkeypatch) -> None:
        _patch_paths(monkeypatch, tmp_path)

        at = AppTest.from_file(HOME).run()

        assert not at.exception
        assert "python -m pipeline --no-llm" in _text(at)
        _assert_no_kpi_tiles(at)
