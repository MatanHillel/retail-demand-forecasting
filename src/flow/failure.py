"""Graceful-stop message templates and the §39 failure handler (US-32, PRD §39).

Every validation checkpoint's exact wording lives with the module that owns the check:
:mod:`pipeline.download` (raw schema / raw hash), :mod:`pipeline.contract` (dataset contract),
:mod:`pipeline.feature_validation` (leakage) and :mod:`flow.steps` (a required artifact missing).
This module re-exports the five §39 templates as one place to read them, and owns the one piece of
behaviour every failure path shares: turn the exception :meth:`flow.main.RetailForecastFlow._run`
caught into ``validation_report.json`` plus a ``status: "failed"`` ``run_log.json``, and archive
(or discard) this run's staging tree so a failed run never overwrites the previous good artifacts.

The five detail templates — ``pipeline.validation.FLOW_STOPPED_PREFIX`` is prepended once by
:class:`~pipeline.validation.FlowValidationError`, never repeated here:

* ``MISSING_COLUMN`` — raw schema (:mod:`pipeline.download`)
* ``RAW_HASH_MISMATCH`` — raw hash check (:mod:`pipeline.download`)
* ``CONTRACT_MISMATCH`` — dataset contract (:mod:`pipeline.contract`)
* ``LEAKAGE`` — feature leakage (:mod:`pipeline.feature_validation`)
* ``ARTIFACT_NOT_GENERATED`` — a required artifact missing (:func:`flow.steps.artifact_validation`)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flow.state import FlowState
from flow.steps import validation_report_path
from pipeline import paths
from pipeline.contract import CONTRACT_MISMATCH_TEMPLATE as CONTRACT_MISMATCH
from pipeline.download import MISSING_COLUMN, RAW_HASH_MISMATCH
from pipeline.feature_validation import LEAKAGE_FAILURE_MESSAGE as LEAKAGE
from pipeline.run_context import RunContext, redact
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

__all__ = [
    "MISSING_COLUMN",
    "RAW_HASH_MISMATCH",
    "CONTRACT_MISMATCH",
    "LEAKAGE",
    "ARTIFACT_NOT_GENERATED",
    "UNEXPECTED_EXCEPTION_RULE",
    "handle_failure",
]

#: Mirrors ``flow.steps.artifact_validation``'s inline wording — kept here for the §39 checklist;
#: that function builds its own message, so this constant documents the shape, not the source.
ARTIFACT_NOT_GENERATED = "{artifact} was not generated"

#: ``Violation.rule`` for an exception that escaped every validation check (docs/interfaces.md §13
#: interface corrections — never write an empty report for this case).
UNEXPECTED_EXCEPTION_RULE = "unexpected_exception"


def handle_failure(
    state: FlowState,
    ctx: RunContext,
    error: Exception,
    *,
    keep_failed: bool = True,
) -> Path | None:
    """The §39 failure path: report + ``status: failed`` run log + staging archived, never promoted.

    ``error`` is whatever :meth:`flow.main.RetailForecastFlow._run` caught: a
    :class:`~pipeline.validation.FlowValidationError` carries the real
    :class:`~pipeline.validation.ValidationResult`; anything else (``MemoryError``, ``KeyError``,
    …) has none, so one is synthesised with ``rule="unexpected_exception"`` — never an empty
    report, which would tell the app a run failed for no stated reason. ``ctx.promote()`` is never
    called here: it refuses once ``ctx.status == "failed"`` anyway.

    Returns the path artifacts were archived to (``logs/failed_runs/<run_id>/``), or ``None`` when
    ``keep_failed`` is ``False`` and the staging tree was discarded instead.
    """
    if isinstance(error, FlowValidationError):
        result = error.result
        log_message = str(error)  # already "FLOW STOPPED: ..." (FlowValidationError.__str__)
    else:
        step = state.current_step or "flow"
        message = redact(str(error))
        result = ValidationResult(
            step=step,
            passed=False,
            violations=[
                Violation(step=step, rule=UNEXPECTED_EXCEPTION_RULE, message=message)
            ],
        )
        log_message = f"FLOW STOPPED: unexpected error in {step} — {message}"

    write_validation_report(result, validation_report_path(ctx), run_id=ctx.run_id)
    ctx.finish("failed")
    state.errors = list(ctx.errors)
    state.status = "failed"
    ctx.logger.error(log_message)

    return _archive_staging(ctx, keep_failed=keep_failed)


def _archive_staging(ctx: RunContext, *, keep_failed: bool) -> Path | None:
    """Move this run's staging tree to ``logs/failed_runs/<run_id>/`` for debugging, or delete it.

    Moving the ``<run_id>`` directory (not its contents) out of ``artifacts/_staging/`` is what
    makes "staging is empty" literally true afterwards — the same reason
    :meth:`~pipeline.run_context.RunContext.discard_staging` deletes the directory rather than its
    files. Uses ``ctx.staging_dir`` / ``ctx.discard_staging()`` rather than a hand-built path
    (docs/interfaces.md §13 interface corrections).
    """
    staging_dir = ctx.staging_dir
    if not keep_failed:
        ctx.discard_staging()
        return None

    destination = (
        ctx.base_dir / paths.FAILED_RUNS_DIR.relative_to(paths.PROJECT_ROOT) / ctx.run_id
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    if staging_dir.exists():
        shutil.move(str(staging_dir), str(destination))
    else:
        # Nothing was ever staged (the run failed before any ctx.out() call) — still leave a
        # marker directory so "logs/failed_runs/<run_id>/ exists" holds unconditionally.
        destination.mkdir(parents=True, exist_ok=True)
    return destination
