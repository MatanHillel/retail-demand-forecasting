"""Validation results and graceful failure (PRD §39).

A deterministic step never decides on its own what to do when the data is wrong: it returns a
:class:`ValidationResult`, the Flow writes it to ``artifacts/validation_report.json`` and — if the
result failed — raises :class:`FlowValidationError`, whose message always starts with
``FLOW STOPPED:``. The run then exits non-zero **without** producing or overwriting forecasts.

Schema of ``validation_report.json`` (read by the Streamlit app when a run failed, US-27 —
keep it stable)::

    {
      "run_id": "20260815T190523Z-3f9a1c" | null,
      "timestamp": "2026-08-15T19:05:23+00:00",
      "step": "contract_validation",
      "passed": false,
      "checked_rows": 412873,
      "violations": [
        {"step": "...", "rule": "...", "message": "...", "count": 3, "examples": [...]}
      ],
      "extra": {}
    }
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pipeline import paths

#: Prefix required by PRD §39 on every graceful-stop message.
FLOW_STOPPED_PREFIX = "FLOW STOPPED:"


class Violation(BaseModel):
    """One broken rule, with enough detail for a human to act on it.

    ``count`` is how many rows or items broke the rule; ``examples`` holds a short, illustrative
    sample (never the whole offending set — the report must stay readable).
    """

    model_config = ConfigDict(extra="forbid")

    step: str
    rule: str
    message: str
    count: int | None = None
    examples: list[Any] | None = None


class ValidationResult(BaseModel):
    """The outcome of one validation step."""

    model_config = ConfigDict(extra="forbid")

    step: str
    passed: bool
    violations: list[Violation] = Field(default_factory=list)
    checked_rows: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """A one-line description suitable for a log line or a `FLOW STOPPED:` message."""
        if self.passed:
            return f"{self.step} passed"
        if len(self.violations) == 1:
            return self.violations[0].message
        return f"{self.step} failed with {len(self.violations)} violations"


def write_validation_report(
    result: ValidationResult,
    path: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Write ``validation_report.json`` and return the path it was written to.

    Written on failure *and* on success, so the app can always tell which step last ran and
    whether it passed. This file deliberately bypasses the staging area: the app must be able to
    read it precisely because the run failed (PRD §39).
    """
    destination = Path(path) if path is not None else paths.VALIDATION_REPORT
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "step": result.step,
        "passed": result.passed,
        "checked_rows": result.checked_rows,
        "violations": [violation.model_dump(mode="json") for violation in result.violations],
        "extra": result.extra,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


class FlowValidationError(Exception):
    """Raised to stop the Flow gracefully; carries the :class:`ValidationResult`.

    ``str(exc)`` always starts with ``FLOW STOPPED:`` — e.g.
    ``FLOW STOPPED: Missing required column Quantity`` (PRD §39).
    """

    def __init__(self, result: ValidationResult, message: str | None = None) -> None:
        self.result = result
        detail = message if message is not None else result.summary()
        super().__init__(f"{FLOW_STOPPED_PREFIX} {detail}")
