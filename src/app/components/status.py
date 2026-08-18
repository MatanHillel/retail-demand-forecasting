"""Run-status / failure banner (US-27, PRD §39).

``run_log.json`` carries three states plus absence (§8 of AI-32): ``missing`` (no run yet),
``running`` (in progress, or a process that was killed before it could finish), ``failed`` and
``success``. Only ``success`` may show forecast widgets — every other case renders a message here
and the caller (a page) stops rendering the rest of itself.

``validation_report.json`` is written on both success and failure and is never cleared between
runs, so its ``run_id`` frequently belongs to an older run than the current ``run_log.json``. It is
trusted as the failure reason only when its ``run_id`` matches and it did not itself pass; every
other combination falls back to ``run_log["errors"][-1]``, and its ``traceback`` is never rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import streamlit as st

from app import data_access

AppRunStatus = Literal["missing", "running", "success", "failed"]


@dataclass(frozen=True)
class FailureReason:
    kind: Literal["violations", "errors", "none"]
    violations: list[dict[str, Any]] | None = None
    error: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunStatusState:
    status: AppRunStatus
    run_id: str | None
    reason: FailureReason | None
    run_log: dict[str, Any] | None


def run_status_banner() -> RunStatusState:
    """Draw the run-status banner and return the state so callers know what to hide."""
    try:
        run_log = data_access.load_run_log()
    except data_access.ArtifactMissing:
        st.info("No pipeline run found yet. Run `python -m pipeline --no-llm` first.")
        return RunStatusState("missing", None, None, None)

    status = run_log.get("status")
    run_id = run_log.get("run_id")

    if status == "running":
        st.warning(
            f"A pipeline run is in progress or did not complete (run id {run_id}) — "
            "forecasts are hidden until it finishes."
        )
        return RunStatusState("running", run_id, None, run_log)

    if status == "failed":
        reason = _resolve_failure_reason(run_log)
        st.error(f"Forecast data unavailable — latest pipeline run failed (run id {run_id})")
        _render_failure_reason(reason)
        return RunStatusState("failed", run_id, reason, run_log)

    st.caption(_format_success_caption(run_log))
    return RunStatusState("success", run_id, None, run_log)


def _resolve_failure_reason(run_log: dict[str, Any]) -> FailureReason:
    report = data_access.load_validation_report_for_run(run_log.get("run_id"))
    if report is not None and report.get("passed") is False:
        return FailureReason("violations", violations=report.get("violations") or [])

    errors = run_log.get("errors") or []
    if errors:
        return FailureReason("errors", error=errors[-1])
    return FailureReason("none")


def _render_failure_reason(reason: FailureReason) -> None:
    if reason.kind == "violations":
        for violation in reason.violations or []:
            with st.expander(f"{violation.get('step')} — {violation.get('rule')}"):
                st.write(violation.get("message"))
                if violation.get("count") is not None:
                    st.caption(f"{violation['count']} row(s) affected")
                if violation.get("examples"):
                    st.write(violation["examples"])
    elif reason.kind == "errors":
        error = reason.error or {}
        st.write(
            f"**Step:** {error.get('step')}  \n"
            f"**Type:** {error.get('type')}  \n"
            f"**Message:** {error.get('message')}"
        )
    else:
        st.caption("The run failed, but no further detail was recorded.")


def _format_success_caption(run_log: dict[str, Any]) -> str:
    timestamp = run_log.get("finished_at") or run_log.get("started_at") or "—"
    sha256 = (run_log.get("data") or {}).get("sha256")
    sha_short = sha256[:12] if sha256 else "—"
    mode = run_log.get("mode") or "—"
    champion_id = _extract_champion_id(run_log.get("champion"))
    return (
        f"run {run_log.get('run_id', '—')} · {timestamp} · data {sha_short} · "
        f"mode {mode} · champion {champion_id}"
    )


def _extract_champion_id(champion: dict[str, Any] | None) -> str:
    if not champion:
        return "—"
    for key in ("champion", "model", "id", "model_id"):
        value = champion.get(key)
        if isinstance(value, str) and value:
            return value
    return "—"
