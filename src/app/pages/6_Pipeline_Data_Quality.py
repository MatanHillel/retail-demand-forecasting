"""Screen 6 — Pipeline & Data Quality (US-30, PRD §33.6).

The diagnostics screen: unlike every other page it renders in full even when the last pipeline
run failed — that is the point of it. It never calls ``st.stop()`` on a non-success status; every
section below is independently guarded against a missing artifact instead.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data_access
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline import paths
from pipeline.config import load_data_sources, load_non_inventory_codes
from pipeline.contract import validate_contract_files

st.set_page_config(page_title="Pipeline & Data Quality", layout="wide")
apply_theme()
st.title("Pipeline & Data Quality")
st.caption(
    "This is the diagnostics screen — it renders even when the last pipeline run failed, so the "
    "failure reason is always visible."
)

def _champion_id(champion: dict | None) -> str:
    if not champion:
        return "—"
    value = champion.get("champion")
    return value if isinstance(value, str) and value else "—"


def _champion_metrics(champion: dict | None, champion_id: str) -> tuple[str, str]:
    if not champion or champion_id == "—":
        return "—", "—"
    candidates = champion.get("candidates") or []
    row = next((c for c in candidates if c.get("model") == champion_id), None)
    if row is None:
        return "—", "—"
    wmape = row.get("wmape")
    bias = row.get("bias")
    return (
        f"{wmape:.1%}" if isinstance(wmape, int | float) else "—",
        f"{bias:.1%}" if isinstance(bias, int | float) else "—",
    )


state = run_status_banner()
run_log = state.run_log
matched_report = data_access.load_validation_report_for_run(
    run_log.get("run_id") if run_log else None
)

# ---------------------------------------------------------------------------
# Last run status
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Last run status")
if run_log is None:
    st.caption("No pipeline run recorded yet.")
else:
    meta_cols = st.columns(4)
    meta_cols[0].metric("Status", run_log.get("status", "—"))
    meta_cols[1].metric("Mode", run_log.get("mode", "—"))
    meta_cols[2].metric("Seed", run_log.get("seed", "—"))
    data_block = run_log.get("data") or {}
    sha256 = data_block.get("sha256")
    meta_cols[3].metric("Data hash", f"{sha256[:12]}…" if sha256 else "—")
    st.caption(
        f"Run id `{run_log.get('run_id', '—')}` · started {run_log.get('started_at', '—')} · "
        f"finished {run_log.get('finished_at') or '—'}"
    )
    versions = run_log.get("versions") or {}
    version_text = ", ".join(
        f"{name} {version or 'not installed'}" for name, version in versions.items()
    )
    st.caption(f"Library versions: {version_text}" if version_text else "Library versions: —")

    steps = run_log.get("steps") or []
    if steps:
        st.markdown("**Steps**")
        steps_display = pd.DataFrame(
            {
                "Step": [step.get("name") for step in steps],
                "Status": [step.get("status") for step in steps],
                "Duration (s)": [step.get("duration_s") for step in steps],
                "Row counts": [step.get("row_counts") or {} for step in steps],
                "Warnings": [len(step.get("warnings") or []) for step in steps],
            }
        )
        st.dataframe(steps_display, use_container_width=True, hide_index=True)
        st.caption(
            "A step is appended to this list when it *starts*: an interrupted run ends with a "
            "'running' step and steps after a failure never appear at all."
        )
    else:
        st.caption("No steps recorded on this run's log.")

    if run_log.get("status") == "failed":
        st.markdown("**Failure detail**")
        errors = run_log.get("errors") or []
        if errors:
            st.error(errors[-1].get("message") or "(no message recorded)")
            for error in errors:
                with st.expander(f"{error.get('step')} — {error.get('type')}"):
                    st.write(f"**Message:** {error.get('message')}")
                    traceback_text = error.get("traceback")
                    if traceback_text:
                        st.caption("Traceback (redacted):")
                        st.code(traceback_text)
        else:
            st.caption("The run failed, but no errors were recorded on the log.")

        if matched_report is not None and matched_report.get("violations"):
            st.markdown(f"Violations from `{matched_report.get('step')}`:")
            st.dataframe(
                pd.DataFrame(matched_report["violations"]),
                use_container_width=True,
                hide_index=True,
            )

# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Run history")
st.caption("`logs/` is git-ignored — this shows only runs that happened on this machine.")
history = data_access.list_run_history()
if not history:
    st.caption("No archived runs found under `logs/`.")
else:
    history_rows = []
    for entry in history:
        champion_id = _champion_id(entry.get("champion"))
        wmape, bias = _champion_metrics(entry.get("champion"), champion_id)
        history_rows.append(
            {
                "Run id": entry.get("run_id", "—"),
                "Started": entry.get("started_at", "—"),
                "Status": entry.get("status", "—"),
                "Champion": champion_id,
                "wMAPE": wmape,
                "Bias": bias,
            }
        )
    st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Cleaning waterfall & data quality
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Cleaning waterfall")
try:
    waterfall = data_access.load_eda_table("E01_cleaning_waterfall")
    st.dataframe(waterfall, use_container_width=True, hide_index=True)
    st.image(str(data_access.figure_path("E01_waterfall")), use_column_width=True)
except data_access.ArtifactMissing as exc:
    st.warning(f"Cleaning waterfall unavailable: {exc}")

try:
    findings = data_access.load_dq_findings()
except data_access.ArtifactMissing as exc:
    findings = None
    st.warning(f"data_quality_findings.json unavailable: {exc}")

if findings is not None:
    duplicates = findings.get("duplicates") or {}
    st.markdown("**Duplicate rows**")
    dup_cols = st.columns(4)
    dup_cols[0].metric("Duplicate rows", f"{duplicates.get('duplicate_rows', 0):,}")
    dup_cols[1].metric("Row share", f"{duplicates.get('duplicate_row_share', 0):.2%}")
    dup_cols[2].metric("Duplicate units", f"{duplicates.get('duplicate_units', 0):,}")
    dup_cols[3].metric("Units share", f"{duplicates.get('duplicate_units_share', 0):.2%}")
    if duplicates.get("warning"):
        st.warning(
            "Above the configured threshold "
            f"({duplicates.get('row_share_threshold', 0):.2%} rows / "
            f"{duplicates.get('units_share_threshold', 0):.2%} units) — flagged (PRD §11)."
        )
    else:
        st.caption("Within the configured duplicate thresholds.")

    st.markdown("**Non-product / non-inventory codes**")
    nonproduct = findings.get("nonproduct_codes") or {}
    st.caption(
        f"{nonproduct.get('codes', 0)} codes documented, "
        f"{nonproduct.get('review_needed', 0)} awaiting review."
    )
    try:
        exclusion_df = load_non_inventory_codes()
        st.dataframe(exclusion_df, use_container_width=True, hide_index=True)
    except (FileNotFoundError, ValueError) as exc:
        st.warning(f"Exclusion list unavailable: {exc}")

    st.markdown("**Abnormal line quantities**")
    abnormal = findings.get("abnormal_quantities") or {}
    st.caption(
        f"{abnormal.get('lines', 0)} line(s) at or above the "
        f"{abnormal.get('threshold', '—')}-unit threshold (flagged, never removed — §10)."
    )
    try:
        abnormal_lines = data_access.load_eda_table("E01_abnormal_lines")
        st.dataframe(abnormal_lines.head(20), use_container_width=True, hide_index=True)
    except data_access.ArtifactMissing:
        pass

    st.markdown("**Partial month**")
    partial = findings.get("partial_month") or {}
    st.caption(
        f"{', '.join(partial.get('months', []))}: {partial.get('rows', 0):,} rows "
        f"({partial.get('note', '')})"
    )

# ---------------------------------------------------------------------------
# Contract validation status
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Contract validation status")
if matched_report is None:
    st.caption("No validation report for this run.")
else:
    checked_rows = matched_report.get("checked_rows")
    violations = matched_report.get("violations") or []
    if matched_report.get("passed"):
        rows_note = f" ({checked_rows:,} rows checked)" if checked_rows is not None else ""
        st.success(f"PASSED — `{matched_report.get('step')}`{rows_note}")
    else:
        st.error(
            f"FAILED — `{matched_report.get('step')}` — {len(violations)} violation(s)"
        )
    if violations:
        st.dataframe(pd.DataFrame(violations), use_container_width=True, hide_index=True)

if st.button("Re-validate now"):
    try:
        revalidate_result = validate_contract_files(paths.CLEAN_DATA, paths.DATASET_CONTRACT)
    except FileNotFoundError as exc:
        st.error(f"Cannot re-validate: {exc}")
    else:
        if revalidate_result.passed:
            st.success(f"PASSED ({revalidate_result.checked_rows:,} rows checked)")
        else:
            st.error(f"FAILED — {len(revalidate_result.violations)} violation(s)")
            st.dataframe(
                pd.DataFrame(
                    [violation.model_dump() for violation in revalidate_result.violations]
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("This check runs in memory only — it never overwrites validation_report.json.")

try:
    contract = data_access.load_contract()
    st.markdown("**Contract summary**")
    st.caption(contract.get("grain", "—"))
    contract_cols = st.columns(4)
    contract_cols[0].metric("Primary key", ", ".join(contract.get("primary_key", [])))
    contract_cols[1].metric("Active-rule k", contract.get("active_rule", {}).get("k", "—"))
    date_range = contract.get("date_range", {})
    contract_cols[2].metric(
        "Date range",
        f"{date_range.get('first_month', '—')} .. {date_range.get('last_full_month', '—')}",
    )
    contract_cols[3].metric("Documented exclusions", len(contract.get("exclusion_list", [])))
except data_access.ArtifactMissing as exc:
    st.warning(f"dataset_contract.json unavailable: {exc}")

try:
    feature_validation = data_access.load_feature_validation()
    st.markdown("**Feature validation summary**")
    checks = pd.DataFrame(feature_validation.get("checks", []))
    st.dataframe(checks, use_container_width=True, hide_index=True)
    overall = "PASSED" if feature_validation.get("passed") else "FAILED"
    st.caption(f"Overall: {overall} (run `{feature_validation.get('run_id', '—')}`)")
except data_access.ArtifactMissing as exc:
    st.warning(f"feature_validation.json unavailable: {exc}")

# ---------------------------------------------------------------------------
# Model card & evaluation report
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Reports")
tab_model_card, tab_evaluation = st.tabs(["Model card", "Evaluation report"])
with tab_model_card:
    try:
        model_card_text = data_access.load_text(paths.MODEL_CARD)
        st.markdown(model_card_text)
        st.download_button(
            "Download model_card.md",
            data=model_card_text,
            file_name="model_card.md",
            mime="text/markdown",
        )
    except data_access.ArtifactMissing as exc:
        st.warning(f"model_card.md unavailable: {exc}")

with tab_evaluation:
    try:
        evaluation_text = data_access.load_text(paths.EVALUATION_REPORT)
        st.markdown(evaluation_text)
        st.download_button(
            "Download evaluation_report.md",
            data=evaluation_text,
            file_name="evaluation_report.md",
            mime="text/markdown",
        )
    except data_access.ArtifactMissing as exc:
        st.warning(f"evaluation_report.md unavailable: {exc}")

# ---------------------------------------------------------------------------
# Ethics & licensing
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Ethics & licensing")
data_sources = load_data_sources()
st.markdown(f"**License:** CC BY 4.0 · {data_sources.citation}")
st.caption(
    "Recommendations rely only on historical sales in this dataset — no external signal is used "
    "(PRD §48)."
)
