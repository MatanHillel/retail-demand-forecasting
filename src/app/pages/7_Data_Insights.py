"""Screen 7 — Data & Insights (EDA) (US-30, PRD §33.7).

Shows the mandatory EDA figures with the matching sentence(s) from ``insights.md``, a table
viewer with CSV download, and a download of the full ``eda_report.html``. Every figure comes from
``artifacts/reports/figures/`` via ``index.json`` — this page never calls ``plt.`` itself.
"""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO

import streamlit as st

from app import data_access
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline import paths
from pipeline.config import load_data_sources

#: E-ids that must be shown expanded, in display order (§33.7, §35A.3).
MANDATORY_IDS: tuple[str, ...] = ("E1", "E2", "E5", "E6", "E7", "E8", "E9", "E11")

#: Matches an insight line's trailing citation, e.g. "... (E2, table `E02_sep_nov_share`)".
_INSIGHT_CITATION = re.compile(
    r"^\d+\.\s+(?P<sentence>.*?)\s*\(E(?P<eid>\d+),\s*table", re.MULTILINE
)


def _insights_by_eid(insights_text: str) -> dict[str, list[str]]:
    """Group each numbered insight sentence under the E-id its trailing citation names."""
    grouped: dict[str, list[str]] = {}
    for match in _INSIGHT_CITATION.finditer(insights_text):
        grouped.setdefault(f"E{match.group('eid')}", []).append(match.group("sentence"))
    return grouped


def _render_figures(figure_names: list[str]) -> None:
    if not figure_names:
        return
    columns = st.columns(len(figure_names))
    for column, figure_name in zip(columns, figure_names, strict=False):
        try:
            column.image(str(data_access.figure_path(figure_name)), use_column_width=True)
        except data_access.ArtifactMissing:
            column.warning(f"Figure `{figure_name}` unavailable.")


def _available_table_names(names: list[str]) -> list[str]:
    """The subset of ``names`` that exist as a CSV table (some index entries are JSON-only)."""
    available = []
    for name in names:
        try:
            data_access.load_eda_table(name)
        except data_access.ArtifactMissing:
            continue
        available.append(name)
    return available


def _tables_zip(names: list[str]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            table = data_access.load_eda_table(name)
            archive.writestr(f"{name}.csv", table.to_csv(index=False))
    return buffer.getvalue()


st.set_page_config(page_title="Data & Insights", layout="wide")
apply_theme()
st.title("Data & Insights")

state = run_status_banner()
if state.status != "success":
    st.stop()

index_path = paths.EDA_TABLES_DIR / "index.json"
try:
    index_entries = json.loads(data_access.load_text(index_path))
except data_access.ArtifactMissing as exc:
    st.warning(f"EDA table index unavailable: {exc}")
    st.stop()

try:
    insights_text = data_access.load_text(paths.INSIGHTS)
except data_access.ArtifactMissing:
    insights_text = ""
insights_by_id = _insights_by_eid(insights_text)

entries_by_id = {entry["id"]: entry for entry in index_entries}

st.subheader("Mandatory analyses")
for eid in MANDATORY_IDS:
    entry = entries_by_id.get(eid)
    if entry is None:
        continue
    st.markdown(f"#### {entry['id']} — {entry['title']}")
    _render_figures(entry.get("figure_names", []))
    sentences = insights_by_id.get(eid, [])
    if sentences:
        for sentence in sentences:
            st.markdown(f"- {sentence}")
    else:
        st.caption("No matching insight sentence found for this analysis.")
    st.divider()

other_entries = [entry for entry in index_entries if entry["id"] not in MANDATORY_IDS]
if other_entries:
    with st.expander(f"Other analyses ({len(other_entries)})"):
        for entry in other_entries:
            st.markdown(f"**{entry['id']} — {entry['title']}**")
            for sentence in insights_by_id.get(entry["id"], []):
                st.caption(sentence)

st.subheader("EDA tables")
all_table_names = sorted({name for entry in index_entries for name in entry["table_names"]})
table_names = _available_table_names(all_table_names)
if not table_names:
    st.info("No EDA tables are available.")
else:
    selected_table = st.selectbox("Select a table", options=table_names)
    table_df = data_access.load_eda_table(selected_table)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download table CSV",
        data=table_df.to_csv(index=False),
        file_name=f"{selected_table}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download all EDA tables (zip)",
        data=_tables_zip(table_names),
        file_name="eda_tables.zip",
        mime="application/zip",
    )

st.subheader("Full report")
try:
    eda_report_html = data_access.load_text(paths.EDA_REPORT)
    st.download_button(
        "Download eda_report.html",
        data=eda_report_html,
        file_name="eda_report.html",
        mime="text/html",
    )
except data_access.ArtifactMissing as exc:
    st.warning(f"eda_report.html unavailable: {exc}")

st.divider()
data_sources = load_data_sources()
st.caption(f"Source: {data_sources.citation}")
st.caption(
    "Every number above comes from a computed table under `artifacts/reports/eda_tables/` "
    "(PRD §35A.2)."
)
