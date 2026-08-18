"""Screen 5 — Inventory Policy Evaluation (US-29, PRD §33.5). Forecast-only vs forecast + robust
safety stock, ML vs baseline, with a z slider (§25)."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app import data_access
from app.components.charts import (
    fill_rate_vs_excess_curve,
    inventory_group_bars,
    inventory_kpi_bar_pairs,
)
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.config import MODEL_IDS, load_inventory_policy, load_model_config
from pipeline.eda.style import ABC_COLORS
from pipeline.inventory import (
    POLICIES,
    POLICY_FORECAST_ONLY,
    POLICY_FORECAST_PLUS_SS,
    ROWS_INPUT_COLUMNS,
    SCOPE_ABC,
    SCOPE_MONTH,
    SCOPE_OVERALL,
    simulate_inventory,
)
from pipeline.inventory import (
    inventory_kpis as compute_kpis,
)
from pipeline.metrics import format_pct

POLICY_LABELS: dict[str, str] = {
    POLICY_FORECAST_ONLY: "Forecast only",
    POLICY_FORECAST_PLUS_SS: "Forecast + robust safety stock",
}
KPI_METRICS: tuple[tuple[str, str], ...] = (
    ("fill_rate", "Fill rate"),
    ("stockout_units", "Stockout units"),
    ("excess_units", "Excess units"),
    ("stockout_skumonth_rate", "Stockout SKU-month rate"),
    ("excess_per_unit_shortage", "Excess per unit of shortage"),
)
KPI_FORMAT = {
    "fill_rate": format_pct,
    "stockout_skumonth_rate": format_pct,
    "stockout_units": lambda v: f"{v:,.1f}",
    "excess_units": lambda v: f"{v:,.1f}",
    "excess_per_unit_shortage": lambda v: "NaN" if pd.isna(v) else f"{v:.2f}",
}
SCOPE_LABELS = {SCOPE_OVERALL: "Overall", SCOPE_MONTH: "By month", SCOPE_ABC: "By ABC"}

st.set_page_config(page_title="Inventory Policy Evaluation", layout="wide")
apply_theme()
st.title("Inventory Policy Evaluation")

state = run_status_banner()
if state.status != "success":
    st.stop()

policy = load_inventory_policy()
model_cfg = load_model_config()

try:
    champion_id = data_access.load_champion_decision().get("champion")
except data_access.ArtifactMissing:
    champion_id = None

if champion_id == model_cfg.main_baseline_id:
    baseline_default = next(b for b in model_cfg.baseline_ids if b != champion_id)
else:
    baseline_default = model_cfg.main_baseline_id
ml_default = champion_id if champion_id else model_cfg.primary_model_id

st.subheader("Controls")
control_cols = st.columns([2, 1, 1])
with control_cols[0]:
    z_choice = st.select_slider(
        "z (safety-stock service level)",
        options=policy.z_options,
        value=policy.z,
        help="Higher z widens the safety-stock band. z = 1.645 does not by itself guarantee a 95% "
        "fill rate — the achieved fill rate is measured empirically in the back-test.",
    )
with control_cols[1]:
    model_a = st.selectbox(
        "Model A", options=list(MODEL_IDS), index=list(MODEL_IDS).index(ml_default)
    )
with control_cols[2]:
    model_b = st.selectbox(
        "Model B", options=list(MODEL_IDS), index=list(MODEL_IDS).index(baseline_default)
    )

with st.expander("What-if: custom z"):
    use_custom_z = st.checkbox("Use a custom z instead of the slider above", value=False)
    custom_z = st.number_input(
        "Custom z",
        min_value=0.01,
        value=float(z_choice),
        step=0.05,
        help="Any z, not just the precomputed options — recomputed live from the stored hold-out "
        "forecast and sigma of each row (no precomputed table covers this value).",
    )
effective_z = float(custom_z) if use_custom_z else float(z_choice)
is_grid_z = any(math.isclose(effective_z, option) for option in policy.z_options)

policy_cols = st.columns(2)
with policy_cols[0]:
    policies_selected = st.multiselect(
        "Policies to compare",
        options=list(POLICIES),
        default=list(POLICIES),
        format_func=lambda p: POLICY_LABELS[p],
    )
with policy_cols[1]:
    scope = st.radio(
        "Scope",
        options=[SCOPE_OVERALL, SCOPE_MONTH, SCOPE_ABC],
        format_func=SCOPE_LABELS.get,
        horizontal=True,
    )

if not policies_selected:
    st.info("Select at least one policy to compare.")
    st.stop()

try:
    kpis_precomputed = data_access.load_inventory_kpis()
except data_access.ArtifactMissing as exc:
    st.warning(f"Inventory KPIs unavailable: {exc}")
    st.stop()


def _rows_for_pair(sim_rows: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Pre-filter the (up to ~350k-row) simulation rows to the two selected models — the
    forecast/sigma/actual of a row are identical across both policies, so one policy's rows are
    enough to reconstruct the pure ``simulate_inventory`` input (issue §3)."""
    subset = sim_rows.loc[
        sim_rows["model"].isin([model_a, model_b]) & (sim_rows["policy"] == POLICY_FORECAST_PLUS_SS)
    ]
    return subset[ROWS_INPUT_COLUMNS].reset_index(drop=True)


def _kpis_for_scope(z: float, scope_key: str) -> pd.DataFrame:
    """Precomputed z_options -> read inventory_kpis.csv; any other z -> recompute via
    pipeline.inventory on the pair-filtered hold-out rows (issue §2/§3 — the only computation
    permitted in the app)."""
    if any(math.isclose(z, option) for option in policy.z_options):
        return kpis_precomputed.loc[
            kpis_precomputed["model"].isin([model_a, model_b])
            & (kpis_precomputed["scope"] == scope_key)
            & kpis_precomputed["z"].apply(lambda value: math.isclose(value, z))
        ].copy()

    try:
        sim_rows = data_access.load_simulation_rows()
    except data_access.ArtifactMissing:
        return pd.DataFrame()
    rows_for_pair = _rows_for_pair(sim_rows, model_a, model_b)
    sim = simulate_inventory(rows_for_pair, z, policy)
    group_col_by_scope = {
        SCOPE_OVERALL: None,
        SCOPE_MONTH: "target_month",
        SCOPE_ABC: "abc_class",
    }
    group_col = group_col_by_scope[scope_key]
    group_cols = [group_col] if group_col else []
    table = compute_kpis(sim, group_cols)
    if group_col:
        table = table.rename(columns={group_col: "group"})
    else:
        table["group"] = "all"
    table["scope"] = scope_key
    return table


scope_kpis = _kpis_for_scope(effective_z, scope)
scope_kpis = scope_kpis.loc[scope_kpis["policy"].isin(policies_selected)]

if not is_grid_z:
    st.caption(
        f"what-if z = {effective_z:g} — recomputed live from holdout_simulation_rows.csv via "
        "pipeline.inventory.simulate_inventory / inventory_kpis, not read from inventory_kpis.csv."
    )

if scope_kpis.empty:
    st.info("No KPI rows available for the current selection.")
else:
    scope_kpis = scope_kpis.copy()
    scope_kpis["series"] = scope_kpis["model"] + " · " + scope_kpis["policy"].map(POLICY_LABELS)

    if scope == SCOPE_OVERALL:
        st.subheader("Overall hold-out KPIs")
        table_display = pd.DataFrame({"Series": scope_kpis["series"]})
        for col, label in KPI_METRICS:
            table_display[label] = scope_kpis[col].map(KPI_FORMAT[col])
        table_display["n rows"] = scope_kpis["n_rows"]
        st.dataframe(table_display, use_container_width=True, hide_index=True)
        st.pyplot(
            inventory_kpi_bar_pairs(
                scope_kpis,
                series_col="series",
                metrics=KPI_METRICS,
                title=f"{model_a} vs {model_b} — hold-out inventory KPIs at z = {effective_z:g}",
            )
        )
    else:
        st.subheader(f"{SCOPE_LABELS[scope]} breakdown")
        metric_col, metric_label = st.selectbox(
            "KPI to break down", options=KPI_METRICS, format_func=lambda item: item[1]
        )
        breakdown_policy = st.selectbox(
            "Policy for the breakdown chart",
            options=policies_selected,
            format_func=lambda p: POLICY_LABELS[p],
        )
        group_col = "target_month" if scope == SCOPE_MONTH else "abc_class"
        table_display = pd.DataFrame(
            {
                "Model": scope_kpis["model"],
                "Policy": scope_kpis["policy"].map(POLICY_LABELS),
                group_col: scope_kpis["group"],
                metric_label: scope_kpis[metric_col].map(KPI_FORMAT[metric_col]),
                "n rows": scope_kpis["n_rows"],
            }
        )
        st.dataframe(table_display, use_container_width=True, hide_index=True)

        breakdown = scope_kpis.loc[scope_kpis["policy"] == breakdown_policy]
        if scope == SCOPE_MONTH:
            group_order = sorted(breakdown["group"].dropna().unique())
            color_map = None
        else:
            group_order = [c for c in ("A", "B", "C") if c in breakdown["group"].unique()]
            color_map = ABC_COLORS
        panels = tuple(
            (model_id, breakdown.loc[breakdown["model"] == model_id])
            for model_id in (model_a, model_b)
        )
        if group_order and all(not sub.empty for _, sub in panels):
            st.pyplot(
                inventory_group_bars(
                    panels,
                    metric_col=metric_col,
                    group_col="group",
                    group_order=group_order,
                    ylabel=metric_label,
                    title=f"{metric_label} by {SCOPE_LABELS[scope].lower()} "
                    f"({POLICY_LABELS[breakdown_policy]}, z = {effective_z:g})",
                    color_map=color_map,
                )
            )

st.subheader("Fill rate vs excess, across z options")
curve_policy = st.selectbox(
    "Policy for the trade-off curve",
    options=policies_selected,
    format_func=lambda p: POLICY_LABELS[p],
    key="curve_policy",
)
curve_df = kpis_precomputed.loc[
    kpis_precomputed["model"].isin([model_a, model_b])
    & (kpis_precomputed["policy"] == curve_policy)
    & (kpis_precomputed["scope"] == SCOPE_OVERALL)
]
if curve_df.empty:
    st.info("No data available for the trade-off curve.")
else:
    st.pyplot(
        fill_rate_vs_excess_curve(curve_df, series_col="model", series_values=(model_a, model_b))
    )

st.subheader("Excess concentration")
try:
    concentration = data_access.load_eval_table("excess_concentration")
    concentration = concentration.loc[
        concentration["model"].isin([model_a, model_b])
        & concentration["policy"].isin(policies_selected)
    ]
    for _, row in concentration.iterrows():
        st.caption(
            f"{row['model']} ({POLICY_LABELS[row['policy']]}): "
            f"{format_pct(row['top_1pct_share'])} of excess comes from the top 1% of SKU-months "
            f"({format_pct(row['top_5pct_share'])} from the top 5%)."
        )
except data_access.ArtifactMissing as exc:
    st.warning(f"Excess-concentration data unavailable: {exc}")

st.divider()
st.caption(policy.disclaimer)
