"""Screen 1 — Executive Dashboard (US-27, PRD §33.1). Entry point: streamlit run src/app/Home.py."""

from __future__ import annotations

import numpy as np
import streamlit as st

from app import data_access
from app.components.charts import monthly_demand_line, seasonal_index_bar
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.config import load_inventory_policy

st.set_page_config(page_title="Retail Demand Forecasting & Inventory Planning", layout="wide")
apply_theme()
st.title("Retail Demand Forecasting & Inventory Planning")

state = run_status_banner()
if state.status != "success":
    st.stop()

plan = data_access.load_inventory_plan()
active = plan.loc[plan["status"] == "Forecast"]

col1, col2, col3 = st.columns(3)
col1.metric("Active products", f"{len(active):,}")
col2.metric("Total next-month forecast (units)", f"{active['forecast'].sum():,.0f}")
col3.metric("Recommended Target Inventory (units)", f"{active['target_inventory'].sum():,.0f}")

champion_decision = data_access.load_champion_decision()
champion_id = champion_decision.get("champion")
holdout = data_access.load_eval_table("holdout_metrics_overall")
champion_row = holdout.loc[holdout["model"] == champion_id]

col4, col5, col6 = st.columns(3)
col4.metric("Champion model", champion_id or "—")
if not champion_row.empty:
    row = champion_row.iloc[0]
    col5.metric("Hold-out wMAPE", f"{row['wmape']:.1%}")
    col6.metric("Hold-out Bias", f"{row['bias']:.1%}")

policy = load_inventory_policy()
kpis = data_access.load_inventory_kpis()
kpi_row = kpis.loc[
    (kpis["model"] == champion_id)
    & (kpis["policy"] == "forecast_plus_ss")
    & (kpis["scope"] == "overall")
    & (np.isclose(kpis["z"], policy.z))
]

col7, col8, col9 = st.columns(3)
if not kpi_row.empty:
    row = kpi_row.iloc[0]
    col7.metric("Hold-out fill rate", f"{row['fill_rate']:.1%}")
    col8.metric("Hold-out stockout units", f"{row['stockout_units']:,.0f}")
    col9.metric("Hold-out excess units", f"{row['excess_units']:,.0f}")

st.divider()
st.subheader("Monthly demand")
st.pyplot(monthly_demand_line(data_access.load_eda_table("E02_monthly_units_revenue")))
st.subheader("Seasonality")
st.pyplot(seasonal_index_bar(data_access.load_eda_table("E02_seasonal_index")))

with st.expander("How to read this dashboard"):
    st.markdown(
        "This system analyses historical sales, predicts how many units each active product "
        "will sell next month, and converts that forecast into a Recommended Target Inventory "
        "using safety stock derived from past forecast errors."
    )
    st.caption(policy.disclaimer)
    st.markdown("Recommendations rely only on historical sales in this dataset (PRD §48).")
