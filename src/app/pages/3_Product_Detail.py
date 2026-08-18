"""Screen 3 — Product Detail (US-28, PRD §33.3). History chart, back-test forecasts,
± z*sigma band, hatched partial December 2011, inventory recommendation and metadata."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app import data_access
from app.components.charts import product_history_chart
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.config import load_inventory_policy
from pipeline.latest_forecast import STATUS_FORECAST
from pipeline.split import ABC_SOURCE_DEFAULT_NO_HISTORY, ABC_SOURCE_TRAINING_WINDOW

st.set_page_config(page_title="Product Detail", layout="wide")
apply_theme()
st.title("Product Detail")

state = run_status_banner()
if state.status != "success":
    st.stop()

panel = data_access.load_panel()
plan = data_access.load_inventory_plan()

products = panel.groupby("stock_code", sort=True)["description"].first().reset_index()
product_codes = products["stock_code"].tolist()
description_by_code = dict(zip(products["stock_code"], products["description"], strict=False))

if not product_codes:
    st.info("No products found in the current run's panel.")
    st.stop()

requested_code = st.session_state.get("selected_stock_code")
if requested_code in description_by_code:
    default_code = requested_code
else:
    forecast_rows = plan.loc[plan["status"] == STATUS_FORECAST, "stock_code"]
    default_code = forecast_rows.iloc[0] if not forecast_rows.empty else product_codes[0]

default_index = product_codes.index(default_code)
selected_code = st.selectbox(
    "Product (type to search by code or description)",
    options=product_codes,
    index=default_index,
    format_func=lambda code: f"{code} — {description_by_code.get(code, '')}",
)
st.session_state["selected_stock_code"] = selected_code

plan_rows = plan.loc[plan["stock_code"] == selected_code]
if plan_rows.empty:
    st.error(f"Product {selected_code} was not found in the current run's inventory plan.")
    st.stop()
plan_row = plan_rows.iloc[0]
status = plan_row["status"]
description = description_by_code.get(selected_code, plan_row["description"])

policy = load_inventory_policy()
z = st.select_slider(
    "Safety-stock service level (z)",
    options=policy.z_options,
    value=policy.z,
    help="Higher z widens the safety-stock band. z = 1.645 does not by itself guarantee a 95% "
    "fill rate — the achieved fill rate is measured empirically in the back-test.",
)

try:
    champion_id = data_access.load_champion_decision().get("champion")
except data_access.ArtifactMissing:
    champion_id = None

backtest_for_product = pd.DataFrame(columns=["target_month", "prediction"])
if champion_id:
    try:
        backtest_all = data_access.load_backtest(model=champion_id)
        product_backtest = backtest_all.loc[backtest_all["stock_code"] == selected_code]
        backtest_for_product = (
            product_backtest[["target_month", "prediction"]]
            .sort_values("target_month")
            .reset_index(drop=True)
        )
    except data_access.ArtifactMissing:
        st.caption(
            "Back-test forecasts are unavailable for this run (backtest_predictions.csv missing)."
        )
else:
    st.caption("No champion model recorded for this run — back-test forecasts cannot be shown.")

is_forecast = status == STATUS_FORECAST
has_forecast = is_forecast and pd.notna(plan_row["forecast"])
has_sigma = is_forecast and pd.notna(plan_row["sigma"])
next_month = plan_row["target_month"] if is_forecast else None
next_forecast = float(plan_row["forecast"]) if has_forecast else None
next_sigma = float(plan_row["sigma"]) if has_sigma else None

history_columns = ["month", "units_sold", "is_partial_month"]
history = panel.loc[panel["stock_code"] == selected_code, history_columns]
history = history.sort_values("month").reset_index(drop=True)

abc_class = plan_row["abc_class"]

fig = product_history_chart(
    history_df=history,
    backtest_df=backtest_for_product,
    stock_code=selected_code,
    description=str(description),
    abc_class=abc_class if pd.notna(abc_class) else None,
    next_month=next_month,
    next_forecast=next_forecast,
    next_sigma=next_sigma,
    z=z,
)
st.pyplot(fig)
st.caption(
    "December 2011 is shown hatched and labelled 'partial' because it only runs through 9 Dec "
    "and is never scored (PRD §8)."
)
if backtest_for_product.empty:
    st.caption(
        "No back-test forecasts are available for this product — it has too little history to "
        "have been included in the rolling-origin back-test."
    )

st.divider()
st.subheader("Inventory recommendation")
if is_forecast and next_forecast is not None and next_sigma is not None:
    safety_stock_whatif = z * next_sigma
    target_whatif = math.ceil(max(0.0, next_forecast + safety_stock_whatif))
    is_default_z = math.isclose(z, policy.z)
    sigma_source = plan_row["sigma_source"]
    if not is_default_z:
        st.caption(f"what-if (z = {z:g}) — the stored plan uses z = {policy.z:g}")
    st.markdown(
        f"Product **{selected_code}** is expected to sell **{next_forecast:.1f}** units next "
        f"month ({next_month}). Its forecast uncertainty (σ = {next_sigma:.1f}, "
        f"source = {sigma_source}) requires **{safety_stock_whatif:.1f}** units of safety stock "
        f"at z = {z:g}. The **Recommended Target Inventory** for the period is "
        f"**{target_whatif}** units."
    )
    st.caption(policy.disclaimer)
else:
    st.info(f"No operational forecast for this product — status: {status}.")

st.divider()
st.subheader("Product metadata")


def _fmt(value: object, spec: str = "{:.0f}") -> str:
    if pd.isna(value):
        return "—"
    return spec.format(value)


sales_months = history.loc[history["units_sold"] > 0, "month"]
first_sale_month = history["month"].iloc[0] if not history.empty else "—"
last_sale_month = sales_months.max() if not sales_months.empty else "—"

try:
    abc_train = data_access.load_eval_table("abc_train")
    abc_rows = abc_train.loc[abc_train["stock_code"] == selected_code]
    abc_source = abc_rows.iloc[0]["abc_source"] if not abc_rows.empty else None
except data_access.ArtifactMissing:
    abc_source = None

if abc_source == ABC_SOURCE_TRAINING_WINDOW:
    abc_note = "based on revenue in the training window"
elif abc_source == ABC_SOURCE_DEFAULT_NO_HISTORY:
    abc_note = "default class — no sales observed in the training window"
else:
    abc_note = "training-window ABC unavailable for this run"

meta_cols = st.columns(4)
meta_cols[0].metric("Product age (months)", _fmt(plan_row["product_age_months"]))
meta_cols[1].metric("Months since last sale", _fmt(plan_row["months_since_last_sale"]))
meta_cols[2].metric("ABC class", abc_class if pd.notna(abc_class) else "—")
sigma_source_display = plan_row["sigma_source"] if pd.notna(plan_row["sigma_source"]) else "—"
meta_cols[3].metric("Sigma source", sigma_source_display)
st.caption(f"ABC class ({abc_class if pd.notna(abc_class) else '—'}): {abc_note}.")

meta_cols_2 = st.columns(4)
meta_cols_2[0].metric("Residuals used for sigma", _fmt(plan_row["n_residuals_product"]))
meta_cols_2[1].metric("First sale month", first_sale_month)
meta_cols_2[2].metric("Last sale month", last_sale_month)
meta_cols_2[3].metric("Status", status)

st.markdown(f"**Canonical description:** {description}")
st.caption(
    "Status legend: \"Forecast\" — an operational forecast is available; an inactive status — no "
    "sales in the configured lookback window; \"Insufficient History / New Product\" — no sales "
    "observed at or before the forecast origin (cold start, PRD §15)."
)
