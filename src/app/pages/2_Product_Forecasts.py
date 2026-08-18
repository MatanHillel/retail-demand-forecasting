"""Screen 2 — Product Forecasts (US-28, PRD §33.2). Table, filters and CSV download."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import data_access
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.latest_forecast import STATUS_FORECAST

#: "High uncertainty" default threshold (§4 of the issue). Kept as a plain page-level default
#: rather than a new ``inventory_policy.yaml -> app.*`` key: ``InventoryPolicy`` forbids unknown
#: top-level keys, so an unreviewed schema addition here would break the whole pipeline for every
#: config consumer, not just this screen (AI-33 §8 interface correction — this is the sanctioned
#: alternative).
DEFAULT_HIGH_UNCERTAINTY_RATIO = 1.0

st.set_page_config(page_title="Product Forecasts", layout="wide")
apply_theme()
st.title("Product Forecasts")

state = run_status_banner()
if state.status != "success":
    st.stop()

plan = data_access.load_inventory_plan()
latest = data_access.load_latest_forecast()

table = plan.merge(
    latest[["stock_code", "lag_1", "rolling_mean_3"]], on="stock_code", how="left"
)

st.subheader("Filters")
filter_cols = st.columns([2, 1, 1, 1])

with filter_cols[0]:
    search = st.text_input(
        "Search by StockCode or description", value="", help="Case-insensitive substring match."
    )

with filter_cols[1]:
    abc_options = sorted(table["abc_class"].dropna().unique())
    selected_abc = st.multiselect("ABC class", options=abc_options, default=abc_options)

with filter_cols[2]:
    forecast_positive = st.toggle("Forecast > 0", value=False)

with filter_cols[3]:
    high_uncertainty_only = st.toggle(
        "High uncertainty only",
        value=False,
        help="sigma_source != product OR uncertainty_ratio >= threshold",
    )

if high_uncertainty_only:
    uncertainty_threshold = st.slider(
        "High-uncertainty ratio threshold",
        min_value=0.1,
        max_value=3.0,
        value=DEFAULT_HIGH_UNCERTAINTY_RATIO,
        step=0.1,
        help="A product counts as high uncertainty when its sigma was not measured from the "
        "product itself, or when sigma / forecast is at least this ratio.",
    )
else:
    uncertainty_threshold = DEFAULT_HIGH_UNCERTAINTY_RATIO

status_options = sorted(table["status"].dropna().unique())
selected_status = st.multiselect(
    "Status",
    options=status_options,
    default=status_options,
    help='Product status: "Forecast" (an operational forecast is available), an inactive status '
    "(no sales in the configured lookback window), or "
    '"Insufficient History / New Product" (no sales observed at or before the forecast origin).',
)

mask = pd.Series(True, index=table.index)
if search:
    needle = search.strip().lower()
    mask &= (
        table["stock_code"].str.lower().str.contains(needle, na=False)
        | table["description"].str.lower().str.contains(needle, na=False)
    )
mask &= table["abc_class"].isin(selected_abc)
mask &= table["status"].isin(selected_status)
if forecast_positive:
    mask &= table["forecast"] > 0
if high_uncertainty_only:
    mask &= (table["status"] == STATUS_FORECAST) & (
        (table["sigma_source"] != "product") | (table["uncertainty_ratio"] >= uncertainty_threshold)
    )

filtered = table.loc[mask].reset_index(drop=True)

st.caption(f"{len(filtered):,} of {len(table):,} products match the current filters.")

if filtered.empty:
    st.info("No products match the current filters.")
else:
    display = pd.DataFrame(
        {
            "Product": filtered["stock_code"],
            "Description": filtered["description"],
            "Last Month": filtered["lag_1"],
            "3M Avg": filtered["rolling_mean_3"],
            "Forecast": filtered["forecast"],
            "Safety Stock": filtered["safety_stock"],
            "Recommended Target Inventory": filtered["target_inventory"],
            "Sigma source": filtered["sigma_source"],
            "ABC": filtered["abc_class"],
            "Status": filtered["status"],
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Last Month": st.column_config.NumberColumn(format="%d"),
            "3M Avg": st.column_config.NumberColumn(format="%.1f"),
            "Forecast": st.column_config.NumberColumn(format="%.1f"),
            "Safety Stock": st.column_config.NumberColumn(format="%.1f"),
            "Recommended Target Inventory": st.column_config.NumberColumn(format="%d"),
        },
    )

    run_log = data_access.load_run_log()
    run_id = run_log["run_id"]
    export_columns = [
        "stock_code",
        "description",
        "lag_1",
        "rolling_mean_3",
        "forecast",
        "safety_stock",
        "target_inventory",
        "sigma_source",
        "abc_class",
        "status",
        "forecast_origin",
        "target_month",
        "sigma",
        "z",
        "run_id",
    ]
    export_df = filtered.reindex(columns=export_columns)
    st.download_button(
        "Download CSV",
        data=export_df.to_csv(index=False),
        file_name=f"product_forecasts_{run_id}.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Open a product")
    codes = filtered["stock_code"].tolist()
    descriptions = dict(zip(filtered["stock_code"], filtered["description"], strict=False))
    selected_code = st.selectbox(
        "Select a product to view its detail page",
        options=codes,
        format_func=lambda code: f"{code} — {descriptions.get(code, '')}",
    )
    if st.button("Open Product Detail →"):
        st.session_state["selected_stock_code"] = selected_code
        st.switch_page("pages/3_Product_Detail.py")
