"""Screen 4 — Model Evaluation (US-29, PRD §33.4). All candidates, by month / ABC, actual vs
forecast, back-test consistency and the champion decision trace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app import data_access
from app.components.charts import (
    actual_vs_forecast_monthly,
    actual_vs_forecast_scatter,
    model_abc_grouped_bars,
    model_line_chart,
)
from app.components.status import run_status_banner
from app.components.theme import apply_theme
from pipeline.config import MODEL_IDS, load_model_config
from pipeline.metrics import format_pct

#: Sample size for the actual-vs-forecast scatter (a page-level display default, not a modelling
#: parameter — see AI-33's DEFAULT_HIGH_UNCERTAINTY_RATIO precedent for why this isn't a config
#: key).
SCATTER_SAMPLE_SIZE = 2000

st.set_page_config(page_title="Model Evaluation", layout="wide")
apply_theme()
st.title("Model Evaluation")

state = run_status_banner()
if state.status != "success":
    st.stop()

model_cfg = load_model_config()
gates = model_cfg.champion_gates
split = model_cfg.split


def _in_holdout(months: pd.Series) -> pd.Series:
    return (
        (months >= split.holdout_targets.start)
        & (months <= split.holdout_targets.end)
        & ~months.isin(split.never_score)
    )


try:
    champion_decision = data_access.load_champion_decision()
    champion_id = champion_decision.get("champion")
except data_access.ArtifactMissing:
    champion_decision = None
    champion_id = None

st.subheader("All candidates — hold-out overall metrics")
try:
    overall = data_access.load_eval_table("holdout_metrics_overall")
    improvement = data_access.load_eval_table("improvement_vs_b2")
    table = overall.merge(
        improvement[["model", "wmape_points_vs_b2", "relative_pct", "meaningful"]],
        on="model",
        how="left",
    )
    table["model"] = pd.Categorical(table["model"], categories=list(MODEL_IDS), ordered=True)
    table = table.sort_values("model").reset_index(drop=True)
    table["model"] = table["model"].astype(str)

    model_label = table["model"] + np.where(table["model"] == champion_id, "  ← champion", "")
    display = pd.DataFrame(
        {
            "Model": model_label,
            "wMAPE": table["wmape"].map(format_pct),
            "Bias": table["bias"].map(format_pct),
            "MAE": table["mae"].round(1),
            "RMSE": table["rmse"].round(1),
            "n rows": table["n_rows"],
            "Negative share (M1)": table["negative_share_raw"].map(format_pct),
            "Coverage (B3)": table["coverage_share"].map(format_pct),
            "vs B2 (points)": (table["wmape_points_vs_b2"] * 100).round(2),
            "vs B2 (relative)": table["relative_pct"].map(format_pct),
            "Meaningful vs B2": table["meaningful"],
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    notes = table.loc[table["note"].fillna("") != "", ["model", "note"]]
    for _, note_row in notes.iterrows():
        st.caption(f"{note_row['model']}: {note_row['note']}")
    if champion_id:
        st.caption(
            f"Champion: **{champion_id}**. wMAPE and Bias are always reported together (§23)."
        )
except data_access.ArtifactMissing as exc:
    st.warning(f"Candidate table unavailable: {exc}")

tab_month, tab_abc, tab_actual, tab_backtest, tab_trace = st.tabs(
    ["By month", "By ABC", "Actual vs forecast", "Back-test consistency", "Champion decision trace"]
)

with tab_month:
    try:
        by_month = data_access.load_eval_table("holdout_metrics_by_month")
        by_month = by_month.loc[_in_holdout(by_month["target_month"])]
        st.pyplot(
            model_line_chart(
                by_month,
                "wmape",
                x_col="target_month",
                ylabel="wMAPE (share of actual demand)",
                title="wMAPE by hold-out month, per model",
            )
        )
        st.pyplot(
            model_line_chart(
                by_month,
                "bias",
                x_col="target_month",
                ylabel="Bias (share of actual demand)",
                title="Bias by hold-out month, per model",
            )
        )
        st.caption(
            f"Hold-out window: {split.holdout_targets.start} to {split.holdout_targets.end}. "
            f"Never scored: {', '.join(split.never_score)}."
        )
    except data_access.ArtifactMissing as exc:
        st.warning(f"By-month metrics unavailable: {exc}")

with tab_abc:
    try:
        by_abc = data_access.load_eval_table("holdout_metrics_by_abc")
        st.pyplot(
            model_abc_grouped_bars(
                by_abc,
                "wmape",
                ylabel="wMAPE (share of actual demand)",
                title="wMAPE by ABC class, per model",
            )
        )
        st.caption("ABC computed on the training window (PRD §18.2, §27).")
    except data_access.ArtifactMissing as exc:
        st.warning(f"By-ABC metrics unavailable: {exc}")

with tab_actual:
    try:
        rows_all = data_access.load_eval_table("holdout_rows_all_models")
        pred_models = [
            model_id for model_id in MODEL_IDS if f"pred_{model_id}" in rows_all.columns
        ]
        default_model = champion_id if champion_id in pred_models else pred_models[0]
        selected_model = st.selectbox(
            "Model", options=pred_models, index=pred_models.index(default_model)
        )
        st.pyplot(actual_vs_forecast_monthly(rows_all, model_id=selected_model))
        st.pyplot(
            actual_vs_forecast_scatter(
                rows_all,
                model_id=selected_model,
                sample_size=SCATTER_SAMPLE_SIZE,
                seed=model_cfg.seed,
            )
        )
    except data_access.ArtifactMissing as exc:
        st.warning(f"Actual-vs-forecast data unavailable: {exc}")

with tab_backtest:
    try:
        consistency = data_access.load_eval_table("backtest_consistency")
        st.pyplot(
            model_line_chart(
                consistency,
                "wmape",
                x_col="forecast_origin",
                ylabel="wMAPE (share of actual demand)",
                title="Back-test wMAPE per rolling origin, per model",
            )
        )
        st.caption(
            f"Rolling origins {model_cfg.backtest.first_origin} to "
            f"{model_cfg.backtest.last_origin} (§22 back-test consistency)."
        )
    except data_access.ArtifactMissing as exc:
        st.warning(f"Back-test consistency data unavailable: {exc}")

with tab_trace:
    if champion_decision is None:
        st.warning("Champion decision trace unavailable: champion_decision.json is missing.")
    else:
        candidates = pd.DataFrame(champion_decision.get("candidates", []))
        candidates["model"] = pd.Categorical(
            candidates["model"], categories=list(MODEL_IDS), ordered=True
        )
        candidates = candidates.sort_values("model").reset_index(drop=True)
        candidates["model"] = candidates["model"].astype(str)
        months_over_threshold_col = (
            f"Months |bias| > {format_pct(gates.monthly_abs_bias_report_threshold)}"
        )
        trace_display = pd.DataFrame(
            {
                "Model": candidates["model"],
                "Gate 1 (bias) pass": candidates["gate1_pass"],
                "Bias": candidates["bias"].map(format_pct),
                months_over_threshold_col: candidates["months_over_bias_threshold"].map(
                    lambda months: ", ".join(months) if months else "—"
                ),
                "Gate 2 rank (wMAPE)": candidates["gate2_rank"],
                "Gate 3 compared with": candidates["gate3_compared_with"].fillna("—"),
                "Gate 3 decision": candidates["gate3_decision"].fillna("—"),
                "Excluded reason": candidates["excluded_reason"].fillna("—"),
            }
        )
        st.dataframe(trace_display, use_container_width=True, hide_index=True)

        best_baseline = champion_decision.get("best_baseline")
        improvement_points = champion_decision.get("improvement_points")
        meaningful = champion_decision.get("meaningful_improvement")
        champion_kind = champion_decision.get("champion_kind")
        statement = (
            f"Champion: **{champion_id}** (best gate-passing baseline: {best_baseline}). "
            f"Improvement over the best baseline: {improvement_points:.2f} wMAPE points "
            f"(meaningful-improvement threshold: {gates.meaningful_improvement_points:g} points)."
        )
        if champion_kind == "baseline" or meaningful is False:
            statement += (
                " Simple methods are competitive — this is a legitimate, expected outcome and is "
                "reported plainly (PRD §20)."
            )
        st.markdown(statement)

with st.expander("How to read wMAPE and Bias"):
    st.markdown(
        "**wMAPE** = sum of |actual − forecast| divided by sum of actual — the average error size "
        "as a share of total demand; 0% is a perfect forecast.\n\n"
        "**Bias** = sum of (forecast − actual) divided by sum of actual — positive means the model "
        "over-forecasts on average, negative means it under-forecasts. wMAPE and Bias are always "
        "shown together (PRD §23) because a low wMAPE can hide a large systematic bias.\n\n"
        f"**Gate 1 (bias):** a candidate passes only if |Bias| ≤ {format_pct(gates.max_abs_bias)} "
        "on the hold-out; months with |Bias| > "
        f"{format_pct(gates.monthly_abs_bias_report_threshold)} are reported but do not by "
        "themselves fail the gate.\n\n"
        "**Gate 2 (accuracy):** the lowest wMAPE among gate-1 passers.\n\n"
        f"**Gate 3 (inventory tie-break):** within {gates.tie_wmape_points:g} wMAPE points, prefer "
        "less excess inventory at a similar fill rate.\n\n"
        f"**Meaningful improvement:** an ML model only counts as improving on the baseline if it "
        f"beats the best gate-passing baseline by at least {gates.meaningful_improvement_points:g} "
        "wMAPE points."
    )
