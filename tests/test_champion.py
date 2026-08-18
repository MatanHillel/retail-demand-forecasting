"""Champion-selection tests (US-22, PRD §20, §55).

Fixtures build minimal ``overall_df`` / ``by_month_df`` / ``kpis_df`` frames directly — only the
columns :func:`pipeline.champion.select_champion` actually reads — mirroring the worked examples
in the issue's own test list (AI-27 §2) rather than re-deriving them from the real hold-out. Real
config (``load_model_config()``, ``load_inventory_policy()``) is used throughout, never hand-rolled
thresholds, so a config change is felt here rather than silently diverging (same convention as
``tests/test_inventory.py``).

wMAPE/bias are fractions here (``0.53``), and every ``champion_gates`` threshold is in percentage
points of that fraction (module docstring) — a "3 point" improvement in a fixture is written as a
0.03 gap between two wmape fractions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline import paths
from pipeline.champion import (
    ChampionDecision,
    _statement,
    select_champion,
)
from pipeline.config import MODEL_IDS, load_inventory_policy, load_model_config
from pipeline.run_context import RunContext, close_log_handlers

CFG = load_model_config()
GATES = CFG.champion_gates
Z = load_inventory_policy().z


def _overall_df(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if "coverage_share" not in frame.columns:
        frame["coverage_share"] = 1.0
    return frame


def _by_month_df(entries: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"model": model, "target_month": month, "bias": bias} for model, month, bias in entries],
        columns=["model", "target_month", "bias"],
    )


def _kpi_row(model: str, fill_rate: float, excess_units: float, z: float = Z) -> dict:
    return {
        "model": model,
        "policy": "forecast_plus_ss",
        "z": z,
        "scope": "overall",
        "group": "all",
        "fill_rate": fill_rate,
        "excess_units": excess_units,
    }


def _run_ctx(tmp_path):
    return RunContext.start(mode="no-llm", base_dir=tmp_path)


# --------------------------------------------------------------------------
# gate 1 — bias, and its independent monthly report (issue §2)
# --------------------------------------------------------------------------
def test_m4_like_fails_gate1_and_is_not_chosen(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "B1_last_month", "wmape": 0.56, "bias": 0.05},
            {"model": "B2_ma3", "wmape": 0.55, "bias": -0.17},
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},
            {"model": "M4_gbm_absolute", "wmape": 0.50, "bias": -0.24},
        ]
    )
    by_month = _by_month_df([])
    kpi_models = ["B1_last_month", "B2_ma3", "M2_gbm_poisson", "M4_gbm_absolute"]
    kpis = pd.DataFrame([_kpi_row(m, 0.9, 1.0) for m in kpi_models])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        by_model = {c.model: c for c in decision.candidates}

        assert by_model["M4_gbm_absolute"].gate1_pass is False
        assert decision.champion != "M4_gbm_absolute"
        # M2 beats B2 (fails gate 1) and B1 (only gate-1-passing baseline)
        assert decision.champion == "M2_gbm_poisson"
        assert decision.champion_kind == "ml"
        assert decision.best_baseline == "B1_last_month"
        assert decision.improvement_points == pytest.approx(3.0, abs=1e-9)
        assert decision.meaningful_improvement is True
    finally:
        close_log_handlers(ctx.run_id)


def test_monthly_bias_over_threshold_is_reported_but_does_not_fail_gate1(tmp_path) -> None:
    overall = _overall_df([{"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02}])
    by_month = _by_month_df(
        [
            ("M2_gbm_poisson", "2011-06", 0.05),
            ("M2_gbm_poisson", "2011-08", 0.30),  # > monthly_abs_bias_report_threshold (0.25)
        ]
    )
    kpis = pd.DataFrame([_kpi_row("M2_gbm_poisson", 0.9, 1.0)])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        candidate = next(c for c in decision.candidates if c.model == "M2_gbm_poisson")
        assert candidate.gate1_pass is True
        assert candidate.months_over_bias_threshold == ["2011-08"]
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# gate 3 — inventory tie-break, only inside the tie band
# --------------------------------------------------------------------------
def test_gate3_prefers_lower_excess_when_within_tie_band_and_similar_fill_rate(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "M2_gbm_poisson", "wmape": 0.530, "bias": 0.02},  # gate-2 leader
            # 0.5 pt behind the leader -> inside the tie band
            {"model": "M3_gbm_squared", "wmape": 0.535, "bias": 0.01},
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame(
        [
            _kpi_row("M2_gbm_poisson", fill_rate=0.900, excess_units=150_000.0),
            _kpi_row("M3_gbm_squared", fill_rate=0.898, excess_units=100_000.0),  # less excess
        ]
    )
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        # M3 has higher wMAPE but wins gate 3 on lower excess at a similar fill rate.
        assert decision.champion == "M3_gbm_squared"
        m3 = next(c for c in decision.candidates if c.model == "M3_gbm_squared")
        m2 = next(c for c in decision.candidates if c.model == "M2_gbm_poisson")
        assert m3.gate3_decision == "lower_excess"
        assert m2.gate3_decision == "lower_excess"
        assert m3.gate3_compared_with == "M2_gbm_poisson"
    finally:
        close_log_handlers(ctx.run_id)


def test_gate3_is_a_noop_outside_the_tie_band(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},
            # 2 pts behind the leader -> outside the tie band
            {"model": "M3_gbm_squared", "wmape": 0.55, "bias": 0.01},
        ]
    )
    by_month = _by_month_df([])
    # M3 would win a lower-excess comparison if it were even considered.
    kpis = pd.DataFrame(
        [
            _kpi_row("M2_gbm_poisson", fill_rate=0.900, excess_units=150_000.0),
            _kpi_row("M3_gbm_squared", fill_rate=0.899, excess_units=1_000.0),
        ]
    )
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        assert decision.champion == "M2_gbm_poisson"
        m3 = next(c for c in decision.candidates if c.model == "M3_gbm_squared")
        assert m3.gate3_decision is None
        assert m3.gate3_compared_with is None
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# gate 4 — meaningful improvement, both outcomes
# --------------------------------------------------------------------------
def test_improvement_below_threshold_is_reported_as_simple_methods_competitive(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "B1_last_month", "wmape": 0.550, "bias": 0.05},
            {"model": "M2_gbm_poisson", "wmape": 0.535, "bias": 0.02},  # 1.5 pts -> below 2.0
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame(
        [_kpi_row(m, 0.9, 1.0) for m in ["B1_last_month", "M2_gbm_poisson"]]
    )
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        assert decision.champion == "M2_gbm_poisson"
        assert decision.improvement_points == pytest.approx(1.5, abs=1e-9)
        assert decision.meaningful_improvement is False

        statement = _statement(decision)
        assert statement is not None
        assert "Simple methods are competitive" in statement
        assert "M2_gbm_poisson" in statement
        assert "B1_last_month" in statement
    finally:
        close_log_handlers(ctx.run_id)


def test_statement_is_none_when_improvement_is_meaningful_or_champion_is_a_baseline() -> None:
    meaningful = ChampionDecision(
        champion="M2_gbm_poisson",
        champion_kind="ml",
        best_baseline="B1_last_month",
        meaningful_improvement=True,
        improvement_points=3.0,
        gate1_all_failed=False,
        candidates=[],
        rules_version="test",
        config_gates={"meaningful_improvement_points": 2.0},
        generated_at="2026-01-01T00:00:00+00:00",
        run_id="test-run",
    )
    assert _statement(meaningful) is None

    baseline_champion = meaningful.model_copy(
        update={
            "champion_kind": "baseline",
            "meaningful_improvement": None,
            "improvement_points": None,
        }
    )
    assert _statement(baseline_champion) is None


# --------------------------------------------------------------------------
# edge case — no candidate passes gate 1
# --------------------------------------------------------------------------
def test_no_candidate_passes_gate1_champion_by_smallest_bias_among_two_lowest_wmape(
    tmp_path,
) -> None:
    overall = _overall_df(
        [
            {"model": "B1_last_month", "wmape": 0.55, "bias": 0.15},
            {"model": "B2_ma3", "wmape": 0.54, "bias": -0.20},  # lowest wMAPE
            {"model": "M2_gbm_poisson", "wmape": 0.60, "bias": 0.30},
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame(
        [_kpi_row(m, 0.9, 1.0) for m in ["B1_last_month", "B2_ma3", "M2_gbm_poisson"]]
    )
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        assert decision.gate1_all_failed is True
        # two lowest wMAPE: B2 (|bias|=0.20) and B1 (|bias|=0.15) -> B1 wins on smaller |bias|
        assert decision.champion == "B1_last_month"
        assert any("no candidate passed gate 1" in warning for warning in ctx.warnings)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# edge case — a baseline can legitimately win
# --------------------------------------------------------------------------
def test_a_baseline_can_win(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "B2_ma3", "wmape": 0.50, "bias": 0.05},  # passes, lowest wMAPE outright
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},  # passes, 3 pts behind
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame([_kpi_row(m, 0.9, 1.0) for m in ["B2_ma3", "M2_gbm_poisson"]])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        assert decision.champion == "B2_ma3"
        assert decision.champion_kind == "baseline"
        assert decision.best_baseline == "B2_ma3"
        # gate 4 is not applicable to a baseline champion
        assert decision.meaningful_improvement is None
        assert decision.improvement_points is None
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# B3 participation — coverage_share gates inclusion (issue §2)
# --------------------------------------------------------------------------
def test_b3_is_excluded_reference_when_coverage_is_partial(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},
            {"model": "B3_seasonal_naive", "wmape": 0.90, "bias": 0.35, "coverage_share": 0.83},
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame([_kpi_row("M2_gbm_poisson", 0.9, 1.0)])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        b3 = next(c for c in decision.candidates if c.model == "B3_seasonal_naive")
        assert b3.excluded_reason is not None
        assert b3.gate2_rank is None
        assert decision.champion != "B3_seasonal_naive"
    finally:
        close_log_handlers(ctx.run_id)


def test_b3_participates_when_coverage_is_complete(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "B3_seasonal_naive", "wmape": 0.40, "bias": 0.02, "coverage_share": 1.0},
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame([_kpi_row(m, 0.9, 1.0) for m in ["B3_seasonal_naive", "M2_gbm_poisson"]])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        b3 = next(c for c in decision.candidates if c.model == "B3_seasonal_naive")
        assert b3.excluded_reason is None
        assert decision.champion == "B3_seasonal_naive"
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# schema — every published key present (issue §2 "Outputs")
# --------------------------------------------------------------------------
def test_champion_decision_schema_keys_present(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": model_id, "wmape": 0.5 + i * 0.01, "bias": 0.02}
            for i, model_id in enumerate(MODEL_IDS)
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame([_kpi_row(model_id, 0.9, 1.0) for model_id in MODEL_IDS])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        payload = decision.model_dump(mode="json")
        expected_keys = {
            "champion",
            "champion_kind",
            "best_baseline",
            "meaningful_improvement",
            "improvement_points",
            "gate1_all_failed",
            "candidates",
            "rules_version",
            "config_gates",
            "generated_at",
            "run_id",
        }
        assert expected_keys.issubset(payload.keys())

        candidate_keys = {
            "model",
            "wmape",
            "bias",
            "gate1_pass",
            "months_over_bias_threshold",
            "gate2_rank",
            "gate3_compared_with",
            "gate3_fill_rate",
            "gate3_excess_units",
            "gate3_decision",
            "excluded_reason",
        }
        for candidate in payload["candidates"]:
            assert candidate_keys.issubset(candidate.keys())

        # all seven candidates appear in the trace (issue §6 AC)
        assert {c["model"] for c in payload["candidates"]} == set(MODEL_IDS)
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# ctx integration — champion_decision.json, ctx.champion, run_log metrics
# --------------------------------------------------------------------------
def test_writes_champion_decision_and_populates_ctx_champion(tmp_path) -> None:
    overall = _overall_df(
        [
            {"model": "B1_last_month", "wmape": 0.56, "bias": 0.05},
            {"model": "M2_gbm_poisson", "wmape": 0.53, "bias": 0.02},
        ]
    )
    by_month = _by_month_df([])
    kpis = pd.DataFrame([_kpi_row(m, 0.9, 1.0) for m in ["B1_last_month", "M2_gbm_poisson"]])
    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)
        destination = tmp_path / "artifacts" / "reports" / "champion_decision.json"
        assert destination.is_file()
        assert ctx.artifacts["champion_decision"] == "artifacts/reports/champion_decision.json"
        assert ctx.champion is not None
        assert ctx.champion["champion"] == decision.champion
        champion_row = next(c for c in decision.candidates if c.model == decision.champion)
        assert ctx.metrics["champion_wmape"] == pytest.approx(champion_row.wmape)

        ctx.finish()
        log_path = tmp_path / "artifacts" / "run_log.json"
        assert log_path.is_file()
        import json

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        assert payload["champion"] is not None
        assert payload["champion"]["champion"] == decision.champion
        assert payload["status"] == "success"
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# end-to-end against the real, committed US-19 / US-21 artifacts
# --------------------------------------------------------------------------
def test_select_champion_end_to_end_on_real_artifacts(tmp_path) -> None:
    overall_path = paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv"
    by_month_path = paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv"
    if not (overall_path.is_file() and by_month_path.is_file() and paths.INVENTORY_KPIS.is_file()):
        pytest.skip("US-19 / US-21 artifacts not present in this checkout")

    overall = pd.read_csv(overall_path, dtype={"model": "string"})
    by_month = pd.read_csv(by_month_path, dtype={"model": "string", "target_month": "string"})
    kpis = pd.read_csv(
        paths.INVENTORY_KPIS,
        dtype={"model": "string", "policy": "string", "scope": "string", "group": "string"},
    )

    ctx = _run_ctx(tmp_path)
    try:
        decision = select_champion(overall, by_month, kpis, CFG, ctx)

        assert set(MODEL_IDS).issuperset({c.model for c in decision.candidates})
        assert decision.champion in set(MODEL_IDS)

        destination = tmp_path / "artifacts" / "reports" / "champion_decision.json"
        assert destination.is_file()
    finally:
        close_log_handlers(ctx.run_id)


# --------------------------------------------------------------------------
# guard: no gate literal, no manual override (CLAUDE.md §2 rule 4; issue §6 AC)
# --------------------------------------------------------------------------
def test_forbidden_literals_never_appear_in_champion_module() -> None:
    text = (paths.PROJECT_ROOT / "src" / "pipeline" / "champion.py").read_text(encoding="utf-8")
    for token in ("0.10", "0.25", "1.0", "2.0"):
        assert token not in text, token
    assert "force" not in text.lower()
