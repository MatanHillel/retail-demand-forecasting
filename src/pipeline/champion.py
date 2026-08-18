"""Champion selection — the four §20 gates, executed by code (US-22, PRD §20, §37 step 7, §45,
§46, §49, §52, §57).

The champion is never picked by hand. :func:`select_champion` applies, in strict order, to every
candidate (the three baselines and the four fitted models, §19) — including baselines, which can
and do win (§20: "gates apply to every candidate, baselines included"):

1. **Bias** — ``|bias_overall| <= champion_gates.max_abs_bias``. Candidates that fail are excluded
   from gates 2-4 but stay in the trace with ``gate1_pass=False``. Independently of pass/fail,
   every candidate's hold-out months with
   ``|bias_month| > champion_gates.monthly_abs_bias_report_threshold`` are listed — report-only,
   never a gate-1 failure by themselves.
2. **Accuracy** — gate-1 passers ranked by ascending ``wmape_overall``; the lowest is the
   provisional champion.
3. **Inventory tie-break** — if the running leader and a challenger are within
   ``champion_gates.tie_wmape_points`` (percentage points) of each other, they are compared on
   ``inventory_kpis.csv`` at ``policy=forecast_plus_ss``, the *inventory* policy's own default z
   (``inventory_policy.yaml`` — ``ModelConfig`` has no z field, issue §8) and ``scope=overall``: a
   fill-rate gap within ``champion_gates.similar_fill_rate_tolerance`` hands the win to whichever
   has less ``excess_units``; a larger gap keeps the lower-wMAPE candidate in the lead.
4. **Meaningful improvement** — only assessed when the winner is an ML model: it "improves on the
   baseline" only if it beats the best gate-1-passing baseline by
   ``>= champion_gates.meaningful_improvement_points`` (percentage points). Falling short is a
   legitimate finding ("simple methods are competitive"), never an error, and never changes who won.

wMAPE and bias in the input tables are fractions (``0.53``, not ``53``); every gate threshold is
expressed in "points" i.e. percentage points of that fraction (PRD §20's own wording: "beats ... by
>= 2 wMAPE points") — :data:`_POINTS_PER_UNIT` is the unit conversion between the two, not a
threshold, and stays a plain ``100.0`` rather than a config field for the same reason
:func:`pipeline.metrics.format_pct` does.

Ties are broken deterministically by ``champion_gates.tie_break_order`` (new config field, issue §8)
at the point candidates are ranked, so no separate "post-gate-3" tie-break code path is needed — any
residual tie already sorted deterministically going into gate 3.

No number in this module is hard-coded (CLAUDE.md §2 rule 4) and there is no manual override of
the outcome anywhere in this file (§20) — every threshold comes from
``model_config.yaml -> champion_gates`` or ``inventory_policy.yaml -> z``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from pipeline import paths
from pipeline.baselines import B3
from pipeline.config import MODEL_IDS, ModelConfig, load_inventory_policy, load_model_config
from pipeline.metrics import format_pct
from pipeline.run_context import RunContext

#: Step name on ``ctx.step(...)`` and in log lines.
STEP_NAME = "champion_selection"

#: Recorded on every decision so a later reader knows which rulebook produced it.
RULES_VERSION = "PRD-v1.3-section20"

#: The four fitted models (§19) — everything else in ``MODEL_IDS`` is a baseline.
ML_MODEL_IDS = frozenset({"M1_linear", "M2_gbm_poisson", "M3_gbm_squared", "M4_gbm_absolute"})

#: Gate 3 reads exactly this slice of ``inventory_kpis.csv`` (issue §2).
_KPI_POLICY = "forecast_plus_ss"
_KPI_SCOPE = "overall"
_KPI_GROUP = "all"

#: Unit conversion, not a threshold — see module docstring.
_POINTS_PER_UNIT = 100.0


# --------------------------------------------------------------------------
# ChampionDecision — the published schema (issue §2 "Outputs")
# --------------------------------------------------------------------------
class ChampionCandidate(BaseModel):
    """One row of the decision trace."""

    model_config = ConfigDict(extra="forbid")

    model: str
    wmape: float | None
    bias: float | None
    gate1_pass: bool
    months_over_bias_threshold: list[str] = Field(default_factory=list)
    gate2_rank: int | None = None
    gate3_compared_with: str | None = None
    gate3_fill_rate: float | None = None
    gate3_excess_units: float | None = None
    gate3_decision: str | None = None
    excluded_reason: str | None = None


class ChampionDecision(BaseModel):
    """The full trace, written to ``champion_decision.json`` and mirrored onto ``ctx.champion``."""

    model_config = ConfigDict(extra="forbid")

    champion: str
    champion_kind: Literal["ml", "baseline"]
    best_baseline: str
    meaningful_improvement: bool | None
    improvement_points: float | None
    gate1_all_failed: bool
    candidates: list[ChampionCandidate]
    rules_version: str
    config_gates: dict[str, Any]
    generated_at: str
    run_id: str


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _safe_float(value: Any) -> float | None:
    """``float(value)``, or ``None`` for a missing/NaN metric — never a bare NaN in the schema."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _rank_key(model_id: str, wmape: float | None, tie_rank: dict[str, int]) -> tuple[float, int]:
    """Ascending-wMAPE sort key with a deterministic tie-break (``champion_gates.tie_break_order``).

    A missing wMAPE sorts last (``inf``) rather than raising, so a candidate with no usable metric
    never wins a ranking by accident.
    """
    value = wmape if wmape is not None else float("inf")
    return (value, tie_rank.get(model_id, len(tie_rank)))


def _months_over_threshold(
    by_month_df: pd.DataFrame, model_id: str, threshold: float
) -> list[str]:
    """Hold-out months where ``|bias| > threshold`` for one model — report-only (gate 1
    docstring)."""
    rows = by_month_df.loc[by_month_df["model"] == model_id]
    over = rows.loc[rows["bias"].abs() > threshold, "target_month"]
    return sorted(str(month) for month in over)


def _kpi_lookup(kpis_df: pd.DataFrame, z: float) -> dict[str, dict[str, float]]:
    """``{model: {fill_rate, excess_units}}`` at the inventory policy's default z,
    ``forecast_plus_ss``, ``scope=overall`` — the exact slice gate 3 compares on (issue §2)."""
    mask = (
        (kpis_df["policy"] == _KPI_POLICY)
        & (kpis_df["scope"] == _KPI_SCOPE)
        & (kpis_df["group"] == _KPI_GROUP)
        & np.isclose(kpis_df["z"].to_numpy(dtype=float), z)
    )
    subset = kpis_df.loc[mask]
    return {
        str(row["model"]): {
            "fill_rate": float(row["fill_rate"]),
            "excess_units": float(row["excess_units"]),
        }
        for row in subset.to_dict(orient="records")
    }


def _gate3_compare(
    leader_id: str,
    challenger_id: str,
    kpi_lookup: dict[str, dict[str, float]],
    tolerance: float,
) -> tuple[str, str]:
    """One gate-3 comparison: ``(winner_model_id, decision_label)``.

    ``"no_kpi_data"`` when either side is missing from ``inventory_kpis.csv`` — the leader keeps
    its position rather than the run guessing. ``"kept_lower_wmape"`` when the fill-rate gap
    exceeds ``similar_fill_rate_tolerance`` — the rule never fires, so the lower-wMAPE candidate
    stays ahead.
    ``"lower_excess"`` when the rule does fire — whichever side has less ``excess_units`` wins,
    including the leader itself if it already has the lower excess.
    """
    leader_kpi = kpi_lookup.get(leader_id)
    challenger_kpi = kpi_lookup.get(challenger_id)
    if leader_kpi is None or challenger_kpi is None:
        return leader_id, "no_kpi_data"

    fill_rate_gap = abs(leader_kpi["fill_rate"] - challenger_kpi["fill_rate"])
    if fill_rate_gap > tolerance:
        return leader_id, "kept_lower_wmape"

    if challenger_kpi["excess_units"] < leader_kpi["excess_units"]:
        return challenger_id, "lower_excess"
    return leader_id, "lower_excess"


# --------------------------------------------------------------------------
# select_champion — the public entry point (issue §2 / §8)
# --------------------------------------------------------------------------
def select_champion(
    overall_df: pd.DataFrame,
    by_month_df: pd.DataFrame,
    kpis_df: pd.DataFrame,
    cfg: ModelConfig,
    ctx: RunContext,
) -> ChampionDecision:
    """Apply gates 1-4 to every candidate in ``overall_df`` and return the full decision trace.

    Never re-reads ``paths.*`` mid-run: every input is a DataFrame the caller already has (the
    US-19 / US-21 tables), which is what keeps this safe to call from inside the orchestrated
    Flow, where those files are still sitting under ``artifacts/_staging/<run_id>/``
    (``docs/interfaces.md`` §6 rule 7, issue §8).
    """
    with ctx.step(STEP_NAME):
        gates = cfg.champion_gates
        policy_cfg = load_inventory_policy()
        tie_rank = {model_id: index for index, model_id in enumerate(gates.tie_break_order)}

        overall_by_model = overall_df.set_index("model")
        kpi_lookup = _kpi_lookup(kpis_df, policy_cfg.z)

        candidate_ids = [model_id for model_id in MODEL_IDS if model_id in overall_by_model.index]

        # -- participation: B3 only when its hold-out coverage is complete (issue §2) -----------
        excluded: dict[str, str] = {}
        if B3 in candidate_ids:
            coverage = float(overall_by_model.loc[B3].get("coverage_share", np.nan))
            if not np.isclose(coverage, 1):
                excluded[B3] = (
                    f"reference_only model, partial hold-out coverage (share={coverage:.6f})"
                )
        participating = [model_id for model_id in candidate_ids if model_id not in excluded]

        # -- gate 1: bias (every candidate, participating or not, for the trace) ----------------
        gate1_pass: dict[str, bool] = {}
        wmape_by_model: dict[str, float | None] = {}
        bias_by_model: dict[str, float | None] = {}
        for model_id in candidate_ids:
            row = overall_by_model.loc[model_id]
            wmape_value = _safe_float(row["wmape"])
            bias_value = _safe_float(row["bias"])
            wmape_by_model[model_id] = wmape_value
            bias_by_model[model_id] = bias_value
            gate1_pass[model_id] = bias_value is not None and abs(bias_value) <= gates.max_abs_bias

        passers = [model_id for model_id in participating if gate1_pass[model_id]]
        gate1_all_failed = len(passers) == 0

        # -- gate 2: accuracy ranking, gate-1 passers only ---------------------------------------
        gate2_rank: dict[str, int] = {}
        ranked_passers = sorted(passers, key=lambda m: _rank_key(m, wmape_by_model[m], tie_rank))
        for rank, model_id in enumerate(ranked_passers, start=1):
            gate2_rank[model_id] = rank

        gate3_info: dict[str, dict[str, Any]] = {}

        if gate1_all_failed:
            # edge case: no candidate passes gate 1 (issue §2 edge cases)
            ranked_all = sorted(
                participating, key=lambda m: _rank_key(m, wmape_by_model[m], tie_rank)
            )
            two_lowest = ranked_all[:2]
            champion_id = min(
                two_lowest,
                key=lambda m: (
                    abs(bias_by_model[m]) if bias_by_model[m] is not None else float("inf"),
                    tie_rank.get(m, len(tie_rank)),
                ),
            )
            ctx.warn(
                "no candidate passed gate 1 (bias) — champion selected by smallest |bias| among "
                f"the two lowest-wMAPE candidates ({', '.join(two_lowest)}); selected {champion_id}"
            )
        else:
            # -- gate 3: inventory tie-break, re-based to the running leader --------------------
            leader_id = ranked_passers[0]
            for challenger_id in ranked_passers[1:]:
                gap_points = (
                    abs(wmape_by_model[challenger_id] - wmape_by_model[leader_id])
                    * _POINTS_PER_UNIT
                )
                if gap_points > gates.tie_wmape_points:
                    continue
                winner_id, decision_label = _gate3_compare(
                    leader_id, challenger_id, kpi_lookup, gates.similar_fill_rate_tolerance
                )
                gate3_info[challenger_id] = {
                    "compared_with": leader_id,
                    "fill_rate": kpi_lookup.get(challenger_id, {}).get("fill_rate"),
                    "excess_units": kpi_lookup.get(challenger_id, {}).get("excess_units"),
                    "decision": decision_label,
                }
                gate3_info[leader_id] = {
                    "compared_with": challenger_id,
                    "fill_rate": kpi_lookup.get(leader_id, {}).get("fill_rate"),
                    "excess_units": kpi_lookup.get(leader_id, {}).get("excess_units"),
                    "decision": decision_label,
                }
                if winner_id == challenger_id:
                    leader_id = challenger_id
            champion_id = leader_id

        champion_kind: Literal["ml", "baseline"] = (
            "ml" if champion_id in ML_MODEL_IDS else "baseline"
        )

        # -- best gate-1-passing baseline (gate 4 input; also reported when no baseline passes) --
        baseline_ids = [model_id for model_id in cfg.baseline_ids if model_id in participating]
        passing_baselines = [model_id for model_id in baseline_ids if gate1_pass.get(model_id)]
        if passing_baselines:
            best_baseline = min(
                passing_baselines, key=lambda m: _rank_key(m, wmape_by_model[m], tie_rank)
            )
        elif baseline_ids:
            best_baseline = min(
                baseline_ids, key=lambda m: _rank_key(m, wmape_by_model[m], tie_rank)
            )
            ctx.warn(
                "no baseline passed gate 1 — best_baseline fallback to lowest wMAPE overall: "
                f"{best_baseline}"
            )
        else:
            best_baseline = champion_id  # degenerate: no baseline in this candidate set at all

        # -- gate 4: meaningful improvement, only meaningful for an ML champion ------------------
        if champion_kind == "ml":
            improvement_points = (
                wmape_by_model[best_baseline] - wmape_by_model[champion_id]
            ) * _POINTS_PER_UNIT
            meaningful_improvement = improvement_points >= gates.meaningful_improvement_points
        else:
            improvement_points = None
            meaningful_improvement = None

        # -- assemble the trace, canonical MODEL_IDS order ---------------------------------------
        threshold = gates.monthly_abs_bias_report_threshold
        candidates: list[ChampionCandidate] = []
        for model_id in candidate_ids:
            info = gate3_info.get(model_id, {})
            candidates.append(
                ChampionCandidate(
                    model=model_id,
                    wmape=wmape_by_model[model_id],
                    bias=bias_by_model[model_id],
                    gate1_pass=gate1_pass[model_id],
                    months_over_bias_threshold=_months_over_threshold(
                        by_month_df, model_id, threshold
                    ),
                    gate2_rank=gate2_rank.get(model_id),
                    gate3_compared_with=info.get("compared_with"),
                    gate3_fill_rate=info.get("fill_rate"),
                    gate3_excess_units=info.get("excess_units"),
                    gate3_decision=info.get("decision"),
                    excluded_reason=excluded.get(model_id),
                )
            )

        decision = ChampionDecision(
            champion=champion_id,
            champion_kind=champion_kind,
            best_baseline=best_baseline,
            meaningful_improvement=meaningful_improvement,
            improvement_points=improvement_points,
            gate1_all_failed=gate1_all_failed,
            candidates=candidates,
            rules_version=RULES_VERSION,
            config_gates=gates.model_dump(mode="json"),
            generated_at=datetime.now(UTC).isoformat(),
            run_id=ctx.run_id,
        )

        _write_decision(decision, ctx)
        ctx.champion = decision.model_dump(mode="json")
        ctx.record_metrics(
            {
                "champion_wmape": wmape_by_model[champion_id],
                "champion_bias": bias_by_model[champion_id],
                "improvement_points": improvement_points,
            }
        )

    return decision


def _write_decision(decision: ChampionDecision, ctx: RunContext) -> Path:
    """Write ``champion_decision.json`` through ``ctx.out()`` (``docs/interfaces.md`` §6 rule 1)."""
    canonical = paths.CHAMPION_DECISION
    destination = ctx.out(_repo_relative(canonical))
    payload = json.dumps(
        decision.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False
    )
    destination.write_text(payload + "\n", encoding="utf-8")
    ctx.record_artifact("champion_decision", _repo_relative(canonical))
    return destination


# --------------------------------------------------------------------------
# CLI: python -m pipeline.champion
# --------------------------------------------------------------------------
def _read_overall(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"model": "string"})


def _read_by_month(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"model": "string", "target_month": "string"})


def _read_kpis(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, dtype={"model": "string", "policy": "string", "scope": "string", "group": "string"}
    )


def _statement(decision: ChampionDecision) -> str | None:
    """The §20 gate-4 wording for display only — not part of the JSON schema (issue §2).

    The threshold quoted in the message comes from ``decision.config_gates`` (the exact
    ``champion_gates`` snapshot the run used), never a literal, so the wording never drifts from
    the config that actually decided the outcome.
    """
    if decision.champion_kind != "ml" or decision.improvement_points is None:
        return None
    if decision.meaningful_improvement:
        return None
    threshold = decision.config_gates["meaningful_improvement_points"]
    return (
        f"Simple methods are competitive: {decision.champion} improves on "
        f"{decision.best_baseline} by {decision.improvement_points:.2f} wMAPE points "
        f"(< {threshold:.1f})"
    )


def run() -> int:
    """Standalone entry point: ``python -m pipeline.champion`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        cfg = load_model_config()
        overall_path = paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv"
        by_month_path = paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv"
        for required in (overall_path, by_month_path, paths.INVENTORY_KPIS):
            if not required.is_file():
                raise FileNotFoundError(
                    f"{required} not found. Run `python -m pipeline.evaluate` and "
                    "`python -m pipeline.inventory` first."
                )

        overall_df = _read_overall(overall_path)
        by_month_df = _read_by_month(by_month_path)
        kpis_df = _read_kpis(paths.INVENTORY_KPIS)

        decision = select_champion(overall_df, by_month_df, kpis_df, cfg, ctx)

        display = pd.DataFrame(
            [
                {
                    "model": candidate.model,
                    "wmape": format_pct(candidate.wmape) if candidate.wmape is not None else "NaN",
                    "bias": format_pct(candidate.bias) if candidate.bias is not None else "NaN",
                    "gate1_pass": candidate.gate1_pass,
                    "gate2_rank": candidate.gate2_rank,
                    "gate3_decision": candidate.gate3_decision,
                    "excluded_reason": candidate.excluded_reason,
                }
                for candidate in decision.candidates
            ]
        )
        print(display.to_string(index=False))
        print(
            f"\nchampion: {decision.champion} ({decision.champion_kind}) | "
            f"best_baseline: {decision.best_baseline} | "
            f"meaningful_improvement: {decision.meaningful_improvement} | "
            f"improvement_points: {decision.improvement_points}"
        )
        statement = _statement(decision)
        if statement:
            print(statement)
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
