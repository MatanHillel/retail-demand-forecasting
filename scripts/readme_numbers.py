"""Print a markdown snippet of the latest pipeline results, for README's Results snapshot section.

Reads only artifacts already on disk (`run_log.json`, `champion_decision.json`,
`holdout_metrics_overall.csv`, `inventory_kpis.csv`) through `pipeline.paths` — it opens no
`RunContext`, so it allocates no run id and writes nothing under `artifacts/` (docs/interfaces.md
"Interface corrections", item 2).

Usage: python scripts/readme_numbers.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from pipeline import paths

MAIN_BASELINE = "B2_ma3"  # PRD's main baseline (CLAUDE.md §6) — always shown when available


def _load_run_log() -> dict:
    if not paths.RUN_LOG.exists():
        return {}
    return json.loads(paths.RUN_LOG.read_text(encoding="utf-8"))


def _load_champion_decision() -> dict | None:
    if not paths.CHAMPION_DECISION.exists():
        return None
    return json.loads(paths.CHAMPION_DECISION.read_text(encoding="utf-8"))


def _metric_row(df: pd.DataFrame, model_id: str) -> dict | None:
    rows = df.loc[df["model"] == model_id]
    return None if rows.empty else rows.iloc[0].to_dict()


def _kpi_row(df: pd.DataFrame, model_id: str) -> dict | None:
    is_model = df["model"] == model_id
    is_overall = df["scope"] == "overall"
    is_with_safety_stock = df["policy"] == "forecast_plus_ss"
    rows = df.loc[is_model & is_overall & is_with_safety_stock]
    return None if rows.empty else rows.iloc[0].to_dict()


def _statement(decision: dict) -> str | None:
    """The §20 gate-4 wording, echoing the private `_statement` in `pipeline.champion` from the
    public fields of `champion_decision.json` alone (that helper takes a `ChampionDecision`
    object and is not part of the public interface — see docs/interfaces.md)."""
    if decision.get("champion_kind") != "ml" or decision.get("improvement_points") is None:
        return None
    if decision.get("meaningful_improvement"):
        return None
    threshold = decision["config_gates"]["meaningful_improvement_points"]
    return (
        f"Simple methods are competitive: {decision['champion']} improves on "
        f"{decision['best_baseline']} by {decision['improvement_points']:.2f} wMAPE points "
        f"(< {threshold:.1f})."
    )


def render() -> str:
    run_log = _load_run_log()
    if not run_log:
        return "_No `artifacts/run_log.json` found — run `python -m pipeline --no-llm` first._"

    run_id = run_log.get("run_id", "unknown")
    status = run_log.get("status")
    lines = [f"_As of run `{run_id}` (status: `{status}`)._", ""]

    if status != "success":
        # `status` is running | success | failed (docs/interfaces.md §3) — only a successful
        # run's numbers are safe to publish; a running/failed run's artifacts are the
        # *previous* run's.
        lines.append(
            "**Not final** — the latest run did not finish successfully; no results shown."
        )
        return "\n".join(lines)

    decision = _load_champion_decision()
    if decision is None:
        lines.append("_No `champion_decision.json` found — champion selection has not run yet._")
        return "\n".join(lines)

    champion = decision.get("champion")
    if champion is None:
        lines.append(
            "_The champion decision has no champion recorded "
            "(no candidate passed the bias gate — PRD §20 gate 1)._"
        )
        return "\n".join(lines)

    overall = pd.read_csv(paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv")
    kpis = pd.read_csv(paths.INVENTORY_KPIS)
    best_baseline = decision.get("best_baseline")

    lines.append(f"**Champion:** `{champion}` ({decision.get('champion_kind')})")
    lines.append("")
    lines.append("| Model | wMAPE | Bias |")
    lines.append("|---|---|---|")
    champion_row = _metric_row(overall, champion)
    if champion_row:
        lines.append(
            f"| {champion} (champion) | {champion_row['wmape']:.3f} | "
            f"{champion_row['bias']:+.3f} |"
        )
    baseline_row = _metric_row(overall, MAIN_BASELINE)
    if baseline_row and MAIN_BASELINE != champion:
        lines.append(
            f"| {MAIN_BASELINE} (baseline) | {baseline_row['wmape']:.3f} | "
            f"{baseline_row['bias']:+.3f} |"
        )

    champion_kpi = _kpi_row(kpis, champion)
    baseline_kpi = _kpi_row(kpis, best_baseline) if best_baseline else None
    if champion_kpi or baseline_kpi:
        lines.append("")
        lines.append("| Model | Fill rate | Excess units |")
        lines.append("|---|---|---|")
        if champion_kpi:
            lines.append(
                f"| {champion} (champion) | {champion_kpi['fill_rate']:.3f} | "
                f"{champion_kpi['excess_units']:.0f} |"
            )
        if baseline_kpi and best_baseline != champion:
            lines.append(
                f"| {best_baseline} (best gate-1 baseline) | {baseline_kpi['fill_rate']:.3f} | "
                f"{baseline_kpi['excess_units']:.0f} |"
            )

    statement = _statement(decision)
    if statement:
        lines.append("")
        lines.append(f"> {statement}")

    return "\n".join(lines)


def main() -> int:
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
