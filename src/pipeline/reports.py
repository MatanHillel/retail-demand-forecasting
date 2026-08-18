"""``evaluation_report.md`` & ``model_card.md`` — deterministic narrative generators
(US-25, PRD §20, §22, §23, §30, §32, §36, §38, §41, §47, §48, §49).

Both artifacts are assembled entirely from tables and JSON files earlier steps already computed —
:mod:`pipeline.evaluate` (US-19), :mod:`pipeline.sigma` (US-20), :mod:`pipeline.inventory` (US-21),
:mod:`pipeline.champion` (US-22), :mod:`pipeline.latest_forecast` (US-23),
:mod:`pipeline.quarterly` (US-24), :mod:`pipeline.quality` (US-07) and
:mod:`pipeline.contract` (US-08). This module never computes a metric, a gate or a σ value itself;
it only formats numbers that already exist on disk (CLAUDE.md §2 rule 3-4). The one exception is
reproducibility bookkeeping — a written artifact's byte size — which is a direct disk measurement,
not a modelling number, and is folded into the same ``numbers_in_tables`` check rather than
special-cased.

**No agent writes these files.** In ``--no-llm`` mode this is the only version that ever exists; in
LLM mode the Model Evaluation & Inventory Scientist (US-26) may rewrite the prose, but every number
in either version must still trace back to a table — the finished markdown is passed through
:func:`pipeline.narrative.numbers_in_tables` before it is accepted, and generation **fails** if a
single number cannot be traced (§38).

**Reads, never re-derives, the staged copy of every input.** Every table is resolved with this
run's staged copy first and the final (previous-run) location second, mirroring
:mod:`pipeline.eda.io`'s ``_resolve_read`` — the same reason: under the Flow (US-31), the upstream
steps' outputs are still sitting in ``artifacts/_staging/<run_id>/`` until ``promote()`` runs, and a
reader that only looked at the final path would silently embed the *previous* run's numbers.

**Percentages go through** :class:`_PctBacking`, **not** :func:`pipeline.metrics.format_pct`
**directly.** ``pipeline.narrative``'s number regex has no sign support — a written ``-23.4 %`` is
parsed as the positive magnitude ``23.4`` — and its tolerance can be tighter than a percentage's
own rounding error for small magnitudes (e.g. a 1-decimal display of ``0.021454`` rounds to
``2.1``, a 0.045-point error the guard's own tolerance does not cover). Both are properties of the
shared, foundational guard, not something a caller can format around case by case. ``_PctBacking``
records the exact value each display rounds to and folds it into the tables the guard checks
against, so the check verifies "is this an accurate rounding of a real cell" rather than tripping
over the guard's own blind spots — it is not a new number, just the sign and the precision the
guard's regex already discards.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline import paths
from pipeline.baselines import B2
from pipeline.champion import ChampionDecision, _statement
from pipeline.config import (
    InventoryPolicy,
    ModelConfig,
    load_data_sources,
    load_inventory_policy,
    load_model_config,
)
from pipeline.inventory import POLICIES, SCOPE_OVERALL
from pipeline.latest_forecast import INVENTORY_OUTPUT_NAME
from pipeline.metrics import format_pct
from pipeline.narrative import numbers_in_tables
from pipeline.run_context import RunContext

#: Where the Jinja2 templates live. Shipped as package data (see ``pyproject.toml``).
TEMPLATE_DIR = Path(__file__).parent / "templates"
EVALUATION_TEMPLATE_NAME = "evaluation_report.md.j2"
MODEL_CARD_TEMPLATE_NAME = "model_card.md.j2"

#: The five mandatory model-card sections plus Configuration/Version, in the required order
#: (course requirement, PRD §36, §49). ``## `` headings only ever come from this list — the guard
#: on this shape lives in :func:`write_model_card`'s own render check below.
MODEL_CARD_HEADINGS: tuple[str, ...] = (
    "## 1. Model purpose",
    "## 2. Training data summary",
    "## 3. Metrics",
    "## 4. Limitations",
    "## 5. Ethical considerations",
    "## Configuration",
    "## Version",
)


class _PctBacking:
    """Formats a fraction as a percentage (:func:`format_pct`) and records what it displayed.

    See the module docstring for why this exists instead of calling :func:`format_pct` directly.
    """

    def __init__(self, digits: int = 1) -> None:
        self._digits = digits
        self.values: list[float] = []

    def __call__(self, value: Any) -> str:
        text = format_pct(value, self._digits)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            self.values.append(abs(round(float(value) * 100, self._digits)) / 100.0)
        return text

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({"pct_backing": self.values})


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _resolve_read(ctx: RunContext, relative: Path) -> Path:
    """Locate an artifact for *reading*: this run's staged copy first, the final one second.

    Deliberately does not call ``ctx.out()`` — that would register the path for promotion and make
    ``promote()`` warn about a file this run only read (``pipeline.eda.io`` uses the identical
    pattern for the same reason).
    """
    candidates = [ctx.staging_dir / relative, ctx.base_dir / relative]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"artifact not found in staging or final location: {searched}")


def _read_csv(ctx: RunContext, canonical: Path, **kwargs: Any) -> pd.DataFrame:
    source = _resolve_read(ctx, _repo_relative(canonical))
    return pd.read_csv(source, **kwargs)


def _read_json(ctx: RunContext, canonical: Path) -> dict[str, Any]:
    source = _resolve_read(ctx, _repo_relative(canonical))
    return json.loads(source.read_text(encoding="utf-8"))


def _read_text(ctx: RunContext, canonical: Path) -> str:
    source = _resolve_read(ctx, _repo_relative(canonical))
    return source.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# loading every input (issue §3 "Inputs")
# --------------------------------------------------------------------------
def _load_inputs(ctx: RunContext) -> dict[str, Any]:
    """Read every table/JSON both reports draw on, staged copy first (module docstring)."""
    overall = _read_csv(
        ctx, paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv", dtype={"model": "string"}
    )
    by_month = _read_csv(
        ctx,
        paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv",
        dtype={"model": "string", "target_month": "string"},
    )
    by_abc = _read_csv(
        ctx,
        paths.EVAL_TABLES_DIR / "holdout_metrics_by_abc.csv",
        dtype={"model": "string", "abc_class": "string"},
    )
    improvement = _read_csv(
        ctx, paths.EVAL_TABLES_DIR / "improvement_vs_b2.csv", dtype={"model": "string"}
    )
    consistency = _read_csv(
        ctx,
        paths.EVAL_TABLES_DIR / "backtest_consistency.csv",
        dtype={"model": "string", "forecast_origin": "string", "target_month": "string"},
    )
    excess = _read_csv(
        ctx, paths.EXCESS_CONCENTRATION, dtype={"model": "string", "policy": "string"}
    )
    sigma_summary = _read_csv(
        ctx,
        paths.EVAL_TABLES_DIR / "sigma_summary.csv",
        dtype={"model": "string", "target_month": "string"},
    )
    kpis = _read_csv(
        ctx,
        paths.INVENTORY_KPIS,
        dtype={"model": "string", "policy": "string", "scope": "string", "group": "string"},
    )
    quarterly_metrics = _read_csv(
        ctx,
        paths.QUARTERLY_METRICS,
        dtype={"model": "string", "scope": "string", "quarter": "string"},
    )
    quarterly_limitation = _read_text(ctx, paths.QUARTERLY_LIMITATION)

    decision_raw = _read_json(ctx, paths.CHAMPION_DECISION)
    decision = ChampionDecision.model_validate(decision_raw)

    return {
        "model_cfg": load_model_config(),
        "policy_cfg": load_inventory_policy(),
        "data_sources": load_data_sources(),
        "overall": overall,
        "by_month": by_month,
        "by_abc": by_abc,
        "improvement": improvement,
        "consistency": consistency,
        "excess": excess,
        "sigma_summary": sigma_summary,
        "kpis": kpis,
        "quarterly_metrics": quarterly_metrics,
        "quarterly_limitation": quarterly_limitation,
        "decision_raw": decision_raw,
        "decision": decision,
        "model_meta": _read_json(ctx, paths.MODEL_META),
        "candidates_meta": _read_json(ctx, paths.CANDIDATES_META),
        "contract": _read_json(ctx, paths.DATASET_CONTRACT),
        "findings": _read_json(ctx, paths.DATA_QUALITY_FINDINGS),
    }


def _config_numbers(model_cfg: ModelConfig, policy_cfg: InventoryPolicy) -> dict[str, Any]:
    """Curated config values a report may quote — never the whole snapshot (interface note §8)."""
    return {
        "seed": model_cfg.seed,
        "active_rule": model_cfg.active_rule.model_dump(mode="json"),
        "split": model_cfg.split.model_dump(mode="json"),
        "champion_gates": model_cfg.champion_gates.model_dump(mode="json"),
        "lead_time_months": policy_cfg.lead_time_months,
        "z": policy_cfg.z,
        "z_options": policy_cfg.z_options,
        "sigma": policy_cfg.sigma.model_dump(mode="json"),
    }


def _numeric_tables(
    inputs: dict[str, Any], artifact_sizes: pd.DataFrame, pct: _PctBacking
) -> dict[str, Any]:
    """Every table/JSON a rendered report may cite a number from (``numbers_in_tables`` input)."""
    return {
        "holdout_metrics_overall": inputs["overall"],
        "holdout_metrics_by_month": inputs["by_month"],
        "holdout_metrics_by_abc": inputs["by_abc"],
        "improvement_vs_b2": inputs["improvement"],
        "backtest_consistency": inputs["consistency"],
        "excess_concentration": inputs["excess"],
        "sigma_summary": inputs["sigma_summary"],
        "inventory_kpis": inputs["kpis"],
        "quarterly_metrics": inputs["quarterly_metrics"],
        "champion_decision": inputs["decision_raw"],
        "model_meta": inputs["model_meta"],
        "candidates_meta": inputs["candidates_meta"],
        "dataset_contract": inputs["contract"],
        "data_quality_findings": inputs["findings"],
        "config_snapshot": _config_numbers(inputs["model_cfg"], inputs["policy_cfg"]),
        "artifact_sizes": artifact_sizes,
        "pct_backing": pct.table(),
    }


# --------------------------------------------------------------------------
# shared context builders
# --------------------------------------------------------------------------
def _config_summary(
    model_cfg: ModelConfig, policy_cfg: InventoryPolicy, pct: _PctBacking
) -> dict[str, Any]:
    split = model_cfg.split
    gates = model_cfg.champion_gates
    return {
        "seed": model_cfg.seed,
        "k": model_cfg.active_rule.k,
        "train_start": split.train_targets.start,
        "train_end": split.train_targets.end,
        "validation_start": split.validation_targets.start,
        "validation_end": split.validation_targets.end,
        "holdout_start": split.holdout_targets.start,
        "holdout_end": split.holdout_targets.end,
        "never_score": ", ".join(split.never_score) if split.never_score else "none",
        "z": f"{policy_cfg.z:g}",
        "z_options": ", ".join(f"{option:g}" for option in policy_cfg.z_options),
        "lead_time_months": policy_cfg.lead_time_months,
        "mad_scale": f"{policy_cfg.sigma.mad_scale:g}",
        "min_residuals_product": policy_cfg.sigma.min_residuals_product,
        "fallback_levels": " -> ".join(policy_cfg.sigma.fallback_levels),
        "max_abs_bias": pct(gates.max_abs_bias),
        "monthly_abs_bias_report_threshold": pct(gates.monthly_abs_bias_report_threshold),
        "tie_wmape_points": f"{gates.tie_wmape_points:.2f}",
        "meaningful_improvement_points": f"{gates.meaningful_improvement_points:.2f}",
        "similar_fill_rate_tolerance": pct(gates.similar_fill_rate_tolerance),
    }


def _candidate_rows(
    overall: pd.DataFrame, improvement: pd.DataFrame, pct: _PctBacking
) -> list[dict[str, Any]]:
    """The seven-candidate comparison table (issue §2 point 2): overall metrics + improvement."""
    merged = overall.merge(
        improvement[["model", "wmape_points_vs_b2", "relative_pct", "meaningful"]],
        on="model",
        how="left",
    )
    rows = []
    for record in merged.to_dict(orient="records"):
        rows.append(
            {
                "model": record["model"],
                "wmape": pct(record["wmape"]),
                "bias": pct(record["bias"]),
                "mae": f"{record['mae']:.1f}",
                "rmse": f"{record['rmse']:.1f}",
                "n_rows": f"{int(record['n_rows']):,}",
                "coverage_share": pct(record["coverage_share"]),
                "negative_share_raw": pct(record["negative_share_raw"]),
                "note": "" if pd.isna(record.get("note")) else record["note"],
                "wmape_points_vs_b2": pct(record.get("wmape_points_vs_b2")),
                "relative_pct": pct(record.get("relative_pct")),
                "meaningful": (
                    bool(record["meaningful"]) if pd.notna(record.get("meaningful")) else None
                ),
            }
        )
    return rows


def _month_rows(by_month: pd.DataFrame, pct: _PctBacking) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "target_month": row["target_month"],
            "wmape": pct(row["wmape"]),
            "bias": pct(row["bias"]),
            "mae": f"{row['mae']:.1f}",
            "rmse": f"{row['rmse']:.1f}",
            "n_rows": f"{int(row['n_rows']):,}",
        }
        for row in by_month.to_dict(orient="records")
    ]


def _abc_rows(by_abc: pd.DataFrame, pct: _PctBacking) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "abc_class": row["abc_class"],
            "wmape": pct(row["wmape"]),
            "bias": pct(row["bias"]),
            "mae": f"{row['mae']:.1f}",
            "rmse": f"{row['rmse']:.1f}",
            "n_rows": f"{int(row['n_rows']):,}",
        }
        for row in by_abc.to_dict(orient="records")
    ]


def _consistency_rows(consistency: pd.DataFrame, pct: _PctBacking) -> list[dict[str, Any]]:
    """One row per model: the per-origin summary columns, deduplicated (module docstring §22)."""
    summary_cols = [
        "model",
        "wmape_mean",
        "wmape_std",
        "wmape_min",
        "wmape_max",
        "bias_mean",
        "months_abs_bias_gt_threshold",
    ]
    summary = consistency[summary_cols].drop_duplicates(subset="model", keep="first")
    return [
        {
            "model": row["model"],
            "wmape_mean": pct(row["wmape_mean"]),
            "wmape_std": pct(row["wmape_std"]),
            "wmape_min": pct(row["wmape_min"]),
            "wmape_max": pct(row["wmape_max"]),
            "bias_mean": pct(row["bias_mean"]),
            "months_over_threshold": int(row["months_abs_bias_gt_threshold"]),
        }
        for row in summary.to_dict(orient="records")
    ]


def _champion_context(decision: ChampionDecision, pct: _PctBacking) -> dict[str, Any]:
    gates = decision.config_gates
    candidates = [
        {
            "model": c.model,
            "wmape": pct(c.wmape) if c.wmape is not None else "NaN",
            "bias": pct(c.bias) if c.bias is not None else "NaN",
            "gate1_pass": c.gate1_pass,
            "months_over_bias_threshold": (
                ", ".join(c.months_over_bias_threshold) if c.months_over_bias_threshold else "none"
            ),
            "gate2_rank": c.gate2_rank if c.gate2_rank is not None else "—",
            "gate3_decision": c.gate3_decision or "—",
            "excluded_reason": c.excluded_reason or "—",
        }
        for c in decision.candidates
    ]
    return {
        "champion": decision.champion,
        "champion_kind": decision.champion_kind,
        "best_baseline": decision.best_baseline,
        "meaningful_improvement": decision.meaningful_improvement,
        "improvement_points": (
            None if decision.improvement_points is None else f"{decision.improvement_points:.2f}"
        ),
        "gate1_all_failed": decision.gate1_all_failed,
        "candidates": candidates,
        "statement": _statement(decision),
        "max_abs_bias": pct(gates["max_abs_bias"]),
        "meaningful_improvement_points": f"{gates['meaningful_improvement_points']:.2f}",
    }


def _kpi_rows(
    kpis: pd.DataFrame, models: list[str], default_z: float, pct: _PctBacking
) -> list[dict[str, Any]]:
    focus = kpis.loc[
        kpis["model"].isin(models)
        & (kpis["scope"] == SCOPE_OVERALL)
        & np.isclose(kpis["z"].to_numpy(dtype=float), default_z)
    ].copy()
    focus["model"] = pd.Categorical(focus["model"], categories=models, ordered=True)
    focus["policy"] = pd.Categorical(focus["policy"], categories=list(POLICIES), ordered=True)
    focus = focus.sort_values(["model", "policy"], kind="mergesort")
    return [
        {
            "model": row["model"],
            "policy": row["policy"],
            "fill_rate": pct(row["fill_rate"]),
            "stockout_units": f"{row['stockout_units']:,.0f}",
            "excess_units": f"{row['excess_units']:,.0f}",
            "stockout_skumonth_rate": pct(row["stockout_skumonth_rate"]),
            "excess_per_unit_shortage": (
                "NaN"
                if pd.isna(row["excess_per_unit_shortage"])
                else f"{row['excess_per_unit_shortage']:.2f}"
            ),
        }
        for row in focus.to_dict(orient="records")
    ]


def _z_sensitivity_rows(
    kpis: pd.DataFrame, champion_id: str, z_options: list[float], pct: _PctBacking
) -> list[dict[str, Any]]:
    focus = kpis.loc[
        (kpis["model"] == champion_id)
        & (kpis["policy"] == "forecast_plus_ss")
        & (kpis["scope"] == SCOPE_OVERALL)
    ].copy()
    focus["z"] = focus["z"].astype(float)
    rows = []
    for z in z_options:
        match = focus.loc[np.isclose(focus["z"], z)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "z": f"{z:g}",
                "fill_rate": pct(row["fill_rate"]),
                "excess_units": f"{row['excess_units']:,.0f}",
                "stockout_units": f"{row['stockout_units']:,.0f}",
            }
        )
    return rows


def _excess_sentence(
    excess: pd.DataFrame, champion_id: str, pct: _PctBacking
) -> dict[str, Any] | None:
    match = excess.loc[(excess["model"] == champion_id) & (excess["policy"] == "forecast_plus_ss")]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "model": champion_id,
        "top_1pct_share": pct(row["top_1pct_share"]),
        "top_5pct_share": pct(row["top_5pct_share"]),
        "total_excess": f"{row['total_excess']:,.0f}",
    }


def _sigma_shares(
    sigma_summary: pd.DataFrame, champion_id: str, pct: _PctBacking
) -> dict[str, Any] | None:
    rows = sigma_summary.loc[sigma_summary["model"] == champion_id].sort_values("target_month")
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    return {
        "target_month": latest["target_month"],
        "n_products": int(latest["n_products"]),
        "share_product": pct(latest["share_product"]),
        "share_abc_group": pct(latest["share_abc_group"]),
        "share_global": pct(latest["share_global"]),
        "share_zero_mad": pct(latest["share_zero_mad"]),
    }


def _quarterly_rows(quarterly_metrics: pd.DataFrame, pct: _PctBacking) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "scope": row["scope"],
            "quarter": row["quarter"],
            "wmape": pct(row["wmape"]),
            "bias": pct(row["bias"]),
            "n_rows": f"{int(row['n_rows']):,}",
        }
        for row in quarterly_metrics.to_dict(orient="records")
    ]


def _demote_heading(text: str) -> str:
    """Drop the leading ``# `` heading line so the text nests under this report's own heading."""
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _artifact_size_table(ctx: RunContext) -> pd.DataFrame:
    """Byte size of every artifact this run has registered so far — a disk measurement, not a
    modelling number, but still routed through the guard (module docstring)."""
    rows = []
    for key, relative in sorted(ctx.artifacts.items()):
        try:
            source = _resolve_read(ctx, Path(relative))
        except FileNotFoundError:
            continue
        rows.append({"key": key, "path": relative, "size_bytes": source.stat().st_size})
    return pd.DataFrame(rows, columns=["key", "path", "size_bytes"])


def _artifact_rows(sizes: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"key": row["key"], "path": row["path"], "size_bytes": f"{int(row['size_bytes']):,}"}
        for row in sizes.to_dict(orient="records")
    ]


def _version_rows(ctx: RunContext) -> list[dict[str, str]]:
    return [
        {"package": name, "version": version or "not installed"}
        for name, version in ctx.versions.items()
    ]


def _data_hash(ctx: RunContext, contract: dict[str, Any]) -> str:
    """``ctx.data.sha256`` first, else the contract's, else "not recorded" (interface §8)."""
    if ctx.data.sha256:
        return ctx.data.sha256
    contract_hash = contract.get("data_sha256")
    if contract_hash:
        return str(contract_hash)
    return "not recorded in this run"


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


# --------------------------------------------------------------------------
# write_evaluation_report — issue §2
# --------------------------------------------------------------------------
def write_evaluation_report(ctx: RunContext, out: Path | None = None) -> Path:
    """Render, verify and write ``evaluation_report.md``; return the path written."""
    inputs = _load_inputs(ctx)
    model_cfg: ModelConfig = inputs["model_cfg"]
    policy_cfg: InventoryPolicy = inputs["policy_cfg"]
    decision: ChampionDecision = inputs["decision"]
    champion_id = decision.champion
    pct = _PctBacking()

    sizes = _artifact_size_table(ctx)
    champion_context = _champion_context(decision, pct)
    excess_sentence = _excess_sentence(inputs["excess"], champion_id, pct)
    sigma_shares = _sigma_shares(inputs["sigma_summary"], champion_id, pct)

    context = {
        "run_id": ctx.run_id,
        "generated_at": _now(),
        "data_hash": _data_hash(ctx, inputs["contract"]),
        "config": _config_summary(model_cfg, policy_cfg, pct),
        "candidates": _candidate_rows(inputs["overall"], inputs["improvement"], pct),
        "by_month": _month_rows(inputs["by_month"], pct),
        "by_abc": _abc_rows(inputs["by_abc"], pct),
        "train_end": model_cfg.split.train_targets.end,
        "champion": champion_context,
        "consistency": _consistency_rows(inputs["consistency"], pct),
        "kpi_rows": _kpi_rows(inputs["kpis"], [champion_id, B2], policy_cfg.z, pct),
        "z_sensitivity": _z_sensitivity_rows(
            inputs["kpis"], champion_id, policy_cfg.z_options, pct
        ),
        "excess_sentence": excess_sentence,
        "sigma_shares": sigma_shares,
        "quarterly_rows": _quarterly_rows(inputs["quarterly_metrics"], pct),
        "quarterly_limitation": _demote_heading(inputs["quarterly_limitation"]),
        "seed": model_cfg.seed,
        "versions": _version_rows(ctx),
        "artifacts": _artifact_rows(sizes),
        "inventory_output_name": INVENTORY_OUTPUT_NAME,
    }

    environment = _environment()
    markdown = environment.get_template(EVALUATION_TEMPLATE_NAME).render(**context)

    tables = _numeric_tables(inputs, sizes, pct)
    check = numbers_in_tables(markdown, tables)
    if not check.passed:
        raise ValueError(
            f"evaluation_report.md would state a number that is in no computed table — "
            f"{check.summary()}. Every narrative number must be traceable to a table (PRD §38)."
        )

    canonical = paths.EVALUATION_REPORT if out is None else Path(out)
    relative = canonical.relative_to(paths.PROJECT_ROOT)
    destination = ctx.out(relative)
    destination.write_text(markdown, encoding="utf-8", newline="\n")
    ctx.record_artifact("evaluation_report", relative)
    ctx.logger.info(
        f"evaluation_report.md written: {relative.as_posix()} "
        f"({check.checked} numbers all backed by tables)"
    )
    return destination


# --------------------------------------------------------------------------
# write_model_card — issue §2
# --------------------------------------------------------------------------
def write_model_card(ctx: RunContext, out: Path | None = None) -> Path:
    """Render, verify and write ``model_card.md``; return the path written."""
    inputs = _load_inputs(ctx)
    model_cfg: ModelConfig = inputs["model_cfg"]
    policy_cfg: InventoryPolicy = inputs["policy_cfg"]
    decision: ChampionDecision = inputs["decision"]
    contract = inputs["contract"]
    model_meta = inputs["model_meta"]
    candidates_meta = inputs["candidates_meta"]
    findings = inputs["findings"]
    waterfall = findings.get("waterfall") or []
    pct = _PctBacking()

    sizes = _artifact_size_table(ctx)

    context = {
        "run_id": ctx.run_id,
        "generated_at": _now(),
        "data_hash": _data_hash(ctx, contract),
        "config": _config_summary(model_cfg, policy_cfg, pct),
        "active_k": model_cfg.active_rule.k,
        "inventory_output_name": INVENTORY_OUTPUT_NAME,
        "contract_rows": f"{int(contract['row_counts']['rows']):,}",
        "contract_products": f"{int(contract['row_counts']['products']):,}",
        "first_month": contract["date_range"]["first_month"],
        "last_full_month": contract["date_range"]["last_full_month"],
        "partial_months": ", ".join(contract["date_range"]["partial_months"]),
        "train_start": model_cfg.split.train_targets.start,
        "train_end": model_cfg.split.train_targets.end,
        "holdout_start": model_cfg.split.holdout_targets.start,
        "holdout_end": model_cfg.split.holdout_targets.end,
        "n_train_rows": f"{int(candidates_meta['models'][0]['n_train_rows']):,}",
        "refit_train_end": model_meta["train_targets"]["end"],
        "operational_target_month": model_meta["target_month"],
        "raw_rows": f"{int(waterfall[0]['rows_before']):,}" if waterfall else "n/a",
        "clean_rows": f"{int(waterfall[-1]['rows_after']):,}" if waterfall else "n/a",
        "dataset_source": contract["source"],
        "citation": inputs["data_sources"].citation,
        "candidates": _candidate_rows(inputs["overall"], inputs["improvement"], pct),
        "by_month": _month_rows(inputs["by_month"], pct),
        "by_abc": _abc_rows(inputs["by_abc"], pct),
        "champion": _champion_context(decision, pct),
        "disclaimer": policy_cfg.disclaimer,
        "quarterly_limitation": _demote_heading(inputs["quarterly_limitation"]),
        "seed": model_cfg.seed,
        "versions": _version_rows(ctx),
        "artifacts": _artifact_rows(sizes),
    }

    environment = _environment()
    markdown = environment.get_template(MODEL_CARD_TEMPLATE_NAME).render(**context)

    headings = [line for line in markdown.splitlines() if line.startswith("## ")]
    if headings != list(MODEL_CARD_HEADINGS):
        raise ValueError(
            f"model_card.md must contain exactly the headings {list(MODEL_CARD_HEADINGS)} in "
            f"that order (PRD §36, §49); got {headings}"
        )

    tables = _numeric_tables(inputs, sizes, pct)
    check = numbers_in_tables(markdown, tables)
    if not check.passed:
        raise ValueError(
            f"model_card.md would state a number that is in no computed table — "
            f"{check.summary()}. Every narrative number must be traceable to a table (PRD §38)."
        )

    canonical = paths.MODEL_CARD if out is None else Path(out)
    relative = canonical.relative_to(paths.PROJECT_ROOT)
    destination = ctx.out(relative)
    destination.write_text(markdown, encoding="utf-8", newline="\n")
    ctx.record_artifact("model_card", relative)
    ctx.logger.info(
        f"model_card.md written: {relative.as_posix()} "
        f"({check.checked} numbers all backed by tables)"
    )
    return destination


def write_all_reports(ctx: RunContext) -> dict[str, Path]:
    """Write both required narrative artifacts; return their paths (issue §2)."""
    return {
        "evaluation_report": write_evaluation_report(ctx),
        "model_card": write_model_card(ctx),
    }


# --------------------------------------------------------------------------
# CLI: python -m pipeline.reports
# --------------------------------------------------------------------------
def run() -> int:
    """Standalone entry point: ``python -m pipeline.reports`` (AC 1)."""
    ctx = RunContext.start(mode="no-llm")
    try:
        with ctx.step("reports"):
            write_all_reports(ctx)
        print("narrative check: passed (evaluation_report.md)")
        print("narrative check: passed (model_card.md)")
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
