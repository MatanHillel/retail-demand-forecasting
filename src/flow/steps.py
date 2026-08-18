"""The ten deterministic Flow steps (US-31, PRD §37) — pure functions over existing tools.

Each function takes ``(state, ctx, data)`` and returns the updated :class:`~flow.state.FlowState`:

* ``state`` — the JSON-serialisable :class:`~flow.state.FlowState` the routers read.
* ``ctx`` — the :class:`~pipeline.run_context.RunContext` (staging on: every artifact write goes
  through ``ctx.out()`` and lands under ``artifacts/_staging/<run_id>/`` until step 10 promotes).
* ``data`` — :class:`FlowData`, the in-memory carrier of the DataFrames the steps hand to each
  other. Frames must travel in memory, never through the final ``paths.*`` locations: mid-run
  those still hold the *previous* run's files (``docs/interfaces.md`` §6 rule 7).

Nothing here computes a new number and nothing here touches CrewAI — every call goes to the same
:mod:`pipeline` function the standalone CLIs use, which is what makes an LLM run and a ``--no-llm``
run numerically identical (§37, §38). A validation failure raises
:class:`~pipeline.validation.FlowValidationError` (message without the ``FLOW STOPPED:`` prefix —
the exception adds it); :class:`flow.main.RetailForecastFlow` catches it and routes the run to the
failure handler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from flow.state import FlowState
from pipeline import paths
from pipeline.backtest import backtest, backtest_summary, write_backtest_summary
from pipeline.baselines import predict_baselines, write_baseline_predictions
from pipeline.champion import select_champion
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import (
    load_cleaning_config,
    load_data_sources,
    load_inventory_policy,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.contract import contract_failure_message, validate_contract, write_contract
from pipeline.download import download_raw, load_raw, validate_raw_schema, verify_hash
from pipeline.eda.run_eda import run_eda
from pipeline.evaluate import evaluate
from pipeline.feature_validation import (
    LEAKAGE_FAILURE_MESSAGE,
    validate_features,
    write_feature_validation,
)
from pipeline.feature_validation import (
    STEP as FEATURE_VALIDATION_STEP,
)
from pipeline.feature_validation import (
    leakage_check as run_leakage_check,
)
from pipeline.features import build_features, write_features
from pipeline.inventory import STEP_NAME as INVENTORY_STEP
from pipeline.inventory import run_inventory_simulation
from pipeline.latest_forecast import STEP_NAME as LATEST_FORECAST_STEP
from pipeline.latest_forecast import run_latest_forecast
from pipeline.models import train_models, tune
from pipeline.panel import build_panel, validate_panel
from pipeline.quarterly import run_quarterly_aggregation
from pipeline.reports import write_all_reports
from pipeline.run_context import RunContext
from pipeline.sigma import run_sigma
from pipeline.split import SplitSpec, abc_train, write_abc_train
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: ``step`` recorded on step 9's violations and validation report.
ARTIFACT_VALIDATION_STEP = "artifact_validation"

#: The eleven files step 9 requires: the eight §41 artifacts plus the three forecast files the
#: issue names. Read from :mod:`pipeline.paths`, never restated as string literals (issue §8).
REQUIRED_FLOW_ARTIFACTS: tuple[Path, ...] = (
    *paths.REQUIRED_ARTIFACTS,
    paths.BACKTEST_PREDICTIONS,
    paths.LATEST_FORECAST,
    paths.INVENTORY_PLAN,
)


@dataclass
class FlowData:
    """In-memory hand-off between steps — DataFrames never enter the JSON ``FlowState``.

    ``raw_path`` is the explicit raw file (``--raw`` / ``--sample``); ``None`` means the canonical
    download. ``skip_tuning`` skips the grid search and uses the parameters already in
    ``model_config.yaml``.
    """

    raw_path: Path | None = None
    skip_tuning: bool = False
    raw_df: pd.DataFrame | None = None
    clean_df: pd.DataFrame | None = None
    waterfall_df: pd.DataFrame | None = None
    panel_df: pd.DataFrame | None = None
    contract: dict[str, Any] | None = None
    features_df: pd.DataFrame | None = None
    split: SplitSpec | None = None
    abc_train_df: pd.DataFrame | None = None
    backtest_df: pd.DataFrame | None = None
    holdout_predictions_df: pd.DataFrame | None = None
    eval_frames: dict[str, Any] = field(default_factory=dict)
    sigma_table_df: pd.DataFrame | None = None
    kpis_df: pd.DataFrame | None = None
    latest: dict[str, Any] = field(default_factory=dict)


def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _resolve_read(ctx: RunContext, relative: Path) -> Path:
    """This run's staged copy of a file if there is one, else the final one — never ``ctx.out()``,
    which registers the path for promotion and would make ``promote()`` warn about a file this
    run only read (``docs/interfaces.md`` §6 rule 7)."""
    staged = ctx.staging_dir / relative
    return staged if staged.is_file() else ctx.base_dir / relative


def validation_report_path(ctx: RunContext) -> Path:
    """Where this run's ``validation_report.json`` goes — the canonical location rebased onto
    ``ctx.base_dir`` so a test run never touches the repo's own copy. Bypasses staging on purpose
    (``docs/interfaces.md`` §6 rule 2)."""
    return ctx.base_dir / _repo_relative(paths.VALIDATION_REPORT)


# --------------------------------------------------------------------------
# 1. Dataset intake (§37 step 1) — download / hash / load / raw schema
# --------------------------------------------------------------------------
def dataset_intake(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Acquire the raw extract, verify its hash and validate the raw schema."""
    with ctx.step("dataset_intake", inputs={"raw": str(data.raw_path or "canonical download")}):
        if data.raw_path is None:
            sources = load_data_sources()
            ctx.logger.info(f"dataset: {sources.citation}")
            raw_path = download_raw(ctx=ctx)
            if sources.expected_sha256 is None:
                ctx.warn(
                    "expected_sha256 is null; run `python -m pipeline.download --record-hash` "
                    "to record it"
                )
            else:
                hash_result = verify_hash(raw_path, expected=sources.expected_sha256)
                if not hash_result.passed:
                    write_validation_report(
                        hash_result, validation_report_path(ctx), run_id=ctx.run_id
                    )
                    state.validation.raw_schema = False
                    raise FlowValidationError(hash_result)
        else:
            raw_path = Path(data.raw_path)
            ctx.logger.info(
                f"raw file supplied explicitly ({raw_path}); recorded-hash check skipped"
            )

        frame, meta = load_raw(raw_path, ctx=ctx)
        data.raw_df = frame

        schema_result = validate_raw_schema(frame)
        write_validation_report(schema_result, validation_report_path(ctx), run_id=ctx.run_id)
        state.validation.raw_schema = schema_result.passed
        if not schema_result.passed:
            raise FlowValidationError(schema_result)
        ctx.logger.info(f"raw data OK: {meta['rows']} rows, {len(meta['columns'])} columns")

    state.data = ctx.data.model_dump(mode="json")
    return state


# --------------------------------------------------------------------------
# 2. Data Analyst work (§37 step 2) — clean → panel → EDA → contract
# --------------------------------------------------------------------------
def data_analyst_work(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """The deterministic tools of Crew 1 in fixed order (US-33 kicks the crew off around them)."""
    cleaning_cfg = load_cleaning_config()

    with ctx.step("clean_transactions"):
        clean_df, waterfall_df = clean_transactions(data.raw_df, cleaning_cfg, ctx)
        data.clean_df = clean_df
        data.waterfall_df = waterfall_df

    returns_path = _resolve_read(ctx, _repo_relative(paths.PROCESSED_DIR / RETURNS_FILENAME))
    with ctx.step("build_panel"):
        panel = build_panel(clean_df, pd.read_parquet(returns_path), cleaning_cfg, ctx)
        result = validate_panel(panel, cleaning_cfg)
        write_validation_report(result, validation_report_path(ctx), run_id=ctx.run_id)
        if not result.passed:
            raise FlowValidationError(result)
        data.panel_df = panel

    with ctx.step("run_eda"):
        run_eda(clean_df, panel, waterfall_df, cleaning_cfg, ctx, raw_df=data.raw_df)

    with ctx.step("write_contract"):
        data.contract = write_contract(
            panel, cleaning_cfg, load_model_config(), load_non_inventory_codes(), ctx
        )

    return state


# --------------------------------------------------------------------------
# 3. Contract validation (§37 step 3)
# --------------------------------------------------------------------------
def contract_validation(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Check ``clean_data`` against the contract this run just wrote."""
    with ctx.step("contract_validation"):
        result = validate_contract(data.panel_df, data.contract)
        write_validation_report(result, validation_report_path(ctx), run_id=ctx.run_id)
        state.validation.contract = result.passed
        if not result.passed:
            raise FlowValidationError(result, contract_failure_message(result))
    return state


# --------------------------------------------------------------------------
# 4. Data Scientist work (§37 step 4) — features.csv
# --------------------------------------------------------------------------
def data_scientist_work(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Build and write the §17 feature table."""
    model_cfg = load_model_config()
    cleaning_cfg = load_cleaning_config()

    with ctx.step("build_features"):
        k = model_cfg.active_rule.k
        first_target = model_cfg.split.first_target_month
        last_target = cleaning_cfg.raw.last_full_month
        targets = pd.period_range(first_target, last_target, freq="M")
        candidates = int(data.panel_df["stock_code"].nunique() * len(targets))
        frame = build_features(data.panel_df, k, first_target, last_target, model_cfg)
        ctx.log_rows(
            "features_active_filter",
            before=candidates,
            removed=candidates - len(frame),
            after=int(len(frame)),
        )
        write_features(frame, ctx)
        data.features_df = frame

    return state


# --------------------------------------------------------------------------
# 5. Feature validation (§37 step 5) — structure + leakage
# --------------------------------------------------------------------------
def feature_validation(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Validate the features structurally and prove the no-leakage boundary (§16)."""
    model_cfg = load_model_config()

    with ctx.step(FEATURE_VALIDATION_STEP):
        k = model_cfg.active_rule.k
        feature_result = validate_features(data.features_df, data.panel_df, model_cfg, k)
        leakage_result = run_leakage_check(data.features_df, data.panel_df, model_cfg, ctx.seed)
        write_feature_validation([feature_result, leakage_result], ctx)

        state.validation.features = feature_result.passed
        state.validation.leakage = leakage_result.passed

        combined = ValidationResult(
            step=FEATURE_VALIDATION_STEP,
            passed=feature_result.passed and leakage_result.passed,
            violations=feature_result.violations + leakage_result.violations,
            checked_rows=int(len(data.features_df)),
        )
        write_validation_report(combined, validation_report_path(ctx), run_id=ctx.run_id)
        if not combined.passed:
            message = (
                LEAKAGE_FAILURE_MESSAGE
                if not leakage_result.passed
                else f"feature validation failed ({len(combined.violations)} violations)"
            )
            raise FlowValidationError(combined, message)
    return state


# --------------------------------------------------------------------------
# 6. Temporal training & back-testing (§37 step 6)
# --------------------------------------------------------------------------
def training_and_backtest(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """ABC (training window), baselines, optional tuning, candidate training, rolling back-test."""
    model_cfg = load_model_config()
    policy_cfg = load_inventory_policy()
    split = SplitSpec.load()

    with ctx.step("abc_train"):
        abc_df = abc_train(data.panel_df, split, policy_cfg)
        write_abc_train(abc_df, ctx)
        data.abc_train_df = abc_df

    with ctx.step("predict_baselines"):
        baseline_df = predict_baselines(data.features_df, data.panel_df)
        write_baseline_predictions(baseline_df, ctx)

    if data.skip_tuning:
        ctx.logger.info("tuning skipped (--skip-tuning); using the parameters in model_config.yaml")
    else:
        with ctx.step("tune"):
            tune(data.features_df, model_cfg, ctx, split)
        # tune() rewrote model_config.yaml and cleared the config cache — reload both.
        model_cfg = load_model_config()
        split = SplitSpec.load()

    with ctx.step("train_models"):
        train_models(data.features_df, model_cfg, ctx, split)

    # backtest() opens its own ctx.step("backtest").
    backtest_df = backtest(data.features_df, data.panel_df, model_cfg, ctx, split)
    data.backtest_df = backtest_df
    data.split = split

    with ctx.step("backtest_summary"):
        summary = backtest_summary(backtest_df)
        write_backtest_summary(summary, ctx)

    holdout_relative = _repo_relative(paths.FORECASTS_DIR / "holdout_predictions.csv")
    data.holdout_predictions_df = pd.read_csv(
        _resolve_read(ctx, holdout_relative),
        dtype={
            "stock_code": "string",
            "forecast_origin": "string",
            "target_month": "string",
            "model": "string",
        },
    )
    return state


# --------------------------------------------------------------------------
# 7. Model evaluation & champion selection (§37 step 7)
# --------------------------------------------------------------------------
def evaluation_and_champion(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Hold-out metrics, robust σ, inventory simulation and the §20 champion gates — all by code."""
    model_cfg = load_model_config()

    # evaluate() and select_champion() open their own steps; run_sigma() does too.
    eval_frames = evaluate(
        data.holdout_predictions_df, data.backtest_df, data.abc_train_df, model_cfg, ctx
    )
    data.eval_frames = eval_frames

    sigma_df, _ = run_sigma(data.backtest_df, data.abc_train_df, model_cfg, ctx)
    data.sigma_table_df = sigma_df

    with ctx.step(INVENTORY_STEP):
        simulation = run_inventory_simulation(
            model_cfg,
            ctx,
            wide_df=eval_frames["holdout_rows_all_models"],
            sigma_df=sigma_df,
        )
    data.kpis_df = simulation["inventory_kpis"]

    select_champion(
        eval_frames["holdout_metrics_overall"],
        eval_frames["holdout_metrics_by_month"],
        data.kpis_df,
        model_cfg,
        ctx,
    )
    state.champion = dict(ctx.champion) if ctx.champion else None
    return state


# --------------------------------------------------------------------------
# 8. Inventory policy calibration (§37 step 8)
# --------------------------------------------------------------------------
def inventory_policy_calibration(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Refit the champion, build the operational forecast, σ, plan, quarterly view and reports."""
    model_cfg = load_model_config()
    cleaning_cfg = load_cleaning_config()

    with ctx.step(LATEST_FORECAST_STEP):
        data.latest = run_latest_forecast(
            model_cfg,
            ctx,
            panel_df=data.panel_df,
            train_features_df=data.features_df,
            backtest_df=data.backtest_df,
            abc_train_df=data.abc_train_df,
        )

    # run_quarterly_aggregation() opens its own step.
    run_quarterly_aggregation(
        model_cfg,
        ctx,
        backtest_df=data.backtest_df,
        latest_df=data.latest["latest_forecast"],
        panel_df=data.panel_df,
        cleaning_cfg=cleaning_cfg,
        champion=data.latest["champion"],
    )

    with ctx.step("reports"):
        write_all_reports(ctx)

    return state


# --------------------------------------------------------------------------
# 9. Artifact validation (§37 step 9) — STAGED paths, never the final ones
# --------------------------------------------------------------------------
def artifact_validation(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Every required artifact exists **in staging** with a non-zero size (issue §8).

    Checking the final locations instead would find the *previous* run's files and pass even when
    this run produced nothing — the false green §39 exists to prevent.
    """
    with ctx.step(ARTIFACT_VALIDATION_STEP):
        missing: list[str] = []
        artifact_paths: dict[str, str] = {}
        for canonical in REQUIRED_FLOW_ARTIFACTS:
            relative = _repo_relative(canonical)
            staged = ctx.staging_dir / relative
            artifact_paths[canonical.name] = relative.as_posix()
            if not (staged.is_file() and staged.stat().st_size > 0):
                missing.append(canonical.name)

        result = ValidationResult(
            step=ARTIFACT_VALIDATION_STEP,
            passed=not missing,
            violations=[
                Violation(
                    step=ARTIFACT_VALIDATION_STEP,
                    rule="required_artifact",
                    message=f"{name} was not generated",
                )
                for name in missing
            ],
            checked_rows=len(REQUIRED_FLOW_ARTIFACTS),
            extra={"artifacts": artifact_paths},
        )
        write_validation_report(result, validation_report_path(ctx), run_id=ctx.run_id)
        state.validation.artifacts = result.passed
        state.artifact_paths = artifact_paths
        if missing:
            raise FlowValidationError(result, f"{missing[0]} was not generated")
    return state


# --------------------------------------------------------------------------
# 10. Publish (§37 step 10) — promote staging, finalise run_log.json, summarise
# --------------------------------------------------------------------------
def publish(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Promote every staged artifact to its final location and finish the run."""
    with ctx.step("publish"):
        warnings_before = len(ctx.warnings)
        promoted = ctx.promote()
        stale = [
            message
            for message in ctx.warnings[warnings_before:]
            if "staged artifact was never written" in message
        ]
        if stale:
            # A registered path was never written: promote() only warns (docs/interfaces.md §6
            # rule 8), but publishing a partially promoted run would defeat §39 — fail instead.
            raise RuntimeError(
                f"promotion incomplete: {len(stale)} registered artifact(s) were never written: "
                + "; ".join(stale)
            )
        ctx.record_artifact_checksums()
        ctx.discard_staging()

    state.metrics = dict(ctx.metrics)
    state.errors = list(ctx.errors)
    ctx.finish("success")
    state.status = "success"

    champion = (state.champion or {}).get("champion", "—")
    lines = [
        f"run {ctx.run_id} finished: success (mode={ctx.mode})",
        f"champion: {champion}",
        f"artifacts promoted: {len(promoted)}",
        "steps:",
    ]
    for record in ctx.steps:
        duration = "-" if record.duration_s is None else f"{record.duration_s:.1f}s"
        lines.append(f"  {record.name}: {record.status} ({duration})")
    print("\n".join(lines))
    return state
