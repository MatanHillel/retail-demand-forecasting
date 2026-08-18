"""The Flow's two crew kickoffs — LLM mode (US-33, PRD §34, §37, §38, §39, §47).

``python -m pipeline`` runs exactly the same ten deterministic steps as ``python -m pipeline
--no-llm``. LLM mode adds two steps *between* them, and nothing else:

* :func:`data_analyst_crew_review` — after step 3 (``contract_validation``) passes and before
  step 4. The Data Analyst Crew reviews the cleaning and the EDA and writes
  ``data_quality_review.md`` plus a polished ``insights.md``.
* :func:`data_scientist_crew_review` — after step 9 (``artifact_validation``) passes and before
  step 10 (``publish``). The Data Scientist Crew runs **narrative-only**: steps 4-8 already
  executed the identical deterministic tools, so the crew is given the T3-narrative task and the
  five reading/guarded-writing tools, and cannot reach a tool that computes a number.

Four rules make this safe enough that an LLM run and a ``--no-llm`` run are numerically identical
(§38, §40), and all four are mechanical rather than hoped for:

1. **A crew is never kicked off after a failure, and never in ``--no-llm`` mode.** Both step
   bodies return immediately unless ``ctx.mode == "llm"``; the Flow only reaches them through a
   router that already returned its *continue* label (§37).
2. **A crew may not change a number.** :func:`snapshot_guarded` takes a sha256 *and a byte copy*
   of every numeric artifact this run has staged, before the crew starts;
   :func:`restore_guarded` compares afterwards and puts the original back, recording
   ``crew modified numeric artifact <name> — restored``. The copy is essential: with staging on,
   ``ctx.out(paths.INSIGHTS)`` hands the crew the *same* staged path the deterministic writer
   used, so "restore from staging" would restore the overwritten file, and restoring from the
   final path would publish the **previous** run's numbers under this run's id (§8 of the issue).
3. **A crew's own mistakes are warnings, not failures.** The cost cap, a guard restore and an LLM
   or agent error are all caught *inside* the ``ctx.step(...)`` block and reported with
   ``ctx.warn``: ``ctx.step`` sets ``ctx.status = "failed"`` on any exception it sees, ``finish()``
   cannot undo that and ``promote()`` then refuses, so an escaping exception would cost the run
   its artifacts over a rejected paragraph. The one thing that does travel on is a failure a
   crew's *deterministic tool* raised inside its own step — that already flipped ``ctx.status``
   and is a genuine validation stop, so it is re-raised and routed to the §39 failure handler
   rather than left to surface later as a confusing refusal to promote.
4. **Completeness is re-checked after crew 2.** Step 9 ran *before* the narrative rewrite, so a
   crew that truncated ``evaluation_report.md`` would slip past it. The same staged existence and
   non-zero-size check is repeated afterwards, and a failure there is a graceful stop.

Cost (§47) is accumulated across both crews from the token counters
:func:`crews.common.record_token_usage` merges into ``ctx.metrics``, priced by
:func:`crews.environment.estimate_cost_usd` against ``model_config.yaml -> llm.pricing``. It is
recorded as ``ctx.record_metrics({"llm": …})`` — ``RunContext`` is ``extra="forbid"`` with a
published field list, so ``run_log.json`` carries this under ``metrics.llm``, never as a
top-level ``llm`` key (§8 of the issue).

**CrewAI is imported lazily**, inside :func:`run_analyst_crew` / :func:`run_scientist_crew`. Those
two functions are the seams the whole LLM stack sits behind: importing this module costs nothing,
and a test can replace either seam with a stub and prove the wiring without a network call.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from flow.state import FlowState
from flow.steps import REQUIRED_FLOW_ARTIFACTS, FlowData, validation_report_path
from pipeline import paths
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: Step names, matching the ``@listen`` methods of :class:`flow.main.RetailForecastFlow`.
CREW1_STEP = "data_analyst_crew_review"
CREW2_STEP = "data_scientist_crew_review"

#: ``step`` recorded on the violations of the post-narrative completeness re-check.
NARRATIVE_VALIDATION_STEP = "narrative_artifact_validation"

#: Per-crew status recorded in ``metrics.llm``. ``not_run`` is a ``--no-llm`` run (the step body
#: returned before doing anything); ``cost_capped`` is the §47 cap aborting the narrative;
#: ``failed`` is a crew that raised something recoverable and left the deterministic text in place.
STATUS_NOT_RUN = "not_run"
STATUS_COMPLETED = "completed"
STATUS_COST_CAPPED = "cost_capped"
STATUS_FAILED = "failed"

#: The numeric artifacts a crew may never change (§2 of the issue). Every one of them is written
#: by a deterministic ``pipeline`` function; a crew that rewrites one has broken §38, whatever its
#: intention was. Files absent from staging when the snapshot is taken are simply not watched —
#: crew 1 runs before the modelling artifacts exist.
GUARDED_ARTIFACTS: tuple[Path, ...] = (
    paths.CLEAN_DATA,
    paths.FEATURES,
    paths.MODEL,
    paths.MODEL_META,
    paths.DATASET_CONTRACT,
    paths.BACKTEST_PREDICTIONS,
    paths.LATEST_FORECAST,
    paths.INVENTORY_PLAN,
    paths.SIGMA_TABLE,
    paths.INVENTORY_KPIS,
    paths.HOLDOUT_SIMULATION_ROWS,
    paths.QUARTERLY_FORECAST,
    paths.CHAMPION_DECISION,
    paths.DATA_QUALITY_FINDINGS,
    paths.FEATURE_VALIDATION,
)

#: Directory holding the guard's byte copies. Deliberately a sibling of the run's staging tree
#: rather than a child: ``promote()`` must never see these files, ``handle_failure`` moves the
#: staging tree whole, and each crew step deletes its own copies when it is done.
GUARD_DIRNAME = "_guard"

#: The three narratives an LLM may rewrite, and therefore the three the guard reports on.
NARRATIVE_KEYS: tuple[str, ...] = ("insights", "evaluation_report", "model_card")

#: ``metrics.llm.tokens`` keys, paired with the suffix each is summed from. Longest suffix first:
#: ``crew_x_cached_prompt_tokens`` also ends with ``_prompt_tokens``, and must be counted once.
_TOKEN_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("cached_prompt", "_cached_prompt_tokens"),
    ("prompt", "_prompt_tokens"),
    ("completion", "_completion_tokens"),
    ("total", "_total_tokens"),
)


# --------------------------------------------------------------------------
# path helpers (mirrors flow.steps — a mid-run reader sees THIS run's files)
# --------------------------------------------------------------------------
def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical path constant (``docs/interfaces.md`` §6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _staged(ctx: RunContext, canonical: Path) -> Path:
    """Where this run wrote ``canonical`` — never the final location, which still holds the
    previous run's copy until step 10 promotes (``docs/interfaces.md`` §6 rule 7)."""
    return ctx.staging_dir / _repo_relative(canonical)


def guard_dir(ctx: RunContext) -> Path:
    """Scratch directory holding the pre-kickoff byte copies for this run."""
    return ctx.staging_dir.parent / GUARD_DIRNAME / ctx.run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _watched(ctx: RunContext) -> list[Path]:
    """Every guarded artifact this run has already staged, plus every evaluation table.

    The evaluation tables are globbed rather than listed: they are written as a set by
    ``evaluate``/``run_sigma``/``run_quarterly_aggregation`` and their number is not a constant.
    """
    watched = [path for path in GUARDED_ARTIFACTS if _staged(ctx, path).is_file()]
    tables_dir = _staged(ctx, paths.EVAL_TABLES_DIR)
    if tables_dir.is_dir():
        watched.extend(
            paths.EVAL_TABLES_DIR / found.name
            for found in sorted(tables_dir.iterdir())
            if found.is_file()
        )
    return watched


# --------------------------------------------------------------------------
# the determinism guard (§38): numbers in, same numbers out
# --------------------------------------------------------------------------
def snapshot_guarded(ctx: RunContext) -> dict[str, str]:
    """Copy and checksum every numeric artifact this run has staged. Returns ``{relative: sha}``.

    Both halves matter. The checksum detects a change; the copy is the only thing that can undo
    one, because the crew writes to the very path the deterministic step wrote to.
    """
    destination_root = guard_dir(ctx)
    snapshot: dict[str, str] = {}
    for canonical in _watched(ctx):
        relative = _repo_relative(canonical)
        source = _staged(ctx, canonical)
        copy = destination_root / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copy)
        snapshot[relative.as_posix()] = _sha256(source)
    return snapshot


def restore_guarded(ctx: RunContext, snapshot: dict[str, str]) -> list[str]:
    """Put back any numeric artifact the crew changed or deleted. Returns what was restored.

    The wording of the warning is fixed by §2 of the issue, so it can be grepped for in a log:
    ``crew modified numeric artifact <name> — restored``.
    """
    root = guard_dir(ctx)
    restored: list[str] = []
    for relative_posix, expected in sorted(snapshot.items()):
        relative = Path(relative_posix)
        current = ctx.staging_dir / relative
        if current.is_file() and _sha256(current) == expected:
            continue
        copy = root / relative
        if not copy.is_file():  # pragma: no cover - the snapshot always writes one
            ctx.warn(f"crew modified numeric artifact {relative_posix} — no copy to restore from")
            continue
        current.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(copy, current)
        ctx.warn(f"crew modified numeric artifact {relative_posix} — restored")
        restored.append(relative_posix)
    return restored


def clear_guard(ctx: RunContext) -> None:
    """Delete this run's guard copies — they must never outlive the step that took them."""
    shutil.rmtree(guard_dir(ctx), ignore_errors=True)


# --------------------------------------------------------------------------
# cost accounting (§47)
# --------------------------------------------------------------------------
def token_totals(ctx: RunContext) -> dict[str, int]:
    """Sum every crew's token counters recorded so far into one ``{prompt, completion, …}``."""
    totals = {key: 0 for key, _ in _TOKEN_SUFFIXES}
    for name, value in ctx.metrics.items():
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        for key, suffix in _TOKEN_SUFFIXES:
            if name.endswith(suffix):
                totals[key] += int(value)
                break
    return totals


def llm_summary(ctx: RunContext, state: FlowState) -> dict[str, Any]:
    """This run's ``metrics.llm`` block so far, defaulted on the first call."""
    if state.llm:
        return dict(state.llm)
    from crews.environment import llm_model_name  # lazy: no LLM import for a --no-llm run

    return {
        "crew1_status": STATUS_NOT_RUN,
        "crew2_status": STATUS_NOT_RUN,
        "model": llm_model_name() if ctx.mode == "llm" else None,
        "tokens": {key: 0 for key, _ in _TOKEN_SUFFIXES},
        "cost_usd": 0.0,
        "max_cost_usd": max_cost_usd(),
        "narrative_accepted": dict.fromkeys(NARRATIVE_KEYS, False),
        "guard_restored": [],
    }


def max_cost_usd() -> float:
    """The configured cap (``model_config.yaml -> llm.max_cost_usd``), overridable at the CLI."""
    from pipeline.config import load_model_config

    return float(load_model_config().llm.max_cost_usd)


def priced(ctx: RunContext, summary: dict[str, Any]) -> dict[str, Any]:
    """Refresh ``tokens`` and ``cost_usd`` on ``summary`` from what the crews have recorded."""
    from crews.environment import estimate_cost_usd  # lazy (module docstring)

    tokens = token_totals(ctx)
    summary["tokens"] = tokens
    summary["cost_usd"] = estimate_cost_usd(
        prompt_tokens=tokens["prompt"],
        cached_prompt_tokens=tokens["cached_prompt"],
        completion_tokens=tokens["completion"],
        model=summary.get("model"),
    )
    return summary


def record(ctx: RunContext, state: FlowState, summary: dict[str, Any]) -> dict[str, Any]:
    """Publish the summary to both places a reader looks: ``metrics.llm`` and ``FlowState.llm``.

    ``record_metrics`` is the published API (``RunContext`` is ``extra="forbid"`` and has no
    ``llm`` field), so ``run_log.json`` carries this under ``metrics.llm`` (§8 of the issue).
    """
    ctx.record_metrics({"llm": summary})
    state.llm = summary
    return summary


# --------------------------------------------------------------------------
# the two seams every LLM call goes through
# --------------------------------------------------------------------------
def run_analyst_crew(ctx: RunContext) -> dict[str, Any]:
    """Kick off the Data Analyst Crew (US-12) — the only place crew 1 is constructed."""
    from crews.data_analyst.crew import run_data_analyst_crew

    return run_data_analyst_crew(ctx)


def run_scientist_crew(ctx: RunContext) -> dict[str, Any]:
    """Kick off the Data Scientist Crew narrative task (US-26 T3-narrative, US-33).

    ``narrative_only=True``: the deterministic T1-T3 tools were already executed by the Flow's
    steps 4-8, so the crew is given the narrative task alone over a state hydrated from this
    run's staged artifacts.
    """
    from crews.data_scientist.crew import run_data_scientist_crew

    return run_data_scientist_crew(ctx, narrative_only=True)


# --------------------------------------------------------------------------
# shared kickoff body
# --------------------------------------------------------------------------
def _kickoff(
    state: FlowState,
    ctx: RunContext,
    *,
    step_name: str,
    status_key: str,
    label: str,
    runner: Any,
    accepted_from: dict[str, str],
) -> dict[str, Any]:
    """Run one crew inside one step, guarded, priced and incapable of failing the run.

    ``accepted_from`` maps a ``metrics.llm.narrative_accepted`` key to the key of the crew
    summary that reports it, so the two crews share this body verbatim.
    """
    summary = llm_summary(ctx, state)
    with ctx.step(step_name):
        cap = summary["max_cost_usd"]
        spent = priced(ctx, summary)["cost_usd"]
        if spent >= cap:
            # §47: the cap aborts the narrative, never the run. Raising here would set
            # ctx.status = "failed" and cost the run its artifacts (§8 of the issue).
            ctx.warn(
                f"{label} not started: estimated LLM cost ${spent:.4f} has reached the "
                f"${cap:.2f} cap (model_config.yaml -> llm.max_cost_usd); the deterministic "
                "narrative stands"
            )
            summary[status_key] = STATUS_COST_CAPPED
            return record(ctx, state, summary)

        snapshot = snapshot_guarded(ctx)
        ctx.logger.info(f"{label}: kicking off over {len(snapshot)} guarded artifact(s)")
        crew_summary: dict[str, Any] | None = None
        try:
            crew_summary = runner(ctx)
            summary[status_key] = STATUS_COMPLETED
        except Exception as error:
            if ctx.status == "failed":
                # A deterministic tool inside the crew stopped the run in its own ctx.step; that
                # is a real validation failure and must reach the §39 handler (module docstring).
                clear_guard(ctx)
                raise
            ctx.warn(f"{label} failed ({type(error).__name__}: {error}); deterministic "
                     "artifacts are unchanged and the run continues")
            summary[status_key] = STATUS_FAILED
        finally:
            restored = restore_guarded(ctx, snapshot)
            clear_guard(ctx)

        if restored:
            summary["guard_restored"] = sorted({*summary.get("guard_restored", []), *restored})
        if crew_summary is not None:
            for narrative_key, summary_key in accepted_from.items():
                summary["narrative_accepted"][narrative_key] = bool(
                    crew_summary.get(summary_key, False)
                )
        priced(ctx, summary)
        ctx.logger.info(
            f"{label}: {summary[status_key]}, {summary['tokens']['total']} token(s), "
            f"estimated ${summary['cost_usd']:.4f}"
        )
    return record(ctx, state, summary)


# --------------------------------------------------------------------------
# step: crew 1, after step 3 (contract_validation)
# --------------------------------------------------------------------------
def data_analyst_crew_review(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Kick off the Data Analyst Crew once the contract has been validated (§37).

    In ``--no-llm`` mode this returns before doing anything at all: no step is opened, no crew is
    built and no LLM class is imported. The crew's tools are idempotent — they may re-run cleaning
    and the panel — so the determinism guard proves ``clean_data.csv`` came out byte-identical
    rather than assuming it.
    """
    if ctx.mode != "llm":
        return state
    _kickoff(
        state,
        ctx,
        step_name=CREW1_STEP,
        status_key="crew1_status",
        label="Data Analyst Crew",
        runner=run_analyst_crew,
        accepted_from={"insights": "insights_narrative_accepted"},
    )
    return state


# --------------------------------------------------------------------------
# step: crew 2, after step 9 (artifact_validation), before step 10 (publish)
# --------------------------------------------------------------------------
def data_scientist_crew_review(state: FlowState, ctx: RunContext, data: FlowData) -> FlowState:
    """Kick off the Data Scientist Crew's narrative task, then re-check completeness (§37).

    Step 9 validated the artifacts *before* this rewrite, so the same staged existence and
    non-zero-size check runs again afterwards: a crew that truncated a report must stop the run,
    not be published by step 10 (§8 of the issue).
    """
    if ctx.mode != "llm":
        return state
    _kickoff(
        state,
        ctx,
        step_name=CREW2_STEP,
        status_key="crew2_status",
        label="Data Scientist Crew",
        runner=run_scientist_crew,
        accepted_from={
            "evaluation_report": "evaluation_report_accepted",
            "model_card": "model_card_accepted",
        },
    )
    verify_artifacts_after_narrative(state, ctx)
    return state


def verify_artifacts_after_narrative(state: FlowState, ctx: RunContext) -> ValidationResult:
    """Re-run step 9's check on the staged paths, after the narrative rewrite.

    Raises:
        FlowValidationError: if a required artifact is now missing or empty. The wording matches
            step 9's (``<name> was not generated``) so the §39 message set stays one set.
    """
    missing = [
        canonical.name
        for canonical in REQUIRED_FLOW_ARTIFACTS
        if not (
            _staged(ctx, canonical).is_file() and _staged(ctx, canonical).stat().st_size > 0
        )
    ]
    result = ValidationResult(
        step=NARRATIVE_VALIDATION_STEP,
        passed=not missing,
        violations=[
            Violation(
                step=NARRATIVE_VALIDATION_STEP,
                rule="required_artifact",
                message=f"{name} was not generated",
            )
            for name in missing
        ],
        checked_rows=len(REQUIRED_FLOW_ARTIFACTS),
    )
    write_validation_report(result, validation_report_path(ctx), run_id=ctx.run_id)
    state.validation.artifacts = result.passed
    if missing:
        raise FlowValidationError(result, f"{missing[0]} was not generated")
    return result
