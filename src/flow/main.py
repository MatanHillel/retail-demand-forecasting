"""``RetailForecastFlow`` — the CrewAI Flow orchestrating the ten §37 steps (US-31, US-32).

The Flow is the conductor, not the orchestra: every step body lives in :mod:`flow.steps` and calls
the same deterministic :mod:`pipeline` tools the standalone CLIs use. A ``@router`` after steps 1,
3, 5 and 9 (the validation checkpoints) routes to *continue* or ``"fail"``; the ``@listen("fail")``
handler delegates to :func:`flow.failure.handle_failure`, which writes ``validation_report.json``
and a ``status: failed`` ``run_log.json``, never calls ``promote()``, and archives (or, with
``--no-keep-failed``, discards) this run's staging tree — so the previous run's published
artifacts are left untouched (§39).

**crewai 0.86.0 adaptation** (issue §3 says to document divergences — details in ``docs/flow.md``):

* A router is registered per trigger method and its *returned string* becomes the trigger label
  that ``@listen("<label>")`` methods match on — the label replaces the method-name trigger, so
  each router's continue label must be distinct (``"intake_ok"``, ``"contract_ok"``, …) while all
  four share the single ``"fail"`` label.
* ``Flow._execute_single_listener`` **swallows** any exception raised inside a ``@listen`` method
  (it prints a traceback and the chain simply stops; ``kickoff()`` returns normally). The Flow can
  therefore never rely on an exception escaping ``kickoff()``: :meth:`RetailForecastFlow._run`
  catches everything itself, records the failure on the state, and the failure travels *through
  the state* to the next router, which returns ``"fail"``. Steps between two routers short-circuit
  via the same state guard.

This module is the only place under ``src/flow/`` that imports CrewAI; :mod:`flow.steps` and
:mod:`flow.state` stay import-clean so ``--no-llm`` mode never touches an LLM class (§37 No-LLM
mode — ``make_llm`` exists only in :mod:`crews.common` and is never called here).
"""

from __future__ import annotations

import os
from pathlib import Path

# Disable CrewAI's OpenTelemetry client before the crewai import: the Flow must run offline and in
# CI, and a telemetry exporter has no place in a deterministic pipeline (§40).
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai.flow.flow import Flow, listen, router, start  # noqa: E402

from flow import failure as flow_failure  # noqa: E402
from flow import steps as flow_steps  # noqa: E402
from flow.state import FlowState  # noqa: E402
from flow.steps import FlowData  # noqa: E402
from pipeline.run_context import RunContext, RunMode  # noqa: E402
from pipeline.validation import FlowValidationError  # noqa: E402

#: The shared failure label every router returns on a failed state (§37).
FAIL = "fail"

#: The four continue labels, one per router — distinct because a router's label *replaces* its
#: method-name trigger in crewai 0.86.0 (module docstring).
INTAKE_OK = "intake_ok"
CONTRACT_OK = "contract_ok"
FEATURES_OK = "features_ok"
ARTIFACTS_OK = "artifacts_ok"


class RetailForecastFlow(Flow[FlowState]):
    """Raw data → 1 intake → 2 analyst work → 3 contract validation → 4 scientist work
    → 5 feature validation → 6 training & back-test → 7 evaluation & champion
    → 8 inventory policy calibration → 9 artifact validation → 10 publish (§37)."""

    def __init__(
        self,
        ctx: RunContext,
        *,
        raw_path: Path | str | None = None,
        skip_tuning: bool = False,
        keep_failed: bool = True,
    ) -> None:
        self._ctx = ctx
        self._data = FlowData(
            raw_path=None if raw_path is None else Path(raw_path),
            skip_tuning=skip_tuning,
        )
        #: ``--keep-failed`` / ``--no-keep-failed`` (US-32): archive vs. discard a failed run's
        #: staging tree. Read by :meth:`_finalize_failure` via :func:`flow.failure.handle_failure`.
        self._keep_failed = keep_failed
        #: The exception behind a graceful stop or an unexpected failure — ``None`` until one hits.
        self._failure_error: Exception | None = None
        super().__init__()
        self.state.run_id = ctx.run_id
        self.state.started_at = ctx.started_at
        self.state.mode = ctx.mode

    # -- shared step execution ---------------------------------------------
    def _run(self, name: str, step_fn) -> None:
        """Run one step body, converting any exception into state the routers can act on.

        crewai 0.86.0 swallows exceptions raised in listeners (module docstring), so nothing may
        escape: a :class:`FlowValidationError` is a graceful stop, anything else an unexpected
        failure. Either way the state flips to ``failed`` and every later step skips itself until
        a router funnels the run into :meth:`handle_failure`.
        """
        if self.state.status == "failed":
            return
        self.state.current_step = name
        ctx = self._ctx
        try:
            step_fn(self.state, ctx, self._data)
        except FlowValidationError as error:
            self._failure_error = error
            self._record_failure(name, error)
        except Exception as error:
            self._failure_error = error
            ctx.logger.exception(f"unexpected error in step {name}")
            self._record_failure(name, error)

    def _record_failure(self, name: str, error: Exception) -> None:
        """Mark the run failed on both the context and the state.

        ``ctx.step(...)`` already recorded the error and set ``ctx.status = "failed"`` when the
        exception happened inside a step block; code between steps (config loads, staged reads)
        is covered here instead.
        """
        ctx = self._ctx
        if ctx.status != "failed":
            ctx.status = "failed"
            ctx.errors.append(
                {"step": name, "type": type(error).__name__, "message": str(error)}
            )
        self.state.errors = list(ctx.errors)
        self.state.status = "failed"

    def _finalize_failure(self) -> None:
        """The §39 failure path — delegates to :func:`flow.failure.handle_failure`.

        ``promote()`` would raise on a failed context anyway; the handler simply never calls it,
        so the previously published artifacts stay untouched, and it archives (or discards) this
        run's staging tree so ``artifacts/_staging/`` never keeps a failed run's leftovers.
        """
        error = self._failure_error or RuntimeError(
            (self.state.errors[-1]["message"] if self.state.errors else None)
            or "unexpected failure with no recorded error"
        )
        flow_failure.handle_failure(self.state, self._ctx, error, keep_failed=self._keep_failed)

    # -- 1 → router --------------------------------------------------------
    @start()
    def dataset_intake(self) -> str:
        self._run("dataset_intake", flow_steps.dataset_intake)
        return self.state.status

    @router(dataset_intake)
    def route_after_intake(self) -> str:
        return FAIL if self.state.status == "failed" else INTAKE_OK

    # -- 2, 3 → router -----------------------------------------------------
    @listen(INTAKE_OK)
    def data_analyst_work(self) -> str:
        self._run("data_analyst_work", flow_steps.data_analyst_work)
        return self.state.status

    @listen(data_analyst_work)
    def contract_validation(self) -> str:
        self._run("contract_validation", flow_steps.contract_validation)
        return self.state.status

    @router(contract_validation)
    def route_after_contract(self) -> str:
        return FAIL if self.state.status == "failed" else CONTRACT_OK

    # -- 4, 5 → router -----------------------------------------------------
    @listen(CONTRACT_OK)
    def data_scientist_work(self) -> str:
        self._run("data_scientist_work", flow_steps.data_scientist_work)
        return self.state.status

    @listen(data_scientist_work)
    def feature_validation(self) -> str:
        self._run("feature_validation", flow_steps.feature_validation)
        return self.state.status

    @router(feature_validation)
    def route_after_features(self) -> str:
        return FAIL if self.state.status == "failed" else FEATURES_OK

    # -- 6, 7, 8, 9 → router -----------------------------------------------
    @listen(FEATURES_OK)
    def training_and_backtest(self) -> str:
        self._run("training_and_backtest", flow_steps.training_and_backtest)
        return self.state.status

    @listen(training_and_backtest)
    def evaluation_and_champion(self) -> str:
        self._run("evaluation_and_champion", flow_steps.evaluation_and_champion)
        return self.state.status

    @listen(evaluation_and_champion)
    def inventory_policy_calibration(self) -> str:
        self._run("inventory_policy_calibration", flow_steps.inventory_policy_calibration)
        return self.state.status

    @listen(inventory_policy_calibration)
    def artifact_validation(self) -> str:
        self._run("artifact_validation", flow_steps.artifact_validation)
        return self.state.status

    @router(artifact_validation)
    def route_after_artifacts(self) -> str:
        return FAIL if self.state.status == "failed" else ARTIFACTS_OK

    # -- 10 ----------------------------------------------------------------
    @listen(ARTIFACTS_OK)
    def publish(self) -> str:
        self._run("publish", flow_steps.publish)
        if self.state.status == "failed":
            # No router follows step 10, so a failed promote finalises here.
            self._finalize_failure()
        return self.state.status

    @listen(FAIL)
    def handle_failure(self) -> str:
        self._finalize_failure()
        return self.state.status


def run_flow(
    *,
    mode: RunMode = "no-llm",
    raw_path: Path | str | None = None,
    skip_tuning: bool = False,
    base_dir: Path | None = None,
    keep_failed: bool = True,
) -> tuple[FlowState, RunContext]:
    """Start a run context (staging on), kick the Flow off and return ``(state, ctx)``.

    ``run_log.json`` is written immediately after ``RunContext.start()`` so a run that dies before
    its first write leaves an honest ``status: "running"`` on disk instead of the previous run's
    ``success`` (issue §8). The Flow's own handlers finish the context on both the success and the
    failure path; the ``except`` below only covers infrastructure errors (e.g. the Flow failing to
    construct), where the log must still flip to ``failed`` before re-raising.

    ``keep_failed`` (US-32, ``--keep-failed`` / ``--no-keep-failed`` at the CLI) controls what
    happens to a failed run's staging tree: archived under ``logs/failed_runs/<run_id>/`` for
    debugging by default, or discarded outright.
    """
    ctx = RunContext.start(mode=mode, staging=True, base_dir=base_dir)
    ctx.write_run_log()
    flow = RetailForecastFlow(
        ctx, raw_path=raw_path, skip_tuning=skip_tuning, keep_failed=keep_failed
    )
    try:
        flow.kickoff()
    except Exception:
        ctx.finish("failed")
        raise
    if ctx.finished_at is None:  # defensive: no handler ran (should not happen)
        ctx.finish("failed" if flow.state.status != "success" else "success")
    return flow.state, ctx
