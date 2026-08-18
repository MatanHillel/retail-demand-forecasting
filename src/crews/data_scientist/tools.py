"""The approved deterministic tools of the Data Scientist Crew (US-26, PRD §36, §38).

Every tool here is a thin wrapper around a :mod:`pipeline` function — the same function
:func:`flow.steps.data_scientist_work` / ``training_and_backtest`` / ``evaluation_and_champion`` /
``inventory_policy_calibration`` calls, with the same arguments, in the same order. That is what
makes an LLM run and a ``--no-llm`` run numerically identical: the agents decide *when* to call a
tool and what its result means, never what the result is (§38).

Three consequences shape the design, exactly as in :mod:`crews.data_analyst.tools` (US-12):

* **The tools take almost no arguments.** Nearly every function they wrap needs DataFrames, a
  config object and the run context — none of which a language model can supply. So the tools are
  built by a factory that closes over the run (:func:`make_tools`) and carry the frames between
  calls on :class:`DataScientistState`.
* **Every artifact write goes through ``ctx.out(...)``.** Functions that already open their own
  named step (``backtest``, ``evaluate``, ``run_sigma``, ``select_champion``,
  ``run_quarterly_aggregation``) are called directly; the rest are wrapped in ``self._step(...)``
  (``docs/interfaces.md`` §6 rules 1, 3).
* **Stop-on-validation-failure is the wrapper's job, not the agent's** (issue §8 interface
  corrections). ``validate_contract_tool`` and ``leakage_check_tool`` write the validation report
  and raise :class:`~pipeline.validation.FlowValidationError` themselves the moment a check comes
  back ``passed=false`` — the agent is never trusted to "stop" on its own reading of a JSON blob.
  A tool called before its inputs exist instead returns a recoverable JSON ``{"error": ...}``
  *without* opening a step, exactly like the Data Analyst Crew's tools.

Training-window ABC (:mod:`pipeline.split`) is needed by ``evaluate_tool``, ``robust_sigma_tool``
and ``latest_forecast_tool`` but has no tool of its own in the issue's list — it is computed once,
lazily, the first time any of them needs it (:meth:`DataScientistToolset._abc_train_df`), the same
way :class:`crews.data_analyst.tools.DataAnalystToolset` lazily loads the raw extract.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from crews.common import GuardDecision, NarrativeGuard
from pipeline import paths
from pipeline.backtest import backtest, backtest_summary, write_backtest_summary
from pipeline.champion import ChampionDecision, select_champion
from pipeline.config import load_cleaning_config, load_inventory_policy, load_model_config
from pipeline.contract import contract_failure_message, read_panel, validate_contract
from pipeline.evaluate import evaluate
from pipeline.feature_validation import (
    LEAKAGE_FAILURE_MESSAGE,
    leakage_check,
    validate_features,
    write_feature_validation,
)
from pipeline.feature_validation import STEP as FEATURE_VALIDATION_STEP
from pipeline.features import build_features, write_features
from pipeline.inventory import STEP_NAME as INVENTORY_STEP_NAME
from pipeline.inventory import run_inventory_simulation
from pipeline.latest_forecast import STEP_NAME as LATEST_FORECAST_STEP_NAME
from pipeline.latest_forecast import run_latest_forecast
from pipeline.models import TRAINABLE_MODEL_IDS, train_models, tune
from pipeline.narrative import extract_numbers
from pipeline.quarterly import run_quarterly_aggregation
from pipeline.reports import write_all_reports
from pipeline.run_context import RunContext
from pipeline.sigma import run_sigma
from pipeline.split import SplitSpec, abc_train, write_abc_train
from pipeline.validation import FlowValidationError, ValidationResult, write_validation_report

#: Rows of a table an agent is shown — the same cap the Data Analyst Crew uses (§3 of the issue).
TABLE_ROW_CAP = 20

#: Step names recorded in ``run_log.json``, prefixed so a crew run's steps are distinguishable
#: from the Flow's own.
STEP_PREFIX = "crew_data_scientist"

#: The tools the narrative task needs, and the only ones a ``narrative_only`` crew is given
#: (US-33): three readers and the two guarded writers. Everything else recomputes a number the
#: Flow has already computed, which is exactly what narrative-only mode exists to avoid.
NARRATIVE_TOOL_NAMES: tuple[str, ...] = (
    "read_eval_table_tool",
    "read_champion_decision_tool",
    "read_model_meta_tool",
    "write_evaluation_narrative_tool",
    "write_model_card_narrative_tool",
)

#: Table name -> the file it is read back from when the state is hydrated instead of computed
#: (US-33 narrative-only mode). Every entry is a table one of the tools would have put on
#: :attr:`DataScientistState.tables`, written by the very same pipeline function the Flow's
#: steps 6-8 called — so hydrating changes where a number is read from, never what it is.
HYDRATED_TABLES: dict[str, Path] = {
    "holdout_metrics_overall": paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv",
    "holdout_metrics_by_month": paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv",
    "holdout_metrics_by_abc": paths.EVAL_TABLES_DIR / "holdout_metrics_by_abc.csv",
    "improvement_vs_b2": paths.EVAL_TABLES_DIR / "improvement_vs_b2.csv",
    "backtest_consistency": paths.EVAL_TABLES_DIR / "backtest_consistency.csv",
    "backtest_by_origin": paths.EVAL_TABLES_DIR / "backtest_by_origin.csv",
    "sigma_summary": paths.EVAL_TABLES_DIR / "sigma_summary.csv",
    "excess_concentration": paths.EXCESS_CONCENTRATION,
    "inventory_kpis": paths.INVENTORY_KPIS,
    "quarterly_metrics": paths.QUARTERLY_METRICS,
}


def relative_path(path: Path) -> Path:
    """Repo-relative form — the only form that is safe for ``ctx.out()`` (§6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def resolve_read(ctx: RunContext, relative: Path) -> Path:
    """This run's staged copy of a file if there is one, else the final one.

    Mirrors :func:`crews.data_analyst.tools.resolve_read`: a mid-run reader must see what *this*
    run wrote, never ``ctx.out()`` — that would register the path for promotion and make
    ``promote()`` warn about a file this run only read (§6 rule 7).
    """
    staged = ctx.staging_dir / relative
    return staged if staged.is_file() else ctx.base_dir / relative


def _json(payload: dict[str, Any]) -> str:
    """A tool result: compact JSON, never a DataFrame (§3 of the issue)."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(message: str, **extra: Any) -> str:
    """A recoverable mistake by the agent — reported to it, not recorded as a run failure."""
    return _json({"error": message, **extra})


def _frame_records(frame: pd.DataFrame, limit: int = TABLE_ROW_CAP) -> list[dict[str, Any]]:
    """The first ``limit`` rows as plain records, so nothing DataFrame-shaped reaches the model."""
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso"))


def _headings(text: str) -> list[str]:
    """Every ``## `` heading line, in order — the structural skeleton a narrative may not change."""
    return [line for line in text.splitlines() if line.startswith("## ")]


def _table_blocks(text: str) -> list[str]:
    """Every contiguous block of markdown-table rows (``|...|``) in ``text``.

    A narrative rewrite must not alter a single cell of a rendered table — altering any cell breaks
    the verbatim match this is used for (§2 of the issue: "section headings and tables must remain
    unchanged").
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("|"):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


class DataScientistState:
    """What the tools hand to each other across a crew run — never shown to the model directly."""

    def __init__(self) -> None:
        self.panel_df: pd.DataFrame | None = None
        self.contract: dict[str, Any] | None = None
        self.features_df: pd.DataFrame | None = None
        self.split: SplitSpec | None = None
        self.tuned: bool = False
        self.holdout_predictions_df: pd.DataFrame | None = None
        self.backtest_df: pd.DataFrame | None = None
        self.abc_train_df: pd.DataFrame | None = None
        self.eval_frames: dict[str, Any] | None = None
        self.sigma_table_df: pd.DataFrame | None = None
        self.kpis_df: pd.DataFrame | None = None
        self.decision: ChampionDecision | None = None
        self.latest: dict[str, Any] | None = None
        self.quarterly: dict[str, Any] | None = None
        self.reports_written: bool = False
        self.deterministic_evaluation_report: str | None = None
        self.deterministic_model_card: str | None = None
        self.evaluation_decision: Any | None = None
        self.model_card_decision: Any | None = None
        #: name -> DataFrame, populated as each stage's table is computed — the only source
        #: ``read_eval_table_tool`` and the narrative guard are allowed to cite (§38).
        self.tables: dict[str, pd.DataFrame] = {}

    def narrative_tables(self, ctx: RunContext) -> dict[str, Any]:
        """Every table/JSON a narrative rewrite may cite a number from."""
        tables: dict[str, Any] = dict(self.tables)
        if self.decision is not None:
            tables["champion_decision"] = self.decision.model_dump(mode="json")
        if self.contract is not None:
            tables["dataset_contract"] = self.contract
        for key, canonical in (
            ("model_meta", paths.MODEL_META),
            ("candidates_meta", paths.MODELS_DIR / "candidates_meta.json"),
        ):
            source = resolve_read(ctx, relative_path(canonical))
            if source.is_file():
                tables[key] = json.loads(source.read_text(encoding="utf-8"))
        return tables


def hydrate_for_narrative(ctx: RunContext, state: DataScientistState) -> list[str]:
    """Fill ``state`` from the artifacts already on disk, for a narrative-only crew (US-33).

    In narrative-only mode the deterministic tools are not run — the Flow's steps 4-8 already ran
    the identical pipeline functions — so nothing has populated the state the narrative tools
    read: ``write_evaluation_narrative_tool`` would refuse every draft with "run
    write_reports_tool first", and ``read_eval_table_tool`` would report no table by any name.
    This reads the same tables back from **this run's staged copies** (``resolve_read``, never the
    final locations, which still hold the previous run's files — ``docs/interfaces.md`` §6 rule 7)
    and puts them where the tools expect them.

    The distinction that matters: this changes where a number is *read from*, never what it is.
    Every file here was written by the same ``pipeline`` function a ``--no-llm`` run calls, so a
    narrative checked against a hydrated table is checked against exactly the numbers the
    deterministic run produced (§38).

    Returns the names of the sources that were missing, so the caller can decide whether the
    narrative step can run at all. An empty list means the state is complete.
    """
    missing: list[str] = []

    for name, canonical in HYDRATED_TABLES.items():
        source = resolve_read(ctx, relative_path(canonical))
        if not source.is_file():
            missing.append(name)
            continue
        state.tables[name] = pd.read_csv(source)

    decision_path = resolve_read(ctx, relative_path(paths.CHAMPION_DECISION))
    if decision_path.is_file():
        state.decision = ChampionDecision.model_validate(
            json.loads(decision_path.read_text(encoding="utf-8"))
        )
    else:
        missing.append("champion_decision")

    contract_path = resolve_read(ctx, relative_path(paths.DATASET_CONTRACT))
    if contract_path.is_file():
        state.contract = json.loads(contract_path.read_text(encoding="utf-8"))
    else:
        missing.append("dataset_contract")

    # ``read_model_meta_tool`` gates on ``state.latest`` because latest_forecast_tool is what
    # writes model_meta.json; here the Flow's step 8 wrote it, so the gate is opened explicitly
    # and the tool goes on to read the file itself.
    meta_path = resolve_read(ctx, relative_path(paths.MODEL_META))
    if meta_path.is_file():
        state.latest = {"hydrated_from": relative_path(paths.MODEL_META).as_posix()}
    else:
        missing.append("model_meta")

    for attribute, canonical in (
        ("deterministic_evaluation_report", paths.EVALUATION_REPORT),
        ("deterministic_model_card", paths.MODEL_CARD),
    ):
        source = resolve_read(ctx, relative_path(canonical))
        if source.is_file():
            setattr(state, attribute, source.read_text(encoding="utf-8"))
        else:
            missing.append(canonical.name)

    # The two deterministic reports exist and their text is now the fallback every rewrite is
    # judged against — which is precisely what ``reports_written`` gates.
    state.reports_written = (
        state.deterministic_evaluation_report is not None
        and state.deterministic_model_card is not None
    )
    return missing


# --------------------------------------------------------------------------
# argument schemas — what the model is allowed to fill in
# --------------------------------------------------------------------------
class NoArgs(BaseModel):
    """No arguments: this tool's inputs are DataFrames and the run context (§8 of the issue)."""

    model_config = ConfigDict(extra="forbid")


class TableNameArgs(BaseModel):
    """The name of one computed evaluation table."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Table name, e.g. 'holdout_metrics_overall'.")


class MarkdownArgs(BaseModel):
    """A narrative the agent has written."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(description="The complete replacement markdown document to publish.")


class _BoundTool(BaseTool):
    """A deterministic pipeline function exposed to an agent, bound to one run.

    ``runner`` closes over the :class:`~pipeline.run_context.RunContext` and the shared
    :class:`DataScientistState`, which is the whole point: the model picks the tool, the closure
    supplies every argument the model must not choose.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Callable[..., str] = Field(exclude=True)

    def _run(self, **kwargs: Any) -> str:
        return self.runner(**kwargs)


class DataScientistToolset:
    """The tools of the Data Scientist Crew, grouped by the agent that owns them (PRD §36)."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.state = DataScientistState()
        self.by_agent: dict[str, list[BaseTool]] = {
            "feature_engineering_specialist": [
                self._tool(
                    "validate_contract_tool",
                    "Validate clean_data.csv against dataset_contract.json. Stops the run if the "
                    "check fails - it never attempts to fix the data. No arguments.",
                    NoArgs,
                    self._validate_contract,
                ),
                self._tool(
                    "build_features_tool",
                    "Build features.csv from the validated panel: lags, rolling windows, the "
                    "active-product rule and the two calendar columns known in advance. Requires "
                    "validate_contract_tool to have run. No arguments.",
                    NoArgs,
                    self._build_features,
                ),
                self._tool(
                    "leakage_check_tool",
                    "Validate the feature table's structure and prove the no-leakage boundary: "
                    "lags recomputed from clean_data, future months permuted and rebuilt, calendar "
                    "columns checked. Stops the run if either check fails. Requires "
                    "build_features_tool to have run. No arguments.",
                    NoArgs,
                    self._leakage_check,
                ),
            ],
            "forecasting_model_scientist": [
                self._tool(
                    "tune_tool",
                    "Grid-search hyper-parameters for the three GBM model variants on rolling "
                    "folds strictly inside the training window, and write the winning parameters "
                    "back into model_config.yaml. Optional - training uses whatever parameters are "
                    "already configured if this is skipped. Requires build_features_tool to have "
                    "run. No arguments.",
                    NoArgs,
                    self._tune,
                ),
                self._tool(
                    "train_models_tool",
                    "Fit each of the four candidate models (M1-M4) through the training window and "
                    "score them once against the whole hold-out. Requires build_features_tool to "
                    "have run. No arguments.",
                    NoArgs,
                    self._train_models,
                ),
                self._tool(
                    "backtest_tool",
                    "Replay every candidate model and baseline at every rolling origin "
                    "(2010-05...2011-10), producing one prediction per origin per product. "
                    "Requires build_features_tool to have run. No arguments.",
                    NoArgs,
                    self._backtest,
                ),
            ],
            "model_evaluation_inventory_scientist": [
                self._tool(
                    "evaluate_tool",
                    "Score every candidate on the hold-out: overall, by month, by training-window "
                    "ABC class, improvement vs. the B2 baseline and back-test consistency across "
                    "every origin. Requires train_models_tool and backtest_tool to have run. No "
                    "arguments.",
                    NoArgs,
                    self._evaluate,
                ),
                self._tool(
                    "robust_sigma_tool",
                    "Compute the robust sigma (1.4826 x MAD of out-of-sample residuals) for every "
                    "candidate, with the product -> ABC-group -> global fallback. Requires "
                    "backtest_tool to have run. No arguments.",
                    NoArgs,
                    self._robust_sigma,
                ),
                self._tool(
                    "simulate_inventory_tool",
                    "Simulate the forecast_only and forecast_plus_ss inventory policies for every "
                    "candidate on the hold-out and compute fill rate, stockout and excess KPIs. "
                    "Requires evaluate_tool and robust_sigma_tool to have run. No arguments.",
                    NoArgs,
                    self._simulate_inventory,
                ),
                self._tool(
                    "select_champion_tool",
                    "Apply the four PRD champion gates (bias, accuracy, inventory tie-break, "
                    "meaningful improvement) to every candidate, in code, and record the decision. "
                    "Requires evaluate_tool and simulate_inventory_tool to have run. No arguments.",
                    NoArgs,
                    self._select_champion,
                ),
                self._tool(
                    "latest_forecast_tool",
                    "Refit the champion through the last full month and produce the operational "
                    "forecast and Recommended Target Inventory plan for next month. Requires "
                    "select_champion_tool to have run. No arguments.",
                    NoArgs,
                    self._latest_forecast,
                ),
                self._tool(
                    "quarterly_tool",
                    "Aggregate the back-tested one-step-ahead forecasts into quarters and add the "
                    "current partial quarter's rolling operational estimate. Requires "
                    "latest_forecast_tool to have run. No arguments.",
                    NoArgs,
                    self._quarterly,
                ),
                self._tool(
                    "write_reports_tool",
                    "Render the deterministic evaluation_report.md and model_card.md from the "
                    "tables computed above. Requires select_champion_tool, latest_forecast_tool "
                    "and quarterly_tool to have run. No arguments.",
                    NoArgs,
                    self._write_reports,
                ),
                self._tool(
                    "read_eval_table_tool",
                    "Read one computed evaluation table by name, capped at "
                    f"{TABLE_ROW_CAP} rows. These tables, plus read_champion_decision_tool and "
                    "read_model_meta_tool, are the ONLY place your numbers may come from. "
                    "Argument: name.",
                    TableNameArgs,
                    self._read_eval_table,
                ),
                self._tool(
                    "read_champion_decision_tool",
                    "Read the full champion-selection decision trace: the winner, the best "
                    "baseline, whether the improvement was meaningful and each candidate's gate "
                    "results. No arguments.",
                    NoArgs,
                    self._read_champion_decision,
                ),
                self._tool(
                    "read_model_meta_tool",
                    "Read model_meta.json - the refit champion's training window, row count and "
                    "hold-out metrics reference. Requires latest_forecast_tool to have run. No "
                    "arguments.",
                    NoArgs,
                    self._read_model_meta,
                ),
                self._tool(
                    "write_evaluation_narrative_tool",
                    "Publish your rewrite of evaluation_report.md's narrative paragraphs "
                    "(interpretation of the comparison, the gate-trace explanation, the business "
                    "reading of the inventory KPIs). Every heading and every table must be byte-"
                    "identical to the generated version, and every number is checked against the "
                    "computed tables; on any failure the deterministic version is restored and "
                    "your text is discarded. Requires write_reports_tool to have run. Argument: "
                    "markdown.",
                    MarkdownArgs,
                    self._write_evaluation_narrative,
                ),
                self._tool(
                    "write_model_card_narrative_tool",
                    "Publish your rewrite of model_card.md's narrative wording (purpose, "
                    "limitations, ethical considerations). Every heading and every table must be "
                    "byte-identical to the generated version - the five mandatory sections may "
                    "never be renamed, reordered or removed - and every number is checked against "
                    "the computed tables; on any failure the deterministic version is restored and "
                    "your text is discarded. Requires write_reports_tool to have run. Argument: "
                    "markdown.",
                    MarkdownArgs,
                    self._write_model_card_narrative,
                ),
            ],
        }

    # -- plumbing -------------------------------------------------------------
    def _tool(
        self,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        runner: Callable[..., str],
    ) -> BaseTool:
        return _BoundTool(
            name=name, description=description, args_schema=args_schema, runner=runner
        )

    @property
    def tools(self) -> list[BaseTool]:
        """Every tool, in agent order."""
        return [tool for group in self.by_agent.values() for tool in group]

    @property
    def narrative_tools(self) -> list[BaseTool]:
        """Only the tools :data:`NARRATIVE_TOOL_NAMES` names — a narrative-only crew's whole kit."""
        by_name = {tool.name: tool for tool in self.tools}
        return [by_name[name] for name in NARRATIVE_TOOL_NAMES]

    def _step(self, name: str) -> Any:
        return self.ctx.step(f"{STEP_PREFIX}:{name}")

    def _split(self) -> SplitSpec:
        if self.state.split is None:
            self.state.split = SplitSpec.load()
        return self.state.split

    def _abc_train_df(self) -> pd.DataFrame:
        """Training-window ABC, computed once and cached (module docstring)."""
        if self.state.abc_train_df is None:
            policy_cfg = load_inventory_policy()
            with self._step("abc_train"):
                frame = abc_train(self.state.panel_df, self._split(), policy_cfg)
                write_abc_train(frame, self.ctx)
                self.state.abc_train_df = frame
        return self.state.abc_train_df

    # -- Feature Engineering Specialist ---------------------------------------
    def _validate_contract(self) -> str:
        panel_path = resolve_read(self.ctx, relative_path(paths.CLEAN_DATA))
        contract_path = resolve_read(self.ctx, relative_path(paths.DATASET_CONTRACT))
        if not panel_path.is_file() or not contract_path.is_file():
            return _error(
                "clean_data.csv and dataset_contract.json must both exist before this crew can "
                "run - produce them with the Data Analyst Crew or `python -m pipeline` first"
            )
        with self._step("validate_contract"):
            panel = read_panel(panel_path)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            result = validate_contract(panel, contract)
            write_validation_report(result, run_id=self.ctx.run_id)
            if not result.passed:
                raise FlowValidationError(result, contract_failure_message(result))
            self.state.panel_df = panel
            self.state.contract = contract
        return _json(
            {
                "passed": result.passed,
                "checked_rows": result.checked_rows,
                "row_counts": result.extra.get("row_counts", {}),
                "dataset": contract["dataset"],
                "grain": contract["grain"],
            }
        )

    def _build_features(self) -> str:
        if self.state.panel_df is None:
            return _error("run validate_contract_tool first - features are built from its panel")
        cfg = load_model_config()
        cleaning_cfg = load_cleaning_config()
        with self._step("build_features"):
            k = cfg.active_rule.k
            first_target = cfg.split.first_target_month
            last_target = cleaning_cfg.raw.last_full_month
            targets = pd.period_range(first_target, last_target, freq="M")
            candidates = int(self.state.panel_df["stock_code"].nunique() * len(targets))
            frame = build_features(self.state.panel_df, k, first_target, last_target, cfg)
            self.ctx.log_rows(
                "features_active_filter",
                before=candidates,
                removed=candidates - len(frame),
                after=int(len(frame)),
            )
            write_features(frame, self.ctx)
            self.state.features_df = frame
        return _json(
            {
                "rows": int(len(frame)),
                "products": int(frame["stock_code"].nunique()),
                "first_target": first_target,
                "last_target": last_target,
                "written": relative_path(paths.FEATURES).as_posix(),
            }
        )

    def _leakage_check(self) -> str:
        if self.state.features_df is None or self.state.panel_df is None:
            return _error("run build_features_tool first - validation reads its output")
        cfg = load_model_config()
        with self._step(FEATURE_VALIDATION_STEP):
            k = cfg.active_rule.k
            feature_result = validate_features(self.state.features_df, self.state.panel_df, cfg, k)
            leakage_result = leakage_check(
                self.state.features_df, self.state.panel_df, cfg, self.ctx.seed
            )
            write_feature_validation([feature_result, leakage_result], self.ctx)
            combined = ValidationResult(
                step=FEATURE_VALIDATION_STEP,
                passed=feature_result.passed and leakage_result.passed,
                violations=feature_result.violations + leakage_result.violations,
                checked_rows=int(len(self.state.features_df)),
            )
            if not combined.passed:
                message = (
                    LEAKAGE_FAILURE_MESSAGE
                    if not leakage_result.passed
                    else f"feature validation failed ({len(combined.violations)} violations)"
                )
                write_validation_report(combined, run_id=self.ctx.run_id)
                raise FlowValidationError(combined, message)
        return _json(
            {
                "feature_validation_passed": feature_result.passed,
                "leakage_passed": leakage_result.passed,
                "checks": (
                    feature_result.extra.get("checks", []) + leakage_result.extra.get("checks", [])
                ),
            }
        )

    # -- Forecasting Model Scientist -------------------------------------------
    def _tune(self) -> str:
        if self.state.features_df is None:
            return _error("run build_features_tool first - tuning replays the feature rows")
        cfg = load_model_config()
        split = self._split()
        with self._step("tune"):
            result = tune(self.state.features_df, cfg, self.ctx, split)
        # tune() rewrote model_config.yaml and cleared the config cache - reload the split too.
        self.state.split = SplitSpec.load()
        self.state.tuned = True
        best = {
            model_id: {"params": params, "wmape": result["best_scores"][model_id]["wmape"]}
            for model_id, params in result["best_params"].items()
        }
        return _json({"best_params": best})

    def _train_models(self) -> str:
        if self.state.features_df is None:
            return _error("run build_features_tool first - training reads the feature rows")
        cfg = load_model_config()
        split = self._split()
        with self._step("train_models"):
            train_models(self.state.features_df, cfg, self.ctx, split)
            holdout_relative = relative_path(paths.FORECASTS_DIR / "holdout_predictions.csv")
            self.state.holdout_predictions_df = pd.read_csv(
                resolve_read(self.ctx, holdout_relative),
                dtype={
                    "stock_code": "string",
                    "forecast_origin": "string",
                    "target_month": "string",
                    "model": "string",
                },
            )
        return _json(
            {
                "models_trained": list(TRAINABLE_MODEL_IDS),
                "holdout_rows": int(len(self.state.holdout_predictions_df)),
                "written": relative_path(paths.MODELS_DIR / "candidates_meta.json").as_posix(),
            }
        )

    def _backtest(self) -> str:
        if self.state.features_df is None or self.state.panel_df is None:
            return _error("run build_features_tool first - the back-test replays its rows")
        cfg = load_model_config()
        split = self._split()
        # backtest() opens its own ctx.step("backtest") - it is called directly, not wrapped.
        predictions = backtest(self.state.features_df, self.state.panel_df, cfg, self.ctx, split)
        self.state.backtest_df = predictions
        with self._step("backtest_summary"):
            summary = backtest_summary(predictions)
            write_backtest_summary(summary, self.ctx)
            self.state.tables["backtest_by_origin"] = summary
        return _json(
            {
                "rows": int(len(predictions)),
                "origins": sorted(str(value) for value in predictions["forecast_origin"].unique()),
                "written": relative_path(paths.BACKTEST_PREDICTIONS).as_posix(),
            }
        )

    # -- Model Evaluation & Inventory Scientist --------------------------------
    def _evaluate(self) -> str:
        if self.state.holdout_predictions_df is None or self.state.backtest_df is None:
            return _error("run train_models_tool and backtest_tool first - evaluation needs both")
        cfg = load_model_config()
        abc_train_df = self._abc_train_df()
        # evaluate() opens its own ctx.step("evaluate") - called directly, not wrapped.
        frames = evaluate(
            self.state.holdout_predictions_df, self.state.backtest_df, abc_train_df, cfg, self.ctx
        )
        self.state.eval_frames = frames
        self.state.tables.update(
            {
                "holdout_metrics_overall": frames["holdout_metrics_overall"],
                "holdout_metrics_by_month": frames["holdout_metrics_by_month"],
                "holdout_metrics_by_abc": frames["holdout_metrics_by_abc"],
                "improvement_vs_b2": frames["improvement_vs_b2"],
                "backtest_consistency": frames["backtest_consistency"],
            }
        )
        return _json(
            {
                "overall": _frame_records(frames["holdout_metrics_overall"]),
                "written": [
                    relative_path(paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv").as_posix(),
                    relative_path(paths.EVAL_TABLES_DIR / "improvement_vs_b2.csv").as_posix(),
                ],
            }
        )

    def _robust_sigma(self) -> str:
        if self.state.backtest_df is None:
            return _error("run backtest_tool first - sigma is computed from its residuals")
        cfg = load_model_config()
        abc_train_df = self._abc_train_df()
        # run_sigma() opens its own ctx.step("sigma") - called directly, not wrapped.
        table, summary = run_sigma(self.state.backtest_df, abc_train_df, cfg, self.ctx)
        self.state.sigma_table_df = table
        self.state.tables["sigma_summary"] = summary
        return _json(
            {
                "rows": int(len(table)),
                "written": relative_path(paths.SIGMA_TABLE).as_posix(),
            }
        )

    def _simulate_inventory(self) -> str:
        if self.state.eval_frames is None or self.state.sigma_table_df is None:
            return _error(
                "run evaluate_tool and robust_sigma_tool first - the simulation needs both"
            )
        cfg = load_model_config()
        with self._step(INVENTORY_STEP_NAME):
            result = run_inventory_simulation(
                cfg,
                self.ctx,
                wide_df=self.state.eval_frames["holdout_rows_all_models"],
                sigma_df=self.state.sigma_table_df,
            )
        self.state.kpis_df = result["inventory_kpis"]
        self.state.tables["inventory_kpis"] = result["inventory_kpis"]
        self.state.tables["excess_concentration"] = result["excess_concentration"]
        return _json(
            {
                "written": [
                    relative_path(paths.INVENTORY_KPIS).as_posix(),
                    relative_path(paths.HOLDOUT_SIMULATION_ROWS).as_posix(),
                ],
            }
        )

    def _select_champion(self) -> str:
        if self.state.eval_frames is None or self.state.kpis_df is None:
            return _error(
                "run evaluate_tool and simulate_inventory_tool first - gate 3 needs the KPIs"
            )
        cfg = load_model_config()
        # select_champion() opens its own ctx.step - called directly, not wrapped.
        decision = select_champion(
            self.state.eval_frames["holdout_metrics_overall"],
            self.state.eval_frames["holdout_metrics_by_month"],
            self.state.kpis_df,
            cfg,
            self.ctx,
        )
        self.state.decision = decision
        return _json(
            {
                "champion": decision.champion,
                "champion_kind": decision.champion_kind,
                "best_baseline": decision.best_baseline,
                "meaningful_improvement": decision.meaningful_improvement,
                "written": relative_path(paths.CHAMPION_DECISION).as_posix(),
            }
        )

    def _latest_forecast(self) -> str:
        if self.state.decision is None:
            return _error(
                "run select_champion_tool first - the operational forecast refits the champion"
            )
        cfg = load_model_config()
        with self._step(LATEST_FORECAST_STEP_NAME):
            result = run_latest_forecast(
                cfg,
                self.ctx,
                panel_df=self.state.panel_df,
                train_features_df=self.state.features_df,
                backtest_df=self.state.backtest_df,
                abc_train_df=self._abc_train_df(),
            )
        self.state.latest = result
        return _json(
            {
                "champion": result["champion"],
                "rows": int(len(result["latest_forecast"])),
                "written": [
                    relative_path(paths.LATEST_FORECAST).as_posix(),
                    relative_path(paths.INVENTORY_PLAN).as_posix(),
                ],
            }
        )

    def _quarterly(self) -> str:
        if not self.state.latest:
            return _error("run latest_forecast_tool first - quarterly aggregation reads its output")
        cfg = load_model_config()
        cleaning_cfg = load_cleaning_config()
        # run_quarterly_aggregation() opens its own ctx.step - called directly, not wrapped.
        result = run_quarterly_aggregation(
            cfg,
            self.ctx,
            backtest_df=self.state.backtest_df,
            latest_df=self.state.latest["latest_forecast"],
            panel_df=self.state.panel_df,
            cleaning_cfg=cleaning_cfg,
            champion=self.state.latest["champion"],
        )
        self.state.quarterly = result
        self.state.tables["quarterly_metrics"] = result["quarterly_metrics"]
        return _json(
            {
                "rows": int(len(result["quarterly_forecast"])),
                "written": [
                    relative_path(paths.QUARTERLY_FORECAST).as_posix(),
                    relative_path(paths.QUARTERLY_METRICS).as_posix(),
                ],
            }
        )

    def _write_reports(self) -> str:
        if self.state.decision is None or not self.state.latest or not self.state.quarterly:
            return _error(
                "run select_champion_tool, latest_forecast_tool and quarterly_tool first - the "
                "reports read their output"
            )
        with self._step("reports"):
            write_all_reports(self.ctx)
            self.state.reports_written = True
            self.state.deterministic_evaluation_report = resolve_read(
                self.ctx, relative_path(paths.EVALUATION_REPORT)
            ).read_text(encoding="utf-8")
            self.state.deterministic_model_card = resolve_read(
                self.ctx, relative_path(paths.MODEL_CARD)
            ).read_text(encoding="utf-8")
        return _json(
            {
                "written": [
                    relative_path(paths.EVALUATION_REPORT).as_posix(),
                    relative_path(paths.MODEL_CARD).as_posix(),
                ]
            }
        )

    def _read_eval_table(self, name: str) -> str:
        frame = self.state.tables.get(name)
        if frame is None:
            return _error(
                f"no table named {name!r} has been computed yet",
                available=sorted(self.state.tables),
            )
        return _json(
            {
                "name": name,
                "columns": [str(column) for column in frame.columns],
                "rows": int(len(frame)),
                "rows_shown": min(int(len(frame)), TABLE_ROW_CAP),
                "records": _frame_records(frame),
            }
        )

    def _read_champion_decision(self) -> str:
        if self.state.decision is None:
            return _error("run select_champion_tool first")
        decision = self.state.decision
        return _json(
            {
                "champion": decision.champion,
                "champion_kind": decision.champion_kind,
                "best_baseline": decision.best_baseline,
                "meaningful_improvement": decision.meaningful_improvement,
                "improvement_points": decision.improvement_points,
                "gate1_all_failed": decision.gate1_all_failed,
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in decision.candidates
                ],
            }
        )

    def _read_model_meta(self) -> str:
        if not self.state.latest:
            return _error("run latest_forecast_tool first - it writes model_meta.json")
        source = resolve_read(self.ctx, relative_path(paths.MODEL_META))
        if not source.is_file():
            return _error("model_meta.json does not exist yet")
        return _json(json.loads(source.read_text(encoding="utf-8")))

    # -- narrative writers: the guard can only lose (§38) ----------------------
    def _write_narrative(
        self, label: str, canonical: Path, deterministic: str | None, markdown: str
    ) -> str:
        if not self.state.reports_written or deterministic is None:
            return _error(
                "run write_reports_tool first - the narrative rewrites its deterministic output"
            )
        relative = relative_path(canonical)

        if _headings(markdown) != _headings(deterministic):
            return _error(
                f"rejected: {canonical.name} headings must stay exactly as generated",
                expected=_headings(deterministic),
                got=_headings(markdown),
            )
        missing = [block for block in _table_blocks(deterministic) if block not in markdown]
        if missing:
            return _error(
                f"rejected: {canonical.name} tables must not be altered",
                unaltered_tables_required=len(_table_blocks(deterministic)),
                broken_tables=len(missing),
            )

        tables = self.state.narrative_tables(self.ctx)
        # The deterministic version has already been proven against pipeline.reports's own,
        # richer backing (which additionally includes artifact byte sizes and the exact rounded
        # percentage each display shows — see that module's docstring). Any number that survives
        # unchanged from that text is legitimate by construction, so it backs itself here too;
        # this is what lets a rewrite keep the report's own figures without hitting a rounding
        # edge case this crew's narrower table set does not otherwise cover.
        tables["deterministic_report_numbers"] = extract_numbers(deterministic)
        guard = NarrativeGuard(label, tables, deterministic)
        with self._step(f"write_{label}_narrative"):
            destination = self.ctx.out(relative)
            if markdown == deterministic:
                # Already proven: write_all_reports() ran this exact text through its own,
                # richer numbers_in_tables check (it includes artifact byte sizes and the exact
                # rounded percentage backing — see pipeline.reports's module docstring) and would
                # have raised had it failed. Re-checking against this crew's narrower table set
                # could reject a text that is provably correct, so an unmodified resubmission is
                # accepted without re-running the guard.
                destination.write_text(markdown, encoding="utf-8", newline="\n")
                decision = GuardDecision(label=label, accepted=True, text=markdown)
                self.ctx.logger.info(
                    f"{label} narrative accepted: text is byte-identical to the deterministic "
                    "version, already proven against every table by write_all_reports()"
                )
            else:
                decision = guard.publish(markdown, destination, self.ctx)
        if label == "evaluation_report":
            self.state.evaluation_decision = decision
        else:
            self.state.model_card_decision = decision
        return _json(
            {
                "accepted": decision.accepted,
                "numbers_checked": decision.checked,
                "unmatched": decision.unmatched,
                "published": relative.as_posix(),
                "note": (
                    "your text was published"
                    if decision.accepted
                    else "your text was discarded and the deterministic version restored"
                ),
            }
        )

    def _write_evaluation_narrative(self, markdown: str) -> str:
        return self._write_narrative(
            "evaluation_report",
            paths.EVALUATION_REPORT,
            self.state.deterministic_evaluation_report,
            markdown,
        )

    def _write_model_card_narrative(self, markdown: str) -> str:
        return self._write_narrative(
            "model_card", paths.MODEL_CARD, self.state.deterministic_model_card, markdown
        )


def make_tools(ctx: RunContext) -> list[BaseTool]:
    """Every Data Scientist tool, bound to ``ctx`` (§8 of the issue)."""
    return DataScientistToolset(ctx).tools
