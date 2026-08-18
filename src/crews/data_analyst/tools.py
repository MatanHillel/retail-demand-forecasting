"""The approved deterministic tools of the Data Analyst Crew (US-12, PRD §35, §38).

Every tool here is a thin wrapper. It re-computes nothing: it calls the same
:mod:`pipeline` function the ``--no-llm`` run calls, with the same arguments, and returns a
compact JSON summary of what happened. That is what makes an LLM run and a ``--no-llm`` run
numerically identical — the agents decide *when* to call a tool and what the result means, never
what the result is.

Three consequences shape the design, all of them forced by the code these tools wrap:

* **The tools take almost no arguments.** ``clean_transactions(raw_df, cfg, ctx)`` and
  ``run_eda(clean_df, panel_df, waterfall_df, cfg, ctx)`` need DataFrames, configuration objects
  and the run context — none of which a language model can supply. So the tools are built by a
  factory that closes over the run (:func:`make_tools`) and carries the frames between calls in
  :class:`DataAnalystState`. What the model chooses is the *order*, and the two narratives.
* **Every artifact write goes through ``ctx.out(...)``** and every call that writes runs inside
  ``ctx.step(...)`` — ``ctx.log_rows`` raises outside one (``docs/interfaces.md`` §6 rules 1, 3).
* **A misuse by the agent is not a failed run.** Calling a tool before its inputs exist returns a
  JSON error *without* opening a step, so the agent can recover; nothing is recorded in
  ``ctx.errors`` and the run stays promotable. A genuine exception inside a step propagates and
  fails the run, which is what should happen when a deterministic tool breaks.

``data_quality_findings.json`` is written by ``run_eda`` (E1 assembles it from the waterfall and
the panel, which do not exist until cleaning has run), not by the T1 profiling tools.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from crews.common import NarrativeGuard
from pipeline import paths
from pipeline.cleaning import RETURNS_FILENAME, clean_transactions
from pipeline.config import (
    load_cleaning_config,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.contract import write_contract
from pipeline.download import load_raw
from pipeline.eda.io import NAME_PATTERN, load_table
from pipeline.eda.run_eda import run_eda
from pipeline.panel import build_panel, validate_panel
from pipeline.quality import (
    confirm_exclusion_list,
    detect_duplicates,
    list_nonproduct_codes,
    profile_raw,
)
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    write_validation_report,
)

#: Rows of a table an agent is shown. §3 of the issue: only T3 needs table content, and it needs
#: the head of a ranking, not the whole frame — the cheapest prompt that still supports a claim.
TABLE_ROW_CAP = 20

#: Filename of the agent-written data-quality review, kept as a name for backwards compatibility;
#: the location itself is :data:`pipeline.paths.DATA_QUALITY_REVIEW` (added by US-33 §8 so the
#: Flow can guard and re-validate the file without rebuilding its path by hand).
DATA_QUALITY_REVIEW_FILENAME = paths.DATA_QUALITY_REVIEW.name

#: Step names recorded in ``run_log.json``, one per tool call, prefixed so a crew run's steps are
#: distinguishable from the Flow's own.
STEP_PREFIX = "crew_data_analyst"


def review_path() -> Path:
    """Canonical location of ``data_quality_review.md`` (repo-absolute, like every other path)."""
    return paths.DATA_QUALITY_REVIEW


def relative_path(path: Path) -> Path:
    """Repo-relative form — the only form that is safe for ``ctx.out()`` (§6 rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def resolve_read(ctx: RunContext, relative: Path) -> Path:
    """This run's staged copy of a file if there is one, else the final one.

    Mirrors :func:`pipeline.eda.io._resolve_read`: a mid-run reader must see what *this* run
    wrote, and must not use ``ctx.out()`` to find it — that would register the path for
    promotion and make ``promote()`` warn about a file this run only read (§6 rule 7).
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


class DataAnalystState:
    """What the tools hand to each other across a crew run.

    The agent cannot pass a DataFrame from one tool call to the next — it only sees JSON — so the
    frames live here, written by the tool that produced them and read by the tool that needs
    them. Nothing here is ever shown to the model.
    """

    def __init__(self) -> None:
        self.raw_df: pd.DataFrame | None = None
        self.clean_df: pd.DataFrame | None = None
        self.waterfall_df: pd.DataFrame | None = None
        self.panel_df: pd.DataFrame | None = None
        self.profile: dict[str, Any] | None = None
        self.duplicates: dict[str, Any] | None = None
        self.codes_df: pd.DataFrame | None = None
        self.codes_summary: dict[str, Any] | None = None
        self.exclusion_df: pd.DataFrame | None = None
        self.tables: dict[str, pd.DataFrame] = {}
        self.deterministic_insights: str | None = None
        self.contract: dict[str, Any] | None = None
        self.review_written: bool = False
        self.insights_decision: Any | None = None

    def quality_tables(self) -> dict[str, Any]:
        """Everything T1 computed, in the shape the numbers-in-tables guard accepts.

        The data-quality review is checked against *these*, not against the EDA tables: at T1 the
        EDA has not run, and the review's job is to interpret the profiling output.
        """
        tables: dict[str, Any] = {}
        if self.profile is not None:
            tables["profile_raw"] = self.profile
        if self.duplicates is not None:
            tables["detect_duplicates"] = self.duplicates
        if self.codes_df is not None:
            tables["nonproduct_codes"] = self.codes_df
        if self.codes_summary is not None:
            tables["exclusion_summary"] = self.codes_summary
        if self.exclusion_df is not None:
            tables["exclusion_list"] = self.exclusion_df
        return tables


# --------------------------------------------------------------------------
# argument schemas — what the model is allowed to fill in
# --------------------------------------------------------------------------
class NoArgs(BaseModel):
    """No arguments: this tool's inputs are DataFrames and the run context (§8 of the issue)."""

    model_config = ConfigDict(extra="forbid")


class TableNameArgs(BaseModel):
    """The name of one computed EDA table."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Table name in the E<nn>_<topic> form, e.g. 'E06_abc_table'.")


class MarkdownArgs(BaseModel):
    """A narrative the agent has written."""

    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(description="The complete markdown document to publish.")


class _BoundTool(BaseTool):
    """A deterministic pipeline function exposed to an agent, bound to one run.

    ``runner`` is a closure over the ``RunContext`` and the shared :class:`DataAnalystState`,
    which is the whole point: the model picks the tool, the closure supplies every argument the
    model must not choose.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Callable[..., str] = Field(exclude=True)

    def _run(self, **kwargs: Any) -> str:
        return self.runner(**kwargs)


class DataAnalystToolset:
    """The eleven tools of the Data Analyst Crew, all bound to one run.

    Grouped by the agent that owns them, exactly as the PRD §35 table assigns them, plus the
    read-only helpers and the two narrative writers §2 of the issue adds.
    """

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.state = DataAnalystState()
        self.cleaning_cfg = load_cleaning_config()
        self.by_agent: dict[str, list[BaseTool]] = {
            "data_quality_analyst": [
                self._tool(
                    "profile_raw",
                    "Profile the raw extract: row counts, missing values per column, "
                    "cancellations and adjustments, non-positive prices and quantities, the "
                    "partial month and the source-sheet overlap. Computes nothing new - it "
                    "returns the pipeline's own profile. No arguments.",
                    NoArgs,
                    self._profile_raw,
                ),
                self._tool(
                    "detect_duplicates",
                    "Count exact duplicate rows in the raw extract and apply the configured "
                    "warning thresholds. Returns the counts, the shares and whether the "
                    "threshold was breached. No arguments.",
                    NoArgs,
                    self._detect_duplicates,
                ),
                self._tool(
                    "list_nonproduct_codes",
                    "List every stock code that does not match the inventory code pattern, then "
                    "confirm the exclusion list against the data: listed codes found in the data "
                    "become 'confirmed', codes found but not listed are appended as "
                    "'review_needed' and are NOT excluded. No arguments.",
                    NoArgs,
                    self._list_nonproduct_codes,
                ),
                self._tool(
                    "write_data_quality_review",
                    "Publish your data-quality review as markdown. Every number in it is checked "
                    "against the tool outputs above and the document is rejected if any number "
                    "is not among them. Argument: markdown.",
                    MarkdownArgs,
                    self._write_data_quality_review,
                ),
            ],
            "data_preparation_analyst": [
                self._tool(
                    "clean_transactions",
                    "Run the cleaning pipeline: normalise keys, derive the month and the "
                    "partial-month flag, remove cancellations and adjustments, drop non-positive "
                    "quantities and prices, de-duplicate and apply the exclusion list. Writes "
                    "clean_transactions.parquet and the cleaning waterfall. No arguments.",
                    NoArgs,
                    self._clean_transactions,
                ),
                self._tool(
                    "build_panel",
                    "Aggregate the cleaned sales lines to one row per product per month, "
                    "zero-fill the months with no sales, attach returned units and the canonical "
                    "description, then validate the result. Writes clean_data.csv. "
                    "Requires clean_transactions to have run. No arguments.",
                    NoArgs,
                    self._build_panel,
                ),
            ],
            "business_eda_analyst": [
                self._tool(
                    "run_eda",
                    "Run all fourteen analyses (E1-E14), write every figure and table, assemble "
                    "eda_report.html, data_quality_findings.json and the deterministic "
                    "insights.md. Requires build_panel to have run. No arguments.",
                    NoArgs,
                    self._run_eda,
                ),
                self._tool(
                    "read_table",
                    "Read one computed EDA table by name, capped at "
                    f"{TABLE_ROW_CAP} rows. These tables are the ONLY place your numbers may "
                    "come from. Argument: name, in the E<nn>_<topic> form.",
                    TableNameArgs,
                    self._read_table,
                ),
                self._tool(
                    "read_findings",
                    "Read data_quality_findings.json - the duplicate counts, non-product codes, "
                    "abnormal lines and panel preview. No arguments.",
                    NoArgs,
                    self._read_findings,
                ),
                self._tool(
                    "write_insights_narrative",
                    "Publish your rewritten insights.md. Every number in it is checked against "
                    "the computed EDA tables; if any number is not in a table the deterministic "
                    "version is restored instead and your text is discarded. Argument: markdown.",
                    MarkdownArgs,
                    self._write_insights_narrative,
                ),
                self._tool(
                    "write_contract",
                    "Write dataset_contract.json from the panel and the configuration: schema, "
                    "grain, primary key, date range, cleaning assumptions, exclusion list, the "
                    "active rule, the partial-month rule and the leakage rules. "
                    "Requires build_panel to have run. No arguments.",
                    NoArgs,
                    self._write_contract,
                ),
            ],
        }

    # -- plumbing -----------------------------------------------------------
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

    def _step(self, name: str) -> Any:
        return self.ctx.step(f"{STEP_PREFIX}:{name}")

    def _raw(self) -> pd.DataFrame:
        """The raw extract, loaded once per run and recorded on it."""
        if self.state.raw_df is None:
            frame, _ = load_raw(ctx=self.ctx)
            self.state.raw_df = frame
        return self.state.raw_df

    # -- Data Quality Analyst ----------------------------------------------
    def _profile_raw(self) -> str:
        with self._step("profile_raw"):
            profile = profile_raw(self._raw(), self.cleaning_cfg)
            self.state.profile = profile
        raw_profile = profile["raw_profile"]
        cancellations = profile["cancellations_adjustments"]["cancellations"]
        return _json(
            {
                "rows": raw_profile["rows"],
                "columns": raw_profile["columns"],
                "distinct_invoices": raw_profile["distinct_invoices"],
                "distinct_stock_codes": raw_profile["distinct_stock_codes"],
                "date_range": raw_profile["date_range"],
                "missing_customer_id": profile["missing_values"]["customer_id"],
                "cancellation_rows": cancellations["rows"],
                "cancellation_row_share": cancellations["row_share"],
                "adjustment_rows": profile["cancellations_adjustments"]["adjustments"]["rows"],
                "nonpositive_price_rows": raw_profile["nonpositive_price"]["rows"],
                "nonpositive_quantity_rows": (
                    raw_profile["nonpositive_quantity_without_cancellation"]["rows"]
                ),
                "partial_month": profile["partial_month"],
                "source_sheet_overlap": raw_profile["source_sheets"].get("overlap", {}),
            }
        )

    def _detect_duplicates(self) -> str:
        with self._step("detect_duplicates"):
            duplicates = detect_duplicates(self._raw(), self.cleaning_cfg)
            self.state.duplicates = duplicates
        return _json(
            {key: value for key, value in duplicates.items() if key != "top_duplicated_lines"}
        )

    def _list_nonproduct_codes(self) -> str:
        with self._step("list_nonproduct_codes"):
            codes_df = list_nonproduct_codes(self._raw(), self.cleaning_cfg)
            exclusion_df = confirm_exclusion_list(codes_df, self.ctx)
            self.state.codes_df = codes_df
            self.state.exclusion_df = exclusion_df
        review_needed = exclusion_df.loc[exclusion_df["status"] == "review_needed", "stock_code"]
        # Counted here, in deterministic code, and kept on the state — the exclusion list itself
        # holds no numeric column, so a narrative citing "28 confirmed codes" would otherwise be
        # quoting a number that is in no table, which is precisely what §38 forbids. Recording
        # the summary is what makes those counts quotable by the review.
        summary = {
            "codes_found": int(len(codes_df)),
            "already_on_exclusion_list": int(codes_df["in_exclusion_list"].sum())
            if len(codes_df)
            else 0,
            "exclusion_list_rows": int(len(exclusion_df)),
            "confirmed": int((exclusion_df["status"] == "confirmed").sum()),
            "review_needed": int(len(review_needed)),
            "review_needed_codes": sorted(review_needed.tolist()),
        }
        self.state.codes_summary = summary
        return _json(
            {
                **summary,
                "codes": _frame_records(codes_df),
                "note": (
                    "review_needed codes are flagged for a human decision and are NOT excluded."
                ),
            }
        )

    def _write_data_quality_review(self, markdown: str) -> str:
        tables = self.state.quality_tables()
        if not tables:
            return _error(
                "run profile_raw, detect_duplicates and list_nonproduct_codes first — there is "
                "nothing yet to check your numbers against"
            )
        guard = NarrativeGuard("data quality review", tables, fallback="")
        decision = guard.review(markdown)
        if not decision.accepted:
            return _error(
                "rejected: these numbers are in none of the tool outputs — remove them or quote "
                "the tool output exactly",
                unmatched=decision.unmatched,
            )
        with self._step("write_data_quality_review"):
            relative = relative_path(review_path())
            self.ctx.out(relative).write_text(markdown, encoding="utf-8", newline="\n")
            self.ctx.record_artifact("data_quality_review", relative)
            self.state.review_written = True
            self.ctx.logger.info(decision.message)
        return _json(
            {
                "written": relative.as_posix(),
                "numbers_checked": decision.checked,
                "accepted": True,
            }
        )

    # -- Data Preparation Analyst ------------------------------------------
    def _clean_transactions(self) -> str:
        with self._step("clean_transactions"):
            clean_df, waterfall_df = clean_transactions(self._raw(), self.cleaning_cfg, self.ctx)
            self.state.clean_df = clean_df
            self.state.waterfall_df = waterfall_df
        last = waterfall_df.iloc[-1]
        return _json(
            {
                "rows_in": int(waterfall_df["rows_before"].iloc[0]),
                "rows_out": int(last["rows_after"]),
                "units_out": int(last["units_after"]),
                "steps": _frame_records(waterfall_df),
                "written": [
                    relative_path(paths.CLEAN_TRANSACTIONS).as_posix(),
                    relative_path(paths.PROCESSED_DIR / RETURNS_FILENAME).as_posix(),
                ],
            }
        )

    def _build_panel(self) -> str:
        if self.state.clean_df is None:
            return _error("run clean_transactions first — the panel is built from its output")
        returns_path = resolve_read(self.ctx, relative_path(paths.PROCESSED_DIR / RETURNS_FILENAME))
        with self._step("build_panel"):
            panel = build_panel(
                self.state.clean_df, pd.read_parquet(returns_path), self.cleaning_cfg, self.ctx
            )
            result = validate_panel(panel, self.cleaning_cfg)
            write_validation_report(result, run_id=self.ctx.run_id)
            if not result.passed:
                raise FlowValidationError(result)
            self.state.panel_df = panel
        metrics = self.ctx.metrics
        return _json(
            {
                "rows": metrics["panel_rows"],
                "products": metrics["panel_products"],
                "nonzero_product_months": metrics["panel_nonzero_rows"],
                "zero_filled_rows": metrics["panel_zero_filled_rows"],
                "zero_share": metrics["panel_zero_share"],
                "partial_month_rows": metrics["panel_partial_rows"],
                "validation_passed": result.passed,
                "written": relative_path(paths.CLEAN_DATA).as_posix(),
            }
        )

    # -- Business & EDA Analyst --------------------------------------------
    def _run_eda(self) -> str:
        missing = [
            name
            for name, frame in (
                ("clean_transactions", self.state.clean_df),
                ("build_panel", self.state.panel_df),
            )
            if frame is None
        ]
        if missing:
            return _error(f"run {' and '.join(missing)} first — the analyses read their output")
        with self._step("run_eda"):
            result = run_eda(
                self.state.clean_df,
                self.state.panel_df,
                self.state.waterfall_df,
                self.cleaning_cfg,
                self.ctx,
                raw_df=self.state.raw_df,
            )
            self.state.tables = result["tables"]
            # The deterministic insights.md, read back from the path this run wrote it to. It has
            # already passed the numbers-in-tables guard inside generate_insights(), so it is a
            # known-good fallback rather than an assumed-good one (§38).
            self.state.deterministic_insights = Path(result["insights_path"]).read_text(
                encoding="utf-8"
            )
        return _json(
            {
                "analyses": [entry["id"] for entry in result["index"]],
                "tables": sorted(result["tables"]),
                "insights": len(result["insights"]),
                "written": [
                    relative_path(paths.EDA_REPORT).as_posix(),
                    relative_path(paths.INSIGHTS).as_posix(),
                    relative_path(paths.DATA_QUALITY_FINDINGS).as_posix(),
                ],
            }
        )

    def _read_table(self, name: str) -> str:
        if not NAME_PATTERN.match(name):
            return _error(
                f"{name!r} is not a table name — use the E<nn>_<topic> form",
                available=sorted(self.state.tables),
            )
        frame = self.state.tables.get(name)
        if frame is None:
            try:
                frame = load_table(name, self.ctx)
            except (FileNotFoundError, ValueError):
                return _error(
                    f"no table named {name!r} was computed",
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

    def _read_findings(self) -> str:
        source = resolve_read(self.ctx, relative_path(paths.DATA_QUALITY_FINDINGS))
        if not source.is_file():
            return _error("data_quality_findings.json does not exist yet — run run_eda first")
        findings = json.loads(source.read_text(encoding="utf-8"))
        return _json(
            {
                "duplicates": {
                    key: value
                    for key, value in findings["duplicates"].items()
                    if key != "top_duplicated_lines"
                },
                "nonproduct_codes": {
                    key: value
                    for key, value in findings["nonproduct_codes"].items()
                    if key != "table"
                },
                "abnormal_quantities": {
                    key: value
                    for key, value in findings["abnormal_quantities"].items()
                    if key != "top_lines"
                },
                "partial_month": findings["partial_month"],
                "panel_preview": findings["panel_preview"],
                "warnings": findings["warnings"],
            }
        )

    def _write_insights_narrative(self, markdown: str) -> str:
        if self.state.deterministic_insights is None:
            return _error("run run_eda first — it writes the insights.md your rewrite replaces")
        guard = NarrativeGuard("insights", self.state.tables, self.state.deterministic_insights)
        with self._step("write_insights_narrative"):
            # ``ctx.out()`` resolves the destination for *this* run, so a rejection restores the
            # deterministic text into staging rather than republishing the previous run's file.
            destination = self.ctx.out(relative_path(paths.INSIGHTS))
            decision = guard.publish(markdown, destination, self.ctx)
            self.state.insights_decision = decision
        return _json(
            {
                "accepted": decision.accepted,
                "numbers_checked": decision.checked,
                "unmatched": decision.unmatched,
                "published": relative_path(paths.INSIGHTS).as_posix(),
                "note": (
                    "your text was published"
                    if decision.accepted
                    else "your text was discarded and the deterministic insights.md restored"
                ),
            }
        )

    def _write_contract(self) -> str:
        if self.state.panel_df is None:
            return _error("run build_panel first — the contract describes its output")
        with self._step("write_contract"):
            contract = write_contract(
                self.state.panel_df,
                self.cleaning_cfg,
                load_model_config(),
                load_non_inventory_codes(),
                self.ctx,
            )
            self.state.contract = contract
        return _json(
            {
                "dataset": contract["dataset"],
                "version": contract["version"],
                "grain": contract["grain"],
                "primary_key": contract["primary_key"],
                "date_range": contract["date_range"],
                # ``columns`` is a mapping of column name -> spec, in panel order.
                "columns": list(contract["columns"]),
                "row_counts": contract["row_counts"],
                "written": relative_path(paths.DATASET_CONTRACT).as_posix(),
            }
        )


def make_tools(ctx: RunContext) -> list[BaseTool]:
    """Every Data Analyst tool, bound to ``ctx`` (§8 of the issue).

    Use :class:`DataAnalystToolset` directly when the per-agent grouping or the shared state is
    needed — this is the flat list the signature in the issue names.
    """
    return DataAnalystToolset(ctx).tools


def deterministic_review(state: DataAnalystState) -> str:
    """The fallback ``data_quality_review.md``, built from tool output only.

    Written when the agent never produced a review the guard would accept, so the artifact always
    exists. Every number below is copied out of a tool result — none is computed here — which is
    the same rule the agent is held to (§38).
    """
    profile = state.profile or {}
    raw_profile = profile.get("raw_profile", {})
    duplicates = state.duplicates or {}
    cancellations = profile.get("cancellations_adjustments", {}).get("cancellations", {})
    customer = profile.get("missing_values", {}).get("customer_id", {})
    partial = profile.get("partial_month", {})
    # Counted by the list_nonproduct_codes tool, not here — this function quotes, it does not
    # calculate, which is the rule it exists to demonstrate.
    codes = state.codes_summary or {}
    review_needed = codes.get("review_needed_codes", [])

    lines = [
        "# Data-quality review",
        "",
        "*Deterministic fallback: the Data Quality Analyst agent did not produce a review that "
        "passed the numbers-in-tables guard, so this version was generated from the tool output "
        "(PRD §38).*",
        "",
        "## What the raw extract contains",
        "",
        f"- {raw_profile.get('rows')} raw rows across {raw_profile.get('distinct_invoices')} "
        f"invoices and {raw_profile.get('distinct_stock_codes')} distinct stock codes.",
        f"- {customer.get('rows')} rows have no customer id "
        f"({customer.get('row_share')} of rows). They are **kept**: an anonymous sale still "
        "consumed inventory (§13.2).",
        f"- {cancellations.get('rows')} cancellation rows ({cancellations.get('row_share')} of "
        "rows) are removed from the sales table but never subtracted from demand (§9).",
        f"- {duplicates.get('duplicate_rows')} exact duplicate rows "
        f"({duplicates.get('duplicate_row_share')} of rows); threshold breached: "
        f"{duplicates.get('warning')}.",
        f"- The partial month(s) {', '.join(partial.get('months', []))} hold "
        f"{partial.get('rows')} rows and are never scored (§8).",
        "",
        "## Exclusion list",
        "",
    ]
    if codes:
        lines.append(
            f"- {codes.get('confirmed')} listed non-inventory code(s) were found in the data and "
            f"confirmed; the list holds {codes.get('exclusion_list_rows')} row(s) in total, and "
            f"{codes.get('codes_found')} code(s) in the data match no inventory pattern."
        )
    if review_needed:
        lines.append(
            f"- {codes.get('review_needed')} code(s) match no inventory pattern but are not on "
            f"the list and were flagged `review_needed`, **not** excluded: "
            f"{', '.join(f'`{code}`' for code in review_needed)}. Excluding a code is a human "
            "decision (§12)."
        )
    else:
        lines.append("- No unlisted non-product codes were found; the list needs no additions.")
    lines += [
        "",
        "**Recommendation:** proceed to cleaning. Nothing above blocks the panel; the "
        "`review_needed` codes stay in the data until a human rules on them.",
        "",
    ]
    return "\n".join(lines)
