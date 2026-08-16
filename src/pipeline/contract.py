"""The dataset contract — the machine-checked hand-off between the two crews (PRD Appendix A).

``dataset_contract.json`` is the written promise of what ``clean_data.csv`` contains: its grain,
its primary key, every column with type/nullability/minimum, the date range, the cleaning
assumptions, the documented exclusions (§12), the active rule, the leakage rules, the modelling
split and the row counts. The Business & EDA Analyst (Crew 1, §35) writes it with
:func:`write_contract`; the Feature Engineering Specialist (Crew 2, §36) and Flow step 3 check the
panel against it with :func:`validate_contract` and stop the run gracefully when it does not match
(§37, §39).

Nothing here is typed by hand except the fixed prose of Appendix A. Every month, threshold, the
stock-code pattern and ``k`` are read from ``cleaning_config.yaml`` / ``model_config.yaml``
(§40), and every count is computed from the panel that was actually built.

Division of labour (``docs/interfaces.md`` §4 and §6 rule 5): :func:`validate_contract` is pure —
it returns a :class:`~pipeline.validation.ValidationResult` and touches neither disk nor data. The
caller writes ``validation_report.json`` with ``run_id=ctx.run_id`` and raises
:class:`~pipeline.validation.FlowValidationError` with :func:`contract_failure_message`, whose
wording §39 fixes and US-31/US-32 reuse verbatim.

Inside a Flow run the contract must be passed as the dict returned by :func:`write_contract`,
never re-read from ``paths.DATASET_CONTRACT``: writes are staged, so until ``promote()`` the final
path still holds the *previous* run's contract (§6 rule 7). :func:`validate_contract_files` exists
for the CLI and CI, where both files are final.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.config import (
    CleaningConfig,
    ModelConfig,
    load_cleaning_config,
    load_model_config,
    load_non_inventory_codes,
)
from pipeline.panel import PANEL_COLUMNS
from pipeline.run_context import RunContext
from pipeline.validation import (
    FlowValidationError,
    ValidationResult,
    Violation,
    write_validation_report,
)

#: ``step`` recorded on every violation and on the validation report (§37 step 3).
CONTRACT_STEP = "contract_validation"

#: Schema version of the contract document itself, not of the data.
CONTRACT_VERSION = "1.0"

#: ``dataset`` field — the artifact this contract describes.
DATASET_NAME = "clean_data"

#: Provenance, Appendix A verbatim.
SOURCE = "UCI Online Retail II (CC BY 4.0); Kaggle mirror mashlyn/online-retail-ii-uci"

#: The seven cleaning assumptions of Appendix A, in order (§10, §11, §12, §9).
CLEANING_ASSUMPTIONS: list[str] = [
    "cancellations (C/A invoices) removed",
    "Quantity <= 0 removed",
    "Price <= 0 removed",
    "non-inventory stock codes removed (see config/non_inventory_stockcodes.csv)",
    "exact duplicate rows removed",
    "rows without Customer ID kept",
    "gross demand (returns not subtracted)",
]

#: The four leakage rules of Appendix A (§16, §17, §18.2, §21).
LEAKAGE_RULES: list[str] = [
    "features use months <= forecast_origin only",
    "target_month = forecast_origin + 1",
    "ABC / sigma computed on training window only",
    "the partial month is never a target",
]

#: The left-censoring conventions of §17 — stated in the contract because they explain values that
#: would otherwise look like missing data.
FEATURE_CONVENTIONS: list[str] = [
    "lags are 0 for months before a product's first observed sale",
    "rolling statistics use only observed months (truncated windows) before the dataset start",
    "product_age_months is a lower bound (left-censoring)",
]

#: Failure wording fixed by §39 and reused verbatim by the Flow (US-31/US-32). The
#: ``FLOW STOPPED: `` prefix is added by ``FlowValidationError`` — never include it here.
CONTRACT_MISMATCH_TEMPLATE = "clean_data does not match dataset_contract.json ({count} violations)"

#: ``month`` is a calendar month, not a date: four digits, a hyphen, a valid month number.
MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: Note carried on ``returned_units`` so no later reader mistakes it for a feature (§13.2).
RETURNED_UNITS_NOTE = "EDA only - never a feature"

#: Columns whose contract entry is an integer count with a floor of zero.
_COUNT_COLUMNS: list[str] = [
    "units_sold",
    "invoice_count",
    "sale_line_count",
    "customer_count",
    "max_line_qty",
    "returned_units",
]

#: Columns whose contract entry is a non-negative float.
_FLOAT_COLUMNS: list[str] = ["gross_revenue", "avg_unit_price"]

#: Floor shared by every numeric column of §13.2 — a panel never holds a negative quantity.
_NON_NEGATIVE_MINIMUM = 0

#: How many offending values a violation carries as illustration.
_EXAMPLE_LIMIT = 5


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _repo_relative(path: Path) -> Path:
    """Repo-relative form of a canonical ``paths.*`` constant (``docs/interfaces.md`` rule 12)."""
    return path.relative_to(paths.PROJECT_ROOT)


def _panel_end(date_range: dict[str, Any]) -> str:
    """Last month the panel runs to: the latest partial month, or the last full one."""
    return max([date_range["last_full_month"], *date_range["partial_months"]])


def _month_ordinals(months: pd.Series) -> pd.Series:
    """``YYYY-MM`` strings as a monotone integer, so contiguity is a subtraction."""
    periods = pd.PeriodIndex(months.astype(str), freq="M")
    return pd.Series(periods.astype("int64"), index=months.index)


def _column_specs(cleaning_cfg: CleaningConfig) -> dict[str, dict[str, Any]]:
    """The ``columns`` block: one entry per §13.2 field, in the panel's own column order."""
    specs: dict[str, dict[str, Any]] = {
        "month": {"type": "string", "format": "YYYY-MM", "nullable": False},
        "stock_code": {
            "type": "string",
            "pattern": cleaning_cfg.rules.inventory_code_pattern,
            "nullable": False,
        },
        # 0.4 % of raw rows carry no description, and the panel keeps those products (§9).
        "description": {"type": "string", "nullable": True},
        "is_partial_month": {"type": "bool", "nullable": False},
    }
    for column in _COUNT_COLUMNS:
        specs[column] = {"type": "int", "min": _NON_NEGATIVE_MINIMUM, "nullable": False}
    for column in _FLOAT_COLUMNS:
        specs[column] = {"type": "float", "min": _NON_NEGATIVE_MINIMUM, "nullable": False}
    specs["returned_units"]["note"] = RETURNED_UNITS_NOTE
    # PANEL_COLUMNS is the published §13.2 order; following it keeps the JSON readable and makes
    # a column added to the panel but not to the contract fail loudly here rather than silently.
    return {column: specs[column] for column in PANEL_COLUMNS}


def _date_range(panel_df: pd.DataFrame, cleaning_cfg: CleaningConfig) -> dict[str, Any]:
    """``date_range`` computed from the panel, with the configured boundaries as the reference."""
    months = panel_df["month"].astype(str)
    observed_first = str(months.min()) if len(months) else cleaning_cfg.raw.first_month
    return {
        "first_month": observed_first,
        "last_full_month": cleaning_cfg.raw.last_full_month,
        "partial_months": list(cleaning_cfg.raw.partial_months),
    }


def _row_counts(panel_df: pd.DataFrame) -> dict[str, int]:
    """``row_counts`` computed from the panel that was actually built."""
    if panel_df.empty:
        return {"rows": 0, "products": 0, "zero_rows": 0, "partial_rows": 0}
    units = pd.to_numeric(panel_df["units_sold"], errors="coerce")
    return {
        "rows": int(len(panel_df)),
        "products": int(panel_df["stock_code"].astype(str).nunique()),
        "zero_rows": int((units == 0).sum()),
        "partial_rows": int(panel_df["is_partial_month"].astype(bool).sum()),
    }


def contract_failure_message(result: ValidationResult) -> str:
    """The §39 wording for a contract mismatch, without the ``FLOW STOPPED: `` prefix.

    ``ValidationResult.summary()`` deliberately produces something else (the single violation's
    message, or ``"<step> failed with <n> violations"``), so every caller that stops the Flow on a
    contract mismatch must build the message through this function to keep the wording identical.
    """
    return CONTRACT_MISMATCH_TEMPLATE.format(count=len(result.violations))


# --------------------------------------------------------------------------
# writer (Crew 1, §35)
# --------------------------------------------------------------------------
def write_contract(
    panel_df: pd.DataFrame,
    cleaning_cfg: CleaningConfig,
    model_cfg: ModelConfig,
    exclusion_df: pd.DataFrame,
    ctx: RunContext,
) -> dict[str, Any]:
    """Build the Appendix A contract from the panel and the configuration, write it, return it.

    The returned dict is what Flow step 3 must validate against: the file itself is written
    through ``ctx.out(...)``, so under staging it does not reach ``paths.DATASET_CONTRACT`` until
    ``promote()`` and that location still holds the previous run's contract (§6 rule 7).

    ``data_sha256`` comes from ``ctx.data``, which is populated by ``load_raw`` (US-03). A
    standalone run that never read the raw file therefore records ``null`` — the field identifies
    the input of a Flow run, and is not fabricated when there was none.
    """
    date_range = _date_range(panel_df, cleaning_cfg)
    partial_months = date_range["partial_months"]
    active_k = model_cfg.active_rule.k
    split = model_cfg.split

    contract: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "version": CONTRACT_VERSION,
        "source": SOURCE,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_id": ctx.run_id,
        "data_sha256": ctx.data.sha256,
        "grain": (
            f"one row per stock_code x month, from first observed sale to "
            f"{date_range['last_full_month']} ({', '.join(partial_months)} flagged partial)"
        ),
        "primary_key": ["stock_code", "month"],
        "date_range": date_range,
        "columns": _column_specs(cleaning_cfg),
        "cleaning_assumptions": list(CLEANING_ASSUMPTIONS),
        "exclusion_list": exclusion_df.to_dict("records"),
        "active_rule": {
            "k": active_k,
            "definition": (
                f">= 1 positive sale in the {active_k} months before the target month"
            ),
        },
        "partial_month_rule": (
            f"{', '.join(partial_months)} is incomplete: it may be shown as a partial actual "
            f"(hatched and labelled) and is the target of the latest operational forecast, but it "
            f"never enters a metric ({', '.join(split.never_score)} is never scored)"
        ),
        "leakage_rules": list(LEAKAGE_RULES),
        "modeling_split": {
            "train_targets": f"{split.train_targets.start}..{split.train_targets.end}",
            "test_targets": f"{split.holdout_targets.start}..{split.holdout_targets.end}",
        },
        "row_counts": _row_counts(panel_df),
        "feature_conventions": list(FEATURE_CONVENTIONS),
    }

    observed_first = date_range["first_month"]
    if observed_first != cleaning_cfg.raw.first_month:
        ctx.warn(
            f"panel starts at {observed_first} but cleaning_config.raw.first_month is "
            f"{cleaning_cfg.raw.first_month}; the contract records the observed month"
        )
    observed_end = str(panel_df["month"].astype(str).max()) if not panel_df.empty else None
    expected_end = _panel_end(date_range)
    if observed_end is not None and observed_end != expected_end:
        ctx.warn(
            f"panel ends at {observed_end} but the configured last month is {expected_end}"
        )

    destination = ctx.out(_repo_relative(paths.DATASET_CONTRACT))
    # newline="\n" explicitly: text mode would otherwise write CRLF on Windows, and this file is
    # committed — the same panel must produce the same bytes on every platform (§40).
    destination.write_text(
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    ctx.record_artifact("dataset_contract", _repo_relative(paths.DATASET_CONTRACT))
    ctx.logger.info(
        f"dataset contract written: {contract['row_counts']['rows']} rows, "
        f"{contract['row_counts']['products']} products, "
        f"{len(contract['exclusion_list'])} documented exclusions"
    )
    return contract


# --------------------------------------------------------------------------
# validator (Crew 2 / Flow step 3, §36, §37)
# --------------------------------------------------------------------------
def validate_contract(panel_df: pd.DataFrame, contract: dict[str, Any]) -> ValidationResult:
    """Check ``clean_data`` against the contract — pure: no ``ctx``, no disk, never raises.

    Every failure is one :class:`~pipeline.validation.Violation` with a stable ``rule`` name, so
    the app and the report can group them: ``columns``, ``unexpected_columns``, ``dtype``,
    ``nullable``, ``primary_key``, ``month_format``, ``month_range``, ``first_row_is_a_sale``,
    ``contiguous_months``, ``panel_end``, ``non_negative``, ``stock_code_pattern``,
    ``is_partial_month``.

    A missing column short-circuits: every later rule would fail on the same cause and the run
    would report a dozen violations for one defect.

    A ``row_counts`` difference is **not** a violation (§3) — a fresh contract is written each
    run, so it only means the panel changed since this contract was generated. It is returned in
    ``extra["row_counts"]`` for the caller to re-emit through ``ctx.warn``.
    """
    specs: dict[str, dict[str, Any]] = contract["columns"]
    date_range = contract["date_range"]
    violations: list[Violation] = []
    rows = int(len(panel_df))

    def add(rule: str, message: str, count: int | None = None, examples: list | None = None):
        violations.append(
            Violation(
                step=CONTRACT_STEP, rule=rule, message=message, count=count, examples=examples
            )
        )

    def result(extra: dict[str, Any] | None = None) -> ValidationResult:
        return ValidationResult(
            step=CONTRACT_STEP,
            passed=not violations,
            violations=violations,
            checked_rows=rows,
            extra=extra or {},
        )

    # -- schema ------------------------------------------------------------
    missing = [column for column in specs if column not in panel_df.columns]
    if missing:
        add(
            "columns",
            f"clean_data is missing column(s) {', '.join(missing)}",
            count=len(missing),
            examples=missing,
        )
        return result()

    extra_columns = [column for column in panel_df.columns if column not in specs]
    if extra_columns:
        add(
            "unexpected_columns",
            f"clean_data has column(s) not described by the contract: {', '.join(extra_columns)}",
            count=len(extra_columns),
            examples=extra_columns,
        )

    # -- types and nulls ---------------------------------------------------
    for column, spec in specs.items():
        series = panel_df[column]
        if not spec.get("nullable", True):
            null_count = int(series.isna().sum())
            if null_count:
                add(
                    "nullable",
                    f"{column} is not nullable but has {null_count} missing value(s)",
                    count=null_count,
                )
        present = series[series.notna()]
        bad = _uncoercible(present, spec["type"])
        if bad:
            add(
                "dtype",
                f"{column} holds {len(bad)} value(s) that are not {spec['type']}",
                count=len(bad),
                examples=bad[:_EXAMPLE_LIMIT],
            )

    # -- primary key -------------------------------------------------------
    key = contract["primary_key"]
    duplicated = panel_df.duplicated(subset=key)
    if duplicated.any():
        add(
            "primary_key",
            f"({', '.join(key)}) must be unique",
            count=int(duplicated.sum()),
            examples=panel_df.loc[duplicated, key].head(_EXAMPLE_LIMIT).to_dict("records"),
        )

    # -- month format and range -------------------------------------------
    months = panel_df["month"].astype(str)
    malformed = months[~months.str.match(MONTH_PATTERN)]
    if not malformed.empty:
        add(
            "month_format",
            f"month must match {specs['month']['format']}",
            count=int(len(malformed)),
            examples=sorted(malformed.unique())[:_EXAMPLE_LIMIT],
        )
    panel_end = _panel_end(date_range)
    outside = months[(months < date_range["first_month"]) | (months > panel_end)]
    if not outside.empty:
        add(
            "month_range",
            f"month must lie between {date_range['first_month']} and {panel_end}",
            count=int(len(outside)),
            examples=sorted(outside.unique())[:_EXAMPLE_LIMIT],
        )

    # -- values ------------------------------------------------------------
    for column, spec in specs.items():
        if "min" not in spec:
            continue
        values = pd.to_numeric(panel_df[column], errors="coerce")
        below = values < spec["min"]
        if below.any():
            add(
                "non_negative",
                f"{column} must never be below {spec['min']}",
                count=int(below.sum()),
                examples=values[below].head(_EXAMPLE_LIMIT).tolist(),
            )

    pattern = specs["stock_code"]["pattern"]
    codes = panel_df["stock_code"].astype(str)
    unmatched = codes[~codes.str.contains(pattern, regex=True, na=False)]
    if not unmatched.empty:
        add(
            "stock_code_pattern",
            (
                f"every stock_code must match {pattern} — non-inventory codes are removed by "
                "cleaning and documented in exclusion_list"
            ),
            count=int(len(unmatched)),
            examples=sorted(unmatched.unique())[:_EXAMPLE_LIMIT],
        )

    expected_partial = months.isin(date_range["partial_months"])
    wrong_flag = panel_df["is_partial_month"].astype(bool) != expected_partial
    if wrong_flag.any():
        add(
            "is_partial_month",
            (
                "is_partial_month must be true exactly for "
                f"{', '.join(date_range['partial_months'])}"
            ),
            count=int(wrong_flag.sum()),
            examples=sorted(months[wrong_flag].unique())[:_EXAMPLE_LIMIT],
        )

    # -- grain -------------------------------------------------------------
    violations.extend(_grain_violations(panel_df, months))

    counted = _row_counts(panel_df)
    extra: dict[str, Any] = {}
    if counted != contract["row_counts"]:
        extra["row_counts"] = {"contract": contract["row_counts"], "panel": counted}
    return result(extra)


def _uncoercible(values: pd.Series, declared: str) -> list[Any]:
    """Values of ``values`` that cannot be read as the contract's ``declared`` type."""
    if declared == "string":
        return []
    if declared == "bool":
        allowed = {True, False}
        return [value for value in values.unique() if value not in allowed]
    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.isna()
    if declared == "int":
        # A float column is fine as long as every value is whole: read back from CSV, an integer
        # column that ever held a NaN comes back as float.
        bad = bad | ((numeric % 1) != 0)
    return values[bad].head(_EXAMPLE_LIMIT).tolist()


def _grain_violations(panel_df: pd.DataFrame, months: pd.Series) -> list[Violation]:
    """The §13.2 grain: no row before the first sale, no gap, every product runs to the end."""
    if panel_df.empty:
        return []
    violations: list[Violation] = []
    ordered = panel_df.assign(_month=months).sort_values(
        ["stock_code", "_month"], kind="mergesort"
    )

    first_rows = ordered.groupby("stock_code", sort=False).head(1)
    not_a_sale = pd.to_numeric(first_rows["units_sold"], errors="coerce").fillna(0) <= 0
    if not_a_sale.any():
        violations.append(
            Violation(
                step=CONTRACT_STEP,
                rule="first_row_is_a_sale",
                message="a product's first month must be its first observed sale",
                count=int(not_a_sale.sum()),
                examples=first_rows.loc[not_a_sale, "stock_code"].head(_EXAMPLE_LIMIT).tolist(),
            )
        )

    try:
        ordinals = _month_ordinals(months)
    except ValueError:
        # Malformed months are already reported by the month_format rule; contiguity is
        # meaningless until they are fixed.
        return violations

    spans = (
        pd.DataFrame({"stock_code": panel_df["stock_code"].astype(str), "ordinal": ordinals})
        .groupby("stock_code")["ordinal"]
        .agg(["min", "max", "nunique"])
    )
    gaps = spans[(spans["max"] - spans["min"] + 1) != spans["nunique"]]
    if not gaps.empty:
        violations.append(
            Violation(
                step=CONTRACT_STEP,
                rule="contiguous_months",
                message="each product must have one row per month with no gap",
                count=int(len(gaps)),
                examples=gaps.index[:_EXAMPLE_LIMIT].tolist(),
            )
        )
    # Compared against the *observed* last month, not the configured one: a panel that stops early
    # is one month_range violation, not one per product.
    short = spans[spans["max"] != spans["max"].max()]
    if not short.empty:
        violations.append(
            Violation(
                step=CONTRACT_STEP,
                rule="panel_end",
                message="every product must be zero-filled through the last panel month",
                count=int(len(short)),
                examples=short.index[:_EXAMPLE_LIMIT].tolist(),
            )
        )
    return violations


def read_panel(path: Path) -> pd.DataFrame:
    """Read ``clean_data.csv`` without letting pandas reinterpret the key columns.

    ``stock_code`` must stay a string — inferred as a number, ``01234`` loses its leading zero and
    ``month`` becomes a timestamp, and both would fail the contract for a reason that is not in
    the data.
    """
    return pd.read_csv(path, dtype={"stock_code": "string", "month": "string"})


def validate_contract_files(clean_data_path: Path, contract_path: Path) -> ValidationResult:
    """Validate two files on disk — for the CLI and CI, where both are final (§6 rule 7).

    The Flow must not use this: mid-run the contract exists only under staging, and this would
    silently check against the previous run's promise.
    """
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    return validate_contract(read_panel(Path(clean_data_path)), contract)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _write_command(ctx: RunContext, clean_data_path: Path) -> int:
    with ctx.step("write_contract", inputs={"clean_data": str(clean_data_path)}):
        if not clean_data_path.is_file():
            raise FileNotFoundError(
                f"{clean_data_path} not found. Run `python -m pipeline.panel` first."
            )
        contract = write_contract(
            read_panel(clean_data_path),
            load_cleaning_config(),
            load_model_config(),
            load_non_inventory_codes(),
            ctx,
        )
    counts = contract["row_counts"]
    print(
        f"contract written: {paths.DATASET_CONTRACT.relative_to(paths.PROJECT_ROOT).as_posix()}\n"
        f"rows: {counts['rows']}\n"
        f"products: {counts['products']}\n"
        f"documented exclusions: {len(contract['exclusion_list'])}"
    )
    return 0


def _validate_command(ctx: RunContext, clean_data_path: Path, contract_path: Path) -> int:
    with ctx.step(
        CONTRACT_STEP,
        inputs={"clean_data": str(clean_data_path), "contract": str(contract_path)},
    ):
        for required in (clean_data_path, contract_path):
            if not required.is_file():
                raise FileNotFoundError(f"{required} not found")
        result = validate_contract_files(clean_data_path, contract_path)
        write_validation_report(result, run_id=ctx.run_id)
        if "row_counts" in result.extra:
            ctx.warn(
                "row_counts differ from the contract "
                f"({result.extra['row_counts']}); a fresh contract is written each run"
            )
        if not result.passed:
            raise FlowValidationError(result, contract_failure_message(result))
    print(f"contract validation: PASSED ({result.checked_rows} rows checked)")
    return 0


def run(argv: list[str] | None = None) -> int:
    """Standalone entry point (AC 1, AC 2). Exit codes: 0 success, 2 validation stop, 1 crash."""
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.contract",
        description="Write or validate artifacts/contracts/dataset_contract.json (PRD Appendix A)",
    )
    parser.add_argument("command", choices=["write", "validate"])
    parser.add_argument(
        "--clean-data",
        type=Path,
        default=paths.CLEAN_DATA,
        help="panel to read (default: the committed data/processed/clean_data.csv)",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=paths.DATASET_CONTRACT,
        help="contract to validate against (default: the promoted contract)",
    )
    args = parser.parse_args(argv)

    ctx = RunContext.start(mode="no-llm")
    try:
        if args.command == "write":
            code = _write_command(ctx, args.clean_data)
        else:
            code = _validate_command(ctx, args.clean_data, args.contract)
    except FlowValidationError as error:
        ctx.finish(status="failed")
        print(str(error), file=sys.stderr)
        return 2
    except Exception:
        ctx.finish(status="failed")
        raise
    ctx.finish()
    return code


if __name__ == "__main__":
    raise SystemExit(run())
