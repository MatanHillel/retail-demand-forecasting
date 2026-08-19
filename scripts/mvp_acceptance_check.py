"""MVP acceptance audit — every clause of PRD §49, checked mechanically (US-37).

    python scripts/mvp_acceptance_check.py [--out artifacts/reports/acceptance_report.md]
                                           [--skip-slow]

**Acceptance criteria** are the checklist that says when the product is "done". PRD §49 lists 22 of
them. This script is the **audit**: a program that walks that checklist, decides PASS / FAIL /
MANUAL / PENDING for each clause from the artifacts, the configuration and the test suite, and
writes ``artifacts/reports/acceptance_report.md`` plus ``acceptance_summary.json``. It exits 0 only
when nothing is FAIL, so it can gate the pre-flight for M6.

Four rules shape the implementation, all of them from the issue's §8 interface corrections:

* **It is a post-run audit, not a pipeline step.** It opens no
  :class:`~pipeline.run_context.RunContext` — ``finish()`` would overwrite the very
  ``run_log.json`` being audited, and ``ctx.out()`` would hide the report inside
  ``artifacts/_staging/``. It never calls ``write_validation_report()`` either: that function
  writes ``artifacts/validation_report.json`` in place, which would replace the audited run's
  report with an untraceable one. It calls the pure validators and reads the returned
  :class:`~pipeline.validation.ValidationResult` instead. **It modifies nothing except its own two
  report files.**
* **Existence is never proof.** The final artifact locations still hold the *previous* successful
  run's files before promotion, and after any failed run. So the audit reads ``run_log.json``
  first, names the run it is auditing at the top of the report, gates every run-derived criterion
  on ``status == "success"`` (``status`` has three values — ``running`` is what a killed process
  leaves behind), reads ``warnings[]`` for ``promote()``'s "staged artifact was never written", and
  cross-checks the eight required artifacts against ``run_log.json → artifacts``.
* **The run log corroborates; on its own it does not condemn.** A partial re-run — the report
  generator, say — legitimately registers only its own outputs, so a value the audited run did not
  record is reported in the evidence column rather than turned into a failure. A value that
  *contradicts* the artifacts, and a required file that is missing or empty on disk, are failures.
* **No number is typed from the PRD.** Model ids, months, ``k``, ``z``, the eight artifact names
  and every path come from :mod:`pipeline.config` and :mod:`pipeline.paths`.

MANUAL is used only where the issue allows it: a GitHub setting this machine cannot read. PENDING is
used only for deliverables a later story produces (US-38's slides, US-39's video) and for an
optional configuration value that is not set.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import paths
from pipeline.config import (
    MODEL_IDS,
    load_cleaning_config,
    load_data_sources,
    load_model_config,
)
from pipeline.contract import read_panel, validate_contract_files
from pipeline.eda.insights import MAX_INSIGHTS, MIN_INSIGHTS
from pipeline.inventory import target_inventory
from pipeline.narrative import numbers_in_tables
from pipeline.panel import PANEL_COLUMNS, validate_panel

# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------
PASS = "PASS"
FAIL = "FAIL"
MANUAL = "MANUAL"
PENDING = "PENDING"
STATUSES = (PASS, FAIL, MANUAL, PENDING)

#: ``run_log.json → status`` value a finished, promoted run must carry (docs/interfaces.md §3).
SUCCESS_STATUS = "success"

#: The EDA sections §49.16 names, by id. This is the criterion's own wording, not a threshold — the
#: figure *files* behind each id are looked up in ``eda_tables/index.json`` rather than listed here.
REQUIRED_EDA_SECTIONS: tuple[str, ...] = ("E1", "E2", "E5", "E6", "E7", "E8", "E9", "E11")

#: The five sections a model card must carry (§49.19), matched as ``## <n>. <title>`` headings.
MODEL_CARD_SECTIONS = 5

#: §53.2 / §53.3 — the shape of the two deliverables US-38 and US-39 produce.
PRESENTATION_SLIDES_MIN = 10
PRESENTATION_SLIDES_MAX = 12
DEMO_MAX_SECONDS = 300

#: Test files the audit runs. The fast ones always; the slow one only without ``--skip-slow``.
TEST_FILES: tuple[str, ...] = (
    "test_cleaning.py",
    "test_leakage.py",
    "test_app_smoke.py",
    "test_flow.py",
    "test_flow_no_llm.py",
    "test_readme.py",
)
SLOW_TEST_FILES: tuple[str, ...] = ("test_determinism.py",)

#: ``promote()`` appends this phrase to ``run_log.json → warnings`` and carries on (§8).
NEVER_WRITTEN_WARNING = "staged artifact was never written"


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------
@dataclass
class Result:
    """One row of the report: what was checked, how, the verdict and the evidence behind it."""

    number: int
    criterion: str
    how: str
    status: str
    evidence: str

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {STATUSES}")


@dataclass
class Verdict:
    """What a single check returns: a status and the evidence sentence that justifies it."""

    status: str
    evidence: str


def _ok(evidence: str) -> Verdict:
    return Verdict(PASS, evidence)


def _no(evidence: str) -> Verdict:
    return Verdict(FAIL, evidence)


# --------------------------------------------------------------------------
# running the test suite — one subprocess, results mapped back per file
# --------------------------------------------------------------------------
def run_pytest(test_files: Sequence[Path]) -> dict[str, bool]:
    """Run ``pytest`` once over ``test_files`` and return ``{filename: passed}``.

    One process rather than one per file: the suite's import cost dominates, and the JUnit XML
    pytest writes carries a ``classname`` per test case, which is enough to attribute every failure
    back to its file. A file that produced no test case at all — a collection error — counts as
    failed; silence is not a pass.
    """
    if not test_files:
        return {}
    seen = {path.name: 0 for path in test_files}
    passed = dict.fromkeys(seen, True)
    with tempfile.TemporaryDirectory(prefix="acceptance_junit_") as tmp:
        report = Path(tmp) / "junit.xml"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={report}",
                *[str(path) for path in test_files],
            ],
            cwd=paths.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if not report.is_file():
            sys.stderr.write(
                "acceptance: pytest produced no JUnit report; treating every named file as "
                f"failed.\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}\n"
            )
            return dict.fromkeys(seen, False)
        for case in ET.parse(report).iter("testcase"):
            name = _file_of(case.get("classname", ""), seen)
            if name is None:
                continue
            seen[name] += 1
            if case.find("failure") is not None or case.find("error") is not None:
                passed[name] = False
    for name, cases in seen.items():
        if cases == 0:
            passed[name] = False
    return passed


def _file_of(classname: str, known: dict[str, int]) -> str | None:
    """Map a JUnit ``classname`` (``tests.test_flow`` / ``tests.test_flow.TestX``) to a filename."""
    for part in classname.split("."):
        candidate = f"{part}.py"
        if candidate in known:
            return candidate
    return None


# --------------------------------------------------------------------------
# small readers, all read-only
# --------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _rel(path: Path) -> str:
    """Repo-relative POSIX form — the shape ``run_log.json → artifacts`` uses."""
    try:
        return path.relative_to(paths.PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _months_between(start: str, end: str) -> list[str]:
    """Every ``YYYY-MM`` from ``start`` to ``end``, inclusive."""
    months: list[str] = []
    year, month = (int(part) for part in start.split("-"))
    while f"{year:04d}-{month:02d}" <= end:
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def _missing_columns(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------
class Auditor:
    """Evaluates the §49 checklist. Construct it, call :meth:`run`, then render the two reports."""

    def __init__(
        self,
        *,
        skip_slow: bool = False,
        pytest_runner: Callable[[Sequence[Path]], dict[str, bool]] | None = None,
    ) -> None:
        self.skip_slow = skip_slow
        self._pytest_runner = pytest_runner or run_pytest
        self._pytest_results: dict[str, bool] | None = None
        self._panel: pd.DataFrame | None = None
        self._eda_tables: dict[str, Any] | None = None
        self.run_log: dict[str, Any] | None = None
        self.run_log_error: str | None = None
        self._load_run_log()

    # -- run log -----------------------------------------------------------
    def _load_run_log(self) -> None:
        if not paths.RUN_LOG.is_file():
            self.run_log_error = f"{_rel(paths.RUN_LOG)} does not exist — there is no run to audit"
            return
        try:
            self.run_log = _read_json(paths.RUN_LOG)
        except (OSError, json.JSONDecodeError) as exc:
            self.run_log_error = f"{_rel(paths.RUN_LOG)} is unreadable: {exc}"
            return
        if self.run_status != SUCCESS_STATUS:
            self.run_log_error = (
                f"the audited run {self.run_id} has status {self.run_status!r}, not "
                f"{SUCCESS_STATUS!r} — its artifacts are the previous run's, or half-written"
            )

    @property
    def run_id(self) -> str:
        return str((self.run_log or {}).get("run_id", "unknown"))

    @property
    def run_status(self) -> str:
        return str((self.run_log or {}).get("status", "unknown"))

    @property
    def logged_artifacts(self) -> dict[str, str]:
        return dict((self.run_log or {}).get("artifacts") or {})

    @property
    def run_warnings(self) -> list[str]:
        return [str(item) for item in (self.run_log or {}).get("warnings") or []]

    @property
    def registered_required_artifacts(self) -> int:
        """How many of the eight the audited run itself registered."""
        logged = set(self.logged_artifacts.values())
        return sum(1 for path in paths.REQUIRED_ARTIFACTS if _rel(path) in logged)

    def _gate(self) -> Verdict | None:
        """FAIL for every run-derived criterion unless the audited run really succeeded (§8)."""
        return _no(self.run_log_error) if self.run_log_error else None

    def _corroboration(self, key: str) -> str:
        """How ``run_log.json → artifacts`` speaks about one artifact key."""
        if self.logged_artifacts.get(key):
            return f"registered by run {self.run_id}"
        return (
            f"not registered by run {self.run_id} (a partial re-run registers only its own "
            "outputs)"
        )

    # -- shared readers ----------------------------------------------------
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = read_panel(paths.CLEAN_DATA)
        return self._panel

    def eda_tables(self) -> dict[str, Any]:
        """Every computed EDA table, keyed by name — the universe ``insights.md`` may cite."""
        if self._eda_tables is None:
            tables: dict[str, Any] = {}
            for path in sorted(paths.EDA_TABLES_DIR.glob("*.csv")):
                tables[path.stem] = pd.read_csv(path)
            for path in sorted(paths.EDA_TABLES_DIR.glob("*.json")):
                if path.name == "index.json":
                    continue
                tables[path.stem] = _read_json(path)
            self._eda_tables = tables
        return self._eda_tables

    def test_passed(self, filename: str) -> bool | None:
        """``True``/``False`` for a test file, or ``None`` when it was skipped as slow."""
        if self._pytest_results is None:
            wanted = list(TEST_FILES) + ([] if self.skip_slow else list(SLOW_TEST_FILES))
            self._pytest_results = self._pytest_runner([paths.TESTS_DIR / name for name in wanted])
        return self._pytest_results.get(filename)

    def _test_evidence(self, *filenames: str) -> tuple[bool, str]:
        """Run the named test files and summarise them as ``(all_passed, evidence)``."""
        parts: list[str] = []
        ok = True
        for name in filenames:
            outcome = self.test_passed(name)
            if outcome is None:
                parts.append(f"{name} skipped (--skip-slow)")
            else:
                parts.append(f"{name} {'passed' if outcome else 'FAILED'}")
                ok = ok and outcome
        return ok, "; ".join(parts)

    # ----------------------------------------------------------------------
    # 1. raw data loads and hashes
    # ----------------------------------------------------------------------
    def check_01(self) -> Verdict:
        gate = self._gate()
        if gate:
            return gate
        expected = load_data_sources().expected_sha256
        recorded = ((self.run_log or {}).get("data") or {}).get("sha256")
        if expected is None:
            return Verdict(
                PENDING,
                "config/data_sources.yaml sets no expected_sha256 (the field is optional), so "
                "there is nothing for a download to be compared against",
            )
        if recorded and recorded != expected:
            return _no(
                f"run {self.run_id} recorded data.sha256={recorded}, contradicting the expected "
                f"{expected} in config/data_sources.yaml"
            )
        if not _exists_nonempty(paths.CLEAN_TRANSACTIONS):
            return _no(
                f"{_rel(paths.CLEAN_TRANSACTIONS)} missing — no raw extract was ever loaded"
            )
        provenance = (
            f"run {self.run_id} recorded data.sha256={recorded}"
            if recorded
            else f"run {self.run_id} recorded no data.sha256 (it re-ran a later step only)"
        )
        return _ok(
            f"expected_sha256={expected}; the raw extract loaded into "
            f"{_rel(paths.CLEAN_TRANSACTIONS)} "
            f"({paths.CLEAN_TRANSACTIONS.stat().st_size} bytes); {provenance}"
        )

    # ----------------------------------------------------------------------
    # 2. cleaning waterfall reproducible
    # ----------------------------------------------------------------------
    def check_02(self) -> Verdict:
        table = paths.EDA_TABLES_DIR / "E01_cleaning_waterfall.csv"
        if not _exists_nonempty(table):
            return _no(f"{_rel(table)} missing")
        frame = pd.read_csv(table)
        missing = _missing_columns(frame, ["step_no", "step", "rows_before", "rows_after"])
        if missing:
            return _no(f"{_rel(table)} is missing column(s) {missing}")
        steps = [int(value) for value in frame["step_no"]]
        if steps != list(range(1, len(frame) + 1)):
            return _no(f"{_rel(table)} step_no is not 1..{len(frame)}: {steps}")
        chained = all(
            int(frame["rows_before"].iloc[index]) == int(frame["rows_after"].iloc[index - 1])
            for index in range(1, len(frame))
        )
        if not chained:
            return _no(
                f"{_rel(table)} does not chain — a step's rows_before differs from the previous "
                "step's rows_after, so the waterfall does not add up"
            )
        ok, tests = self._test_evidence("test_cleaning.py", "test_determinism.py")
        return Verdict(
            PASS if ok else FAIL,
            f"{_rel(table)}: {len(frame)} chained steps, "
            f"{int(frame['rows_before'].iloc[0])} -> {int(frame['rows_after'].iloc[-1])} rows; "
            f"{tests}",
        )

    # ----------------------------------------------------------------------
    # 3. clean_data.csv generated
    # ----------------------------------------------------------------------
    def check_03(self) -> Verdict:
        if not _exists_nonempty(paths.CLEAN_DATA):
            return _no(f"{_rel(paths.CLEAN_DATA)} missing or empty")
        panel = self.panel()
        if list(panel.columns) != PANEL_COLUMNS:
            return _no(
                f"{_rel(paths.CLEAN_DATA)} columns {list(panel.columns)} != the §13.2 order "
                f"{PANEL_COLUMNS}"
            )
        duplicates = int(panel.duplicated(subset=["stock_code", "month"]).sum())
        if duplicates:
            return _no(
                f"{_rel(paths.CLEAN_DATA)} has {duplicates} duplicate (stock_code, month) key(s)"
            )
        return _ok(
            f"{_rel(paths.CLEAN_DATA)}: {len(panel)} rows, {len(panel.columns)} columns in the "
            f"§13.2 order, (stock_code, month) unique; {self._corroboration('clean_data')}"
        )

    # ----------------------------------------------------------------------
    # 4. contract generated and validated by code
    # ----------------------------------------------------------------------
    def check_04(self) -> Verdict:
        if not _exists_nonempty(paths.DATASET_CONTRACT):
            return _no(f"{_rel(paths.DATASET_CONTRACT)} missing or empty")
        result = validate_contract_files(paths.CLEAN_DATA, paths.DATASET_CONTRACT)
        evidence = (
            f"validate_contract_files({_rel(paths.CLEAN_DATA)}, {_rel(paths.DATASET_CONTRACT)}) "
            f"-> {result.summary()} ({result.checked_rows} rows checked)"
        )
        return _ok(evidence) if result.passed else _no(evidence)

    # ----------------------------------------------------------------------
    # 5. product x month panel with explicit zero months
    # ----------------------------------------------------------------------
    def check_05(self) -> Verdict:
        panel = self.panel()
        result = validate_panel(panel, load_cleaning_config())
        zero_rows = int((panel["units_sold"].astype("int64") == 0).sum())
        share = zero_rows / len(panel) if len(panel) else 0.0
        evidence = (
            f"validate_panel -> {result.summary()} (its rules include contiguous_months and "
            f"first_row_is_a_sale); {zero_rows} of {len(panel)} rows are zero-sales months "
            f"({share:.1%})"
        )
        if not result.passed:
            return _no(evidence)
        if zero_rows == 0:
            return _no(f"no zero-sales month in the panel, so it was not zero-filled — {evidence}")
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 6. active rule k
    # ----------------------------------------------------------------------
    def check_06(self) -> Verdict:
        k = load_model_config().active_rule.k
        table = paths.EDA_TABLES_DIR / "E08_zero_share_by_k.csv"
        if not _exists_nonempty(table):
            return _no(f"{_rel(table)} missing")
        if not _exists_nonempty(paths.FEATURES):
            return _no(f"{_rel(paths.FEATURES)} missing or empty")
        frame = pd.read_csv(table)
        row = frame[frame["k"] == k]
        if row.empty:
            return _no(f"{_rel(table)} has no row for the configured k={k}")
        expected_rows = int(row["rows"].iloc[0])
        with paths.FEATURES.open(encoding="utf-8") as handle:
            actual_rows = sum(1 for _ in handle) - 1
        evidence = (
            f"model_config.active_rule.k={k}; {_rel(table)} row k={k} expects {expected_rows} "
            f"product-months; {_rel(paths.FEATURES)} has {actual_rows}"
        )
        if "is_configured_k" in frame.columns and not bool(row["is_configured_k"].iloc[0]):
            return _no(f"{evidence}; but that row is not flagged is_configured_k")
        return _ok(evidence) if expected_rows == actual_rows else _no(evidence)

    # ----------------------------------------------------------------------
    # 7. features without leakage
    # ----------------------------------------------------------------------
    def check_07(self) -> Verdict:
        if not _exists_nonempty(paths.FEATURE_VALIDATION):
            return _no(f"{_rel(paths.FEATURE_VALIDATION)} missing")
        payload = _read_json(paths.FEATURE_VALIDATION)
        ok, tests = self._test_evidence("test_leakage.py")
        checks = payload.get("checks") or []
        evidence = (
            f"{_rel(paths.FEATURE_VALIDATION)}: passed={payload.get('passed')} over "
            f"{len(checks)} checks (run {payload.get('run_id')}); {tests}"
        )
        return _ok(evidence) if (payload.get("passed") is True and ok) else _no(evidence)

    # ----------------------------------------------------------------------
    # 8. baselines computed
    # ----------------------------------------------------------------------
    def check_08(self) -> Verdict:
        table = paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv"
        if not _exists_nonempty(table):
            return _no(f"{_rel(table)} missing")
        scored = set(pd.read_csv(table)["model"])
        baselines = load_model_config().baseline_ids
        missing = [model for model in baselines if model not in scored]
        evidence = (
            f"{_rel(table)} scores {sorted(set(baselines) & scored)} of the configured baselines "
            f"{list(baselines)}"
        )
        return _ok(evidence) if not missing else _no(f"{evidence}; missing {missing}")

    # ----------------------------------------------------------------------
    # 9. at least two ML model variations trained
    # ----------------------------------------------------------------------
    def check_09(self) -> Verdict:
        cfg = load_model_config()
        ml_ids = [model for model in MODEL_IDS if model not in cfg.baseline_ids]
        table = paths.EVAL_TABLES_DIR / "holdout_metrics_overall.csv"
        if not _exists_nonempty(table):
            return _no(f"{_rel(table)} missing")
        scored = set(pd.read_csv(table)["model"])
        missing_files = [
            model for model in ml_ids if not _exists_nonempty(paths.candidate_model(model))
        ]
        missing_rows = [model for model in ml_ids if model not in scored]
        evidence = (
            f"{len(ml_ids) - len(missing_files)} of {len(ml_ids)} ML candidates {ml_ids} have a "
            f"{_rel(paths.MODELS_DIR)}/<id>.joblib file, and "
            f"{len(ml_ids) - len(missing_rows)} have a row in {_rel(table)}"
        )
        if missing_files or missing_rows:
            return _no(f"{evidence}; missing files {missing_files}, missing rows {missing_rows}")
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 10. temporal hold-out and rolling origin
    # ----------------------------------------------------------------------
    def check_10(self) -> Verdict:
        cfg = load_model_config()
        by_month = paths.EVAL_TABLES_DIR / "holdout_metrics_by_month.csv"
        for path in (by_month, paths.BACKTEST_PREDICTIONS):
            if not _exists_nonempty(path):
                return _no(f"{_rel(path)} missing")
        expected_months = _months_between(
            cfg.split.holdout_targets.start, cfg.split.holdout_targets.end
        )
        scored_months = sorted(set(pd.read_csv(by_month)["target_month"].astype(str)))
        backtest = pd.read_csv(
            paths.BACKTEST_PREDICTIONS, usecols=["forecast_origin", "target_month"], dtype=str
        )
        origins = sorted(set(backtest["forecast_origin"]))
        expected_origins = _months_between(cfg.backtest.first_origin, cfg.backtest.last_origin)
        never = set(cfg.split.never_score)
        offenders = sorted(never & (set(scored_months) | set(backtest["target_month"])))
        evidence = (
            f"hold-out months {scored_months[0]}..{scored_months[-1]} vs the configured "
            f"{expected_months[0]}..{expected_months[-1]}; back-test origins "
            f"{origins[0]}..{origins[-1]} vs the configured {cfg.backtest.first_origin}.."
            f"{cfg.backtest.last_origin}; never_score {sorted(never)} appears "
            f"{'in ' + str(offenders) if offenders else 'in no scored target month'}"
        )
        if scored_months != expected_months:
            return _no(f"{evidence}; the scored months are not the configured hold-out window")
        if origins != expected_origins:
            return _no(f"{evidence}; the back-test origins are not the configured origin window")
        if offenders:
            return _no(evidence)
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 11. wMAPE and Bias overall / by month / by ABC
    # ----------------------------------------------------------------------
    def check_11(self) -> Verdict:
        names = ("holdout_metrics_overall", "holdout_metrics_by_month", "holdout_metrics_by_abc")
        problems: list[str] = []
        seen: list[str] = []
        for name in names:
            path = paths.EVAL_TABLES_DIR / f"{name}.csv"
            if not _exists_nonempty(path):
                problems.append(f"{_rel(path)} missing")
                continue
            frame = pd.read_csv(path)
            missing = _missing_columns(frame, ["wmape", "bias"])
            if missing:
                problems.append(f"{_rel(path)} lacks {missing}")
            seen.append(f"{name}.csv ({len(frame)} rows)")
        if problems:
            return _no("; ".join(problems))
        return _ok(
            f"{_rel(paths.EVAL_TABLES_DIR)}: {', '.join(seen)} — each carrying both a wmape and a "
            "bias column"
        )

    # ----------------------------------------------------------------------
    # 12. champion selected by the §20 gates
    # ----------------------------------------------------------------------
    def check_12(self) -> Verdict:
        gate = self._gate()
        if gate:
            return gate
        if not _exists_nonempty(paths.CHAMPION_DECISION):
            return _no(f"{_rel(paths.CHAMPION_DECISION)} missing")
        decision = _read_json(paths.CHAMPION_DECISION)
        champion = decision.get("champion")
        candidates = decision.get("candidates") or []
        gate_fields = ("gate1_pass", "gate2_rank", "gate3_decision", "wmape", "bias")
        incomplete = [
            candidate.get("model")
            for candidate in candidates
            if any(name not in candidate for name in gate_fields)
        ]
        logged = (self.run_log or {}).get("champion")
        logged_id = logged.get("champion") if isinstance(logged, dict) else logged
        evidence = (
            f"{_rel(paths.CHAMPION_DECISION)}: champion={champion}, {len(candidates)} candidates "
            f"each carrying {list(gate_fields)}, best_baseline={decision.get('best_baseline')}; "
            f"run {self.run_id} logged champion={logged_id!r}"
        )
        if champion not in MODEL_IDS:
            return _no(f"{evidence}; the champion is not one of the canonical {list(MODEL_IDS)}")
        if incomplete:
            return _no(f"{evidence}; candidates missing gate fields: {incomplete}")
        if logged_id is not None and logged_id != champion:
            return _no(f"{evidence}; the run log contradicts champion_decision.json")
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 13. robust sigma, safety stock, target inventory
    # ----------------------------------------------------------------------
    def check_13(self) -> Verdict:
        for path in (paths.SIGMA_TABLE, paths.INVENTORY_PLAN):
            if not _exists_nonempty(path):
                return _no(f"{_rel(path)} missing")
        plan = pd.read_csv(paths.INVENTORY_PLAN)
        missing = _missing_columns(
            plan, ["forecast", "sigma", "z", "safety_stock", "target_inventory", "status"]
        )
        if missing:
            return _no(f"{_rel(paths.INVENTORY_PLAN)} lacks {missing}")
        # An inactive or brand-new product carries no forecast, so there is nothing to recompute
        # for it — those rows are counted in the evidence rather than silently dropped.
        planned = plan.dropna(subset=["forecast", "sigma", "z", "target_inventory"])
        skipped = len(plan) - len(planned)
        sample = planned.head(load_model_config().validation.lag_sample_rows)
        if sample.empty:
            return _no(f"{_rel(paths.INVENTORY_PLAN)} has no row carrying a forecast")
        mismatches = 0
        for z in sorted(sample["z"].unique()):
            rows = sample[sample["z"] == z]
            recomputed = pd.Series(
                target_inventory(rows["forecast"], rows["sigma"], float(z)), index=rows.index
            )
            mismatches += int((recomputed != rows["target_inventory"].astype(int)).sum())
        sigma_sources = pd.read_csv(paths.SIGMA_TABLE, usecols=["sigma_source"])["sigma_source"]
        evidence = (
            f"{_rel(paths.SIGMA_TABLE)}: sigma_source levels "
            f"{sorted(sigma_sources.dropna().unique().tolist())}; "
            f"inventory.target_inventory() recomputed on {len(sample)} of the "
            f"{len(planned)} forecast rows of {_rel(paths.INVENTORY_PLAN)} "
            f"(z={sorted(sample['z'].unique())}, {skipped} rows carry no forecast: "
            f"{sorted(plan.loc[plan['forecast'].isna(), 'status'].unique().tolist())}) — "
            f"{mismatches} mismatch(es)"
        )
        return _ok(evidence) if mismatches == 0 else _no(evidence)

    # ----------------------------------------------------------------------
    # 14. inventory simulated for the ML and the baseline policy
    # ----------------------------------------------------------------------
    def check_14(self) -> Verdict:
        if not _exists_nonempty(paths.INVENTORY_KPIS):
            return _no(f"{_rel(paths.INVENTORY_KPIS)} missing")
        cfg = load_model_config()
        kpis = pd.read_csv(paths.INVENTORY_KPIS)
        champion = (
            _read_json(paths.CHAMPION_DECISION).get("champion")
            if _exists_nonempty(paths.CHAMPION_DECISION)
            else None
        )
        wanted = [model for model in (champion, cfg.main_baseline_id) if model]
        policies = sorted(set(kpis["policy"]))
        missing = [
            f"{model}/{policy}"
            for model in wanted
            for policy in policies
            if kpis[(kpis["model"] == model) & (kpis["policy"] == policy)].empty
        ]
        evidence = (
            f"{_rel(paths.INVENTORY_KPIS)}: policies {policies}; the champion {champion} and the "
            f"main baseline {cfg.main_baseline_id} are simulated under every one of them"
        )
        if len(policies) < 2:
            return _no(f"{evidence}; fewer than two policies were simulated")
        return _ok(evidence) if not missing else _no(f"{evidence}; missing {missing}")

    # ----------------------------------------------------------------------
    # 15. Streamlit screens 1-7 and the CSV download
    # ----------------------------------------------------------------------
    def check_15(self) -> Verdict:
        app_dir = paths.PROJECT_ROOT / "src" / "app"
        home = app_dir / "Home.py"
        pages = sorted((app_dir / "pages").glob("[0-9]*.py"))
        forecasts_page = next((page for page in pages if page.name.startswith("2_")), None)
        ok, tests = self._test_evidence("test_app_smoke.py")
        has_download = bool(
            forecasts_page and "st.download_button" in forecasts_page.read_text(encoding="utf-8")
        )
        evidence = (
            f"{_rel(home)} plus pages {[page.name for page in pages]} — screens "
            f"1-{len(pages) + 1}; st.download_button in "
            f"{forecasts_page.name if forecasts_page else '(no screen 2)'}: {has_download}; {tests}"
        )
        if not home.is_file() or len(pages) < 6 or not has_download or not ok:
            return _no(evidence)
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 16. eda_report.html figures and insights.md
    # ----------------------------------------------------------------------
    def check_16(self) -> Verdict:
        for path in (paths.EDA_REPORT, paths.INSIGHTS):
            if not _exists_nonempty(path):
                return _no(f"{_rel(path)} missing")
        index_path = paths.EDA_TABLES_DIR / "index.json"
        if not _exists_nonempty(index_path):
            return _no(f"{_rel(index_path)} missing — cannot resolve section ids to figures")
        index = {entry["id"]: entry for entry in _read_json(index_path)}
        html = paths.EDA_REPORT.read_text(encoding="utf-8")
        missing_sections = []
        for section in REQUIRED_EDA_SECTIONS:
            figures = ((index.get(section) or {}).get("figure_names")) or []
            if not figures or not all(figure in html for figure in figures):
                missing_sections.append(section)
        embedded = html.count("data:image/png;base64,")

        insights_text = paths.INSIGHTS.read_text(encoding="utf-8")
        numbered = re.findall(r"^\d+\. ", insights_text, flags=re.MULTILINE)
        guard = numbers_in_tables(insights_text, self.eda_tables())
        evidence = (
            f"{_rel(paths.EDA_REPORT)} embeds {embedded} base64 figures and names the figures of "
            f"{len(REQUIRED_EDA_SECTIONS) - len(missing_sections)} of the required sections "
            f"{list(REQUIRED_EDA_SECTIONS)}; {_rel(paths.INSIGHTS)} has {len(numbered)} "
            f"insights (allowed {MIN_INSIGHTS}-{MAX_INSIGHTS}); numbers_in_tables checked "
            f"{guard.checked} numbers against {guard.table_values} table values -> "
            f"passed={guard.passed}"
        )
        if missing_sections:
            return _no(f"{evidence}; sections with no embedded figure: {missing_sections}")
        if embedded == 0:
            return _no(f"{evidence}; the report embeds no base64 figure at all")
        if not MIN_INSIGHTS <= len(numbered) <= MAX_INSIGHTS:
            return _no(evidence)
        if not guard.passed:
            return _no(f"{evidence}; unbacked numbers {guard.unmatched[:5]}")
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 17. the Flow hands off, validates and fails gracefully
    # ----------------------------------------------------------------------
    def check_17(self) -> Verdict:
        flow_doc = paths.DOCS_DIR / "flow.md"
        ok, tests = self._test_evidence("test_flow.py", "test_flow_no_llm.py")
        evidence = (
            f"{tests}; {_rel(flow_doc)} present={flow_doc.is_file()}; "
            f"CI: {_github_latest_run()}"
        )
        if not ok or not flow_doc.is_file():
            return _no(evidence)
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 18. all eight required artifacts, exact names
    # ----------------------------------------------------------------------
    def check_18(self) -> Verdict:
        gate = self._gate()
        if gate:
            return gate
        missing = [path.name for path in paths.REQUIRED_ARTIFACTS if not _exists_nonempty(path)]
        never_written = [
            warning for warning in self.run_warnings if NEVER_WRITTEN_WARNING in warning
        ]
        total = len(paths.REQUIRED_ARTIFACTS)
        evidence = (
            f"{total - len(missing)} of {total} artifacts exist non-empty "
            f"({', '.join(path.name for path in paths.REQUIRED_ARTIFACTS)}); "
            f"{self.registered_required_artifacts} of them are registered in run "
            f"{self.run_id}'s artifacts map; promote() warnings about unwritten artifacts: "
            f"{len(never_written)}"
        )
        if missing:
            return _no(f"{evidence}; missing or empty: {missing}")
        if never_written:
            return _no(f"{evidence}; {never_written}")
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 19. model-card sections and the evaluation report's champion trace
    # ----------------------------------------------------------------------
    def check_19(self) -> Verdict:
        for path in (paths.MODEL_CARD, paths.EVALUATION_REPORT):
            if not _exists_nonempty(path):
                return _no(f"{_rel(path)} missing or empty")
        card = paths.MODEL_CARD.read_text(encoding="utf-8")
        sections = re.findall(r"^## \d+\.\s+(.+)$", card, flags=re.MULTILINE)
        report = paths.EVALUATION_REPORT.read_text(encoding="utf-8")
        champion = (
            _read_json(paths.CHAMPION_DECISION).get("champion")
            if _exists_nonempty(paths.CHAMPION_DECISION)
            else None
        )
        traced = bool(champion) and champion in report
        evidence = (
            f"{_rel(paths.MODEL_CARD)} carries {len(sections)} numbered sections {sections} "
            f"(required {MODEL_CARD_SECTIONS}); {_rel(paths.EVALUATION_REPORT)} names the champion "
            f"{champion}: {traced}"
        )
        if len(sections) < MODEL_CARD_SECTIONS or not traced:
            return _no(evidence)
        return _ok(evidence)

    # ----------------------------------------------------------------------
    # 20. README
    # ----------------------------------------------------------------------
    def check_20(self) -> Verdict:
        readme = paths.PROJECT_ROOT / "README.md"
        ok, tests = self._test_evidence("test_readme.py")
        evidence = f"{_rel(readme)} present={_exists_nonempty(readme)}; {tests}"
        return _ok(evidence) if (ok and _exists_nonempty(readme)) else _no(evidence)

    # ----------------------------------------------------------------------
    # 21. PR-based history and branch protection
    # ----------------------------------------------------------------------
    def check_21(self) -> Verdict:
        doc = paths.DOCS_DIR / "branch_protection.md"
        merges = _pull_request_merges()
        protected, protection_evidence = _github_branch_protection()
        shallow = _is_shallow_clone()
        evidence = (
            f"git log --merges: {len(merges)} merge commit(s) referencing a pull request (latest: "
            f"{merges[0] if merges else 'none'}); {_rel(doc)} present={doc.is_file()}; branch "
            f"protection on main: {protection_evidence}"
        )
        if not doc.is_file():
            return _no(f"{evidence}; the branch-protection settings are not documented")
        if not merges:
            # A CI checkout is shallow by default, so there is no history here to read. That is a
            # property of the checkout, not of the repository — it cannot be called a failure.
            if shallow:
                return Verdict(
                    MANUAL,
                    f"{evidence}; this is a shallow clone, so the merge history is not present "
                    "locally — check the pull-request history on GitHub",
                )
            return _no(f"{evidence}; no pull-request merge found in the history")
        if protected is None:
            return Verdict(MANUAL, f"{evidence} — confirm the GitHub setting in the web UI")
        return _ok(evidence) if protected else _no(evidence)

    # ----------------------------------------------------------------------
    # 22. presentation and demo video (US-38 / US-39)
    # ----------------------------------------------------------------------
    def check_22(self) -> Verdict:
        deck = paths.DOCS_DIR / "presentation.pptx"
        video = paths.DOCS_DIR / "demo.mp4"
        if not deck.is_file() and not video.is_file():
            return Verdict(
                PENDING,
                f"{_rel(deck)} and {_rel(video)} do not exist yet — US-38 and US-39 produce them",
            )
        slides = _slide_count(deck)
        duration = _video_seconds(video)
        evidence = (
            f"{_rel(deck)}: {slides if slides is not None else 'unreadable'} slides (allowed "
            f"{PRESENTATION_SLIDES_MIN}-{PRESENTATION_SLIDES_MAX}); {_rel(video)}: "
            f"{duration if duration is not None else 'unmeasurable'} s (allowed <= "
            f"{DEMO_MAX_SECONDS})"
        )
        if slides is None or duration is None:
            return Verdict(PENDING, f"{evidence} — one deliverable is still missing or unreadable")
        if not PRESENTATION_SLIDES_MIN <= slides <= PRESENTATION_SLIDES_MAX:
            return _no(evidence)
        if duration > DEMO_MAX_SECONDS:
            return _no(evidence)
        return _ok(evidence)

    # ----------------------------------------------------------------------
    def run(self) -> list[Result]:
        """Evaluate all 22 clauses, in order."""
        results: list[Result] = []
        for number, criterion, how in CRITERIA:
            check = getattr(self, f"check_{number:02d}")
            try:
                verdict = check()
            except Exception as exc:  # noqa: BLE001 - an audit never dies on one bad clause
                verdict = _no(f"the check itself raised {type(exc).__name__}: {exc}")
            results.append(
                Result(
                    number=number,
                    criterion=criterion,
                    how=how,
                    status=verdict.status,
                    evidence=verdict.evidence,
                )
            )
        return results


# --------------------------------------------------------------------------
# optional, environment-dependent probes
# --------------------------------------------------------------------------
def _git(*args: str) -> str | None:
    if shutil.which("git") is None:
        return None
    completed = subprocess.run(
        ["git", *args], cwd=paths.PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout if completed.returncode == 0 else None


def _pull_request_merges() -> list[str]:
    """Merge-commit subjects on the main line that reference a pull request."""
    for ref in ("main", "origin/main", "HEAD"):
        output = _git("log", "--merges", "--format=%h %s", ref)
        if not output:
            continue
        subjects = [
            line.strip()
            for line in output.splitlines()
            if re.search(r"(pull request #\d+|\(#\d+\))", line, flags=re.IGNORECASE)
        ]
        if subjects:
            return subjects
    return []


def _is_shallow_clone() -> bool:
    """A CI checkout is shallow by default, so its merge history is simply not there to read."""
    return (_git("rev-parse", "--is-shallow-repository") or "").strip() == "true"


def _gh(*args: str) -> str | None:
    if shutil.which("gh") is None:
        return None
    completed = subprocess.run(
        ["gh", *args], cwd=paths.PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout if completed.returncode == 0 else None


def _github_branch_protection() -> tuple[bool | None, str]:
    """``(protected, evidence)``; ``protected is None`` when this machine cannot tell."""
    output = _gh("api", "repos/{owner}/{repo}/branches/main/protection")
    if output is None:
        return None, "gh unavailable or unauthorised — not readable from here"
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None, "gh returned no JSON"
    contexts = (payload.get("required_status_checks") or {}).get("contexts") or []
    reviews = payload.get("required_pull_request_reviews")
    return bool(contexts and reviews), (
        f"required checks {contexts}, reviews required={bool(reviews)}"
    )


def _github_latest_run() -> str:
    output = _gh("run", "list", "--limit", "1", "--json", "conclusion,name,headBranch")
    if output is None:
        return "gh unavailable — the failure path is verified locally by the flow tests instead"
    try:
        runs = json.loads(output)
    except json.JSONDecodeError:
        return "gh returned no JSON"
    if not runs:
        return "no workflow run found"
    latest = runs[0]
    return (
        f"latest run '{latest.get('name')}' on {latest.get('headBranch')}: "
        f"{latest.get('conclusion')}"
    )


def _slide_count(deck: Path) -> int | None:
    if not deck.is_file():
        return None
    try:
        from pptx import Presentation  # noqa: PLC0415 - optional, only needed once US-38 lands
    except ImportError:
        return None
    try:
        return len(Presentation(str(deck)).slides)
    except Exception:  # noqa: BLE001 - a corrupt deck is "unreadable", not a crash
        return None


def _video_seconds(video: Path) -> float | None:
    if not video.is_file() or shutil.which("ffprobe") is None:
        return None
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


# --------------------------------------------------------------------------
# the checklist itself — number, criterion (PRD §49), how it is checked
# --------------------------------------------------------------------------
CRITERIA: tuple[tuple[int, str, str], ...] = (
    (
        1,
        "Raw data loads and its integrity hash is checked",
        "`data_sources.expected_sha256` vs `run_log.json → data.sha256`, plus the loaded extract",
    ),
    (
        2,
        "The cleaning waterfall is reproducible",
        "`E01_cleaning_waterfall.csv` steps chain; `test_cleaning.py` + `test_determinism.py`",
    ),
    (
        3,
        "`clean_data.csv` is generated",
        "`paths.CLEAN_DATA` exists; columns equal `panel.PANEL_COLUMNS`; the key is unique",
    ),
    (
        4,
        "The dataset contract is generated and validated by code",
        "`contract.validate_contract_files()` over the two final files",
    ),
    (
        5,
        "Product × month panel with explicit zero months",
        "`panel.validate_panel()` plus the share of zero-sales rows",
    ),
    (
        6,
        "The active-product rule uses the configured k",
        "`model_config.active_rule.k` vs `E08_zero_share_by_k.csv` vs the `features.csv` row count",
    ),
    (
        7,
        "Features carry no leakage",
        "`feature_validation.json → passed`; `test_leakage.py`",
    ),
    (
        8,
        "The baselines are computed",
        "every `config.baseline_ids` present in `holdout_metrics_overall.csv`",
    ),
    (
        9,
        "At least two ML model variations are trained",
        "`paths.candidate_model(id)` per non-baseline `MODEL_IDS`, plus a row in the overall table",
    ),
    (
        10,
        "Temporal hold-out and rolling-origin back-test",
        "months vs `split.holdout_targets`, origins vs `backtest.*`, `split.never_score` absent",
    ),
    (
        11,
        "wMAPE and Bias reported overall, by month and by ABC",
        "the three `evaluation_tables/*.csv` each carry a `wmape` and a `bias` column",
    ),
    (
        12,
        "The champion is selected by the §20 gates",
        "`champion_decision.json` gate fields, cross-checked against `run_log.json → champion`",
    ),
    (
        13,
        "Robust σ, safety stock and target inventory computed",
        "`sigma_table.csv` levels; `inventory.target_inventory()` recomputed on plan rows",
    ),
    (
        14,
        "Inventory simulated for the ML and the baseline policy",
        "`inventory_kpis.csv` covers the champion and `main_baseline_id` under every policy",
    ),
    (
        15,
        "Streamlit shows screens 1–7 with a CSV download",
        "`Home.py` + `pages/*.py`; `st.download_button` on screen 2; `test_app_smoke.py`",
    ),
    (
        16,
        "EDA report figures and 8–12 backed insights",
        "`eda_tables/index.json` → figures embedded in the HTML; `narrative.numbers_in_tables()`",
    ),
    (
        17,
        "The Flow hands off, validates and fails gracefully",
        "`test_flow.py`, `test_flow_no_llm.py`, `docs/flow.md`, the latest CI run",
    ),
    (
        18,
        "All required artifacts saved under their exact names",
        "`paths.REQUIRED_ARTIFACTS` exist non-empty; `run_log.json` artifacts map and warnings",
    ),
    (
        19,
        "Model-card sections and the evaluation report's champion trace",
        "`## n.` headings in `model_card.md`; the champion id present in `evaluation_report.md`",
    ),
    (
        20,
        "README with the required sections",
        "`README.md` exists; `test_readme.py` (US-36) passes",
    ),
    (
        21,
        "PR-based history with branch protection",
        "`git log --merges` for pull-request merges; `docs/branch_protection.md`; `gh api`",
    ),
    (
        22,
        "Presentation (10–12 slides) and demo video (≤ 5 min)",
        "`docs/presentation.pptx` via python-pptx; `docs/demo.mp4` via ffprobe",
    ),
)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def counts(results: Sequence[Result]) -> dict[str, int]:
    return {status: sum(1 for result in results if result.status == status) for status in STATUSES}


def _cell(text: str) -> str:
    """A markdown table cell cannot contain a bare pipe or a newline."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_report(auditor: Auditor, results: Sequence[Result], *, generated_at: str) -> str:
    tally = counts(results)
    lines = [
        "# MVP acceptance report",
        "",
        f"*Generated:* {generated_at} · *Checklist:* PRD §49, {len(results)} criteria",
        "",
        f"*Audited run:* `{auditor.run_id}` · *status:* `{auditor.run_status}` · *mode:* "
        f"`{(auditor.run_log or {}).get('mode', 'unknown')}` · *finished:* "
        f"{(auditor.run_log or {}).get('finished_at', 'unknown')}",
        "",
        f"*Run-log corroboration:* {auditor.registered_required_artifacts} of "
        f"{len(paths.REQUIRED_ARTIFACTS)} required artifacts are registered in that run's "
        f"`artifacts` map, and it recorded {len(auditor.run_warnings)} warning(s). A partial "
        "re-run registers only its own outputs, so a gap here is reported as evidence; a "
        "*contradiction*, a missing file, or a `promote()` warning about an unwritten artifact is "
        "a FAIL.",
        "",
        "*Scope:* this audit is read-only — it opens no `RunContext`, never calls "
        "`write_validation_report()`, and modifies nothing except "
        f"`{_rel(paths.ACCEPTANCE_REPORT)}` and `{_rel(paths.ACCEPTANCE_SUMMARY)}`.",
        "",
        "## Summary",
        "",
        "| Result | Count |",
        "|---|---|",
    ]
    lines += [f"| {status} | {tally[status]} |" for status in STATUSES]
    lines += [
        f"| **Total** | **{len(results)}** |",
        "",
        "## Criteria",
        "",
        "| # | Criterion | How checked | Result | Evidence |",
        "|---:|---|---|---|---|",
    ]
    lines += [
        f"| {result.number} | {_cell(result.criterion)} | {_cell(result.how)} | "
        f"**{result.status}** | {_cell(result.evidence)} |"
        for result in results
    ]
    lines += [
        "",
        "## What the statuses mean",
        "",
        "* **PASS** — the check ran, and the evidence in the last column supports the clause.",
        "* **FAIL** — the check ran and the clause does not hold. Any FAIL makes this script exit "
        "non-zero.",
        "* **MANUAL** — only for a GitHub setting this machine cannot read (`gh` missing or "
        "unauthorised); confirm it in the web UI.",
        "* **PENDING** — a deliverable a later story produces (US-38's slides, US-39's video), or "
        "an optional configuration value that is not set.",
        "",
        f"Regenerate with `make acceptance`. `{_rel(paths.DOCS_DIR / 'acceptance.md')}` explains "
        "what each criterion checks and why.",
        "",
    ]
    return "\n".join(lines)


def render_summary(auditor: Auditor, results: Sequence[Result], *, generated_at: str) -> dict:
    return {
        "generated_at": generated_at,
        "run_id": auditor.run_id,
        "run_status": auditor.run_status,
        "total": len(results),
        "counts": counts(results),
        "failed": [result.number for result in results if result.status == FAIL],
        "criteria": [
            {
                "number": result.number,
                "criterion": result.criterion,
                "status": result.status,
                "evidence": result.evidence,
            }
            for result in results
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mvp_acceptance_check",
        description="Audit every PRD §49 acceptance criterion and write acceptance_report.md.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.ACCEPTANCE_REPORT,
        help=f"where to write the markdown report (default: {_rel(paths.ACCEPTANCE_REPORT)})",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help=f"do not run {', '.join(SLOW_TEST_FILES)} (the full pipeline runs twice inside it)",
    )
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    """Print the report without depending on the console's code page.

    The criteria carry §, × and σ. On Windows a redirected stdout defaults to cp1252, which cannot
    encode them, and the whole run would die on the summary print *after* the two report files were
    already written — the audit's verdict lost to a console setting.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: Sequence[str] | None = None) -> int:
    _use_utf8_console()
    args = _parse_args(argv)
    auditor = Auditor(skip_slow=args.skip_slow)
    results = auditor.run()
    generated_at = datetime.now(UTC).isoformat()

    report_path = Path(args.out)
    summary_path = report_path.with_name(paths.ACCEPTANCE_SUMMARY.name)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(auditor, results, generated_at=generated_at), encoding="utf-8", newline="\n"
    )
    summary_path.write_text(
        json.dumps(
            render_summary(auditor, results, generated_at=generated_at), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    tally = counts(results)
    print(f"MVP acceptance audit — run {auditor.run_id} (status: {auditor.run_status})")
    for result in results:
        print(f"  {result.number:>2}. {result.status:<7} {result.criterion}")
    print("  " + " · ".join(f"{status}: {tally[status]}" for status in STATUSES))
    print(f"  report:  {_rel(report_path)}")
    print(f"  summary: {_rel(summary_path)}")
    if tally[FAIL]:
        failing = [result.number for result in results if result.status == FAIL]
        print(f"  FAILING criteria: {failing}")
    return 1 if tally[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
