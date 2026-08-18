"""Reproducibility & determinism (US-34, PRD §40, §55).

**Determinism** — running the same steps on the same data always gives exactly the same result.
This file runs the full ``--no-llm`` pipeline twice on the identical CI sample fixture, into two
separate temporary output roots (``--out-root`` / ``base_dir``, so the two runs never collide), and
proves the numeric artifacts that come out are byte-for-byte identical. Only run-scoped bookkeeping
that is *supposed* to differ between two executions — run id, timestamps, per-step wall-clock
durations — is excluded, per ``docs/interfaces.md`` §8 interface corrections on this issue.

Both runs execute in the same interpreter, so ``pipeline.config``'s ``lru_cache(1)`` loaders are
cleared between them (§2 interface corrections) — otherwise the second run would silently keep
reading the first run's already-parsed configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from flow.main import run_flow
from pipeline import config, paths
from pipeline.run_context import RunContext, close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: Artifacts that must be byte-for-byte identical between the two runs (issue §2.1).
EXACT_ARTIFACT_KEYS = (
    "clean_data",
    "features",
    "cleaning_waterfall",
    "holdout_metrics_overall",
    "holdout_metrics_by_month",
    "holdout_metrics_by_abc",
    "inventory_kpis",
    "backtest_predictions",
)

#: ``champion_decision.json`` (and its mirror at ``run_log.json["champion"]``) carries its own
#: run id and generation timestamp — excluded the same way the issue excludes them from the file.
CHAMPION_EXCLUDED_KEYS = {"run_id", "generated_at"}

#: The deterministic subtree of ``run_log.json`` (issue §8): a naive "the whole file is identical"
#: criterion cannot pass because ``steps[*].started_at`` / ``duration_s`` are wall-clock by design.
DETERMINISTIC_RUN_LOG_KEYS = ("seed", "data", "config_snapshot", "versions", "artifacts")
DETERMINISTIC_STEP_KEYS = ("name", "status", "inputs", "outputs", "row_counts", "warnings")

#: ``metrics["backtest_seconds_by_origin"]`` (src/pipeline/backtest.py) is a per-origin wall-clock
#: duration, the one non-timing exception baked into ``metrics`` — same wall-clock reasoning as
#: ``steps[*].duration_s``, just recorded under a different top-level key. Every other metric
#: (wMAPE, Bias, KPIs, …) is a computed number and must match exactly.
NON_DETERMINISTIC_METRIC_KEYS = {"backtest_seconds_by_origin"}


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """Run the deterministic ``--no-llm`` pipeline twice on the sample fixture, into two roots."""
    base1 = tmp_path_factory.mktemp("det_run1")
    base2 = tmp_path_factory.mktemp("det_run2")

    state1, ctx1 = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=base1)
    config.clear_config_cache()
    state2, ctx2 = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=base2)

    yield SimpleNamespace(
        base1=base1, base2=base2, ctx1=ctx1, ctx2=ctx2, state1=state1, state2=state2
    )
    close_log_handlers(ctx1.run_id)
    close_log_handlers(ctx2.run_id)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_path(base: Path, ctx: RunContext, key: str) -> Path:
    return base / ctx.artifacts[key]


def _without_keys(payload: dict, excluded: set[str]) -> dict:
    return {key: value for key, value in payload.items() if key not in excluded}


# --------------------------------------------------------------------------
# both runs succeed, as two distinct runs
# --------------------------------------------------------------------------
def test_both_runs_succeed(two_runs) -> None:
    assert two_runs.state1.status == "success"
    assert two_runs.state2.status == "success"
    assert two_runs.ctx1.run_id != two_runs.ctx2.run_id


def test_repo_artifacts_are_untouched_by_an_out_root_run(two_runs) -> None:
    """An ``--out-root`` run must never write into the real repository (issue §8)."""
    assert not (paths.ARTIFACTS_DIR / "_staging" / two_runs.ctx1.run_id).exists()
    assert not (paths.ARTIFACTS_DIR / "_staging" / two_runs.ctx2.run_id).exists()


# --------------------------------------------------------------------------
# byte-identical numeric artifacts
# --------------------------------------------------------------------------
@pytest.mark.parametrize("key", EXACT_ARTIFACT_KEYS)
def test_numeric_artifact_is_byte_identical(two_runs, key) -> None:
    first = _artifact_path(two_runs.base1, two_runs.ctx1, key)
    second = _artifact_path(two_runs.base2, two_runs.ctx2, key)
    assert first.is_file(), f"run 1 did not produce artifact {key!r}"
    assert second.is_file(), f"run 2 did not produce artifact {key!r}"
    assert first.read_bytes() == second.read_bytes(), f"{key} differs between two identical runs"


def test_champion_decision_is_identical_excluding_run_scoped_fields(two_runs) -> None:
    first = _read_json(_artifact_path(two_runs.base1, two_runs.ctx1, "champion_decision"))
    second = _read_json(_artifact_path(two_runs.base2, two_runs.ctx2, "champion_decision"))
    assert _without_keys(first, CHAMPION_EXCLUDED_KEYS) == _without_keys(
        second, CHAMPION_EXCLUDED_KEYS
    )


def test_inventory_plan_is_identical_excluding_run_id_column(two_runs) -> None:
    first = pd.read_csv(
        _artifact_path(two_runs.base1, two_runs.ctx1, "inventory_plan"),
        dtype=str,
        keep_default_na=False,
    ).drop(columns=["run_id"])
    second = pd.read_csv(
        _artifact_path(two_runs.base2, two_runs.ctx2, "inventory_plan"),
        dtype=str,
        keep_default_na=False,
    ).drop(columns=["run_id"])
    assert first.equals(second)


# --------------------------------------------------------------------------
# run_log.json — the deterministic subtree only
# --------------------------------------------------------------------------
def test_run_log_deterministic_subtree_is_identical(two_runs) -> None:
    first = _read_json(two_runs.base1 / "artifacts" / "run_log.json")
    second = _read_json(two_runs.base2 / "artifacts" / "run_log.json")

    for key in DETERMINISTIC_RUN_LOG_KEYS:
        assert first[key] == second[key], f"run_log.json[{key!r}] differs between two runs"

    metrics1 = _without_keys(first["metrics"], NON_DETERMINISTIC_METRIC_KEYS)
    metrics2 = _without_keys(second["metrics"], NON_DETERMINISTIC_METRIC_KEYS)
    assert metrics1 == metrics2

    champion1 = _without_keys(first["champion"] or {}, CHAMPION_EXCLUDED_KEYS)
    champion2 = _without_keys(second["champion"] or {}, CHAMPION_EXCLUDED_KEYS)
    assert champion1 == champion2

    assert [step["name"] for step in first["steps"]] == [step["name"] for step in second["steps"]]
    for step1, step2 in zip(first["steps"], second["steps"], strict=True):
        for field in DETERMINISTIC_STEP_KEYS:
            assert step1[field] == step2[field], (
                f"step {step1['name']!r} field {field!r} differs between two runs"
            )


def test_run_log_run_scoped_fields_are_allowed_to_differ(two_runs) -> None:
    """The excluded fields are genuinely run-scoped, not accidentally identical by fixture luck."""
    first = _read_json(two_runs.base1 / "artifacts" / "run_log.json")
    second = _read_json(two_runs.base2 / "artifacts" / "run_log.json")
    assert first["run_id"] != second["run_id"]
