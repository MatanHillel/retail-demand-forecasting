"""Run-metadata completeness (US-34, PRD §40, §55).

Every result shown in the report, the app or the presentation must be traceable back to the exact
run that produced it. This file runs the full ``--no-llm --sample`` pipeline once and checks that
``artifacts/run_log.json`` — the one file the whole project treats as the "receipt" for a run —
carries everything PRD §40 promises: who ran it, on what data, with what configuration and seed,
against which library versions, with what result, and a **checksum** (a fingerprint of a file, used
to prove two copies are exactly the same) for every artifact it produced.
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import pytest

from flow.main import run_flow
from pipeline import paths
from pipeline.config import load_model_config
from pipeline.download import compute_sha256
from pipeline.run_context import close_log_handlers

RAW_SAMPLE = paths.FIXTURES_DIR / "raw_sample.csv"

#: run_log.json["versions"] key -> installed-distribution name (issue §8: not the same string —
#: "python" is not a package at all, and "sklearn" maps to the "scikit-learn" distribution).
VERSION_KEY_TO_DISTRIBUTION = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "crewai": "crewai",
    "streamlit": "streamlit",
}

CONFIG_SNAPSHOT_KEYS = {
    "cleaning_config",
    "model_config",
    "inventory_policy",
    "data_sources",
    "non_inventory_stockcodes",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    base = tmp_path_factory.mktemp("run_metadata")
    state, ctx = run_flow(mode="no-llm", raw_path=RAW_SAMPLE, skip_tuning=True, base_dir=base)
    yield SimpleNamespace(base=base, state=state, ctx=ctx)
    close_log_handlers(ctx.run_id)


@pytest.fixture(scope="module")
def run_log(run) -> dict:
    path = run.base / "artifacts" / "run_log.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def requirements_pins() -> dict[str, str]:
    """``{distribution_name: pinned_version}`` parsed from ``requirements.txt``."""
    text = (paths.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)$", line)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


# --------------------------------------------------------------------------
# top-level identity: run id, status, mode, seed
# --------------------------------------------------------------------------
def test_run_succeeded(run) -> None:
    assert run.state.status == "success"


def test_run_log_identity_fields(run_log, run) -> None:
    assert run_log["run_id"] == run.ctx.run_id
    assert run_log["started_at"]
    assert run_log["finished_at"]
    assert run_log["status"] == "success"
    assert run_log["mode"] == "no-llm"


def test_seed_matches_model_config(run_log) -> None:
    assert run_log["seed"] == load_model_config().seed


# --------------------------------------------------------------------------
# data identity: sha256, rows
# --------------------------------------------------------------------------
def test_data_sha256_matches_the_raw_input_file(run_log) -> None:
    assert run_log["data"]["sha256"] == compute_sha256(RAW_SAMPLE)


def test_data_rows_is_recorded(run_log) -> None:
    assert isinstance(run_log["data"]["rows"], int)
    assert run_log["data"]["rows"] > 0


# --------------------------------------------------------------------------
# config snapshot: exactly the five keys (issue §8 — not four files plus a hash)
# --------------------------------------------------------------------------
def test_config_snapshot_has_exactly_five_keys(run_log) -> None:
    assert set(run_log["config_snapshot"]) == CONFIG_SNAPSHOT_KEYS


def test_config_snapshot_model_config_matches_the_seed(run_log) -> None:
    assert run_log["config_snapshot"]["model_config"]["seed"] == load_model_config().seed


# --------------------------------------------------------------------------
# versions: match installed packages and requirements.txt pins
# --------------------------------------------------------------------------
def test_python_version_is_recorded_and_is_3_11(run_log) -> None:
    assert run_log["versions"]["python"].startswith("3.11")
    assert sys.version.startswith("3.11")


@pytest.mark.parametrize("key,distribution", sorted(VERSION_KEY_TO_DISTRIBUTION.items()))
def test_version_matches_installed_package(run_log, key, distribution) -> None:
    recorded = run_log["versions"][key]
    assert recorded is not None, f"{key} ({distribution}) was not recorded — is it installed?"
    assert recorded == importlib_metadata.version(distribution)


@pytest.mark.parametrize("key,distribution", sorted(VERSION_KEY_TO_DISTRIBUTION.items()))
def test_version_matches_requirements_txt_pin(
    run_log, requirements_pins, key, distribution
) -> None:
    pinned = requirements_pins.get(distribution.lower())
    assert pinned is not None, f"{distribution} has no == pin in requirements.txt"
    assert run_log["versions"][key] == pinned


# --------------------------------------------------------------------------
# metrics, champion, steps
# --------------------------------------------------------------------------
def test_metrics_reports_wmape_and_bias_together(run_log) -> None:
    """PRD §23: wMAPE and Bias are always reported together, never one without the other."""
    holdout = run_log["metrics"].get("holdout")
    assert holdout, "no holdout metrics recorded in run_log.json"
    payload = json.dumps(holdout).lower()
    assert "wmape" in payload
    assert "bias" in payload


def test_champion_is_recorded(run_log) -> None:
    champion = run_log["champion"]
    assert champion is not None
    assert champion["champion"] in {
        model_id for model_id in load_model_config().models
    }


def test_steps_carry_durations_and_row_counts(run_log) -> None:
    steps = run_log["steps"]
    assert steps, "no steps recorded"
    for step in steps:
        assert step["status"] == "success"
        assert isinstance(step["duration_s"], float)
        assert step["duration_s"] >= 0

    cleaning_steps = [s for s in steps if s["name"] in {"clean_transactions", "build_panel"}]
    assert cleaning_steps, "no cleaning step recorded"
    assert any(step["row_counts"] for step in cleaning_steps), (
        "no cleaning step recorded a row-count waterfall"
    )


# --------------------------------------------------------------------------
# artifact checksums (issue §8: a new field, not a retyped `artifacts`)
# --------------------------------------------------------------------------
def test_artifacts_is_a_path_map_and_artifact_checksums_is_separate(run_log) -> None:
    assert run_log["artifacts"], "no artifacts recorded"
    for value in run_log["artifacts"].values():
        assert isinstance(value, str), "artifacts values must stay a plain path map (§8)"
    assert set(run_log["artifact_checksums"]) == set(run_log["artifacts"])


def test_required_artifacts_have_a_correct_checksum(run_log, run) -> None:
    required_by_key = {
        "clean_data": paths.CLEAN_DATA,
        "features": paths.FEATURES,
        "model": paths.MODEL,
        "eda_report": paths.EDA_REPORT,
        "insights": paths.INSIGHTS,
        "dataset_contract": paths.DATASET_CONTRACT,
        "evaluation_report": paths.EVALUATION_REPORT,
        "model_card": paths.MODEL_CARD,
    }
    for key, canonical in required_by_key.items():
        entry = run_log["artifact_checksums"].get(key)
        assert entry is not None, f"no checksum recorded for required artifact {key!r}"
        relative = canonical.relative_to(paths.PROJECT_ROOT)
        resolved = run.base / relative
        assert entry["path"] == relative.as_posix()
        assert entry["bytes"] == resolved.stat().st_size
        assert entry["sha256"] == compute_sha256(resolved)
