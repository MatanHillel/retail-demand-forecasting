"""The only module in ``src/app`` that reads files (US-27, PRD §33).

Every screen goes through the loaders here so that no two screens can ever disagree about a
number because one of them opened a file differently. Each public loader checks the artifact
exists (raising :class:`ArtifactMissing` if not — the one exception is
:func:`load_validation_report`, whose absence is a normal state) and then delegates to a private,
``st.cache_data``-backed reader keyed on the file's path and modification time, so a new pipeline
run is picked up automatically without a manual cache-clear and without caches leaking across the
different temporary artifact directories used in tests.

Path constants are looked up as ``paths.X`` (module attribute access) rather than imported by name,
so that tests can ``monkeypatch.setattr(paths, "RUN_LOG", ...)`` and have it take effect here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import paths


class ArtifactMissing(Exception):
    """Raised when a required pipeline artifact is absent from disk."""

    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        super().__init__(f"required artifact missing: {name} ({path})")


def _require(path: Path, name: str) -> Path:
    if not path.exists():
        raise ArtifactMissing(name, path)
    return path


def _key(path: Path) -> tuple[str, int]:
    return (str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def _read_json(path_str: str, mtime_ns: int) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def _read_text(path_str: str, mtime_ns: int) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def load_run_log() -> dict:
    return _read_json(*_key(_require(paths.RUN_LOG, "run_log.json")))


def load_validation_report() -> dict | None:
    """Return the parsed report, or ``None`` when absent — that is a normal state (§8)."""
    if not paths.VALIDATION_REPORT.exists():
        return None
    return _read_json(*_key(paths.VALIDATION_REPORT))


def load_inventory_plan() -> pd.DataFrame:
    return _read_csv(*_key(_require(paths.INVENTORY_PLAN, "inventory_plan.csv")))


def load_latest_forecast() -> pd.DataFrame:
    return _read_csv(*_key(_require(paths.LATEST_FORECAST, "latest_forecast.csv")))


def load_champion_decision() -> dict:
    return _read_json(*_key(_require(paths.CHAMPION_DECISION, "champion_decision.json")))


def load_eval_table(name: str) -> pd.DataFrame:
    path = paths.EVAL_TABLES_DIR / f"{name}.csv"
    return _read_csv(*_key(_require(path, f"evaluation_tables/{name}.csv")))


def load_inventory_kpis() -> pd.DataFrame:
    return _read_csv(*_key(_require(paths.INVENTORY_KPIS, "inventory_kpis.csv")))


def load_simulation_rows() -> pd.DataFrame:
    return _read_csv(
        *_key(_require(paths.HOLDOUT_SIMULATION_ROWS, "holdout_simulation_rows.csv"))
    )


def load_panel() -> pd.DataFrame:
    return _read_csv(*_key(_require(paths.CLEAN_DATA, "clean_data.csv")))


def load_backtest(model: str | None = None) -> pd.DataFrame:
    df = _read_csv(*_key(_require(paths.BACKTEST_PREDICTIONS, "backtest_predictions.csv")))
    return df if model is None else df.loc[df["model"] == model].reset_index(drop=True)


def load_eda_table(name: str) -> pd.DataFrame:
    path = paths.EDA_TABLES_DIR / f"{name}.csv"
    return _read_csv(*_key(_require(path, f"eda_tables/{name}.csv")))


def figure_path(name: str) -> Path:
    path = paths.FIGURES_DIR / f"{name}.png"
    return _require(path, f"figures/{name}.png")


def load_text(path: Path) -> str:
    target = _require(Path(path), Path(path).name)
    return _read_text(*_key(target))


def load_contract() -> dict:
    return _read_json(*_key(_require(paths.DATASET_CONTRACT, "dataset_contract.json")))


def load_dq_findings() -> dict:
    return _read_json(*_key(_require(paths.DATA_QUALITY_FINDINGS, "data_quality_findings.json")))
