"""Canonical filesystem locations (PRD §41).

Every module resolves artifact locations through this file so that a rename happens in exactly
one place. Paths are absolute and derived from the repository root; nothing here touches disk.
"""

from __future__ import annotations

from pathlib import Path

# src/pipeline/paths.py -> src/pipeline -> src -> repository root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# --- top-level directories -------------------------------------------------
CONFIG_DIR: Path = PROJECT_ROOT / "config"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS_DIR / "models"
FORECASTS_DIR: Path = ARTIFACTS_DIR / "forecasts"
REPORTS_DIR: Path = ARTIFACTS_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
EDA_TABLES_DIR: Path = REPORTS_DIR / "eda_tables"
EVAL_TABLES_DIR: Path = REPORTS_DIR / "evaluation_tables"
CONTRACTS_DIR: Path = ARTIFACTS_DIR / "contracts"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"

# --- configuration files ---------------------------------------------------
CLEANING_CONFIG: Path = CONFIG_DIR / "cleaning_config.yaml"
MODEL_CONFIG: Path = CONFIG_DIR / "model_config.yaml"
INVENTORY_POLICY: Path = CONFIG_DIR / "inventory_policy.yaml"
NON_INVENTORY_STOCKCODES: Path = CONFIG_DIR / "non_inventory_stockcodes.csv"
DATA_SOURCES: Path = CONFIG_DIR / "data_sources.yaml"

# --- processed data --------------------------------------------------------
CLEAN_TRANSACTIONS: Path = PROCESSED_DIR / "clean_transactions.parquet"
CLEAN_DATA: Path = PROCESSED_DIR / "clean_data.csv"          # required artifact
FEATURES: Path = PROCESSED_DIR / "features.csv"              # required artifact

# --- models ----------------------------------------------------------------
MODEL: Path = MODELS_DIR / "model.joblib"                    # required artifact (champion)
MODEL_META: Path = MODELS_DIR / "model_meta.json"            # provenance of the refit champion


def candidate_model(model_id: str) -> Path:
    """Path of a candidate model file, e.g. ``candidate_model("M2_gbm_poisson")``."""
    return MODELS_DIR / f"{model_id}.joblib"


# --- forecasts & inventory -------------------------------------------------
BACKTEST_PREDICTIONS: Path = FORECASTS_DIR / "backtest_predictions.csv"
LATEST_FORECAST: Path = FORECASTS_DIR / "latest_forecast.csv"
INVENTORY_PLAN: Path = FORECASTS_DIR / "inventory_plan.csv"
SIGMA_TABLE: Path = FORECASTS_DIR / "sigma_table.csv"
INVENTORY_KPIS: Path = FORECASTS_DIR / "inventory_kpis.csv"
HOLDOUT_SIMULATION_ROWS: Path = FORECASTS_DIR / "holdout_simulation_rows.csv"

# --- reports ---------------------------------------------------------------
EDA_REPORT: Path = REPORTS_DIR / "eda_report.html"            # required artifact
INSIGHTS: Path = REPORTS_DIR / "insights.md"                  # required artifact
EVALUATION_REPORT: Path = REPORTS_DIR / "evaluation_report.md"  # required artifact
MODEL_CARD: Path = REPORTS_DIR / "model_card.md"              # required artifact
CHAMPION_DECISION: Path = REPORTS_DIR / "champion_decision.json"
DATA_QUALITY_FINDINGS: Path = REPORTS_DIR / "data_quality_findings.json"
FEATURE_VALIDATION: Path = REPORTS_DIR / "feature_validation.json"
HOLDOUT_ROWS_ALL_MODELS: Path = EVAL_TABLES_DIR / "holdout_rows_all_models.csv"
EXCESS_CONCENTRATION: Path = EVAL_TABLES_DIR / "excess_concentration.csv"

# --- contracts & run bookkeeping -------------------------------------------
DATASET_CONTRACT: Path = CONTRACTS_DIR / "dataset_contract.json"  # required artifact
VALIDATION_REPORT: Path = ARTIFACTS_DIR / "validation_report.json"
RUN_LOG: Path = ARTIFACTS_DIR / "run_log.json"

# The eight artifacts required by the course brief, under their exact names (PRD §41).
REQUIRED_ARTIFACTS: tuple[Path, ...] = (
    CLEAN_DATA,
    FEATURES,
    MODEL,
    EDA_REPORT,
    INSIGHTS,
    DATASET_CONTRACT,
    EVALUATION_REPORT,
    MODEL_CARD,
)
