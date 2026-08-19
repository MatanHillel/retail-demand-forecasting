# Retail Demand Forecasting & Inventory Planning System

[![ci](https://github.com/danielbfogel-lgtm/retail-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/danielbfogel-lgtm/retail-demand-forecasting/actions/workflows/ci.yml)

A Streamlit application that analyses historical sales from the UCI *Online Retail II* dataset
(Chen, 2019 — CC BY 4.0), uses **one global machine-learning model** to predict how many units each
active product will sell next month, converts that forecast into a **Recommended Target Inventory**
using safety stock derived from out-of-sample forecast errors, and lets users evaluate product
forecasts, model performance and the trade-off between stockouts and excess inventory. Forecasting
(machine learning) and inventory policy (a deterministic rule) are two separate layers — the model
never predicts inventory. See PRD §51.

The specification of record is [`docs/PRD.md`](docs/PRD.md) (PRD v1.3). Working conventions for
contributors and for Claude Code are in [`CLAUDE.md`](CLAUDE.md).

> **Status:** bootstrap (US-00). The pipeline, crews and app are added by later issues; the full
> README is written in US-36.

## Setup

Python **3.11** only (PRD §43).

```bash
# 1. create a virtual environment — a private folder holding this project's exact package versions
python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell:  .\.venv\Scripts\Activate.ps1

# 2. install pinned dependencies and this project in editable mode
pip install -r requirements.txt
pip install -e .

# 3. run the checks
pytest -q
ruff check src tests
```

`pip install -e .` puts `src/` on the import path, so the packages are imported by their top-level
names: `from pipeline.config import ...`, `from flow.main import ...`, `from app.data_access import
...`.

`make install` / `make test` / `make lint` wrap the same three commands. Without GNU make (e.g.
Windows), use `scripts/install.sh`, `scripts/test.sh`, `scripts/lint.sh`.

`uv` is a supported alternative for creating the environment, and it can fetch CPython 3.11 itself:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt pip
.venv/Scripts/python.exe -m pip install -e .    # Linux/macOS: .venv/bin/python -m pip install -e .
```

## Repository layout (PRD §41)

```
data/raw/          raw dataset — downloaded by script, NEVER committed
data/processed/    clean_data.csv, features.csv, clean_transactions.parquet
artifacts/         models, forecasts, reports, contracts, run logs
config/            all thresholds, dates, seeds and gates (never hard-coded in code)
src/pipeline/      deterministic data & modelling pipeline
src/crews/         CrewAI agent crews
src/flow/          CrewAI Flow orchestration
src/app/           Streamlit application
tests/  docs/  scripts/  logs/
```

## Data

The raw dataset is **not** part of the repository. A download script fetches it and verifies a
SHA-256 hash (US-03). Source: UCI Machine Learning Repository, *Online Retail II* (Chen, 2019),
CC BY 4.0; Kaggle mirror `mashlyn/online-retail-ii-uci`.

## Contributing

Branch as `feature/US-NN-short-name`, open a pull request against protected `main` using the
template, and get at least one review with all four CI checks green (`lint-test`,
`pipeline-no-llm`, `failure-path`, `determinism`). Run `make ci-local` first to reproduce them
on your own machine. See [`docs/contributing.md`](docs/contributing.md) and
[`docs/branch_protection.md`](docs/branch_protection.md).
