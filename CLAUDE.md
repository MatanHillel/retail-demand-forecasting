# CLAUDE.md

Guidance for Claude Code when working in this repository.

## 1. What this project is

A **Retail Demand Forecasting & Inventory Planning System** built on the UCI *Online Retail II*
dataset (Chen, 2019 — CC BY 4.0; Kaggle mirror `mashlyn/online-retail-ii-uci`).

> A Streamlit application that analyses historical sales, uses **one global machine-learning model**
> to predict how many units each active product will sell next month, converts the forecast into a
> **Recommended Target Inventory** using safety stock derived from out-of-sample forecast errors, and
> lets users evaluate product forecasts, model performance and the trade-off between stockouts and
> excess inventory.

Two separate layers — never conflate them:

1. **Forecasting layer (ML)** — predicts *demand* (SKU × month → units sold).
2. **Inventory-policy layer (deterministic rule)** — converts the forecast into target inventory.
   The model never predicts inventory.

The specification of record is **`docs/PRD.md` (PRD v1.3)**. Every section reference below (`§10`,
`§20`, `§35A`, `Appendix A`) points there. **If this file and the PRD disagree, the PRD wins** —
and fix this file in the same PR.

## 2. Non-negotiable rules

These are the invariants the whole project is graded on. Violating one is a bug even if the tests
pass. Read this list before writing any code.

1. **No leakage.** No feature may use information from the target month `t` or later. Forecast
   origin is `t−1`; allowed information is everything with `month ≤ t−1`. Calendar attributes of
   `t` (`target_month_of_year`, `target_quarter`) are the only exception — they are known in
   advance. (§16, §17)
2. **`train_test_split(shuffle=True)` is forbidden.** The split is temporal (§21). A test enforces
   that `sklearn.model_selection.train_test_split` is never imported under `src/`.
3. **LLM agents never compute numbers.** No wMAPE, Bias, aggregation, MAD, safety stock, prediction
   or split is ever produced by a model. Agents call approved deterministic tools, interpret their
   output and write prose. Every LLM-written narrative passes the `numbers_in_tables` guard before
   it is accepted; on failure the deterministic version is restored. (§38)
4. **No number is hard-coded.** Thresholds, dates, `k`, `z`, gates, ABC cut-offs, seeds — all live in
   `config/*.yaml`. Indicative numbers from the PRD (row counts, wMAPE ≈ 55 %) are *expectations to
   log and compare against*, never literals in code and never asserted in tests. (§14 convention, §40)
5. **December 2011 is never scored.** It is a partial month (through 9 Dec). It may be shown as
   "partial actual" (hatched, labelled), and it is the target of the latest operational forecast —
   but it never enters a metric. Last full month is 2011-11. (§8, §16, §21)
6. **Gross demand.** The target is units sold on positive sales lines. Cancellations (`C` invoices)
   and adjustments are **not** subtracted — inventory was needed to fulfil the original order.
   `returned_units` is EDA-only and is **never** a feature. Rows without `Customer ID` are **kept**.
   (§9, §13.2)
7. **wMAPE and Bias are always reported together.** Never one without the other. (§23)
8. **Robust σ only.** `Robust σ = 1.4826 × MAD` of **out-of-sample** residuals from origins strictly
   before the evaluation month. Never `np.std`, never in-sample residuals. (§22, §26)
9. **ABC from the training window only** for anything touching modelling, evaluation or σ fallback.
   Full-period ABC exists for descriptive EDA (E6) and must be labelled as such. (§18.2, §23, §27)
10. **The output is "Recommended Target Inventory", never "Order Quantity".** The dataset has no
    on-hand / on-order data, so an order quantity cannot be computed. (§7)
11. **`z = 1.645` does not guarantee a 95 % fill rate.** The achieved fill rate is measured
    empirically in the back-test. Any UI or document text implying otherwise is a defect. (§25)
12. **Post-hoc bias correction is out of scope.** It proved unstable and is explicitly excluded from
    the MVP. (§20)
13. **Failed runs never overwrite good results.** Everything is written to a staging directory and
    promoted only on success. A failed run writes `validation_report.json`, sets
    `run_log.json → status: failed`, exits non-zero and leaves the previous forecasts untouched. (§39)
14. **Exact artifact names.** Eight artifacts are required by the course brief and must exist with
    exactly these names (§41):
    `clean_data.csv`, `features.csv`, `model.joblib`, `eda_report.html`, `insights.md`,
    `dataset_contract.json`, `evaluation_report.md`, `model_card.md`.
15. **Raw data is never committed.** `data/raw/*` is git-ignored; a script downloads it and verifies
    a SHA-256 hash. `clean_data.csv` and `features.csv` *are* committed. (§42)
16. **Determinism.** Single global seed (`model_config.yaml → seed`, 42). Two runs on the same data
    must produce byte-identical `clean_data.csv`, `features.csv` and metrics. (§40)

## 3. Repository layout

```
data/
    raw/                      # NOT committed (download script + hash check)
    processed/
        clean_transactions.parquet    # audit / EDA on invoices & customers
        clean_data.csv                # REQUIRED — hand-off panel, StockCode × Month
        features.csv                  # REQUIRED — model input
artifacts/
    models/model.joblib               # REQUIRED — champion (refit through 2011-11)
    models/<model_id>.joblib          # candidates
    forecasts/backtest_predictions.csv, latest_forecast.csv, inventory_plan.csv,
              sigma_table.csv, inventory_kpis.csv, holdout_simulation_rows.csv
    reports/eda_report.html, insights.md, evaluation_report.md, model_card.md   # REQUIRED
    reports/figures/, eda_tables/, evaluation_tables/, champion_decision.json,
            data_quality_findings.json, feature_validation.json
    contracts/dataset_contract.json   # REQUIRED
    validation_report.json, run_log.json
config/
    cleaning_config.yaml, model_config.yaml, inventory_policy.yaml,
    non_inventory_stockcodes.csv, data_sources.yaml
src/pipeline/   src/crews/   src/flow/   src/app/
tests/   logs/   docs/   scripts/
```

Import style is src-layout with an editable install: `from pipeline.config import ...`,
`from flow.main import ...`, `from app.data_access import ...`. Never rename these paths — later
issues reference them literally.

## 4. Commands

```bash
# setup (Python 3.11 only)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# data
python -m pipeline.download --record-hash     # download + hash
python -m pipeline.download --make-sample     # tests/fixtures/raw_sample.csv for CI

# pipeline
python -m pipeline --no-llm                   # full deterministic run (CI mode)
python -m pipeline --no-llm --sample --skip-tuning   # fast run on the fixture
python -m pipeline                            # LLM mode (crews run; needs an API key)

# app
streamlit run src/app/Home.py

# quality
pytest -q
pytest -q -m "not slow"
ruff check src tests
python scripts/mvp_acceptance_check.py        # §49 checklist -> acceptance_report.md
```

Exit codes: `0` success, `2` validation failure (graceful stop), `1` unexpected exception.

## 5. Architecture

### CrewAI Flow (§37) — ten steps

```
Raw data → 1 Dataset intake → 2 Data Analyst Crew work → 3 Contract validation
→ 4 Data Scientist Crew work → 5 Feature validation → 6 Temporal training & back-testing
→ 7 Model evaluation & champion selection → 8 Inventory policy calibration
→ 9 Artifact validation → 10 Publish to Streamlit
```

- State is a Pydantic `FlowState` (run id, timestamp, data hash, artifact paths, validation flags,
  metrics, champion, errors).
- A `@router` after steps 1, 3, 5 and 9 returns `continue` or `fail`.
- **The Flow always executes the deterministic tools itself.** In LLM mode the crews are kicked off
  *around* those steps to review results and write narrative — so `--no-llm` and LLM runs produce
  numerically identical artifacts. Crews never run after a failure.

### Crews (≥ 3 agents each — course requirement)

| Crew | Agents |
|---|---|
| Data Analyst (§35) | Data Quality Analyst · Data Preparation Analyst · Business & EDA Analyst |
| Data Scientist (§36) | Feature Engineering Specialist · Forecasting Model Scientist · Model Evaluation & Inventory Scientist |

### Canonical model ids — use these strings everywhere

`B1_last_month` · `B2_ma3` (**main baseline**) · `B3_seasonal_naive` (reference only) ·
`M1_linear` · `M2_gbm_poisson` (**primary ML**) · `M3_gbm_squared` · `M4_gbm_absolute` (challenger)

### Champion gates (§20) — executed by code, never by hand

1. **Bias gate** — `|Bias| ≤ 0.10` on the hold-out; months with `|Bias| > 0.25` are reported but do
   not fail the gate.
2. **Accuracy** — lowest wMAPE among gate-1 passers.
3. **Inventory tie-break** — within 1 wMAPE point, prefer less excess at similar fill rate.
4. **Meaningful improvement** — an ML model "improves on the baseline" only if it beats the best
   gate-passing baseline by **≥ 2 wMAPE points**.

Gates apply to **every** candidate including baselines. A baseline winning, or
"simple methods are competitive", is a **legitimate, expected outcome** — report it plainly. Never
tune the pipeline to make ML look better. There is no `--force-champion` flag and there must never be.

### Key formulas — copy verbatim

```text
wMAPE           = Σ|Actual − Forecast| / Σ Actual
Bias            = Σ(Forecast − Actual) / Σ Actual
residual e      = Actual − Forecast
Robust σ        = 1.4826 × median(|e − median(e)|)          # ≥ 6 out-of-sample residuals per product
Safety Stock    = z × σ                                      # z = 1.645 default
Target Inventory= ceil( max(0, Forecast + z × σ) )           # 820, σ 70, z 1.645 → 936
Fill Rate       = Σ Fulfilled / Σ Actual
```

σ fallback hierarchy: **product** (≥ 6 residuals) → **ABC group** (training-window ABC) →
**global**. The level used is stored per product as `sigma_source` and shown in the app.

Active product (§14): at least one positive sale in the `k = 6` months before the target month.

Temporal split (§21): train targets **2010-03 … 2011-05**; internal validation folds
**2011-01 … 2011-05**; hold-out **2011-06 … 2011-11**; back-test origins **2010-05 … 2011-10**.

## 6. Configuration

Never write a constant into code — read it from config.

| File | Owns |
|---|---|
| `cleaning_config.yaml` | cancellation/adjustment prefixes, price & quantity rules, duplicate policy, warning thresholds, month boundaries |
| `model_config.yaml` | seed, `k`, feature list, split & back-test dates, model ids and hyper-parameters, tuning grid, champion gates |
| `inventory_policy.yaml` | lead time, `z` and `z_options`, MAD scale, min residuals for σ, ABC thresholds |
| `non_inventory_stockcodes.csv` | excluded codes with `reason` and `status` |
| `data_sources.yaml` | dataset URLs, expected SHA-256, citation |

`config_snapshot()` is recorded in every `run_log.json`.

## 7. Working on an issue

Work is tracked in **Linear**, project `AI_DEV_final_poject` (team `AI_DEV_final_project`), issues
`AI-5` … `AI-44`, titled `US-00` … `US-39` in execution order with `blocked by` relations and
category labels (`data-pipeline`, `eda`, `modeling`, `evaluation`, `inventory-policy`, `crewai`,
`orchestration`, `streamlit-ui`, `testing`, `ci-cd`, `docs`, `deliverable`) plus milestone labels
`M1`–`M6`.

Each issue body is the source of truth for that unit of work and contains: the user story, an
**AI Prompt** (exact files, paths, formulas, PRD sections), **Technical notes**, **Notes**,
a **Report for review** instruction, **Acceptance criteria**, and **Claude Code execution settings**
(model / reasoning effort / plan mode). Follow the issue's own execution settings.

**Definition of done for any issue:**

1. Every acceptance-criteria checkbox in the issue verifiably passes — run them, do not assume.
2. The named tests exist and pass; `ruff check src tests` is clean.
3. No PRD invariant from §2 above is broken (grep for the forbidden patterns listed in the issue).
4. The **Report for review** is written: a step-by-step, plain-language explanation for a
   non-programmer, defining every technical term on first use, plus a one-line description of every
   file created or changed. This is a deliverable, not a formality.
5. A PR is opened against protected `main` with the template filled in (What / Why / How tested) and
   the CI checks green.

Branch naming: `feature/US-NN-short-name`. Every change merges through a PR with ≥ 1 reviewer.
Each milestone ends with a merged PR and a green CI run (§54).

**Ownership** — Daniel: platform/config/logging (US-00–02), Streamlit (US-27–30), CI, docs,
determinism, presentation, demo. מתן הלל: raw data → cleaning → panel → data quality → EDA →
contract → Data Analyst Crew (US-03–12), then the Flow (US-31–33). Dor Hll: features → leakage →
metrics → models → back-test → evaluation → σ → inventory → champion → reports → Data Scientist Crew
(US-13–26). Check the Linear assignee before starting something outside your slice.

## 8. Tests (§55)

| File | Proves |
|---|---|
| `test_cleaning.py` | waterfall counts, exclusion list, duplicate policy |
| `test_panel.py` | zero-fill, no rows before first sale, partial flag |
| `test_contract.py` | validation passes, and fails on corrupted samples |
| `test_features.py` | lag correctness, active rule, NaN policy |
| `test_leakage.py` | permutation of future months leaves features unchanged |
| `test_metrics.py` | wMAPE / Bias on known arrays |
| `test_inventory.py` | σ, safety stock, target, simulation |
| `test_flow.py` | graceful failure path produces `validation_report.json` |
| `test_determinism.py` | two runs identical |

Add tests in the file the PRD names — do not invent a parallel structure.

## 9. Style and conventions

- Python 3.11, pandas / numpy / scikit-learn / CrewAI / Streamlit / Matplotlib+Seaborn, Joblib,
  Pydantic v2, pytest. Versions pinned in `requirements.txt`.
- Charts: **Matplotlib / Seaborn only** (course requirement). One consistent colour-blind-safe
  palette; ABC classes always the same three colours; every figure has a title, axis labels *with
  units*, a source footnote, and one message. Log scale must be stated in the axis label. Figures
  are ≥ 150 dpi PNGs under `artifacts/reports/figures/` with stable `E<nn>_<name>.png` names.
- `eda_report.html` is a **single self-contained file** — PNGs embedded as base64, no external
  or CDN references, must render offline.
- Numbers in `insights.md` (8–12 insights) must all exist in a computed table under
  `eda_tables/`. The `numbers_in_tables` guard enforces this.
- Months are always `YYYY-MM` strings; timestamps are ISO-8601 UTC.
- Deterministic file writes: fixed sort order, fixed `float_format`, no index column.
- Never log or commit an API key. `--no-llm` mode must import no LLM class at all.

## 10. Common pitfalls

- Computing a rolling window without excluding month `t` — the single most likely way to break the
  project. Recompute a lag by hand from `clean_data.csv` when in doubt.
- Using full-period ABC where training-window ABC is required.
- Using `np.std` instead of the MAD-based robust σ; a few 80,000-unit wholesale lines will destroy it.
- Scoring December 2011, or letting it into a feature window.
- Subtracting returns from the target.
- Letting an agent state a number that is not in a table.
- Writing an artifact directly instead of through the staging path, so a failed run corrupts good output.
- Silently dropping rows so two models are compared on different row sets — all policies and
  candidates are scored on identical rows.

## 11. Glossary

**Forecast origin** — last month whose data may be used. **Active product** — ≥ 1 sale in the
previous `k` months. **wMAPE** — Σ|error| / Σ actual. **Bias** — Σ(forecast − actual) / Σ actual.
**Robust σ** — 1.4826 × MAD of out-of-sample residuals. **Fill rate** — fulfilled / actual demand.
**Champion** — the model selected by the §20 gates. **Left-censoring** — products already selling in
Dec 2009 may be older, so `product_age_months` is a lower bound.
