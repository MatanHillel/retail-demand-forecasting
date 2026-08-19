# Retail Demand Forecasting & Inventory Planning System

[![ci](https://github.com/danielbfogel-lgtm/retail-demand-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/danielbfogel-lgtm/retail-demand-forecasting/actions/workflows/ci.yml)

> A Streamlit application that analyses historical sales, uses **one global machine-learning model**
> to predict how many units each active product will sell next month, converts the forecast into a
> **Recommended Target Inventory** using safety stock derived from out-of-sample forecast errors,
> and lets users evaluate product forecasts, model performance and the trade-off between stockouts
> and excess inventory (PRD §51).

Built on the UCI *Online Retail II* dataset — Chen (2019), **CC BY 4.0**; see
[`DATA_LICENSE.md`](DATA_LICENSE.md) for the full attribution and [`CITATION.cff`](CITATION.cff) for
the machine-readable citation.

The specification of record is [`docs/PRD.md`](docs/PRD.md) (PRD v1.3) — every `§NN` reference below
points there. Working conventions for contributors and for Claude Code are in
[`CLAUDE.md`](CLAUDE.md). The full documentation set is indexed in [`docs/README.md`](docs/README.md).

---

## 1. What this product does — and does not do

**Two separate layers, never conflated (PRD §7, §52):**

1. **Forecasting (machine learning)** — predicts *demand*: SKU × month → units sold.
2. **Inventory policy (a deterministic rule)** — converts that forecast into a target inventory
   level using robust safety stock. The model never predicts inventory.

> *Illustrative example (PRD §4):* Product X is expected to sell **820** units next month. Its
> forecast uncertainty requires **116** units of safety stock. The recommended target inventory for
> the period is **936** units. These three numbers are an example of the shape of the output, not a
> claim about any specific run — see [Results snapshot](#7-results-snapshot) for real numbers.

The dataset carries no on-hand / on-order inventory or purchase-order data, so the system cannot
compute `Order Quantity = Target Inventory − On Hand − On Order + Backorders`. Its output is
therefore always **"Recommended Target Inventory"**, never *"Order Quantity"* (PRD §7).

**Out of scope for this MVP** (PRD §6.2): forecasts for products with no sales history (cold start);
forecasting by country, store or warehouse; supplier management and real lead times; current
on-hand / on-order inventory and purchase orders; price optimisation; true reorder-quantity
calculation; direct 2- and 3-month-ahead forecasts; full-quarter forecasts at the start of the
quarter; forecasting returns or cancellations; customer-level analytics (RFM, churn) beyond
descriptive EDA.

---

## 2. Quick start

**Prerequisites:** Python **3.11** only — `pyproject.toml` pins `>=3.11,<3.12` (PRD §43); CI runs
3.11. A venv built with 3.12 or 3.14 fails `pip install -e .` with `requires a different Python`.
Never widen the pin to make an install succeed.

```powershell
# Windows (primary dev environment) — PowerShell 5.1 has no `&&`, run one line at a time
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
python --version                 # must print 3.11.x
uv pip install -r requirements.txt
uv pip install -e .
```

```bash
# Linux / macOS / CI
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

`make install` wraps the same three commands; without GNU make (Windows), use `scripts/install.sh`.
`pip install -e .` puts `src/` on the import path, so packages import by their top-level names:
`from pipeline.config import ...`, `from flow.main import ...`, `from app.data_access import ...`.

Call the interpreter by path rather than relying on an activated environment — each shell
invocation is independent:

```bash
.venv\Scripts\python.exe -m pytest -q          # Windows
.venv/bin/python -m pytest -q                  # Linux / macOS / CI
```

**Get the data, run the pipeline, run the app:**

```bash
# 1. download the raw dataset and record its SHA-256 hash (data/raw/ is never committed)
python -m pipeline.download --record-hash

# 2. run the full pipeline deterministically (no LLM class is imported in this mode)
python -m pipeline --no-llm

#    useful variants:
python -m pipeline --no-llm --sample --skip-tuning   # fast run on the committed CI fixture
python -m pipeline --no-llm --skip-tuning            # skip the hyper-parameter grid search

# 3. run the app
python -m streamlit run src/app/Home.py

# 4. run the tests
pytest -q
```

**LLM mode** (optional — the pipeline runs fully without it):

```bash
cp .env.example .env               # fill in OPENAI_API_KEY or ANTHROPIC_API_KEY
python -m pipeline                 # runs the same ten steps + two crew narrative kickoffs
```

`.env` is read at the process entry point by `python-dotenv`, and **an environment variable that
is already set always wins over the file** (`load_dotenv(override=False)`). This matters more than
it looks: a stale `OPENAI_API_KEY` exported in your shell, your Windows user profile, or the
launcher that started your editor silently takes precedence, and `.env` is then never consulted at
all. The symptom is indistinguishable from a bad key in `.env` — the provider simply answers
`401 Unauthorized`. If you get a 401 while the key in `.env` is known-good, check for a leftover
variable first:

```bash
python -c "import os; print('set in environment:', 'OPENAI_API_KEY' in os.environ)"
# PowerShell: Remove-Item Env:OPENAI_API_KEY     (this session only)
# permanent:  setx OPENAI_API_KEY ""             then restart the terminal/editor
```

Startup prints which file it read (`loaded environment file: …`), and the mode line names the
*variable* the credential came from — never its value.

Without a key, `python -m pipeline` prints `LLM mode requires an API key — falling back to
--no-llm` and runs deterministically anyway (exit 0); `--llm` instead requires the key and exits 2
without it. Spend is capped by `model_config.yaml → llm.max_cost_usd` (overridable with
`--max-llm-cost-usd`) — reaching the cap skips that run's narrative step, never the run itself. See
[LLM mode](docs/flow.md#llm-mode-us-33) for the full design.

`--out-root <dir>` is a **testing/CI-only** flag (`make ci-local` and the workflow use it) that
redirects `artifacts/` and `logs/` to a scratch directory so a test run never touches the real
repository state; production runs never pass it (`docs/interfaces.md` §3, "Interface corrections").

---

## 3. Architecture

`python -m pipeline --no-llm` runs the whole project through a CrewAI **Flow** of ten deterministic
steps, with validation checkpoints and a graceful failure path:

```text
Raw data → 1 Dataset intake → 2 Data Analyst Crew work → 3 Contract validation
→ 4 Data Scientist Crew work → 5 Feature validation → 6 Temporal training & back-testing
→ 7 Model evaluation & champion selection → 8 Inventory policy calibration
→ 9 Artifact validation → 10 Publish to Streamlit
```

The Flow is the conductor, not the orchestra: every step body calls the same `pipeline.*` functions
the standalone CLIs use, so a Flow run and the per-module commands produce byte-identical artifacts,
and an LLM run and a `--no-llm` run produce numerically identical results. Full diagram, `FlowState`
and the crewai-0.86.0 adaptation notes: [`docs/flow.md`](docs/flow.md). The generated,
authoritative surface every module is checked against — `pipeline.paths`, `pipeline.config`,
`pipeline.run_context`, `pipeline.validation` and every foundational module since — is
[`docs/interfaces.md`](docs/interfaces.md).

**Two crews, three agents each**, wired in only when the pipeline runs in LLM mode (§34–§36):

| Crew | Agents | Writes |
|---|---|---|
| **Data Analyst** | Data Quality Analyst · Data Preparation Analyst · Business & EDA Analyst | `data_quality_review.md`, a reviewed `insights.md` |
| **Data Scientist** (narrative only) | Feature Engineering Specialist · Forecasting Model Scientist · Model Evaluation & Inventory Scientist | reviewed `evaluation_report.md`, `model_card.md` |

**Core principle (§38): LLM agents never compute numbers.** They call approved deterministic
Python tools, interpret the tools' output and write prose. Every narrative an agent writes is
checked by the `numbers_in_tables` guard — a rewrite is published only if every number in it exists
in a computed table; on rejection the already-guard-checked deterministic text is published
instead, and the run continues.

**Failure semantics (§39):** every artifact write goes through `ctx.out(path)`, which — when the run
is started with `staging=True` — redirects it into `artifacts/_staging/<run_id>/`. `ctx.promote()`
moves the staged files to their final location atomically and refuses once the run has failed;
`ctx.discard_staging()` removes the (now file-less) staging tree afterwards — `promote()` alone
leaves it standing, empty. `run_log.json` and `validation_report.json` deliberately **bypass**
staging, because they are the files that *report* success or failure, so both exist even when a run
fails. `run_log.json → status` is one of three values — `running` (a process killed before
`finish()`: Ctrl-C, OOM, a CI timeout — this can persist on disk), `success`, or `failed`. A failed
run writes `validation_report.json`, sets `status: failed`, exits non-zero and never touches the
previous run's published forecasts. Details: [`docs/flow.md`](docs/flow.md#failure-handling-us-32-39).

**Reproducibility (§40):** a single global seed (`model_config.yaml → seed`, `42`) seeds every
source of randomness; two runs on the same input produce byte-identical `clean_data.csv`,
`features.csv` and metrics. Full guarantee, exclusions and how to run two copies side by side:
[`docs/reproducibility.md`](docs/reproducibility.md).

---

## 4. Repository layout and required artifacts

```text
data/
    raw/                      # NOT committed (download script + hash check)
    processed/
        clean_transactions.parquet    # audit / EDA on invoices & customers
        clean_data.csv                # required — hand-off panel, StockCode × Month
        features.csv                  # required — model input
artifacts/
    models/model.joblib               # required — champion (refit through 2011-11) + candidates
    forecasts/backtest_predictions.csv, latest_forecast.csv, inventory_plan.csv,
              sigma_table.csv, inventory_kpis.csv, holdout_simulation_rows.csv, quarterly_forecast.csv
    reports/eda_report.html, insights.md, evaluation_report.md, model_card.md   # required
    reports/figures/, eda_tables/, evaluation_tables/, champion_decision.json,
            data_quality_findings.json, feature_validation.json
    contracts/dataset_contract.json   # required
    validation_report.json, run_log.json
config/
    cleaning_config.yaml, model_config.yaml, inventory_policy.yaml,
    non_inventory_stockcodes.csv, data_sources.yaml
src/pipeline/   src/crews/   src/flow/   src/app/
tests/   logs/   docs/   scripts/
```

The **eight artifacts required by the course brief** must exist with exactly these names (§41):
`clean_data.csv`, `features.csv`, `model.joblib`, `eda_report.html`, `insights.md`,
`dataset_contract.json`, `evaluation_report.md`, `model_card.md`. Every canonical location is
resolved through `pipeline.paths` (`docs/interfaces.md` §1) — nothing in the project builds a path
by hand.

`logs/` is not part of a fresh clone — it is created locally by the first run and holds
`run_<id>.log` files and, for a failed run kept with `--keep-failed` (the default),
`logs/failed_runs/<run_id>/`.

---

## 5. Methodology summary

| Topic | Summary | PRD |
|---|---|---|
| Cleaning waterfall | Deterministic, config-driven row-count waterfall: schema validation, key normalisation, cancellation/adjustment removal, price/quantity rules, duplicate policy — every step logged with rows removed | §10 |
| Active-product rule | Active at month `t` ⇔ ≥ 1 positive sale in months `t−k … t−1` (`k = 6`, `model_config.yaml`); month `t` itself is never inspected | §14 |
| Features | 15 features, all from information ≤ forecast origin `t−1` (lags, rolling stats, calendar attributes of `t` as the one allowed exception) | §17 |
| Models | `B1_last_month`, `B2_ma3` (main baseline), `B3_seasonal_naive` (reference only), `M1_linear`, `M2_gbm_poisson` (primary ML), `M3_gbm_squared`, `M4_gbm_absolute` (challenger) | §19 |
| Temporal split & rolling origin | Never `shuffle=True`; train targets 2010-03…2011-05, hold-out 2011-06…2011-11, back-test origins 2010-05…2011-10; **December 2011 is never scored** (partial month) | §21–§22 |
| Metrics | wMAPE and Bias always reported together — never one without the other | §23 |
| Champion gates | (1) bias gate `|Bias| ≤ 0.10`, (2) lowest wMAPE among gate-1 passers, (3) inventory tie-break within 1 wMAPE point, (4) an ML model "improves on the baseline" only at ≥ 2 wMAPE points — a baseline winning is a legitimate, reported outcome, never tuned away | §20 |
| Robust σ and hierarchy | `Robust σ = 1.4826 × MAD` of **out-of-sample** residuals only; fallback product → ABC group (training-window ABC only) → global | §26–§27 |
| Inventory policy | `Safety Stock = z × σ`, `Target Inventory = ceil(max(0, Forecast + z·σ))`; `z = 1.645` default does **not** guarantee a 95 % fill rate — the achieved rate is measured empirically in the back-test | §24–§30 |
| Quarterly aggregation | One-step-ahead monthly forecasts summed into quarters, reported only for complete quarters (3 of 3 months) | §31–§32 |

---

## 6. Results snapshot

Generated from the artifacts of the latest successful run — never typed by hand:

```bash
python scripts/readme_numbers.py
```

`scripts/readme_numbers.py` reads `artifacts/run_log.json` for the run id and its `success |
failed | running` status, and `champion_decision.json`, `evaluation_tables/holdout_metrics_overall.csv`
and `forecasts/inventory_kpis.csv` for the numbers, all through `pipeline.paths` — it opens no
`RunContext`, so running it never allocates a new run id or touches `artifacts/`. It refuses to
print a results table unless the latest run's status is `success`, and states plainly when no
champion was recorded.

<!-- BEGIN readme_numbers -->
_As of run `20260818T095823Z-fdcd09` (status: `success`)._

**Champion:** `M2_gbm_poisson` (ml)

| Model | wMAPE | Bias |
|---|---|---|
| M2_gbm_poisson (champion) | 0.526 | +0.007 |
| B2_ma3 (baseline) | 0.547 | -0.174 |

| Model | Fill rate | Excess units |
|---|---|---|
| M2_gbm_poisson (champion) | 0.880 | 1880261 |
| B1_last_month (best gate-1 baseline) | 0.857 | 1726902 |
<!-- END readme_numbers -->

The PRD's indicative expectation (§46) is baseline wMAPE ≈ 55 % — an expectation to compare against,
never a literal asserted in code or tests. Re-run `python scripts/readme_numbers.py` and replace the
block above whenever a newer successful run should be reflected here.

---

## 7. Screens overview

Seven Streamlit screens (§33), all reading from the artifacts the pipeline produced — never
recomputing a number the pipeline already computed:

1. **Executive Dashboard** — active products, total next-month forecast and target inventory,
   champion wMAPE/Bias, hold-out fill rate, stockout/excess units, run id and data hash.
2. **Product Forecasts** — one row per active product (forecast, safety stock, target inventory,
   σ source, ABC class) with filters and CSV download.
3. **Product Detail** — monthly chart of actuals, back-test forecasts and the next-month forecast
   with a ± z·σ band, plus product metadata.
4. **Model Evaluation** — every candidate's wMAPE/Bias/MAE/RMSE overall, by month and by ABC, and
   the champion decision trace (which gate each candidate passed).
5. **Inventory Policy Evaluation** — forecast-only vs. forecast + robust safety stock, ML vs.
   baseline, an adjustable `z`; fill rate, stockout units, excess units.
6. **Pipeline & Data Quality** — cleaning waterfall, contract validation status, last run status
   and failure message if any, the rendered model card and evaluation report.
7. **Data & Insights (EDA)** — the mandatory EDA figures (E1, E2, E5–E9, E11) with the matching
   `insights.md` narrative, and EDA table downloads.

---

## 8. Configuration

No threshold, date, seed, `k`, `z` or gate is ever a literal in code — every one lives in
`config/*.yaml` and is snapshotted into every run's `run_log.json` (§40, §56):

| File | Owns |
|---|---|
| `config/cleaning_config.yaml` | cancellation/adjustment prefixes, price & quantity rules, duplicate policy, warning thresholds, month boundaries |
| `config/model_config.yaml` | `seed`, active-rule `k`, feature list, split & back-test dates, model ids and hyper-parameters, tuning grid, champion gates, LLM pricing/cost cap |
| `config/inventory_policy.yaml` | lead time, `z` and `z_options`, MAD scale, minimum residuals for σ, ABC thresholds |
| `config/non_inventory_stockcodes.csv` | excluded codes with `reason` and `status` |
| `config/data_sources.yaml` | dataset URLs, expected SHA-256, citation |

To change the active-product window, edit `active_rule.k` in `model_config.yaml`. To change the
safety-stock service factor, edit `z` (and the `z_options` offered in the app's slider) in
`inventory_policy.yaml`. To change a champion gate, edit `models.champion_gates` in
`model_config.yaml` — the gates are executed by code (`pipeline.champion`) and applied identically
to every candidate, including baselines; there is no `--force-champion` flag.

---

## 9. Testing & CI

| File | Proves |
|---|---|
| `tests/test_cleaning.py` | waterfall counts, exclusion list, duplicate policy |
| `tests/test_panel.py` | zero-fill, no rows before first sale, partial flag |
| `tests/test_contract.py` | validation passes, and fails on corrupted samples |
| `tests/test_features.py` | lag correctness, active rule, NaN policy |
| `tests/test_leakage.py` | permutation of future months leaves features unchanged |
| `tests/test_metrics.py` | wMAPE / Bias on known arrays |
| `tests/test_inventory.py` | σ, safety stock, target, simulation |
| `tests/test_flow.py`, `tests/test_flow_no_llm.py` | graceful failure path produces `validation_report.json` |
| `tests/test_determinism.py` | two runs identical |
| `tests/test_acceptance_script.py` | the §49 acceptance audit checks every clause and exits non-zero on a FAIL |

`main` is protected: every change merges through a PR (template: What / Why / How tested) with at
least one reviewer and four required, green CI checks (`lint-test`, `pipeline-no-llm`,
`failure-path`, `determinism`). Run the same four jobs locally before opening a PR:

```bash
make ci-local          # or scripts/ci_local.sh on Windows without GNU make
```

Full contributor workflow: [`docs/contributing.md`](docs/contributing.md). Branch-protection
settings applied to `main`: [`docs/branch_protection.md`](docs/branch_protection.md).

The MVP acceptance criteria (§49) are audited mechanically — one row per clause, with the evidence
behind each verdict, in `artifacts/reports/acceptance_report.md`:

```bash
make acceptance                                   # or: python scripts/mvp_acceptance_check.py
python scripts/mvp_acceptance_check.py --skip-slow # without the two-full-pipeline determinism test
```

What each clause checks, and what the four verdicts mean: [`docs/acceptance.md`](docs/acceptance.md).

The business presentation is generated from the artifacts rather than maintained by hand, for the
same reason every other number here is (§14): a deck edited in PowerPoint states figures that
nothing computed, and goes stale the moment a model is retrained.

```bash
python scripts/build_presentation.py     # -> docs/presentation.pptx + docs/presentation_notes.md
python scripts/capture_screens.py        # re-capture docs/img/screens/ from the running app
```

`build_presentation.py` refuses to build unless `run_log.json` says the audited run succeeded, and
it names on slide 1 any input table that run did not itself write. The deck:
[`docs/presentation.pptx`](docs/presentation.pptx), notes:
[`docs/presentation_notes.md`](docs/presentation_notes.md).

---

## 10. Limitations & ethics

Full detail lives in the model card (`artifacts/reports/model_card.md`, generated by the pipeline)
and PRD §47–§48. In short: only 24 full months of history (one prior season for month-of-year
effects); December 2011 is partial and excluded from every metric; no on-hand/on-order inventory
data, so the output is a target level, never an order quantity; cold-start products (no sales
history) are out of scope; extreme wholesale orders dominate ordinary variance, which is why robust
σ is used instead of `np.std`; products first observed in the dataset's first month have a
left-censored, lower-bound `product_age_months`. No decision is made about individuals — `Customer
ID` is never a model feature — and the application surfaces its own uncertainty and limitations
rather than presenting a forecast as certain.

---

## 11. Roadmap and team

**Version 2 (PRD §50):** quantile regression (P50/P90/P95 as direct inventory levels); direct
multi-horizon models (t+1, t+2, t+3) for start-of-quarter planning; a cold-start rule from a
product's first 14 days; real inventory data (on hand, on order, backorders, lead times) to compute
actual order quantities; cost optimisation (holding vs. stockout cost, margin); demand
capping / wholesale-order flags in evaluation.

**Team & milestones (§54):** Daniel B. Fogel — platform/config/logging, Streamlit, CI, docs,
determinism, presentation, demo. Matan Hillel — raw data → cleaning → panel → data quality → EDA →
contract → Data Analyst Crew, then the Flow. Dor Hll — features → leakage → metrics → models →
back-test → evaluation → σ → inventory → champion → reports → Data Scientist Crew. Milestones M1
(data pipeline + contract + EDA) through M6 (presentation + demo); each ends with a merged PR and a
green CI run.

---

## 12. Glossary (§57)

**Forecast origin** — the last month whose data may be used for a prediction; forecasting month `t`
uses information through `t−1` only. **Active product** — a product with ≥ 1 sale in the previous
`k` months (`k = 6`). **wMAPE** — weighted Mean Absolute Percentage Error, `Σ|error| / Σ actual`.
**Bias** — `Σ(forecast − actual) / Σ actual`; positive means over-forecasting. **Robust σ** —
`1.4826 × MAD` (median absolute deviation) of out-of-sample forecast residuals — resistant to the
extreme wholesale orders that would break a plain standard deviation. **Fill rate** — the share of
actual demand fulfilled by the recommended inventory, measured empirically in the back-test, never
assumed from `z` alone. **Champion** — the model selected by the four-gate rule in §20; a baseline
winning is a legitimate, expected outcome. **Left-censoring** — products already selling in the
dataset's first observed month may in reality be older, so their `product_age_months` is a lower
bound, not an exact age.
