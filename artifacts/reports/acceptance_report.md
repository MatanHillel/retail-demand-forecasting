# MVP acceptance report

*Generated:* 2026-08-19T19:54:44.075329+00:00 · *Checklist:* PRD §49, 22 criteria

*Audited run:* `20260818T095823Z-fdcd09` · *status:* `success` · *mode:* `no-llm` · *finished:* 2026-08-18T09:58:25.335261+00:00

*Run-log corroboration:* 2 of 8 required artifacts are registered in that run's `artifacts` map, and it recorded 0 warning(s). A partial re-run registers only its own outputs, so a gap here is reported as evidence; a *contradiction*, a missing file, or a `promote()` warning about an unwritten artifact is a FAIL.

*Scope:* this audit is read-only — it opens no `RunContext`, never calls `write_validation_report()`, and modifies nothing except `artifacts/reports/acceptance_report.md` and `artifacts/reports/acceptance_summary.json`.

## Summary

| Result | Count |
|---|---|
| PASS | 20 |
| FAIL | 0 |
| MANUAL | 1 |
| PENDING | 1 |
| **Total** | **22** |

## Criteria

| # | Criterion | How checked | Result | Evidence |
|---:|---|---|---|---|
| 1 | Raw data loads and its integrity hash is checked | `data_sources.expected_sha256` vs `run_log.json → data.sha256`, plus the loaded extract | **PASS** | expected_sha256=bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980; the raw extract loaded into data/processed/clean_transactions.parquet (9861840 bytes); run 20260818T095823Z-fdcd09 recorded no data.sha256 (it re-ran a later step only) |
| 2 | The cleaning waterfall is reproducible | `E01_cleaning_waterfall.csv` steps chain; `test_cleaning.py` + `test_determinism.py` | **PASS** | artifacts/reports/eda_tables/E01_cleaning_waterfall.csv: 10 chained steps, 1067371 -> 1003338 rows; test_cleaning.py passed; test_determinism.py passed |
| 3 | `clean_data.csv` is generated | `paths.CLEAN_DATA` exists; columns equal `panel.PANEL_COLUMNS`; the key is unique | **PASS** | data/processed/clean_data.csv: 100717 rows, 12 columns in the §13.2 order, (stock_code, month) unique; not registered by run 20260818T095823Z-fdcd09 (a partial re-run registers only its own outputs) |
| 4 | The dataset contract is generated and validated by code | `contract.validate_contract_files()` over the two final files | **PASS** | validate_contract_files(data/processed/clean_data.csv, artifacts/contracts/dataset_contract.json) -> contract_validation passed (100717 rows checked) |
| 5 | Product × month panel with explicit zero months | `panel.validate_panel()` plus the share of zero-sales rows | **PASS** | validate_panel -> panel passed (its rules include contiguous_months and first_row_is_a_sale); 34853 of 100717 rows are zero-sales months (34.6%) |
| 6 | The active-product rule uses the configured k | `model_config.active_rule.k` vs `E08_zero_share_by_k.csv` vs the `features.csv` row count | **PASS** | model_config.active_rule.k=6; artifacts/reports/eda_tables/E08_zero_share_by_k.csv row k=6 expects 72182 product-months; data/processed/features.csv has 72182 |
| 7 | Features carry no leakage | `feature_validation.json → passed`; `test_leakage.py` | **PASS** | artifacts/reports/feature_validation.json: passed=True over 10 checks (run 20260817T081028Z-e986bd); test_leakage.py passed |
| 8 | The baselines are computed | every `config.baseline_ids` present in `holdout_metrics_overall.csv` | **PASS** | artifacts/reports/evaluation_tables/holdout_metrics_overall.csv scores ['B1_last_month', 'B2_ma3', 'B3_seasonal_naive'] of the configured baselines ['B1_last_month', 'B2_ma3', 'B3_seasonal_naive'] |
| 9 | At least two ML model variations are trained | `paths.candidate_model(id)` per non-baseline `MODEL_IDS`, plus a row in the overall table | **PASS** | 4 of 4 ML candidates ['M1_linear', 'M2_gbm_poisson', 'M3_gbm_squared', 'M4_gbm_absolute'] have a artifacts/models/<id>.joblib file, and 4 have a row in artifacts/reports/evaluation_tables/holdout_metrics_overall.csv |
| 10 | Temporal hold-out and rolling-origin back-test | months vs `split.holdout_targets`, origins vs `backtest.*`, `split.never_score` absent | **PASS** | hold-out months 2011-06..2011-11 vs the configured 2011-06..2011-11; back-test origins 2010-05..2011-10 vs the configured 2010-05..2011-10; never_score ['2011-12'] appears in no scored target month |
| 11 | wMAPE and Bias reported overall, by month and by ABC | the three `evaluation_tables/*.csv` each carry a `wmape` and a `bias` column | **PASS** | artifacts/reports/evaluation_tables: holdout_metrics_overall.csv (7 rows), holdout_metrics_by_month.csv (42 rows), holdout_metrics_by_abc.csv (21 rows) — each carrying both a wmape and a bias column |
| 12 | The champion is selected by the §20 gates | `champion_decision.json` gate fields, cross-checked against `run_log.json → champion` | **PASS** | artifacts/reports/champion_decision.json: champion=M2_gbm_poisson, 7 candidates each carrying ['gate1_pass', 'gate2_rank', 'gate3_decision', 'wmape', 'bias'], best_baseline=B1_last_month; run 20260818T095823Z-fdcd09 logged champion=None |
| 13 | Robust σ, safety stock and target inventory computed | `sigma_table.csv` levels; `inventory.target_inventory()` recomputed on plan rows | **PASS** | artifacts/forecasts/sigma_table.csv: sigma_source levels ['abc_group', 'product']; inventory.target_inventory() recomputed on 500 of the 3390 forecast rows of artifacts/forecasts/inventory_plan.csv (z=[1.645], 1333 rows carry no forecast: ['Inactive (no sales in last 6 months)', 'Insufficient History / New Product']) — 0 mismatch(es) |
| 14 | Inventory simulated for the ML and the baseline policy | `inventory_kpis.csv` covers the champion and `main_baseline_id` under every policy | **PASS** | artifacts/forecasts/inventory_kpis.csv: policies ['forecast_only', 'forecast_plus_ss']; the champion M2_gbm_poisson and the main baseline B2_ma3 are simulated under every one of them |
| 15 | Streamlit shows screens 1–7 with a CSV download | `Home.py` + `pages/*.py`; `st.download_button` on screen 2; `test_app_smoke.py` | **PASS** | src/app/Home.py plus pages ['2_Product_Forecasts.py', '3_Product_Detail.py', '4_Model_Evaluation.py', '5_Inventory_Policy.py', '6_Pipeline_Data_Quality.py', '7_Data_Insights.py'] — screens 1-7; st.download_button in 2_Product_Forecasts.py: True; test_app_smoke.py passed |
| 16 | EDA report figures and 8–12 backed insights | `eda_tables/index.json` → figures embedded in the HTML; `narrative.numbers_in_tables()` | **PASS** | artifacts/reports/eda_report.html embeds 17 base64 figures and names the figures of 8 of the required sections ['E1', 'E2', 'E5', 'E6', 'E7', 'E8', 'E9', 'E11']; artifacts/reports/insights.md has 12 insights (allowed 8-12); numbers_in_tables checked 24 numbers against 91382 table values -> passed=True |
| 17 | The Flow hands off, validates and fails gracefully | `test_flow.py`, `test_flow_no_llm.py`, `docs/flow.md`, the latest CI run | **PASS** | test_flow.py passed; test_flow_no_llm.py passed; docs/flow.md present=True; CI: latest run 'ci' on main: success |
| 18 | All required artifacts saved under their exact names | `paths.REQUIRED_ARTIFACTS` exist non-empty; `run_log.json` artifacts map and warnings | **PASS** | 8 of 8 artifacts exist non-empty (clean_data.csv, features.csv, model.joblib, eda_report.html, insights.md, dataset_contract.json, evaluation_report.md, model_card.md); 2 of them are registered in run 20260818T095823Z-fdcd09's artifacts map; promote() warnings about unwritten artifacts: 0 |
| 19 | Model-card sections and the evaluation report's champion trace | `## n.` headings in `model_card.md`; the champion id present in `evaluation_report.md` | **PASS** | artifacts/reports/model_card.md carries 5 numbered sections ['Model purpose', 'Training data summary', 'Metrics', 'Limitations', 'Ethical considerations'] (required 5); artifacts/reports/evaluation_report.md names the champion M2_gbm_poisson: True |
| 20 | README with the required sections | `README.md` exists; `test_readme.py` (US-36) passes | **PASS** | README.md present=True; test_readme.py passed |
| 21 | PR-based history with branch protection | `git log --merges` for pull-request merges; `docs/branch_protection.md`; `gh api` | **MANUAL** | git log --merges: 34 merge commit(s) referencing a pull request (latest: faa7208 Merge pull request #35 from MatanHillel/ai-42-us-37-mvp-acceptance); docs/branch_protection.md present=True; branch protection on main: gh unavailable or unauthorised — not readable from here — confirm the GitHub setting in the web UI |
| 22 | Presentation (10–12 slides) and demo video (≤ 5 min) | `docs/presentation.pptx` via python-pptx; `docs/demo.mp4` via ffprobe | **PENDING** | docs/presentation.pptx: 12 slides (allowed 10-12); docs/demo.mp4: unmeasurable s (allowed <= 300) — one deliverable is still missing or unreadable |

## What the statuses mean

* **PASS** — the check ran, and the evidence in the last column supports the clause.
* **FAIL** — the check ran and the clause does not hold. Any FAIL makes this script exit non-zero.
* **MANUAL** — only for a GitHub setting this machine cannot read (`gh` missing or unauthorised); confirm it in the web UI.
* **PENDING** — a deliverable a later story produces (US-38's slides, US-39's video), or an optional configuration value that is not set.

Regenerate with `make acceptance`. `docs/acceptance.md` explains what each criterion checks and why.
