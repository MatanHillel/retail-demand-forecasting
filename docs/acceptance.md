# MVP acceptance audit

**Acceptance criteria** are the checklist that says when the product is "done". PRD §49 lists 22 of
them. `scripts/mvp_acceptance_check.py` is the **audit** — a program that walks that checklist and
reports a verdict for each clause, so nobody has to remember what "finished" meant.

```bash
python scripts/mvp_acceptance_check.py                    # the full audit
python scripts/mvp_acceptance_check.py --skip-slow        # without the determinism test
python scripts/mvp_acceptance_check.py --out other.md     # a different report location
make acceptance                                            # the same thing, via make
```

It writes two files and nothing else:

| File | What it is |
|---|---|
| `artifacts/reports/acceptance_report.md` | the human-readable table — one row per clause, with the evidence behind each verdict |
| `artifacts/reports/acceptance_summary.json` | the same verdicts as data, for CI or the app to read |

It exits `0` only when no clause is FAIL, so it can be used as a gate before the M6 presentation.

## The four verdicts

| Verdict | Meaning |
|---|---|
| **PASS** | the check ran and the evidence supports the clause |
| **FAIL** | the check ran and the clause does not hold — the script exits non-zero |
| **MANUAL** | a GitHub setting this machine cannot read (`gh` missing, unauthorised, or a shallow clone with no history) — confirm it in the web UI |
| **PENDING** | a deliverable a later story produces (US-38's slides, US-39's video), or an optional configuration value that is not set |

MANUAL is deliberately restricted: the issue allows it only for GitHub settings and for the content
of the recorded video. Everything else is decided from files on disk, from the typed configuration,
or by running the project's own tests.

## What the audit is allowed to touch

The audit runs **after** the pipeline has finished and promoted its artifacts, and it is read-only
apart from its own two report files. Three rules make that true, and they are not decoration:

* **It opens no `RunContext`.** A `RunContext` would end in `finish()` / `write_run_log()`, which
  overwrite the very `artifacts/run_log.json` the audit is reading, and with staging on, `ctx.out()`
  would bury the report inside `artifacts/_staging/<run_id>/`.
* **It never calls `write_validation_report()`.** That function writes
  `artifacts/validation_report.json` in place — calling it would replace the audited run's report
  with an untraceable one and break the app's failure screen. The audit calls the pure validators
  (`validate_contract_files`, `validate_panel`) and reads the `ValidationResult` they return.
* **Existence is never treated as proof.** Before promotion — and after any failed run, because
  `promote()` refuses to run on a failed one — the final artifact locations still hold the
  *previous* run's files. So the audit reads `run_log.json` first, prints the run id and status at
  the top of the report, and marks every run-derived clause FAIL unless that run's status is exactly
  `success`. (`status` has three values: `running` is what a killed process leaves behind.)

### How the run log is used

The run log **corroborates**; on its own it does not condemn:

* a value that **contradicts** the artifacts — a data hash different from `data_sources.yaml`, a
  champion different from `champion_decision.json` — is a FAIL;
* a `promote()` warning saying `staged artifact was never written` is a FAIL;
* a required artifact **missing or empty on disk** is a FAIL;
* a value the audited run simply **did not record** is reported in the evidence column. A partial
  re-run — the report generator, say — legitimately registers only its own outputs, and calling that
  a failure would mean reporting a red checklist for a repository where every deliverable is
  present. The report's header states how many of the eight required artifacts the audited run
  registered, so the gap is visible rather than hidden.

### No number is typed from the PRD

Model ids come from `pipeline.config.MODEL_IDS`, `baseline_ids`, `primary_model_id` and
`main_baseline_id`; months from `split.holdout_targets`, `backtest.first_origin` / `last_origin` and
`split.never_score`; the look-back from `active_rule.k`; every path from `pipeline.paths`, including
the eight required artifact names in `paths.REQUIRED_ARTIFACTS`. Change the configuration and the
audit follows it.

## The 22 checks

| # | Criterion (PRD §49) | How it is decided |
|---:|---|---|
| 1 | Raw data loads and its hash is checked | `data_sources.expected_sha256` (optional — PENDING when unset) against `run_log.json → data.sha256`, plus the loaded `clean_transactions.parquet`. A contradicting hash is a FAIL. |
| 2 | The cleaning waterfall is reproducible | `E01_cleaning_waterfall.csv` numbers its steps `1..n` and each step's `rows_before` equals the previous step's `rows_after`; `test_cleaning.py` and `test_determinism.py` pass. |
| 3 | `clean_data.csv` is generated | `paths.CLEAN_DATA` exists, its columns are exactly `panel.PANEL_COLUMNS` in order, and `(stock_code, month)` is unique. |
| 4 | The contract is generated and validated by code | `contract.validate_contract_files()` over the two final files; the `ValidationResult.summary()` is the evidence. |
| 5 | Panel with explicit zero months | `panel.validate_panel()` — whose rules include `contiguous_months` and `first_row_is_a_sale` — plus a non-zero share of zero-sales rows. |
| 6 | The active-product rule uses the configured `k` | `model_config.active_rule.k` selects the row in `E08_zero_share_by_k.csv`; its `rows` must equal the `features.csv` row count. |
| 7 | Features carry no leakage | `feature_validation.json → passed` is `true` and `test_leakage.py` passes. |
| 8 | The baselines are computed | every id in `config.baseline_ids` has a row in `holdout_metrics_overall.csv`. |
| 9 | At least two ML variations are trained | every non-baseline id in `MODEL_IDS` has a `paths.candidate_model(id)` file **and** a row in the overall metrics table. |
| 10 | Temporal hold-out and rolling origin | the scored months equal `split.holdout_targets` exactly, the back-test origins equal `backtest.first_origin … last_origin` exactly, and no `split.never_score` month is scored anywhere. |
| 11 | wMAPE and Bias overall / by month / by ABC | the three `evaluation_tables/*.csv` each carry both a `wmape` and a `bias` column. |
| 12 | The champion is selected by the §20 gates | `champion_decision.json` names a champion from `MODEL_IDS` and every candidate carries the gate fields; the run log's `champion` must not contradict it (`null` is reported, not failed). |
| 13 | Robust σ, safety stock, target inventory | `sigma_table.csv` shows which fallback levels were used, and `inventory.target_inventory()` is recomputed on a sample of `inventory_plan.csv` rows — every recomputed value must match the file. |
| 14 | Inventory simulated for both policies | `inventory_kpis.csv` covers the champion and `main_baseline_id` under every policy it contains, and contains at least two policies. |
| 15 | Streamlit screens 1–7 with a CSV download | `src/app/Home.py` plus the numbered pages exist, screen 2 calls `st.download_button`, and `test_app_smoke.py` passes. |
| 16 | EDA figures and 8–12 backed insights | `eda_tables/index.json` maps each required section id to its figures, and each of those figures is embedded in `eda_report.html` as base64; `insights.md` has between `MIN_INSIGHTS` and `MAX_INSIGHTS` numbered findings and passes `narrative.numbers_in_tables()` against every computed EDA table. |
| 17 | The Flow hands off and fails gracefully | `test_flow.py` and `test_flow_no_llm.py` pass and `docs/flow.md` exists; the latest CI run is quoted as evidence when `gh` is available. |
| 18 | All required artifacts, exact names | every path in `paths.REQUIRED_ARTIFACTS` exists non-empty, no `promote()` warning says an artifact was never written, and the run-log corroboration count is reported. |
| 19 | Model card and champion trace | `model_card.md` carries its five `## n.` sections, and `evaluation_report.md` names the champion. |
| 20 | README | `README.md` exists and `test_readme.py` (US-36) passes. |
| 21 | PR-based history and branch protection | `git log --merges` finds merges referencing pull requests, `docs/branch_protection.md` documents the settings, and `gh api` confirms the protection when it can — otherwise MANUAL. A shallow clone (what CI checks out by default) is MANUAL, not FAIL: the history is simply not there to read. |
| 22 | Presentation and demo video | `docs/presentation.pptx` slide count via python-pptx and `docs/demo.mp4` duration via `ffprobe`. PENDING until US-38 and US-39 deliver them. |

## Running the project's tests

Seven clauses are only true if a test says so. The audit runs those test files in **one** pytest
subprocess and maps each failure back to its file through the JUnit XML report, so the suite is
imported once rather than seven times. A file that produces no test case at all — a collection
error — counts as failed: silence is not a pass.

`--skip-slow` drops `test_determinism.py`, which runs the whole `--no-llm` pipeline twice. Use it
while iterating; run the full audit before a milestone.

## Related

* `docs/interfaces.md` — the module surfaces the audit reads.
* `docs/branch_protection.md` — the GitHub settings behind clause 21.
* `docs/reproducibility.md` — determinism, which clause 2 leans on.
