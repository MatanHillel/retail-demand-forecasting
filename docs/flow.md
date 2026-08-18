# The CrewAI Flow (US-31, PRD §37)

`python -m pipeline --no-llm` runs the whole project end to end: ten deterministic steps, four
validation checkpoints, staging-safe artifact writes and a graceful failure path. The Flow is the
conductor, not the orchestra — every step body in `src/flow/steps.py` calls the same
`pipeline.*` tools the standalone CLIs use, so a Flow run and the per-module commands produce
byte-identical artifacts, and (once US-33 lands) an LLM run and a `--no-llm` run produce identical
numbers.

## Step diagram

```
Raw data
  → 1  dataset_intake            download / hash check / load_raw / raw-schema validation
  ── @router ──────────────────  "intake_ok" | "fail"
  → 2  data_analyst_work         clean_transactions → build_panel (+ validate_panel)
                                 → run_eda (E1–E14, eda_report.html, insights.md,
                                   data_quality_findings.json) → write_contract
  → 3  contract_validation       validate_contract(panel, contract)
  ── @router ──────────────────  "contract_ok" | "fail"
  → 4  data_scientist_work       build_features → features.csv
  → 5  feature_validation        validate_features + leakage_check → feature_validation.json
  ── @router ──────────────────  "features_ok" | "fail"
  → 6  training_and_backtest     abc_train → predict_baselines → [tune] → train_models
                                 → backtest → backtest_summary
  → 7  evaluation_and_champion   evaluate → run_sigma → run_inventory_simulation
                                 → select_champion (§20 gates, sets ctx.champion)
  → 8  inventory_policy_calibration
                                 run_latest_forecast (refit_champion, build_latest_forecast,
                                 operational_sigma, build_inventory_plan)
                                 → run_quarterly_aggregation → write_all_reports
  → 9  artifact_validation       every §41 artifact exists in STAGING with non-zero size
  ── @router ──────────────────  "artifacts_ok" | "fail"
  → 10 publish                   ctx.promote() → discard_staging() → run_log.json: success

  "fail" ──→ handle_failure      validation_report.json (stamped with the run id)
                                 + run_log.json: failed — promote() is never called (§39)
```

## FlowState (`src/flow/state.py`)

The state is the pipeline's notebook — JSON-serialisable, no DataFrames:

| Field | Meaning |
|---|---|
| `run_id`, `started_at`, `mode` | mirrored from the `RunContext` at construction |
| `data` | `{file, sha256, rows, columns}` — mirrored from `ctx.data` after step 1 |
| `artifact_paths` | required artifact name → repo-relative path, recorded by step 9 |
| `validation` | `raw_schema / contract / features / leakage / artifacts` — `None` until reached |
| `metrics` | mirrored from `ctx.metrics` at the end of the run |
| `champion` | the §20 decision, mirrored from `ctx.champion` |
| `errors` | mirrored from `ctx.errors` (`{step, type, message, traceback}`) |
| `status` | `running → success \| failed` |
| `current_step` | the step being executed (the failed one, after a stop) |

Every field has a default because crewai's `Flow[FlowState]` instantiates the class with no
arguments. `run_log.json` is **never** written from the state — the `RunContext` is the single
source (US-02); the state mirrors the same facts for routing and for tests.

DataFrames travel between steps in `flow.steps.FlowData`, an in-memory carrier on the Flow
instance. They must never be re-read from the final `paths.*` locations mid-run: with staging on,
those still hold the *previous* run's files (`docs/interfaces.md` §6 rule 7).

## crewai 0.86.0 adaptation (issue §3: divergences are documented here)

The pinned crewai version's Flow semantics differ from the issue's idealised
`@router → "continue" | "fail"` description in two ways, both verified against
`crewai/flow/flow.py`:

1. **A router's returned string replaces its method-name trigger.** `@router(method)` registers a
   router for `method`; after `method` completes, the router runs and its return value becomes the
   trigger label that `@listen("<label>")` methods match on. Two routers returning the same
   `"continue"` would therefore be indistinguishable — so each router returns a distinct continue
   label (`intake_ok`, `contract_ok`, `features_ok`, `artifacts_ok`) while all four share the
   single `"fail"` label into one `@listen("fail")` handler.
2. **Exceptions raised inside `@listen` methods are swallowed.** `Flow._execute_single_listener`
   catches every exception, prints a traceback and returns; the chain simply stops and `kickoff()`
   returns normally (only `@start` methods propagate). The Flow can therefore never rely on an
   exception escaping `kickoff()`. Instead, `RetailForecastFlow._run` catches every step exception
   itself — `FlowValidationError` as a graceful stop, anything else as an unexpected failure —
   records it on the state and the context, and the failure travels *through the state*: steps
   between two routers short-circuit via a state guard, and the next router returns `"fail"`.
   A failure inside step 10 (which no router follows) finalises itself.

## Staging lifecycle (§39)

`run_flow()` starts the context with `staging=True`, so every artifact written through
`ctx.out()` lands under `artifacts/_staging/<run_id>/…` and the final locations keep the last
successful run's files until step 10 promotes. Consequences:

* **Step 9 checks the staged paths** (`ctx.staging_dir / relative`) — checking the final paths
  would find the previous run's files and pass even when this run produced nothing.
* **Step 10 treats a `promote()` warning as a failure**: a registered-but-never-written path only
  warns inside `promote()`, but publishing a partially promoted run would defeat §39. After a
  clean promote, `discard_staging()` removes the (now file-less) staging tree.
* **The failure path never promotes.** It writes `validation_report.json` (stamped with
  `run_id` — the app ignores a report whose id does not match `run_log.json`) and finishes the
  context with `status: failed`. `promote()` would raise on a failed context anyway.
* `run_log.json` is written once immediately after `RunContext.start()`, so a run that dies before
  its first write leaves an honest `status: "running"` on disk instead of the previous run's
  `success`. Both `run_log.json` and `validation_report.json` bypass staging by design (§6 rule 2).

## CLI

```
python -m pipeline --no-llm [--skip-tuning] [--raw <path> | --sample]
```

* `--no-llm` — run fully deterministically; no LLM class is imported or instantiated. Without the
  flag the command prints a notice and behaves identically until US-33 adds the LLM mode.
* `--skip-tuning` — skip the grid search and use the parameters already in `model_config.yaml`
  (tuning rewrites that file in place; CI and tests always skip).
* `--raw <path>` / `--sample` — run on an explicit raw CSV / on `tests/fixtures/raw_sample.csv`
  (the CI fixture). The recorded-hash check applies only to the canonical download.

The CrewAI import lives inside `pipeline.__main__.main()`, so importing any `pipeline.*` module
never pulls the LLM stack in (`docs/interfaces.md` §6 rule 10).

Exit codes: `0` success · `2` graceful validation stop (`FLOW STOPPED: …` on stderr) · `1`
unexpected exception.
