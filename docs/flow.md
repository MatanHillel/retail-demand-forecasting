# The CrewAI Flow (US-31, PRD §37)

`python -m pipeline --no-llm` runs the whole project end to end: ten deterministic steps, four
validation checkpoints, staging-safe artifact writes and a graceful failure path. The Flow is the
conductor, not the orchestra — every step body in `src/flow/steps.py` calls the same
`pipeline.*` tools the standalone CLIs use, so a Flow run and the per-module commands produce
byte-identical artifacts, and an LLM run and a `--no-llm` run produce identical numbers.

`python -m pipeline` (no flag) runs **LLM mode** (US-33): the same ten steps, plus two crew
kickoffs woven between them. See [LLM mode](#llm-mode-us-33) below.

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
  → C1 data_analyst_crew_review  LLM MODE ONLY — Data Analyst Crew (US-12):
                                 data_quality_review.md + polished insights.md
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
  → C2 data_scientist_crew_review
                                 LLM MODE ONLY — Data Scientist Crew, narrative task only:
                                 polished evaluation_report.md + model_card.md,
                                 then step 9's completeness check runs again
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
| `llm` | LLM mode only: crew statuses, tokens, cost and the accepted-narrative flags (US-33) |
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

## Failure handling (US-32, §39)

`src/flow/failure.py` owns the single failure-finalising step every graceful stop and every
unexpected exception shares. `RetailForecastFlow._run` (`src/flow/main.py`) catches everything a
step raises (crewai 0.86.0 swallows exceptions raised in listeners — see the adaptation notes
above) and remembers it; the `@listen("fail")` handler and a failure inside step 10 both call
`flow.failure.handle_failure(state, ctx, error, keep_failed=...)`, which:

1. Builds the `ValidationResult` to report. A `FlowValidationError` already carries one. Any other
   exception (`MemoryError`, `KeyError`, …) has none, so one is synthesised with
   `rule="unexpected_exception"` and the (redacted) exception text — never an empty report, which
   would tell the app a run failed for no stated reason.
2. Writes `artifacts/validation_report.json` stamped with `run_id=ctx.run_id` (never `None` —
   `validation_report.json` is not cleared between runs, so an unstamped report can never be tied
   back to the run that produced it).
3. Calls `ctx.finish("failed")` — idempotent for a run `ctx.step(...)` already marked failed,
   and it stamps `finished_at` even when the failure happened between two steps.
4. **Never calls `ctx.promote()`.** It would raise on a failed context anyway; the previous run's
   `artifacts/forecasts/*`, `artifacts/models/model.joblib` and `data/processed/*` are simply never
   touched.
5. Archives this run's staging tree. `ctx.staging_dir` (`artifacts/_staging/<run_id>/`) is moved
   whole to `logs/failed_runs/<run_id>/` — moving the `<run_id>` directory, not its contents, is
   what makes `artifacts/_staging/` literally empty afterwards. `--no-keep-failed` calls
   `ctx.discard_staging()` instead, deleting the tree rather than preserving it for debugging.

The five §39 message templates each live with the check that raises them, and `flow.failure`
re-exports all five as one place to read or grep them: `MISSING_COLUMN` and `RAW_HASH_MISMATCH`
(`pipeline.download`), `CONTRACT_MISMATCH` (`pipeline.contract`'s `CONTRACT_MISMATCH_TEMPLATE`),
`LEAKAGE` (`pipeline.feature_validation`'s `LEAKAGE_FAILURE_MESSAGE`), and
`ARTIFACT_NOT_GENERATED` (`flow.steps.artifact_validation`'s inline wording). None of them repeat
the `FLOW STOPPED:` prefix — `FlowValidationError` adds it exactly once.

### Streamlit banner

`app.components.status.run_status_banner()` (US-27) reads `run_log.json` and shows
`Forecast data unavailable — latest pipeline run failed (run id …)` whenever `status == "failed"`,
with the reason underneath. Two rules keep the banner honest, because `validation_report.json` is
written on success *and* failure and is never cleared between runs:

* the report is trusted as the failure reason only when its `run_id` matches the current
  `run_log.json`'s `run_id` and it did not itself pass — otherwise the banner falls back to
  `run_log["errors"][-1]`;
* a `status == "running"` run log (a process killed before `finish()` — Ctrl-C, OOM, a CI timeout)
  is shown as in progress, not as failed or successful.

## LLM mode (US-33)

`python -m pipeline` runs **exactly the same ten deterministic steps** as `--no-llm`. LLM mode
adds two steps between them and changes nothing else — which is the whole design: the numbers are
produced by `pipeline.*` functions in both modes, so an LLM run and a `--no-llm` run are
numerically identical (§38, §40), and CI can stay LLM-free without testing a different pipeline.

```
                 ┌── router after step 3 returns "contract_ok"
                 ▼
   step 3  ──►  C1  data_analyst_crew_review  ──►  step 4 …
                     Data Analyst Crew (US-12), all three agents, full tool set.
                     Writes data_quality_review.md and rewrites insights.md.

                 ┌── router after step 9 returns "artifacts_ok"
                 ▼
   step 9  ──►  C2  data_scientist_crew_review  ──►  step 10 publish
                     Data Scientist Crew (US-26), T3-narrative ONLY (narrative_only=True).
                     Rewrites the prose of evaluation_report.md and model_card.md.
```

### Where the crews run, and why there

| | Crew 1 | Crew 2 |
|---|---|---|
| Kicked off after | step 3, `contract_validation` | step 9, `artifact_validation` |
| Runs before | step 4 | step 10, `publish` |
| Scope | the full crew — its tools are idempotent and may re-run cleaning and the panel | narrative only — steps 4–8 already ran the identical deterministic tools |
| Writes | `data_quality_review.md`, `insights.md` | `evaluation_report.md`, `model_card.md` |

§37's "LLM-narrative agents run only when the pipeline succeeded" is ambiguous about *which*
success. It is resolved here as: **each crew runs only after the validation checkpoint that
governs its own inputs has passed** — crew 1 after the contract check that proves the cleaning and
the panel are sound, crew 2 after every checkpoint including step 9. A run routed to `"fail"`
reaches neither, and no LLM call is made at all: both step bodies return before opening a step
when `ctx.mode != "llm"`, and the Flow only reaches them through a router that returned its
*continue* label.

Crew 2 runs `narrative_only=True`, which does two things: it builds the crew from the
T3-narrative task and the five reading/guarded-writing tools alone, and it **hydrates**
`DataScientistState` from this run's staged tables (`crews.data_scientist.tools.
hydrate_for_narrative`) so the narrative tools have the tables the guard checks against. Nothing
in that kit can compute a number.

### Four guarantees

1. **A crew is never kicked off after a failure, and never in `--no-llm` mode.**
2. **A crew may not change a number.** Before each kickoff, `flow.llm_mode.snapshot_guarded`
   takes a sha256 *and a byte copy* of every numeric artifact already staged — `clean_data.csv`,
   `features.csv`, `model.joblib`, the forecast CSVs, `champion_decision.json`, every evaluation
   table. Afterwards `restore_guarded` compares, restores from the copy and records
   `crew modified numeric artifact <name> — restored`. The copy is not optional: with staging on,
   `ctx.out(paths.INSIGHTS)` hands the crew *the same path* the deterministic writer used, so
   "restore from staging" would restore the overwritten file — and restoring from the final path
   would publish the **previous** run's numbers under this run's id. The copies live in
   `artifacts/_staging/_guard/<run_id>/` and are deleted by the step that took them.
3. **A crew's own mistakes are warnings, not failures.** The cost cap, a guard restore and an LLM
   or agent error are all caught *inside* the `ctx.step(...)` block and reported with `ctx.warn`.
   This is forced by `RunContext`: `ctx.step` sets `status = "failed"` on any exception it sees,
   `finish()` cannot undo that and `promote()` then refuses — an escaping exception would cost the
   run every artifact over a rejected paragraph. The one exception is a failure a crew's
   *deterministic tool* raised inside its own step: that already flipped `ctx.status`, is a
   genuine validation stop, and is re-raised so it reaches the §39 failure handler with a proper
   message instead of surfacing later as an unexplained refusal to promote.
4. **Completeness is re-checked after crew 2.** Step 9 ran *before* the narrative rewrite, so a
   crew that truncated `evaluation_report.md` would slip past it. `verify_artifacts_after_narrative`
   repeats the same staged existence and non-zero-size check and stops the run gracefully with
   step 9's own wording (`<name> was not generated`).

Every narrative is additionally checked by `numbers_in_tables` inside the crew that wrote it
(`crews.common.NarrativeGuard`): a rewrite is published only if every number in it is in a
computed table, and otherwise the deterministic text is written to *this* run's staged
destination. The three verdicts are reported in `metrics.llm.narrative_accepted`.

### Cost and caching (§47)

`crews.common.make_llm` uses `temperature=0` and the project seed. Prompt caching is measured
rather than assumed: OpenAI (the default provider) caches long prompt prefixes automatically and
reports the hits back as `cached_prompt_tokens`, which `record_token_usage` records and
`crews.environment.estimate_cost_usd` prices at the lower cached rate. A provider that needs an
explicit caching header gets it from `model_config.yaml → llm.extra_params`, which is passed
straight to `crewai.LLM(**extra_params)` → `litellm.completion` — no code change, and no
credential may ever be put there (`config_snapshot` is written into `run_log.json` verbatim).
`Crew(cache=True)` additionally serves a repeated tool call with identical arguments from memory.

Every rate lives in `model_config.yaml → llm.pricing`, keyed by LiteLLM model id with a mandatory
`default` entry — no price is written in code (§40). The estimate exists to drive the cap and to
make spend visible; the provider's invoice is the authority.

The cap is `llm.max_cost_usd` (2.0), overridable per run with `--max-llm-cost-usd`. It is checked
*before* each kickoff: once the estimated spend has reached the cap, the narrative step is
aborted with a warning and the run **continues and succeeds**, with that crew's status
`cost_capped` and its narratives marked not accepted.

### What lands in `run_log.json`

Under `metrics.llm` — not as a top-level key. `RunContext` is a pydantic model with
`extra="forbid"` and a published field list, so `ctx.record_metrics({"llm": …})` is the only way
to record it (and `FlowState.llm` mirrors it for the Flow's own use):

```json
"metrics": {
  "llm": {
    "crew1_status": "completed",          // not_run | completed | cost_capped | failed
    "crew2_status": "completed",
    "model": "gpt-4o-mini",
    "tokens": {"prompt": 0, "cached_prompt": 0, "completion": 0, "total": 0},
    "cost_usd": 0.0,
    "max_cost_usd": 2.0,
    "narrative_accepted": {"insights": true, "evaluation_report": true, "model_card": true},
    "guard_restored": []
  }
}
```

`mode` is `"llm"` only when a crew could actually have run: `RunMode` has exactly two values, so a
run that asked for LLM mode and fell back for want of a credential starts as `"no-llm"` and the
log says so.

### The import boundary

`flow.llm_mode` imports the crews **lazily**, inside `run_analyst_crew` and `run_scientist_crew`.
Those two functions are the only place an LLM stack is constructed, which keeps three things
true: importing `flow.llm_mode` costs nothing, `src/pipeline/` still imports no CrewAI
(`docs/interfaces.md` §6 rule 10 — the CLI's credential check uses `crews.environment`, which is
CrewAI-free), and a test can replace either seam with a stub and prove the wiring without a
network call (`tests/test_flow_llm_mode.py`).

## CLI

```
python -m pipeline [--no-llm | --llm] [--max-llm-cost-usd USD] [--skip-tuning]
                   [--raw <path> | --sample] [--keep-failed | --no-keep-failed]
```

* `--no-llm` — run fully deterministically; no LLM class is imported or instantiated.
* *(no mode flag)* — LLM mode when a credential is set, otherwise it prints
  `LLM mode requires an API key — falling back to --no-llm` and runs deterministically, exit 0.
* `--llm` — require LLM mode: without a credential it prints the how-to and exits 2, before any
  run context is started, so no run log is left stranded at `status: "running"`.
* `--max-llm-cost-usd USD` — override `llm.max_cost_usd` for this run. Reaching the cap aborts the
  narrative step, never the run.
* `--skip-tuning` — skip the grid search and use the parameters already in `model_config.yaml`
  (tuning rewrites that file in place; CI and tests always skip).
* `--raw <path>` / `--sample` — run on an explicit raw CSV / on `tests/fixtures/raw_sample.csv`
  (the CI fixture). The recorded-hash check applies only to the canonical download.
* `--keep-failed` (default) / `--no-keep-failed` — archive a failed run's staging tree under
  `logs/failed_runs/<run_id>/` for debugging, or delete it outright (US-32, §39).

The CrewAI import lives inside `pipeline.__main__.main()`, so importing any `pipeline.*` module
never pulls the LLM stack in (`docs/interfaces.md` §6 rule 10).

Exit codes: `0` success · `2` graceful validation stop (`FLOW STOPPED: …` on stderr) · `1`
unexpected exception.
