# Cross-cutting interfaces — the single source of truth

**Status:** generated from the merged code of US-00, US-01 and US-02 (branch `feature/US-02-run-context`).
**Rule:** every issue that uses these modules links to this file instead of restating the API. If an
issue's prompt and this file disagree, **this file wins** — it is derived from code that exists, the
issue text was written before the code did.

**Maintenance:** regenerate and re-sweep the open issues after every foundational merge (US-02 here;
next US-05 `clean_data.csv`, US-13 `features.csv`). Add a section per foundational module; never
document a function that is not yet merged.

---

## 1. `pipeline.paths` — every filesystem location

Nothing in the project builds a path by hand. Import the constant.

| Constant | Location |
|---|---|
| `PROJECT_ROOT` | repository root |
| `CLEANING_CONFIG`, `MODEL_CONFIG`, `INVENTORY_POLICY`, `NON_INVENTORY_STOCKCODES`, `DATA_SOURCES` | `config/…` |
| `CLEAN_TRANSACTIONS` | `data/processed/clean_transactions.parquet` |
| `CLEAN_DATA` ★ | `data/processed/clean_data.csv` |
| `FEATURES` ★ | `data/processed/features.csv` |
| `MODEL` ★ | `artifacts/models/model.joblib` |
| `candidate_model(model_id)` | `artifacts/models/<model_id>.joblib` |
| `BACKTEST_PREDICTIONS`, `LATEST_FORECAST`, `INVENTORY_PLAN`, `SIGMA_TABLE`, `INVENTORY_KPIS`, `HOLDOUT_SIMULATION_ROWS` | `artifacts/forecasts/…` |
| `EDA_REPORT` ★, `INSIGHTS` ★, `EVALUATION_REPORT` ★, `MODEL_CARD` ★ | `artifacts/reports/…` |
| `CHAMPION_DECISION`, `DATA_QUALITY_FINDINGS`, `FEATURE_VALIDATION` | `artifacts/reports/…` |
| `DATASET_CONTRACT` ★ | `artifacts/contracts/dataset_contract.json` |
| `VALIDATION_REPORT`, `RUN_LOG` | `artifacts/…` |
| `FIGURES_DIR`, `EDA_TABLES_DIR`, `EVAL_TABLES_DIR`, `LOGS_DIR`, `FIXTURES_DIR` | directories |
| `REQUIRED_ARTIFACTS` | tuple of the eight ★ artifacts (PRD §41) |

## 2. `pipeline.config` — typed configuration

```python
load_cleaning_config() -> CleaningConfig      # lru_cache(1)
load_model_config()    -> ModelConfig         # lru_cache(1)
load_inventory_policy()-> InventoryPolicy     # lru_cache(1)
load_data_sources()    -> DataSources         # lru_cache(1)
load_non_inventory_codes() -> pd.DataFrame    # NOT cached — re-reads the CSV on every call
clear_config_cache()   -> None                # tests only; clears the four cached loaders
config_snapshot()      -> dict                # five keys: cleaning_config, model_config,
                                              # inventory_policy, data_sources,
                                              # non_inventory_stockcodes
```

No threshold, month, seed or model parameter is ever written in code — it comes from these loaders
(PRD §40). `ModelConfig` carries `seed`, `active_rule.k`, `features`, `split`, `backtest`, `models`,
`tuning`, `champion_gates`.

## 3. `pipeline.run_context` — one run, its log and its safety net

```python
new_run_id() -> str                       # "20260815T190523Z-3f9a1c"; RUN_ID_PATTERN validates it
set_global_seed(seed: int | None = None) -> int      # None → read from model_config.yaml
get_logger(run_id=None, base_dir=None) -> logging.Logger
close_log_handlers(run_id: str) -> None
redact(text: str) -> str

RunContext.start(mode="no-llm", *, staging=False, seed=None, base_dir=None) -> RunContext
```

`start()` allocates the run id, seeds randomness, snapshots configuration and records library
versions. `base_dir` exists **only** so tests can redirect `artifacts/` and `logs/` to a temporary
folder — production callers never pass it.

`start()` does **not** write `run_log.json`. Nothing reaches that file until someone calls
`write_run_log()` or `finish()`. A run that dies before its first write therefore leaves the
*previous* run's log on disk, still saying `success` — so any long-running caller should write the
log once immediately after `start()`.

`RunContext(staging=True)` is not constructible: the model is `extra="forbid"` and `staging` is not
a field. Always go through `RunContext.start(..., staging=True)`.

### Instance API

```python
ctx.step(name, inputs=None)                # context manager: times the step, records the
                                           # exception into ctx.errors and re-raises
ctx.log_rows(name, before, removed, after) # ← MUST be inside a step (raises otherwise)
ctx.warn(message)                          # safe anywhere; redacted
ctx.record_data(file=, sha256=, rows=, columns=)
ctx.record_metrics(dict)
ctx.record_artifact(key, path)
ctx.out(path) -> Path                      # ← EVERY artifact write goes through this
ctx.promote() -> list[Path]                # refuses when status == "failed"
ctx.discard_staging()
ctx.finish(status="success") -> Path       # a failed run stays failed
ctx.write_run_log(path=None, archive_dir=None) -> Path
ctx.logger, ctx.base_dir, ctx.staging_dir, ctx.current_step
```

### `run_log.json` — published schema, extend but never rename

`run_id, started_at, finished_at, mode, status, seed, data{file,sha256,rows,columns},
config_snapshot, versions{python,pandas,numpy,sklearn,crewai,streamlit}, steps[], warnings[],
metrics{}, champion|null, errors[{step,type,message,traceback}], artifacts{key: path}`

`status` is **`running` | `success` | `failed`** — three values, not two. `running` persists on disk
whenever a process is killed before `finish()` (Ctrl-C, OOM, CI timeout), so every reader must
handle it.

Each entry of `steps[]` is `{name, status, started_at, duration_s, inputs, outputs, row_counts,
warnings}`.

## 4. `pipeline.validation` — graceful stop

```python
Violation(step, rule, message, count=None, examples=None)
ValidationResult(step, passed, violations=[], checked_rows=None, extra={})
ValidationResult.summary() -> str
write_validation_report(result, path=None, *, run_id: str) -> Path   # run_id is mandatory
FlowValidationError(result, message=None)          # str(exc) always starts "FLOW STOPPED: "
```

A deterministic step never decides what to do about bad data: it **returns** a `ValidationResult`.
The caller writes the report and raises `FlowValidationError`.

---

## 5. Usage rules — the checklist every issue is swept against

1. **Every artifact write goes through `ctx.out(path)`.** `promote()` only moves paths registered by
   that call; a direct write to a final path bypasses staging, and for that file the §39 guarantee
   silently does not hold. Costs nothing standalone: with `staging=False`, `ctx.out()` returns the
   path unchanged and creates the parent directory.
2. **Two files deliberately bypass staging:** `run_log.json` and `validation_report.json`. They are
   the files that *report* a failure, so they must be readable precisely because the run failed.
   Never route these through `ctx.out()`.
3. **`ctx.log_rows()` only works inside `ctx.step(...)`** — it raises `RuntimeError` otherwise. Any
   standalone entry point (`python -m pipeline.<module>`) must therefore open a step itself:
   ```python
   ctx = RunContext.start(mode="no-llm")
   with ctx.step("<name>"):
       ...
   ctx.finish()
   ```
   `ctx.warn()` has no such constraint.
4. **`write_validation_report` requires `run_id=ctx.run_id`.** The argument is keyword-only and
   has no default, so a call that omits it fails immediately with a `TypeError` rather than
   silently writing a report that cannot be tied to a run (readers need it — see rule 6).
5. **A function that writes a file needs `ctx` in its signature.** Check functions stay pure
   (compute a `ValidationResult`, touch no disk); a separate writer takes `ctx` for the run id and
   the staging redirect.
6. **`validation_report.json` is written on success *and* on failure, and is not cleared between
   runs.** A reader must compare its `run_id` with the `run_id` in `run_log.json` and ignore it when
   they differ — otherwise a failed run displays the previous run's reason, possibly `passed: true`.
7. **Artifact-completeness checks run against the staged paths**, not the final ones. Before
   promotion the final locations still hold the previous successful run's files, so a check on them
   passes on stale leftovers.
8. **`promote()` only warns** when a registered path was never written. Callers that care about
   completeness must treat that warning as a failure.
9. **`promote()` leaves empty directories** under `artifacts/_staging/<run_id>/` — the files are
   unlinked, the tree is not. Call `ctx.discard_staging()` if "staging is empty" must hold literally.
10. **No CrewAI import under `src/pipeline/`.** Library versions are read from package metadata,
    which does not import the package, so `--no-llm` runs stay LLM-free.
11. **No secrets in artifacts.** `redact()` protects log lines and error messages, but
    `config_snapshot` is serialised into `run_log.json` verbatim and `artifacts/` is committed. If a
    credential ever enters a YAML file, extend redaction to the snapshot first.
