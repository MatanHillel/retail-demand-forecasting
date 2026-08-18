# Cross-cutting interfaces — the single source of truth

**Status:** generated from the merged code of US-00, US-01 and US-02, extended with the US-05 panel
surface (branch `matan/pr1-us03-05-data-pipeline`), the US-06 EDA foundations (branch
`matan/pr2-us06-08`), the US-13 feature surface (§10) and the US-23 operational-forecast surface
(§11).
**Rule:** every issue that uses these modules links to this file instead of restating the API. If an
issue's prompt and this file disagree, **this file wins** — it is derived from code that exists, the
issue text was written before the code did.

**Maintenance:** regenerate and re-sweep the open issues after every foundational merge (US-13
here; next the model surface of US-17). Add a section per foundational module; never document a
function that is not yet merged.

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
| `MODEL_META` | `artifacts/models/model_meta.json` |
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
`tuning`, `champion_gates`, `validation` (`lag_sample_rows`, `permutation_products` — US-14 sample
sizes).

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

## 5. `pipeline.panel` & `pipeline.active` — the hand-off panel (US-05)

```python
PANEL_COLUMNS: list[str]                                            # the 12 columns, in order
build_panel(clean_df, returns_lines, cfg: CleaningConfig, ctx) -> pd.DataFrame
validate_panel(panel, cfg: CleaningConfig) -> ValidationResult      # pure: no ctx, no disk
active_mask(panel, k: int | None = None) -> pd.DataFrame            # k=None → active_rule.k
run() -> int                                                        # python -m pipeline.panel
```

`build_panel` **must run inside `ctx.step(...)`** — it calls `ctx.log_rows`. It writes
`data/processed/clean_data.csv` through `ctx.out(...)`, registers it as artifact key
`clean_data`, records the shape change as `log_rows("panel_zero_fill", …)` and the breakdown as
metrics (`panel_rows`, `panel_products`, `panel_nonzero_rows`, `panel_zero_filled_rows`,
`panel_partial_rows`, `panel_zero_share`, `returns_without_panel_row`). It does **not** validate:
the caller runs `validate_panel`, writes the report with `run_id=ctx.run_id` and raises
`FlowValidationError` — same division of labour as §4.

### `clean_data.csv` — published schema, extend but never rename

Grain: one row per `(stock_code, month)` — the **primary key**. Sorted by `stock_code, month`.

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `month` | `str` `YYYY-MM` | calendar month |
| 2 | `stock_code` | `str` | the key; normalised (stripped, upper-case) |
| 3 | `description` | `str`, nullable | canonical description — **display only** |
| 4 | `units_sold` | `int64 ≥ 0` | **the target**: gross demand (§9) |
| 5 | `gross_revenue` | `float ≥ 0` | Σ quantity × price |
| 6 | `avg_unit_price` | `float ≥ 0` | revenue-weighted; last known price in a zero month |
| 7 | `invoice_count` | `int64 ≥ 0` | distinct invoices |
| 8 | `sale_line_count` | `int64 ≥ 0` | sales lines |
| 9 | `customer_count` | `int64 ≥ 0` | distinct customers — **diagnostic, never a feature** |
| 10 | `max_line_qty` | `int64 ≥ 0` | largest single line |
| 11 | `returned_units` | `int64 ≥ 0` | Σ \|qty\| on `C` invoices — **EDA only, never a feature** |
| 12 | `is_partial_month` | `bool` | true only for `cleaning_config → raw.partial_months` |

Invariants enforced by `validate_panel` (rule names are the `Violation.rule` values):
`schema`, `primary_key`, `non_negative`, `is_partial_month`, `month_range`,
`first_row_is_a_sale`, `contiguous_months`, `panel_end`. In words: every product runs from its
**first observed sale** (that first row always has `units_sold > 0` — there are no rows before it)
to the last panel month, one row per month with **no gap**, and zero-sales months are explicit
rows, not missing ones.

### `active_mask` — the §14 rule, one definition for the whole project

`is_active(t) = any(units_sold > 0 in months t−k … t−1)`. Month `t` itself is **never** inspected,
which is the same no-leakage boundary as the forecast origin (§16) — changing the sales of month
`t` can only change months after `t`. Returns `stock_code, month, is_active` for every panel row.
It counts **rows**, so it is only correct on the zero-filled panel (`contiguous_months` above);
never call it on a frame with missing months. EDA (US-10) sweeps several `k`; feature engineering
(US-13) uses the configured one — do not re-implement either.

---

## 6. Usage rules — the checklist every issue is swept against

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
12. **Hand a canonical path to `ctx.out()` and `ctx.record_artifact()` in repo-relative form:**
    `ctx.out(paths.CLEAN_DATA.relative_to(paths.PROJECT_ROOT))`. `out()` rebases a *relative* path
    onto the run's base directory, while an *absolute* `paths.*` constant is returned unchanged —
    so the absolute form silently escapes a test `base_dir` (writing into the real repo) and raises
    under `staging=True`. `record_artifact()` has the mirror-image problem: it stores
    `path.relative_to(base_dir)` and falls back to the **absolute** string when that fails, so an
    absolute constant under a test base dir lands in `run_log.json` as a machine-specific path.
    The relative form is correct in all three modes.
13. **Raw data and inputs are not artifacts.** `ctx.out()` is for run *outputs*. Downloaded raw
    files, the parquet read-cache and committed test fixtures are written to their real locations
    directly — staging them would copy git-ignored bulk into `artifacts/_staging/` and promote it
    into the repo on every successful run.

---

## 7. `pipeline.eda.style`, `pipeline.eda.io` & `pipeline.abc` — EDA foundations (US-06)

Numbered after the usage rules on purpose: §6 rule numbers are cited from several open issues and
must not shift.

```python
# pipeline.eda.style — one look for every figure (§35A.2). Backend is forced to Agg on import.
PALETTE: list[str]                      # Okabe–Ito, colour-blind safe, ordered for series
ABC_COLORS: dict[str, str]              # {"A","B","C"} -> hex; fixed forever
FIGURE_DPI: int                         # 150 — the §35A.2 floor
FIGURE_SIZE, BASE_FONT_SIZE, DEFAULT_FOOTNOTE, LOG_SCALE_SUFFIX, PARTIAL_HATCH, PARTIAL_LABEL
apply_style() -> None                   # mutates global rcParams + Seaborn theme
finalize(fig, title, xlabel, ylabel, footnote=DEFAULT_FOOTNOTE, log_y=False) -> Figure
hatch_partial(ax, x_positions) -> list  # hatches + labels "partial" months (§8)

# pipeline.eda.io — the single choke point for EDA artifact reads and writes
NAME_PATTERN                            # ^E\d{2}_[A-Za-z0-9_]+$ — enforced, not advisory
figure_path(name) -> Path               # repo-relative
table_path(name, fmt="csv") -> Path     # repo-relative
save_figure(fig, name, ctx) -> Path     # >=150 dpi PNG, closes the figure
save_table(df, name, ctx, fmt="csv") -> Path        # fmt in ("csv", "json")
load_table(name, ctx, fmt="csv") -> pd.DataFrame    # staged copy first, final second
figure_to_base64(figure: str | Path, ctx=None) -> str   # name needs ctx; Path does not

# pipeline.abc — one ABC definition for EDA, evaluation, σ fallback and inventory KPIs
ABC_COLUMNS: list[str]                  # stock_code, revenue, revenue_share, cum_share, abc_class
ABC_CLASSES: tuple[str, str, str]
compute_abc(panel, through_month, a_cum_share=None, b_cum_share=None) -> pd.DataFrame  # pure
```

Rules these modules add to §6:

* **Artifact names are validated, not merely conventional.** `save_figure`/`save_table`/
  `load_table` raise `ValueError` on anything that is not `E<nn>_<topic>`. `E01_cleaning_waterfall`
  (written by `pipeline.cleaning`) already follows it.
* **Both savers take `ctx` and write through `ctx.out()`** with the **repo-relative** form of
  `paths.FIGURES_DIR` / `paths.EDA_TABLES_DIR` (§6 rule 12 — the absolute constant would escape a
  test `base_dir`). This is where the §39 guarantee is enforced for all seventeen figures and
  every EDA table, so no analysis issue may write a figure by hand.
* **Readers resolve staged-first and never call `ctx.out()`.** `out()` registers a path for
  promotion, so using it to *locate* a file makes `promote()` warn "staged artifact was never
  written". `load_table` and `figure_to_base64` look in `ctx.staging_dir` first, then
  `ctx.base_dir`, and raise `FileNotFoundError` naming both.
* **`save_table` preserves the caller's row and column order** and writes `index=False`,
  `float_format="%.4f"`, `lineterminator="\n"` — deterministic bytes (§40). It does not sort:
  a top-20 ranking and the cleaning waterfall are ordered on purpose.
* **`compute_abc` is pure** — no `ctx`, no disk. Thresholds default to
  `load_inventory_policy().abc.{a_cum_share,b_cum_share}`; nothing is hard-coded (§40). Persisting
  the table goes through `save_table(..., ctx)`.
* **`through_month` is the leakage boundary.** Revenue is summed over months `≤ through_month`
  only. Modelling, evaluation and σ fallback pass the **last training target month** (§18.2, §23,
  §27); descriptive EDA (E6) may pass the panel end but must label the figure full-period.
  Products first seen after the cut-off are absent from the result — at that origin they had not
  been observed. Class A while `cum_share ≤ a_cum_share`, B while `≤ b_cum_share`, else C, with a
  `1e-9` tolerance so a product landing exactly on a boundary does not fall a class on floating
  point noise. Zero-revenue products are always C.
* **`apply_style()` is global state.** `save_figure` passes `dpi` explicitly, so the ≥ 150 dpi
  guarantee holds even when a caller forgot to call it.

---

## 8. `pipeline.contract` — the dataset contract (US-08)

```python
CONTRACT_STEP = "contract_validation"       # the step name on every Violation and on the report
CONTRACT_MISMATCH_TEMPLATE                  # "clean_data does not match dataset_contract.json ({count} violations)"
CONTRACT_VERSION, DATASET_NAME, SOURCE, MONTH_PATTERN, RETURNED_UNITS_NOTE
CLEANING_ASSUMPTIONS, LEAKAGE_RULES, FEATURE_CONVENTIONS    # the fixed Appendix A prose

write_contract(panel_df, cleaning_cfg, model_cfg, exclusion_df, ctx) -> dict
validate_contract(panel_df, contract: dict) -> ValidationResult      # pure: no ctx, no disk
validate_contract_files(clean_data_path, contract_path) -> ValidationResult   # CLI/CI only
contract_failure_message(result) -> str     # the §39 wording, WITHOUT the "FLOW STOPPED: " prefix
read_panel(path) -> pd.DataFrame            # stock_code and month stay strings
run(argv=None) -> int                       # python -m pipeline.contract write|validate
```

`dataset_contract.json` keys, in written order: `dataset, version, source, generated_at, run_id,
data_sha256, grain, primary_key, date_range, columns, cleaning_assumptions, exclusion_list,
active_rule, partial_month_rule, leakage_rules, modeling_split, row_counts, feature_conventions`.
`columns` holds exactly the twelve `PANEL_COLUMNS` in panel order, each with `type` and `nullable`,
plus `min` on the numeric ones, `format` on `month`, `pattern` on `stock_code` and `note` on
`returned_units`.

Rules for callers:

* **Flow step 3 validates the dict `write_contract` returned**, never `paths.DATASET_CONTRACT`.
  Step 2 stages the write, so until `promote()` the final path still holds the *previous* run's
  contract (§6 rule 7). `validate_contract_files` is for the CLI and CI, where both files are final.
* **The failure wording comes from `contract_failure_message(result)`, not `summary()`.**
  `summary()` returns the single violation's message, or `"<step> failed with <n> violations"` —
  neither is the string §39 fixes. Raise
  `FlowValidationError(result, contract_failure_message(result))`; the exception adds
  `FLOW STOPPED: ` itself, so never include that prefix.
* **The caller writes the report**: `write_validation_report(result, run_id=ctx.run_id)`, bypassing
  `ctx.out()` (§6 rules 2 and 4).
* **Violation rule names** (stable; the app groups on them): `columns`, `unexpected_columns`,
  `dtype`, `nullable`, `primary_key`, `month_format`, `month_range`, `first_row_is_a_sale`,
  `contiguous_months`, `panel_end`, `non_negative`, `stock_code_pattern`, `is_partial_month`.
  A missing column **short-circuits** — one violation, not a dozen consequences of one defect.
* **`row_counts` differences are not violations** (§3): a fresh contract is written each run. They
  are returned in `ValidationResult.extra["row_counts"]` as `{"contract": …, "panel": …}` for the
  caller to re-emit through `ctx.warn`.
* **`panel_end` compares against the observed last month**, not the configured one, so a panel that
  stops a month early is one `month_range` violation rather than one violation per product.
* **`data_sha256` is `null` unless `ctx.record_data(...)` ran** (US-03's `load_raw` does it). A
  standalone `python -m pipeline.contract write` records `null` rather than fabricating a hash.
* **Read `clean_data.csv` with `read_panel`.** Plain `pd.read_csv` infers `stock_code` as an
  integer — `01234` loses its leading zero and then fails a pattern it actually matches.

---

## 9. `crews.common` & `crews.data_analyst` — the crew layer (US-12)

Numbered after §8 for the same reason §7 was: the §6 rule numbers are cited from open issues and
must not shift. **CrewAI may be imported here and nowhere else** (§6 rule 10) — that one-way
direction is what keeps `--no-llm` runs free of any LLM import.

```python
# crews.common — shared by both crews (US-12 here, US-26 next)
API_KEY_VARIABLES: tuple[str, ...]      # ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), tried in order
MODEL_VARIABLE: str                     # "CREWAI_LLM_MODEL"
DEFAULT_MODEL, LLM_TEMPERATURE, NO_API_KEY_MESSAGE
MissingAPIKeyError(RuntimeError)
api_key_variable() -> str | None        # the variable NAME; the value is never returned
require_api_key() -> str                # raises MissingAPIKeyError
llm_model_name() -> str
make_llm(*, seed=None, temperature=0.0) -> crewai.LLM
GuardDecision(label, accepted, text, checked, unmatched); .message -> str
NarrativeGuard(label, tables, fallback)
    .review(candidate) -> GuardDecision                  # pure: decides, writes nothing
    .publish(candidate, destination, ctx) -> GuardDecision
record_token_usage(ctx, label, usage) -> dict[str, int]  # merges into run_log.json -> metrics

# crews.data_analyst
make_tools(ctx) -> list[BaseTool]                        # every tool, bound to the run
DataAnalystToolset(ctx)                                  # .tools, .by_agent, .state
DataAnalystCrew(ctx, llm=None)                           # .agents, .tasks, .crew()
build_crew(ctx, llm=None) -> Crew
run_data_analyst_crew(ctx) -> dict
verify_outputs(ctx) -> list[str]                         # required outputs not written this run
AGENT_ORDER, TASK_ORDER, REQUIRED_OUTPUTS, METRICS_LABEL
relative_path(path) -> Path                              # repo-relative, for ctx.out()
resolve_read(ctx, relative) -> Path                      # staged copy first, final second
review_path() -> Path                                    # artifacts/reports/data_quality_review.md
deterministic_review(state) -> str
```

Rules these modules add to §6:

* **A crew tool takes no argument the model must not choose.** Every deterministic function the
  crew wraps needs DataFrames, a config object and `ctx` — none of which a language model can
  supply — so the tools are built by a factory closing over the run and carry frames between calls
  on `DataAnalystToolset.state`. The four writing tools expose an *empty* argument schema. A later
  crew must do the same: `make_tools(ctx)`, never a module-level tool list.
* **An agent's mistake is not a failed run.** Calling a tool before its inputs exist returns a JSON
  `{"error": ...}` **without opening a step**, so nothing lands in `ctx.errors`, `ctx.status` stays
  `running` and the run remains promotable. Only a genuine exception inside `ctx.step(...)` fails
  the run. Getting this backwards makes a recoverable retry poison the whole run, because
  `ctx.step` sets `status = "failed"` on any exception it sees.
* **The narrative guard decides, the deterministic version wins ties.** `insights.md` is written
  deterministically by `generate_insights` (US-11), which has *already* run `numbers_in_tables` on
  it — so the fallback is known-good. A rewrite is published only if every number in it is in a
  computed table; on rejection the deterministic text is written to `ctx.out(paths.INSIGHTS…)`,
  i.e. **this** run's destination, never the final path (which still holds the previous run's copy
  under staging). `run_data_analyst_crew` re-checks the published file after `kickoff()`.
* **A narrative may only quote numbers that exist in a table — including the crew's own.** The
  exclusion list has no numeric column, so counts like "28 confirmed codes" are computed by the
  `list_nonproduct_codes` tool and recorded on the state; the review quotes them from there. A
  count computed inside a narrative writer is a §38 violation even when the writer is Python.
* **Completeness is checked, not inferred.** `promote()` only warns about a registered path that
  was never written (§6 rule 8), so `verify_outputs(ctx)` checks the required artifacts against the
  **staged** paths (rule 7) and `run_data_analyst_crew` raises when any is missing.
* **The credential stays in the environment.** `make_llm` confirms a key is present and lets
  LiteLLM read it; the value is never passed as an argument, never stored and never logged — only
  the variable *name* is. No credential may enter `config/*.yaml`, which `config_snapshot()`
  serialises verbatim into `run_log.json` (§6 rule 11).
* **Check for the key before `RunContext.start()`.** The exit-2 path must leave no run log stranded
  at `status: "running"` for a run that never began.
* **Two dependencies are pinned for import-time reasons, not features.** `setuptools` (crewai 0.86
  imports `pkg_resources`) and `onnxruntime==1.20.1` (crewai → chromadb instantiates its default
  ONNX embedding function at import time; ≥ 1.21 and pyarrow 17 ship incompatible DLLs on Windows,
  so `import pandas` followed by `import crewai` dies). Both are in `requirements.txt`.

---

## 10. `pipeline.features` — the model input (US-13)

```python
FEATURE_COLUMNS: list[str]      # the 15 §17 features, in model_config.yaml -> features order
FEATURES_COLUMNS: list[str]     # the 20 columns of features.csv, in order

build_features(panel_df, k: int, first_target: str, last_target: str,
               cfg: ModelConfig, include_target: bool = True) -> pd.DataFrame   # pure
build_features_for_origin(panel_df, origin: str, k: int, cfg: ModelConfig) -> pd.DataFrame
write_features(frame, ctx) -> Path
run() -> int                    # python -m pipeline.features
```

**`build_features` is pure and takes no `ctx`.** The issue's original signature had `ctx=None` on a
function that also had to write through `ctx.out()` — the two cannot both hold, so the write lives
in `write_features(frame, ctx)` instead (§6 rules 1 and 5). `build_features_for_origin` likewise
returns a frame. Callers that need the file (US-16, US-23) call both, inside a `ctx.step(...)`.

### `features.csv` — published schema, extend but never rename

Grain: one row per `(stock_code, target_month)` — the **primary key**. Sorted by `stock_code,
target_month`. Written with `float_format="%.6f"`, `index=False`, `lineterminator="\n"`.

`stock_code, forecast_origin, target_month,` then the fifteen features `lag_1, lag_2, lag_3,
rolling_mean_3, rolling_mean_6, rolling_median_6, rolling_std_3, rolling_max_6, nonzero_months_6,
months_since_last_sale, product_age_months, invoice_count_lag_1, avg_unit_price_lag_1,
target_month_of_year, target_quarter,` then `y, is_active`.

`forecast_origin` is always `target_month − 1` month. `is_active` is always true in the saved file —
the column documents the §14 filter that produced the rows. `y` is `units_sold` in the target month
and is **absent** when `include_target=False` (and from `build_features_for_origin`). The file
covers `split.first_target_month … raw.last_full_month` (2010-03 … 2011-11) only: **2011-12 is
never a target here** (§16, §21), it is served on demand by `build_features_for_origin`.

### The three window conventions — the part that is easy to get wrong

Every feature is computed from the units series **shifted one month back inside each product**, so
month `t` is unreachable rather than filtered out. Only `target_month_of_year` and `target_quarter`
come from `t` — calendar attributes, known in advance, the one exception §17 allows.

* **Months before a product's first sale are observed zeros.** The panel has no rows there (§5
  `first_row_is_a_sale`), so `build_features` reindexes onto the full observed month grid first.
  They count as zero-demand months: a product launched 2010-06 has `rolling_mean_6` divided by
  **six** at target 2010-09.
* **Months before the dataset's first observed month are unobserved and excluded**, so early
  windows are *truncated*: at target 2010-03 only three months exist and `rolling_mean_6` is
  divided by **three**. Same arithmetic sum, different divisor — do not conflate the two cases.
* **`months_since_last_sale` and `product_age_months` both count months of history through the
  origin**, so each reads `1` when the relevant sale was in `t−1`. `product_age_months` is a lower
  bound for products already selling in the first observed month (left-censoring, §47).

`rolling_std_3` is the **population** standard deviation (`ddof=0`), which also makes a
one-observation window `0.0` rather than `NaN`. `build_features` raises rather than filling if any
feature is `NaN`, if `cfg.features` disagrees with `FEATURE_COLUMNS`, or if any `forecast_origin` is
not one month before its target.

Rules this module adds to §6:

* **Reuse `active_mask`, never re-derive the §14 rule** — `build_features` merges its output rather
  than recomputing "sold in the last k months", which is what keeps `features.csv` and E8's
  `E08_zero_share_by_k.csv` in agreement (they match exactly at `k = 6`: 72,182 rows, 4,688
  products, 25.94 % zero targets).
* **`k` is a parameter; the window lengths 3 and 6 are not.** Those are part of the feature
  *definitions* — the names encode them — and live as module constants; the active-rule `k` comes
  from `model_config.yaml` and is never a literal.
* **Read `clean_data.csv` with `contract.read_panel`**, not `pd.read_csv` (§8) — `stock_code` must
  stay a string.
* **The column and uniqueness guarantees describe a standalone run.** Under the Flow
  (`staging=True`) the file is at `artifacts/_staging/<run_id>/data/processed/features.csv` until
  `promote()`; a check against the final path before promotion reads the previous run's file (§6
  rule 7).

---

## 11. `pipeline.latest_forecast` — the operational forecast & inventory plan (US-23)

```python
STEP_NAME = "latest_forecast"
INVENTORY_OUTPUT_NAME = "Recommended Target Inventory"      # §7 — never a re-order size
STATUS_FORECAST, STATUS_NEW_PRODUCT, INACTIVE_STATUS_TEMPLATE
LATEST_FORECAST_COLUMNS, INVENTORY_PLAN_COLUMNS             # published column orders
inactive_status(k) -> str                                   # "Inactive (no sales in last 6 months)"

BaselineForecaster(model_id)                                # .model_id, .feature, .predict(features)

resolve_champion(ctx) -> dict                               # ctx.champion first, the JSON second
champion_id(decision) -> str
holdout_metrics_reference(decision, model_id, ctx) -> dict  # {wmape, bias}, never recomputed

operational_origin(cleaning_cfg) -> str                     # raw.last_full_month (2011-11)
operational_features(panel_df, cfg, origin) -> pd.DataFrame # pure
validate_operational_inputs(operational_df, train_features_df, panel_df,
                            origin, cfg, cleaning_cfg) -> ValidationResult      # pure

refit_champion(features_df, champion, cfg, ctx, *, decision=None, origin=None) -> estimator
build_latest_forecast(panel_df, champion_model, cfg, ctx, *, champion, abc_train_df,
                      origin=None, features_df=None) -> pd.DataFrame
operational_sigma(backtest_df, abc_train_df, latest_df, champion, policy_cfg) -> pd.DataFrame
build_inventory_plan(latest_df, sigma_df, policy_cfg, ctx, *, panel_df, abc_train_df,
                     features_df, cfg, origin=None) -> pd.DataFrame
sanity_report(plan, latest, policy_cfg) -> str
run_latest_forecast(cfg, ctx) -> dict                       # the Flow step-8 entry point
run(argv=None) -> int                                       # python -m pipeline.latest_forecast
```

`run_latest_forecast` returns `{champion, model, latest_forecast, sigma_table, inventory_plan,
validation}` and writes four artifacts, **all four through `ctx.out()`** (§6 rule 1):
`artifacts/models/model.joblib` (`paths.MODEL`), `artifacts/models/model_meta.json`
(`paths.MODEL_META`), `artifacts/forecasts/latest_forecast.csv` and
`artifacts/forecasts/inventory_plan.csv`. It **must run inside `ctx.step(...)`** — it calls
`ctx.log_rows("inventory_plan_status", …)`.

### `latest_forecast.csv` — published schema, extend but never rename

One row per **active** product (§14), sorted by `stock_code`:
`stock_code, description, forecast_origin, target_month, model, prediction, lag_1,
rolling_mean_3, abc_class, is_active, status`. `forecast_origin` is always
`cleaning_config → raw.last_full_month` and `target_month` always the month after it (§16);
`prediction` is clipped at zero; `is_active` is always true and `status` always `"Forecast"` —
both document the filter that produced the rows, exactly as `features.csv` does.

### `inventory_plan.csv` — published schema, extend but never rename

One row per product **in the whole panel**, sorted by `stock_code`:
`stock_code, description, forecast_origin, target_month, model, forecast, sigma, sigma_source,
n_residuals_product, z, safety_stock, target_inventory, uncertainty_ratio, abc_class,
last_month_units, ma3_units, months_since_last_sale, product_age_months, status, run_id`.

`status` is one of three values and nothing else: `"Forecast"`, `inactive_status(k)` or
`"Insufficient History / New Product"` (§15 — a product with no observed month at or before the
origin gets no model forecast and no invented history). Only a `"Forecast"` row carries `forecast`,
`sigma`, `sigma_source`, `n_residuals_product`, `z`, `safety_stock`, `target_inventory`,
`uncertainty_ratio`, `ma3_units`, `months_since_last_sale` and `product_age_months`; every other
row leaves them empty rather than claiming a zero. `last_month_units` is read from the panel for
**every** row and equals the `lag_1` feature on a forecast row. Whole-number columns are pandas
nullable `Int64`, so the CSV holds `936` and `` — never `936.000000` and never a phantom `0`.

Rules this module adds to §6:

* **The champion is never named by hand.** `resolve_champion` reads `ctx.champion` (set by US-22)
  and falls back to `paths.CHAMPION_DECISION` only for the standalone CLI, where the producing run
  has already promoted. Under `staging=True` the final JSON still holds the *previous* run's
  decision (§6 rule 7), so the Flow must set `ctx.champion` rather than rely on the file. A missing
  decision raises `FileNotFoundError`: PRD §20 is executed by code and there is no
  `--force-champion` flag.
* **`model.joblib` is the champion *refit through the origin*, not a hold-out candidate.** US-17's
  candidates stop at the training window and stay at `paths.candidate_model(model_id)`;
  `model_meta.json` records `train_targets`, `n_rows`, `seed`, `sklearn_version`, `run_id` and the
  `holdout_metrics_reference` so the two can never be confused. A champion that is a **baseline** is
  persisted as a `BaselineForecaster` (`kind: "baseline"`, `n_rows: 0`), because a baseline winning
  the §20 gates is a legitimate outcome and `model.joblib` is a required artifact (§41).
  `B3_seasonal_naive` is unsupported by design — reference only (§19), and its rule reads the panel's
  month `t−12` rather than a feature column.
* **The December-2011 boundary is proved, not asserted.** `validate_operational_inputs` rebuilds the
  operational features from a panel whose `raw.partial_months` rows carry corrupted measurements and
  requires an identical frame (rule `partial_month_not_used`), plus `forecast_origin`, `target_month`
  and `refit_window`. It is pure and returns a `ValidationResult`; the caller writes the report with
  `write_validation_report(result, run_id=ctx.run_id)` and raises `FlowValidationError` (§6 rules 4
  and 5).
* **σ for the operational month needs a universe row.** `pipeline.sigma.sigma_table` takes the
  products to price from the rows it finds *at* the evaluation month, and the back-test's last target
  is 2011-11 — so `operational_sigma` appends one residual-free placeholder per active product at the
  target month. Eligibility (`target_month < t` **and** the residual is not NaN) is unchanged, so the
  placeholders price the products without ever pricing themselves. Calling `sigma_table` directly
  with `eval_months=["2011-12"]` returns an empty frame; use `operational_sigma`.
* **The two formulas come from `pipeline.inventory`.** `safety_stock` and `target_inventory` are
  imported, never restated — one definition of §25 and §28 for the hold-out simulation and the
  operational plan alike.
* **`run_latest_forecast` reads its four inputs from the canonical `paths.*` locations**, which is
  correct standalone (`staging=False`) and wrong under the Flow: US-33 step 8 must hand in the frames
  the producing steps returned (§6 rule 7).
