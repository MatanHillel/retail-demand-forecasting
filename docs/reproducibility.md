# Reproducibility & determinism (US-34, PRD §40)

**Determinism** — running the same steps on the same data always gives exactly the same result.
**Seed** — a fixed starting number handed to anything "random" (like a machine-learning model's
initial state), so a "random" choice repeats identically every time. **Checksum** — a fingerprint
of a file (a short string computed from its bytes) used to prove two copies are exactly the same
without comparing every byte by eye.

This project promises: two runs of `python -m pipeline --no-llm` on the same input data produce
byte-identical `clean_data.csv`, `features.csv` and evaluation metrics, and every run leaves behind
a complete, checkable record of exactly what produced it. This document says how to reproduce a
run, what is guaranteed identical, what is explicitly excluded from that guarantee, and how to run
two copies of the pipeline side by side without them overwriting each other.

## How to reproduce a run

1. Get the exact input: either the committed sample fixture (`tests/fixtures/raw_sample.csv`, used
   in CI) or the canonical download (`python -m pipeline.download --record-hash`), which verifies a
   SHA-256 hash before proceeding — see `config/data_sources.yaml → expected_sha256`.
2. Confirm the single global seed: `config/model_config.yaml → seed` (currently `42`). Every source
   of randomness in the project — Python's `random`, NumPy's global RNG, and every model's
   `random_state` — is seeded from this one value, by the single call to `set_global_seed()` inside
   `RunContext.start()` (`docs/interfaces.md` §3, §8). No module seeds anything on its own, and no
   module hard-codes a seed literal; `tests/test_seed_audit.py` enforces both.
3. Run the pipeline:
   ```powershell
   .venv\Scripts\python.exe -m pipeline --no-llm --sample --skip-tuning
   ```
   (drop `--sample` to use the canonical download, `--skip-tuning` to skip the hyper-parameter grid
   search and use `config/model_config.yaml` as committed).
4. Every run writes `artifacts/run_log.json` — the receipt for that run. It records the run id, the
   input file's SHA-256, the full configuration snapshot, the seed, library versions, per-step
   timings and row counts, the final metrics, the champion decision, and a checksum for every
   artifact it produced (`artifact_checksums`, added by this issue — see below).

## What is deterministic

Given the same input file and the same `config/*.yaml`, two runs produce:

* **Byte-identical** files: `data/processed/clean_data.csv`, `data/processed/features.csv`,
  `artifacts/reports/eda_tables/E01_cleaning_waterfall.csv`,
  `artifacts/reports/evaluation_tables/holdout_metrics_overall.csv` (and `_by_month`, `_by_abc`),
  `artifacts/forecasts/inventory_kpis.csv`, `artifacts/forecasts/backtest_predictions.csv`.
* **Identical except a handful of run-scoped fields**: `artifacts/reports/champion_decision.json`
  (excluding `run_id`, `generated_at`) and `artifacts/forecasts/inventory_plan.csv` (excluding the
  `run_id` column) — every number in them is the same, only the label of *which run* produced them
  differs.
* **Identical metrics**: `artifacts/run_log.json → metrics` (every wMAPE / Bias / KPI computed for
  the run), `seed`, `data` (the input file's identity — same file, same hash, same row count),
  `config_snapshot`, `versions` and `artifacts` (the map of artifact name → file path).

Two mechanisms make this hold, both fixed at the source rather than worked around in a test:

* `HistGradientBoostingRegressor` candidates are built with `early_stopping=False` and
  `random_state=<the single seed>` (`src/pipeline/models.py`) — early stopping's internal
  train/validation split is itself a source of run-to-run variance, so it is disabled outright.
* Every `pandas.groupby(...)` under `src/pipeline` that a result's row order could depend on passes
  `sort=False` deliberately: with deterministic input order (the panel is always written sorted by
  `stock_code, month`), group order follows first-appearance order, which is itself deterministic —
  not the same thing as being *unsorted* and therefore unstable.
* The one place NumPy's un-seeded random module could leak in — `np.random.*` used without going
  through the single seeded entry point — is checked directly:
  `tests/test_seed_audit.py::test_no_bare_np_random_under_pipeline_outside_the_seeded_entry_points`.

`tests/test_determinism.py` runs the `--no-llm --sample --skip-tuning` pipeline twice, into two
separate output roots, and asserts all of the above.

## What is explicitly excluded

* **Run-scoped bookkeeping** — `run_id`, `started_at`, `finished_at`, every step's `started_at` and
  wall-clock `duration_s`, `champion_decision.json`'s (and `run_log.json["champion"]`'s)
  `generated_at` and `run_id`, `inventory_plan.csv`'s `run_id` column, and
  `validation_report.json`'s `timestamp`. These differ by design — they say *when* and *which* run
  produced a result, not what the result was.
* **LLM narrative** (PRD §47). In LLM mode, the crews review the same deterministic numbers and
  write prose around them — but an LLM is not seeded the way NumPy is, so the exact wording is not
  guaranteed to repeat. Every run records whether the narrative it published passed the
  `numbers_in_tables` guard (no LLM output may state a number that is not in a computed table) as
  `narrative_accepted` — on rejection the deterministic version (already guard-checked) is
  published instead, so the *numbers* stay guaranteed even when the *prose* does not.
* **Model artifact bytes** (`artifacts/models/model.joblib` and friends). The fitted parameters are
  deterministic — same seed, same data, same result — but `joblib`'s pickle serialization of a
  fitted scikit-learn estimator is not itself asserted byte-identical.

## Artifact checksums

Every artifact a run writes (not just the eight required by the course brief) is fingerprinted in
`artifacts/run_log.json → artifact_checksums`, computed **after** `ctx.promote()` moves the run's
staged files to their final location — computing it any earlier would fingerprint the *previous*
run's files instead of this run's (`docs/interfaces.md` §6 rule 7). Each entry is
`{path, bytes, sha256}`:

```json
"artifact_checksums": {
  "clean_data": {"path": "data/processed/clean_data.csv", "bytes": 512044, "sha256": "…"},
  "model": {"path": "artifacts/models/model.joblib", "bytes": 88213, "sha256": "…"}
}
```

This is a field of its own — `artifacts` (the existing `{key: path}` map the app and CI already
read) is never retyped into `{key: {path, sha256}}`, which would silently break every reader of the
existing schema.

## Running two copies side by side: `--out-root`

`python -m pipeline --no-llm --out-root <dir>` writes `artifacts/` and `logs/` under `<dir>` instead
of the repository root — so two runs (or a test and a real run) never collide. It can also be set
via the `RDF_OUT_ROOT` environment variable; the explicit flag wins if both are given. Neither the
flag nor the environment variable changes `pipeline.paths` itself: every canonical path constant
(`paths.CLEAN_DATA`, `paths.MODEL`, …) still points at the repository root exactly as before.
`--out-root` is wired straight to `RunContext.start(base_dir=<dir>)`, which is what every path
inside a run is actually rebased onto (`docs/interfaces.md` §3, §8). Leaving both unset — the
default, and the only mode production runs use — writes into the real repository exactly as always.

```powershell
.venv\Scripts\python.exe -m pipeline --no-llm --sample --skip-tuning --out-root C:\tmp\r1
.venv\Scripts\python.exe -m pipeline --no-llm --sample --skip-tuning --out-root C:\tmp\r2
diff C:\tmp\r1\data\processed\clean_data.csv C:\tmp\r2\data\processed\clean_data.csv   # prints nothing
```

Two files bypass `--out-root`'s staging redirection on purpose, `run_log.json` and
`validation_report.json` — they are the files that *report* success or failure, so both are still
written under `<dir>` directly rather than through the staging area (`docs/interfaces.md` §6 rule
2); `--out-root` relocates where they land, it does not change that rule.
