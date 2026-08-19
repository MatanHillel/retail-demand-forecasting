# Contributing

How a change gets from an idea to `main` in this repository. The rules here are the ones CI and
branch protection actually enforce — everything else is in [`CLAUDE.md`](../CLAUDE.md) (working
conventions) and [`docs/PRD.md`](PRD.md) (the specification of record).

## 1. Environment

**Python 3.11 only** — `pyproject.toml` pins `>=3.11,<3.12` (PRD §43) and CI runs 3.11. A venv
built with 3.12 or 3.14 fails at `pip install -e .` with
`requires a different Python`. Never widen the pin to make an install succeed.

```bash
# Linux / macOS / CI
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

```powershell
# Windows (primary dev environment) — PowerShell 5.1 has no `&&`, run these one line at a time
uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
python --version                 # must print 3.11.x
uv pip install -r requirements.txt
uv pip install -e .
```

Call the interpreter by path (`.venv\Scripts\python.exe -m pytest`) rather than relying on an
activated environment — each shell invocation is independent, and `python -m <tool>` guarantees
the tool runs under the interpreter you named instead of a globally installed copy.

## 2. Branches and PRs

* Branch from `main`, named **`feature/US-NN-short-name`** (e.g. `feature/US-35-ci-end-to-end`).
* One Linear issue per branch. The issue body is the source of truth for that unit of work.
* Never push to `main` — it is protected and the push is refused (see
  [`branch_protection.md`](branch_protection.md)).
* Open a PR with the template filled in (**What / Why / How tested**) and request **at least one
  reviewer**. Reference the issue as `US-NN` / `AI-NN` in the title.
* Merging needs a review **and** all four CI checks green.

### Before you open the PR

Run the same four jobs GitHub will run:

```bash
make ci-local
```

It is the exact command set from `.github/workflows/ci.yml`, so a green `ci-local` means the PR
will almost certainly be green too. The individual jobs are also available on their own while you
iterate:

```bash
make ci-lint-test        # ruff + pip check + the unit suite
make ci-pipeline         # the pipeline end to end on the sample fixture
make ci-failure-path     # a broken input must stop gracefully with exit code 2
make ci-determinism      # the reproducibility suites + two runs compared byte for byte
make clean-ci            # delete the .ci-local/ output roots
```

Everything writes into `.ci-local/` via `--out-root`, so your working tree's `artifacts/` and
`data/processed/` are never touched — two of the jobs assert exactly that with
`git diff --exit-code`. If that assertion fails, something wrote to a final path instead of
going through `ctx.out()`.

On Windows there is usually no GNU make. `scripts/ci_local.sh` is the same four jobs as a shell
script, for Git Bash:

```bash
scripts/ci_local.sh                   # all four jobs
scripts/ci_local.sh failure-path      # one job
PY=.venv/Scripts/python.exe scripts/ci_local.sh   # point it at the venv interpreter explicitly
```

## 3. What CI checks, and why each exists

| Job | Command | The failure it exists to catch |
|---|---|---|
| `lint-test` | `ruff check src tests`, `pip check`, `pytest -q -m "not slow" --maxfail=1` | style drift, an inconsistent dependency set, a broken unit |
| `pipeline-no-llm` | `python -m pipeline --no-llm --sample --skip-tuning --out-root ./ci_out` | the pipeline no longer runs end to end, or stops producing one of the eight required artifacts |
| `failure-path` | the same, on `raw_sample_missing_quantity.csv` | a bad input *crashes* (exit 1) instead of stopping gracefully (exit 2) with a readable reason, or a failed run overwrites good forecasts (§39) |
| `determinism` | US-34's reproducibility suites, then two runs compared | a number moved between two runs of identical code on identical data (§40) |

CI needs **no credential**: the pipeline runs in `--no-llm` mode, which imports no LLM class at
all (PRD §37), and reads only the committed sample fixture — the raw dataset is never downloaded
in CI (§42). `tests/test_ci_workflow.py` asserts the workflow references no `secrets.` at all, so
a future job cannot quietly acquire one.

The `pipeline-no-llm` job uploads `run_log.json`, `validation_report.json`, `evaluation_report.md`,
`model_card.md`, `insights.md` and `eda_report.html` as downloadable workflow artifacts — the
fastest way to see what a run actually produced without reproducing it locally.

## 4. Things that will fail review

* **A hard-coded number.** Thresholds, dates, `k`, `z`, gates, ABC cut-offs, seeds and prices live
  in `config/*.yaml` (PRD §40). Indicative figures from the PRD are expectations to log, never
  literals in code and never assertions in tests.
* **An artifact written outside `ctx.out()`.** It bypasses staging, so a failed run can corrupt
  good output (`docs/interfaces.md` §6 rule 1).
* **A CrewAI import under `src/pipeline/`.** That is what keeps `--no-llm` runs LLM-free
  (rule 10).
* **`train_test_split(shuffle=True)`.** The split is temporal; a test enforces the import never
  appears under `src/`.
* **A number in LLM-written prose that is not in a computed table.** The `numbers_in_tables` guard
  rejects it and restores the deterministic text (§38).
* **A new module that later issues import, without a `docs/interfaces.md` section.** Do the drift
  check described in `CLAUDE.md` §8 before writing code, and update the interface doc in the same
  PR.

## 5. Determinism, and one thing that looks like a bug

Two runs on the same data must produce byte-identical `clean_data.csv`, `features.csv` and
metrics (PRD §40). What legitimately differs between two runs is *provenance*: `run_id`,
`started_at` / `finished_at` / `generated_at` / `refit_at`, per-step `duration_s`, `fit_seconds`,
and the `run_id` column inside `inventory_plan.csv`. `docs/reproducibility.md` lists the full
exclusion set; the determinism job compares only files with none of those fields in them.

`PYTHONHASHSEED=0` is set at the workflow level so hash randomisation cannot make an
order-dependent bug intermittent. **It cannot affect an interpreter that is already running** —
Python reads it at startup — and every run re-exports it with the configured seed for its own
child processes (`run_context.set_global_seed`). So do not write a test asserting the pipeline
process sees `PYTHONHASHSEED == "0"`; the variable's real job is to forbid any dependence on
set/dict iteration order in the first place.

## 6. Definition of done

1. Every acceptance-criteria checkbox in the Linear issue verifiably passes — run them, do not
   assume, and paste the output into the issue as evidence.
2. The named tests exist and pass; `ruff check src tests` is clean; `make ci-local` is green.
3. No PRD invariant from `CLAUDE.md` §2 is broken.
4. The **Report for review** is written: a step-by-step, plain-language explanation for a
   non-programmer, defining every technical term on first use, plus a one-line description of
   every file created or changed.
5. A PR is open against protected `main` with the template filled in and CI green.
