#!/usr/bin/env bash
# Same as `make ci-local` — for machines without GNU make (e.g. Windows + Git Bash).
#
# Runs the four jobs of .github/workflows/ci.yml in the same order, with the same commands, so a
# red check can be found here instead of on GitHub. Everything is written under $CI_OUT via
# --out-root, so the working tree's artifacts/ and data/processed/ are never touched — two of the
# jobs assert exactly that.
#
#   scripts/ci_local.sh                  # all four jobs
#   scripts/ci_local.sh lint-test        # one job
#   CI_OUT=/tmp/ci scripts/ci_local.sh   # somewhere else
#
# Run with the project virtual environment already activated.
set -euo pipefail

CI_OUT="${CI_OUT:-.ci-local}"
PY="${PY:-python}"

banner() { printf '\n== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }

job_lint_test() {
  banner "lint-test"
  ruff check src tests
  "$PY" -m pip check
  pytest -q -m "not slow" --maxfail=1 --durations=15
}

job_pipeline_no_llm() {
  banner "pipeline-no-llm"
  "$PY" -m pipeline --no-llm --sample --skip-tuning --out-root "$CI_OUT/pipeline"
  "$PY" scripts/ci_check_success_run.py "$CI_OUT/pipeline"
  git diff --exit-code -- artifacts data/processed
}

job_failure_path() {
  banner "failure-path"
  local code=0
  set +e
  "$PY" -m pipeline --no-llm --skip-tuning \
    --raw tests/fixtures/raw_sample_missing_quantity.csv \
    --out-root "$CI_OUT/failure" 2> "$CI_OUT-stderr.txt"
  code=$?
  set -e
  cat "$CI_OUT-stderr.txt"
  "$PY" scripts/ci_check_failure_run.py "$CI_OUT/failure" "$code" "$CI_OUT-stderr.txt"
  git diff --exit-code -- artifacts data/processed
}

job_determinism() {
  banner "determinism"
  pytest -q tests/test_determinism.py tests/test_seed_audit.py \
            tests/test_run_metadata.py tests/test_requirements_pinned.py
  pytest -q -m slow
  "$PY" -m pipeline --no-llm --sample --skip-tuning --out-root "$CI_OUT/det_a"
  "$PY" -m pipeline --no-llm --sample --skip-tuning --out-root "$CI_OUT/det_b"
  "$PY" scripts/ci_check_determinism.py "$CI_OUT/det_a" "$CI_OUT/det_b"
}

mkdir -p "$CI_OUT"

case "${1:-all}" in
  lint-test)       job_lint_test ;;
  pipeline-no-llm) job_pipeline_no_llm ;;
  failure-path)    job_failure_path ;;
  determinism)     job_determinism ;;
  all)
    job_lint_test
    job_pipeline_no_llm
    job_failure_path
    job_determinism
    printf '\nci-local: all four jobs passed.\n'
    ;;
  *)
    echo "usage: scripts/ci_local.sh [lint-test|pipeline-no-llm|failure-path|determinism|all]" >&2
    exit 2
    ;;
esac
