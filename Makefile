.PHONY: install test test-fast lint ci-local ci-lint-test ci-pipeline ci-failure-path ci-determinism clean-ci

# Create the environment first:  python3.11 -m venv .venv && . .venv/bin/activate
# (Windows PowerShell:           py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1)

# Where the ci-local targets write. Each job gets its own root so two runs never collide and the
# repository's own artifacts/ and data/processed/ are never touched (US-34's --out-root).
CI_OUT ?= .ci-local

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

test-fast:
	pytest -q -m "not slow"

lint:
	ruff check src tests

# --------------------------------------------------------------------------
# ci-local — the same four jobs .github/workflows/ci.yml runs, in the same order (US-35).
# Run it before opening a PR to find a red check on your own machine instead of on GitHub.
# Without GNU make (e.g. Windows + Git Bash): scripts/ci_local.sh does the same.
# --------------------------------------------------------------------------
ci-local: ci-lint-test ci-pipeline ci-failure-path ci-determinism
	@echo ""
	@echo "ci-local: all four jobs passed."

ci-lint-test:
	@echo "== lint-test =================================================="
	ruff check src tests
	python -m pip check
	pytest -q -m "not slow" --maxfail=1 --durations=15

ci-pipeline:
	@echo "== pipeline-no-llm ============================================"
	python -m pipeline --no-llm --sample --skip-tuning --out-root $(CI_OUT)/pipeline
	python scripts/ci_check_success_run.py $(CI_OUT)/pipeline
	git diff --exit-code -- artifacts data/processed

ci-failure-path:
	@echo "== failure-path ==============================================="
	@set +e; \
	python -m pipeline --no-llm --skip-tuning \
	  --raw tests/fixtures/raw_sample_missing_quantity.csv \
	  --out-root $(CI_OUT)/failure 2> $(CI_OUT)-stderr.txt; \
	code=$$?; \
	set -e; \
	cat $(CI_OUT)-stderr.txt; \
	python scripts/ci_check_failure_run.py $(CI_OUT)/failure $$code $(CI_OUT)-stderr.txt
	git diff --exit-code -- artifacts data/processed

ci-determinism:
	@echo "== determinism ================================================"
	pytest -q tests/test_determinism.py tests/test_seed_audit.py \
	          tests/test_run_metadata.py tests/test_requirements_pinned.py
	pytest -q -m slow
	python -m pipeline --no-llm --sample --skip-tuning --out-root $(CI_OUT)/det_a
	python -m pipeline --no-llm --sample --skip-tuning --out-root $(CI_OUT)/det_b
	python scripts/ci_check_determinism.py $(CI_OUT)/det_a $(CI_OUT)/det_b

clean-ci:
	rm -rf $(CI_OUT) $(CI_OUT)-stderr.txt
