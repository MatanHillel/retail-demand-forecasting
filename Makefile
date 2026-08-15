.PHONY: install test test-fast lint

# Create the environment first:  python3.11 -m venv .venv && . .venv/bin/activate
# (Windows PowerShell:           py -3.11 -m venv .venv; .\.venv\Scripts\Activate.ps1)

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
