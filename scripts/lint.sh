#!/usr/bin/env bash
# Same as `make lint`.
set -euo pipefail
ruff check src tests
