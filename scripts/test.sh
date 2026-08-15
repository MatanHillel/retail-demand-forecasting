#!/usr/bin/env bash
# Same as `make test`.
set -euo pipefail
pytest -q "$@"
