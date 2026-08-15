#!/usr/bin/env bash
# Same as `make install` — for machines without GNU make (e.g. Windows + Git Bash).
# Run with the project virtual environment already activated.
set -euo pipefail
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
