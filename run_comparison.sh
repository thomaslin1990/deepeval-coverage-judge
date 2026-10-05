#!/usr/bin/env bash
# Run the coverage judge on the CSVs in data/ by calling the Python code directly.
#
# Usage:
#   ./run_comparison.sh [extra flags, e.g. --top-k 5 --judge-threshold 0.75]
set -euo pipefail

# Always run from the project folder, wherever the script is called from.
cd "$(dirname "$0")"

# Use the virtual environment's Python (it has deepeval, pandas, etc. installed).
PYTHON=.venv/bin/python

# Let Python find the coverage_judge package inside src/.
export PYTHONPATH=src

# Run src/coverage_judge/cli.py as a module; it calls main() at the bottom.
"$PYTHON" -m coverage_judge.cli \
  --baseline data/baseline.example.csv \
  --candidate data/candidate.example.csv \
  --output judge_results.csv \
  "$@"
