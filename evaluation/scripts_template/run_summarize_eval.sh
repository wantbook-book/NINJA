#!/bin/bash
set -euo pipefail

RESULT_ROOT="${RESULT_ROOT:-/path/to/result_root}"

python evaluation/summarize_eval_runs.py \
  "$RESULT_ROOT" \
  --recursive \
  --time-dirs-only
