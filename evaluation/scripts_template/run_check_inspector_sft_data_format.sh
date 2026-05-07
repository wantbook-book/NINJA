#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

INPUT_FILE="${1:-${INPUT_FILE:-/path/to/inspector_sft/train.parquet}}"
OUTPUT_FILE="${2:-${OUTPUT_FILE:-/path/to/wrong_assistant_contents.jsonl}}"
CLEAN_OUTPUT_FILE="${3:-${CLEAN_OUTPUT_FILE:-}}"
SUMMARY_OUTPUT_FILE="${4:-${SUMMARY_OUTPUT_FILE:-}}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"

CMD=(
  "$PYTHON_BIN" -m evaluation.check_inspector_sft_data_format
  --input "$INPUT_FILE"
  --output "$OUTPUT_FILE"
  --messages-field "$MESSAGES_FIELD"
)

if [[ -n "$CLEAN_OUTPUT_FILE" ]]; then
  CMD+=(--clean-output "$CLEAN_OUTPUT_FILE")
fi

if [[ -n "$SUMMARY_OUTPUT_FILE" ]]; then
  CMD+=(--summary-output "$SUMMARY_OUTPUT_FILE")
fi

if [[ $# -ge 4 ]]; then
  shift 4
elif [[ $# -ge 3 ]]; then
  shift 3
elif [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

CMD+=("$@")

"${CMD[@]}"
