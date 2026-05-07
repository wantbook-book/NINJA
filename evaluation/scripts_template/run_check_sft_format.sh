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
OUTPUT_FILE="${2:-${OUTPUT_FILE:-/path/to/wrong_format_contents.jsonl}}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"

CMD=(
  "$PYTHON_BIN" -m evaluation.check_sft_format
  --input "$INPUT_FILE"
  --output "$OUTPUT_FILE"
  --messages-field "$MESSAGES_FIELD"
)

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

CMD+=("$@")

"${CMD[@]}"
