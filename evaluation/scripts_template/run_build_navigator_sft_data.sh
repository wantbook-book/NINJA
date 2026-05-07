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

INPUT_FILE="${1:-${INPUT_FILE:-/path/to/trajs.jsonl}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-/path/to/navigator_sft}}"
VAL_SIZE="${VAL_SIZE:-}"
VAL_RATIO="${VAL_RATIO:-0.0}"
SEED="${SEED:-42}"
MESSAGES_FIELD="${MESSAGES_FIELD:-messages}"
KEEP_MESSAGE_METADATA="${KEEP_MESSAGE_METADATA:-0}"  # 0 | 1

CMD=(
  "$PYTHON_BIN" -m evaluation.build_navigator_sft_data
  --input "$INPUT_FILE"
  --output-dir "$OUTPUT_DIR"
  --val-ratio "$VAL_RATIO"
  --seed "$SEED"
  --messages-field "$MESSAGES_FIELD"
)

if [[ -n "$VAL_SIZE" ]]; then
  CMD+=(--val-size "$VAL_SIZE")
fi

if [[ "$KEEP_MESSAGE_METADATA" == "1" ]]; then
  CMD+=(--keep-message-metadata)
fi

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

CMD+=("$@")

"${CMD[@]}"
