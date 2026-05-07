#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

INPUT_FILE="${1:-${INPUT_FILE:-/path/to/trajs.jsonl}}"
OUTPUT_FILE="${2:-${OUTPUT_FILE:-}}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
NO_COMPONENT_DETAILS="${NO_COMPONENT_DETAILS:-0}"  # 0 | 1
PRESERVE_EXISTING="${PRESERVE_EXISTING:-0}"  # 0 | 1

CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/evaluation/backfill_tool_call_stats.py"
  --input "$INPUT_FILE"
)

if [[ -n "$OUTPUT_FILE" ]]; then
  CMD+=(--output "$OUTPUT_FILE")
fi

if [[ -n "$SUMMARY_FILE" ]]; then
  CMD+=(--summary "$SUMMARY_FILE")
fi

if [[ "$NO_COMPONENT_DETAILS" == "1" ]]; then
  CMD+=(--no_component_details)
fi

if [[ "$PRESERVE_EXISTING" == "1" ]]; then
  CMD+=(--preserve_existing)
fi

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

CMD+=("$@")

"${CMD[@]}"
