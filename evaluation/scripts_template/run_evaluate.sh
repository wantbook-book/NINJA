#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

INPUT_FILE="${1:-/path/to/trajs.jsonl}"
OUTPUT_FILE="${2:-}"
K_VALUES="${K_VALUES:-}"
MAIN_AGENT_TOKEN_BUDGET="${MAIN_AGENT_TOKEN_BUDGET:-}"
SUB_AGENT_TOKEN_BUDGET="${SUB_AGENT_TOKEN_BUDGET:-}"

CMD=(
  python "$ROOT_DIR/evaluation/evaluate.py"
  --input "$INPUT_FILE"
)

if [[ -n "$OUTPUT_FILE" ]]; then
  CMD+=(--output "$OUTPUT_FILE")
fi

if [[ -n "$K_VALUES" ]]; then
  CMD+=(--k "$K_VALUES")
fi

if [[ -n "$MAIN_AGENT_TOKEN_BUDGET" ]]; then
  CMD+=(--main_agent_token_budget "$MAIN_AGENT_TOKEN_BUDGET")
fi

if [[ -n "$SUB_AGENT_TOKEN_BUDGET" ]]; then
  CMD+=(--sub_agent_token_budget "$SUB_AGENT_TOKEN_BUDGET")
fi

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

CMD+=("$@")

"${CMD[@]}"
