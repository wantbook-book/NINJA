#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── List the input files to evaluate here ────────────────────────────────
INPUT_FILES=(
  "/path/to/exp1/traj/trajs.jsonl"
  "/path/to/exp2/traj/trajs.jsonl"
)
# ──────────────────────────────────────────────────────────────────────────

K_VALUES="${K_VALUES:-}"
MAIN_AGENT_TOKEN_BUDGET="${MAIN_AGENT_TOKEN_BUDGET:-}"
SUB_AGENT_TOKEN_BUDGET="${SUB_AGENT_TOKEN_BUDGET:-}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-_eval}"

ok=0
fail=0

for INPUT_FILE in "${INPUT_FILES[@]}"; do
  if [[ "$INPUT_FILE" == *.jsonl ]]; then
    OUTPUT_FILE="${INPUT_FILE%.jsonl}${OUTPUT_SUFFIX}.jsonl"
  else
    OUTPUT_FILE="${INPUT_FILE}${OUTPUT_SUFFIX}.jsonl"
  fi

  CMD=(
    python "$ROOT_DIR/evaluation/evaluate.py"
    --input "$INPUT_FILE"
    --output "$OUTPUT_FILE"
  )

  if [[ -n "$K_VALUES" ]]; then
    CMD+=(--k "$K_VALUES")
  fi

  if [[ -n "$MAIN_AGENT_TOKEN_BUDGET" ]]; then
    CMD+=(--main_agent_token_budget "$MAIN_AGENT_TOKEN_BUDGET")
  fi

  if [[ -n "$SUB_AGENT_TOKEN_BUDGET" ]]; then
    CMD+=(--sub_agent_token_budget "$SUB_AGENT_TOKEN_BUDGET")
  fi

  echo "==> Evaluating: $INPUT_FILE -> $OUTPUT_FILE"
  if "${CMD[@]}"; then
    ok=$((ok + 1))
  else
    echo "ERROR: evaluation failed for $INPUT_FILE" >&2
    fail=$((fail + 1))
  fi
done

echo ""
echo "Done: $ok succeeded, $fail failed."
[[ $fail -eq 0 ]]
