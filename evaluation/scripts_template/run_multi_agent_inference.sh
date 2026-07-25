#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

cd "$PROJECT_ROOT"
export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

INPUT_FILE="${INPUT_FILE:-}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_BACKEND="${MODEL_BACKEND:-openai}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"

GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
MAX_SUB_AGENT_TURNS="${MAX_SUB_AGENT_TURNS:-120}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
ASSISTANT_RESPONSE_PREFILL="${ASSISTANT_RESPONSE_PREFILL:-${ASSISTANT_FIRST_TOKEN:-}}"
SUB_AGENT_PARALLELISM="${SUB_AGENT_PARALLELISM:-2}"
SUB_AGENT_MODEL_NAME="${SUB_AGENT_MODEL_NAME:-}"
SUB_AGENT_MODEL_BACKEND="${SUB_AGENT_MODEL_BACKEND:-}"
SUB_AGENT_API_KEY="${SUB_AGENT_API_KEY:-}"
SUB_AGENT_BASE_URL="${SUB_AGENT_BASE_URL:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
MAIN_AGENT_TOKEN_BUDGET="${MAIN_AGENT_TOKEN_BUDGET:-32768}"
SUB_AGENT_TOKEN_BUDGET="${SUB_AGENT_TOKEN_BUDGET:-32768}"
# Ablation modes: none, no_independent_context, no_dynamic_scheduling, no_hierarchical_search
ABLATION_MODE="${ABLATION_MODE:-none}"

INSTANCE_IDS="${INSTANCE_IDS:-}"
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-}"
LOAD_CONFIG="${LOAD_CONFIG:-}"

export PROJECT_FILE_LOC="${PROJECT_FILE_LOC:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# ---------------------------------------------------------------------------
# Validate required variables
# ---------------------------------------------------------------------------

if [[ -z "$RESUME_OUTPUT_DIR" ]]; then
  REQUIRED_VARS=(
    INPUT_FILE
    BASE_OUTPUT_DIR
    RUN_NAME
    MODEL_NAME
    MODEL_BACKEND
  )
  for VAR_NAME in "${REQUIRED_VARS[@]}"; do
    if [[ -z "${!VAR_NAME}" ]]; then
      echo "$VAR_NAME is required (unless RESUME_OUTPUT_DIR is set)." >&2
      exit 1
    fi
  done
else
  # When resuming, only INPUT_FILE is required (others come from args.json)
  if [[ -z "$INPUT_FILE" ]]; then
    echo "INPUT_FILE is required." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

CMD=(
  "$PYTHON_BIN" -c '
from evaluation.multi_agent_shared import PROJECT_ROOT as _PROJECT_ROOT
import evaluation.multi_agent_inference as _multi_agent_inference

_multi_agent_inference.PROJECT_ROOT = _PROJECT_ROOT
raise SystemExit(_multi_agent_inference.main())
'
  --input_file "$INPUT_FILE"
)

if [[ -n "$RESUME_OUTPUT_DIR" ]]; then
  CMD+=(--resume_output_dir "$RESUME_OUTPUT_DIR")
fi

if [[ -n "$LOAD_CONFIG" ]]; then
  CMD+=(--load_config "$LOAD_CONFIG")
fi

if [[ -n "$BASE_OUTPUT_DIR" ]]; then
  CMD+=(--base_output_dir "$BASE_OUTPUT_DIR")
fi

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

if [[ -n "$MODEL_NAME" ]]; then
  CMD+=(--model_name "$MODEL_NAME")
fi

if [[ -n "$MODEL_BACKEND" ]]; then
  CMD+=(--model_backend "$MODEL_BACKEND")
fi

if [[ -n "$API_KEY" ]]; then
  CMD+=(--api_key "$API_KEY")
fi

if [[ -n "$BASE_URL" ]]; then
  CMD+=(--base_url "$BASE_URL")
fi

if [[ -n "$GRAPH_INDEX_DIR" ]]; then
  CMD+=(--graph_index_dir "$GRAPH_INDEX_DIR")
fi

if [[ -n "$MAX_ROUNDS" ]]; then
  CMD+=(--max_rounds "$MAX_ROUNDS")
fi

if [[ -n "$MAX_SUB_AGENT_TURNS" ]]; then
  CMD+=(--max_sub_agent_turns "$MAX_SUB_AGENT_TURNS")
fi

if [[ -n "$MAX_TOKENS" ]]; then
  CMD+=(--max_tokens "$MAX_TOKENS")
fi

if [[ -n "$TEMPERATURE" ]]; then
  CMD+=(--temperature "$TEMPERATURE")
fi

if [[ -n "$TOP_P" ]]; then
  CMD+=(--top_p "$TOP_P")
fi

if [[ -n "$ASSISTANT_RESPONSE_PREFILL" ]]; then
  CMD+=(--assistant_response_prefill "$ASSISTANT_RESPONSE_PREFILL")
fi

if [[ -n "$SUB_AGENT_PARALLELISM" ]]; then
  CMD+=(--sub_agent_parallelism "$SUB_AGENT_PARALLELISM")
fi

if [[ -n "$SUB_AGENT_MODEL_NAME" ]]; then
  CMD+=(--sub_agent_model_name "$SUB_AGENT_MODEL_NAME")
fi

if [[ -n "$SUB_AGENT_MODEL_BACKEND" ]]; then
  CMD+=(--sub_agent_model_backend "$SUB_AGENT_MODEL_BACKEND")
fi

if [[ -n "$SUB_AGENT_API_KEY" ]]; then
  CMD+=(--sub_agent_api_key "$SUB_AGENT_API_KEY")
fi

if [[ -n "$SUB_AGENT_BASE_URL" ]]; then
  CMD+=(--sub_agent_base_url "$SUB_AGENT_BASE_URL")
fi

if [[ -n "$TOKENIZER_PATH" ]]; then
  CMD+=(--tokenizer_path "$TOKENIZER_PATH")
fi

if [[ -n "$MAIN_AGENT_TOKEN_BUDGET" ]]; then
  CMD+=(--main_agent_token_budget "$MAIN_AGENT_TOKEN_BUDGET")
fi

if [[ -n "$SUB_AGENT_TOKEN_BUDGET" ]]; then
  CMD+=(--sub_agent_token_budget "$SUB_AGENT_TOKEN_BUDGET")
fi

if [[ -n "$ABLATION_MODE" ]]; then
  CMD+=(--ablation_mode "$ABLATION_MODE")
fi

if [[ -n "$INSTANCE_IDS" ]]; then
  CMD+=(--instance_ids "$INSTANCE_IDS")
fi

if [[ -n "$INSTANCE_IDS_FILE" ]]; then
  CMD+=(--instance_ids_file "$INSTANCE_IDS_FILE")
fi

if [[ -n "$START_INDEX" ]]; then
  CMD+=(--start_index "$START_INDEX")
fi

if [[ -n "$END_INDEX" ]]; then
  CMD+=(--end_index "$END_INDEX")
fi

# Pass through any extra arguments
CMD+=("$@")

"${CMD[@]}"
