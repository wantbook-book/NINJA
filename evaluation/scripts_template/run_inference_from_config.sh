#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
LOAD_CONFIG="${LOAD_CONFIG:-/path/to/output_dir/args.json}"
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-}"
RUN_NAME="${RUN_NAME:-}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
CHAIN_SUMMARY_DIR="${CHAIN_SUMMARY_DIR:-}"
INSTANCE_IDS="${INSTANCE_IDS:-}"
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"
MOCK_MODE="${MOCK_MODE:-keep}"  # keep | on | off

export PROJECT_FILE_LOC="${PROJECT_FILE_LOC:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export SGLANG_BASE_URL="${SGLANG_BASE_URL:-}"

if [[ -n "$RESUME_OUTPUT_DIR" && "$LOAD_CONFIG" == "/path/to/output_dir/args.json" ]]; then
  LOAD_CONFIG="$RESUME_OUTPUT_DIR/args.json"
fi

if [[ ! -f "$LOAD_CONFIG" ]]; then
  echo "LOAD_CONFIG does not exist: $LOAD_CONFIG" >&2
  exit 1
fi

CMD=(
  "$PYTHON_BIN" -m evaluation.inference
  --load_config "$LOAD_CONFIG"
)

if [[ -n "$RUN_NAME" ]]; then
  CMD+=(--run_name "$RUN_NAME")
fi

if [[ -n "$BASE_OUTPUT_DIR" ]]; then
  CMD+=(--base_output_dir "$BASE_OUTPUT_DIR")
fi

if [[ -n "$RESUME_OUTPUT_DIR" ]]; then
  CMD+=(--resume_output_dir "$RESUME_OUTPUT_DIR")
fi

if [[ -n "$CHAIN_SUMMARY_DIR" ]]; then
  CMD+=(--chain_summary_dir "$CHAIN_SUMMARY_DIR")
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

case "$MOCK_MODE" in
  keep)
    ;;
  on)
    CMD+=(--mock)
    ;;
  off)
    CMD+=(--no-mock)
    ;;
  *)
    echo "MOCK_MODE must be one of: keep, on, off" >&2
    exit 1
    ;;
esac

CMD+=("$@")

"${CMD[@]}"
