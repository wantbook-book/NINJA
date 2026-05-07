#!/bin/bash
set -euo pipefail

INPUT="${INPUT:-/path/to/generated_swebench_tool_agent_data}"
SPLIT="${SPLIT:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/sample_output}"
SAMPLE_PER_REPO="${SAMPLE_PER_REPO:-2}"
SEED="${SEED:-42}"
INSTANCE_ID_FILE="${INSTANCE_ID_FILE:-}"

cmd=(
  python -m examples.data_preprocess.sample_swebench_tool_agent_data
  --input "$INPUT"
  --split "$SPLIT"
  --output_dir "$OUTPUT_DIR"
  --sample_per_repo "$SAMPLE_PER_REPO"
  --seed "$SEED"
)

if [[ -n "$INSTANCE_ID_FILE" ]]; then
  cmd+=(--instance_id_file "$INSTANCE_ID_FILE")
fi

"${cmd[@]}"
