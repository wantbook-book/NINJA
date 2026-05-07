#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export PYTHONPATH="$ROOT_DIR/LocAgent:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

DATASET="${DATASET:-czlll/SWE-bench_Lite}"
INPUT_FILE="${INPUT_FILE:-}"
SPLIT="${SPLIT:-test}"
NUM_PROCESSES="${NUM_PROCESSES:-30}"
REPO_PATH="${REPO_PATH:-$ROOT_DIR/LocAgent/playground/build_graph}"
INDEX_DIR="${INDEX_DIR:-$ROOT_DIR/LocAgent/index_data}"
INSTANCE_ID_PATH="${INSTANCE_ID_PATH:-}"
DOWNLOAD_REPO="${DOWNLOAD_REPO:-1}"

CMD=(
  python3 "$ROOT_DIR/LocAgent/dependency_graph/batch_build_graph.py"
  --split "$SPLIT"
  --num_processes "$NUM_PROCESSES"
  --repo_path "$REPO_PATH"
  --index_dir "$INDEX_DIR"
)

if [[ -n "$INPUT_FILE" ]]; then
  CMD+=(--input-file "$INPUT_FILE")
else
  CMD+=(--dataset "$DATASET")
fi

if [[ "$DOWNLOAD_REPO" == "1" ]]; then
  CMD+=(--download_repo)
fi

if [[ -n "$INSTANCE_ID_PATH" ]]; then
  CMD+=(--instance_id_path "$INSTANCE_ID_PATH")
fi

CMD+=("$@")

"${CMD[@]}"
