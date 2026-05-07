#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

INPUT_FILE="${INPUT_FILE:-/path/to/input.parquet}"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-$ROOT_DIR/tools/RepoSearch/playground}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/tools/RepoSearch/repo_strucs}"
ERROR_FILE="${ERROR_FILE:-$ROOT_DIR/tools/RepoSearch/repo_structure_errors.txt}"
INSTANCE_ID_LIST_FILE="${INSTANCE_ID_LIST_FILE:-}"

# INPUT_FILE may omit the base_commit column. In that case the cloned repo's
# current checkout is copied and parsed directly.

CMD=(
  python3 "$ROOT_DIR/tools/RepoSearch/get_repo_structure.py"
  batch
  --input-file "$INPUT_FILE"
  --playground-dir "$PLAYGROUND_DIR"
  --output-dir "$OUTPUT_DIR"
  --error-file "$ERROR_FILE"
)

if [[ -n "$INSTANCE_ID_LIST_FILE" ]]; then
  CMD+=(--instance-id-list-file "$INSTANCE_ID_LIST_FILE")
fi

CMD+=("$@")

"${CMD[@]}"
