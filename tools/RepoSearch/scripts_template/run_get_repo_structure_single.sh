#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

REPO_NAME="${REPO_NAME:-psf/requests}"
BASE_COMMIT="${BASE_COMMIT:-}"
INSTANCE_ID="${INSTANCE_ID:-demo-instance}"
PLAYGROUND_DIR="${PLAYGROUND_DIR:-$ROOT_DIR/tools/RepoSearch/playground}"
OUTPUT_FILE="${OUTPUT_FILE:-$ROOT_DIR/tools/RepoSearch/repo_strucs/${INSTANCE_ID}.json}"

CMD=(
  python3 "$ROOT_DIR/tools/RepoSearch/get_repo_structure.py"
  single
  --repo "$REPO_NAME"
  --instance-id "$INSTANCE_ID"
  --playground-dir "$PLAYGROUND_DIR"
  --output-file "$OUTPUT_FILE"
)

if [[ -n "$BASE_COMMIT" ]]; then
  CMD+=(--commit "$BASE_COMMIT")
fi

CMD+=("$@")

"${CMD[@]}"
