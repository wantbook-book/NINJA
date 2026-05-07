#!/usr/bin/env bash
set -euo pipefail

PARQUET_FILE="${1:?Usage: bash examples/data_preprocess/scripts_template/run_extract_ground_truth_from_patch.sh /path/to/input.parquet [output.parquet]}"
OUTPUT_FILE="${2:-}"

CMD=(
  python3 -m examples.data_preprocess.extract_ground_truth_from_patch
  --parquet "$PARQUET_FILE"
)

if [[ -n "$OUTPUT_FILE" ]]; then
  CMD+=(--output "$OUTPUT_FILE")
fi

"${CMD[@]}"
