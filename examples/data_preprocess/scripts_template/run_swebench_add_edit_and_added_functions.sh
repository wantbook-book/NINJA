#!/usr/bin/env bash
set -euo pipefail

DATA_SOURCE="${SWEBENCH_DATA_SOURCE:-princeton-nlp/SWE-bench_Verified}"
LOCAL_SAVE_DIR="${SWEBENCH_LOCAL_SAVE_DIR:-$HOME/data/swebench_with_functions}"
LOCAL_DATASET_PATH="${SWEBENCH_LOCAL_DATASET_PATH:-}"
HDFS_DIR="${SWEBENCH_HDFS_DIR:-}"

CMD=(
  python3 -m examples.data_preprocess.swebench_add_edit_and_added_functions
  --data_source "$DATA_SOURCE"
  --local_save_dir "$LOCAL_SAVE_DIR"
)

if [[ -n "$LOCAL_DATASET_PATH" ]]; then
  CMD+=(--local_dataset_path "$LOCAL_DATASET_PATH")
fi

if [[ -n "$HDFS_DIR" ]]; then
  CMD+=(--hdfs_dir "$HDFS_DIR")
fi

"${CMD[@]}"
