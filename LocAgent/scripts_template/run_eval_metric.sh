#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export LOCAGENT_DIR
export PYTHONPATH="${PYTHONPATH:-}:${LOCAGENT_DIR}"

# evaluation mode:
#   dataset: use `evaluate_results` on a HuggingFace dataset split
#   file:    use `eval_w_file` with a local ground-truth jsonl file
MODE="${MODE:-dataset}"

# required
LOC_FILE="${LOC_FILE:-YOUR_LOC_RESULT_JSONL}"

# dataset mode arguments
# DATASET can be:
#   1. a HuggingFace dataset name
#   2. a local `datasets.save_to_disk(...)` directory
#   3. a local parquet/json/jsonl file containing `edit_functions`
#      or `reward_model.ground_truth`
DATASET="${DATASET:-czlll/SWE-bench_Lite}"
SPLIT="${SPLIT:-test}"

# file mode arguments
GT_FILE="${GT_FILE:-}"

# common optional arguments
SELECTED_INSTANCE_ID_FILE="${SELECTED_INSTANCE_ID_FILE:-}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
METRICS="${METRICS:-acc,ndcg,precision,recall,map}"
FILE_K_VALUES="${FILE_K_VALUES:-1,3,5}"
MODULE_K_VALUES="${MODULE_K_VALUES:-5,10}"
FUNCTION_K_VALUES="${FUNCTION_K_VALUES:-5,10}"

# prediction field mapping in localization output jsonl
FILE_KEY="${FILE_KEY:-found_files}"
MODULE_KEY="${MODULE_KEY:-found_modules}"
FUNCTION_KEY="${FUNCTION_KEY:-found_entities}"

export MODE
export LOC_FILE
export DATASET
export SPLIT
export GT_FILE
export SELECTED_INSTANCE_ID_FILE
export OUTPUT_CSV
export METRICS
export FILE_K_VALUES
export MODULE_K_VALUES
export FUNCTION_K_VALUES
export FILE_KEY
export MODULE_KEY
export FUNCTION_KEY

python3 "${LOCAGENT_DIR}/evaluation/run_eval_metric.py" \
  --mode "$MODE" \
  --loc-file "$LOC_FILE" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --gt-file "$GT_FILE" \
  --selected-instance-id-file "$SELECTED_INSTANCE_ID_FILE" \
  --output-csv "$OUTPUT_CSV" \
  --metrics "$METRICS" \
  --file-k-values "$FILE_K_VALUES" \
  --module-k-values "$MODULE_K_VALUES" \
  --function-k-values "$FUNCTION_K_VALUES" \
  --file-key "$FILE_KEY" \
  --module-key "$MODULE_KEY" \
  --function-key "$FUNCTION_KEY"
