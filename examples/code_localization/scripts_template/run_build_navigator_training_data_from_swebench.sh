#!/bin/bash
# Build Navigator training data from preprocessed SWE-bench tool-agent parquet.
#
# Prerequisite:
#   Run examples/data_preprocess/scripts_template/run_swebench_tool_agent_loop_reposearch.sh
#   and use the produced parquet file as INPUT_FILE.
#
# Output:
#   - train.parquet and val.parquet in OUTPUT_DIR
#   - sample.json for quick inspection
#   - train_sample.json and val_sample.json when the corresponding split is non-empty

set -ex

# ---- User-configurable variables ----
INPUT_FILE="${INPUT_FILE:-$HOME/data/swebench/test.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/data/code_loc/navigator_from_swebench}"
AGENT_NAME="${AGENT_NAME:-code_localization}"

# Splitting
VAL_SIZE="${VAL_SIZE:-100}"
SEED="${SEED:-42}"

# ---- Run build ----
python3 -m examples.code_localization.build_navigator_training_data_from_swebench \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --val_size "$VAL_SIZE" \
    --seed "$SEED" \
    --agent_name "$AGENT_NAME"
