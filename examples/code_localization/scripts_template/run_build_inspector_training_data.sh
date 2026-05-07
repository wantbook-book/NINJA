#!/bin/bash
# Step 2 of Inspector training data pipeline:
# Build verl-compatible training data from extracted Inspector trajectories.
#
# Prerequisites:
# 1. Run run_extract_inspector_trajectories.sh to get trajectory parquet
#
# Filtering:
#    - Keeps only trajectories where entry_file matches a ground-truth file
#    - Caps per-instance trajectory count to avoid data imbalance
#
# Output:
#    - train.parquet and val.parquet in OUTPUT_DIR
#    - Ready to use with run_train_inspector_template.sh

set -x

# ---- User-configurable variables ----
TRAJECTORY_FILE="${TRAJECTORY_FILE:-$HOME/data/code_loc/inspector_trajectories.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/data/code_loc/inspector_training}"
AGENT_NAME="${AGENT_NAME:-code_localization}"

# Filtering / splitting
MAX_PER_INSTANCE="${MAX_PER_INSTANCE:-5}"
VAL_SIZE="${VAL_SIZE:-100}"
SEED="${SEED:-42}"

# ---- Run build ----
python3 -m examples.code_localization.build_inspector_training_data \
    --trajectory_file ${TRAJECTORY_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --max_per_instance ${MAX_PER_INSTANCE} \
    --val_size ${VAL_SIZE} \
    --seed ${SEED} \
    --agent_name ${AGENT_NAME}
