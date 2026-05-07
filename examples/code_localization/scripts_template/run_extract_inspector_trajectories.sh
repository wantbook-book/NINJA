#!/bin/bash
# Step 1 of Inspector training data pipeline:
# Extract Inspector (sub-agent) trajectories from multi-agent inference outputs.
#
# Prerequisites:
# 1. Run multi_agent_inference.py to produce an output directory with
#    traj/trajs.jsonl and traj/sub_agents/{instance_id}/*.json
# 2. Have the original input data file (parquet or jsonl) available
#
# Output:
#    - A parquet file with one row per (instance_id, entry_file) sub-agent trajectory

set -x

# ---- User-configurable variables ----

# Path to the multi_agent_inference.py output directory
# e.g. outputs/multi_agent/20240101_120000/
INFERENCE_OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:?Please set INFERENCE_OUTPUT_DIR}"

# Original input data file (for problem_statement / structure / patch lookup)
INPUT_FILE="${INPUT_FILE:-$HOME/data/code_loc/train.parquet}"

# Output parquet
OUTPUT_FILE="${OUTPUT_FILE:-$HOME/data/code_loc/inspector_trajectories.parquet}"

# Data selection (optional)
INSTANCE_IDS="${INSTANCE_IDS:-}"
INSTANCE_IDS_FILE="${INSTANCE_IDS_FILE:-}"

# ---- Run extraction ----
CMD="python3 -m examples.code_localization.extract_inspector_trajectories \
    --inference_output_dir ${INFERENCE_OUTPUT_DIR} \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE}"

# Append optional arguments
if [ -n "${INSTANCE_IDS}" ]; then
    CMD="${CMD} --instance_ids ${INSTANCE_IDS}"
fi
if [ -n "${INSTANCE_IDS_FILE}" ]; then
    CMD="${CMD} --instance_ids_file ${INSTANCE_IDS_FILE}"
fi

eval ${CMD}
