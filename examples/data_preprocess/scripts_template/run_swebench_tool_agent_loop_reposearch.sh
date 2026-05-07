#!/bin/bash
set -ex
data_source="data_name"
agent_name="tool_agent"
local_save_dir="/path/to/data_dir/data_name"
project_file_loc="/path/to/strucs"
split="test"
gt_locations_file="/path/to/gt_locations_file.jsonl"
python -m examples.data_preprocess.swebench_tool_agent_loop_reposearch \
    --data_source "$data_source" \
    --agent_name "$agent_name" \
    --local_save_dir "$local_save_dir" \
    --project_file_loc "$project_file_loc" \
    --split "$split" \
    --gt_locations_file "$gt_locations_file"
