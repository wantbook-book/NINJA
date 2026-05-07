#!/bin/bash
set -ex

# Path to the original gt_location.jsonl (edit location only)
old_file="evaluation/gt_location/SWE-bench_Lite/test/gt_location.jsonl"

# Path to the new gt_location_with_support.jsonl produced by the updated
# gen_oracle_locations.py
new_file="evaluation/gt_location/swe_bench_lite/test/gt_location_with_support.jsonl"

# Change keys to compare (comma-separated).
# Core keys: edited_entities,added_entities
# Extended: edited_entities,added_entities,edited_modules,added_modules
keys="edited_entities,added_entities"

# Number of differing instances to print in full detail
show_sample=5

python LocAgent/util/benchmark/compare_file_changes.py \
    --old        "$old_file" \
    --new        "$new_file" \
    --keys       "$keys" \
    --show_sample "$show_sample"
