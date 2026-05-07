#!/bin/bash
set -ex
parquet=""
jsonl=""
python LocAgent/util/benchmark/compare_locations.py \
    --parquet $parquet \
    --jsonl $jsonl