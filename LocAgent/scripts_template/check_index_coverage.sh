#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
INDEX_DIR="${INDEX_DIR:-}"
GRAPH_INDEX_DIR="${GRAPH_INDEX_DIR:-}"
BM25_INDEX_DIR="${BM25_INDEX_DIR:-}"
USED_LIST="${USED_LIST:-selected_ids}"
EVAL_N_LIMIT="${EVAL_N_LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-index_check_results}"

CMD=(
  python3 check_index_coverage.py
  --dataset "$DATASET"
  --split "$SPLIT"
  --used_list "$USED_LIST"
  --eval_n_limit "$EVAL_N_LIMIT"
  --output_dir "$OUTPUT_DIR"
)

if [[ -n "$INDEX_DIR" ]]; then
  CMD+=(--index_dir "$INDEX_DIR")
else
  if [[ -n "$GRAPH_INDEX_DIR" ]]; then
    CMD+=(--graph_index_dir "$GRAPH_INDEX_DIR")
  fi
  if [[ -n "$BM25_INDEX_DIR" ]]; then
    CMD+=(--bm25_index_dir "$BM25_INDEX_DIR")
  fi
fi

"${CMD[@]}"
