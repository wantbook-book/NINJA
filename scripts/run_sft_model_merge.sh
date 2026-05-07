#!/usr/bin/env bash
# Merge an SFT FSDP checkpoint saved by verl.trainer.sft_trainer into HuggingFace format.
#
# Usage:
#   bash scripts/run_sft_model_merge.sh [CKPT_ROOT] [STEP|global_step_STEP|latest] [TARGET_DIR]
#
# Examples:
#   bash scripts/run_sft_model_merge.sh
#   bash scripts/run_sft_model_merge.sh ./checkpoints/code-agent-sft/code-agent-sft-qwen2.5-7b 500
#   bash scripts/run_sft_model_merge.sh ./checkpoints/code-agent-sft/code-agent-sft-qwen2.5-7b global_step_500 ./merged_hf
#
# You can pass extra merger options through EXTRA_MERGE_ARGS, for example:
#   EXTRA_MERGE_ARGS="--hf_upload_path your-org/your-model --private" bash scripts/run_sft_model_merge.sh

set -euo pipefail

CKPT_ROOT=${1:-./checkpoints/code-agent-sft/code-agent-sft-qwen2.5-7b}
STEP_ARG=${2:-latest}
TARGET_DIR_ARG=${3:-}

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "Checkpoint root does not exist: $CKPT_ROOT" >&2
  exit 1
fi

if [[ "$STEP_ARG" == "latest" ]]; then
  TRACKER_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"
  if [[ ! -f "$TRACKER_FILE" ]]; then
    echo "Latest checkpoint tracker not found: $TRACKER_FILE" >&2
    echo "Pass a step explicitly, e.g. bash scripts/run_sft_model_merge.sh $CKPT_ROOT 500" >&2
    exit 1
  fi
  STEP=$(tr -d '[:space:]' < "$TRACKER_FILE")
  STEP_DIR="global_step_${STEP}"
elif [[ "$STEP_ARG" == global_step_* ]]; then
  STEP_DIR="$STEP_ARG"
  STEP="${STEP_ARG#global_step_}"
else
  STEP="$STEP_ARG"
  STEP_DIR="global_step_${STEP}"
fi

LOCAL_DIR="${CKPT_ROOT}/${STEP_DIR}"
TARGET_DIR=${TARGET_DIR_ARG:-"${CKPT_ROOT}/hf_global_step_${STEP}"}

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Checkpoint step directory does not exist: $LOCAL_DIR" >&2
  exit 1
fi

if [[ ! -f "${LOCAL_DIR}/fsdp_config.json" ]]; then
  echo "Missing ${LOCAL_DIR}/fsdp_config.json; this script expects a new-format SFT FSDP checkpoint." >&2
  echo "For old checkpoints, use scripts/legacy_model_merger.py with --local_dir pointing to the global_step directory." >&2
  exit 1
fi

if ! compgen -G "${LOCAL_DIR}/model_world_size_*_rank_0.pt" > /dev/null; then
  echo "No model shard found under $LOCAL_DIR" >&2
  exit 1
fi

echo "Merging SFT checkpoint:"
echo "  local_dir:  $LOCAL_DIR"
echo "  target_dir: $TARGET_DIR"

# shellcheck disable=SC2086
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR" \
  --trust-remote-code \
  ${EXTRA_MERGE_ARGS:-}
