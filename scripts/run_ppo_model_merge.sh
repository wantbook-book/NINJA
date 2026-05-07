#!/usr/bin/env bash
# Merge a PPO FSDP checkpoint (actor or critic) saved by verl into HuggingFace format.
#
# Usage:
#   bash scripts/run_ppo_model_merge.sh [CKPT_ROOT] [STEP|global_step_STEP|latest] [ROLE] [TARGET_DIR]
#
# Arguments:
#   CKPT_ROOT   Root checkpoint directory (contains global_step_* subdirs).
#   STEP        Step number, "global_step_N", or "latest" (reads latest_checkpointed_iteration.txt).
#   ROLE        "actor" (default) or "critic".
#   TARGET_DIR  Output directory for the merged HF model (default: CKPT_ROOT/hf_ROLE_global_step_STEP).
#
# Examples:
#   bash scripts/run_ppo_model_merge.sh
#   bash scripts/run_ppo_model_merge.sh /path/to/code_agent/outputs/inspector/checkpoints 500
#   bash scripts/run_ppo_model_merge.sh /path/to/code_agent/outputs/inspector/checkpoints 500 critic
#   bash scripts/run_ppo_model_merge.sh /path/to/code_agent/outputs/inspector/checkpoints latest actor ./merged_hf/actor
#
# Extra merger options via EXTRA_MERGE_ARGS:
#   EXTRA_MERGE_ARGS="--hf_upload_path your-org/your-model --private" bash scripts/run_ppo_model_merge.sh

set -euo pipefail

CKPT_ROOT=${1:-./checkpoints}
STEP_ARG=${2:-latest}
ROLE=${3:-actor}
TARGET_DIR_ARG=${4:-}

# ---- Validate role ----
if [[ "$ROLE" != "actor" && "$ROLE" != "critic" ]]; then
  echo "ROLE must be 'actor' or 'critic', got: $ROLE" >&2
  exit 1
fi

# ---- Validate checkpoint root ----
if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "Checkpoint root does not exist: $CKPT_ROOT" >&2
  exit 1
fi

# ---- Resolve step ----
if [[ "$STEP_ARG" == "latest" ]]; then
  TRACKER_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"
  if [[ ! -f "$TRACKER_FILE" ]]; then
    echo "Latest checkpoint tracker not found: $TRACKER_FILE" >&2
    echo "Pass a step explicitly, e.g. bash scripts/run_ppo_model_merge.sh $CKPT_ROOT 500" >&2
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

# ---- PPO checkpoints store actor/critic in subdirectories ----
LOCAL_DIR="${CKPT_ROOT}/${STEP_DIR}/${ROLE}"
TARGET_DIR=${TARGET_DIR_ARG:-"${CKPT_ROOT}/hf_${ROLE}_global_step_${STEP}"}

# ---- Validate role subdirectory ----
if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Role directory does not exist: $LOCAL_DIR" >&2
  echo "Available contents of ${CKPT_ROOT}/${STEP_DIR}:" >&2
  ls "${CKPT_ROOT}/${STEP_DIR}" 2>/dev/null || true
  exit 1
fi

if [[ ! -f "${LOCAL_DIR}/fsdp_config.json" ]]; then
  echo "Missing ${LOCAL_DIR}/fsdp_config.json" >&2
  exit 1
fi

if ! compgen -G "${LOCAL_DIR}/model_world_size_*_rank_0.pt" > /dev/null; then
  echo "No model shard found under $LOCAL_DIR" >&2
  exit 1
fi

echo "Merging PPO ${ROLE} checkpoint:"
echo "  local_dir:  $LOCAL_DIR"
echo "  target_dir: $TARGET_DIR"

# shellcheck disable=SC2086
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$LOCAL_DIR" \
  --target_dir "$TARGET_DIR" \
  --trust-remote-code \
  ${EXTRA_MERGE_ARGS:-}
