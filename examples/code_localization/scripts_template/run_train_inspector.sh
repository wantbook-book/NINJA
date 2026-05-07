#!/bin/bash
# Phase 1: Train Inspector with PPO (with pre-extracted training data)
# Inspector generates via local rollout engine.
# Uses PPO (GAE advantage estimation + critic value function).
#
# Prerequisites:
# 1. Run extract_inspector_trajectories.py to get sub-agent trajectories
# 2. Run build_inspector_training_data.py to build training data with entry_file
# 3. Set TRAIN_DATA / VAL_DATA to the output parquet files
# 4. No remote vLLM server needed (entry_file is pre-specified in data)
#
# Environment variables:
#    - PROJECT_FILE_LOC: directory with {instance_id}.json repo structure files
#    - GRAPH_INDEX_DIR: directory with {instance_id}.pkl dependency graph files

# ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
set -x
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/code_localization/config"

# ---- User-configurable variables ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
CRITIC_MODEL_PATH="${CRITIC_MODEL_PATH:-${MODEL_PATH}}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/code_loc/inspector_training/train.parquet}"
VAL_DATA="${VAL_DATA:-$HOME/data/code_loc/inspector_training/val.parquet}"

# Training hyperparameters
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
CRITIC_LEARNING_RATE="${CRITIC_LEARNING_RATE:-1e-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LENGTH_PENALTY_COEF="${LENGTH_PENALTY_COEF:-1.0}"
TOP_K_PREDICTIONS="${TOP_K_PREDICTIONS:-5}"
REWARD_METRIC="${REWARD_METRIC:-dice}"
REWARD_ACC_AT_K="${REWARD_ACC_AT_K:-null}"
REWARD_ACC_AT_K_FROM_GT="${REWARD_ACC_AT_K_FROM_GT:-false}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
N_GPUS="${N_GPUS:-8}"
TP_SIZE="${TP_SIZE:-2}"
SAVE_FREQ="${SAVE_FREQ:-50}"
AGENT_LOOP_NUM_WORKERS="${AGENT_LOOP_NUM_WORKERS:-4}"

# PPO / GAE hyperparameters
GAE_GAMMA="${GAE_GAMMA:-1.0}"
GAE_LAMBDA="${GAE_LAMBDA:-1.0}"
KL_COEF="${KL_COEF:-0.001}"
CLIPRANGE_VALUE="${CLIPRANGE_VALUE:-0.5}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
CRITIC_PARAM_OFFLOAD="${CRITIC_PARAM_OFFLOAD:-False}"
CRITIC_OPTIMIZER_OFFLOAD="${CRITIC_OPTIMIZER_OFFLOAD:-False}"

# Overlong buffer (soft penalty for responses in [max_response_length, max_response_length + buffer_len])
ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-True}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/code_agent/outputs/inspector/checkpoints}"
TRAJECTORY_SAVE_DIR="${TRAJECTORY_SAVE_DIR:-/path/to/code_agent/outputs/inspector/trajectories}"

NNODES=2

SP_SIZE=4

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ray job submit --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='code_loc_grpo' \
    actor_rollout_ref.rollout.name=vllm \
    algorithm.adv_estimator=gae \
    algorithm.gamma=${GAE_GAMMA} \
    algorithm.lam=${GAE_LAMBDA} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=${KL_COEF} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_LOOP_NUM_WORKERS} \
    actor_rollout_ref.rollout.data_parallel_size=$((N_GPUS * NNODES / TP_SIZE)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.model.path="${CRITIC_MODEL_PATH}" \
    critic.optim.lr=${CRITIC_LEARNING_RATE} \
    critic.optim.lr_warmup_steps_ratio=0.05 \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    critic.cliprange_value=${CLIPRANGE_VALUE} \
    critic.fsdp.param_offload=${CRITIC_PARAM_OFFLOAD} \
    critic.fsdp.optimizer_offload=${CRITIC_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.training_role=inspector \
    actor_rollout_ref.rollout.multi_turn.max_inspector_turns=10 \
    actor_rollout_ref.rollout.multi_turn.length_penalty_coef=${LENGTH_PENALTY_COEF} \
    actor_rollout_ref.rollout.multi_turn.top_k_predictions=${TOP_K_PREDICTIONS} \
    actor_rollout_ref.rollout.multi_turn.reward_metric="${REWARD_METRIC}" \
    actor_rollout_ref.rollout.multi_turn.reward_acc_at_k="${REWARD_ACC_AT_K}" \
    actor_rollout_ref.rollout.multi_turn.reward_acc_at_k_from_gt=${REWARD_ACC_AT_K_FROM_GT} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8000 \
    actor_rollout_ref.rollout.multi_turn.inspector_tool_config_path="$PROJECT_DIR/examples/code_localization/config/tool_config/inspector_tool_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=code_localization \
    trainer.critic_warmup=${CRITIC_WARMUP} \
    trainer.project_name='code_localization_rl' \
    trainer.experiment_name='inspector_ppo_training' \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=20 \
    trainer.logger='["console","mlflow","swanlab"]' \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    reward.reward_manager.source=register \
    reward.reward_manager.name=code_localization \
    reward.reward_kwargs.max_resp_len=4096 \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE} \
    critic.ulysses_sequence_parallel_size=${SP_SIZE} \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    actor_rollout_ref.rollout.multi_turn.trajectory_save_dir="${TRAJECTORY_SAVE_DIR}" \
    "actor_rollout_ref.rollout.multi_turn.run_id=${RUN_ID}"
