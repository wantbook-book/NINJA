#!/bin/bash
# Phase 1: Train Inspector with DAPO (with pre-extracted training data)
# Inspector generates via local rollout engine.
# Uses DAPO (GRPO group-relative advantage, no critic, asymmetric clipping).
#
# Prerequisites:
# 1. Run extract_inspector_trajectories.py to get sub-agent trajectories
# 2. Run build_inspector_training_data.py to build training data with entry_file
# 3. Set TRAIN_FILE / TEST_FILE to the output parquet files
# 4. No remote vLLM server needed (entry_file is pre-specified in data)
#
# Environment variables:
#    - PROJECT_FILE_LOC: directory with {instance_id}.json repo structure files
#    - GRAPH_INDEX_DIR: directory with {instance_id}.pkl dependency graph files

set -x
ulimit -n 65535

project_name='code_localization_rl'
exp_name='inspector_dapo_training'

# Algorithm
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10

train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=8
train_prompt_mini_bsz=1

# Multi-turn agent settings
length_penalty_coef=1.0
top_k_predictions=5
reward_metric=${reward_metric:-dice}
reward_acc_at_k=${reward_acc_at_k:-null}
reward_acc_at_k_from_gt=${reward_acc_at_k_from_gt:-false}
agent_loop_num_workers=4

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-2}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/code_loc/inspector_training/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/code_loc/inspector_training/val.parquet"}
TRAJECTORY_SAVE_DIR=${TRAJECTORY_SAVE_DIR:-"${RAY_DATA_HOME}/trajectories/${project_name}/${exp_name}"}

PROJECT_DIR="${WORKING_DIR}"
CONFIG_PATH="${PROJECT_DIR}/examples/code_localization/config"

# Performance
temperature=1.0
top_p=1.0
top_k=-1
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
offload=True
gen_tp=2
MICRO_BATCH_SIZE=1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

ray job submit --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='code_loc_grpo' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.return_raw_chat=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.agent.num_workers=${agent_loop_num_workers} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.multi_turn.enable=true \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.training_role=inspector \
    actor_rollout_ref.rollout.multi_turn.max_inspector_turns=10 \
    actor_rollout_ref.rollout.multi_turn.length_penalty_coef=${length_penalty_coef} \
    actor_rollout_ref.rollout.multi_turn.top_k_predictions=${top_k_predictions} \
    actor_rollout_ref.rollout.multi_turn.reward_metric="${reward_metric}" \
    actor_rollout_ref.rollout.multi_turn.reward_acc_at_k="${reward_acc_at_k}" \
    actor_rollout_ref.rollout.multi_turn.reward_acc_at_k_from_gt=${reward_acc_at_k_from_gt} \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=8000 \
    actor_rollout_ref.rollout.multi_turn.inspector_tool_config_path="${PROJECT_DIR}/examples/code_localization/config/tool_config/inspector_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.trajectory_save_dir="${TRAJECTORY_SAVE_DIR}" \
    "actor_rollout_ref.rollout.multi_turn.run_id=${RUN_ID}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=code_localization \
    actor_rollout_ref.rollout.trace.backend=mlflow \
    actor_rollout_ref.rollout.trace.token2text=True \
    reward.reward_manager.source=register \
    reward.reward_manager.name=code_localization \
    trainer.logger='["console","mlflow","swanlab"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=20 \
    trainer.save_freq=50 \
    trainer.total_epochs=5 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
