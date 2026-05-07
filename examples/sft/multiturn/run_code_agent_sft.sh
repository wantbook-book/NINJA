#!/bin/bash
# Multi-turn tool-calling SFT training for code localization agents (Inspector / Navigator).
#
# Data is produced by:
#   python -m evaluation.build_inspector_sft_data  --input ... --output-dir $DATA_DIR
#   python -m evaluation.build_navigator_sft_data  --input ... --output-dir $DATA_DIR
#
# The output parquet contains a `messages` column with multi-turn conversations
# (user / assistant / tool roles, including tool_calls).
#
# Usage:
#   bash examples/sft/multiturn/run_code_agent_sft.sh

# from transformers import AutoTokenizer
# import pandas as pd

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
# df = pd.read_parquet("train.parquet")

# def count_tokens(messages):
#     text = tokenizer.apply_chat_template(messages, tokenize=False)
#     return len(tokenizer.encode(text))

# lens = df["messages"].apply(count_tokens)
# print(lens.describe(percentiles=[.5, .75, .9, .95, .99]))
# print(f"Ratio exceeding 16384: {(lens > 16384).mean():.1%}")
# print(f"Ratio exceeding 32768: {(lens > 32768).mean():.1%}")

set -xeuo pipefail

export SWANLAB_API_KEY=${SWANLAB_API_KEY:-your_key_here}

# ============================================================================
#  Cluster / Hardware
# ============================================================================
NNODES=1                          # number of nodes
NPROC_PER_NODE=8                  # GPUs per node
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# ============================================================================
#  Model
# ============================================================================
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct   # HuggingFace model name or local path

# ============================================================================
#  Data
# ============================================================================
TRAIN_FILES=$HOME/data/code_agent_sft/train.parquet
VAL_FILES=$HOME/data/code_agent_sft/val.parquet

# ============================================================================
#  Output / Logging
# ============================================================================
PROJECT_NAME=code-agent-sft
EXPERIMENT_NAME=code-agent-sft-qwen2.5-7b
SAVE_DIR=./checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}

# ============================================================================
#  Training Hyperparameters
# ============================================================================
TOTAL_EPOCHS=3                    # number of full passes over the dataset
MAX_TRAINING_STEPS=null           # set to an integer to cap total steps (overrides epochs)
TRAIN_BATCH_SIZE=128              # global batch size across all GPUs
MICRO_BATCH_SIZE=2                # per-GPU micro batch size
MAX_LENGTH=8192                   # max sequence length (tokens); set based on your data's P95
MAX_TOKEN_LEN_PER_GPU=16384       # dynamic batching: max tokens packed per GPU per step

LR=1e-5                          # learning rate
LR_SCHEDULER=cosine               # "cosine" or "constant"
LR_WARMUP_STEPS_RATIO=0.03       # fraction of total steps used for warmup
MIN_LR_RATIO=0.1                 # minimum lr ratio at end of cosine decay
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0

SEED=42

# ============================================================================
#  Sequence Parallel
# ============================================================================
SP_SIZE=2                         # Ulysses sequence parallel size; 1 = disabled

# ============================================================================
#  FSDP Engine
# ============================================================================
FSDP_STRATEGY=fsdp                # "fsdp" or "fsdp2"
DTYPE=bfloat16                    # mixed-precision param dtype: bfloat16 | float16
PARAM_OFFLOAD=false               # offload params to CPU (saves GPU memory, slower)
OPTIMIZER_OFFLOAD=false           # offload optimizer states to CPU
RESHARD_AFTER_FORWARD=true        # trade compute for memory

# ============================================================================
#  Checkpointing & Evaluation
# ============================================================================
SAVE_FREQ=500                     # save checkpoint every N steps (-1 = only at end)
TEST_FREQ=200                     # run validation every N steps (-1 = disabled)
MAX_CKPT_TO_KEEP=3                # max checkpoints to keep on disk (null = keep all)

# ============================================================================
#  Launch
# ============================================================================
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m verl.trainer.sft_trainer \
    \
    model.path=$MODEL_PATH \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.use_remove_padding=true \
    \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.messages_key=messages \
    data.tools_key=tools \
    data.max_length=$MAX_LENGTH \
    data.pad_mode=no_padding \
    data.truncation=right \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_token_len_per_gpu=$MAX_TOKEN_LEN_PER_GPU \
    data.use_dynamic_bsz=true \
    data.ignore_input_ids_mismatch=true \
    data.num_workers=8 \
    \
    optim.lr=$LR \
    optim.lr_scheduler_type=$LR_SCHEDULER \
    optim.lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO \
    optim.min_lr_ratio=$MIN_LR_RATIO \
    optim.weight_decay=$WEIGHT_DECAY \
    optim.clip_grad=$GRAD_CLIP \
    \
    engine.dtype=$DTYPE \
    engine.param_offload=$PARAM_OFFLOAD \
    engine.optimizer_offload=$OPTIMIZER_OFFLOAD \
    engine.reshard_after_forward=$RESHARD_AFTER_FORWARD \
    engine.strategy=$FSDP_STRATEGY \
    \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$MAX_TRAINING_STEPS \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger='[console,swanlab]' \
    trainer.seed=$SEED \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.max_ckpt_to_keep=$MAX_CKPT_TO_KEEP \
    trainer.balance_batch=true \
    trainer.resume_mode=auto \
    \
    engine.ulysses_sequence_parallel_size=$SP_SIZE \
