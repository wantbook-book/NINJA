#!/bin/bash
set -ex

# vllm serve --config vllm.yaml
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# model_path="Qwen/Qwen3-30B-A3B-Instruct-2507"
model_path="Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
# served_model_name="ft-qwen3-8B-128K"
served_model_name="Qwen3-80B"
# vllm serve $model_path --port 8000 --host 0.0.0.0 --served-model-name $served_model_name --max-model-len 32K -tp 2 -dp 4 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
# vllm serve $model_path --port 8000 -tp 2 -dp 4 --max-model-len 262144 --served-model-name $served_model_name --reasoning-parser qwen3 --language-model-only --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192
vllm serve $model_path --port 8000 -tp 4 -dp 1 --served-model-name $served_model_name --reasoning-parser qwen3 --gpu-memory-utilization 0.8 --max-num-batched-tokens 8192 --max_num_seqs 512
