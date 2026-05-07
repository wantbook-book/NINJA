python scripts/legacy_model_merger.py merge \
  --backend fsdp \
  --local_dir /path/to/checkpoints/.../actor \
  --target_dir /path/to/merged_hf_model

# python -m verl.model_merger merge \
#     --backend megatron \
#     --tie-word-embedding \
#     --local_dir checkpoints/verl_megatron_gsm8k_examples/qwen2_5_0b5_megatron_saveload/global_step_1/actor \
#     --target_dir /path/to/merged_hf_model