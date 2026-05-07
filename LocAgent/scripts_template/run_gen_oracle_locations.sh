repo_base_dir=""
dataset="czlll/SWE-bench_Lite"
output_dir="/code_agent/dataset"
max_edit_file_num=1
split=test
python LocAgent/util/benchmark/gen_oracle_locations.py \
    --repo_base_dir $repo_base_dir \
    --dataset $dataset \
    --output_dir $output_dir \
    --max_edit_file_num $max_edit_file_num \
    --split $split


