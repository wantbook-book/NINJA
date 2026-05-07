export PYTHONPATH=$PYTHONPATH:$(pwd)

repo_path='playground/build_graph'
index_dir='index_data'

# generate graph index for SWE-bench_Lite
python dependency_graph/batch_build_graph.py \
        --dataset 'czlll/SWE-bench_Lite' \
        --split 'test' \
        --repo_path "$repo_path" \
        --index_dir "$index_dir" \
        --num_processes 50 \
        --download_repo

# generate graph index for Loc-Bench
python dependency_graph/batch_build_graph.py \
        --dataset 'czlll/Loc-Bench_V1' \
        --split 'test' \
        --repo_path "$repo_path" \
        --index_dir "$index_dir" \
        --num_processes 50 \
        --download_repo

# failures will be written to:
# ${index_dir}/{dataset_name}/graph_index_v2.3/failed_instance_ids.txt
