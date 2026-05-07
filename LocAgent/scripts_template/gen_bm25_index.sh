export PYTHONPATH=$PYTHONPATH:$(pwd)

dataset='czlll/Loc-Bench_V1'
repo_path='playground/build_graph'
index_dir='index_data'

python build_bm25_index.py \
        --dataset "$dataset" \
        --split 'test' \
        --repo_path "$repo_path" \
        --index_dir "$index_dir" \
        --num_processes 100 \
        --download_repo

# failures will be written to:
# ${index_dir}/${dataset##*/}/BM25_index/failed_instance_ids.txt
