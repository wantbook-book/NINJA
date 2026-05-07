import argparse
import json
import os
import shutil
import time
from pathlib import Path
import subprocess
import threading
import torch.multiprocessing as mp
import os.path as osp
from queue import Empty
from tqdm import tqdm
# from dependency_graph.build_graph import build_graph, VERSION
from util.benchmark.git_repo_manager import remove_repo_worktree
from util.benchmark.setup_repo import setup_repo_worktree
from util.dataset_utils import dataset_spec_to_name, load_benchmark_dataset
from plugins.location_tools.retriever.bm25_retriever import (
    build_code_retriever_from_repo as build_code_retriever
)


def list_folders(path):
    return [p.name for p in Path(path).iterdir() if p.is_dir()]


def append_failed_instance(instance_id, failure_file, failure_file_lock):
    with failure_file_lock:
        with open(failure_file, 'a', encoding='utf-8') as f:
            f.write(f"{instance_id}\n")


def report_progress(progress_queue, instance_id, status):
    if progress_queue is not None:
        progress_queue.put({
            'instance_id': instance_id,
            'status': status,
        })


def monitor_progress(progress_queue, total, desc):
    status_counts = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
    }
    with tqdm(total=total, desc=desc, unit='instance') as pbar:
        completed = 0
        while completed < total:
            try:
                event = progress_queue.get(timeout=0.2)
            except Empty:
                continue
            status = event.get('status', 'unknown')
            if status not in status_counts:
                status_counts[status] = 0
            status_counts[status] += 1
            completed += 1
            pbar.update(1)
            pbar.set_postfix(
                processed=status_counts.get('processed', 0),
                skipped=status_counts.get('skipped', 0),
                failed=status_counts.get('failed', 0),
            )


def run(rank, repo_queue, repo_path, out_path,
        download_repo=False, instance_data=None, similarity_top_k=10,
        failure_file=None, failure_file_lock=None, progress_queue=None):
    while True:
        try:
            repo_name = repo_queue.get_nowait()
        except Exception:
            # Queue is empty
            break

        output_file = osp.join(out_path, repo_name)
        if osp.exists(output_file):
            # print(f'[{rank}] {repo_name} already processed, skipping.')
            report_progress(progress_queue, repo_name, 'skipped')
            continue

        if download_repo:
            # use a shared base dir for cached repos and worktrees
            repo_base_dir = str(repo_path)
            os.makedirs(repo_base_dir, exist_ok=True)
            repo_source_dir = None
            worktree_dir = None
            try:
                repo_source_dir, worktree_dir = setup_repo_worktree(
                    instance_data=instance_data[repo_name],
                    repo_base_dir=repo_base_dir,
                    dataset=None,
                )
                repo_dir = worktree_dir
            except subprocess.CalledProcessError as e:
                print(f'[{rank}] Error preparing repo {repo_name}: {e}')
                if failure_file and failure_file_lock:
                    append_failed_instance(repo_name, failure_file, failure_file_lock)
                report_progress(progress_queue, repo_name, 'failed')
                continue
        else:
            repo_dir = osp.join(repo_path, repo_name)
            repo_source_dir = None
            worktree_dir = None

        print(f'[{rank}] Start process {repo_name}')
        try:
            retriever = build_code_retriever(repo_dir, persist_path=output_file,
                                         similarity_top_k=similarity_top_k)
            # G = build_graph(repo_dir, global_import=True)
            # with open(output_file, 'wb') as f:
            #     pickle.dump(G, f)
            print(f'[{rank}] Processed {repo_name}')
            report_progress(progress_queue, repo_name, 'processed')
        except Exception as e:
            print(f'[{rank}] Error processing {repo_name}: {e}')
            if osp.isdir(output_file):
                shutil.rmtree(output_file, ignore_errors=True)
            elif osp.exists(output_file):
                os.remove(output_file)
            if failure_file and failure_file_lock:
                append_failed_instance(repo_name, failure_file, failure_file_lock)
            report_progress(progress_queue, repo_name, 'failed')
        finally:
            if worktree_dir:
                try:
                    remove_repo_worktree(repo_source_dir, worktree_dir)
                except subprocess.CalledProcessError as e:
                    print(f'[{rank}] Error removing worktree {worktree_dir}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="czlll/SWE-bench_Lite",
        help=(
            "Hugging Face dataset name, a local dataset directory, or a local "
            "data file such as .parquet/.json/.jsonl/.csv."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--num_processes', type=int, default=30)
    parser.add_argument('--download_repo', action='store_true', 
                        help='Whether to download the codebase to `repo_path` before indexing. When base_commit is missing, the cloned repo current checkout is copied and indexed.')
    parser.add_argument('--repo_path', type=str, default='playground/build_graph', 
                        help='When --download_repo is set, use this shared base dir for cloned repos and worktrees; otherwise treat it as the directory containing local repos.')
    parser.add_argument('--index_dir', type=str, default='index_data', 
                        help='The base directory where the generated graph index will be saved.')
    parser.add_argument('--instance_id_path', type=str, default='', 
                        help='Path to a file containing a list of selected instance IDs.')
    args = parser.parse_args()

    
    dataset_name = dataset_spec_to_name(args.dataset)
    args.index_dir = f'{args.index_dir}/{dataset_name}/BM25_index/'
    os.makedirs(args.index_dir, exist_ok=True)
    failure_file = osp.join(args.index_dir, 'failed_instance_ids.txt')
    with open(failure_file, 'w', encoding='utf-8') as f:
        f.write('')
        
    # load selected repo instance id and instance_data
    if args.download_repo:
        selected_instance_data = {}
        bench_data = load_benchmark_dataset(args.dataset, args.split)
        if args.instance_id_path and osp.exists(args.instance_id_path):
            with open(args.instance_id_path, 'r') as f:
                repo_folders = json.loads(f.read())
            for instance in bench_data:
                if instance['instance_id'] in repo_folders:
                    selected_instance_data[instance['instance_id']] = instance
        else:
            repo_folders = []
            for instance in bench_data:
                repo_folders.append(instance['instance_id'])
                selected_instance_data[instance['instance_id']] = instance
    else:
        if args.instance_id_path and osp.exists(args.instance_id_path):
            with open(args.instance_id_path, 'r') as f:
                repo_folders = json.loads(f.read())
        else:
            repo_folders = list_folders(args.repo_path)
        selected_instance_data = None

    os.makedirs(args.repo_path, exist_ok=True)

    # Create a shared queue and add repositories to it
    manager = mp.Manager()
    queue = manager.Queue()
    failure_file_lock = manager.Lock()
    progress_queue = manager.Queue()
    for repo in repo_folders:
        queue.put(repo)

    start_time = time.time()

    total_repos = len(repo_folders)
    progress_thread = None
    if total_repos > 0:
        progress_thread = threading.Thread(
            target=monitor_progress,
            args=(progress_queue, total_repos, 'Building BM25 indexes'),
            daemon=True,
        )
        progress_thread.start()

    # Start multiprocessing with a global queue
    if total_repos > 0:
        mp.spawn(
            run,
            nprocs=args.num_processes,
            args=(queue, args.repo_path, args.index_dir,
                  args.download_repo, selected_instance_data, 10,
                  failure_file, failure_file_lock, progress_queue),
            join=True
        )
        if progress_thread is not None:
            progress_thread.join()
    else:
        print('No repositories to process.')

    end_time = time.time()
    print(f'Total Execution time = {end_time - start_time:.3f}s')
    print(f'Failed instance ids are recorded at: {failure_file}')
