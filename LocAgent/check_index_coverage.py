import argparse
import json
import os
from typing import Iterable

import toml
from datasets import load_dataset


def filter_dataset(dataset, filter_column: str, used_list: str):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            data = toml.load(file)
            if used_list in data:
                selected_ids = data[used_list]

                def filter_function(example):
                    return example[filter_column] in selected_ids

                return dataset.filter(filter_function)
    return dataset


def _derive_index_dirs(
    dataset_name: str,
    index_dir: str | None,
    graph_index_dir: str | None,
    bm25_index_dir: str | None,
) -> tuple[str, str]:
    if index_dir:
        dataset_suffix = dataset_name.split("/")[-1]
        if not graph_index_dir:
            graph_index_dir = os.path.join(index_dir, dataset_suffix, "graph_index_v2.3")
        if not bm25_index_dir:
            bm25_index_dir = os.path.join(index_dir, dataset_suffix, "BM25_index")

    if not graph_index_dir:
        graph_index_dir = os.environ.get("GRAPH_INDEX_DIR")
    if not bm25_index_dir:
        bm25_index_dir = os.environ.get("BM25_INDEX_DIR")

    if not graph_index_dir or not bm25_index_dir:
        raise ValueError(
            "You must provide --index_dir or both --graph_index_dir and --bm25_index_dir, "
            "or set GRAPH_INDEX_DIR and BM25_INDEX_DIR."
        )

    return os.path.abspath(graph_index_dir), os.path.abspath(bm25_index_dir)


def _write_instance_ids(instance_ids: Iterable[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for instance_id in instance_ids:
            file.write(f"{instance_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Check whether graph and BM25 indexes required by auto_search_main.py already exist."
    )
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--used_list", type=str, default="selected_ids")
    parser.add_argument("--eval_n_limit", type=int, default=0)
    parser.add_argument("--index_dir", type=str, default="")
    parser.add_argument("--graph_index_dir", type=str, default="")
    parser.add_argument("--bm25_index_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="index_check_results")
    args = parser.parse_args()

    graph_index_dir, bm25_index_dir = _derive_index_dirs(
        dataset_name=args.dataset,
        index_dir=args.index_dir or None,
        graph_index_dir=args.graph_index_dir or None,
        bm25_index_dir=args.bm25_index_dir or None,
    )

    bench_data = load_dataset(args.dataset, split=args.split)
    bench_tests = filter_dataset(bench_data, "instance_id", args.used_list)
    if args.eval_n_limit:
        eval_n_limit = min(args.eval_n_limit, len(bench_tests))
        bench_tests = bench_tests.select(range(0, eval_n_limit))

    missing_graph = []
    missing_bm25 = []
    missing_any = []
    status_rows = []

    for bug in bench_tests:
        instance_id = str(bug["instance_id"])
        graph_exists = os.path.exists(os.path.join(graph_index_dir, f"{instance_id}.pkl"))
        bm25_exists = os.path.exists(os.path.join(bm25_index_dir, instance_id, "corpus.jsonl"))

        if not graph_exists:
            missing_graph.append(instance_id)
        if not bm25_exists:
            missing_bm25.append(instance_id)
        if not graph_exists or not bm25_exists:
            missing_any.append(instance_id)

        status_rows.append(
            {
                "instance_id": instance_id,
                "graph_exists": graph_exists,
                "bm25_exists": bm25_exists,
            }
        )

    dataset_suffix = args.dataset.split("/")[-1]
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "used_list": args.used_list,
        "eval_n_limit": args.eval_n_limit,
        "graph_index_dir": graph_index_dir,
        "bm25_index_dir": bm25_index_dir,
        "total_instances": len(status_rows),
        "missing_graph_count": len(missing_graph),
        "missing_bm25_count": len(missing_bm25),
        "missing_any_count": len(missing_any),
    }

    summary_path = os.path.join(output_dir, f"{dataset_suffix}_{args.split}_summary.json")
    all_status_path = os.path.join(output_dir, f"{dataset_suffix}_{args.split}_status.jsonl")
    missing_graph_path = os.path.join(output_dir, f"{dataset_suffix}_{args.split}_missing_graph.txt")
    missing_bm25_path = os.path.join(output_dir, f"{dataset_suffix}_{args.split}_missing_bm25.txt")
    missing_any_path = os.path.join(output_dir, f"{dataset_suffix}_{args.split}_missing_any.txt")

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(all_status_path, "w", encoding="utf-8") as file:
        for row in status_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_instance_ids(missing_graph, missing_graph_path)
    _write_instance_ids(missing_bm25, missing_bm25_path)
    _write_instance_ids(missing_any, missing_any_path)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Missing graph instance ids written to: {missing_graph_path}")
    print(f"Missing BM25 instance ids written to: {missing_bm25_path}")
    print(f"Missing any-index instance ids written to: {missing_any_path}")


if __name__ == "__main__":
    main()
