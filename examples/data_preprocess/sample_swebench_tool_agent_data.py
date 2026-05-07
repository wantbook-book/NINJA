import argparse
import json
import os
import random
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


def _load_rows(parquet_path: str) -> tuple[list[dict[str, Any]], pa.Schema]:
    table = pq.read_table(parquet_path)
    return table.to_pylist(), table.schema


def _write_parquet(rows: list[dict[str, Any]], output_path: str, schema: pa.Schema) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
    else:
        table = pa.Table.from_batches([], schema=schema)
    pq.write_table(table, output_path)


def _write_json(data: Any, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_lines(lines: Iterable[str], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _write_sample_json(rows: list[dict[str, Any]], output_path: str) -> None:
    if not rows:
        return
    _write_json(rows[0], output_path)


def _read_instance_ids(path: str) -> list[str]:
    instance_ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            instance_id = line.strip()
            if instance_id:
                instance_ids.append(instance_id)
    return instance_ids


def _extra_info(row: dict[str, Any]) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    return extra_info if isinstance(extra_info, dict) else {}


def _instance_id(row: dict[str, Any]) -> str:
    value = _extra_info(row).get("instance_id", row.get("instance_id"))
    return "" if value is None else str(value)


def _repo(row: dict[str, Any]) -> str:
    value = _extra_info(row).get("repo", row.get("repo"))
    return "" if value is None else str(value)


def _normalize_ground_truth(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return [text]
            return _normalize_ground_truth(parsed)
        return [text]
    if isinstance(value, dict):
        normalized = []
        for item in value.values():
            normalized.extend(_normalize_ground_truth(item))
        return normalized
    return [str(value)]


def _ground_truth(row: dict[str, Any]) -> list[str]:
    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        return []
    return _normalize_ground_truth(reward_model.get("ground_truth"))


def _repo_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(_repo(row) for row in rows)
    return [
        {"repo": repo, "count": count}
        for repo, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _write_repo_counts(rows: list[dict[str, Any]], output_dir: str) -> None:
    count_rows = _repo_count_rows(rows)
    _write_json(
        {
            "num_repos": len(count_rows),
            "total_instances": len(rows),
            "repo_counts": count_rows,
        },
        os.path.join(output_dir, "repo_counts.json"),
    )


def _ground_truth_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt_counts = [_ground_truth_count(row) for row in rows]
    histogram = Counter(gt_counts)
    by_repo: dict[str, list[int]] = defaultdict(list)
    for row, count in zip(rows, gt_counts):
        by_repo[_repo(row)].append(count)

    repo_stats = []
    for repo, counts in sorted(by_repo.items(), key=lambda item: (-len(item[1]), item[0])):
        repo_stats.append(
            {
                "repo": repo,
                "instances": len(counts),
                "total_ground_truth_locations": sum(counts),
                "empty_ground_truth_count": sum(1 for count in counts if count == 0),
                "min_ground_truth_locations": min(counts) if counts else 0,
                "max_ground_truth_locations": max(counts) if counts else 0,
                "mean_ground_truth_locations": mean(counts) if counts else 0,
                "median_ground_truth_locations": median(counts) if counts else 0,
            }
        )

    return {
        "total_instances": len(rows),
        "total_ground_truth_locations": sum(gt_counts),
        "empty_ground_truth_count": sum(1 for count in gt_counts if count == 0),
        "non_empty_ground_truth_count": sum(1 for count in gt_counts if count > 0),
        "min_ground_truth_locations": min(gt_counts) if gt_counts else 0,
        "max_ground_truth_locations": max(gt_counts) if gt_counts else 0,
        "mean_ground_truth_locations": mean(gt_counts) if gt_counts else 0,
        "median_ground_truth_locations": median(gt_counts) if gt_counts else 0,
        "ground_truth_location_count_histogram": {
            str(count): freq for count, freq in sorted(histogram.items())
        },
        "by_repo": repo_stats,
    }


def _ground_truth_count(row: dict[str, Any]) -> int:
    return len(_ground_truth(row))


def _sample_per_repo(rows: list[dict[str, Any]], sample_per_repo: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_repo[_repo(row)].append(row)

    sampled_rows = []
    for repo in sorted(rows_by_repo):
        repo_rows = rows_by_repo[repo]
        sample_size = min(sample_per_repo, len(repo_rows))
        sampled_rows.extend(rng.sample(repo_rows, sample_size))
    return sampled_rows


def _select_by_instance_ids(
    rows: list[dict[str, Any]], instance_ids: list[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    rows_by_id = {_instance_id(row): row for row in rows}
    selected_rows = []
    missing_ids = []
    seen_ids = set()
    for instance_id in instance_ids:
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)
        row = rows_by_id.get(instance_id)
        if row is None:
            missing_ids.append(instance_id)
        else:
            selected_rows.append(row)
    return selected_rows, missing_ids


def _resolve_input_parquet(input_path: str, split: str) -> str:
    if os.path.isdir(input_path):
        parquet_path = os.path.join(input_path, f"{split}.parquet")
    else:
        parquet_path = input_path
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}")
    return parquet_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample and filter data generated by swebench_tool_agent_loop.py, "
            "and export repo/ground-truth statistics."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to generated parquet file or a directory containing <split>.parquet.",
    )
    parser.add_argument("--split", default="train", help="Split name used when --input is a directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for sampled data and statistics.")
    parser.add_argument("--sample_per_repo", type=int, default=2, help="Random samples per repo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for per-repo sampling.")
    parser.add_argument(
        "--instance_id_file",
        default=None,
        help="Optional txt file containing one instance_id per line to extract exactly those rows.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parquet_path = _resolve_input_parquet(args.input, args.split)
    rows, schema = _load_rows(parquet_path)

    _write_repo_counts(rows, args.output_dir)
    _write_json(_ground_truth_stats(rows), os.path.join(args.output_dir, "ground_truth_stats.json"))

    sampled_rows = _sample_per_repo(rows, args.sample_per_repo, args.seed)
    _write_parquet(sampled_rows, os.path.join(args.output_dir, "sampled_per_repo.parquet"), schema)
    _write_sample_json(sampled_rows, os.path.join(args.output_dir, "sampled_per_repo_sample.json"))
    _write_lines(
        [_instance_id(row) for row in sampled_rows],
        os.path.join(args.output_dir, "sampled_instance_ids.txt"),
    )

    if args.instance_id_file:
        requested_ids = _read_instance_ids(args.instance_id_file)
        selected_rows, missing_ids = _select_by_instance_ids(rows, requested_ids)
        _write_parquet(selected_rows, os.path.join(args.output_dir, "selected_by_instance_ids.parquet"), schema)
        _write_sample_json(selected_rows, os.path.join(args.output_dir, "selected_by_instance_ids_sample.json"))
        _write_lines(
            [_instance_id(row) for row in selected_rows],
            os.path.join(args.output_dir, "selected_instance_ids.txt"),
        )
        _write_lines(missing_ids, os.path.join(args.output_dir, "missing_instance_ids.txt"))

    summary = {
        "input_parquet": parquet_path,
        "output_dir": args.output_dir,
        "total_instances": len(rows),
        "num_repos": len(_repo_count_rows(rows)),
        "sample_per_repo": args.sample_per_repo,
        "sampled_instances": len(sampled_rows),
        "instance_id_file": args.instance_id_file,
    }
    if args.instance_id_file:
        summary["selected_instances"] = len(selected_rows)
        summary["missing_instance_ids"] = len(missing_ids)
    _write_json(summary, os.path.join(args.output_dir, "summary.json"))

    print(f"Loaded {len(rows)} rows from {parquet_path}")
    print(f"Wrote sampled data and stats to {args.output_dir}")


if __name__ == "__main__":
    main()
