import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return [value]


def _normalize_entities(value: Any) -> List[str]:
    items = _normalize_list(value)
    normalized: List[str] = []
    for item in items:
        if item is None:
            continue
        normalized.append(str(item))
    return normalized


def _extract_edit_entities_from_changes(changes: Any) -> List[str]:
    if isinstance(changes, dict):
        return _normalize_entities(changes.get("edited_entities"))
    if isinstance(changes, list):
        merged: List[str] = []
        for entry in changes:
            if isinstance(entry, dict) and "edited_entities" in entry:
                merged.extend(_normalize_entities(entry.get("edited_entities")))
        return merged
    return []


def merge_edit_entities(file_changes: Any) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()
    for change in _normalize_list(file_changes):
        if not isinstance(change, dict):
            continue
        changes = change.get("changes")
        entities = _extract_edit_entities_from_changes(changes)
        for entity in entities:
            if entity in seen:
                continue
            seen.add(entity)
            merged.append(entity)
    return merged


def _build_parquet_index(
    parquet_path: str,
    instance_id_key: str,
    reward_model_key: str,
    ground_truth_key: str,
) -> Dict[str, List[str]]:
    df = pd.read_parquet(parquet_path)
    if instance_id_key not in df.columns:
        raise KeyError(f"Missing column '{instance_id_key}' in parquet.")
    if reward_model_key not in df.columns:
        raise KeyError(f"Missing column '{reward_model_key}' in parquet.")

    index: Dict[str, List[str]] = {}
    for row in df.to_dict(orient="records"):
        instance_id = row.get(instance_id_key)
        reward_model = row.get(reward_model_key)
        ground_truth = None
        if isinstance(reward_model, dict):
            ground_truth = reward_model.get(ground_truth_key)
        normalized = ground_truth.tolist()
        index[str(instance_id)] = normalized
    return index


def compare_locations(
    parquet_path: str,
    jsonl_path: str,
    output_path: str,
    instance_id_key: str = "instance_id",
    file_changes_key: str = "file_changes",
    reward_model_key: str = "reward_model",
    ground_truth_key: str = "ground_truth",
    include_missing: bool = True,
) -> Tuple[int, int, int]:
    parquet_index = _build_parquet_index(
        parquet_path,
        instance_id_key=instance_id_key,
        reward_model_key=reward_model_key,
        ground_truth_key=ground_truth_key,
    )

    mismatches: List[Dict[str, Any]] = []
    total = 0
    missing = 0

    for record in _iter_jsonl(jsonl_path):
        total += 1
        instance_id = str(record.get(instance_id_key))
        file_changes = record.get(file_changes_key)
        edit_entities = merge_edit_entities(file_changes)

        ground_truth = parquet_index.get(instance_id)
        if ground_truth is None:
            missing += 1
            if include_missing:
                mismatches.append(
                    {
                        "instance_id": instance_id,
                        "edited_entities": edit_entities,
                        "ground_truth": None,
                        "reason": "missing_parquet",
                    }
                )
            continue

        if set(edit_entities) != set(ground_truth):
            mismatches.append(
                {
                    "instance_id": instance_id,
                    "edited_entities": edit_entities,
                    "ground_truth": ground_truth,
                }
            )

    with open(output_path, "w", encoding="utf-8") as f:
        for item in mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return total, len(mismatches), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare edit_entities from jsonl with ground_truth in parquet.")
    parser.add_argument("--parquet", required=True, help="Input parquet file path.")
    parser.add_argument("--jsonl", required=True, help="Input jsonl file path.")
    parser.add_argument("--output", default=None, help="Output jsonl path for mismatches.")
    parser.add_argument("--instance-id-key", default="instance_id", help="Instance id field name.")
    parser.add_argument("--file-changes-key", default="file_changes", help="File changes field name in jsonl.")
    parser.add_argument("--reward-model-key", default="reward_model", help="Reward model field name in parquet.")
    parser.add_argument("--ground-truth-key", default="ground_truth", help="Ground truth field name in reward_model.")
    parser.add_argument(
        "--exclude-missing",
        action="store_true",
        help="Exclude records that are missing from the parquet index.",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        base, _ = os.path.splitext(args.jsonl)
        output_path = f"{base}_mismatches.jsonl"

    total, mismatches, missing = compare_locations(
        parquet_path=args.parquet,
        jsonl_path=args.jsonl,
        output_path=output_path,
        instance_id_key=args.instance_id_key,
        file_changes_key=args.file_changes_key,
        reward_model_key=args.reward_model_key,
        ground_truth_key=args.ground_truth_key,
        include_missing=not args.exclude_missing,
    )

    print(f"Total jsonl records: {total}")
    print(f"Mismatches saved: {mismatches}")
    print(f"Missing in parquet: {missing}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
