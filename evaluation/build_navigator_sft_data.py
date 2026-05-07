"""Build Navigator SFT data from Navigator trajectories.

The script keeps only trajectories whose function-level Acc@k is correct under
``evaluation.evaluate`` semantics, with ``k = len(gt)`` for each row.
Output parquet files use verl SFT's multi-turn ``messages`` format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from evaluation.evaluate import _extract_trace_locs
from evaluation.inspector_evaluate import parse_ground_truth_value
from evaluation.sft_data_utils import (
    coerce_messages,
    has_assistant_turn,
    load_records,
    parse_jsonish,
    sanitize_messages,
    write_dataset_outputs,
)
from evaluation.utils import (
    construct_pred_func,
    extract_locs,
    extract_predicted_methods,
    locagent_acc_at_k,
    parse_gt_methods,
)


def _normalized_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    for key in ("messages", "ground_truth", "pred_methods", "trace_locs"):
        if key in normalized:
            normalized[key] = parse_jsonish(normalized[key])
    return normalized


def _ground_truth_locations(entry: dict[str, Any]) -> list[str]:
    if "ground_truth" in entry:
        locations = parse_ground_truth_value(entry.get("ground_truth"))
        if locations:
            return locations
    reward_model = entry.get("reward_model")
    if isinstance(reward_model, dict):
        locations = parse_ground_truth_value(reward_model.get("ground_truth"))
        if locations:
            return locations
    extra_info = entry.get("extra_info")
    if isinstance(extra_info, dict):
        locations = parse_ground_truth_value(extra_info.get("ground_truth"))
        if locations:
            return locations
    return []


def _extract_pred_methods(entry: dict[str, Any], messages: list[dict[str, Any]]) -> list[str]:
    pred_methods = entry.get("pred_methods")
    if isinstance(pred_methods, list):
        return [str(method) for method in pred_methods if str(method or "").strip()]

    trace_locs = entry.get("trace_locs")
    if not isinstance(trace_locs, str) or not trace_locs.strip():
        trace_locs = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                trace_locs = _extract_trace_locs(str(message.get("content", "") or ""))
                break

    found_related_locs = extract_locs(trace_locs)
    pred_funcs = construct_pred_func(found_related_locs)
    return extract_predicted_methods(pred_funcs)


def _reward_model(gt_locations: list[str]) -> dict[str, str]:
    return {
        "style": "rule",
        "ground_truth": json.dumps(gt_locations, ensure_ascii=False),
    }


def build_navigator_sft_records(
    entries: list[dict[str, Any]],
    *,
    keep_message_metadata: bool = False,
    messages_field: str = "messages",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source_index, raw_entry in enumerate(entries):
        entry = _normalized_entry(raw_entry)
        gt_locations = _ground_truth_locations(entry)
        _, gt_methods = parse_gt_methods(gt_locations)
        if not gt_methods:
            skipped["empty_function_ground_truth"] += 1
            continue

        raw_messages = coerce_messages(entry.get(messages_field))
        pred_methods = _extract_pred_methods(entry, raw_messages)
        k = 5
        if locagent_acc_at_k(gt_methods, pred_methods, k) < 1.0:
            skipped["function_acc_at_len_gt_incorrect"] += 1
            continue

        messages = sanitize_messages(raw_messages, keep_message_metadata=keep_message_metadata)
        if not messages:
            skipped["empty_messages"] += 1
            continue
        if not has_assistant_turn(messages):
            skipped["no_assistant_turn"] += 1
            continue

        record = {
            "data_source": str(entry.get("data_source") or "code_localization_navigator_sft"),
            "agent_name": "navigator",
            "ability": "code_localization",
            "messages": messages,
            "reward_model": _reward_model(gt_locations),
            "extra_info": {
                "index": len(records),
                "split": "train",
                "source_index": source_index,
                "instance_id": str(entry.get("instance_id", "") or ""),
                "repo": str(entry.get("repo", "") or ""),
                "base_commit": str(entry.get("base_commit", "") or ""),
                "ground_truth": gt_locations,
                "gt_methods": sorted(gt_methods),
                "gt_count": len(gt_locations),
                "gt_method_count": len(gt_methods),
                "pred_methods": pred_methods,
                "function_acc_k": 5,
                "search_execution_mode": str(entry.get("search_execution_mode", "") or ""),
            },
        }
        records.append(record)

    stats = {
        "total_records": len(entries),
        "selected_records": len(records),
        "skipped": dict(sorted(skipped.items())),
        "filter": "navigator function_level Acc@len(gt) == 1",
    }
    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verl SFT data from successful Navigator trajectories.")
    parser.add_argument("--input", required=True, help="Navigator trajectory/evaluation JSONL/parquet file.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.parquet, val.parquet, and metadata.")
    parser.add_argument("--val-size", type=int, default=None, help="Absolute validation sample count.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation ratio used when --val-size is unset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val splitting.")
    parser.add_argument(
        "--messages-field",
        default="messages",
        help="Input field containing the trajectory messages.",
    )
    parser.add_argument(
        "--keep-message-metadata",
        action="store_true",
        help="Keep non-chat message fields such as token_count and _message_kind.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_records(args.input)
    records, stats = build_navigator_sft_records(
        entries,
        keep_message_metadata=args.keep_message_metadata,
        messages_field=args.messages_field,
    )
    if not records:
        raise SystemExit(f"No Navigator trajectories passed filtering. Stats: {stats}")

    train_path, val_path = write_dataset_outputs(
        records,
        output_dir=args.output_dir,
        val_size=args.val_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        metadata={"input": args.input, **stats},
    )
    print(f"Loaded {stats['total_records']} Navigator trajectories")
    print(f"Selected {stats['selected_records']} trajectories with function Acc@len(gt)=1")
    print(f"Skipped: {stats['skipped']}")
    print(f"Saved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")


if __name__ == "__main__":
    main()
