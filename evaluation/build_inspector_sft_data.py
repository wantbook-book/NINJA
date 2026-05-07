"""Build Inspector SFT data from Inspector trajectories.

The script keeps only trajectories whose function-level Acc@k is correct under
``evaluation.inspector_evaluate`` semantics, with ``k = len(gt_locations)`` for
each row. Output parquet files use verl SFT's multi-turn ``messages`` format.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from evaluation.inspector_evaluate import (
    acc_at_k,
    dedupe_preserve_order,
    extract_ground_truth_locations,
    extract_predicted_locations,
)
from evaluation.sft_data_utils import (
    coerce_messages,
    has_assistant_turn,
    load_records,
    parse_jsonish,
    sanitize_messages,
    write_dataset_outputs,
)


def _normalized_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    for key in (
        "messages",
        "ground_truth",
        "gt_locations",
        "pred_locations",
        "predicted_locations",
        "suspicious",
        "confirmed_suspicious",
        "final_result",
    ):
        if key in normalized:
            normalized[key] = parse_jsonish(normalized[key])
    if "suspicious" not in normalized and "confirmed_suspicious" in normalized:
        normalized["suspicious"] = normalized["confirmed_suspicious"]
    return normalized


def _reward_model(gt_locations: list[str]) -> dict[str, str]:
    return {
        "style": "rule",
        "ground_truth": json.dumps(gt_locations, ensure_ascii=False),
    }


def build_inspector_sft_records(
    entries: list[dict[str, Any]],
    *,
    keep_message_metadata: bool = False,
    messages_field: str = "messages",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source_index, raw_entry in enumerate(entries):
        entry = _normalized_entry(raw_entry)
        gt_locations = dedupe_preserve_order(extract_ground_truth_locations(entry))
        if not gt_locations:
            skipped["empty_ground_truth"] += 1
            continue

        k = len(gt_locations)
        pred_locations = extract_predicted_locations(entry)
        if acc_at_k(gt_locations, pred_locations, k) < 1.0:
            skipped["function_acc_at_len_gt_incorrect"] += 1
            continue

        messages = sanitize_messages(
            coerce_messages(entry.get(messages_field)),
            keep_message_metadata=keep_message_metadata,
        )
        if not messages:
            skipped["empty_messages"] += 1
            continue
        if not has_assistant_turn(messages):
            skipped["no_assistant_turn"] += 1
            continue

        record = {
            "data_source": str(entry.get("data_source") or "code_localization_inspector_sft"),
            "agent_name": "inspector",
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
                "entry_file": str(entry.get("entry_file", "") or ""),
                "ground_truth": gt_locations,
                "pred_locations": pred_locations,
                "function_acc_k": k,
            },
        }
        records.append(record)

    stats = {
        "total_records": len(entries),
        "selected_records": len(records),
        "skipped": dict(sorted(skipped.items())),
        "filter": "inspector function_level Acc@len(gt_locations) == 1",
    }
    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verl SFT data from successful Inspector trajectories.")
    parser.add_argument("--input", required=True, help="Inspector trajectory JSONL/parquet file.")
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
    records, stats = build_inspector_sft_records(
        entries,
        keep_message_metadata=args.keep_message_metadata,
        messages_field=args.messages_field,
    )
    if not records:
        raise SystemExit(f"No Inspector trajectories passed filtering. Stats: {stats}")

    train_path, val_path = write_dataset_outputs(
        records,
        output_dir=args.output_dir,
        val_size=args.val_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        metadata={"input": args.input, **stats},
    )
    print(f"Loaded {stats['total_records']} Inspector trajectories")
    print(f"Selected {stats['selected_records']} trajectories with function Acc@len(gt)=1")
    print(f"Skipped: {stats['skipped']}")
    print(f"Saved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")


if __name__ == "__main__":
    main()
