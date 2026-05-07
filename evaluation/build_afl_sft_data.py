"""Build SFT data from RepoDeepSearch AFL trajectories.

Supports two trajectory types:
  - **file-level**: output of ``AFL_localize_file.py`` (``file_loc_trajs``)
  - **func-level**: output of ``AFL_localize_func.py`` (``func_loc_trajs``)

Filtering uses acc@k from ``eval_loc_acc_f1.py`` semantics (all ground-truth
items must appear in top-k predictions). k is configurable (default 5).

Usage::

    # file-level SFT data
    python -m evaluation.build_afl_sft_data \\
        --input file_loc_outputs.jsonl \\
        --gt-file evaluation/gt_verified.json \\
        --level file \\
        --output-dir output/afl_file_sft

    # func-level SFT data
    python -m evaluation.build_afl_sft_data \\
        --input func_loc_outputs.jsonl \\
        --gt-file evaluation/gt_verified.json \\
        --level func \\
        --output-dir output/afl_func_sft
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from evaluation.sft_data_utils import (
    has_assistant_turn,
    load_records,
    parse_jsonish,
    write_dataset_outputs,
)


# ---------------------------------------------------------------------------
# Ground-truth / prediction helpers (mirrors eval_loc_acc_f1.py)
# ---------------------------------------------------------------------------

def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _gt_files(entities: list[str]) -> list[str]:
    return _dedupe(e.split("::", 1)[0] for e in entities)


def _gt_entities(entities: list[str]) -> list[str]:
    return _dedupe(e for e in entities if "::" in e)


def _pred_files(record: dict[str, Any]) -> list[str]:
    return _dedupe(record.get("found_files", []))


def _pred_entities(record: dict[str, Any]) -> list[str]:
    predictions: list[str] = []
    found_related_locs = record.get("found_related_locs", {}) or {}
    if isinstance(found_related_locs, str):
        found_related_locs = parse_jsonish(found_related_locs)
    if not isinstance(found_related_locs, dict):
        return []
    for file_name in _pred_files(record):
        loc_blocks = found_related_locs.get(file_name, [])
        if isinstance(loc_blocks, str):
            loc_blocks = [loc_blocks]
        for block in loc_blocks:
            for line in str(block).splitlines():
                line = line.strip()
                if line.startswith("function:") or line.startswith("class:"):
                    _, value = line.split(":", 1)
                    value = value.strip()
                    if value:
                        predictions.append(f"{file_name}::{value}")
    return _dedupe(predictions)


def _acc_at_k(gt: list[str], pred: list[str], k: int) -> float:
    gt_set = set(gt)
    if not gt_set:
        return 0.0
    return 1.0 if gt_set <= set(pred[:k]) else 0.0


# ---------------------------------------------------------------------------
# Trajectory -> messages conversion
# ---------------------------------------------------------------------------

def _reconstruct_messages_from_trajs(trajs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct the full multi-turn conversation from AFL trajectory list.

    Each traj dict has ``prompt`` (message list snapshot) and ``response``.
    The last traj's ``prompt`` + its ``response`` gives the complete conversation.
    """
    if not trajs:
        return []
    last_traj = trajs[-1]
    prompt = last_traj.get("prompt")
    if isinstance(prompt, str):
        prompt = parse_jsonish(prompt)
    if not isinstance(prompt, list):
        return []
    messages = list(prompt)
    response = last_traj.get("response", "")
    if response and isinstance(response, str) and response.strip():
        messages.append({"role": "assistant", "content": response})
    return messages


def _reconstruct_messages_from_single_traj(traj: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct messages from a single traj dict (evaluation JSONL format)."""
    prompt = traj.get("prompt")
    if isinstance(prompt, str):
        prompt = parse_jsonish(prompt)
    if not isinstance(prompt, list):
        return []
    messages = list(prompt)
    response = traj.get("response", "")
    if response and isinstance(response, str) and response.strip():
        messages.append({"role": "assistant", "content": response})
    return messages


def _extract_messages(entry: dict[str, Any], level: str) -> list[dict[str, Any]]:
    """Extract the conversation messages from an AFL output record."""
    if level == "file":
        trajs_key = "file_loc_trajs"
    else:
        trajs_key = "func_loc_trajs"

    trajs = entry.get(trajs_key)
    if isinstance(trajs, str):
        trajs = parse_jsonish(trajs)

    if isinstance(trajs, list) and trajs:
        return _reconstruct_messages_from_trajs(trajs)

    traj_key = "file_traj" if level == "file" else "func_traj"
    traj = entry.get(traj_key)
    if isinstance(traj, str):
        traj = parse_jsonish(traj)
    if isinstance(traj, dict):
        return _reconstruct_messages_from_single_traj(traj)

    return []


# ---------------------------------------------------------------------------
# Sanitize messages (keep only role + content)
# ---------------------------------------------------------------------------

_KEEP_KEYS = ("role", "content", "tool_calls", "name", "tool_call_id")


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role", "") or "").strip()
        if not role:
            continue
        clean = {k: msg[k] for k in _KEEP_KEYS if k in msg and msg[k] is not None}
        clean["role"] = role
        if "content" not in clean:
            clean["content"] = ""
        sanitized.append(clean)
    return sanitized


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _reward_model(gt: list[str]) -> dict[str, str]:
    return {
        "style": "rule",
        "ground_truth": json.dumps(gt, ensure_ascii=False),
    }


def build_afl_sft_records(
    entries: list[dict[str, Any]],
    gt_data: dict[str, list[str]],
    *,
    level: str,
    k: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for source_index, raw_entry in enumerate(entries):
        entry = dict(raw_entry)
        for key in ("file_loc_trajs", "func_loc_trajs", "func_traj",
                     "file_traj", "found_related_locs", "found_files"):
            if key in entry and isinstance(entry[key], str):
                entry[key] = parse_jsonish(entry[key])

        instance_id = str(entry.get("instance_id", "") or "")
        if not instance_id:
            skipped["no_instance_id"] += 1
            continue

        gt_entities_raw = gt_data.get(instance_id)
        if not gt_entities_raw:
            skipped["not_in_ground_truth"] += 1
            continue

        # Compute acc@k
        if level == "file":
            gt = _gt_files(gt_entities_raw)
            pred = _pred_files(entry)
        else:
            gt = _gt_entities(gt_entities_raw)
            pred = _pred_entities(entry)

        if not gt:
            skipped["empty_ground_truth"] += 1
            continue

        if _acc_at_k(gt, pred, k) < 1.0:
            skipped[f"{level}_acc_at_{k}_incorrect"] += 1
            continue

        # Reconstruct messages
        messages = _extract_messages(entry, level)
        messages = _sanitize_messages(messages)
        if not messages:
            skipped["empty_messages"] += 1
            continue
        if not has_assistant_turn(messages):
            skipped["no_assistant_turn"] += 1
            continue

        record = {
            "data_source": f"afl_{level}_sft",
            "agent_name": f"afl_{level}",
            "ability": "code_localization",
            "messages": messages,
            "reward_model": _reward_model(gt),
            "extra_info": {
                "index": len(records),
                "split": "train",
                "source_index": source_index,
                "instance_id": instance_id,
                "repo": str(entry.get("repo", "") or ""),
                "base_commit": str(entry.get("base_commit", "") or ""),
                "ground_truth": gt,
                "predictions": pred[:k],
                "acc_at_k": k,
                "level": level,
            },
        }
        records.append(record)

    stats = {
        "total_records": len(entries),
        "selected_records": len(records),
        "skipped": dict(sorted(skipped.items())),
        "filter": f"AFL {level}_level Acc@{k} == 1",
    }
    return records, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build verl SFT data from successful AFL trajectories."
    )
    parser.add_argument(
        "--input", required=True,
        help="AFL trajectory JSONL file (output of AFL_localize_file or AFL_localize_func).",
    )
    parser.add_argument(
        "--gt-file", required=True,
        help="Ground-truth JSON file mapping instance_id -> list of 'file::entity' strings.",
    )
    parser.add_argument(
        "--level", required=True, choices=["file", "func"],
        help="Trajectory level: 'file' for file-level, 'func' for function-level.",
    )
    parser.add_argument("--k", type=int, default=5, help="k for acc@k filtering (default: 5).")
    parser.add_argument("--output-dir", required=True, help="Directory for output parquet and metadata.")
    parser.add_argument("--val-size", type=int, default=None, help="Absolute validation sample count.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation ratio when --val-size is unset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    entries = load_records(args.input)
    records, stats = build_afl_sft_records(
        entries,
        gt_data,
        level=args.level,
        k=args.k,
    )
    if not records:
        raise SystemExit(f"No AFL trajectories passed filtering. Stats: {stats}")

    train_path, val_path = write_dataset_outputs(
        records,
        output_dir=args.output_dir,
        val_size=args.val_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        metadata={"input": args.input, "gt_file": args.gt_file, "level": args.level, "k": args.k, **stats},
    )
    print(f"Loaded {stats['total_records']} AFL {args.level}-level trajectories")
    print(f"Selected {stats['selected_records']} trajectories with {args.level} Acc@{args.k}=1")
    print(f"Skipped: {stats['skipped']}")
    print(f"Saved train data to: {train_path}")
    print(f"Saved val data to: {val_path}")


if __name__ == "__main__":
    main()
