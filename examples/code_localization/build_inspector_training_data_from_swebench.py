"""Build Inspector training data from preprocessed SWE-bench tool-agent data.

This script mirrors the output schema of ``build_inspector_training_data.py``,
but it does not require extracted Inspector trajectories.  It reads the parquet
files produced by ``examples.data_preprocess.swebench_tool_agent_loop_reposearch``
and creates one Inspector sample for each unique ground-truth file in an
instance:

    - entry_file is the unique file path from the ground-truth locations.
    - reward_model.ground_truth contains only ground-truth functions in that
      entry_file.

Usage:
    python -m examples.code_localization.build_inspector_training_data_from_swebench \
        --input_file data/swebench/test.parquet \
        --output_dir data/code_loc/inspector_from_swebench \
        --val_size 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from string import Template
from typing import Any

import numpy as np
import pandas as pd

from evaluation.multi_agent_shared import (
    PROMPTS_DIR,
    _load_text,
    augment_system_prompt_with_tool_instructions,
)


def _extract_file_part(location: str) -> str:
    """Extract the file part from a 'file:function' location string."""
    if ":" in location:
        return location.split(":")[0].strip()
    return location.strip()


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_location_list(value: Any) -> list[str]:
    """Normalize a parquet cell into a list of function location strings."""
    if value is None or _is_missing_scalar(value):
        return []
    if isinstance(value, np.ndarray):
        return normalize_location_list(value.tolist())
    if isinstance(value, list | tuple | set):
        return [loc for loc in (str(item).strip() for item in value) if loc]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text:
                return [loc.strip() for loc in text.splitlines() if loc.strip()]
            if "," in text:
                return [loc.strip() for loc in text.split(",") if loc.strip()]
            return [text]
        return normalize_location_list(parsed)
    return [str(value).strip()] if str(value).strip() else []


def group_locations_by_file(locations: list[str]) -> dict[str, list[str]]:
    """Group function locations by their file path."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for location in locations:
        file_path = _extract_file_part(location)
        if file_path:
            grouped[file_path].append(location)
    return {
        file_path: _dedupe_preserve_order(file_locations)
        for file_path, file_locations in grouped.items()
    }


def load_prompt_templates() -> dict[str, str]:
    """Load Inspector prompt templates from prompts/multi_agent/ directory."""
    return {
        "insp_system": _load_text(os.path.join(PROMPTS_DIR, "sub_agent_system_prompt.txt")),
        "insp_user": _load_text(os.path.join(PROMPTS_DIR, "sub_agent_user_prompt.txt")),
    }


def load_tool_schemas() -> list[dict]:
    """Load sub-agent tool schemas."""
    tool_schemas_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tools",
        "tool_schemas",
        "sub_agent_tools.json",
    )
    try:
        with open(tool_schemas_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _build_inspector_tools_kwargs(
    instance_id: str,
    tool_schemas: list[dict],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build per-tool create kwargs for Inspector tool initialization."""
    create_kwargs = {"instance_id": instance_id}
    tools_kwargs = {}
    for schema in tool_schemas:
        tool_name = schema.get("function", {}).get("name")
        if tool_name:
            tools_kwargs[tool_name] = {"create_kwargs": create_kwargs}
    return tools_kwargs


def build_inspector_prompt(
    problem_statement: str,
    entry_file: str,
    templates: dict[str, str],
    tool_schemas: list[dict],
) -> list[dict]:
    """Build Inspector prompt messages using loaded templates."""
    system_prompt = augment_system_prompt_with_tool_instructions(
        templates["insp_system"], tool_schemas
    )
    user_content = Template(templates["insp_user"]).safe_substitute(
        problem_statement=problem_statement,
        entry_file=entry_file,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_training_record(
    row: dict[str, Any],
    templates: dict[str, str],
    tool_schemas: list[dict],
    agent_name: str,
) -> dict[str, Any]:
    """Convert a per-file SWE-bench row into an Inspector training record."""
    instance_id = str(row.get("instance_id", ""))
    problem_statement = str(row.get("problem_statement", ""))
    entry_file = str(row.get("entry_file", ""))
    ground_truth = str(row.get("ground_truth", ""))

    try:
        gt_parsed = json.loads(ground_truth)
        if not isinstance(gt_parsed, list):
            gt_parsed = [str(gt_parsed)]
    except (json.JSONDecodeError, TypeError):
        gt_parsed = [ground_truth] if ground_truth else []

    return {
        "data_source": "code_localization_inspector",
        "agent_name": agent_name,
        "prompt": build_inspector_prompt(
            problem_statement,
            entry_file,
            templates,
            tool_schemas,
        ),
        "ability": "code_localization",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(gt_parsed),
        },
        "extra_info": {
            "index": 0,
            "split": "train",
            "instance_id": instance_id,
            "repo": str(row.get("repo", "")),
            "problem_statement": problem_statement,
            "structure": str(row.get("structure", "")),
            "ground_truth": ground_truth,
            "entry_file": entry_file,
            "patch": str(row.get("patch", "")),
            "need_tools_kwargs": True,
            "tools_kwargs": _build_inspector_tools_kwargs(instance_id, tool_schemas),
        },
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_records_from_processed_rows(
    processed_rows: Any,
    *,
    agent_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create one Inspector training record per unique GT file."""
    templates = load_prompt_templates()
    tool_schemas = load_tool_schemas()

    records = []
    skipped_instance_ids = []
    for example in processed_rows:
        reward_model = _as_dict(example.get("reward_model"))
        extra_info = _as_dict(example.get("extra_info"))
        instance_id = str(extra_info.get("instance_id") or example.get("instance_id", ""))
        ground_truth = reward_model.get("ground_truth", [])
        file_to_locations = group_locations_by_file(normalize_location_list(ground_truth))
        if not file_to_locations:
            if instance_id:
                skipped_instance_ids.append(instance_id)
            continue

        for entry_file, file_ground_truth in file_to_locations.items():
            row = {
                "instance_id": instance_id,
                "repo": extra_info.get("repo", example.get("repo", "")),
                "problem_statement": extra_info.get(
                    "problem_statement",
                    example.get("problem_statement", ""),
                ),
                "structure": extra_info.get("structure", example.get("structure", "")),
                "entry_file": entry_file,
                "ground_truth": json.dumps(file_ground_truth),
                "patch": extra_info.get("patch", example.get("patch", "")),
            }
            record = build_training_record(row, templates, tool_schemas, agent_name)
            records.append(record)

    for index, record in enumerate(records):
        record["extra_info"]["index"] = index

    return records, skipped_instance_ids


build_records_from_raw_dataset = build_records_from_processed_rows


def split_records(
    records: list[dict[str, Any]],
    *,
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size < 0:
        raise ValueError("--val_size must be non-negative")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    val_indices = set(indices[: min(val_size, len(records))])

    train_data = []
    val_data = []
    for index, record in enumerate(records):
        target = val_data if index in val_indices else train_data
        item = dict(record)
        item["extra_info"] = dict(record["extra_info"])
        item["extra_info"]["split"] = "test" if index in val_indices else "train"
        target.append(item)

    return train_data, val_data


def save_records(
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    pd.DataFrame(train_data).to_parquet(train_path, index=False)
    pd.DataFrame(val_data).to_parquet(val_path, index=False)

    if train_data:
        train_sample_path = os.path.join(output_dir, "train_sample.json")
        with open(train_sample_path, "w", encoding="utf-8") as f:
            json.dump(train_data[0], f, ensure_ascii=False, indent=2)
    if val_data:
        val_sample_path = os.path.join(output_dir, "val_sample.json")
        with open(val_sample_path, "w", encoding="utf-8") as f:
            json.dump(val_data[0], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Inspector training data from preprocessed SWE-bench tool-agent parquet"
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Parquet file produced by swebench_tool_agent_loop_reposearch.py.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--val_size",
        type=int,
        required=True,
        help="Absolute number of samples to reserve for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--agent_name",
        default="code_localization",
        help="The agent_name field to write into each output sample.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_file)
    records, skipped_instance_ids = build_records_from_processed_rows(
        df.to_dict("records"),
        agent_name=args.agent_name,
    )
    train_data, val_data = split_records(records, val_size=args.val_size, seed=args.seed)
    save_records(train_data, val_data, args.output_dir)

    skipped_path = os.path.join(args.output_dir, "skipped_instance_ids.txt")
    with open(skipped_path, "w", encoding="utf-8") as f:
        for instance_id in skipped_instance_ids:
            f.write(f"{instance_id}\n")

    print(f"Loaded {len(df)} preprocessed tool-agent rows from {args.input_file}")
    print(f"Built {len(records)} Inspector samples")
    print(
        f"Saved {len(train_data)} training samples to "
        f"{os.path.join(args.output_dir, 'train.parquet')}"
    )
    print(
        f"Saved {len(val_data)} validation samples to "
        f"{os.path.join(args.output_dir, 'val.parquet')}"
    )
    print(f"Skipped {len(skipped_instance_ids)} raw instances without GT function locations")


if __name__ == "__main__":
    main()
