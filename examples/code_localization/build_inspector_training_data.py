"""
Build Inspector training data from extracted sub-agent trajectories.

Takes the trajectory parquet produced by ``extract_inspector_trajectories.py``
and converts it into verl-compatible parquet format for Inspector RL training.

Filtering logic:
    - For each trajectory, check whether the Inspector's explored entry_file
      contains any ground-truth location at the *file level*.  Only trajectories
      where at least one ground-truth function resides in the explored entry_file
      are kept.
    - A per-instance cap limits the number of trajectories per instance_id to
      avoid data imbalance.

Output schema follows the tool-agent compatible preprocessing format:
    - data_source, agent_name, prompt, ability, reward_model, extra_info
    - extra_info.need_tools_kwargs, extra_info.tools_kwargs

Usage:
    python -m examples.code_localization.build_inspector_training_data \
        --trajectory_file data/code_loc/inspector_trajectories.parquet \
        --output_dir data/code_loc/inspector_training/ \
        --max_per_instance 5 \
        --val_size 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from string import Template
from typing import Any

import pandas as pd

from evaluation.multi_agent_shared import (
    PROMPTS_DIR,
    _load_text,
    augment_system_prompt_with_tool_instructions,
)
from verl.utils.reward_score.code_localization import parse_ground_truth


def _extract_file_part(location: str) -> str:
    """Extract the file part from a 'file:function' location string."""
    if ":" in location:
        return location.split(":")[0].strip()
    return location.strip()


def _has_file_level_overlap(entry_file: str, ground_truth_locs: list[str]) -> bool:
    """Check if any ground-truth location shares the same file as entry_file."""
    for loc in ground_truth_locs:
        gt_file = _extract_file_part(loc)
        if gt_file == entry_file:
            return True
    return False


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
        "tools", "tool_schemas", "sub_agent_tools.json"
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
    row: dict,
    templates: dict[str, str],
    tool_schemas: list[dict],
    agent_name: str,
) -> dict:
    """Convert a filtered trajectory row into a verl-compatible training record."""
    instance_id = str(row.get("instance_id", ""))
    problem_statement = str(row.get("problem_statement", ""))
    entry_file = str(row.get("entry_file", ""))
    ground_truth = str(row.get("ground_truth", ""))

    # Build prompt
    prompt = build_inspector_prompt(problem_statement, entry_file, templates, tool_schemas)

    # Parse ground truth for reward computation
    try:
        gt_parsed = json.loads(ground_truth)
        if not isinstance(gt_parsed, list):
            gt_parsed = [str(gt_parsed)]
    except (json.JSONDecodeError, TypeError):
        gt_parsed = [ground_truth] if ground_truth else []

    return {
        "data_source": "code_localization_inspector",
        "agent_name": agent_name,
        "prompt": prompt,
        "ability": "code_localization",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(gt_parsed),
        },
        "extra_info": {
            "index": 0,  # Set later
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


def main():
    parser = argparse.ArgumentParser(
        description="Build Inspector training data from extracted trajectories"
    )
    parser.add_argument(
        "--trajectory_file", required=True,
        help="Input trajectory parquet from extract_inspector_trajectories.py"
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for training data")
    parser.add_argument(
        "--max_per_instance", type=int, default=5,
        help="Maximum number of trajectories per instance_id"
    )
    parser.add_argument(
        "--val_size",
        type=int,
        required=True,
        help="Absolute number of samples to reserve for validation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--agent_name",
        default="code_localization",
        help="The agent_name field to write into each output sample.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)

    # Load trajectories
    df = pd.read_parquet(args.trajectory_file)
    print(f"Loaded {len(df)} trajectory records from {args.trajectory_file}")
    print(f"Unique instances: {df['instance_id'].nunique()}")

    # Step 1: Filter by file-level overlap with ground truth
    kept_indices = []
    for idx, row in df.iterrows():
        gt_locs = parse_ground_truth(row.get("ground_truth", ""))
        entry_file = str(row.get("entry_file", ""))
        if _has_file_level_overlap(entry_file, gt_locs):
            kept_indices.append(idx)

    df_filtered = df.loc[kept_indices].copy()
    print(f"After file-level ground truth filtering: {len(df_filtered)} trajectories "
          f"({len(df_filtered['instance_id'].unique())} unique instances)")

    if len(df_filtered) == 0:
        print("No trajectories passed filtering! Check ground truth format.")
        return

    # Step 2: Cap per-instance count
    if args.max_per_instance > 0:
        grouped = df_filtered.groupby("instance_id")
        capped_frames = []
        for instance_id, group in grouped:
            if len(group) > args.max_per_instance:
                group = group.sample(n=args.max_per_instance, random_state=args.seed)
            capped_frames.append(group)
        df_filtered = pd.concat(capped_frames, ignore_index=True)
        print(f"After per-instance cap ({args.max_per_instance}): {len(df_filtered)} trajectories")

    # Step 3: Build training records
    templates = load_prompt_templates()
    tool_schemas = load_tool_schemas()

    records = []
    for idx, (_, row) in enumerate(df_filtered.iterrows()):
        record = build_training_record(row.to_dict(), templates, tool_schemas, args.agent_name)
        record["extra_info"]["index"] = idx
        records.append(record)

    # Step 4: Train/val split
    if args.val_size < 0:
        raise ValueError("--val_size must be non-negative")
    if args.val_size > len(records):
        raise ValueError(
            f"--val_size ({args.val_size}) cannot exceed the number of records ({len(records)})"
        )

    indices = list(range(len(records)))
    random.shuffle(indices)
    val_indices = indices[:args.val_size]
    train_indices = indices[args.val_size:]

    train_data = [records[i] for i in train_indices]
    val_data = [records[i] for i in val_indices]

    for item in train_data:
        item["extra_info"]["split"] = "train"
    for item in val_data:
        item["extra_info"]["split"] = "test"

    # Step 5: Save
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    train_sample_path = os.path.join(args.output_dir, "train_sample.json")
    val_sample_path = os.path.join(args.output_dir, "val_sample.json")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    if train_data:
        with open(train_sample_path, "w", encoding="utf-8") as f:
            json.dump(train_data[0], f, ensure_ascii=False, indent=2)
    if val_data:
        with open(val_sample_path, "w", encoding="utf-8") as f:
            json.dump(val_data[0], f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")
    if train_data:
        print(f"Saved one training sample to {train_sample_path}")
    if val_data:
        print(f"Saved one validation sample to {val_sample_path}")

    # Summary statistics
    if train_data:
        sample = train_data[0]
        print("\nSample entry:")
        print(f"  data_source: {sample['data_source']}")
        print(f"  agent_name: {sample['agent_name']}")
        print(f"  ability: {sample['ability']}")
        print(f"  prompt messages: {len(sample['prompt'])}")
        print(f"  reward_model: {sample['reward_model']}")
        print(f"  extra_info keys: {list(sample['extra_info'].keys())}")
        print(f"  instance_id: {sample['extra_info']['instance_id']}")
        print(f"  entry_file: {sample['extra_info']['entry_file']}")

    # Count instances with training data
    train_instances = set(item["extra_info"]["instance_id"] for item in train_data)
    val_instances = set(item["extra_info"]["instance_id"] for item in val_data)
    print(f"\nUnique instances in train: {len(train_instances)}")
    print(f"Unique instances in val: {len(val_instances)}")


if __name__ == "__main__":
    main()
