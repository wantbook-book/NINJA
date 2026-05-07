"""
Data preprocessing script for code localization RL training.

Converts code localization dataset (parquet with instance_id, problem_statement,
structure, ground_truth) into verl-compatible parquet format.

Prompts are loaded from ``prompts/multi_agent/`` to stay consistent with
the inference pipeline (``evaluation/multi_agent_inference.py``).

Usage:
    python examples/code_localization/preprocess_data.py \
        --input_file data/code_loc/raw_data.parquet \
        --output_dir data/code_loc/ \
        --train_ratio 0.9

Input parquet columns:
    - instance_id: str
    - problem_statement: str
    - structure: str (repo directory structure)
    - ground_truth: str (JSON list of "file:function" locations)

Optional columns:
    - repo: str
    - base_commit: str
    - data_source: str
"""

import argparse
import json
import os
from string import Template

import pandas as pd

from evaluation.multi_agent_shared import (
    PROMPTS_DIR,
    _load_text,
)


def load_prompt_templates() -> dict[str, str]:
    """Load prompt templates from prompts/multi_agent/ directory."""
    return {
        "nav_system": _load_text(os.path.join(PROMPTS_DIR, "main_agent_system_prompt.txt")),
        "nav_user": _load_text(os.path.join(PROMPTS_DIR, "main_agent_user_prompt.txt")),
        "insp_system": _load_text(os.path.join(PROMPTS_DIR, "sub_agent_system_prompt.txt")),
        "insp_user": _load_text(os.path.join(PROMPTS_DIR, "sub_agent_user_prompt.txt")),
    }


def build_navigator_prompt(
    problem_statement: str,
    structure: str,
    templates: dict[str, str],
) -> list[dict]:
    """Build Navigator prompt messages using the loaded templates."""
    user_content = Template(templates["nav_user"]).safe_substitute(
        problem_statement=problem_statement,
        repo_structure=structure,
    )
    return [{"role": "user", "content": user_content}]


def process_row(row: dict, templates: dict[str, str]) -> dict:
    """Convert a single data row to verl-compatible format."""
    instance_id = str(row.get("instance_id", ""))
    problem_statement = str(row.get("problem_statement", ""))
    structure = str(row.get("structure", ""))
    ground_truth = str(row.get("ground_truth", ""))

    # Build prompt using loaded templates (system prompt will be added at runtime
    # by the agent loop via augment_system_prompt_with_tool_instructions)
    prompt = build_navigator_prompt(problem_statement, structure, templates)

    # Parse ground truth for reward computation
    try:
        gt_parsed = json.loads(ground_truth)
        if not isinstance(gt_parsed, list):
            gt_parsed = [str(gt_parsed)]
    except (json.JSONDecodeError, TypeError):
        gt_parsed = [ground_truth] if ground_truth else []

    return {
        "data_source": "code_localization",
        "prompt": prompt,
        "ability": "code_localization",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(gt_parsed),
        },
        "extra_info": {
            "index": 0,  # Will be set later
            "split": "train",
            "instance_id": instance_id,
            "repo": str(row.get("repo", "")),
            "problem_statement": problem_statement,
            "structure": structure,
            "ground_truth": ground_truth,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for code localization RL training")
    parser.add_argument("--input_file", required=True, help="Input parquet/jsonl file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load input data
    if args.input_file.endswith(".parquet"):
        df = pd.read_parquet(args.input_file)
    elif args.input_file.endswith(".jsonl"):
        df = pd.read_json(args.input_file, lines=True)
    elif args.input_file.endswith(".json"):
        df = pd.read_json(args.input_file)
    else:
        raise ValueError(f"Unsupported file format: {args.input_file}")

    print(f"Loaded {len(df)} rows from {args.input_file}")
    print(f"Columns: {list(df.columns)}")

    # Load prompt templates from prompts/multi_agent/
    templates = load_prompt_templates()
    print(f"Loaded prompt templates from: {PROMPTS_DIR}")

    # Process each row
    processed = []
    for idx, row in df.iterrows():
        item = process_row(row.to_dict(), templates)
        item["extra_info"]["index"] = idx
        processed.append(item)

    # Split into train and val
    import random
    random.seed(args.seed)
    indices = list(range(len(processed)))
    random.shuffle(indices)
    split_point = int(len(indices) * args.train_ratio)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_data = [processed[i] for i in train_indices]
    val_data = [processed[i] for i in val_indices]

    # Set split field
    for item in train_data:
        item["extra_info"]["split"] = "train"
    for item in val_data:
        item["extra_info"]["split"] = "test"

    # Save as parquet
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved {len(train_data)} training samples to {train_path}")
    print(f"Saved {len(val_data)} validation samples to {val_path}")

    # Print a sample
    if train_data:
        sample = train_data[0]
        print("\nSample entry:")
        print(f"  data_source: {sample['data_source']}")
        print(f"  ability: {sample['ability']}")
        print(f"  prompt (first message role): {sample['prompt'][0]['role']}")
        print(f"  reward_model: {sample['reward_model']}")
        print(f"  extra_info keys: {list(sample['extra_info'].keys())}")
        print(f"  instance_id: {sample['extra_info']['instance_id']}")


if __name__ == "__main__":
    main()
