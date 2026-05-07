"""Build Navigator training data from preprocessed SWE-bench tool-agent data.

This script reads the parquet files produced by
``examples.data_preprocess.swebench_tool_agent_loop_reposearch`` and converts
each valid SWE-bench instance into one Navigator training sample for
``verl.experimental.agent_loop.code_localization_loop``.

The output schema follows the code-localization agent loop requirements:

    - ``data_source`` is ``code_localization``.
    - ``agent_name`` defaults to the registered ``code_localization`` loop.
    - ``prompt`` contains the Navigator system/user messages with main-agent
      tool schemas embedded.
    - ``reward_model.ground_truth`` and ``extra_info.ground_truth`` contain the
      full ground-truth function-location list for the instance.
    - ``extra_info`` carries ``problem_statement`` and ``structure``, which the
      Navigator loop uses to rebuild its runtime prompt.

Usage:
    python -m examples.code_localization.build_navigator_training_data_from_swebench \
        --input_file data/swebench/test.parquet \
        --output_dir data/code_loc/navigator_from_swebench \
        --val_size 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
from string import Template
from typing import Any

import numpy as np
import pandas as pd

from evaluation.multi_agent_shared import (
    PROMPTS_DIR,
    _load_text,
    augment_system_prompt_with_tool_instructions,
)


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
    if isinstance(value, list | tuple | set | dict | np.ndarray):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _string_value(value: Any) -> str:
    if value is None or _is_missing_scalar(value):
        return ""
    return str(value)


def normalize_location_list(value: Any) -> list[str]:
    """Normalize a parquet cell into a list of function location strings."""
    if value is None or _is_missing_scalar(value):
        return []
    if isinstance(value, np.ndarray):
        return normalize_location_list(value.tolist())
    if isinstance(value, list | tuple | set):
        return _dedupe_preserve_order(
            [loc for loc in (str(item).strip() for item in value) if loc]
        )
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "\n" in text:
                return _dedupe_preserve_order(
                    [loc.strip() for loc in text.splitlines() if loc.strip()]
                )
            if "," in text:
                return _dedupe_preserve_order(
                    [loc.strip() for loc in text.split(",") if loc.strip()]
                )
            return [text]
        return normalize_location_list(parsed)
    return [str(value).strip()] if str(value).strip() else []


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


def load_prompt_templates() -> dict[str, str]:
    """Load Navigator prompt templates from prompts/multi_agent/ directory."""
    return {
        "nav_system": _load_text(os.path.join(PROMPTS_DIR, "main_agent_system_prompt.txt")),
        "nav_user": _load_text(os.path.join(PROMPTS_DIR, "main_agent_user_prompt.txt")),
    }


def load_tool_schemas() -> list[dict]:
    """Load main-agent tool schemas."""
    tool_schemas_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tools",
        "tool_schemas",
        "main_agent_tools.json",
    )
    try:
        with open(tool_schemas_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _build_navigator_tools_kwargs(
    instance_id: str,
    tool_schemas: list[dict],
) -> dict[str, dict[str, dict[str, str]]]:
    """Build per-tool create kwargs for Navigator tool initialization."""
    create_kwargs = {"instance_id": instance_id}
    tools_kwargs = {}
    for schema in tool_schemas:
        tool_name = schema.get("function", {}).get("name")
        if tool_name:
            tools_kwargs[tool_name] = {"create_kwargs": create_kwargs}
    return tools_kwargs


def build_navigator_prompt(
    problem_statement: str,
    structure: str,
    templates: dict[str, str],
    tool_schemas: list[dict],
) -> list[dict]:
    """Build Navigator prompt messages using loaded templates."""
    system_prompt = augment_system_prompt_with_tool_instructions(
        templates["nav_system"], tool_schemas
    )
    user_content = Template(templates["nav_user"]).safe_substitute(
        problem_statement=problem_statement,
        repo_structure=structure,
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
    """Convert one SWE-bench row into a Navigator training record."""
    instance_id = _string_value(row.get("instance_id", ""))
    problem_statement = _string_value(row.get("problem_statement", ""))
    structure = _string_value(row.get("structure", ""))
    ground_truth_locations = normalize_location_list(row.get("ground_truth", []))
    ground_truth = json.dumps(ground_truth_locations)

    return {
        "data_source": "code_localization",
        "agent_name": agent_name,
        "prompt": build_navigator_prompt(
            problem_statement,
            structure,
            templates,
            tool_schemas,
        ),
        "ability": "code_localization",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "index": 0,
            "split": "train",
            "instance_id": instance_id,
            "repo": _string_value(row.get("repo", "")),
            "base_commit": _string_value(row.get("base_commit", "")),
            "problem_statement": problem_statement,
            "structure": structure,
            "ground_truth": ground_truth,
            "patch": _string_value(row.get("patch", "")),
            "need_tools_kwargs": True,
            "tools_kwargs": _build_navigator_tools_kwargs(instance_id, tool_schemas),
        },
    }


def _ground_truth_from_example(
    example: dict[str, Any],
    reward_model: dict[str, Any],
    extra_info: dict[str, Any],
) -> list[str]:
    for key in ("ground_truth", "edited_entities"):
        locations = normalize_location_list(reward_model.get(key))
        if locations:
            return locations
    for value in (
        extra_info.get("ground_truth"),
        example.get("ground_truth"),
        example.get("edited_entities"),
    ):
        locations = normalize_location_list(value)
        if locations:
            return locations
    return []


def build_records_from_processed_rows(
    processed_rows: Any,
    *,
    agent_name: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create one Navigator training record per SWE-bench instance."""
    templates = load_prompt_templates()
    tool_schemas = load_tool_schemas()

    records = []
    skipped_instance_ids = []
    for example in processed_rows:
        reward_model = _as_dict(example.get("reward_model"))
        extra_info = _as_dict(example.get("extra_info"))
        instance_id = _string_value(extra_info.get("instance_id") or example.get("instance_id", ""))
        ground_truth_locations = _ground_truth_from_example(example, reward_model, extra_info)
        if not ground_truth_locations:
            if instance_id:
                skipped_instance_ids.append(instance_id)
            continue

        row = {
            "instance_id": instance_id,
            "repo": extra_info.get("repo", example.get("repo", "")),
            "base_commit": extra_info.get("base_commit", example.get("base_commit", "")),
            "problem_statement": extra_info.get(
                "problem_statement",
                example.get("problem_statement", ""),
            ),
            "structure": extra_info.get("structure", example.get("structure", "")),
            "ground_truth": json.dumps(ground_truth_locations),
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
    sample_path = os.path.join(output_dir, "sample.json")
    pd.DataFrame(train_data).to_parquet(train_path, index=False)
    pd.DataFrame(val_data).to_parquet(val_path, index=False)

    sample = train_data[0] if train_data else val_data[0] if val_data else None
    if sample:
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
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
        description="Build Navigator training data from preprocessed SWE-bench tool-agent parquet"
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
    print(f"Built {len(records)} Navigator samples")
    print(
        f"Saved {len(train_data)} training samples to "
        f"{os.path.join(args.output_dir, 'train.parquet')}"
    )
    print(
        f"Saved {len(val_data)} validation samples to "
        f"{os.path.join(args.output_dir, 'val.parquet')}"
    )
    if records:
        print(f"Saved one sample to {os.path.join(args.output_dir, 'sample.json')}")
    print(f"Skipped {len(skipped_instance_ids)} raw instances without GT function locations")


if __name__ == "__main__":
    main()
