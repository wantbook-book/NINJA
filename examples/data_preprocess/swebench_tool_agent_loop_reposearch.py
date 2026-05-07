# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the SWEBench dataset to parquet format
"""

import argparse
import copy
import json
import os
from functools import lru_cache
from typing import Any

from tqdm import tqdm
from LocAgent.util.utils import load_json
import datasets

from examples.data_preprocess.extract_ground_truth_from_patch import extract_edit_functions_from_patch

def _load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

from verl.utils.hdfs_io import makedirs
from verl.utils.hdfs_io import copy as hdfs_copy
from tools.RepoSearch.preprocess_data import filter_out_test_files, show_project_structure, filter_none_python


_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM_PROMPT = _load_prompt(os.path.join(_ROOT_DIR, "prompts", "repo_search", "system_prompt.txt"))
TASK_PROMPT = _load_prompt(os.path.join(_ROOT_DIR, "prompts", "repo_search", "task_prompt.txt"))


def _project_json_path(project_file_loc: str, instance_id: str) -> str:
    return os.path.join(project_file_loc, f"{instance_id}.json")


def _build_signature_repo_search_tools_kwargs(instance_id: str) -> dict:
    create_kwargs = {"loc_instance_id": instance_id}
    return {
        "get_methods_of_class": {"create_kwargs": create_kwargs},
        "get_file_functions": {"create_kwargs": create_kwargs},
        "get_file_classes": {"create_kwargs": create_kwargs},
        "get_code_of_class_function": {"create_kwargs": create_kwargs},
        "get_code_of_file_function": {"create_kwargs": create_kwargs},
        "exit": {"create_kwargs": create_kwargs},
    }


@lru_cache(maxsize=None)
def _load_project_record_cached(project_json_path: str) -> dict[str, Any]:
    return load_json(project_json_path)


def _load_project_record(project_file_loc: str, instance_id: str) -> dict[str, Any]:
    return copy.deepcopy(
        _load_project_record_cached(_project_json_path(project_file_loc, instance_id))
    )


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0 or all(_is_empty_value(item) for item in value)
    if isinstance(value, dict):
        return len(value) == 0
    return False


def _build_structure_text(project_record: dict[str, Any]) -> str:
    filtered_project_record = copy.deepcopy(project_record)
    filter_none_python(filtered_project_record)
    filter_out_test_files(filtered_project_record)
    return show_project_structure(filtered_project_record).strip()


def _build_ground_truth_details(
    gt_location_map: dict[str, dict[str, Any]], instance_id: str
) -> dict[str, Any] | None:
    ground_truth_details = copy.deepcopy(gt_location_map.get(instance_id))
    if ground_truth_details is None:
        return None
    ground_truth_details["ground_truth"] = ground_truth_details.get("edited_entities", [])
    return ground_truth_details


def _build_output_data(
    example: dict[str, Any],
    *,
    split: str,
    original_index: int,
    data_source: str,
    agent_name: str,
    project_record: dict[str, Any],
    ground_truth_details: dict[str, Any],
) -> dict[str, Any]:
    instance_id = example.get("instance_id")
    problem_statement = example.get("problem_statement")
    structure_text = _build_structure_text(project_record)

    return {
        "data_source": data_source,
        "agent_name": agent_name,
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": TASK_PROMPT.format(
                    problem_statement=problem_statement,
                    structure=structure_text,
                ).strip(),
            },
        ],
        "ability": "code",
        "reward_model": copy.deepcopy(ground_truth_details),
        "extra_info": {
            "structure": structure_text,
            "split": split,
            "index": original_index,
            "instance_id": instance_id,
            "repo": example.get("repo"),
            "base_commit": example.get("base_commit"),
            "problem_statement": problem_statement,
            "patch": example.get("patch"),
            "need_tools_kwargs": True,
            "tools_kwargs": _build_signature_repo_search_tools_kwargs(instance_id),
        },
    }


def _split_missing_project_files(
    split_dataset: datasets.Dataset, project_file_loc: str
) -> tuple[datasets.Dataset, list[str], list[int]]:
    valid_indices = []
    skipped_instance_ids = []
    for idx, example in enumerate(tqdm(split_dataset, desc="Checking project files")):
        instance_id = example.get("instance_id")
        project_json_path = _project_json_path(project_file_loc, instance_id)
        if os.path.isfile(project_json_path):
            valid_indices.append(idx)
        else:
            skipped_instance_ids.append(instance_id)

    return split_dataset.select(valid_indices), skipped_instance_ids, valid_indices


def _split_empty_required_fields(
    split_dataset: datasets.Dataset,
    original_indices: list[int],
    *,
    project_file_loc: str,
    gt_location_map: dict[str, dict[str, Any]],
) -> tuple[datasets.Dataset, list[str], list[int]]:
    valid_indices = []
    skipped_instance_ids = []
    filtered_original_indices = []

    for idx, example in enumerate(tqdm(split_dataset, desc="Checking required fields")):
        instance_id = example.get("instance_id")
        project_record = _load_project_record(project_file_loc, instance_id)
        structure_text = _build_structure_text(project_record)
        ground_truth_details = _build_ground_truth_details(gt_location_map, instance_id)
        ground_truth = None
        if ground_truth_details is not None:
            ground_truth = ground_truth_details.get("ground_truth")

        if (
            _is_empty_value(structure_text)
            or _is_empty_value(example.get("problem_statement"))
            or _is_empty_value(ground_truth)
        ):
            skipped_instance_ids.append(instance_id)
            continue

        valid_indices.append(idx)
        filtered_original_indices.append(original_indices[idx])

    return split_dataset.select(valid_indices), skipped_instance_ids, filtered_original_indices


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source", default="czlll/Loc-Bench_V1", help="The data source to load the raw dataset from."
    )
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/swebench", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--project_file_loc", default=None, help="The local path to save the project files.")
    parser.add_argument(
        "--repo_base_dir",
        default=os.getenv("REPO_DIR", "/tmp/repos"),
        help="Base directory for cached repos and temporary worktrees used by oracle patch extraction.",
    )
    parser.add_argument(
        "--split",
        default='test',
        help="The dataset split to preprocess.",
    )
    parser.add_argument(
        '--gt_locations_file',
        required=True,
        help="The file path to provide the ground truth location."
    )
    parser.add_argument(
        "--agent_name",
        default="tool_agent",
        help="The agent_name field to write into each output sample.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    data_source = args.data_source

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_source, "default")

    gt_locations = load_jsonl(args.gt_locations_file)
    gt_location_map = {}
    for item in gt_locations:
        file_changes = item.get('file_changes') or []
        for file_change in file_changes:
            if item['instance_id'] not in gt_location_map:
                gt_location_map[item['instance_id']] = {
                    "edited_entities": [],
                    "added_entities": [],
                    "edited_modules": [],
                    "added_modules": [],
                }
            for key in ("edited_entities", "added_entities", "edited_modules", "added_modules"):
                for location in file_change['changes'].get(key, []):
                    if location not in gt_location_map[item['instance_id']][key]:
                        gt_location_map[item['instance_id']][key].append(location)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # add a row to each data item that represents a unique id
    def make_map_fn(split, original_indices):
        def process_fn(example, idx):
            instance_id = example.get("instance_id")
            original_idx = original_indices[idx]
            project_record = _load_project_record(args.project_file_loc, instance_id)
            ground_truth_details = _build_ground_truth_details(gt_location_map, instance_id)
            if ground_truth_details is None:
                raise ValueError(f"Missing ground truth details for instance {instance_id}")
            return _build_output_data(
                example,
                split=split,
                original_index=original_idx,
                data_source=data_source,
                agent_name=args.agent_name,
                project_record=project_record,
                ground_truth_details=ground_truth_details,
            )

        return process_fn

    processed_datasets = {}
    checkpoint_dir = os.path.join(local_save_dir, ".filter_checkpoint")
    checkpoint_dataset_path = os.path.join(checkpoint_dir, f"{args.split}_filtered")
    checkpoint_meta_path = os.path.join(checkpoint_dir, f"{args.split}_meta.json")

    if os.path.isdir(checkpoint_dataset_path) and os.path.isfile(checkpoint_meta_path):
        print(f"Loading filtered dataset from checkpoint: {checkpoint_dir}")
        available_dataset = datasets.load_from_disk(checkpoint_dataset_path)
        with open(checkpoint_meta_path, "r", encoding="utf-8") as f:
            checkpoint_meta = json.load(f)
        original_indices = checkpoint_meta["original_indices"]
        skipped_instance_ids = checkpoint_meta["skipped_instance_ids"]
    else:
        available_dataset, skipped_instance_ids, original_indices = _split_missing_project_files(
            dataset[args.split], args.project_file_loc
        )
        if skipped_instance_ids:
            print(
                f"Skipping {len(skipped_instance_ids)} instances in split '{args.split}' "
                "because the project JSON file does not exist."
            )
        available_dataset, empty_field_instance_ids, original_indices = _split_empty_required_fields(
            available_dataset,
            original_indices,
            project_file_loc=args.project_file_loc,
            gt_location_map=gt_location_map,
        )
        if empty_field_instance_ids:
            print(
                f"Skipping {len(empty_field_instance_ids)} instances in split '{args.split}' "
                "because structure, problem_statement, or ground_truth is empty."
            )
            skipped_instance_ids.extend(empty_field_instance_ids)

        os.makedirs(checkpoint_dir, exist_ok=True)
        available_dataset.save_to_disk(checkpoint_dataset_path)
        with open(checkpoint_meta_path, "w", encoding="utf-8") as f:
            json.dump({"original_indices": original_indices, "skipped_instance_ids": skipped_instance_ids}, f)
        print(f"Filter checkpoint saved to: {checkpoint_dir}")
    process_fn = make_map_fn(args.split, original_indices)
    processed_examples = [
        process_fn(available_dataset[i], i)
        for i in tqdm(range(len(available_dataset)), desc=f"Processing split '{args.split}'")
    ]
    processed_datasets[args.split] = datasets.Dataset.from_list(processed_examples)

    # record skipped_instance_ids 
    skipped_info_path = os.path.join(local_save_dir, f"{args.split}_skipped_instance_ids.txt")
    with open(skipped_info_path, "w", encoding="utf-8") as f:
        for instance_id in skipped_instance_ids:
            f.write(f"{instance_id}\n")

    for split_name, split_dataset in processed_datasets.items():
        split_dataset.to_parquet(os.path.join(local_save_dir, f"{split_name}.parquet"))
        if len(split_dataset) == 0:
            continue
        sample = split_dataset[0]
        sample_path = os.path.join(local_save_dir, f"{split_name}_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        hdfs_copy(src=local_save_dir, dst=hdfs_dir)
