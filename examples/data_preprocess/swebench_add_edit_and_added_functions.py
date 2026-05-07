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
Load a SWE-bench-style dataset from Hugging Face and add `edit_functions`
and `added_functions` fields while preserving the original schema.
"""

import argparse
import json
import os
from typing import Any

import datasets

from examples.data_preprocess.extract_ground_truth_from_patch import (
    extract_edit_functions_with_metadata,
)
from verl.utils.hdfs_io import copy, makedirs


def _normalize_func_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
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
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        return [text]
    return [str(value)]


def _extract_function_fields(example: dict[str, Any]) -> tuple[list[str], list[str]]:
    existing_edit_functions = _normalize_func_list(example.get("edit_functions"))
    existing_added_functions = _normalize_func_list(example.get("added_functions"))
    patch = example.get("patch")
    if not isinstance(patch, str) or not patch.strip():
        return existing_edit_functions, existing_added_functions

    raw_functions, new_only = extract_edit_functions_with_metadata(patch)
    if not raw_functions:
        return existing_edit_functions, existing_added_functions

    new_only_set = set(new_only)
    edit_functions = [func for func in raw_functions if func not in new_only_set]
    added_functions = [func for func in raw_functions if func in new_only_set]
    return edit_functions, added_functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        default="princeton-nlp/SWE-bench_Verified",
        help="The Hugging Face dataset to load.",
    )
    parser.add_argument("--local_dir", default=None, help="Deprecated alias of --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/swebench_with_functions",
        help="The save directory for the augmented dataset.",
    )

    args = parser.parse_args()
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(args.data_source, "default")

    split_names = list(dataset.keys())
    if not split_names:
        raise ValueError("Dataset has no splits to process.")

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    def make_map_fn():
        def process_fn(example):
            edit_functions, added_functions = _extract_function_fields(example)
            data = dict(example)
            data["edit_functions"] = edit_functions
            data["added_functions"] = added_functions
            return data

        return process_fn

    processed_datasets = {}
    stats = {}
    for split_name in split_names:
        processed_dataset = dataset[split_name].map(function=make_map_fn())
        processed_datasets[split_name] = processed_dataset
        stats[split_name] = {
            "rows": len(processed_dataset),
            "rows_with_edit_functions": sum(1 for funcs in processed_dataset["edit_functions"] if funcs),
            "rows_with_added_functions": sum(1 for funcs in processed_dataset["added_functions"] if funcs),
        }

    for split_name, split_dataset in processed_datasets.items():
        split_dataset.to_parquet(os.path.join(local_save_dir, f"{split_name}.parquet"))
        if len(split_dataset) == 0:
            continue
        sample = split_dataset[0]
        sample_path = os.path.join(local_save_dir, f"{split_name}_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    stats_path = os.path.join(local_save_dir, "function_field_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
