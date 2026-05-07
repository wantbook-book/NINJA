from __future__ import annotations

import json
import unittest

import numpy as np

from examples.code_localization.build_inspector_training_data_from_swebench import (
    build_records_from_processed_rows,
    group_locations_by_file,
    normalize_location_list,
    split_records,
)


class BuildInspectorTrainingDataFromSwebenchTest(unittest.TestCase):
    def test_group_locations_by_file_groups_unique_files(self) -> None:
        locations = [
            "pkg/a.py:f",
            "pkg/a.py:C.m",
            "pkg/b.py:g",
            "pkg/a.py:f",
        ]

        self.assertEqual(
            group_locations_by_file(locations),
            {
                "pkg/a.py": ["pkg/a.py:f", "pkg/a.py:C.m"],
                "pkg/b.py": ["pkg/b.py:g"],
            },
        )

    def test_build_records_keeps_only_entry_file_ground_truth(self) -> None:
        processed_rows = [
            {
                "data_source": "mock/source",
                "reward_model": {
                    "ground_truth": ["pkg/a.py:f", "pkg/a.py:C.m", "pkg/b.py:g"],
                    "edited_entities": ["pkg/a.py:f", "pkg/a.py:C.m", "pkg/b.py:g"],
                },
                "extra_info": {
                    "instance_id": "i1",
                    "repo": "owner/repo",
                    "problem_statement": "Fix the bug.",
                    "patch": "diff --git a/pkg/a.py b/pkg/a.py",
                    "split": "test",
                },
            }
        ]

        records, skipped = build_records_from_processed_rows(
            processed_rows,
            agent_name="code_localization",
        )

        self.assertEqual(skipped, [])
        self.assertEqual(len(records), 2)

        by_file = {record["extra_info"]["entry_file"]: record for record in records}
        self.assertEqual(
            json.loads(by_file["pkg/a.py"]["reward_model"]["ground_truth"]),
            ["pkg/a.py:f", "pkg/a.py:C.m"],
        )
        self.assertEqual(
            json.loads(by_file["pkg/b.py"]["reward_model"]["ground_truth"]),
            ["pkg/b.py:g"],
        )
        self.assertEqual(by_file["pkg/a.py"]["data_source"], "code_localization_inspector")
        self.assertEqual(
            set(by_file["pkg/a.py"]["extra_info"]),
            {
                "index",
                "split",
                "instance_id",
                "repo",
                "problem_statement",
                "structure",
                "ground_truth",
                "entry_file",
                "patch",
                "need_tools_kwargs",
                "tools_kwargs",
            },
        )

    def test_build_records_accepts_numpy_array_ground_truth(self) -> None:
        processed_rows = [
            {
                "data_source": "mock/source",
                "reward_model": {
                    "ground_truth": np.array(
                        ["pkg/a.py:f", "pkg/a.py:C.m", "pkg/b.py:g"],
                        dtype=object,
                    ),
                },
                "extra_info": {
                    "instance_id": "i1",
                    "repo": "owner/repo",
                    "problem_statement": "Fix the bug.",
                    "patch": "diff --git a/pkg/a.py b/pkg/a.py",
                },
            }
        ]

        records, skipped = build_records_from_processed_rows(
            processed_rows,
            agent_name="code_localization",
        )

        self.assertEqual(skipped, [])
        by_file = {record["extra_info"]["entry_file"]: record for record in records}
        self.assertEqual(
            json.loads(by_file["pkg/a.py"]["reward_model"]["ground_truth"]),
            ["pkg/a.py:f", "pkg/a.py:C.m"],
        )
        self.assertEqual(
            json.loads(by_file["pkg/b.py"]["reward_model"]["ground_truth"]),
            ["pkg/b.py:g"],
        )

    def test_normalize_location_list_converts_numpy_array_with_tolist(self) -> None:
        value = np.array(["pkg/a.py:f", "pkg/b.py:g"], dtype=object)

        self.assertEqual(
            normalize_location_list(value),
            ["pkg/a.py:f", "pkg/b.py:g"],
        )

    def test_split_records_uses_all_records_as_validation_when_val_size_is_too_large(self) -> None:
        records = [
            {
                "extra_info": {"split": "train", "index": 0},
                "reward_model": {"ground_truth": "[]"},
            },
            {
                "extra_info": {"split": "train", "index": 1},
                "reward_model": {"ground_truth": "[]"},
            },
        ]

        train_data, val_data = split_records(records, val_size=10, seed=42)

        self.assertEqual(train_data, [])
        self.assertEqual(len(val_data), 2)
        self.assertEqual({item["extra_info"]["split"] for item in val_data}, {"test"})


if __name__ == "__main__":
    unittest.main()
