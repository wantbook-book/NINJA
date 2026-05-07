from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from examples.code_localization.build_navigator_training_data_from_swebench import (
    build_records_from_processed_rows,
    normalize_location_list,
    save_records,
    split_records,
)


class BuildNavigatorTrainingDataFromSwebenchTest(unittest.TestCase):
    def test_build_records_uses_full_instance_ground_truth(self) -> None:
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
                    "base_commit": "abc123",
                    "problem_statement": "Fix the bug.",
                    "structure": "pkg/\n  a.py\n  b.py",
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
        self.assertEqual(len(records), 1)

        record = records[0]
        self.assertEqual(record["data_source"], "code_localization")
        self.assertEqual(record["agent_name"], "code_localization")
        self.assertEqual(record["ability"], "code_localization")
        self.assertEqual(
            json.loads(record["reward_model"]["ground_truth"]),
            ["pkg/a.py:f", "pkg/a.py:C.m", "pkg/b.py:g"],
        )
        self.assertEqual(
            json.loads(record["extra_info"]["ground_truth"]),
            ["pkg/a.py:f", "pkg/a.py:C.m", "pkg/b.py:g"],
        )
        self.assertEqual(record["extra_info"]["base_commit"], "abc123")
        self.assertEqual(record["extra_info"]["index"], 0)
        self.assertEqual(
            set(record["extra_info"]),
            {
                "index",
                "split",
                "instance_id",
                "repo",
                "base_commit",
                "problem_statement",
                "structure",
                "ground_truth",
                "patch",
                "need_tools_kwargs",
                "tools_kwargs",
            },
        )

        self.assertEqual([message["role"] for message in record["prompt"]], ["system", "user"])
        self.assertIn("<tools>", record["prompt"][0]["content"])
        self.assertIn("find_files_by_content", record["prompt"][0]["content"])
        self.assertIn("Fix the bug.", record["prompt"][1]["content"])
        self.assertIn("pkg/", record["prompt"][1]["content"])
        self.assertIn("find_files_by_content", record["extra_info"]["tools_kwargs"])

    def test_build_records_accepts_numpy_array_ground_truth(self) -> None:
        processed_rows = [
            {
                "reward_model": {
                    "ground_truth": np.array(
                        ["pkg/a.py:f", "pkg/a.py:f", "pkg/b.py:g"],
                        dtype=object,
                    ),
                },
                "extra_info": {
                    "instance_id": "i1",
                    "problem_statement": "Fix the bug.",
                    "structure": "pkg/",
                },
            }
        ]

        records, skipped = build_records_from_processed_rows(
            processed_rows,
            agent_name="code_localization",
        )

        self.assertEqual(skipped, [])
        self.assertEqual(
            json.loads(records[0]["reward_model"]["ground_truth"]),
            ["pkg/a.py:f", "pkg/b.py:g"],
        )

    def test_build_records_falls_back_to_edited_entities_and_skips_missing_gt(self) -> None:
        processed_rows = [
            {
                "reward_model": {"edited_entities": ["pkg/a.py:f"]},
                "extra_info": {"instance_id": "has-gt"},
            },
            {
                "reward_model": {"ground_truth": []},
                "extra_info": {"instance_id": "missing-gt"},
            },
        ]

        records, skipped = build_records_from_processed_rows(
            processed_rows,
            agent_name="code_localization",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extra_info"]["instance_id"], "has-gt")
        self.assertEqual(skipped, ["missing-gt"])

    def test_normalize_location_list_converts_json_and_numpy(self) -> None:
        self.assertEqual(
            normalize_location_list('["pkg/a.py:f", "pkg/a.py:f", "pkg/b.py:g"]'),
            ["pkg/a.py:f", "pkg/b.py:g"],
        )
        self.assertEqual(
            normalize_location_list(np.array(["pkg/a.py:f", "pkg/b.py:g"], dtype=object)),
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

    def test_save_records_writes_generic_sample_json(self) -> None:
        val_record = {
            "data_source": "code_localization",
            "agent_name": "code_localization",
            "prompt": [{"role": "user", "content": "hello"}],
            "ability": "code_localization",
            "reward_model": {"style": "rule", "ground_truth": '["pkg/a.py:f"]'},
            "extra_info": {"split": "test", "index": 0, "instance_id": "i1"},
        }

        with tempfile.TemporaryDirectory(prefix="navigator_training_", dir="/tmp") as tmp_dir:
            save_records([], [val_record], tmp_dir)

            with open(os.path.join(tmp_dir, "sample.json"), encoding="utf-8") as f:
                sample = json.load(f)

        self.assertEqual(sample["extra_info"]["instance_id"], "i1")
        self.assertEqual(sample["data_source"], "code_localization")


if __name__ == "__main__":
    unittest.main()
