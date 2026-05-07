from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.modify_ground_truth import (
    attach_ground_truth,
    load_location_support_ground_truth,
    normalize_ground_truth_value,
)


class ModifyGroundTruthTest(unittest.TestCase):
    def test_loads_support_entities_as_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            location_file = Path(tmpdir) / "gt_location_with_support.jsonl"
            records = [
                {
                    "instance_id": "repo__issue-1",
                    "support_entities": [
                        "pkg/a.py:foo",
                        " pkg/b.py:Bar.baz ",
                        "",
                        None,
                    ],
                },
                {
                    "instance_id": "repo__issue-2",
                    "support_entities": [],
                },
            ]
            location_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            ground_truth = load_location_support_ground_truth(str(location_file))

        self.assertEqual(
            ground_truth,
            {
                "repo__issue-1": ["pkg/a.py:foo", "pkg/b.py:Bar.baz"],
                "repo__issue-2": [],
            },
        )

    def test_attach_ground_truth_uses_location_file_index(self) -> None:
        trajs = [{"instance_id": "repo__issue-1", "messages": []}]
        updated = attach_ground_truth(
            trajs,
            {"repo__issue-1": ["pkg/a.py:foo"]},
        )

        self.assertEqual(updated[0]["ground_truth"], ["pkg/a.py:foo"])

    def test_normalize_ground_truth_value_parses_json_string_list(self) -> None:
        self.assertEqual(
            normalize_ground_truth_value('["pkg/a.py:foo", "pkg/b.py:bar"]'),
            ["pkg/a.py:foo", "pkg/b.py:bar"],
        )


if __name__ == "__main__":
    unittest.main()
