from __future__ import annotations

import unittest

from evaluation.build_inspector_sft_data import build_inspector_sft_records
from evaluation.build_navigator_sft_data import build_navigator_sft_records


class BuildSFTDataTest(unittest.TestCase):
    def test_inspector_keeps_only_function_acc_at_gt_size_correct_rows(self) -> None:
        entries = [
            {
                "instance_id": "good",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "pred_locations": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "messages": [
                    {"role": "user", "content": "Inspect pkg/a.py"},
                    {"role": "assistant", "content": "<result>{\"suspicious\": []}</result>"},
                ],
            },
            {
                "instance_id": "bad",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "pred_locations": ["pkg/a.py:foo", "pkg/c.py:baz"],
                "messages": [
                    {"role": "user", "content": "Inspect pkg/a.py"},
                    {"role": "assistant", "content": "<result>{\"suspicious\": []}</result>"},
                ],
            },
        ]

        records, stats = build_inspector_sft_records(entries)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extra_info"]["instance_id"], "good")
        self.assertEqual(records[0]["extra_info"]["function_acc_k"], 2)
        self.assertEqual(stats["skipped"], {"function_acc_at_len_gt_incorrect": 1})

    def test_inspector_parses_messages_and_confirmed_suspicious_from_json_strings(self) -> None:
        entries = [
            {
                "instance_id": "good",
                "ground_truth": "[\"pkg/a.py:foo\"]",
                "confirmed_suspicious": "[{\"location\": \"pkg/a.py:foo\"}]",
                "messages": (
                    "["
                    "{\"role\": \"user\", \"content\": \"Inspect pkg/a.py\"},"
                    "{\"role\": \"assistant\", \"content\": \"done\"}"
                    "]"
                ),
            }
        ]

        records, stats = build_inspector_sft_records(entries)

        self.assertEqual(stats["selected_records"], 1)
        self.assertEqual(records[0]["messages"][0]["role"], "user")
        self.assertEqual(records[0]["extra_info"]["pred_locations"], ["pkg/a.py:foo"])

    def test_navigator_keeps_only_function_acc_at_gt_size_correct_rows(self) -> None:
        entries = [
            {
                "instance_id": "good",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:Bar.baz"],
                "messages": [
                    {"role": "user", "content": "Locate the issue."},
                    {
                        "role": "assistant",
                        "content": (
                            "<trace_locs>\n"
                            "pkg/a.py\n"
                            "function: foo\n"
                            "pkg/b.py\n"
                            "function: Bar.baz\n"
                            "</trace_locs>"
                        ),
                    },
                ],
            },
            {
                "instance_id": "bad",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:Bar.baz"],
                "messages": [
                    {"role": "user", "content": "Locate the issue."},
                    {
                        "role": "assistant",
                        "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                    },
                ],
            },
        ]

        records, stats = build_navigator_sft_records(entries)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["extra_info"]["instance_id"], "good")
        self.assertEqual(records[0]["extra_info"]["function_acc_k"], 2)
        self.assertEqual(records[0]["extra_info"]["pred_methods"], ["foo", "Bar.baz"])
        self.assertEqual(stats["skipped"], {"function_acc_at_len_gt_incorrect": 1})


if __name__ == "__main__":
    unittest.main()
