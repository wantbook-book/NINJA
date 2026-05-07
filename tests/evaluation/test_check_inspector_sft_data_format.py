from __future__ import annotations

import json
import os
import tempfile
import unittest

from evaluation.check_inspector_sft_data_format import (
    clean_assistant_content,
    clean_assistant_content_with_types,
    clean_records,
    collect_wrong_assistant_contents,
    is_valid_assistant_content,
    write_cleaned_records_parquet,
    write_summary,
    write_wrong_assistant_contents,
)
from evaluation.sft_data_utils import load_records


class CheckInspectorSFTDataFormatTest(unittest.TestCase):
    def test_valid_assistant_content_allows_one_think_and_multiple_tool_calls(self) -> None:
        self.assertTrue(
            is_valid_assistant_content(
                '<think>Inspect the file.</think>\n'
                '<tool_call>{"name": "get_file", "arguments": {}}</tool_call>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>'
            )
        )

    def test_valid_assistant_content_allows_result_before_tool_calls(self) -> None:
        self.assertTrue(
            is_valid_assistant_content(
                '<think>Done.</think>\n'
                '<result>{"suspicious": []}</result>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>'
            )
        )

    def test_valid_assistant_content_allows_result_or_trace_locs_only(self) -> None:
        self.assertTrue(is_valid_assistant_content('<result>{"suspicious": []}</result>'))
        self.assertTrue(
            is_valid_assistant_content(
                "<trace_locs>\n"
                "pkg/a.py\n"
                "function: foo\n"
                "</trace_locs>"
            )
        )

    def test_valid_assistant_content_allows_trace_locs_before_tool_call(self) -> None:
        self.assertTrue(
            is_valid_assistant_content(
                "<think></think>\n"
                "<trace_locs>\n"
                "pkg/a.py\n"
                "function: foo\n"
                "</trace_locs>\n"
                '<tool_call>\n{"name": "exit", "arguments": {}}\n</tool_call>'
            )
        )

    def test_invalid_assistant_content_shapes(self) -> None:
        invalid_contents = [
            'prefix <think>Inspect.</think><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
            '<think>Inspect.</think><tool_call>{}</tool_call><result>{"suspicious": []}</result>',
            '<think>Inspect.</think><result>{}</result><result>{}</result><tool_call>{}</tool_call>',
            '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
            '<think>Inspect.</think>',
            '<think>First.</think><think>Second.</think><tool_call>{}</tool_call>',
            '<think>Inspect.</think><tool_call>{}</tool_call><think>More.</think>',
        ]

        for content in invalid_contents:
            with self.subTest(content=content):
                self.assertFalse(is_valid_assistant_content(content))

    def test_collect_wrong_assistant_contents_uses_instance_id_and_messages_json(self) -> None:
        records = [
            {
                "extra_info": {"instance_id": "from-extra"},
                "messages": [
                    {"role": "user", "content": "Inspect pkg/a.py"},
                    {
                        "role": "assistant",
                        "content": '<think>OK.</think><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    },
                    {"role": "assistant", "content": "plain text"},
                ],
            },
            {
                "instance_id": "from-top-level",
                "messages": json.dumps(
                    [
                        {"role": "assistant", "content": '<tool_call>{"name": "exit", "arguments": {}}</tool_call>'}
                    ]
                ),
            },
        ]

        wrong_contents = collect_wrong_assistant_contents(records)

        self.assertEqual(
            wrong_contents,
            [
                {
                    "instance_id": "from-extra",
                    "message_index": 2,
                    "type": ["unclassified"],
                    "content": "plain text",
                    "corrected_content": "plain text",
                    "corrected_valid": False,
                },
                {
                    "instance_id": "from-top-level",
                    "message_index": 0,
                    "type": ["missing_think_block"],
                    "content": '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    "corrected_content": '<think></think>\n'
                    '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    "corrected_valid": True,
                },
            ],
        )

    def test_clean_assistant_content_repairs_common_bad_formats(self) -> None:
        cases = [
            (
                '<think>Done.</thinking><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>Done.</think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["think_closed_by_thinking"],
            ),
            (
                '<thinking>Done.</thinking><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>Done.</think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["thinking_block"],
            ),
            (
                '<think>Done.</think>\nstray text\n<result>{}</result>\nmore stray\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>Done.</think>\n<result>{}</result>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["unwrapped_text_between_blocks"],
            ),
            (
                'initial reasoning\n<result>{}</result>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>initial reasoning</think>\n<result>{}</result>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["missing_think_block"],
            ),
            (
                '<think>Done.</think><tool_call>{"name": "exit", "arguments": {}}</tool_call></tool_call>',
                '<think>Done.</think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["extra_closing_tool_call"],
            ),
            (
                '<think>First.</think><think>Second.</think><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>First.\nSecond.</think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["multiple_think_blocks"],
            ),
            (
                '<think>First.</think><tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
                '<think>Second.</think><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think>First.\nSecond.</think>\n<tool_call>{"name": "lookup", "arguments": {}}</tool_call>\n'
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["multiple_think_blocks"],
            ),
            (
                '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                '<think></think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                ["missing_think_block"],
            ),
            (
                "<think></think><trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                "<think></think>\n<trace_locs>pkg/a.py\nfunction: foo</trace_locs>\n"
                '<tool_call>\n{"name": "exit", "arguments": {}}\n</tool_call>',
                ["trace_locs_with_think_missing_tool_call"],
            ),
        ]

        for content, expected, expected_types in cases:
            with self.subTest(content=content):
                corrected = clean_assistant_content(content)
                self.assertEqual(corrected, expected)
                self.assertTrue(is_valid_assistant_content(corrected))
                corrected_with_types, correction_types = clean_assistant_content_with_types(content)
                self.assertEqual(corrected_with_types, expected)
                self.assertEqual(correction_types, expected_types)

    def test_clean_records_exports_only_records_valid_after_cleaning(self) -> None:
        records = [
            {
                "extra_info": {"instance_id": "repairable"},
                "messages": [
                    {"role": "user", "content": "Inspect pkg/a.py"},
                    {
                        "role": "assistant",
                        "content": '<think>Done.</thinking><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    },
                ],
            },
            {
                "extra_info": {"instance_id": "not-repairable"},
                "messages": [{"role": "assistant", "content": "plain text"}],
            },
        ]

        cleaned_records, wrong_contents, stats = clean_records(records)

        self.assertEqual(stats["wrong_assistant_contents"], 2)
        self.assertEqual(stats["corrected_assistant_contents"], 1)
        self.assertEqual(stats["uncorrected_assistant_contents"], 1)
        self.assertEqual(
            stats["correction_type_counts"],
            {"think_closed_by_thinking": 1, "unclassified": 1},
        )
        self.assertEqual(len(cleaned_records), 1)
        self.assertEqual(cleaned_records[0]["extra_info"]["instance_id"], "repairable")
        self.assertEqual(
            cleaned_records[0]["messages"][1]["content"],
            '<think>Done.</think>\n<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
        )
        self.assertEqual(
            wrong_contents,
            [
                {
                    "instance_id": "repairable",
                    "message_index": 1,
                    "type": ["think_closed_by_thinking"],
                    "content": '<think>Done.</thinking><tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    "corrected_content": '<think>Done.</think>\n'
                    '<tool_call>{"name": "exit", "arguments": {}}</tool_call>',
                    "corrected_valid": True,
                },
                {
                    "instance_id": "not-repairable",
                    "message_index": 0,
                    "type": ["unclassified"],
                    "content": "plain text",
                    "corrected_content": "plain text",
                    "corrected_valid": False,
                },
            ],
        )

    def test_clean_records_counts_multiple_think_and_tool_call_only_correction_types(self) -> None:
        records = [
            {
                "extra_info": {"instance_id": "multiple-think"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": '<think>First.</think><think>Second.</think><tool_call>{}</tool_call>',
                    }
                ],
            },
            {
                "extra_info": {"instance_id": "tool-call-only"},
                "messages": [
                    {"role": "assistant", "content": '<tool_call>{"name": "exit", "arguments": {}}</tool_call>'}
                ],
            },
            {
                "extra_info": {"instance_id": "think-trace-only"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<think></think><trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                    }
                ],
            },
        ]

        cleaned_records, wrong_contents, stats = clean_records(records)

        self.assertEqual(len(cleaned_records), 3)
        self.assertEqual(stats["corrected_assistant_contents"], 3)
        self.assertEqual(stats["uncorrected_assistant_contents"], 0)
        self.assertEqual(
            stats["correction_type_counts"],
            {
                "missing_think_block": 1,
                "multiple_think_blocks": 1,
                "trace_locs_with_think_missing_tool_call": 1,
            },
        )
        self.assertEqual(wrong_contents[0]["type"], ["multiple_think_blocks"])
        self.assertEqual(wrong_contents[1]["type"], ["missing_think_block"])
        self.assertEqual(wrong_contents[2]["type"], ["trace_locs_with_think_missing_tool_call"])

    def test_result_only_and_trace_locs_only_are_not_errors(self) -> None:
        records = [
            {
                "extra_info": {"instance_id": "result-only"},
                "messages": [{"role": "assistant", "content": '<result>{"suspicious": []}</result>'}],
            },
            {
                "extra_info": {"instance_id": "trace-only"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                    }
                ],
            },
        ]

        cleaned_records, wrong_contents, stats = clean_records(records)

        self.assertEqual(cleaned_records, records)
        self.assertEqual(wrong_contents, [])
        self.assertEqual(stats["wrong_assistant_contents"], 0)
        self.assertEqual(stats["correction_type_counts"], {})

    def test_write_cleaned_records_parquet(self) -> None:
        rows = [
            {
                "extra_info": {"instance_id": "i1"},
                "messages": [{"role": "assistant", "content": '<think></think><tool_call>{}</tool_call>'}],
            }
        ]

        with tempfile.TemporaryDirectory(prefix="clean_inspector_sft_", dir="/tmp") as tmp_dir:
            output_path = os.path.join(tmp_dir, "cleaned.parquet")
            write_cleaned_records_parquet(rows, output_path)

            loaded = load_records(output_path)

        self.assertEqual(loaded[0]["extra_info"]["instance_id"], "i1")
        self.assertEqual(loaded[0]["messages"][0]["role"], "assistant")

    def test_write_summary_json(self) -> None:
        summary = {
            "error_type_counts": {"missing_think_block": 1},
            "total_errors": 2,
            "corrected_errors": 1,
            "uncorrected_errors": 1,
            "cleaned_parquet_records": 3,
        }

        with tempfile.TemporaryDirectory(prefix="check_inspector_sft_", dir="/tmp") as tmp_dir:
            output_path = os.path.join(tmp_dir, "summary.json")
            write_summary(summary, output_path)

            with open(output_path, encoding="utf-8") as f:
                loaded = json.load(f)

        self.assertEqual(loaded, summary)

    def test_write_wrong_assistant_contents_jsonl(self) -> None:
        rows = [
            {
                "instance_id": "from-top-level",
                "content": "bad\ncontent",
                "corrected_content": "<think>bad</think>",
                "corrected_valid": False,
            }
        ]

        with tempfile.TemporaryDirectory(prefix="check_inspector_sft_", dir="/tmp") as tmp_dir:
            output_path = os.path.join(tmp_dir, "wrong.jsonl")
            write_wrong_assistant_contents(rows, output_path)

            with open(output_path, encoding="utf-8") as f:
                lines = [json.loads(line) for line in f]

        self.assertEqual(lines, rows)


if __name__ == "__main__":
    unittest.main()
