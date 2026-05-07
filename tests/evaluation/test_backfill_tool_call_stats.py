from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import backfill_tool_call_stats


class BackfillToolCallStatsTest(unittest.TestCase):
    def test_backfills_stats_without_double_counting_assistant_and_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "trajs.jsonl"
            output_path = tmp_path / "trajs_with_stats.jsonl"
            summary_path = tmp_path / "summary.json"
            entry = {
                "instance_id": "demo-1",
                "messages": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"mock_tool","arguments":{"query":"same"}}</tool_call>',
                    },
                    {
                        "role": "user",
                        "_message_kind": "tool_result",
                        "content": (
                            "Here is a result of a function/class code retrived by 'mock_tool' "
                            'with arguments: {"query":"same"}.\n<code>{}</code>'
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"mock_tool","arguments":{"query":"same"}}</tool_call>',
                    },
                    {
                        "role": "user",
                        "_message_kind": "tool_result",
                        "content": (
                            "Here is a result of a function/class code retrived by 'mock_tool' "
                            'with arguments: {"query":"same"}.\n<code>{}</code>'
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"mock_tool","arguments":{"query":"different"}}</tool_call>',
                    },
                    {
                        "role": "user",
                        "_message_kind": "tool_result",
                        "content": (
                            "Here is a result of a function/class code retrived by 'mock_tool' "
                            'with arguments: {"query":"different"}.\n<code>{}</code>'
                        ),
                    },
                    {"role": "assistant", "content": "<trace_locs>\nsrc/foo.py\n</trace_locs>"},
                ],
            }
            input_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

            summary = backfill_tool_call_stats.backfill_file(
                str(input_path),
                str(output_path),
                summary_path=str(summary_path),
                include_component_details=True,
                preserve_existing=False,
            )

            output_entry = json.loads(output_path.read_text(encoding="utf-8").strip())
            stats = output_entry["tool_call_stats"]
            self.assertEqual(stats["total_tool_calls"], 3)
            self.assertEqual(stats["distinct_tool_calls"], 2)
            self.assertEqual(stats["repeated_tool_calls"], 1)
            self.assertAlmostEqual(stats["repeat_rate"], 1 / 3)
            self.assertEqual(stats["tool_name_counts"], {"mock_tool": 3})
            self.assertEqual(stats["distinct_tool_calls_by_tool_name"], {"mock_tool": 2})
            self.assertEqual(summary["total_tool_calls"], 3)
            self.assertTrue(summary_path.exists())

    def test_backfills_component_detail_proactive_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            traj_dir = tmp_path / "run" / "traj"
            detail_dir = tmp_path / "run" / "summary" / "components" / "demo-2"
            traj_dir.mkdir(parents=True)
            detail_dir.mkdir(parents=True)
            input_path = traj_dir / "trajs.jsonl"
            output_path = traj_dir / "trajs_with_stats.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "instance_id": "demo-2",
                        "search_mode": "bfs_component_graph",
                        "messages": [{"role": "assistant", "content": "<trace_locs>\npkg/a.py\n</trace_locs>"}],
                    },
                    ensure_ascii=False,
                ) + "\n",
                encoding="utf-8",
            )
            (detail_dir / "component_0000.json").write_text(
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "_message_kind": "tool_result",
                                "content": (
                                    "Current node code has been proactively retrieved for you.\n"
                                    'Retrieval tool: get_code_of_file_function({"file_name":"pkg/a.py","func_name":"entry"})\n'
                                    "<current_node_code>\ndef entry(): pass\n</current_node_code>"
                                ),
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            backfill_tool_call_stats.backfill_file(
                str(input_path),
                str(output_path),
                summary_path=None,
                include_component_details=True,
                preserve_existing=False,
            )

            output_entry = json.loads(output_path.read_text(encoding="utf-8").strip())
            stats = output_entry["tool_call_stats"]
            self.assertEqual(stats["total_tool_calls"], 1)
            self.assertEqual(stats["distinct_tool_calls"], 1)
            self.assertEqual(stats["tool_name_counts"], {"get_code_of_file_function": 1})
            self.assertEqual(output_entry["tool_call_stats_source"]["component_detail_files"], 1)
            self.assertEqual(output_entry["tool_call_stats_source"]["component_detail_tool_call_records"], 1)


if __name__ == "__main__":
    unittest.main()
