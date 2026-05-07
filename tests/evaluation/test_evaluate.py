from __future__ import annotations

import json
import os
import tempfile
import unittest

from evaluation.evaluate import evaluate_file


class EvaluateFileTest(unittest.TestCase):
    def test_component_graph_usage_sums_component_details(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_component_graph_", dir="/tmp") as tmp_dir:
            run_dir = os.path.join(tmp_dir, "run")
            traj_dir = os.path.join(run_dir, "traj")
            summary_dir = os.path.join(run_dir, "summary")
            os.makedirs(traj_dir, exist_ok=True)
            input_path = os.path.join(traj_dir, "trajs.jsonl")
            output_path = os.path.join(traj_dir, "trajs_eval.jsonl")
            component_dir = os.path.join(summary_dir, "components", "inst1")
            os.makedirs(component_dir, exist_ok=True)
            with open(os.path.join(summary_dir, "inst1.json"), "w", encoding="utf-8") as f:
                json.dump({"components": {}}, f)

            for component_id, total_tokens, tool_error_count in (
                ("component_a", 30, 0),
                ("component_b", 70, 1),
            ):
                with open(
                    os.path.join(component_dir, f"{component_id}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "component_id": component_id,
                            "usage": {
                                "total_tokens": total_tokens,
                                "tool_tokens": total_tokens - 10,
                                "assistant_tokens": 10,
                                "total_turns": 4,
                                "tool_turns": 2,
                                "assistant_turns": 1,
                                "tool_execute_count": 2,
                                "tool_error_count": tool_error_count,
                            },
                        },
                        f,
                    )

            entry = {
                "instance_id": "inst1",
                "search_mode": "component_graph",
                "fail_ratio": 0.25,
                "ground_truth": ["pkg/a.py:foo"],
                "messages": [
                    {"role": "system", "content": "system", "token_count": 1},
                    {"role": "user", "content": "user", "token_count": 1},
                    {
                        "role": "assistant",
                        "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                        "token_count": 1,
                    },
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())
            with open(os.path.join(traj_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertEqual(evaluated_entry["metrics"]["usage"]["usage_source"], "component_details")
            self.assertEqual(evaluated_entry["metrics"]["usage"]["total_tokens"], 100)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["tool_execute_count"], 4)
            self.assertAlmostEqual(metrics["usage"]["avg_total_tokens"], 100.0)
            self.assertAlmostEqual(metrics["usage"]["avg_tool_execute_count"], 4.0)
            self.assertAlmostEqual(metrics["fail_ratio"], 0.25)

    def test_default_usage_uses_main_messages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_default_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "trajs.jsonl")
            output_path = os.path.join(tmp_dir, "trajs_eval.jsonl")
            entry = {
                "instance_id": "inst2",
                "search_mode": "default",
                "fail_ratio": 0.5,
                "ground_truth": ["pkg/b.py:bar"],
                "messages": [
                    {"role": "system", "content": "system", "token_count": 2},
                    {"role": "user", "_message_kind": "tool_result", "content": "tool output", "token_count": 3},
                    {
                        "role": "assistant",
                        "content": "<trace_locs>\npkg/b.py\nfunction: bar\n</trace_locs>",
                        "token_count": 5,
                    },
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())

            self.assertEqual(evaluated_entry["metrics"]["usage"]["usage_source"], "main_messages")
            self.assertEqual(evaluated_entry["metrics"]["usage"]["total_tokens"], 10)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["tool_tokens"], 3)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["assistant_tokens"], 5)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["tool_execute_count"], 1)

    def test_metrics_include_average_tool_call_stats(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_tool_call_stats_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "trajs.jsonl")
            output_path = os.path.join(tmp_dir, "trajs_eval.jsonl")
            entries = [
                {
                    "instance_id": "inst_tool_1",
                    "ground_truth": ["pkg/a.py:foo"],
                    "tool_call_stats": {
                        "total_tool_calls": 3,
                        "distinct_tool_calls": 2,
                        "repeated_tool_calls": 1,
                        "repeat_rate": 1 / 3,
                        "failed_tool_calls": 1,
                    },
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
                {
                    "instance_id": "inst_tool_2",
                    "ground_truth": ["pkg/b.py:bar"],
                    "tool_call_stats": {
                        "total_tool_calls": 1,
                        "distinct_tool_calls": 1,
                        "repeated_tool_calls": 0,
                        "repeat_rate": 0.0,
                        "failed_tool_calls": 0,
                    },
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/b.py\nfunction: bar\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
            ]
            with open(input_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(os.path.join(tmp_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertAlmostEqual(metrics["usage"]["avg_total_tool_calls"], 2.0)
            self.assertAlmostEqual(metrics["usage"]["avg_distinct_tool_calls"], 1.5)
            self.assertAlmostEqual(metrics["usage"]["avg_repeated_tool_calls"], 0.5)
            self.assertAlmostEqual(metrics["usage"]["avg_tool_repeat_rate"], 1 / 6)
            self.assertAlmostEqual(metrics["usage"]["avg_failed_tool_call_rate"], 1 / 6)

    def test_metrics_include_search_execution_mode_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_search_execution_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "trajs.jsonl")
            output_path = os.path.join(tmp_dir, "trajs_eval.jsonl")
            entries = [
                {
                    "instance_id": "inst_exec_1",
                    "search_execution_mode": "component_search_merge",
                    "ground_truth": ["pkg/a.py:foo"],
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
                {
                    "instance_id": "inst_exec_2",
                    "search_execution_mode": "default_search_no_component",
                    "ground_truth": ["pkg/b.py:bar"],
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/b.py\nfunction: bar\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
                {
                    "instance_id": "inst_exec_3",
                    "search_execution_mode": "component_search_with_oversized_default",
                    "ground_truth": ["pkg/c.py:baz"],
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/c.py\nfunction: baz\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
                {
                    "instance_id": "inst_exec_4",
                    "search_execution_mode": "component_search_merge",
                    "ground_truth": ["pkg/d.py:qux"],
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "<trace_locs>\npkg/d.py\nfunction: qux\n</trace_locs>",
                            "token_count": 1,
                        },
                    ],
                },
            ]
            with open(input_path, "w", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_first_entry = json.loads(f.readline())
            with open(os.path.join(tmp_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertEqual(evaluated_first_entry["search_execution_mode"], "component_search_merge")
            self.assertEqual(
                metrics["search_execution_mode_counts"],
                {
                    "component_search_merge": 2,
                    "component_search_with_oversized_default": 1,
                    "default_search_no_component": 1,
                },
            )

    def test_locagent_acc_requires_covering_available_ground_truth_at_k(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_locagent_acc_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "trajs.jsonl")
            output_path = os.path.join(tmp_dir, "trajs_eval.jsonl")
            entry = {
                "instance_id": "inst_acc",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "messages": [
                    {
                        "role": "assistant",
                        "content": "<trace_locs>\npkg/a.py\nfunction: foo\n</trace_locs>",
                        "token_count": 1,
                    },
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())
            with open(os.path.join(tmp_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertEqual(evaluated_entry["metrics"]["file_level"]["acc"]["top3"], 1.0)
            self.assertEqual(evaluated_entry["metrics"]["file_level"]["locagent_acc"]["top3"], 0.0)
            self.assertEqual(metrics["file_level"]["TOP 3"], 100.0)
            self.assertEqual(metrics["file_level"]["Acc@3"], 0.0)

    def test_metrics_include_macro_f1_for_files_and_functions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_f1_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "trajs.jsonl")
            output_path = os.path.join(tmp_dir, "trajs_eval.jsonl")
            entry = {
                "instance_id": "inst_f1",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "messages": [
                    {
                        "role": "assistant",
                        "content": (
                            "<trace_locs>\n"
                            "pkg/a.py\nfunction: foo\n\n"
                            "pkg/c.py\nfunction: baz\n"
                            "</trace_locs>"
                        ),
                        "token_count": 1,
                    },
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())
            with open(os.path.join(tmp_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)

            self.assertAlmostEqual(evaluated_entry["metrics"]["file_level"]["f1"], 0.5)
            self.assertAlmostEqual(evaluated_entry["metrics"]["function_level"]["f1"], 0.5)
            self.assertAlmostEqual(metrics["file_level"]["F1"], 50.0)
            self.assertAlmostEqual(metrics["function_level"]["F1"], 50.0)

    def test_name_guided_component_graph_usage_adds_main_messages(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_name_guided_", dir="/tmp") as tmp_dir:
            run_dir = os.path.join(tmp_dir, "run")
            traj_dir = os.path.join(run_dir, "traj")
            summary_dir = os.path.join(run_dir, "summary")
            os.makedirs(traj_dir, exist_ok=True)
            input_path = os.path.join(traj_dir, "trajs.jsonl")
            output_path = os.path.join(traj_dir, "trajs_eval.jsonl")
            component_dir = os.path.join(summary_dir, "components", "inst3")
            os.makedirs(component_dir, exist_ok=True)
            with open(os.path.join(component_dir, "component_a.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "component_id": "component_a",
                        "usage": {
                            "total_tokens": 40,
                            "tool_tokens": 20,
                            "assistant_tokens": 10,
                            "total_turns": 5,
                            "tool_turns": 1,
                            "assistant_turns": 2,
                            "tool_execute_count": 1,
                            "tool_error_count": 0,
                        },
                    },
                    f,
                )

            entry = {
                "instance_id": "inst3",
                "search_mode": "name_guided_component_graph",
                "fail_ratio": 0.0,
                "ground_truth": ["pkg/c.py:target"],
                "messages": [
                    {"role": "system", "content": "system", "token_count": 2},
                    {"role": "user", "content": "user", "token_count": 3},
                    {"role": "assistant", "content": "<component_name_query>{}</component_name_query>", "token_count": 5},
                    {"role": "assistant", "content": "<trace_locs>\npkg/c.py\nfunction: target\n</trace_locs>", "token_count": 7},
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())

            self.assertEqual(
                evaluated_entry["metrics"]["usage"]["usage_source"],
                "component_details_plus_main_messages",
            )
            self.assertEqual(evaluated_entry["metrics"]["usage"]["total_tokens"], 57)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["tool_execute_count"], 1)

    def test_name_guided_component_graph_usage_adds_separate_name_query_and_default_fallback_details(self) -> None:
        with tempfile.TemporaryDirectory(prefix="eval_name_guided_fallback_", dir="/tmp") as tmp_dir:
            run_dir = os.path.join(tmp_dir, "run")
            traj_dir = os.path.join(run_dir, "traj")
            summary_dir = os.path.join(run_dir, "summary")
            os.makedirs(traj_dir, exist_ok=True)
            input_path = os.path.join(traj_dir, "trajs.jsonl")
            output_path = os.path.join(traj_dir, "trajs_eval.jsonl")
            component_dir = os.path.join(summary_dir, "components", "inst4")
            name_query_dir = os.path.join(summary_dir, "name_query")
            default_fallback_dir = os.path.join(summary_dir, "default_fallback")
            os.makedirs(component_dir, exist_ok=True)
            os.makedirs(name_query_dir, exist_ok=True)
            os.makedirs(default_fallback_dir, exist_ok=True)
            with open(os.path.join(component_dir, "component_a.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "component_id": "component_a",
                        "usage": {
                            "total_tokens": 40,
                            "tool_tokens": 20,
                            "assistant_tokens": 10,
                            "total_turns": 5,
                            "tool_turns": 1,
                            "assistant_turns": 2,
                            "tool_execute_count": 1,
                            "tool_error_count": 0,
                        },
                    },
                    f,
                )
            with open(os.path.join(name_query_dir, "inst4.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "instance_id": "inst4",
                        "usage": {
                            "total_tokens": 23,
                            "tool_tokens": 0,
                            "assistant_tokens": 11,
                            "total_turns": 3,
                            "tool_turns": 0,
                            "assistant_turns": 1,
                            "tool_execute_count": 0,
                            "tool_error_count": 0,
                        },
                    },
                    f,
                )
            with open(os.path.join(default_fallback_dir, "inst4.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "instance_id": "inst4",
                        "usage": {
                            "total_tokens": 41,
                            "tool_tokens": 13,
                            "assistant_tokens": 17,
                            "total_turns": 3,
                            "tool_turns": 1,
                            "assistant_turns": 1,
                            "tool_execute_count": 1,
                            "tool_error_count": 0,
                        },
                    },
                    f,
                )

            entry = {
                "instance_id": "inst4",
                "search_mode": "name_guided_component_graph",
                "search_execution_mode": "component_search_with_oversized_default",
                "fail_ratio": 0.0,
                "ground_truth": ["pkg/c.py:target"],
                "messages": [
                    {"role": "assistant", "content": "<trace_locs>\npkg/c.py\nfunction: target\n</trace_locs>", "token_count": 7},
                ],
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            evaluate_file(input_path, output_path, None)

            with open(output_path, "r", encoding="utf-8") as f:
                evaluated_entry = json.loads(f.readline())

            self.assertEqual(
                evaluated_entry["metrics"]["usage"]["usage_source"],
                "name_query_detail_plus_component_details_plus_main_messages_plus_default_fallback_detail",
            )
            self.assertEqual(evaluated_entry["metrics"]["usage"]["total_tokens"], 111)
            self.assertEqual(evaluated_entry["metrics"]["usage"]["tool_execute_count"], 2)


if __name__ == "__main__":
    unittest.main()
