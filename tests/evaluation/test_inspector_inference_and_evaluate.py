from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from evaluation.inspector_evaluate import evaluate_entries
from evaluation.inspector_inference import InspectorInferenceRunner


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def chat_response(self, messages, **params):  # noqa: ANN001, ANN003
        del messages, params
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return _FakeResponse(response)


def _runner_args(tmp_dir: str, input_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        input_file=input_path,
        base_output_dir=tmp_dir,
        run_name="inspector_test",
        resume_output_dir="",
        model_backend="openai",
        model_name="fake",
        api_key=None,
        base_url=None,
        mock=False,
        max_turns=3,
        max_tokens=128,
        temperature=0.0,
        top_p=None,
        max_tool_response_length=10000,
        tokenizer_path="",
        tool_schemas_path="",
        project_file_loc="",
        graph_index_dir="",
        instance_ids="",
        instance_ids_file="",
        start_index=0,
        end_index=None,
        evaluate=False,
        eval_k_values="1,3,5",
    )


class InspectorEvaluateTest(unittest.TestCase):
    def test_function_level_acc_and_f1_at_k_use_full_locations(self) -> None:
        entries = [
            {
                "instance_id": "i1",
                "ground_truth": ["pkg/a.py:foo", "pkg/b.py:bar"],
                "pred_locations": ["pkg/a.py:foo", "pkg/c.py:baz", "pkg/b.py:bar"],
            }
        ]

        evaluated, metrics = evaluate_entries(entries, [1, 2, 3])

        function_metrics = evaluated[0]["metrics"]["function_level"]
        self.assertEqual(function_metrics["Acc@1"], 1.0)
        self.assertEqual(function_metrics["Acc@2"], 0.0)
        self.assertEqual(function_metrics["Acc@3"], 1.0)
        self.assertAlmostEqual(function_metrics["F1@1"], 2 / 3)
        self.assertAlmostEqual(function_metrics["F1@2"], 0.5)
        self.assertAlmostEqual(function_metrics["F1@3"], 0.8)
        self.assertAlmostEqual(metrics["function_level"]["Acc@1"], 100.0)
        self.assertAlmostEqual(metrics["function_level"]["Acc@2"], 0.0)
        self.assertAlmostEqual(metrics["function_level"]["F1@3"], 80.0)


class InspectorInferenceTest(unittest.TestCase):
    def test_reuses_exit_result_without_finalize_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="inspector_infer_", dir="/tmp") as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.jsonl")
            item = {
                "data_source": "code_localization_inspector",
                "reward_model": {"ground_truth": ["pkg/a.py:foo"]},
                "extra_info": {
                    "instance_id": "i1",
                    "repo": "owner/repo",
                    "problem_statement": "foo should handle the issue",
                    "entry_file": "pkg/a.py",
                },
            }
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")

            fake_llm = _FakeLLM(
                [
                    (
                        "<think>Done.</think>\n"
                        "<result>{\"suspicious\": ["
                        "{\"location\": \"pkg/a.py:foo\", \"reason\": \"matches issue\"}"
                        "]}</result>\n"
                        "<tool_call>{\"name\": \"exit\", \"arguments\": {}}</tool_call>"
                    )
                ]
            )
            runner = InspectorInferenceRunner(_runner_args(tmp_dir, input_path), llm=fake_llm)

            outputs = runner.run()

            self.assertEqual(fake_llm.calls, 1)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(outputs[0]["entry_file"], "pkg/a.py")
            self.assertEqual(outputs[0]["pred_locations"], ["pkg/a.py:foo"])
            self.assertEqual(outputs[0]["tool_call_stats"]["total_tool_calls"], 1)
            self.assertTrue(os.path.exists(runner.trajs_path))


if __name__ == "__main__":
    unittest.main()
