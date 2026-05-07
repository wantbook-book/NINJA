# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(str(text))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = []
        for message in messages:
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            rendered.append(f"<{role}>{content}</{role}>")
        joined = "\n".join(rendered)
        if tokenize:
            return self.encode(joined, add_special_tokens=False)
        return joined


class _DummyFunctionCallParser:
    def __init__(self, *args, **kwargs):
        pass

    def parse_non_stream(self, content):
        return content, []


def _build_args(tmp_dir: Path, **overrides):
    base = {
        "input_file": str(tmp_dir / "input.jsonl"),
        "model_name": "mock-model",
        "model_backend": "openai",
        "api_key": None,
        "base_url": None,
        "base_output_dir": str(tmp_dir / "outputs"),
        "run_name": "mock-run",
        "max_turns": 8,
        "max_tokens": None,
        "temperature": 0.2,
        "top_p": None,
        "tokenizer_path": "dummy-tokenizer",
        "search_mode": "default",
        "tool_schemas_path": None,
        "tool_call_parser_type": None,
        "mock": True,
        "tool_registry_name": "repo_search_tools",
        "finalize_prompt_path": "",
        "instance_ids": "",
        "instance_ids_file": "",
        "start_index": 0,
        "end_index": None,
        "resume_output_dir": "",
        "component_graph_dir": "",
        "component_summary_dir": "",
        "bfs_system_prompt_path": "",
        "bfs_max_tool_rounds_per_node": 2,
        "bfs_component_search_strategy": "ranked_groups",
        "bfs_group_preview_limit": 24,
        "bfs_component_parallelism": 4,
        "bfs_max_components_to_check": 0,
        "enable_component_summary_merge": False,
        "input_cost_per_million_tokens": 0.0,
        "output_cost_per_million_tokens": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tool_call_payload(tool_name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


class InferenceSearchModesTest(unittest.TestCase):
    def setUp(self):
        fake_models = ModuleType("evaluation.models")
        fake_models.build_model = lambda *args, **kwargs: None

        fake_sglang = ModuleType("sglang")
        fake_sglang_srt = ModuleType("sglang.srt")
        fake_sglang_function_call = ModuleType("sglang.srt.function_call")
        fake_sglang_function_call_parser = ModuleType("sglang.srt.function_call.function_call_parser")
        fake_sglang_function_call_parser.FunctionCallParser = _DummyFunctionCallParser
        fake_sglang_managers = ModuleType("sglang.srt.managers")
        fake_sglang_io_struct = ModuleType("sglang.srt.managers.io_struct")
        fake_tqdm = ModuleType("tqdm")
        fake_tqdm.tqdm = lambda iterable=None, **kwargs: iterable if iterable is not None else None
        fake_transformers = ModuleType("transformers")
        fake_regex = ModuleType("regex")
        fake_regex.compile = re.compile
        fake_regex.search = re.search
        fake_regex.sub = re.sub
        fake_regex.findall = re.findall
        fake_regex.split = re.split
        fake_regex.DOTALL = re.DOTALL
        fake_regex.IGNORECASE = re.IGNORECASE

        class _DummyAutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return _DummyTokenizer()

        fake_transformers.AutoTokenizer = _DummyAutoTokenizer
        fake_tools = ModuleType("tools")
        fake_tools.tools_registries = {"repo_search_tools": {}, "mock": {}}
        fake_debugpy = ModuleType("debugpy")
        fake_debugpy.listen = lambda *args, **kwargs: None
        fake_debugpy.wait_for_client = lambda *args, **kwargs: None

        class _DummyFunction:
            def __init__(self, name=None, description=None, parameters=None):
                self.name = name
                self.description = description
                self.parameters = parameters

        class _DummyTool:
            def __init__(self, type="function", function=None):
                self.type = type
                self.function = function

        fake_sglang_io_struct.Function = _DummyFunction
        fake_sglang_io_struct.Tool = _DummyTool

        self._module_patcher = mock.patch.dict(
            sys.modules,
            {
                "evaluation.models": fake_models,
                "sglang": fake_sglang,
                "sglang.srt": fake_sglang_srt,
                "sglang.srt.function_call": fake_sglang_function_call,
                "sglang.srt.function_call.function_call_parser": fake_sglang_function_call_parser,
                "sglang.srt.managers": fake_sglang_managers,
                "sglang.srt.managers.io_struct": fake_sglang_io_struct,
                "tqdm": fake_tqdm,
                "transformers": fake_transformers,
                "regex": fake_regex,
                "tools": fake_tools,
                "debugpy": fake_debugpy,
            },
        )
        self._module_patcher.start()
        self.addCleanup(self._module_patcher.stop)

        import evaluation.inference as inference

        self.inference = inference
        self._tokenizer_patcher = mock.patch.object(
            self.inference.AutoTokenizer,
            "from_pretrained",
            return_value=_DummyTokenizer(),
        )
        self._tokenizer_patcher.start()
        self.addCleanup(self._tokenizer_patcher.stop)

        self._parser_patcher = mock.patch.object(
            self.inference,
            "FunctionCallParser",
            _DummyFunctionCallParser,
        )
        self._parser_patcher.start()
        self.addCleanup(self._parser_patcher.stop)

    def test_default_search_mode_mock_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )
            saved_args = json.loads(Path(runner.args_path).read_text(encoding="utf-8"))
            self.assertEqual(saved_args["run_name"], "mock-run")
            self.assertEqual(saved_args["base_output_dir"], str(Path(args.base_output_dir).absolute()))
            self.assertNotIn("load_config", saved_args)

            loaded_args = self.inference.parse_args(
                [
                    "--load_config",
                    runner.args_path,
                    "--run_name",
                    "replay-run",
                    "--no-mock",
                ]
            )
            self.assertEqual(loaded_args.input_file, saved_args["input_file"])
            self.assertEqual(loaded_args.model_name, saved_args["model_name"])
            self.assertEqual(loaded_args.run_name, "replay-run")
            self.assertFalse(loaded_args.mock)
            self.assertEqual(loaded_args.load_config, str(Path(runner.args_path).absolute()))

            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": "demo-instance",
                "base_commit": "deadbeef",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": "demo-instance",
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }
            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "default")
            self.assertEqual(result["fail_ratio"], 0.0)
            self.assertTrue(any(message.get("_message_kind") == "tool_result" for message in result["messages"]))
            self.assertIn("<trace_locs>", result["messages"][-1]["content"])
            self.assertEqual(result["messages"][-1]["role"], "assistant")

    def test_default_search_mode_builds_fallback_prompt_when_prompt_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )

            item = {
                "prompt": [],
                "repo": "demo/repo",
                "instance_id": "default-fallback-prompt-instance",
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": "default-fallback-prompt-instance",
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py\npkg/b.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "default")
            self.assertEqual(result["messages"][0]["role"], "system")
            self.assertEqual(result["messages"][1]["role"], "user")
            self.assertIn("Helper returns the wrong transformed value.", result["messages"][1]["content"])
            self.assertIn("pkg/a.py", result["messages"][1]["content"])
            self.assertIn("<trace_locs>", result["messages"][-1]["content"])

    def test_run_updates_runtime_token_progress_postfix(self):
        class _DummyProgressBar:
            def __init__(self, iterable, **kwargs):
                self.items = list(iterable)
                self.postfix_history = []

            def __iter__(self):
                return iter(self.items)

            def set_postfix_str(self, payload, refresh=False):
                self.postfix_history.append(str(payload))

            def set_postfix(self, payload, refresh=False):
                self.postfix_history.append(str(payload))

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": "runtime-progress-instance",
                "base_commit": "deadbeef",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": "runtime-progress-instance",
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }
            input_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")

            captured_bars = []

            def _build_progress_bar(iterable, **kwargs):
                bar = _DummyProgressBar(iterable, **kwargs)
                captured_bars.append(bar)
                return bar

            args = _build_args(
                tmp_path,
                input_file=str(input_path),
                search_mode="default",
                input_cost_per_million_tokens=1_000_000.0,
                output_cost_per_million_tokens=1_000_000.0,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )

            with mock.patch.object(self.inference, "tqdm", side_effect=_build_progress_bar):
                outputs = runner.run()

            self.assertEqual(len(outputs), 1)
            self.assertEqual(len(captured_bars), 1)
            self.assertTrue(captured_bars[0].postfix_history)
            final_postfix = captured_bars[0].postfix_history[-1]
            self.assertIn("T", final_postfix)
            self.assertIn(" U", final_postfix)
            self.assertIn(" A", final_postfix)
            self.assertIn(" X", final_postfix)
            self.assertIn(" I", final_postfix)
            self.assertIn(" C", final_postfix)
            self.assertIn("$", final_postfix)
            runtime_cost = outputs[0]["runtime_cost"]
            self.assertEqual(runtime_cost["currency"], "USD")
            self.assertGreater(runtime_cost["model_input_tokens"], 0)
            self.assertGreater(runtime_cost["model_output_tokens"], 0)
            self.assertEqual(runtime_cost["trace_total_tokens"], runner.runtime_total_tokens)
            self.assertEqual(runtime_cost["trace_user_tokens"], runner.runtime_user_tokens)
            self.assertEqual(runtime_cost["trace_assistant_tokens"], runner.runtime_assistant_tokens)
            self.assertEqual(runtime_cost["trace_tool_tokens"], runner.runtime_tool_tokens)
            self.assertAlmostEqual(runtime_cost["total_cost_usd"], runner.runtime_total_cost_usd)
            self.assertGreater(runner.runtime_total_tokens, 0)
            self.assertEqual(
                runner.runtime_total_tokens,
                runner.runtime_user_tokens + runner.runtime_assistant_tokens + runner.runtime_tool_tokens,
            )
            self.assertAlmostEqual(
                runner.runtime_total_cost_usd,
                float(runner.runtime_input_tokens + runner.runtime_output_tokens),
            )
            self.assertEqual(runner.runtime_completed_issue_count, 1)

    def test_count_tokens_uses_tokenizer_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )
            message = {"role": "assistant", "content": "abc"}
            expected = len("<assistant>abc</assistant>")
            self.assertEqual(runner._count_tokens(message), expected)

    def test_function_level_trace_locs_only_removes_file_and_class_only_blocks(self):
        trace_locs = (
            "<trace_locs>\n"
            "pkg/file_only.py\n\n"
            "pkg/class_only.py\n"
            "class: Handler\n\n"
            "pkg/compact.py:Handler.run\n\n"
            "pkg/function.py\n"
            "function: helper\n"
            "</trace_locs>"
        )

        normalized = self.inference._function_level_trace_locs_only(trace_locs)

        self.assertIn("pkg/compact.py\nfunction: Handler.run", normalized)
        self.assertIn("pkg/function.py\nfunction: helper", normalized)
        self.assertNotIn("pkg/file_only.py", normalized)
        self.assertNotIn("class:", normalized)

    def test_model_calls_strip_internal_tool_feedback_metadata(self):
        class _CapturingLLM:
            def __init__(self) -> None:
                self.calls = []
                self.index = 0

            def chat_response(self, messages, **params):
                self.calls.append(copy.deepcopy(messages))
                payloads = [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="",
                                    tool_calls=[
                                        _tool_call_payload(
                                            "mock_tool",
                                            {"query": "search term"},
                                            "call_1",
                                        )
                                    ],
                                )
                            )
                        ],
                        usage=SimpleNamespace(total_tokens=10),
                    ),
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="<trace_locs>src/foo.py:Foo.bar</trace_locs>",
                                    tool_calls=None,
                                )
                            )
                        ],
                        usage=SimpleNamespace(total_tokens=10),
                    ),
                ]
                response = payloads[min(self.index, len(payloads) - 1)]
                self.index += 1
                return response

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(args, llm=_CapturingLLM())
            runner.tool_registry = {"mock_tool": lambda query: {"result": query}}

            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": "demo-instance",
                "base_commit": "deadbeef",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": "demo-instance",
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            runner._run_item(copy.deepcopy(item))

            self.assertGreaterEqual(len(runner.llm.calls), 2)
            second_call_messages = runner.llm.calls[1]
            self.assertTrue(any(message.get("content") == "Locate the bug." for message in second_call_messages))
            for message in second_call_messages:
                self.assertNotIn("tool_call_id", message)
                self.assertNotIn("_message_kind", message)

    def test_session_model_call_retries_five_times_after_provider_errors(self):
        class _FlakyLLM:
            def __init__(self) -> None:
                self.calls = 0

            def chat_response(self, messages, **params):
                self.calls += 1
                if self.calls <= 5:
                    raise RuntimeError("Bedrock is unable to process your request.")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                                tool_calls=None,
                            )
                        )
                    ],
                    usage=SimpleNamespace(total_tokens=10),
                )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            llm = _FlakyLLM()
            runner = self.inference.LocalizeRunner(args, llm=llm)
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[{"role": "user", "content": "Locate the bug."}],
                max_tokens=None,
            )

            with (
                mock.patch.object(self.inference.logger, "warning") as log_warning,
                mock.patch.object(self.inference.time, "sleep") as sleep,
            ):
                response = runner._session_call_model(session, session["messages"], None)

            self.assertEqual(llm.calls, 6)
            self.assertEqual(log_warning.call_count, 5)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [10, 20, 30, 40, 50])
            self.assertIn("<trace_locs>", response.choices[0].message.content)

    def test_session_call_model_does_not_override_max_tokens_from_session_budget(self):
        class _CapturingLLM:
            def __init__(self) -> None:
                self.params = None

            def chat_response(self, messages, **params):
                self.params = copy.deepcopy(params)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content="<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                                tool_calls=None,
                            )
                        )
                    ],
                    usage=SimpleNamespace(total_tokens=10),
                )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            llm = _CapturingLLM()
            runner = self.inference.LocalizeRunner(args, llm=llm)
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[{"role": "user", "content": "Locate the bug."}],
                max_tokens=32,
            )

            runner._session_call_model(session, session["messages"], None)

            self.assertEqual(llm.params, {"temperature": 0.2})

    def test_bfs_session_returns_tool_format_error_for_invalid_arguments_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM([]),
            )
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[],
                max_tokens=None,
            )

            stopped = runner._session_execute_bfs_tool_calls(
                session,
                [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_code_of_file_function",
                            "arguments": "file_name=pkg/a.py, func_name=entry",
                        },
                    }
                ],
            )

            self.assertFalse(stopped)
            self.assertEqual(session["tool_execute_count"], 1)
            self.assertEqual(session["tool_error_count"], 1)
            self.assertEqual(session["messages"][-1]["role"], "user")
            self.assertEqual(session["messages"][-1]["_message_kind"], "tool_result")
            self.assertIn("Wrong tool call format", session["messages"][-1]["content"])
            self.assertIn("<tool_call>", session["messages"][-1]["content"])
            self.assertIn("valid JSON object", session["messages"][-1]["content"])

    def test_bfs_session_exit_short_circuits_other_tool_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM([]),
            )
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[],
                max_tokens=None,
            )
            runner.tool_registry = {
                "mock_tool": lambda query, instance_id: self.fail("mock_tool should not be executed after exit"),
            }

            stopped = runner._session_execute_bfs_tool_calls(
                session,
                [
                    _tool_call_payload("mock_tool", {"query": "should-not-run"}, "call_1"),
                    _tool_call_payload("exit", {}, "call_2"),
                ],
            )

            self.assertTrue(stopped)
            self.assertEqual(session["tool_execute_count"], 0)
            self.assertEqual(session["tool_error_count"], 0)
            self.assertEqual(session["messages"], [])
            self.assertEqual(len(session["tool_call_records"]), 1)
            self.assertEqual(session["tool_call_records"][0]["tool_name"], "exit")

    def test_bfs_session_tool_feedback_matches_default_tool_feedback_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM([]),
            )
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[],
                max_tokens=None,
            )
            runner.tool_registry = {
                "mock_tool": lambda query, instance_id: {"result": f"{instance_id}:{query}"},
            }

            stopped = runner._session_execute_bfs_tool_calls(
                session,
                [_tool_call_payload("mock_tool", {"query": "needle"}, "call_1")],
            )

            self.assertFalse(stopped)
            self.assertEqual(session["tool_execute_count"], 1)
            feedback = session["messages"][-1]["content"]
            self.assertIn("Here is a result of a function/class code retrived by 'mock_tool'", feedback)
            self.assertIn("<code>", feedback)
            self.assertIn("do not repeat a previously executed tool call", feedback)
            self.assertNotIn("Tool result for mock_tool", feedback)

    def test_session_tool_interaction_loop_pops_generation_that_exceeds_token_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = _build_args(tmp_path, search_mode="default")
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(
                    [
                        {
                            "content": "this assistant response is intentionally too long",
                            "tool_calls": None,
                            "usage": {"total_tokens": 10},
                        }
                    ]
                ),
            )
            session = runner._create_bfs_session(
                instance_id="demo-instance",
                seed_messages=[],
                max_tokens=1,
            )

            stop_reason = runner._run_session_tool_interaction_loop(session)

            self.assertEqual(stop_reason, "token_budget_exhausted")
            self.assertEqual(session["messages"], [])
            self.assertEqual(session["token_counts"], [])

    def test_trace_locs_from_last_message_ignores_earlier_trace_locs(self):
        messages = [
            {"role": "assistant", "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>"},
            {"role": "assistant", "content": "No final trace yet."},
        ]

        trace_locs = self.inference.LocalizeRunner._trace_locs_from_last_message(messages)

        self.assertEqual(trace_locs, "<trace_locs>\n</trace_locs>")

    def test_trace_locs_from_last_message_uses_last_assistant_message(self):
        messages = [
            {"role": "assistant", "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>"},
            {"role": "user", "content": "tool feedback"},
            {"role": "assistant", "content": "<trace_locs>\npkg/b.py\nfunction: worker\n</trace_locs>"},
            {"role": "user", "content": "trailing observation"},
        ]

        trace_locs = self.inference.LocalizeRunner._trace_locs_from_last_message(messages)

        self.assertEqual(trace_locs, "<trace_locs>\npkg/b.py\nfunction: worker\n</trace_locs>")

    def test_bfs_component_graph_parallelism_one_runs_without_thread_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "sequential-bfs-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {},
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<node_decision>"
                        "{\"decision\": \"finish_search\", "
                        "\"is_relevant\": true, "
                        "\"confidence\": 0.9, "
                        "\"reason\": \"Entry is relevant.\", "
                        "\"functionality_hint\": \"Entry flow\", "
                        "\"suspected_locations\": ["
                        "{\"node_id\": \"pkg/a.py:entry\", \"reason\": \"Likely bug entry.\", \"confidence\": 0.9}"
                        "]}"
                        "</node_decision>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": (
                        "<component_summary>\n"
                        "{\"functionality_summary\": \"Entry flow\"}\n"
                        "</component_summary>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="bfs_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }

            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Entry returns the wrong value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:entry"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py\npkg/b.py",
                },
            }

            with mock.patch.object(
                self.inference,
                "ThreadPoolExecutor",
                side_effect=AssertionError("ThreadPoolExecutor should not be used when effective_parallelism=1"),
            ):
                result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["bfs_component_parallelism"], 1)
            self.assertEqual(len(result["component_search_results"]), 1)
            self.assertEqual(result["component_search_results"][0]["component_id"], "component_0000")
            self.assertEqual(len(result["messages"]), 3)
            self.assertEqual(result["messages"][-1]["role"], "assistant")
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>")
            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>",
            )
            detail_payload = json.loads(
                (
                    Path(runner.component_summary_dir)
                    / "components"
                    / instance_id
                    / "component_0000.json"
                ).read_text(encoding="utf-8")
            )
            user_messages = [
                message["content"]
                for message in detail_payload["messages"]
                if message.get("role") == "user"
            ]
            self.assertTrue(any("<current_node_code>" in content for content in user_messages))
            self.assertTrue(any("def entry():" in content for content in user_messages))
            summary_trace = detail_payload.get("component_summary_trace", {})
            self.assertEqual(summary_trace.get("reused_existing_summary"), False)
            self.assertGreater(len(summary_trace.get("summary_messages", [])), 0)
            self.assertEqual(summary_trace.get("merge_messages", []), [])

    def test_component_graph_search_mode_merges_per_component_trace_locs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "component-graph-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "entry",
                            },
                            "pkg/a.py:helper": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "helper",
                            },
                            "pkg/b.py:worker": {
                                "file_path": "pkg/b.py",
                                "qualified_name": "worker",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                                "graph_edge_list": [["pkg/a.py:entry", "pkg/a.py:helper"]],
                            },
                            {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/b.py:worker"],
                                "node_ids": ["pkg/b.py:worker"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": "I need to inspect the helper implementation.",
                    "tool_calls": [
                        _tool_call_payload(
                            "get_code_of_file_function",
                            {"file_name": "pkg/a.py", "func_name": "helper"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "This component does not look related.",
                    "tool_calls": [
                        _tool_call_payload(
                            "exit",
                            {},
                            "call_2",
                        )
                    ],
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "<trace_locs>\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 18},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }

            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py\npkg/b.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "component_graph")
            self.assertEqual(result["search_execution_mode"], "component_search_merge")
            self.assertEqual(result["bfs_component_parallelism"], 1)
            self.assertEqual(runner.tool_execute_count, 1)
            self.assertEqual(len(result["messages"]), 2)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>")
            self.assertEqual(len(result["component_search_results"]), 2)
            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
            )
            self.assertEqual(
                result["component_search_results"][1]["trace_locs"],
                "<trace_locs>\n</trace_locs>",
            )
            self.assertEqual(
                result["component_search_results"][1]["stop_reason"],
                "exit_requested",
            )

            detail_payload = json.loads(
                (
                    Path(runner.component_summary_dir)
                    / "components"
                    / instance_id
                    / "component_0000.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(detail_payload["component_search_result"]["trace_locs"], "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>")
            self.assertTrue(any(message.get("_message_kind") == "tool_result" for message in detail_payload["messages"]))
            self.assertTrue(
                any(
                    "pkg/a.py:entry -> pkg/a.py:helper" in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )
            main_usage = runner._build_usage_from_traced_messages(
                runner._messages_with_token_counts(result["messages"])
            )
            detail_dir = Path(runner.component_summary_dir) / "components" / instance_id
            detail_usage_total = 0
            for detail_path in detail_dir.glob("*.json"):
                detail_usage_total += int(
                    json.loads(detail_path.read_text(encoding="utf-8"))["usage"]["total_tokens"]
                )
            self.assertEqual(
                runner.runtime_total_tokens,
                int(main_usage.get("total_tokens", 0) or 0) + detail_usage_total,
            )
            self.assertEqual(
                runner.runtime_total_tokens,
                runner.runtime_user_tokens + runner.runtime_assistant_tokens + runner.runtime_tool_tokens,
            )
            self.assertEqual(runner.runtime_component_count, 2)
            self.assertGreater(runner.runtime_component_tokens, 0)

    def test_component_graph_search_finalizes_after_no_tool_response_without_trace_locs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "component-finalize-after-no-tool-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:helper": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "helper",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:helper"],
                                "node_ids": ["pkg/a.py:helper"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": "The helper node looks relevant, but I should now provide the final location.",
                    "tool_calls": None,
                    "usage": {"total_tokens": 18},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 12},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
            )
            detail_payload = json.loads(
                (Path(runner.component_summary_dir) / "components" / instance_id / "component_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                any(
                    "Finish the current component now." in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )

    def test_component_graph_empty_components_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "component-graph-empty-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {},
                        "components": [],
                    }
                ),
                encoding="utf-8",
            )

            args = _build_args(
                tmp_path,
                search_mode="component_graph",
                component_graph_dir=str(component_graph_dir),
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )
            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the helper bug."},
                ],
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "default")
            self.assertEqual(result["search_execution_mode"], "default_search_no_component")
            self.assertNotIn("component_search_results", result)
            self.assertTrue(any(message.get("_message_kind") == "tool_result" for message in result["messages"]))
            self.assertEqual(result["messages"][-1]["role"], "assistant")
            self.assertIn("<trace_locs>", result["messages"][-1]["content"])

    def test_name_guided_component_graph_filters_components_by_predicted_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "name-guided-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "entry",
                            },
                            "pkg/a.py:helper": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "helper",
                            },
                            "pkg/b.py:worker": {
                                "file_path": "pkg/b.py",
                                "qualified_name": "worker",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                                "graph_edge_list": [["pkg/a.py:entry", "pkg/a.py:helper"]],
                            },
                            {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/b.py:worker"],
                                "node_ids": ["pkg/b.py:worker"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<component_name_query>"
                        "{\"file_names\": [\"pkg/a.py\", \"pkg/missing.py\"], "
                        "\"function_names\": [\"helper\", \"ghost_func\"], "
                        "\"class_names\": [\"GhostClass\"]}"
                        "</component_name_query>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 20},
                },
                {
                    "content": "I should inspect the helper body first.",
                    "tool_calls": [
                        _tool_call_payload(
                            "get_code_of_file_function",
                            {"file_name": "pkg/a.py", "func_name": "helper"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 28},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="name_guided_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }
            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "name_guided_component_graph")
            self.assertEqual(result["search_execution_mode"], "component_search_merge")
            self.assertEqual(result["bfs_component_parallelism"], 1)
            self.assertEqual(len(result["messages"]), 1)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>")
            name_query_detail = json.loads(Path(result["name_query_detail_path"]).read_text(encoding="utf-8"))
            self.assertEqual(name_query_detail["messages"][-1]["content"], script[0]["content"])
            self.assertEqual(runner.tool_execute_count, 1)
            detail_dir = Path(runner.component_summary_dir) / "components" / instance_id
            self.assertTrue((detail_dir / "component_0000.json").exists())
            self.assertFalse((detail_dir / "component_0001.json").exists())
            detail_payload = json.loads((detail_dir / "component_0000.json").read_text(encoding="utf-8"))
            self.assertTrue(
                any(
                    "Suspicious names from issue planning:" in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )
            self.assertTrue(
                any(
                    '"function_names": ["helper"]' in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )
            self.assertTrue(
                any(
                    '"file_names": ["pkg/a.py"]' in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )
            self.assertFalse(
                any(
                    '"class_names":' in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )
            self.assertFalse(
                any(
                    "ghost_func" in message.get("content", "") or "pkg/missing.py" in message.get("content", "")
                    for message in detail_payload["messages"]
                    if message.get("role") == "user"
                )
            )

    def test_name_guided_component_graph_filters_before_component_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "name-guided-limit-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "entry",
                            },
                            "pkg/b.py:worker": {
                                "file_path": "pkg/b.py",
                                "qualified_name": "worker",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry"],
                                "graph_edge_list": [],
                            },
                            {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/b.py:worker"],
                                "node_ids": ["pkg/b.py:worker"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<component_name_query>"
                        "{\"file_names\": [\"pkg/b.py\"], \"function_names\": [\"worker\"], \"class_names\": []}"
                        "</component_name_query>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 20},
                },
                {
                    "content": "Inspect the worker implementation.",
                    "tool_calls": [
                        _tool_call_payload(
                            "get_code_of_file_function",
                            {"file_name": "pkg/b.py", "func_name": "worker"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 28},
                },
                {
                    "content": "<trace_locs>\npkg/b.py\nfunction: worker\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="name_guided_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
                bfs_component_search_strategy="original",
                bfs_max_components_to_check=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }
            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "The worker path handles the wrong branch.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/b.py:worker"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/b.py\nfunction: worker\n</trace_locs>")
            detail_dir = Path(runner.component_summary_dir) / "components" / instance_id
            self.assertFalse((detail_dir / "component_0000.json").exists())
            self.assertTrue((detail_dir / "component_0001.json").exists())

    def test_name_guided_component_graph_filters_unmatched_multi_node_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "name-guided-keep-multi-node-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "entry",
                            },
                            "pkg/a.py:helper": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "helper",
                            },
                            "pkg/c.py:entry": {
                                "file_path": "pkg/c.py",
                                "qualified_name": "entry",
                            },
                            "pkg/c.py:worker": {
                                "file_path": "pkg/c.py",
                                "qualified_name": "worker",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                                "graph_edge_list": [["pkg/a.py:entry", "pkg/a.py:helper"]],
                            },
                            {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/c.py:entry"],
                                "node_ids": ["pkg/c.py:entry", "pkg/c.py:worker"],
                                "graph_edge_list": [["pkg/c.py:entry", "pkg/c.py:worker"]],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<component_name_query>"
                        "{\"file_names\": [\"pkg/a.py\"], \"function_names\": [\"helper\"], \"class_names\": []}"
                        "</component_name_query>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 20},
                },
                {
                    "content": "I should inspect the helper body first.",
                    "tool_calls": [
                        _tool_call_payload(
                            "get_code_of_file_function",
                            {"file_name": "pkg/a.py", "func_name": "helper"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 28},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "<trace_locs>\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 18},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="name_guided_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }
            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py\npkg/c.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "name_guided_component_graph")
            self.assertEqual(len(result["component_search_results"]), 1)
            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
            )
            detail_dir = Path(runner.component_summary_dir) / "components" / instance_id
            self.assertTrue((detail_dir / "component_0000.json").exists())
            self.assertFalse((detail_dir / "component_0001.json").exists())

    def test_name_guided_component_graph_falls_back_to_default_when_no_components_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "name-guided-default-fallback-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {
                                "file_path": "pkg/a.py",
                                "qualified_name": "entry",
                            },
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<component_name_query>"
                        "{\"file_names\": [\"pkg/z.py\"], \"function_names\": [\"missing\"], \"class_names\": []}"
                        "</component_name_query>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 20},
                },
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call_payload(
                            "mock_tool",
                            {"query": "search term"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": "<trace_locs>\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 16},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="name_guided_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {"mock_tool": self.inference._mock_tool}

            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/a.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "name_guided_component_graph")
            self.assertEqual(result["search_execution_mode"], "default_search_no_component")
            self.assertEqual(result["component_search_results"], [])
            self.assertEqual(len(result["messages"]), 1)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>")
            self.assertNotIn("default_fallback_messages", result)
            name_query_detail = json.loads(Path(result["name_query_detail_path"]).read_text(encoding="utf-8"))
            self.assertEqual(name_query_detail["messages"][-1]["content"], script[0]["content"])
            fallback_detail = json.loads(Path(result["default_fallback_detail_path"]).read_text(encoding="utf-8"))
            self.assertTrue(any(message.get("_message_kind") == "tool_result" for message in fallback_detail["messages"]))

    def test_name_guided_component_graph_skips_oversized_components_and_merges_default_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "name-guided-oversized-component-instance"
            oversized_nodes = {
                f"pkg/large.py:func_{index}": {
                    "file_path": "pkg/large.py",
                    "qualified_name": f"func_{index}",
                }
                for index in range(101)
            }
            payload_nodes = {
                **oversized_nodes,
                "pkg/small.py:entry": {
                    "file_path": "pkg/small.py",
                    "qualified_name": "entry",
                },
                "pkg/small.py:helper": {
                    "file_path": "pkg/small.py",
                    "qualified_name": "helper",
                },
            }
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": payload_nodes,
                        "components": [
                            {
                                "component_id": "component_large",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/large.py:func_0"],
                                "node_ids": [f"pkg/large.py:func_{index}" for index in range(101)],
                                "graph_edge_list": [],
                            },
                            {
                                "component_id": "component_small",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/small.py:entry"],
                                "node_ids": ["pkg/small.py:entry", "pkg/small.py:helper"],
                                "graph_edge_list": [["pkg/small.py:entry", "pkg/small.py:helper"]],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "<component_name_query>"
                        "{\"file_names\": [\"pkg/large.py\", \"pkg/small.py\"], "
                        "\"function_names\": [\"helper\"], \"class_names\": []}"
                        "</component_name_query>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 20},
                },
                {
                    "content": "<trace_locs>\npkg/small.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "<trace_locs>\npkg/small.py\nfunction: helper\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
                {
                    "content": "",
                    "tool_calls": [
                        _tool_call_payload(
                            "mock_tool",
                            {"query": "search term"},
                            "call_1",
                        )
                    ],
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": "<trace_locs>\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 16},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="name_guided_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {"mock_tool": self.inference._mock_tool}

            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/small.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                    "structure": "pkg/large.py\npkg/small.py",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "name_guided_component_graph")
            self.assertEqual(result["search_execution_mode"], "component_search_with_oversized_default")
            self.assertEqual(len(result["component_search_results"]), 1)
            self.assertEqual(result["component_search_results"][0]["component_id"], "component_small")
            self.assertEqual(
                result["messages"][-1]["content"],
                "<trace_locs>\npkg/small.py\nfunction: helper\n\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>",
            )
            self.assertFalse(any(message.get("_message_kind") == "tool_result" for message in result["messages"]))
            self.assertNotIn("default_fallback_messages", result)
            self.assertEqual(
                json.loads(Path(result["default_fallback_detail_path"]).read_text(encoding="utf-8"))["trace_locs"],
                "<trace_locs>\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>",
            )
            detail_dir = Path(runner.component_summary_dir) / "components" / instance_id
            self.assertFalse((detail_dir / "component_large.json").exists())
            self.assertTrue((detail_dir / "component_small.json").exists())

    def test_default_search_records_distinct_tool_call_stats_by_name_and_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            script = [
                {
                    "content": "",
                    "tool_calls": [_tool_call_payload("mock_tool", {"query": "same"}, "call_1")],
                    "usage": {"total_tokens": 10},
                },
                {
                    "content": "",
                    "tool_calls": [_tool_call_payload("mock_tool", {"query": "same"}, "call_2")],
                    "usage": {"total_tokens": 10},
                },
                {
                    "content": "",
                    "tool_calls": [_tool_call_payload("mock_tool", {"query": "different"}, "call_3")],
                    "usage": {"total_tokens": 10},
                },
                {
                    "content": "Need final answer.",
                    "tool_calls": None,
                    "usage": {"total_tokens": 10},
                },
                {
                    "content": "<trace_locs>\nsrc/foo.py\nfunction: Foo.bar\n</trace_locs>",
                    "tool_calls": None,
                    "usage": {"total_tokens": 10},
                },
            ]
            args = _build_args(tmp_path, search_mode="default", mock=True)
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {"mock_tool": self.inference._mock_tool}

            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the bug."},
                ],
                "repo": "demo/repo",
                "instance_id": "tool-call-stats-instance",
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
            }

            result = runner._run_item(copy.deepcopy(item))
            stats = result["tool_call_stats"]

            self.assertEqual(stats["total_tool_calls"], 3)
            self.assertEqual(stats["distinct_tool_calls"], 2)
            self.assertEqual(stats["repeated_tool_calls"], 1)
            self.assertAlmostEqual(stats["repeat_rate"], 1 / 3)
            self.assertEqual(stats["tool_name_counts"], {"mock_tool": 3})
            self.assertEqual(stats["distinct_tool_calls_by_tool_name"], {"mock_tool": 2})

    def test_run_supports_subset_selection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            items = []
            for instance_id in ("inst-a", "inst-b", "inst-c", "inst-d"):
                items.append(
                    {
                        "prompt": [
                            {"role": "system", "content": "You are a locator."},
                            {"role": "user", "content": f"Locate bug for {instance_id}."},
                        ],
                        "repo": "demo/repo",
                        "instance_id": instance_id,
                        "base_commit": "deadbeef",
                        "data_source": "mock",
                        "reward_model": {"ground_truth": [f"src/{instance_id}.py:Foo.bar"]},
                        "extra_info": {
                            "instance_id": instance_id,
                            "repo": "demo/repo",
                            "base_commit": "deadbeef",
                        },
                    }
                )
            with open(input_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            selected_ids_file = tmp_path / "selected_ids.txt"
            selected_ids_file.write_text("inst-b\ninst-d\n", encoding="utf-8")

            args = _build_args(
                tmp_path,
                input_file=str(input_path),
                search_mode="default",
                start_index=1,
                end_index=4,
                instance_ids_file=str(selected_ids_file),
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )

            outputs = runner.run()

            self.assertEqual([output["instance_id"] for output in outputs], ["inst-b", "inst-d"])
            traj_lines = [
                json.loads(line)
                for line in Path(runner.trajs_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([record["instance_id"] for record in traj_lines], ["inst-b", "inst-d"])

            resume_args = self.inference.parse_args(
                [
                    "--resume_output_dir",
                    runner.output_dir,
                    "--instance_ids",
                    "inst-a,inst-b,inst-c,inst-d",
                    "--start_index",
                    "0",
                    "--end_index",
                    "4",
                ]
            )
            resumed_runner = self.inference.LocalizeRunner(
                resume_args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )

            resumed_outputs = resumed_runner.run()

            self.assertEqual(resumed_runner.output_dir, runner.output_dir)
            self.assertEqual([output["instance_id"] for output in resumed_outputs], ["inst-a", "inst-c"])
            all_traj_lines = [
                json.loads(line)
                for line in Path(runner.trajs_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [record["instance_id"] for record in all_traj_lines],
                ["inst-b", "inst-d", "inst-a", "inst-c"],
            )
            sample_output = json.loads(Path(runner.traj_sample_path).read_text(encoding="utf-8"))
            self.assertEqual(sample_output["instance_id"], "inst-b")

    def test_extract_tagged_json_does_not_accept_bare_json(self):
        self.assertIsNone(
            self.inference._extract_tagged_json(
                "{\"decision\": \"finish_search\"}",
                "node_decision",
            )
        )
        self.assertEqual(
            self.inference._extract_tagged_json(
                "<node_decision>{\"decision\": \"finish_search\"}</node_decision>",
                "node_decision",
            ),
            {"decision": "finish_search"},
        )

    def test_bfs_component_graph_search_mode_reuses_existing_component_summary_for_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "demo-instance"
            component_graph_file = component_graph_dir / f"{instance_id}.json"
            component_graph_payload = {
                "instance_id": instance_id,
                "repo": "demo/repo",
                "base_commit": "deadbeef",
                "nodes": {
                    "pkg/a.py:entry": {},
                    "pkg/a.py:helper": {},
                    "pkg/b.py:Worker.run": {
                        "parent_type": "class",
                    },
                },
                "components": [
                    {
                        "component_id": "component_0000",
                        "entry_kind": "root",
                        "entry_node_ids": ["pkg/a.py:entry"],
                        "node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                        "graph_edge_list": [["pkg/a.py:entry", "pkg/a.py:helper"]],
                    },
                    {
                        "component_id": "component_0001",
                        "entry_kind": "root",
                        "entry_node_ids": ["pkg/b.py:Worker.run"],
                        "node_ids": ["pkg/b.py:Worker.run"],
                        "graph_edge_list": [],
                    },
                ],
            }
            component_graph_file.write_text(json.dumps(component_graph_payload), encoding="utf-8")

            script = [
                {
                    "content": (
                        "Entry coordinates the user-facing workflow.\n"
                        "<node_decision>"
                        "{\"decision\": \"continue_current_component\", "
                        "\"is_relevant\": false, "
                        "\"confidence\": 0.35, "
                        "\"reason\": \"Entry is orchestration only.\", "
                        "\"functionality_hint\": \"Entry path orchestration\", "
                        "\"suspected_locations\": []}"
                        "</node_decision>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": (
                        "Helper applies the suspicious transformation.\n"
                        "<node_decision>"
                        "{\"decision\": \"switch_to_next_component\", "
                        "\"is_relevant\": true, "
                        "\"confidence\": 0.91, "
                        "\"reason\": \"Helper contains the likely faulty transformation.\", "
                        "\"functionality_hint\": \"Transformation helper\", "
                        "\"suspected_locations\": ["
                        "{\"node_id\": \"pkg/a.py:helper\", \"reason\": \"Likely bug site.\", \"confidence\": 0.91}"
                        "]}"
                        "</node_decision>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 32},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="bfs_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            self.assertEqual(Path(runner.component_summary_dir), Path(runner.output_dir) / "summary")
            self.assertEqual(Path(runner.trajs_path), Path(runner.output_dir) / "traj" / "trajs.jsonl")
            self.assertEqual(Path(runner.traj_sample_path), Path(runner.output_dir) / "traj" / "traj_sample.json")

            existing_summary_file = Path(runner.component_summary_dir) / f"{instance_id}.json"
            existing_summary_file.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "component_graph_file": str(component_graph_file),
                        "updated_at": "2026-03-26T00:00:00",
                        "components": {
                            "component_0000": {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                                "checked_node_ids": ["pkg/a.py:entry", "pkg/a.py:helper"],
                                "functionality_summary": "Entry flow transformation helper chain",
                                "updated_at": "2026-03-26T00:00:00",
                            },
                            "component_0001": {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/b.py:Worker.run"],
                                "node_ids": ["pkg/b.py:Worker.run"],
                                "checked_node_ids": ["pkg/b.py:Worker.run"],
                                "functionality_summary": "Background worker execution path",
                                "updated_at": "2026-03-26T00:00:00",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
                "get_code_of_class_function": lambda file_name, class_name, func_name, instance_id: (
                    f"class {class_name}:\n    def {func_name}(self):\n        return '{instance_id}:{file_name}:{class_name}.{func_name}'"
                ),
            }

            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Helper returns the wrong transformed value in the entry flow.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:helper"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "bfs_component_graph")
            self.assertEqual(result["search_execution_mode"], "component_search_merge")
            self.assertEqual(result["bfs_component_parallelism"], 1)
            self.assertEqual(result["fail_ratio"], 0.0)
            self.assertEqual(runner.tool_execute_count, 2)
            self.assertTrue(result["component_summary_path"].endswith(f"{instance_id}.json"))
            self.assertNotIn("component_brief_path", result)
            self.assertEqual(len(result["messages"]), 3)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>")

            saved_summary = json.loads(existing_summary_file.read_text(encoding="utf-8"))
            component_0 = saved_summary["components"]["component_0000"]
            component_1 = saved_summary["components"]["component_0001"]

            self.assertEqual(saved_summary["updated_at"], "2026-03-26T00:00:00")
            self.assertNotIn("merged_from_existing", component_0)
            self.assertEqual(component_0["functionality_summary"], "Entry flow transformation helper chain")
            self.assertEqual(component_0["checked_node_ids"], ["pkg/a.py:entry", "pkg/a.py:helper"])
            self.assertNotIn("issue_relevance", component_0)
            self.assertNotIn("issue_relevance_reason", component_0)
            self.assertNotIn("likely_related_locations", component_0)
            self.assertNotIn("node_findings", component_0)
            self.assertEqual(component_1["functionality_summary"], "Background worker execution path")
            self.assertEqual(len(result["component_search_results"]), 1)
            self.assertEqual(result["component_search_results"][0]["component_id"], "component_0000")
            self.assertEqual(
                [location["node_id"] for location in result["component_search_results"][0]["suspected_locations"]],
                ["pkg/a.py:helper"],
            )
            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: helper\n</trace_locs>",
            )
            detail_payload = json.loads(
                (Path(runner.component_summary_dir) / "components" / instance_id / "component_0000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(detail_payload["component_summary_reused"])
            self.assertFalse(detail_payload["component_summary_saved"])
            self.assertTrue(detail_payload["summary_complete"])
            self.assertEqual(detail_payload["component_summary_trace"]["summary_messages"], [])

    def test_bfs_component_graph_does_not_save_summary_when_summary_generation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "summary-failure-instance"
            component_graph_file = component_graph_dir / f"{instance_id}.json"
            component_graph_file.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {},
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "Entry looks suspicious.\n"
                        "<node_decision>"
                        "{\"decision\": \"finish_search\", "
                        "\"is_relevant\": true, "
                        "\"confidence\": 0.88, "
                        "\"reason\": \"This entry is likely related.\", "
                        "\"functionality_hint\": \"Primary entry flow\", "
                        "\"suspected_locations\": ["
                        "{\"node_id\": \"pkg/a.py:entry\", \"reason\": \"Likely bug entry.\", \"confidence\": 0.88}"
                        "]}"
                        "</node_decision>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": "not a valid summary payload",
                    "tool_calls": None,
                    "usage": {"total_tokens": 24},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="bfs_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_parallelism=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }

            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Entry returns the wrong value.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:entry"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            summary_payload = json.loads(Path(result["component_summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["components"], {})
            self.assertNotIn("component_brief_path", result)
            self.assertEqual(len(result["messages"]), 3)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>")

            detail_path = Path(runner.component_summary_dir) / "components" / instance_id / "component_0000.json"
            detail_payload = json.loads(detail_path.read_text(encoding="utf-8"))
            self.assertIsNone(detail_payload["component_summary"])
            self.assertFalse(detail_payload["component_summary_saved"])
            self.assertFalse(detail_payload["summary_complete"])
            self.assertEqual(detail_payload["component_search_result"]["component_id"], "component_0000")
            self.assertEqual(
                detail_payload["component_search_result"]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>",
            )
            self.assertGreater(len(detail_payload["component_summary_trace"]["summary_messages"]), 0)
            self.assertIn(
                "not a valid summary payload",
                detail_payload["component_summary_trace"]["summary_messages"][-1]["content"],
            )

    def test_bfs_component_graph_empty_components_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "bfs-empty-instance"
            (component_graph_dir / f"{instance_id}.json").write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {},
                        "components": [],
                    }
                ),
                encoding="utf-8",
            )

            args = _build_args(
                tmp_path,
                search_mode="bfs_component_graph",
                component_graph_dir=str(component_graph_dir),
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(self.inference._default_mock_script()),
            )
            item = {
                "prompt": [
                    {"role": "system", "content": "You are a locator."},
                    {"role": "user", "content": "Locate the helper bug."},
                ],
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["src/foo.py:Foo.bar"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "default")
            self.assertEqual(result["search_execution_mode"], "default_search_no_component")
            self.assertNotIn("component_search_results", result)
            self.assertTrue(any(message.get("_message_kind") == "tool_result" for message in result["messages"]))
            self.assertEqual(result["messages"][-1]["role"], "assistant")
            self.assertIn("<trace_locs>", result["messages"][-1]["content"])

    def test_original_strategy_respects_bfs_max_components_to_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            component_graph_dir = tmp_path / "component_graphs"
            component_graph_dir.mkdir()

            instance_id = "limited-instance"
            component_graph_file = component_graph_dir / f"{instance_id}.json"
            component_graph_file.write_text(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "repo": "demo/repo",
                        "base_commit": "deadbeef",
                        "nodes": {
                            "pkg/a.py:entry": {},
                            "pkg/b.py:worker": {},
                        },
                        "components": [
                            {
                                "component_id": "component_0000",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/a.py:entry"],
                                "node_ids": ["pkg/a.py:entry"],
                                "graph_edge_list": [],
                            },
                            {
                                "component_id": "component_0001",
                                "entry_kind": "root",
                                "entry_node_ids": ["pkg/b.py:worker"],
                                "node_ids": ["pkg/b.py:worker"],
                                "graph_edge_list": [],
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            script = [
                {
                    "content": (
                        "Entry looks directly relevant.\n"
                        "<node_decision>"
                        "{\"decision\": \"finish_search\", "
                        "\"is_relevant\": true, "
                        "\"confidence\": 0.88, "
                        "\"reason\": \"This entry is enough for the limited test search.\", "
                        "\"functionality_hint\": \"Primary entry flow\", "
                        "\"suspected_locations\": ["
                        "{\"node_id\": \"pkg/a.py:entry\", \"reason\": \"Likely bug entry.\", \"confidence\": 0.88}"
                        "]}"
                        "</node_decision>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 32},
                },
                {
                    "content": (
                        "<component_summary>\n"
                        "{\"functionality_summary\": \"Primary entry flow\"}\n"
                        "</component_summary>"
                    ),
                    "tool_calls": None,
                    "usage": {"total_tokens": 48},
                },
            ]

            args = _build_args(
                tmp_path,
                search_mode="bfs_component_graph",
                mock=False,
                component_graph_dir=str(component_graph_dir),
                tool_call_parser_type="mock-parser",
                bfs_component_search_strategy="original",
                bfs_component_parallelism=1,
                bfs_max_components_to_check=1,
            )
            runner = self.inference.LocalizeRunner(
                args,
                llm=self.inference.MockLLM(script),
            )
            runner.tool_registry = {
                "get_code_of_file_function": lambda file_name, func_name, instance_id: (
                    f"def {func_name}():\n    return '{instance_id}:{file_name}:{func_name}'"
                ),
            }

            item = {
                "repo": "demo/repo",
                "instance_id": instance_id,
                "base_commit": "deadbeef",
                "problem_statement": "Only inspect the first component during this test run.",
                "data_source": "mock",
                "reward_model": {"ground_truth": ["pkg/a.py:entry"]},
                "extra_info": {
                    "instance_id": instance_id,
                    "repo": "demo/repo",
                    "base_commit": "deadbeef",
                },
            }

            result = runner._run_item(copy.deepcopy(item))

            self.assertEqual(result["search_mode"], "bfs_component_graph")
            self.assertEqual(result["search_execution_mode"], "component_search_merge")
            self.assertEqual(result["bfs_component_search_strategy"], "original")
            self.assertEqual(result["bfs_component_parallelism"], 1)
            self.assertEqual(result["bfs_max_components_to_check"], 1)
            self.assertEqual(runner.tool_execute_count, 1)
            self.assertNotIn("Search Configuration:", result["messages"][1]["content"])
            self.assertNotIn("Instance ID:", result["messages"][1]["content"])
            self.assertNotIn("Repository:", result["messages"][1]["content"])
            self.assertNotIn("Base Commit:", result["messages"][1]["content"])
            self.assertNotIn("Candidate Component Graphs", result["messages"][1]["content"])
            self.assertNotIn("component_brief_path", result)
            self.assertEqual(len(result["component_search_results"]), 1)
            self.assertEqual(result["component_search_results"][0]["component_id"], "component_0000")
            self.assertEqual(len(result["messages"]), 3)
            self.assertEqual(result["messages"][-1]["content"], "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>")
            self.assertEqual(
                result["component_search_results"][0]["trace_locs"],
                "<trace_locs>\npkg/a.py\nfunction: entry\n</trace_locs>",
            )

            merged_summary = json.loads(Path(result["component_summary_path"]).read_text(encoding="utf-8"))
            self.assertEqual(sorted(merged_summary["components"].keys()), ["component_0000"])


if __name__ == "__main__":
    unittest.main()
