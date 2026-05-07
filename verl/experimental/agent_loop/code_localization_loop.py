"""
Code Localization multi-agent loop for RL training with verl.

Supports alternating training:
- training_role=navigator: Navigator generates via local rollout, Inspectors via remote vLLM
- training_role=inspector: Navigator generates via remote vLLM, Inspector generates via local rollout

Uses <tool_call> XML format (hermes parser) for tool calling.

Shared utilities (prompt construction, tool-call parsing, GlobalState, etc.)
are imported from ``evaluation.multi_agent_shared`` to stay consistent with
the inference pipeline in ``evaluation/multi_agent_inference.py``.
"""

import asyncio
import json
import logging
import os
import re
from string import Template
from typing import Any, Optional
from uuid import uuid4

from evaluation.multi_agent_shared import (
    MAIN_AGENT_FIND_FILES_BY_CONTENT,
    MAIN_AGENT_FIND_FILES_BY_NAME,
    PROJECT_ROOT,
    PROMPTS_DIR,
    RESULT_RE,
    GlobalState,
    _load_text,
    augment_system_prompt_with_tool_instructions,
    build_inspector_messages,
    build_navigator_messages,
    build_sub_agent_tool_registry,
    extract_explored_from_tool_calls,
    parse_tool_calls_from_text,
    truncate_at_second_think,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TOOL_SCHEMAS_DIR = os.path.join(PROJECT_ROOT, "tools", "tool_schemas")



class RemoteVLLMClient:
    """Client for calling a remote vLLM server via OpenAI-compatible API."""

    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    base_url=self.api_base,
                    api_key=self.api_key,
                )
            except ImportError:
                import httpx
                self._client = None
                raise ImportError("openai package required for remote vLLM calls")
        return self._client

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.95,
    ) -> str:
        """Call the remote vLLM chat completion and return the response text."""
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].message.content or ""


class AgentToolStats:
    """Track tool call success/failure counts."""

    def __init__(self):
        self.total = 0
        self.successful = 0

    def record(self, success: bool):
        self.total += 1
        if success:
            self.successful += 1

    def to_dict(self) -> dict[str, int]:
        return {"total": self.total, "successful": self.successful}


# GlobalState imported from evaluation.multi_agent_shared


@register("code_localization")
class CodeLocalizationAgentLoop(AgentLoopBase):
    """Multi-agent code localization loop for RL training.

    Supports two training modes via `training_role`:
    - "navigator": Train Navigator; Inspector runs via remote vLLM
    - "inspector": Train Inspector; Navigator runs via remote vLLM
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        multi_turn_config = self.rollout_config.multi_turn

        # Training role
        self.training_role = getattr(multi_turn_config, "training_role", "navigator")

        # Agent turn limits
        self.max_navigator_turns = getattr(multi_turn_config, "max_navigator_turns", 10)
        self.max_inspector_turns = getattr(multi_turn_config, "max_inspector_turns", 10)
        self.sub_agent_parallelism = getattr(multi_turn_config, "sub_agent_parallelism", 3)
        self.max_tool_response_length = multi_turn_config.max_tool_response_length

        # Remote vLLM config for the fixed (non-training) role
        self.remote_api_base = getattr(multi_turn_config, "remote_api_base", "")
        self.remote_api_key = getattr(multi_turn_config, "remote_api_key", "EMPTY")
        self.remote_model_name = getattr(multi_turn_config, "remote_model_name", "")
        self.remote_client = None
        if self.remote_api_base and self.remote_model_name:
            self.remote_client = RemoteVLLMClient(
                api_base=self.remote_api_base,
                api_key=self.remote_api_key,
                model_name=self.remote_model_name,
            )

        # Tool setup
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        length_penalty_coef = getattr(multi_turn_config, "length_penalty_coef", None)
        if length_penalty_coef is None:
            length_penalty_coef = getattr(multi_turn_config, "length_penalty_coefficient", 1.0)
        self.length_penalty_coef = float(length_penalty_coef)
        self.top_k_predictions = int(getattr(multi_turn_config, "top_k_predictions", 5))
        self.reward_metric = self._normalize_reward_metric(
            getattr(multi_turn_config, "reward_metric", "dice")
        )
        self.reward_acc_at_k_from_gt = self._parse_bool(
            getattr(multi_turn_config, "reward_acc_at_k_from_gt", False)
        )
        self.reward_acc_at_k = self._parse_reward_acc_at_k(
            getattr(
                multi_turn_config,
                "reward_acc_at_k",
                getattr(multi_turn_config, "reward_acc_k", None),
            )
        )

        # Initialize tools
        nav_tool_config = getattr(multi_turn_config, "navigator_tool_config_path", None)
        insp_tool_config = getattr(multi_turn_config, "inspector_tool_config_path", None)

        if nav_tool_config:
            nav_tools = initialize_tools_from_config(nav_tool_config)
            self.navigator_tools = {t.name: t for t in nav_tools}
            self.navigator_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in nav_tools
            ]
        else:
            self.navigator_tools = {}
            self.navigator_tool_schemas = []

        if insp_tool_config:
            insp_tools = initialize_tools_from_config(insp_tool_config)
            self.inspector_tools = {t.name: t for t in insp_tools}
            self.inspector_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in insp_tools
            ]
        else:
            self.inspector_tools = {}
            self.inspector_tool_schemas = []

        # Trajectory saving — append a run_id subdir so each training run is isolated.
        # run_id comes from config (preferred, works with ray submit) or env var fallback.
        # Cast to str: Hydra infers numeric-looking values (e.g. 20250501142035) as int.
        _base_dir = getattr(multi_turn_config, "trajectory_save_dir", None)
        if _base_dir:
            _run_id_raw = getattr(multi_turn_config, "run_id", None)
            _run_id = str(_run_id_raw) if _run_id_raw is not None else os.environ.get("VERL_RUN_TIMESTAMP", "")
            self.trajectory_save_dir = os.path.join(_base_dir, _run_id) if _run_id else _base_dir
        else:
            self.trajectory_save_dir = None

        # Load prompt templates
        self._load_prompts()

    def _load_prompts(self):
        """Load prompt templates from files."""
        self.nav_system_prompt = _load_text(os.path.join(PROMPTS_DIR, "main_agent_system_prompt.txt"))
        self.nav_user_prompt_template = _load_text(os.path.join(PROMPTS_DIR, "main_agent_user_prompt.txt"))
        self.insp_system_prompt = _load_text(os.path.join(PROMPTS_DIR, "sub_agent_system_prompt.txt"))
        self.insp_user_prompt_template = _load_text(os.path.join(PROMPTS_DIR, "sub_agent_user_prompt.txt"))
        self.insp_finalize_prompt_template = _load_text(os.path.join(PROMPTS_DIR, "sub_agent_finalize_prompt.txt"))
        self.finalize_prompt_template = _load_text(os.path.join(PROMPTS_DIR, "finalize_prompt.txt"))

    async def _save_trajectory(
        self,
        trajectory: dict[str, Any],
        instance_id: str,
        global_step: int,
        is_eval: bool,
    ) -> None:
        """Persist one trajectory to disk under trajectory_save_dir.

        Directory layout:
            <trajectory_save_dir>/{train|val}/step_<NNNNNN>/<instance_id>_<hex>.json
        """
        if not self.trajectory_save_dir:
            return
        split = "val" if is_eval else "train"
        step_dir = os.path.join(self.trajectory_save_dir, split, f"step_{global_step:06d}")

        def _write():
            os.makedirs(step_dir, exist_ok=True)
            filename = f"{instance_id}_{uuid4().hex[:8]}.json"
            filepath = os.path.join(step_dir, filename)
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(trajectory, fh, ensure_ascii=False, indent=2)

        try:
            await asyncio.to_thread(_write)
        except Exception as exc:
            logger.warning("Failed to save trajectory for %s: %s", instance_id, exc)

    def _get_remote_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        """Resolve remote sampling params from the local rollout config."""
        if not sampling_params:
            raise ValueError("Remote fixed-agent execution requires local sampling_params.")
        missing = [name for name in ("temperature", "top_p") if name not in sampling_params]
        if missing:
            raise ValueError(
                "Remote fixed-agent execution requires local sampling_params with keys: "
                + ", ".join(missing)
            )
        max_tokens = sampling_params.get("max_tokens", sampling_params.get("max_new_tokens", self.response_length))
        if max_tokens is None:
            raise ValueError(
                "Remote fixed-agent execution requires local max_tokens/max_new_tokens "
                "or a configured response_length."
            )
        return {
            "temperature": sampling_params["temperature"],
            "top_p": sampling_params["top_p"],
            "max_tokens": max_tokens,
        }

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        if self.training_role == "navigator":
            return await self._run_navigator_training(sampling_params, **kwargs)
        else:
            return await self._run_inspector_training(sampling_params, **kwargs)

    # -----------------------------------------------------------------------
    # Navigator Training Mode
    # -----------------------------------------------------------------------
    async def _run_navigator_training(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        """Train Navigator: Navigator uses local rollout, Inspectors use remote vLLM."""
        extra_info = kwargs.get("extra_info", {}) or {}
        instance_id = extra_info.get("instance_id", "")
        problem_statement = extra_info.get("problem_statement", "")
        repo_structure = extra_info.get("structure", "")
        ground_truth = extra_info.get("ground_truth", "")
        global_step: int = kwargs.get("global_step", extra_info.get("global_step", 0))
        is_eval: bool = bool(kwargs.get("is_eval", extra_info.get("is_eval", False)))

        # Build Navigator messages using shared helper
        messages = build_navigator_messages(
            self.nav_system_prompt,
            self.nav_user_prompt_template,
            self.navigator_tool_schemas,
            problem_statement,
            repo_structure,
        )

        # State tracking
        metrics = {}
        request_id = uuid4().hex
        state = GlobalState()
        tool_stats = AgentToolStats()

        # Token tracking
        prompt_ids = await self.apply_chat_template(messages)
        response_ids_all: list[int] = []
        response_mask_all: list[int] = []
        response_logprobs_all: list[float] = []

        # Trajectory message tracking (text-level, for saving)
        trajectory_messages: list[dict[str, str]] = list(messages)
        termination_reason = "max_turns_reached"

        for turn in range(self.max_navigator_turns):
            # Generate via local rollout engine
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids_all,
                    sampling_params=sampling_params,
                )

            # Truncate at second <think> block if present
            gen_ids = output.token_ids
            gen_logprobs = output.log_probs
            full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            truncated_text = truncate_at_second_think(full_text)
            if len(truncated_text) < len(full_text):
                trunc_len = len(self.tokenizer.encode(truncated_text, add_special_tokens=False))
                gen_ids = gen_ids[:trunc_len]
                if gen_logprobs:
                    gen_logprobs = gen_logprobs[:trunc_len]

            response_ids_all += gen_ids
            response_mask_all += [1] * len(gen_ids)
            if gen_logprobs:
                response_logprobs_all += gen_logprobs
            trajectory_messages.append({"role": "assistant", "content": truncated_text})

            if len(response_mask_all) >= self.response_length:
                termination_reason = "response_length_exhausted"
                break

            # Parse tool calls from response
            tool_calls = parse_tool_calls_from_text(truncated_text)

            if not tool_calls:
                termination_reason = "no_tool_calls"
                break

            # Execute tool calls
            tool_feedback_messages = await self._execute_navigator_tools(
                tool_calls, instance_id, state, tool_stats,
                problem_statement, extra_info, sampling_params,
            )

            # Add tool response messages to prompt (each as a separate user message)
            should_break = False
            for feedback_text in tool_feedback_messages:
                tool_message = [{"role": "user", "content": feedback_text}]
                tool_ids = await self.apply_chat_template(tool_message, remove_system_prompt=True)

                if len(response_mask_all) + len(tool_ids) >= self.response_length:
                    should_break = True
                    break

                response_ids_all += tool_ids
                response_mask_all += [0] * len(tool_ids)
                if response_logprobs_all:
                    response_logprobs_all += [0.0] * len(tool_ids)
                trajectory_messages.append({"role": "user", "content": feedback_text})

            if should_break:
                termination_reason = "response_length_exhausted"
                break

            # Check for exit
            if any(tc.get("name") == "exit" for tc in tool_calls):
                termination_reason = "agent_called_exit"
                break

        # Finalize: ask Navigator for ranked top-5 (matches inference pipeline)
        if state.confirmed_suspicious:
            suspicious_summary = "\n".join(
                f"  - {s['location']}: {s['reason']}"
                for s in state.confirmed_suspicious
                if "location" in s
            )
            finalize_content = Template(self.finalize_prompt_template).safe_substitute(
                confirmed_suspicious=suspicious_summary,
            )
            finalize_message = [{"role": "user", "content": finalize_content}]
            finalize_ids = await self.apply_chat_template(finalize_message, remove_system_prompt=True)

            if len(response_mask_all) + len(finalize_ids) < self.response_length:
                response_ids_all += finalize_ids
                response_mask_all += [0] * len(finalize_ids)
                if response_logprobs_all:
                    response_logprobs_all += [0.0] * len(finalize_ids)
                trajectory_messages.append({"role": "user", "content": finalize_content})

                # Generate finalize response via local rollout engine
                with simple_timer("generate_finalize", metrics):
                    finalize_output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids + response_ids_all,
                        sampling_params=sampling_params,
                    )

                finalize_gen_ids = finalize_output.token_ids
                finalize_logprobs = finalize_output.log_probs
                finalize_text = self.tokenizer.decode(finalize_gen_ids, skip_special_tokens=True)
                response_ids_all += finalize_gen_ids
                response_mask_all += [1] * len(finalize_gen_ids)
                if finalize_logprobs:
                    response_logprobs_all += finalize_logprobs
                trajectory_messages.append({"role": "assistant", "content": finalize_text})

        # Compute Navigator reward
        from verl.utils.reward_score.code_localization import compute_navigator_reward, parse_ground_truth
        gt_locs = parse_ground_truth(ground_truth)

        # Extract predicted locations from finalize response or confirmed_suspicious
        predicted = self._extract_predicted_from_response(response_ids_all, state)
        predicted = predicted[:self._prediction_limit_for_reward(gt_locs)]
        if self.reward_metric == "acc_at_k":
            reward_result = self._compute_navigator_acc_at_k_reward(
                predicted_locations=predicted,
                ground_truth_locations=gt_locs,
                total_tool_calls=tool_stats.total,
                successful_tool_calls=tool_stats.successful,
            )
        else:
            reward_result = compute_navigator_reward(
                predicted_locations=predicted,
                ground_truth_locations=gt_locs,
                total_tool_calls=tool_stats.total,
                successful_tool_calls=tool_stats.successful,
            )
            reward_result["reward_metric"] = "dice"
            self._add_common_reward_metrics(
                reward_result,
                predicted_locations=predicted,
                ground_truth_locations=gt_locs,
            )
        reward_score, length_penalty = self._apply_response_length_penalty(
            reward_result["score"],
            len(response_ids_all),
        )
        reward_result["score_before_length_penalty"] = reward_result["score"]
        reward_result["score"] = reward_score
        reward_result["length_penalty"] = length_penalty

        all_ids = (prompt_ids + response_ids_all)
        final_prompt_ids = all_ids[:len(prompt_ids)]
        # AgentLoop postprocess pads responses into fixed-width tensors. Keep
        # the tensor contract here, but apply reward penalty using raw length.
        final_response_ids = response_ids_all[:self.response_length]
        final_response_mask = response_mask_all[:self.response_length]

        await self._save_trajectory(
            trajectory={
                "role": "navigator",
                "instance_id": instance_id,
                "global_step": global_step,
                "is_eval": is_eval,
                "num_turns": turn + 1,
                "termination_reason": termination_reason,
                "messages": trajectory_messages,
                "confirmed_suspicious": state.confirmed_suspicious,
                "tool_stats": tool_stats.to_dict(),
                "reward_breakdown": reward_result,
                "reward_extra_info": self._build_reward_extra_info(reward_result),
            },
            instance_id=instance_id,
            global_step=global_step,
            is_eval=is_eval,
        )

        return AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=response_logprobs_all[:self.response_length] if response_logprobs_all else None,
            reward_score=reward_score,
            num_turns=turn + 1,
            metrics=metrics,
            extra_fields={
                "confirmed_suspicious": state.confirmed_suspicious,
                "tool_stats": tool_stats.to_dict(),
                "reward_breakdown": reward_result,
                "reward_extra_info": self._build_reward_extra_info(reward_result),
            },
        )

    async def _execute_navigator_tools(
        self,
        tool_calls: list[dict[str, Any]],
        instance_id: str,
        state: GlobalState,
        tool_stats: AgentToolStats,
        problem_statement: str,
        extra_info: dict,
        sampling_params: dict[str, Any],
    ) -> list[str]:
        """Execute Navigator tool calls. Returns a list of formatted feedback messages.

        Search tools (find_files_by_content, find_files_by_name) are combined into one message.
        Dispatch produces a separate message with sub-agent results.
        Format matches evaluation/multi_agent_inference.py.
        """
        search_results: list[dict[str, Any]] = []  # {"tool_name", "arguments", "result"}
        dispatch_feedback: Optional[str] = None
        has_exit = False

        # --- Phase 1: Categorize tool calls (arguments already parsed by parse_tool_calls_from_text) ---
        search_calls: list[tuple[str, dict]] = []
        dispatch_calls: list[tuple[str, dict]] = []

        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if not isinstance(args, dict):
                tool_stats.record(False)
                search_results.append({
                    "tool_name": name,
                    "arguments": args,
                    "result": f"Error: Invalid arguments for {name}",
                })
                continue

            if name in (MAIN_AGENT_FIND_FILES_BY_CONTENT, MAIN_AGENT_FIND_FILES_BY_NAME):
                search_calls.append((name, args))
            elif name == "dispatch":
                dispatch_calls.append((name, args))
            elif name == "exit":
                tool_stats.record(True)
                has_exit = True
            else:
                tool_stats.record(False)
                search_results.append({
                    "tool_name": name,
                    "arguments": args,
                    "result": f"Error: Unknown tool '{name}'",
                })

        # --- Phase 2: Execute search tools in parallel ---
        if search_calls:
            from tools.multi_agent_tools import find_files_by_content_repo, find_files_by_name_repo

            def _exec_search(tc_name: str, args: dict) -> str:
                if tc_name == MAIN_AGENT_FIND_FILES_BY_CONTENT:
                    keywords = args.get("keywords")
                    if not keywords:
                        keyword = args.get("keyword", "")
                        keywords = [keyword] if keyword else []
                    return find_files_by_content_repo(keywords, instance_id, args.get("file_pattern", "*.py"))
                else:
                    patterns = args.get("patterns")
                    if not patterns:
                        pattern = args.get("pattern", "")
                        patterns = [pattern] if pattern else []
                    return find_files_by_name_repo(patterns, instance_id)

            # Parallel execution only when there are multiple calls
            if len(search_calls) > 1:
                tasks = [
                    asyncio.to_thread(_exec_search, name, args)
                    for name, args in search_calls
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                name0, args0 = search_calls[0]
                try:
                    raw_results = [_exec_search(name0, args0)]
                except Exception as e:
                    raw_results = [e]

            for (name, args), result in zip(search_calls, raw_results):
                if isinstance(result, Exception):
                    tool_stats.record(False)
                    search_results.append({
                        "tool_name": name,
                        "arguments": args,
                        "result": f"Error: {result}",
                    })
                    continue
                tool_stats.record(True)
                try:
                    parsed = json.loads(result)
                    if name == MAIN_AGENT_FIND_FILES_BY_CONTENT:
                        files = [r["file"] for r in parsed.get("results", [])]
                    else:
                        files = parsed.get("matched_files", [])
                    state.add_pending(files)
                except (json.JSONDecodeError, TypeError):
                    pass
                search_results.append({
                    "tool_name": name,
                    "arguments": args,
                    "result": result,
                })

        # --- Phase 3: Handle dispatch ---
        for dispatch_name, args in dispatch_calls:
            entries = args.get("entries", [])
            entries = [e for e in entries if e not in state.explored_entries]
            if not entries:
                tool_stats.record(True)
                search_results.append({
                    "tool_name": dispatch_name,
                    "arguments": args,
                    "result": json.dumps({"message": "All entries already explored."}),
                })
                continue

            sub_results = await self._run_inspectors_remote(
                entries, instance_id, problem_statement, extra_info, sampling_params
            )
            tool_stats.record(True)

            for sr in sub_results:
                state.add_suspicious(sr.get("suspicious", []))
                state.add_explored_from_functions(sr.get("explored", []))
                entry_file = sr.get("entry_file", "")
                if entry_file:
                    state.add_explored_entry(entry_file)

            # Build dispatch feedback matching inference format
            parts = []
            for sr in sub_results:
                entry = sr.get("entry_file", "unknown")
                suspicious = sr.get("suspicious", [])
                parts.append(f"Sub Agent explored '{entry}':")
                if suspicious:
                    for s in suspicious:
                        parts.append(f"  - {s['location']}: {s['reason']}")
                else:
                    parts.append("  (no suspicious locations found)")

            dispatch_feedback = "Sub Agent results:\n" + "\n".join(parts)
            dispatch_feedback += f"\n\nCurrent state:\n{state.state_summary()}"
            dispatch_feedback += (
                "\n\nBased on these results, decide your next action: "
                "search more with find_files_by_content/find_files_by_name, dispatch more files "
                "to Sub Agents, or call exit if done."
            )

        # --- Phase 4: Build feedback messages list ---
        feedback_messages: list[str] = []

        # Search tool results -> one combined message
        if search_results:
            feedback_parts = []
            for tr in search_results:
                args_str = json.dumps(tr["arguments"], ensure_ascii=False) if isinstance(tr["arguments"], dict) else str(tr["arguments"])
                feedback_parts.append(
                    f"Result of {tr['tool_name']}({args_str}):\n"
                    f"{tr['result']}"
                )
            feedback = "\n\n".join(feedback_parts)
            feedback += f"\n\nCurrent state:\n{state.state_summary()}"
            feedback += (
                "\n\nBased on these results, decide your next action: "
                "search more with find_files_by_content/find_files_by_name, dispatch files "
                "to Sub Agents, or call exit if done."
            )
            feedback_messages.append(feedback)

        # Dispatch results -> separate message
        if dispatch_feedback:
            feedback_messages.append(dispatch_feedback)

        # If only exit was called with no other tools
        # if not feedback_messages and has_exit:
        #     feedback_messages.append("Search completed.")

        return feedback_messages

    async def _run_inspectors_remote(
        self,
        entries: list[str],
        instance_id: str,
        problem_statement: str,
        extra_info: dict,
        sampling_params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run Inspector sub-agents via remote vLLM."""
        if not self.remote_client:
            return [{"entry_file": e, "suspicious": [], "explored": [],
                     "error": "No remote vLLM configured"} for e in entries]

        # Load sub-agent tool schemas
        tool_schemas_path = os.path.join(TOOL_SCHEMAS_DIR, "sub_agent_tools.json")
        try:
            with open(tool_schemas_path, "r", encoding="utf-8") as f:
                sub_tool_schemas = json.load(f)
        except FileNotFoundError:
            sub_tool_schemas = []

        # Run in parallel batches
        results = []
        for i in range(0, len(entries), self.sub_agent_parallelism):
            batch = entries[i:i + self.sub_agent_parallelism]
            tasks = [
                self._run_single_inspector_remote(
                    entry,
                    instance_id,
                    problem_statement,
                    extra_info,
                    sub_tool_schemas,
                    sampling_params,
                )
                for entry in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "entry_file": batch[j], "suspicious": [], "explored": [],
                        "error": str(result),
                    })
                else:
                    results.append(result)
        return results

    async def _run_single_inspector_remote(
        self,
        entry_file: str,
        instance_id: str,
        problem_statement: str,
        extra_info: dict,
        sub_tool_schemas: list[dict],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a single Inspector via remote vLLM with multi-turn tool calling."""
        remote_sampling_params = self._get_remote_sampling_params(sampling_params)

        messages = build_inspector_messages(
            self.insp_system_prompt,
            self.insp_user_prompt_template,
            sub_tool_schemas,
            problem_statement,
            entry_file,
        )

        all_tool_calls: list[dict[str, Any]] = []
        termination_reason = "max_turns_reached"

        for turn in range(self.max_inspector_turns):
            response_text = await self.remote_client.chat_completion(
                messages=messages,
                temperature=remote_sampling_params["temperature"],
                max_tokens=remote_sampling_params["max_tokens"],
                top_p=remote_sampling_params["top_p"],
            )
            response_text = truncate_at_second_think(response_text)
            messages.append({"role": "assistant", "content": response_text})

            # Parse tool calls
            parsed_calls = parse_tool_calls_from_text(response_text)
            if not parsed_calls:
                termination_reason = "no_tool_calls"
                break

            # Check for exit
            has_exit = any(tc.get("name") == "exit" for tc in parsed_calls)

            tool_response_messages = await self._execute_inspector_tool_dicts(
                parsed_calls,
                instance_id,
                all_tool_calls=all_tool_calls,
            )
            for feedback in tool_response_messages:
                messages.append({"role": "user", "content": feedback})

            if has_exit:
                termination_reason = "agent_called_exit"
                break

        suspicious = await self._finalize_inspector_remote(
            entry_file=entry_file,
            messages=messages,
            remote_sampling_params=remote_sampling_params,
            termination_reason=termination_reason,
        )
        explored = extract_explored_from_tool_calls(all_tool_calls)
        return {
            "entry_file": entry_file,
            "suspicious": suspicious,
            "explored": explored,
        }

    @staticmethod
    def _parse_inspector_result(text: str) -> Optional[dict[str, Any]]:
        """Parse the <result> block from an Inspector response."""
        result_match = RESULT_RE.search(text)
        if not result_match:
            return None
        try:
            result_json = json.loads(result_match.group(1).strip())
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(result_json, dict):
            return None
        return {
            "suspicious": result_json.get("suspicious", []),
        }

    def _collect_inspector_suspicious_candidates(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Collect suspicious candidates from the last assistant message only."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            parsed = self._parse_inspector_result(str(msg.get("content", "") or ""))
            if not parsed:
                return []
            suspicious_list = parsed.get("suspicious", [])
            if not isinstance(suspicious_list, list):
                return []

            candidates: list[dict[str, str]] = []
            for item in suspicious_list:
                if not isinstance(item, dict):
                    continue
                location = str(item.get("location", "") or "").strip()
                if not location:
                    continue
                candidates.append(
                    {
                        "location": location,
                        "reason": str(item.get("reason", "") or "").strip(),
                    }
                )
            return candidates

        return []

    async def _finalize_inspector_remote(
        self,
        *,
        entry_file: str,
        messages: list[dict[str, Any]],
        remote_sampling_params: dict[str, Any],
        termination_reason: str,
    ) -> list[dict[str, str]]:
        """Finalize a remote Inspector run in the existing conversation.

        If the agent already called exit and produced a valid <result> block,
        skip the extra remote LLM call to save time.
        """
        del entry_file  # unused

        # Try to reuse an existing <result> when the agent exited cleanly
        existing = self._get_reusable_inspector_exit_result(messages, termination_reason)
        if existing:
            return existing

        messages.append({"role": "user", "content": self.insp_finalize_prompt_template})
        response_text = await self.remote_client.chat_completion(
            messages=messages,
            temperature=remote_sampling_params["temperature"],
            max_tokens=remote_sampling_params["max_tokens"],
            top_p=remote_sampling_params["top_p"],
        )
        response_text = truncate_at_second_think(response_text)
        messages.append({"role": "assistant", "content": response_text})

        parsed = self._parse_inspector_result(response_text)
        if parsed is not None and isinstance(parsed.get("suspicious", []), list):
            return parsed["suspicious"]
        return self._collect_inspector_suspicious_candidates(messages)

    # -----------------------------------------------------------------------
    # Inspector Training Mode
    # -----------------------------------------------------------------------
    async def _run_inspector_training(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        """Train Inspector using pre-extracted training data.

        Requires extra_info['entry_file'] to be set in the training data
        (built by build_inspector_training_data.py). The Inspector generates
        via local rollout engine for the specified entry file.
        """
        extra_info = kwargs.get("extra_info", {}) or {}
        instance_id = extra_info.get("instance_id", "")
        problem_statement = extra_info.get("problem_statement", "")
        ground_truth = extra_info.get("ground_truth", "")
        global_step: int = kwargs.get("global_step", extra_info.get("global_step", 0))
        is_eval: bool = bool(kwargs.get("is_eval", extra_info.get("is_eval", False)))

        entry_file = extra_info.get("entry_file", "")
        if not entry_file:
            logger.warning(
                f"[{instance_id}] No entry_file in extra_info. "
                "Inspector training requires pre-extracted data with entry_file. "
                "Use build_inspector_training_data.py to prepare the data."
            )
            return await self._empty_output(kwargs)

        # Run Inspector via local rollout engine
        return await self._run_inspector_local(
            sampling_params, entry_file, instance_id,
            problem_statement, ground_truth, extra_info,
            global_step=global_step, is_eval=is_eval,
        )

    async def _run_inspector_local(
        self,
        sampling_params: dict[str, Any],
        entry_file: str,
        instance_id: str,
        problem_statement: str,
        ground_truth: str,
        extra_info: dict,
        global_step: int = 0,
        is_eval: bool = False,
    ) -> AgentLoopOutput:
        """Run one Inspector via local rollout engine (tracked for training)."""
        # Build Inspector messages using shared helper
        messages = build_inspector_messages(
            self.insp_system_prompt,
            self.insp_user_prompt_template,
            self.inspector_tool_schemas,
            problem_statement,
            entry_file,
        )

        metrics = {}
        request_id = uuid4().hex
        tool_stats = AgentToolStats()
        all_tool_calls: list[dict[str, Any]] = []
        termination_reason = "max_turns_reached"

        # Tokenize initial prompt
        prompt_ids = await self.apply_chat_template(messages)
        response_ids_all: list[int] = []
        response_mask_all: list[int] = []
        response_logprobs_all: list[float] = []

        for turn in range(self.max_inspector_turns):
            # Generate via local rollout engine
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids_all,
                    sampling_params=sampling_params,
                )

            # Truncate at second <think> block if present
            gen_ids = output.token_ids
            gen_logprobs = output.log_probs
            full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            truncated_text = truncate_at_second_think(full_text)
            if len(truncated_text) < len(full_text):
                trunc_len = len(self.tokenizer.encode(truncated_text, add_special_tokens=False))
                gen_ids = gen_ids[:trunc_len]
                if gen_logprobs:
                    gen_logprobs = gen_logprobs[:trunc_len]

            response_ids_all += gen_ids
            response_mask_all += [1] * len(gen_ids)
            if gen_logprobs:
                response_logprobs_all += gen_logprobs
            messages.append({"role": "assistant", "content": truncated_text})

            if len(response_mask_all) >= self.response_length:
                termination_reason = "response_length_exhausted"
                break

            # Parse tool calls
            parsed_calls = parse_tool_calls_from_text(truncated_text)

            if not parsed_calls:
                termination_reason = "no_tool_calls"
                break

            # Execute Inspector tools
            tool_response_messages = await self._execute_inspector_tool_dicts(
                parsed_calls,
                instance_id,
                tool_stats=tool_stats,
                all_tool_calls=all_tool_calls,
            )

            # Check for exit
            has_exit = any(tc.get("name") == "exit" for tc in parsed_calls)

            # Add tool response messages (each as a separate user message)
            should_break = False
            for feedback_text in tool_response_messages:
                tool_message = [{"role": "user", "content": feedback_text}]
                tool_ids = await self.apply_chat_template(tool_message, remove_system_prompt=True)

                if len(response_mask_all) + len(tool_ids) >= self.response_length:
                    should_break = True
                    break

                response_ids_all += tool_ids
                response_mask_all += [0] * len(tool_ids)
                if response_logprobs_all:
                    response_logprobs_all += [0.0] * len(tool_ids)
                messages.append({"role": "user", "content": feedback_text})

            if should_break:
                termination_reason = "response_length_exhausted"
                break

            if has_exit:
                termination_reason = "agent_called_exit"
                break

        await self._finalize_inspector_local(
            messages=messages,
            sampling_params=sampling_params,
            request_id=request_id,
            prompt_ids=prompt_ids,
            response_ids_all=response_ids_all,
            response_mask_all=response_mask_all,
            response_logprobs_all=response_logprobs_all,
            metrics=metrics,
            termination_reason=termination_reason,
        )

        # Compute Inspector reward
        from verl.utils.reward_score.code_localization import (
            compute_inspector_reward, parse_ground_truth, parse_predicted_locations,
        )

        gt_locs = parse_ground_truth(ground_truth)

        # Extract predictions only from the final assistant message. Parsing the
        # full trajectory can pick up stale or tool-feedback content.
        response_text = self._get_last_assistant_message_text(messages)
        predicted = parse_predicted_locations(response_text)

        # Get explored files from tool calls
        explored_funcs = extract_explored_from_tool_calls(all_tool_calls)
        explored_files = set()
        for func_loc in explored_funcs:
            file_part = func_loc.split(":")[0] if ":" in func_loc else func_loc
            if file_part:
                explored_files.add(file_part)
        # Always include the entry file
        explored_files.add(entry_file)

        predicted = predicted[:self._prediction_limit_for_reward(gt_locs, explored_files=explored_files)]
        if self.reward_metric == "acc_at_k":
            reward_result = self._compute_inspector_acc_at_k_reward(
                predicted_locations=predicted,
                ground_truth_locations=gt_locs,
                explored_files=explored_files,
                total_tool_calls=tool_stats.total,
                successful_tool_calls=tool_stats.successful,
            )
        else:
            reward_result = compute_inspector_reward(
                predicted_locations=predicted,
                ground_truth_locations=gt_locs,
                explored_files=explored_files,
                total_tool_calls=tool_stats.total,
                successful_tool_calls=tool_stats.successful,
            )
            reward_result["reward_metric"] = "dice"
            self._add_common_reward_metrics(
                reward_result,
                predicted_locations=predicted,
                ground_truth_locations=self._filter_ground_truth_by_files(gt_locs, explored_files),
            )

        # If F_relevant is empty (discard=True), set reward to None
        # The reward manager will handle filtering these out
        reward_score = reward_result.get("score")
        reward_score, length_penalty = self._apply_response_length_penalty(
            reward_score,
            len(response_ids_all),
        )
        reward_result["score_before_length_penalty"] = reward_result.get("score")
        reward_result["score"] = reward_score
        reward_result["length_penalty"] = length_penalty

        # AgentLoop postprocess pads responses into fixed-width tensors. Keep
        # the tensor contract here, but apply reward penalty using raw length.
        final_response_ids = response_ids_all[:self.response_length]
        final_response_mask = response_mask_all[:self.response_length]

        await self._save_trajectory(
            trajectory={
                "role": "inspector",
                "instance_id": instance_id,
                "entry_file": entry_file,
                "global_step": global_step,
                "is_eval": is_eval,
                "num_turns": turn + 1,
                "termination_reason": termination_reason,
                "messages": messages,
                "explored_files": list(explored_files),
                "tool_stats": tool_stats.to_dict(),
                "reward_breakdown": reward_result,
                "reward_extra_info": self._build_reward_extra_info(reward_result),
                "discard": reward_result.get("discard", False),
            },
            instance_id=instance_id,
            global_step=global_step,
            is_eval=is_eval,
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=response_logprobs_all[:self.response_length] if response_logprobs_all else None,
            reward_score=reward_score,
            num_turns=turn + 1,
            metrics=metrics,
            extra_fields={
                "entry_file": entry_file,
                "explored_files": list(explored_files),
                "tool_stats": tool_stats.to_dict(),
                "reward_breakdown": reward_result,
                "reward_extra_info": self._build_reward_extra_info(reward_result),
                "discard": reward_result.get("discard", False),
            },
        )

    def _get_reusable_inspector_exit_result(
        self,
        messages: list[dict[str, Any]],
        termination_reason: str,
    ) -> list[dict[str, str]]:
        """Return the final exit result when Inspector already produced one."""
        if termination_reason != "agent_called_exit":
            return []
        return self._collect_inspector_suspicious_candidates(messages)

    @staticmethod
    def _get_last_assistant_message_text(messages: list[dict[str, Any]]) -> str:
        """Return the final assistant message content from an Inspector conversation."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return str(msg.get("content", "") or "")
        return ""

    @staticmethod
    def _normalize_reward_metric(value: Any) -> str:
        metric = str(value or "dice").strip().lower().replace("-", "_")
        aliases = {
            "dice": "dice",
            "f1": "dice",
            "acc": "acc_at_k",
            "acc@k": "acc_at_k",
            "accuracy_at_k": "acc_at_k",
            "top_k_accuracy": "acc_at_k",
            "acc_at_k": "acc_at_k",
        }
        if metric not in aliases:
            raise ValueError(
                "Unsupported code localization reward_metric "
                f"{value!r}. Expected one of: dice, acc_at_k."
            )
        return aliases[metric]

    def _parse_reward_acc_at_k(self, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"none", "null"}:
                return None
            if normalized in {"gt", "len_gt", "gt_size", "ground_truth", "ground_truth_size"}:
                self.reward_acc_at_k_from_gt = True
                return None
            try:
                parsed = int(normalized)
            except ValueError as exc:
                raise ValueError(
                    "reward_acc_at_k must be a positive integer or one of "
                    "{gt, len_gt, gt_size, ground_truth}."
                ) from exc
        else:
            parsed = int(value)
        if parsed <= 0:
            raise ValueError("reward_acc_at_k must be positive when set explicitly.")
        return parsed

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
                return False
        return bool(value)

    @staticmethod
    def _normalize_location(loc: str) -> str:
        return str(loc).strip()

    @classmethod
    def _normalized_location_set(cls, locations: list[str]) -> set[str]:
        return {cls._normalize_location(loc) for loc in locations if cls._normalize_location(loc)}

    @staticmethod
    def _extract_location_file(loc: str) -> str:
        loc = str(loc).strip()
        return loc.split(":", 1)[0].strip() if ":" in loc else loc

    @staticmethod
    def _tool_success_rate(total_tool_calls: int, successful_tool_calls: int) -> float:
        if total_tool_calls == 0:
            return 0.0
        return successful_tool_calls / total_tool_calls

    @classmethod
    def _top_k_accuracy(cls, ground_truth_locations: list[str], predicted_locations: list[str], k: int) -> float:
        if k <= 0:
            return 0.0
        gt_set = cls._normalized_location_set(ground_truth_locations)
        if not gt_set:
            return 0.0
        pred_set = cls._normalized_location_set(predicted_locations[:k])
        return float(gt_set.issubset(pred_set))

    @classmethod
    def _f1_dice_score(cls, ground_truth_locations: list[str], predicted_locations: list[str]) -> float:
        gt_set = cls._normalized_location_set(ground_truth_locations)
        pred_set = cls._normalized_location_set(predicted_locations)
        if not gt_set or not pred_set:
            return 0.0
        return 2.0 * len(gt_set & pred_set) / (len(gt_set) + len(pred_set))

    def _add_common_reward_metrics(
        self,
        reward_result: dict[str, Any],
        *,
        predicted_locations: list[str],
        ground_truth_locations: list[str],
    ) -> None:
        k = self._resolve_reward_acc_at_k(ground_truth_locations)
        reward_result.setdefault(
            "acc_at_k",
            self._top_k_accuracy(ground_truth_locations, predicted_locations, k),
        )
        reward_result.setdefault("k", k)
        reward_result.setdefault("k_from_gt", self.reward_acc_at_k_from_gt)
        reward_result.setdefault("gt_size", len(self._normalized_location_set(ground_truth_locations)))
        f1_dice = self._f1_dice_score(ground_truth_locations, predicted_locations)
        reward_result.setdefault("dice", f1_dice)
        reward_result["f1_dice_score"] = f1_dice

    @staticmethod
    def _build_reward_extra_info(reward_result: dict[str, Any]) -> dict[str, float]:
        extra_info = {}
        for key in ("tool_success_rate", "acc_at_k", "f1_dice_score"):
            value = reward_result.get(key)
            if value is None:
                continue
            try:
                extra_info[key] = float(value)
            except (TypeError, ValueError):
                continue
        return extra_info

    def _resolve_reward_acc_at_k(self, ground_truth_locations: list[str]) -> int:
        if self.reward_acc_at_k_from_gt:
            return len(self._normalized_location_set(ground_truth_locations))
        if self.reward_acc_at_k is not None:
            return self.reward_acc_at_k
        return self.top_k_predictions

    def _prediction_limit_for_reward(
        self,
        ground_truth_locations: list[str],
        *,
        explored_files: Optional[set[str]] = None,
    ) -> int:
        if self.reward_metric != "acc_at_k":
            return self.top_k_predictions
        if explored_files is not None:
            ground_truth_locations = self._filter_ground_truth_by_files(
                ground_truth_locations, explored_files
            )
        reward_k = self._resolve_reward_acc_at_k(ground_truth_locations)
        return max(self.top_k_predictions, reward_k)

    def _compute_navigator_acc_at_k_reward(
        self,
        predicted_locations: list[str],
        ground_truth_locations: list[str],
        total_tool_calls: int = 0,
        successful_tool_calls: int = 0,
    ) -> dict[str, Any]:
        k = self._resolve_reward_acc_at_k(ground_truth_locations)
        acc = self._top_k_accuracy(ground_truth_locations, predicted_locations, k)
        f1_dice = self._f1_dice_score(ground_truth_locations, predicted_locations)
        tool_rate = self._tool_success_rate(total_tool_calls, successful_tool_calls)
        return {
            "score": acc + tool_rate,
            "reward_metric": "acc_at_k",
            "acc_at_k": acc,
            "dice": f1_dice,
            "f1_dice_score": f1_dice,
            "k": k,
            "k_from_gt": self.reward_acc_at_k_from_gt,
            "gt_size": len(self._normalized_location_set(ground_truth_locations)),
            "tool_success_rate": tool_rate,
        }

    def _compute_inspector_acc_at_k_reward(
        self,
        predicted_locations: list[str],
        ground_truth_locations: list[str],
        explored_files: set[str],
        total_tool_calls: int = 0,
        successful_tool_calls: int = 0,
    ) -> dict[str, Any]:
        relevant_ground_truth = self._filter_ground_truth_by_files(
            ground_truth_locations, explored_files
        )
        tool_rate = self._tool_success_rate(total_tool_calls, successful_tool_calls)
        if not relevant_ground_truth:
            return {
                "score": None,
                "reward_metric": "acc_at_k",
                "acc_at_k": 0.0,
                "dice": 0.0,
                "f1_dice_score": 0.0,
                "k": self._resolve_reward_acc_at_k(relevant_ground_truth),
                "k_from_gt": self.reward_acc_at_k_from_gt,
                "gt_size": 0,
                "tool_success_rate": tool_rate,
                "f_relevant_size": 0,
                "discard": True,
            }

        k = self._resolve_reward_acc_at_k(relevant_ground_truth)
        acc = self._top_k_accuracy(relevant_ground_truth, predicted_locations, k)
        f1_dice = self._f1_dice_score(relevant_ground_truth, predicted_locations)
        return {
            "score": acc + tool_rate,
            "reward_metric": "acc_at_k",
            "acc_at_k": acc,
            "dice": f1_dice,
            "f1_dice_score": f1_dice,
            "k": k,
            "k_from_gt": self.reward_acc_at_k_from_gt,
            "gt_size": len(self._normalized_location_set(relevant_ground_truth)),
            "tool_success_rate": tool_rate,
            "f_relevant_size": len(self._normalized_location_set(relevant_ground_truth)),
            "discard": False,
        }

    @classmethod
    def _filter_ground_truth_by_files(
        cls,
        ground_truth_locations: list[str],
        explored_files: set[str],
    ) -> list[str]:
        return [
            loc
            for loc in ground_truth_locations
            if cls._extract_location_file(loc) in explored_files
        ]

    def _apply_response_length_penalty(
        self,
        reward_score: Optional[float],
        response_length: int,
    ) -> tuple[Optional[float], dict[str, Any]]:
        """Subtract proportional reward penalty when the raw response exceeds the limit."""
        limit = self.response_length
        overflow = max(response_length, limit) - limit
        overflow_ratio = overflow / limit if limit > 0 else 0.0
        penalty = self.length_penalty_coef * overflow_ratio
        if limit <= 0 or overflow == 0:
            return reward_score, {
                "raw_response_length": response_length,
                "response_length_limit": limit,
                "overflow_tokens": overflow,
                "overflow_ratio": overflow_ratio,
                "penalty_coef": self.length_penalty_coef,
                "penalty": 0.0,
            }

        penalized_score = reward_score - penalty if reward_score is not None else None
        return penalized_score, {
            "raw_response_length": response_length,
            "response_length_limit": limit,
            "overflow_tokens": overflow,
            "overflow_ratio": overflow_ratio,
            "penalty_coef": self.length_penalty_coef,
            "penalty": penalty,
        }

    async def _execute_inspector_tool_dicts(
        self,
        tool_calls: list[dict[str, Any]],
        instance_id: str,
        tool_stats: Optional[AgentToolStats] = None,
        all_tool_calls: Optional[list[dict[str, Any]]] = None,
    ) -> list[str]:
        """Execute Inspector tool calls shared by local and remote Inspector runs.

        Each non-exit tool result is a separate message with <code> wrapping,
        matching the inference pipeline. Non-exit tools are executed in
        parallel via asyncio.to_thread.
        """
        sub_tool_registry = build_sub_agent_tool_registry(instance_id)

        # --- Phase 1: Parse arguments and categorize ---
        invalid_messages: list[str] = []
        valid_calls: list[tuple[str, dict]] = []  # (tool_name, parsed_args)

        for tc in tool_calls:
            name = str(tc.get("name", "") or "")
            args = tc.get("arguments", {})
            if not isinstance(args, dict):
                if tool_stats is not None:
                    tool_stats.record(False)
                args = {}
                feedback = (
                    f"Result of {name}({json.dumps(args, ensure_ascii=False)}):\n"
                    f"<code>\nError: Invalid arguments for {name}\n</code>\n\n"
                    f"Continue your exploration based on the issue description. "
                    f"Call more tools if needed, or call exit() with a <result> block when you are done."
                )
                invalid_messages.append(feedback)
                continue

            if all_tool_calls is not None:
                all_tool_calls.append({"name": name, "arguments": args})

            if name == "exit":
                if tool_stats is not None:
                    tool_stats.record(True)
                continue

            valid_calls.append((name, args))

        # --- Phase 2: Execute non-exit tools in parallel ---
        formatted_messages: list[str] = list(invalid_messages)

        if valid_calls:
            def _exec_tool(name: str, args: dict) -> tuple[str, bool]:
                """Returns (result_str, success)."""
                if name in sub_tool_registry:
                    try:
                        return sub_tool_registry[name](args), True
                    except Exception as e:
                        return json.dumps({"error": str(e)}), False
                return f"Error: Unknown tool '{name}'", False

            # Parallel execution only when there are multiple calls
            if len(valid_calls) > 1:
                tasks = [
                    asyncio.to_thread(_exec_tool, name, args)
                    for name, args in valid_calls
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                name0, args0 = valid_calls[0]
                try:
                    raw_results = [_exec_tool(name0, args0)]
                except Exception as e:
                    raw_results = [e]

            for (name, args), raw in zip(valid_calls, raw_results):
                if isinstance(raw, Exception):
                    if tool_stats is not None:
                        tool_stats.record(False)
                    result = json.dumps({"error": str(raw)})
                else:
                    result, success = raw
                    if tool_stats is not None:
                        tool_stats.record(success)

                feedback = (
                    f"Result of {name}({json.dumps(args, ensure_ascii=False)}):\n"
                    f"<code>\n{result}\n</code>\n\n"
                    f"Continue your exploration based on the issue description. "
                    f"Call more tools if needed, or call exit() with a <result> block when you are done."
                )
                formatted_messages.append(feedback)

        return formatted_messages

    async def _finalize_inspector_local(
        self,
        *,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        request_id: str,
        prompt_ids: list[int],
        response_ids_all: list[int],
        response_mask_all: list[int],
        response_logprobs_all: list[float],
        metrics: dict[str, Any],
        termination_reason: str,
    ) -> None:
        """Finalize a local Inspector run using the same policy as remote."""
        if self._get_reusable_inspector_exit_result(messages, termination_reason):
            return

        finalize_message = [{"role": "user", "content": self.insp_finalize_prompt_template}]
        finalize_ids = await self.apply_chat_template(finalize_message, remove_system_prompt=True)
        if len(response_mask_all) + len(finalize_ids) >= self.response_length:
            return

        response_ids_all += finalize_ids
        response_mask_all += [0] * len(finalize_ids)
        if response_logprobs_all:
            response_logprobs_all += [0.0] * len(finalize_ids)
        messages.append({"role": "user", "content": self.insp_finalize_prompt_template})

        with simple_timer("generate_finalize", metrics):
            finalize_output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids + response_ids_all,
                sampling_params=sampling_params,
            )

        finalize_gen_ids = finalize_output.token_ids
        finalize_logprobs = finalize_output.log_probs
        full_text = self.tokenizer.decode(finalize_gen_ids, skip_special_tokens=True)
        finalize_text = truncate_at_second_think(full_text)
        if len(finalize_text) < len(full_text):
            trunc_len = len(self.tokenizer.encode(finalize_text, add_special_tokens=False))
            finalize_gen_ids = finalize_gen_ids[:trunc_len]
            if finalize_logprobs:
                finalize_logprobs = finalize_logprobs[:trunc_len]

        response_ids_all += finalize_gen_ids
        response_mask_all += [1] * len(finalize_gen_ids)
        if finalize_logprobs:
            response_logprobs_all += finalize_logprobs
        messages.append({"role": "assistant", "content": finalize_text})

    async def _empty_output(self, kwargs: dict) -> AgentLoopOutput:
        """Return an empty output when no Inspector can be run."""
        messages = [{"role": "system", "content": "No entry files available."}]
        prompt_ids = await self.apply_chat_template(messages)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=[],
            response_mask=[],
            reward_score=None,  # Will be filtered out by reward manager
            num_turns=0,
            metrics={},
            extra_fields={"discard": True},
        )

    def _extract_predicted_from_response(
        self,
        response_ids: list[int],
        state: GlobalState,
    ) -> list[str]:
        """Extract predicted locations from finalize response or confirmed_suspicious.

        Tries to parse <trace_locs> block from the decoded response. Falls back
        to extracting locations from state.confirmed_suspicious.
        """
        # Try to decode response and parse <trace_locs>
        try:
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            trace_locs = self._parse_trace_locs(response_text)
            if trace_locs:
                # Flatten to list of "file:function" format
                predicted = []
                for file_path, func_list in trace_locs.items():
                    for func_str in func_list:
                        for line in func_str.split("\n"):
                            line = line.strip()
                            if line.startswith("function:"):
                                func_name = line[len("function:"):].strip()
                                predicted.append(f"{file_path}:{func_name}")
                if predicted:
                    return predicted
        except Exception:
            pass

        # Fallback: use confirmed_suspicious from state
        return [s["location"] for s in state.confirmed_suspicious if "location" in s]

    @staticmethod
    def _parse_trace_locs(text: str) -> dict[str, list[str]]:
        """Parse <trace_locs>...</trace_locs> block into {file: [functions]}.

        Matches the format used in evaluation/multi_agent_inference.py.
        """
        trace_match = re.search(r"<trace_locs>(.*?)</trace_locs>", text, re.DOTALL)
        if not trace_match:
            return {}

        content = trace_match.group(1).strip()
        file_functions: dict[str, list[str]] = {}
        current_file: Optional[str] = None
        current_functions: list[str] = []

        def _flush():
            nonlocal current_file, current_functions
            if current_file and current_functions:
                if current_file in file_functions:
                    file_functions[current_file].extend(current_functions)
                else:
                    file_functions[current_file] = list(current_functions)
            current_file = None
            current_functions = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                _flush()
                continue
            if line.startswith("function:"):
                current_functions.append(line)
            elif line.endswith(".py"):
                _flush()
                current_file = line

        _flush()
        return {f: ["\n".join(funcs)] for f, funcs in file_functions.items()}
