"""
Multi-Agent Issue Localization Inference Pipeline.

This module implements the multi-agent framework described in TASK.md:
- Main Agent: navigates at repo level, dispatches Sub Agents
- Sub Agent: explores files in depth, traces caller/callee relationships

Usage:
    python -m evaluation.multi_agent_inference \
        --input_file data.parquet \
        --model_name gpt-4 \
        --model_backend openai \
        --api_key $OPENAI_API_KEY \
        --base_output_dir outputs/ \
        --run_name multi_agent_test \
        --graph_index_dir graph_index/ \
        --max_rounds 5 \
        --max_sub_agent_turns 10 \
        --assistant_response_prefill "<think>"
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import os.path as osp
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from string import Template
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Set

from evaluation.multi_agent_shared import (
    MAIN_AGENT_FIND_FILES_BY_CONTENT,
    MAIN_AGENT_FIND_FILES_BY_NAME,
    PROJECT_ROOT,
    PROMPTS_DIR,
    RESULT_RE,
    TOOL_CALL_RE,
    GlobalState,
    _load_text,
    augment_system_prompt_with_tool_instructions as _augment_system_prompt_with_tool_instructions,
    build_inspector_messages,
    build_main_agent_tool_registry,
    build_navigator_messages,
    build_sub_agent_tool_registry,
    extract_explored_from_tool_calls as _extract_explored_from_tool_calls,
    parse_tool_calls_from_text,
    truncate_at_second_think,
)
from evaluation.utils import load_data, extract_locs, _extract_trace_locs
from tqdm import tqdm

from tools.multi_agent_tools import (
    get_methods_of_class,
    get_file_functions,
    get_file_classes,
    get_code_of_function,
    get_code_of_class_method,
    get_callers,
    get_callees,
    find_files_by_content_repo,
    find_files_by_name_repo,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants (extracted from previously hardcoded values)
# ---------------------------------------------------------------------------

UTC_PLUS_8 = timezone(timedelta(hours=8))
MODEL_CALL_RETRY_SLEEP_SECONDS = 5
MODEL_CALL_MAX_RETRIES = 2
MODEL_MESSAGE_KEYS = {"role", "content", "name"}

# Sub Agent
SUB_AGENT_TIMEOUT_SECONDS = 300
SUB_AGENT_EXPLORED_DISPLAY_LIMIT = 100

# no_hierarchical_search ablation: cap on entry files dispatched in the single round
NO_HIERARCHICAL_SEARCH_MAX_ENTRIES = 2

# Termination heuristics
TERMINATION_SUSPICIOUS_THRESHOLD = 10

# Default model parameters
DEFAULT_MAX_TOKENS = 4096
DEFAULT_ASSISTANT_RESPONSE_PREFILL = ""

# Token budget for a single trajectory (Main Agent or Sub Agent conversation)
DEFAULT_MAIN_AGENT_TOKEN_BUDGET = 0        # 0 = unlimited
DEFAULT_SUB_AGENT_TOKEN_BUDGET = 0         # 0 = unlimited

# Ablation modes
ABLATION_NONE = "none"
ABLATION_NO_INDEPENDENT_CONTEXT = "no_independent_context"
ABLATION_NO_DYNAMIC_SCHEDULING = "no_dynamic_scheduling"
ABLATION_NO_HIERARCHICAL_SEARCH = "no_hierarchical_search"
ABLATION_CHOICES = (
    ABLATION_NONE,
    ABLATION_NO_INDEPENDENT_CONTEXT,
    ABLATION_NO_DYNAMIC_SCHEDULING,
    ABLATION_NO_HIERARCHICAL_SEARCH,
)

# PROMPTS_DIR, RESULT_RE, TOOL_CALL_RE imported from evaluation.multi_agent_shared


# _load_text imported from evaluation.multi_agent_shared


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(obj)


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_issue_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a dataset item into standard fields."""
    extra_info = item.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}

    problem_statement = _first_nonempty(
        item.get("problem_statement"),
        extra_info.get("problem_statement"),
        "",
    )
    title = _first_nonempty(
        item.get("title"),
        extra_info.get("title"),
    )
    if not title:
        title = str(problem_statement).splitlines()[0].strip() if problem_statement else "Untitled issue"

    return {
        "instance_id": str(_first_nonempty(item.get("instance_id"), extra_info.get("instance_id"), "unknown")),
        "repo": str(_first_nonempty(item.get("repo"), extra_info.get("repo"), "unknown")),
        "base_commit": _first_nonempty(item.get("base_commit"), extra_info.get("base_commit"), ""),
        "problem_statement": str(problem_statement or ""),
        "title": str(title),
        "data_source": _first_nonempty(item.get("data_source"), extra_info.get("data_source"), ""),
        "structure": str(_first_nonempty(item.get("structure"), extra_info.get("structure"), "") or ""),
    }



# _augment_system_prompt_with_tool_instructions imported from evaluation.multi_agent_shared


# ---------------------------------------------------------------------------
# Global State for Main Agent (imported from evaluation.multi_agent_shared)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool-call statistics tracker
# ---------------------------------------------------------------------------

@dataclass
class ToolCallTracker:
    """Accumulates tool-call statistics for evaluate.py compatibility.

    Tracks total / distinct / repeated / failed calls and computes
    ``tool_call_stats`` and ``fail_ratio`` in the format expected by
    ``evaluate.py``.
    """

    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    # _seen stores canonical (name, args_json) tuples for repeat detection
    _seen: set = field(default_factory=set)
    repeated_tool_calls: int = 0

    def record(self, tool_name: str, arguments: Dict[str, Any], *, failed: bool = False) -> None:
        """Record a single tool-call execution."""
        self.total_tool_calls += 1
        if failed:
            self.failed_tool_calls += 1
        key = (tool_name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
        if key in self._seen:
            self.repeated_tool_calls += 1
        else:
            self._seen.add(key)

    @property
    def distinct_tool_calls(self) -> int:
        return len(self._seen)

    @property
    def repeat_rate(self) -> float:
        if self.total_tool_calls <= 0:
            return 0.0
        return self.repeated_tool_calls / self.total_tool_calls

    @property
    def fail_ratio(self) -> float:
        if self.total_tool_calls <= 0:
            return 0.0
        return self.failed_tool_calls / self.total_tool_calls

    def to_dict(self) -> Dict[str, Any]:
        """Return the ``tool_call_stats`` dict expected by evaluate.py."""
        return {
            "total_tool_calls": self.total_tool_calls,
            "distinct_tool_calls": self.distinct_tool_calls,
            "repeated_tool_calls": self.repeated_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "repeat_rate": self.repeat_rate,
        }

    def merge(self, other: "ToolCallTracker") -> None:
        """Merge stats from *other* (e.g. a sub-agent) into this tracker."""
        self.total_tool_calls += other.total_tool_calls
        self.failed_tool_calls += other.failed_tool_calls
        self.repeated_tool_calls += other.repeated_tool_calls
        # We do NOT merge _seen sets because the sub-agent namespace is
        # separate from the main agent.  The distinct count is therefore
        # the sum of both.
        # But we do add the sub-agent's _seen count to our distinct count
        # conceptually — the property is derived from len(_seen) so we
        # must add sub-agent seen entries to our set under a namespace to
        # keep them separate.
        for key in other._seen:
            namespaced = ("__sub__", key)
            self._seen.add(namespaced)


# SUB_AGENT_TOOL_REGISTRY / MAIN_AGENT_TOOL_REGISTRY:
# Module-level registries kept for local dispatch convenience.
# New code should use build_sub_agent_tool_registry / build_main_agent_tool_registry
# from evaluation.multi_agent_shared.
SUB_AGENT_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "get_methods_of_class": lambda file_name, class_name, instance_id: get_methods_of_class(file_name, class_name, instance_id),
    "get_file_functions": lambda file_name, instance_id: get_file_functions(file_name, instance_id),
    "get_file_classes": lambda file_name, instance_id: get_file_classes(file_name, instance_id),
    "get_code_of_function": lambda file_name, func_name, instance_id: get_code_of_function(file_name, func_name, instance_id),
    "get_code_of_class_method": lambda file_name, class_name, func_name, instance_id: get_code_of_class_method(file_name, class_name, func_name, instance_id),
    "get_callers": lambda location, instance_id: get_callers(location, instance_id),
    "get_callees": lambda location, instance_id: get_callees(location, instance_id),
}

MAIN_AGENT_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    MAIN_AGENT_FIND_FILES_BY_CONTENT: lambda keywords, instance_id, file_pattern="*.py": find_files_by_content_repo(
        keywords, instance_id, file_pattern
    ),
    MAIN_AGENT_FIND_FILES_BY_NAME: lambda patterns, instance_id: find_files_by_name_repo(patterns, instance_id),
}

# _extract_explored_from_tool_calls imported from evaluation.multi_agent_shared


# ---------------------------------------------------------------------------
# Multi-Agent Localize Runner
# ---------------------------------------------------------------------------

@dataclass
class MultiAgentLocalizeRunner:
    """Runs the multi-agent issue localization pipeline."""

    args: Any
    llm: Optional[Any] = None

    def __post_init__(self) -> None:
        self.max_rounds = getattr(self.args, "max_rounds", 5)
        self.max_sub_agent_turns = getattr(self.args, "max_sub_agent_turns", 10)
        self.max_tokens = getattr(self.args, "max_tokens", 8192)
        self.temperature = getattr(self.args, "temperature", 0.2)
        self.top_p = getattr(self.args, "top_p", None)
        self.assistant_response_prefill = getattr(
            self.args,
            "assistant_response_prefill",
            DEFAULT_ASSISTANT_RESPONSE_PREFILL,
        ) or DEFAULT_ASSISTANT_RESPONSE_PREFILL
        self.sub_agent_parallelism = getattr(self.args, "sub_agent_parallelism", 3)
        self.ablation_mode = getattr(self.args, "ablation_mode", ABLATION_NONE) or ABLATION_NONE
        if self.ablation_mode not in ABLATION_CHOICES:
            raise ValueError(
                f"Unknown ablation_mode={self.ablation_mode!r}; expected one of {ABLATION_CHOICES}"
            )
        self.main_agent_token_budget = getattr(self.args, "main_agent_token_budget", DEFAULT_MAIN_AGENT_TOKEN_BUDGET) or 0
        self.sub_agent_token_budget = getattr(self.args, "sub_agent_token_budget", DEFAULT_SUB_AGENT_TOKEN_BUDGET) or 0
        self.base_output_dir = self.args.base_output_dir
        self.run_name = self.args.run_name
        self.time_str = datetime.now(UTC_PLUS_8).strftime("%Y%m%d_%H%M%S")
        self.resume_output_dir = osp.abspath(self.args.resume_output_dir) if getattr(self.args, "resume_output_dir", "") else ""
        if self.resume_output_dir:
            self.output_dir = self.resume_output_dir
        else:
            self.output_dir = os.path.join(self.base_output_dir, self.run_name, self.time_str)
        os.makedirs(self.output_dir, exist_ok=True)
        self.traj_dir = osp.join(self.output_dir, "traj")
        os.makedirs(self.traj_dir, exist_ok=True)
        self.trajs_path = osp.join(self.traj_dir, "trajs.jsonl")

        # Save args (skip overwrite when resuming and file already exists)
        self.args_path = osp.join(self.output_dir, "args.json")
        if not (self.resume_output_dir and osp.exists(self.args_path)):
            with open(self.args_path, "w", encoding="utf-8") as f:
                json.dump(vars(self.args) if hasattr(self.args, "__dict__") else {}, f, ensure_ascii=False, indent=2)

        # Set environment variables
        if hasattr(self.args, "graph_index_dir") and self.args.graph_index_dir:
            os.environ["GRAPH_INDEX_DIR"] = self.args.graph_index_dir

        # Instance filter
        self.instance_id_filter = self._build_instance_id_filter()
        self.start_index = max(int(getattr(self.args, "start_index", 0) or 0), 0)
        self.end_index = getattr(self.args, "end_index", None)
        self.processed_instance_ids: Set[str] = self._load_processed_instance_ids() if self.resume_output_dir else set()

        # Load prompts
        self.main_agent_system_prompt = _load_text(osp.join(PROMPTS_DIR, "main_agent_system_prompt.txt"))
        self.main_agent_user_prompt_template = _load_text(osp.join(PROMPTS_DIR, "main_agent_user_prompt.txt"))
        self.sub_agent_system_prompt = _load_text(osp.join(PROMPTS_DIR, "sub_agent_system_prompt.txt"))
        self.sub_agent_user_prompt_template = _load_text(osp.join(PROMPTS_DIR, "sub_agent_user_prompt.txt"))
        self.sub_agent_finalize_prompt_template = _load_text(osp.join(PROMPTS_DIR, "sub_agent_finalize_prompt.txt"))
        self.single_agent_system_prompt = _load_text(osp.join(PROMPTS_DIR, "single_agent_system_prompt.txt"))
        self.single_agent_user_prompt_template = _load_text(osp.join(PROMPTS_DIR, "single_agent_user_prompt.txt"))
        self.single_agent_finalize_prompt = _load_text(osp.join(PROMPTS_DIR, "single_agent_finalize_prompt.txt"))
        self.finalize_prompt_template = _load_text(osp.join(PROMPTS_DIR, "finalize_prompt.txt"))
        self._apply_ablation_prompt_overrides()

        # Load tool schemas
        tool_schemas_dir = osp.join(PROJECT_ROOT, "tools", "tool_schemas")
        with open(osp.join(tool_schemas_dir, "main_agent_tools.json"), "r") as f:
            self.main_agent_tool_schemas = json.load(f)
        with open(osp.join(tool_schemas_dir, "sub_agent_tools.json"), "r") as f:
            self.sub_agent_tool_schemas = json.load(f)
        with open(osp.join(tool_schemas_dir, "single_agent_tools.json"), "r") as f:
            self.single_agent_tool_schemas = json.load(f)

        # Build augmented system prompts (tools embedded in system prompt, not passed as API param)
        self.main_agent_augmented_system_prompt = _augment_system_prompt_with_tool_instructions(
            self.main_agent_system_prompt, self.main_agent_tool_schemas
        )
        self.sub_agent_augmented_system_prompt = _augment_system_prompt_with_tool_instructions(
            self.sub_agent_system_prompt, self.sub_agent_tool_schemas
        )
        self.single_agent_augmented_system_prompt = _augment_system_prompt_with_tool_instructions(
            self.single_agent_system_prompt,
            self.single_agent_tool_schemas,
        )

        # Build model(s)
        if self.llm is None:
            self.llm = self._build_model()
        self.sub_agent_llm = self._build_sub_agent_model()

        # Load tokenizer (optional, for token budget enforcement)
        self.tokenizer = None
        tokenizer_path = getattr(self.args, "tokenizer_path", "") or ""
        if tokenizer_path:
            self.tokenizer = self._load_tokenizer(tokenizer_path)

    def _apply_ablation_prompt_overrides(self) -> None:
        """Append mode-specific notes without modifying the prompt files."""
        if self.ablation_mode == ABLATION_NO_DYNAMIC_SCHEDULING:
            self.main_agent_user_prompt_template += (
                "\n\n<ablation_note>\n"
                "Ablation mode: dynamic scheduling is disabled. If you dispatch a file "
                "that was already explored, it will still be explored again instead of "
                "being skipped.\n"
                "</ablation_note>"
            )
        elif self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
            self.main_agent_user_prompt_template += (
                "\n\n<ablation_note>\n"
                "Ablation mode: multi-step hierarchical search is disabled. Once you "
                "call `dispatch`, the scheduler will dispatch all current pending "
                "entries, collect those results, and finalize without another search "
                "round.\n"
                "</ablation_note>"
            )

    def _build_model(self):
        from evaluation.models import build_model
        backend_kwargs = {
            "api_key": getattr(self.args, "api_key", None),
            "base_url": getattr(self.args, "base_url", None),
            "timeout": 120,
        }
        return build_model(self.args.model_backend, self.args.model_name, **backend_kwargs)

    def _build_sub_agent_model(self):
        """Build a separate model for Sub Agent (Inspector).

        Falls back to self.llm when no sub-agent-specific model config is provided.
        """
        from evaluation.models import build_model
        sub_model_name = getattr(self.args, "sub_agent_model_name", "") or ""
        sub_model_backend = getattr(self.args, "sub_agent_model_backend", "") or ""
        if not sub_model_name:
            return self.llm
        backend = sub_model_backend or self.args.model_backend
        backend_kwargs = {
            "api_key": getattr(self.args, "sub_agent_api_key", None) or getattr(self.args, "api_key", None),
            "base_url": getattr(self.args, "sub_agent_base_url", None) or getattr(self.args, "base_url", None),
            "timeout": 120,
        }
        return build_model(backend, sub_model_name, **backend_kwargs)

    @staticmethod
    def _load_tokenizer(tokenizer_path: str):
        """Load a HuggingFace tokenizer for token counting."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_path)

    # -------------------------------------------------------------------
    # Token counting helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _normalize_message_content_for_tokenizer(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, sort_keys=True)

    def _message_to_tokenizer_payload(self, message: Dict[str, Any]) -> Dict[str, str]:
        payload: Dict[str, str] = {
            "role": str(message.get("role", "") or "user"),
            "content": self._normalize_message_content_for_tokenizer(message.get("content", "")),
        }
        name = message.get("name")
        if name:
            payload["name"] = str(name)
        return payload

    def _count_tokenized_text(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _estimate_content_tokens(self, content: str) -> int:
        """Estimate token count for a single content string.

        Uses the actual tokenizer when available; otherwise falls back to
        a simple ``len(content) // 4`` heuristic (≈4 chars per token).
        """
        if self.tokenizer is not None:
            return self._count_tokenized_text(content)
        return max(len(content) // 4, 0)

    def _count_messages_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Count the total tokens in a list of messages using the tokenizer.

        Returns 0 when no tokenizer is loaded.
        """
        if self.tokenizer is None:
            return 0
        payloads = [
            self._message_to_tokenizer_payload(m)
            for m in messages
            if isinstance(m, dict)
        ]
        if not payloads:
            return 0
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                rendered = self.tokenizer.apply_chat_template(
                    payloads,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                if isinstance(rendered, str):
                    return self._count_tokenized_text(rendered)
            except TypeError:
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        payloads,
                        tokenize=False,
                    )
                    if isinstance(rendered, str):
                        return self._count_tokenized_text(rendered)
                except Exception:
                    pass
            except Exception:
                pass
        serialized = "\n\n".join(
            json.dumps(p, ensure_ascii=False, sort_keys=True) for p in payloads
        )
        return self._count_tokenized_text(serialized)

    def _is_token_budget_exceeded(self, messages: List[Dict[str, Any]], budget: int) -> bool:
        """Check whether the conversation has exceeded the token budget.

        Returns False when budget is 0 (unlimited) or no tokenizer is loaded.
        """
        if budget <= 0 or self.tokenizer is None:
            return False
        used = self._count_messages_tokens(messages)
        return used >= budget

    def _build_instance_id_filter(self) -> Optional[set]:
        selected_ids = []
        raw_ids = getattr(self.args, "instance_ids", "")
        if raw_ids:
            selected_ids.extend(i.strip() for i in str(raw_ids).split(",") if i.strip())
        ids_file = getattr(self.args, "instance_ids_file", "")
        if ids_file and osp.exists(ids_file):
            with open(ids_file, "r") as f:
                for line in f:
                    v = line.strip()
                    if v and not v.startswith("#"):
                        selected_ids.append(v)
        return set(selected_ids) if selected_ids else None

    def _load_processed_instance_ids(self) -> Set[str]:
        """Read already-completed instance IDs from the existing trajs.jsonl."""
        if not osp.exists(self.trajs_path):
            return set()
        processed_ids: Set[str] = set()
        with open(self.trajs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = record.get("instance_id")
                if instance_id:
                    processed_ids.add(str(instance_id))
        logger.info("Resume: found %d already-processed instances in %s", len(processed_ids), self.trajs_path)
        return processed_ids

    def _select_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        end = self.end_index if self.end_index is not None else len(data)
        selected = data[self.start_index:end]
        if self.instance_id_filter:
            selected = [
                item for item in selected
                if _normalize_issue_item(item)["instance_id"] in self.instance_id_filter
            ]
        if self.processed_instance_ids:
            before = len(selected)
            selected = [
                item for item in selected
                if _normalize_issue_item(item)["instance_id"] not in self.processed_instance_ids
            ]
            skipped = before - len(selected)
            if skipped:
                logger.info("Resume: skipping %d already-processed instances, %d remaining", skipped, len(selected))
        return selected

    def _call_model(self, messages: List[Dict[str, Any]],
                    max_tokens: int = DEFAULT_MAX_TOKENS,
                    llm: Optional[Any] = None) -> Any:
        """Call the LLM with retry logic. Tools are NOT passed as API parameter."""
        if llm is None:
            llm = self.llm
        prefill = getattr(self, "assistant_response_prefill", DEFAULT_ASSISTANT_RESPONSE_PREFILL) or ""
        payload_messages = self._prepare_model_messages(messages, prefill)
        params = {
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if self.top_p is not None:
            params["top_p"] = self.top_p

        for attempt in range(MODEL_CALL_MAX_RETRIES):
            try:
                t0 = time.time()
                response = llm.chat_response(payload_messages, **params)
                if hasattr(self, '_model_call_times'):
                    self._model_call_times.append(time.time() - t0)
                return self._with_assistant_response_prefill(response, prefill)
            except Exception as e:
                logger.warning(f"Model call failed (attempt {attempt + 1}): {e}")
                if attempt < MODEL_CALL_MAX_RETRIES - 1:
                    time.sleep(MODEL_CALL_RETRY_SLEEP_SECONDS)
                else:
                    raise

    def _prepare_model_messages(
        self,
        messages: List[Dict[str, Any]],
        assistant_response_prefill: str = "",
    ) -> List[Dict[str, Any]]:
        """Return API-safe messages, optionally ending with an assistant prefill."""
        payload_messages = [
            {
                key: value
                for key, value in msg.items()
                if key in MODEL_MESSAGE_KEYS
            }
            for msg in messages
            if isinstance(msg, dict)
        ]
        if assistant_response_prefill:
            payload_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response_prefill,
                }
            )
        return payload_messages

    @staticmethod
    def _with_assistant_response_prefill(response: Any, assistant_response_prefill: str = "") -> Any:
        """Expose the prefilled content as part of the returned assistant text."""
        if not assistant_response_prefill:
            return response
        if not response or not getattr(response, "choices", None):
            return response

        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        content = getattr(message, "content", "") if message is not None else ""
        content_text = content if isinstance(content, str) else str(content or "")
        if content_text and content_text.startswith(assistant_response_prefill):
            return response

        wrapped_message = SimpleNamespace(
            **{
                key: value
                for key, value in vars(message).items()
                if not key.startswith("_")
            }
        ) if hasattr(message, "__dict__") else SimpleNamespace()
        wrapped_message.content = assistant_response_prefill + content_text

        wrapped_choice = SimpleNamespace(
            **{
                key: value
                for key, value in vars(first_choice).items()
                if not key.startswith("_")
            }
        ) if hasattr(first_choice, "__dict__") else SimpleNamespace()
        wrapped_choice.message = wrapped_message

        wrapped_response = SimpleNamespace(
            **{
                key: value
                for key, value in vars(response).items()
                if not key.startswith("_")
            }
        ) if hasattr(response, "__dict__") else SimpleNamespace()
        wrapped_response.choices = [wrapped_choice, *list(response.choices[1:])]
        return wrapped_response

    def _extract_response_text(self, response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            return ""
        message = response.choices[0].message
        return getattr(message, "content", "") or ""

    def _make_message(self, role: str, content: str, **extra: Any) -> Dict[str, Any]:
        """Build a message dict with an estimated ``token_count`` field.

        ``evaluate.py`` reads ``token_count`` from every message to compute
        usage statistics.  We estimate it here so that the JSONL output is
        directly compatible.
        """
        msg: Dict[str, Any] = {"role": role, "content": content}
        msg["token_count"] = self._estimate_content_tokens(content)
        msg.update(extra)
        return msg

    @staticmethod
    def _is_tool_result_error(result_str: str) -> bool:
        """Check whether a tool result JSON indicates an error."""
        try:
            parsed = json.loads(result_str)
            if isinstance(parsed, dict) and "error" in parsed:
                return True
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from <tool_call>...</tool_call> tags in text.

        Delegates to the shared implementation in ``evaluation.multi_agent_shared``.
        """
        return parse_tool_calls_from_text(text)

    # -------------------------------------------------------------------
    # Main run loop
    # -------------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        data = load_data(self.args.input_file)
        data = self._select_data(data)
        outputs = []
        progress_bar = tqdm(data, desc="Multi-Agent Localization", dynamic_ncols=True)

        for item in progress_bar:
            meta = _normalize_issue_item(item)
            progress_bar.set_postfix_str(f"issue={meta['instance_id']}")
            result = self._run_item(item)

            instance_id = meta["instance_id"]
            sub_agent_trajs = result.get("sub_agent_trajectories", [])

            # Save each sub-agent trajectory to a separate file and build
            # lightweight summaries (without messages) for the main JSONL.
            sub_agent_summaries: List[Dict[str, Any]] = []
            if sub_agent_trajs:
                sa_dir = osp.join(self.traj_dir, "sub_agents", instance_id)
                os.makedirs(sa_dir, exist_ok=True)
                for idx, sa in enumerate(sub_agent_trajs):
                    entry_file = str(sa.get("entry_file", "") or "")
                    safe_name = entry_file.replace("/", "__").replace("\\", "__") or f"sub_{idx}"
                    sa_path = osp.join(sa_dir, f"{safe_name}.json")
                    with open(sa_path, "w", encoding="utf-8") as f:
                        json.dump(sa, f, ensure_ascii=False, indent=2)
                    # Lightweight summary (no messages) for the JSONL row
                    sub_agent_summaries.append({
                        "entry_file": entry_file,
                        "suspicious": sa.get("suspicious", []),
                        "explored": sa.get("explored", []),
                        "sub_agent_stats": sa.get("sub_agent_stats", {}),
                        "error": sa.get("error"),
                        "sub_agent_detail_path": osp.relpath(sa_path, self.output_dir),
                    })

            output = {
                "repo": meta["repo"],
                "instance_id": instance_id,
                "base_commit": meta.get("base_commit"),
                "data_source": meta.get("data_source"),
                "search_execution_mode": result.get("search_execution_mode", self._search_execution_mode()),
                "ablation_mode": result.get("ablation_mode", self.ablation_mode),
                "found_related_locs": result["found_related_locs"],
                "confirmed_suspicious": result["confirmed_suspicious"],
                "global_state": result["global_state"],
                "messages": result["messages"],
                "round_count": result["round_count"],
                "tool_call_stats": result.get("tool_call_stats", {}),
                "fail_ratio": result.get("fail_ratio", 0.0),
                "sub_agent_trajectories": sub_agent_summaries,
                "sub_agent_tool_call_stats": result.get("sub_agent_tool_call_stats", {}),
                "ground_truth": item['reward_model'].get("ground_truth", []),
                "inference_time_seconds": result.get("inference_time_seconds", 0.0),
                "model_call_time_seconds": result.get("model_call_time_seconds", 0.0),
            }
            self._append_jsonl(self.trajs_path, output)
            outputs.append(output)
            self.processed_instance_ids.add(instance_id)

        return outputs

    def _run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full multi-agent pipeline for a single issue."""
        item_start_time = time.time()
        self._model_call_times = []
        meta = _normalize_issue_item(item)
        instance_id = meta["instance_id"]
        problem_statement = meta["problem_statement"]
        structure = meta["structure"]

        logger.info(f"Processing {instance_id}")

        state = GlobalState()
        all_messages: List[Dict[str, Any]] = []
        tracker = ToolCallTracker()
        sub_agent_trajectories: List[Dict[str, Any]] = []

        if self.ablation_mode == ABLATION_NO_INDEPENDENT_CONTEXT:
            round_count = self._single_agent_loop(
                instance_id=instance_id,
                problem_statement=problem_statement,
                structure=structure,
                state=state,
                messages=all_messages,
                tracker=tracker,
            )
            self._finalize_single_agent_suspicious(
                instance_id=instance_id,
                state=state,
                messages=all_messages,
            )
        else:
            # Run the Main Agent loop (search, dispatch, expand)
            round_count = self._main_agent_loop(
                instance_id=instance_id,
                problem_statement=problem_statement,
                structure=structure,
                state=state,
                messages=all_messages,
                tracker=tracker,
                sub_agent_trajectories=sub_agent_trajectories,
            )

        # Finalize: ask Main Agent for ranked top-5
        found_related_locs = self._finalize(
            instance_id=instance_id,
            state=state,
            messages=all_messages,
        )

        # Aggregate sub-agent stats (kept separate from main agent tracker)
        sub_agent_tool_call_stats = self._aggregate_sub_agent_stats(sub_agent_trajectories)

        inference_time_seconds = time.time() - item_start_time
        model_call_time_seconds = sum(self._model_call_times)

        return {
            "found_related_locs": found_related_locs,
            "confirmed_suspicious": state.confirmed_suspicious,
            "global_state": state.to_dict(),
            "messages": all_messages,
            "round_count": round_count,
            "tool_call_stats": tracker.to_dict(),
            "fail_ratio": tracker.fail_ratio,
            "sub_agent_trajectories": sub_agent_trajectories,
            "sub_agent_tool_call_stats": sub_agent_tool_call_stats,
            "ablation_mode": self.ablation_mode,
            "search_execution_mode": self._search_execution_mode(),
            "inference_time_seconds": inference_time_seconds,
            "model_call_time_seconds": model_call_time_seconds,
        }

    # -------------------------------------------------------------------
    # Single-Agent Ablation Loop
    # -------------------------------------------------------------------

    def _single_agent_loop(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        structure: str,
        state: GlobalState,
        messages: List[Dict[str, Any]],
        tracker: Optional[ToolCallTracker] = None,
    ) -> int:
        """Run the no-independent-context baseline in one shared conversation."""
        system_msg = self._make_message("system", self.single_agent_augmented_system_prompt)
        user_prompt = Template(self.single_agent_user_prompt_template).safe_substitute(
            problem_statement=problem_statement,
            repo_structure=structure,
        )
        user_msg = self._make_message("user", user_prompt)

        conversation = [system_msg, user_msg]
        messages.extend([system_msg, user_msg])

        round_count = 0
        no_new_leads_count = 0

        while round_count < self.max_rounds:
            round_count += 1
            logger.info(
                f"[{instance_id}] Single Agent round {round_count}, "
                f"pending={len(state.pending_entries)}, "
                f"suspicious={len(state.confirmed_suspicious)}"
            )

            if self._is_token_budget_exceeded(conversation, self.main_agent_token_budget):
                logger.info(f"[{instance_id}] Single Agent token budget exhausted.")
                break

            response = self._call_model(conversation)
            assistant_text = truncate_at_second_think(self._extract_response_text(response))
            assistant_msg = self._make_message("assistant", assistant_text)
            conversation.append(assistant_msg)
            messages.append(assistant_msg)

            if self._is_token_budget_exceeded(conversation, self.main_agent_token_budget):
                logger.info(
                    f"[{instance_id}] Single Agent token budget exceeded after response. "
                    "Rolling back last round and triggering finalize."
                )
                conversation.pop()
                messages.pop()
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()
                break

            parsed_result = self._parse_sub_agent_result(assistant_text)
            if parsed_result:
                state.add_suspicious(parsed_result.get("suspicious", []))

            tool_calls = self._parse_tool_calls_from_text(assistant_text)
            has_exit = any(tc["name"] == "exit" for tc in tool_calls)
            if has_exit:
                logger.info(f"[{instance_id}] Single Agent called exit.")
                break

            non_exit_calls = [tc for tc in tool_calls if tc["name"] != "exit"]
            if not non_exit_calls:
                if parsed_result:
                    break
                no_new_leads_count += 1
                if no_new_leads_count >= 2:
                    logger.info(f"[{instance_id}] Single Agent produced no new tool calls. Ending.")
                    break
                nudge_msg = self._make_message(
                    "user",
                    "Please continue with repository search or code-inspection tools, "
                    "or finish with a <result> block and exit when done.",
                )
                conversation.append(nudge_msg)
                messages.append(nudge_msg)
                continue

            if len(non_exit_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(non_exit_calls)) as pool:
                    futures = [
                        pool.submit(
                            self._execute_single_agent_tool,
                            tc["name"],
                            tc.get("arguments", {}),
                            instance_id,
                        )
                        for tc in non_exit_calls
                    ]
                    raw_results = [f.result() for f in futures]
            else:
                raw_results = [
                    self._execute_single_agent_tool(
                        non_exit_calls[0]["name"],
                        non_exit_calls[0].get("arguments", {}),
                        instance_id,
                    )
                ]

            feedback_parts = []
            for tc, tool_result in zip(non_exit_calls, raw_results):
                failed = self._is_tool_result_error(tool_result)
                if tracker:
                    tracker.record(tc["name"], tc.get("arguments", {}), failed=failed)

                if tc["name"] in (MAIN_AGENT_FIND_FILES_BY_CONTENT, MAIN_AGENT_FIND_FILES_BY_NAME):
                    files = self._extract_files_from_tool_result(tc["name"], tool_result)
                    self._add_pending_files(state, files)
                else:
                    state.add_explored_from_functions(
                        _extract_explored_from_tool_calls([tc])
                    )

                feedback_parts.append(
                    f"Result of {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)}):\n"
                    f"<code>\n{tool_result}\n</code>"
                )

            feedback = "\n\n".join(feedback_parts)
            feedback += f"\n\nCurrent state:\n{self._state_summary(state)}"
            feedback += (
                "\n\nContinue searching or inspecting code as needed, or finish with "
                "a <result> block and exit when the suspicious locations are sufficient."
            )
            feedback_msg = self._make_message("user", feedback, _message_kind="tool_result")
            conversation.append(feedback_msg)
            messages.append(feedback_msg)
            no_new_leads_count = 0

            if self._should_terminate(state, round_count):
                logger.info(f"[{instance_id}] Single Agent termination criteria met.")
                break

        return round_count

    def _execute_single_agent_tool(self, tool_name: str, arguments: Dict[str, Any], instance_id: str) -> str:
        """Execute a tool available to the single-agent baseline."""
        if tool_name in (MAIN_AGENT_FIND_FILES_BY_CONTENT, MAIN_AGENT_FIND_FILES_BY_NAME):
            return self._execute_main_agent_tool(tool_name, arguments, instance_id)
        if tool_name == "dispatch":
            return json.dumps({"error": "dispatch is disabled in single-agent ablation mode"})
        return self._execute_sub_agent_tool(tool_name, arguments, instance_id)

    def _finalize_single_agent_suspicious(
        self,
        *,
        instance_id: str,
        state: GlobalState,
        messages: List[Dict[str, Any]],
    ) -> None:
        """Force a single-agent <result> block when the loop ended without one."""
        if state.confirmed_suspicious:
            return

        finalize_msg = self._make_message(
            "user",
            self.single_agent_finalize_prompt,
            _message_kind="finalize",
        )
        messages.append(finalize_msg)

        response = self._call_model(messages)
        finalize_text = truncate_at_second_think(self._extract_response_text(response))
        messages.append(self._make_message("assistant", finalize_text, _message_kind="finalize"))

        result = self._parse_sub_agent_result(finalize_text)
        if result:
            state.add_suspicious(result.get("suspicious", []))
            logger.info(
                f"[{instance_id}] Single Agent finalize collected "
                f"{len(result.get('suspicious', []))} suspicious locations."
            )

    # -------------------------------------------------------------------
    # Main Agent Loop
    # -------------------------------------------------------------------

    def _main_agent_loop(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        structure: str,
        state: GlobalState,
        messages: List[Dict[str, Any]],
        tracker: Optional[ToolCallTracker] = None,
        sub_agent_trajectories: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Main Agent interactive loop: search -> dispatch -> expand -> exit.

        Returns the number of rounds executed.
        """
        system_msg = self._make_message("system", self.main_agent_augmented_system_prompt)
        user_prompt = Template(self.main_agent_user_prompt_template).safe_substitute(
            problem_statement=problem_statement,
            repo_structure=structure,
        )
        user_msg = self._make_message("user", user_prompt)

        conversation = [system_msg, user_msg]
        messages.extend([system_msg, user_msg])

        round_count = 0
        no_new_leads_count = 0

        while round_count < self.max_rounds:
            round_count += 1
            logger.info(f"[{instance_id}] Main Agent round {round_count}, "
                        f"pending={len(state.pending_entries)}, "
                        f"suspicious={len(state.confirmed_suspicious)}")

            # Token budget check
            if self._is_token_budget_exceeded(conversation, self.main_agent_token_budget):
                logger.info(f"[{instance_id}] Main Agent token budget exhausted.")
                break

            # Get Main Agent response
            response = self._call_model(conversation)
            assistant_text = truncate_at_second_think(self._extract_response_text(response))
            assistant_msg = self._make_message("assistant", assistant_text)
            conversation.append(assistant_msg)
            messages.append(assistant_msg)

            # Post-response token budget check: if exceeded, roll back the
            # last user+assistant pair and break so _finalize runs on the
            # trimmed conversation.
            if self._is_token_budget_exceeded(conversation, self.main_agent_token_budget):
                logger.info(f"[{instance_id}] Main Agent token budget exceeded after response. "
                            "Rolling back last round and triggering finalize.")
                # Pop assistant
                conversation.pop()
                messages.pop()
                # Pop the preceding user message (tool feedback from prev round) if exists
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                    if messages and messages[-1].get("role") == "user":
                        messages.pop()
                break

            # Parse tool calls from text only
            tool_calls = self._parse_tool_calls_from_text(assistant_text)

            if not tool_calls:
                # No tool calls — treat as done
                logger.info(f"[{instance_id}] Main Agent produced no tool calls. Ending loop.")
                break

            # Check for exit
            has_exit = any(tc["name"] == "exit" for tc in tool_calls)
            if has_exit:
                logger.info(f"[{instance_id}] Main Agent called exit.")
                break

            # Check for dispatch
            has_dispatch = any(tc["name"] == "dispatch" for tc in tool_calls)

            # Execute file-discovery tool calls in parallel
            search_tool_calls = [
                tc for tc in tool_calls
                if tc["name"] in (MAIN_AGENT_FIND_FILES_BY_CONTENT, MAIN_AGENT_FIND_FILES_BY_NAME)
            ]
            tool_results = []
            if search_tool_calls:
                # Parallel execution only when there are multiple calls
                if len(search_tool_calls) > 1:
                    with ThreadPoolExecutor(max_workers=len(search_tool_calls)) as pool:
                        futures = [
                            pool.submit(self._execute_main_agent_tool, tc["name"], tc.get("arguments", {}), instance_id)
                            for tc in search_tool_calls
                        ]
                        raw_results = [f.result() for f in futures]
                else:
                    raw_results = [self._execute_main_agent_tool(
                        search_tool_calls[0]["name"], search_tool_calls[0].get("arguments", {}), instance_id
                    )]

                for tc, result in zip(search_tool_calls, raw_results):
                    failed = self._is_tool_result_error(result)
                    if tracker:
                        tracker.record(tc["name"], tc.get("arguments", {}), failed=failed)
                    tool_results.append({
                        "tool_name": tc["name"],
                        "arguments": tc.get("arguments", {}),
                        "result": result,
                    })
                    # Auto-add files from search results to pending
                    files = self._extract_files_from_tool_result(tc["name"], result)
                    self._add_pending_files(state, files)

            # Handle dispatch
            if has_dispatch:
                # Collect all entries requested across dispatch calls
                requested_entries: List[str] = []
                for tc in tool_calls:
                    if tc["name"] == "dispatch":
                        requested_entries.extend(
                            tc.get("arguments", {}).get("entries", [])
                        )

                entries_to_explore, skipped_entries, supplemented_entries = self._select_dispatch_entries(
                    requested_entries,
                    state,
                )

                # Remove all entries_to_explore from pending_entries
                explore_set = set(entries_to_explore)
                state.pending_entries = [
                    p for p in state.pending_entries if p not in explore_set
                ]

                if entries_to_explore:
                    # Run sub-agents in sequential batches of sub_agent_parallelism.
                    # After each batch, extract explored files from results and
                    # filter remaining entries that are now explored.
                    all_sub_results: List[Dict[str, Any]] = []
                    remaining = list(entries_to_explore)
                    batch_size = max(self.sub_agent_parallelism, 1)

                    while remaining:
                        batch = remaining[:batch_size]
                        remaining = remaining[batch_size:]

                        batch_results = self._run_sub_agents_parallel(
                            instance_id=instance_id,
                            problem_statement=problem_statement,
                            entry_files=batch,
                            state=state,
                        )

                        # Integrate batch results into state
                        for sr in batch_results:
                            state.add_suspicious(sr.get("suspicious", []))
                            # Extract files from explored functions and add to explored_entries
                            state.add_explored_from_functions(sr.get("explored", []))
                            entry_file = sr.get("entry_file", "")
                            if entry_file:
                                state.add_explored_entry(entry_file)

                        all_sub_results.extend(batch_results)

                        # Filter remaining entries: skip those now in explored_entries
                        if remaining and self.ablation_mode != ABLATION_NO_DYNAMIC_SCHEDULING:
                            filtered = [e for e in remaining if e not in state.explored_entries]
                            skipped_by_batch = len(remaining) - len(filtered)
                            if skipped_by_batch > 0:
                                logger.info(
                                    f"[{instance_id}] Skipping {skipped_by_batch} entries "
                                    f"already covered by previous batch exploration"
                                )
                            remaining = filtered

                    # Also remove newly explored entries from pending
                    if self.ablation_mode != ABLATION_NO_DYNAMIC_SCHEDULING:
                        state.pending_entries = [
                            p for p in state.pending_entries if p not in state.explored_entries
                        ]

                    # Collect sub-agent trajectories for output
                    if sub_agent_trajectories is not None:
                        for sr in all_sub_results:
                            sub_agent_trajectories.append({
                                "entry_file": sr.get("entry_file", ""),
                                "suspicious": sr.get("suspicious", []),
                                "explored": sr.get("explored", []),
                                "messages": sr.get("messages", []),
                                "sub_agent_stats": sr.get("sub_agent_stats", {}),
                                "error": sr.get("error"),
                            })

                    # Build feedback message for Main Agent
                    feedback = self._build_dispatch_feedback(
                        all_sub_results, state,
                        skipped_entries=skipped_entries,
                        supplemented_entries=supplemented_entries,
                    )
                    feedback_msg = self._make_message("user", feedback, _message_kind="tool_result")
                    conversation.append(feedback_msg)
                    messages.append(feedback_msg)
                    if self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
                        logger.info(
                            f"[{instance_id}] No-hierarchical-search ablation finalized after first dispatch."
                        )
                        break
                    no_new_leads_count = 0
                    continue
                else:
                    if self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
                        logger.info(
                            f"[{instance_id}] No-hierarchical-search ablation saw dispatch with no entries; finalizing."
                        )
                        feedback = self._build_dispatch_feedback(
                            [], state,
                            skipped_entries=skipped_entries,
                            supplemented_entries=supplemented_entries,
                        )
                        feedback_msg = self._make_message("user", feedback, _message_kind="tool_result")
                        conversation.append(feedback_msg)
                        messages.append(feedback_msg)
                        break
                    no_new_leads_count += 1

            # Build feedback from search tool results
            if tool_results:
                feedback_parts = []
                for tr in tool_results:
                    feedback_parts.append(
                        f"Result of {tr['tool_name']}({json.dumps(tr['arguments'], ensure_ascii=False)}):\n"
                        f"{tr['result']}"
                    )
                feedback = "\n\n".join(feedback_parts)
                feedback += f"\n\nCurrent state:\n{self._state_summary(state)}"
                feedback += (
                    "\n\nBased on these results, decide your next action: search more with "
                    "find_files_by_content/find_files_by_name, dispatch files to Sub Agents, or call exit if done."
                )
                feedback_msg = self._make_message("user", feedback, _message_kind="tool_result")
                conversation.append(feedback_msg)
                messages.append(feedback_msg)
            elif not has_dispatch:
                # No recognized tools called
                no_new_leads_count += 1
                if no_new_leads_count >= 2:
                    logger.info(f"[{instance_id}] No new leads for {no_new_leads_count} rounds. Ending.")
                    break
                nudge_msg = self._make_message(
                    "user",
                    "Please use find_files_by_content/find_files_by_name to search for files, "
                    "dispatch to explore them, or call exit when done.",
                )
                conversation.append(nudge_msg)
                messages.append(nudge_msg)

            # Check programmatic termination
            if self._should_terminate(state, round_count):
                logger.info(f"[{instance_id}] Termination criteria met.")
                break

        return round_count

    def _build_dispatch_feedback(
        self,
        sub_results: List[Dict[str, Any]],
        state: GlobalState,
        *,
        skipped_entries: Optional[List[str]] = None,
        supplemented_entries: Optional[List[str]] = None,
    ) -> str:
        """Build a feedback message for the Main Agent after sub-agent dispatch."""
        parts = []

        # Explain skipped / supplemented entries so the agent knows what happened
        if skipped_entries:
            parts.append(
                "Note: The following entries were skipped because they have "
                "already been explored:\n"
                + "\n".join(f"  - {e}" for e in skipped_entries)
            )
        if supplemented_entries:
            if self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
                parts.append(
                    "Note: The following pending entries were included for the "
                    "one-shot dispatch ablation:\n"
                    + "\n".join(f"  - {e}" for e in supplemented_entries)
                )
            else:
                parts.append(
                    "Note: The following entries were automatically supplemented "
                    "from the pending queue to fill the dispatch batch:\n"
                    + "\n".join(f"  - {e}" for e in supplemented_entries)
                )

        for sr in sub_results:
            entry = sr.get("entry_file", "unknown")
            suspicious = sr.get("suspicious", [])
            parts.append(f"Sub Agent explored '{entry}':")
            if suspicious:
                for s in suspicious:
                    parts.append(f"  - {s['location']}: {s['reason']}")
            else:
                parts.append("  (no suspicious locations found)")

        feedback = "Sub Agent results:\n" + "\n".join(parts)
        feedback += f"\n\nCurrent state:\n{self._state_summary(state)}"
        if self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
            feedback += "\n\nAblation complete: finalization will run after these dispatch results."
        else:
            feedback += (
                "\n\nBased on these results, decide your next action: search more with "
                "find_files_by_content/find_files_by_name, dispatch more files to Sub Agents, or call exit if done."
            )
        return feedback

    def _execute_main_agent_tool(self, tool_name: str, arguments: Dict[str, Any], instance_id: str) -> str:
        """Execute a Main Agent file-discovery tool."""
        try:
            handler = MAIN_AGENT_TOOL_REGISTRY.get(tool_name)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

            if tool_name == MAIN_AGENT_FIND_FILES_BY_CONTENT:
                # Accept both "keywords" (list) and legacy "keyword" (str)
                keywords = arguments.get("keywords", [])
                if not keywords:
                    keyword = arguments.get("keyword", "")
                    keywords = [keyword] if keyword else []
                return handler(
                    keywords=keywords,
                    instance_id=instance_id,
                    file_pattern=arguments.get("file_pattern", "*.py"),
                )
            elif tool_name == MAIN_AGENT_FIND_FILES_BY_NAME:
                # Accept both "patterns" (list) and legacy "pattern" (str)
                patterns = arguments.get("patterns", [])
                if not patterns:
                    pattern = arguments.get("pattern", "")
                    patterns = [pattern] if pattern else []
                return handler(
                    patterns=patterns,
                    instance_id=instance_id,
                )
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _extract_files_from_tool_result(self, tool_name: str, result_str: str) -> List[str]:
        """Extract file paths from a tool result."""
        try:
            result = json.loads(result_str)
        except json.JSONDecodeError:
            return []

        files = []
        if tool_name == MAIN_AGENT_FIND_FILES_BY_CONTENT:
            for match in result.get("results", []):
                f = match.get("file", "")
                if f:
                    files.append(f)
        elif tool_name == MAIN_AGENT_FIND_FILES_BY_NAME:
            files = result.get("matched_files", [])

        return files

    def _search_execution_mode(self) -> str:
        if self.ablation_mode == ABLATION_NO_INDEPENDENT_CONTEXT:
            return "single_agent"
        return "multi_agent"

    @staticmethod
    def _dedupe_preserve_order(entries: List[Any]) -> List[str]:
        seen: Set[str] = set()
        unique: List[str] = []
        for entry in entries:
            if entry is None:
                continue
            entry_str = str(entry).strip()
            if not entry_str or entry_str in seen:
                continue
            seen.add(entry_str)
            unique.append(entry_str)
        return unique

    def _add_pending_files(self, state: GlobalState, files: List[str]) -> None:
        """Add discovered files to pending according to the active scheduler mode."""
        if self.ablation_mode != ABLATION_NO_DYNAMIC_SCHEDULING:
            state.add_pending(files)
            return

        existing = set(state.pending_entries)
        for file_name in files:
            if not file_name or file_name in existing:
                continue
            state.pending_entries.append(file_name)
            existing.add(file_name)

    def _state_summary(self, state: GlobalState) -> str:
        if self.ablation_mode not in (
            ABLATION_NO_DYNAMIC_SCHEDULING,
            ABLATION_NO_INDEPENDENT_CONTEXT,
        ):
            return state.state_summary()

        explored_note = (
            "files already inspected by the single agent."
            if self.ablation_mode == ABLATION_NO_INDEPENDENT_CONTEXT
            else "files already explored; dynamic scheduling is disabled, so redispatches are not skipped."
        )
        pending_note = (
            "files discovered by search tools, available for direct inspection."
            if self.ablation_mode == ABLATION_NO_INDEPENDENT_CONTEXT
            else "files discovered by search tools, waiting for dispatch."
        )
        return (
            "# Current State\n"
            "# confirmed_suspicious: functions/methods identified as potentially causing the issue.\n"
            f"# explored_entries: {explored_note}\n"
            f"# pending_entries: {pending_note}\n"
            f"\nconfirmed_suspicious ({len(state.confirmed_suspicious)} items):\n"
            + "\n".join(f"  - {s['location']}: {s.get('reason', '')}" for s in state.confirmed_suspicious)
            + f"\n\nexplored_entries ({len(state.explored_entries)} items):\n"
            + "\n".join(f"  - {f}" for f in sorted(state.explored_entries))
            + f"\n\npending_entries ({len(state.pending_entries)} items):\n"
            + "\n".join(f"  - {f}" for f in state.pending_entries)
        )

    def _select_dispatch_entries(
        self,
        requested_entries: List[Any],
        state: GlobalState,
    ) -> tuple[List[str], List[str], List[str]]:
        """Return entries to explore plus skipped and supplemented entries."""
        unique_requested = self._dedupe_preserve_order(requested_entries)
        skip_explored = self.ablation_mode != ABLATION_NO_DYNAMIC_SCHEDULING

        skipped_entries: List[str] = []
        valid_entries: List[str] = []
        seen_valid: Set[str] = set()

        def add_if_schedulable(entry: str) -> None:
            if entry in seen_valid:
                return
            if skip_explored and entry in state.explored_entries:
                skipped_entries.append(entry)
                return
            valid_entries.append(entry)
            seen_valid.add(entry)

        if self.ablation_mode == ABLATION_NO_HIERARCHICAL_SEARCH:
            for entry in unique_requested:
                if len(valid_entries) >= NO_HIERARCHICAL_SEARCH_MAX_ENTRIES:
                    break
                add_if_schedulable(entry)
            requested_set = set(unique_requested)
            supplemented_entries: List[str] = []
            for entry in state.pending_entries:
                if len(valid_entries) >= NO_HIERARCHICAL_SEARCH_MAX_ENTRIES:
                    break
                before = len(valid_entries)
                add_if_schedulable(entry)
                if len(valid_entries) > before and entry not in requested_set:
                    supplemented_entries.append(entry)
            return valid_entries, skipped_entries, supplemented_entries

        for entry in unique_requested:
            add_if_schedulable(entry)

        # Supplement from pending if fewer valid entries than parallelism.
        supplemented_entries = []
        if len(valid_entries) < self.sub_agent_parallelism:
            need = self.sub_agent_parallelism - len(valid_entries)
            for pending_entry in state.pending_entries:
                if pending_entry in seen_valid:
                    continue
                if skip_explored and pending_entry in state.explored_entries:
                    continue
                supplemented_entries.append(pending_entry)
                seen_valid.add(pending_entry)
                if len(supplemented_entries) >= need:
                    break

        return valid_entries + supplemented_entries, skipped_entries, supplemented_entries

    # -------------------------------------------------------------------
    # Sub Agent Execution
    # -------------------------------------------------------------------

    def _run_sub_agents_parallel(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        entry_files: List[str],
        state: GlobalState,
    ) -> List[Dict[str, Any]]:
        """Run multiple Sub Agents in parallel."""
        results = []

        if self.sub_agent_parallelism <= 1 or len(entry_files) <= 1:
            for entry_file in entry_files:
                result = self._run_single_sub_agent(
                    instance_id=instance_id,
                    problem_statement=problem_statement,
                    entry_file=entry_file,
                    explored_entries=set(state.explored_entries),
                )
                results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=self.sub_agent_parallelism) as executor:
                futures = {}
                for entry_file in entry_files:
                    future = executor.submit(
                        self._run_single_sub_agent,
                        instance_id=instance_id,
                        problem_statement=problem_statement,
                        entry_file=entry_file,
                        explored_entries=set(state.explored_entries),
                    )
                    futures[future] = entry_file

                for future in as_completed(futures):
                    entry_file = futures[future]
                    try:
                        result = future.result(timeout=SUB_AGENT_TIMEOUT_SECONDS)
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Sub Agent for {entry_file} failed: {e}")
                        results.append({
                            "entry_file": entry_file,
                            "suspicious": [],
                            "explored": [],
                            "messages": [],
                            "sub_agent_stats": {},
                            "error": str(e),
                        })

        return results

    def _run_single_sub_agent(
        self,
        *,
        instance_id: str,
        problem_statement: str,
        entry_file: str,
        explored_entries: Set[str],
    ) -> Dict[str, Any]:
        """Run a single Sub Agent for a given entry file."""
        logger.info(f"[{instance_id}] Sub Agent starting for: {entry_file}")

        # Build conversation — tools are in the system prompt, not passed as API param
        system_msg = self._make_message("system", self.sub_agent_augmented_system_prompt)

        explored_str = (
            ", ".join(sorted(explored_entries)[:SUB_AGENT_EXPLORED_DISPLAY_LIMIT])
            if explored_entries
            else "None"
        )
        user_prompt = Template(self.sub_agent_user_prompt_template).safe_substitute(
            problem_statement=problem_statement,
            entry_file=entry_file,
            explored_functions=explored_str,
            explored_entries=explored_str,
        )
        user_msg = self._make_message("user", user_prompt)

        conversation = [system_msg, user_msg]

        # Track explored functions programmatically from tool calls (deduplicated)
        all_explored_set: Set[str] = set()
        all_explored: List[str] = []
        # Per-sub-agent tracker (stays independent, not merged into main)
        sub_tracker = ToolCallTracker()
        termination_reason = "max_turns_reached"

        # Tool interaction loop
        for turn in range(self.max_sub_agent_turns):
            # Pre-call token budget check
            if self._is_token_budget_exceeded(conversation, self.sub_agent_token_budget):
                logger.info(f"[{instance_id}] Sub Agent token budget exhausted for {entry_file}.")
                termination_reason = "token_budget_exhausted"
                break

            response = self._call_model(conversation, llm=self.sub_agent_llm)
            assistant_text = truncate_at_second_think(self._extract_response_text(response))
            conversation.append(self._make_message("assistant", assistant_text))

            # Post-response token budget check: roll back last user+assistant
            # and force early finalize so the next model call fits in budget.
            if self._is_token_budget_exceeded(conversation, self.sub_agent_token_budget):
                logger.info(f"[{instance_id}] Sub Agent token budget exceeded after response for {entry_file}. "
                            "Rolling back and forcing finalize.")
                # Pop the assistant we just appended
                conversation.pop()
                # Pop the preceding user message if present
                if conversation and conversation[-1].get("role") == "user":
                    conversation.pop()
                termination_reason = "token_budget_exhausted"
                break

            # Parse tool calls from text only
            tool_calls = self._parse_tool_calls_from_text(assistant_text)

            has_exit = any(tc["name"] == "exit" for tc in tool_calls)

            # Programmatically track explored functions from tool call arguments
            explored = _extract_explored_from_tool_calls(tool_calls)
            for loc in explored:
                if loc not in all_explored_set:
                    all_explored_set.add(loc)
                    all_explored.append(loc)

            if has_exit:
                termination_reason = "agent_called_exit"
                break

            # Execute non-exit tool calls in parallel, then build feedback messages in order
            non_exit_calls = [tc for tc in tool_calls if tc["name"] != "exit"]
            if not non_exit_calls:
                # No tool calls and no exit — ask for clarification
                conversation.append(self._make_message(
                    "user",
                    "Please continue your exploration or call exit() when done. "
                    "Remember to include your findings in a <result> block.",
                ))
                continue

            # Parallel execution only when there are multiple calls
            if len(non_exit_calls) > 1:
                with ThreadPoolExecutor(max_workers=len(non_exit_calls)) as pool:
                    futures = [
                        pool.submit(self._execute_sub_agent_tool, tc["name"], tc.get("arguments", {}), instance_id)
                        for tc in non_exit_calls
                    ]
                    raw_results = [f.result() for f in futures]
            else:
                raw_results = [self._execute_sub_agent_tool(
                    non_exit_calls[0]["name"], non_exit_calls[0].get("arguments", {}), instance_id
                )]

            for tc, tool_result in zip(non_exit_calls, raw_results):
                failed = self._is_tool_result_error(tool_result)
                sub_tracker.record(tc["name"], tc.get("arguments", {}), failed=failed)
                feedback = (
                    f"Result of {tc['name']}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)}):\n"
                    f"<code>\n{tool_result}\n</code>\n\n"
                    f"Continue your exploration based on the issue description. "
                    f"Call more tools if needed, or call exit() with a <result> block when you are done."
                )
                conversation.append(self._make_message("user", feedback, _message_kind="tool_result"))

        return self._finalize_sub_agent(
            instance_id=instance_id,
            entry_file=entry_file,
            conversation=conversation,
            explored=all_explored,
            sub_tracker=sub_tracker,
            termination_reason=termination_reason,
        )

    def _execute_sub_agent_tool(self, tool_name: str, arguments: Dict[str, Any], instance_id: str) -> str:
        """Execute a Sub Agent tool."""
        try:
            handler = SUB_AGENT_TOOL_REGISTRY.get(tool_name)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            return handler(**arguments, instance_id=instance_id)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _parse_sub_agent_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse the <result>...</result> block from Sub Agent response."""
        match = RESULT_RE.search(text)
        if not match:
            return None
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return {
                    "suspicious": result.get("suspicious", []),
                }
        except json.JSONDecodeError:
            pass
        return None

    def _collect_sub_agent_suspicious_candidates(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Collect suspicious candidates from the last assistant message only."""
        for msg in reversed(conversation):
            if msg.get("role") != "assistant":
                continue
            parsed = self._parse_sub_agent_result(str(msg.get("content", "") or ""))
            if not parsed:
                return []
            suspicious_list = parsed.get("suspicious", [])
            if not isinstance(suspicious_list, list):
                return []

            candidates: List[Dict[str, str]] = []
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

    def _extract_fallback_result(self, conversation: List[Dict[str, Any]], entry_file: str) -> Dict[str, Any]:
        """Extract any suspicious findings from conversation as fallback."""
        return {
            "entry_file": entry_file,
            "suspicious": self._collect_sub_agent_suspicious_candidates(conversation),
        }

    def _finalize_sub_agent(
        self,
        *,
        instance_id: str,
        entry_file: str,
        conversation: List[Dict[str, Any]],
        explored: List[str],
        sub_tracker: ToolCallTracker,
        termination_reason: str,
    ) -> Dict[str, Any]:
        """Finalize a sub-agent run and force a clean final <result> response.

        If the agent already called exit and produced a valid <result> block,
        skip the extra LLM call to save time.
        """
        # Try to reuse an existing <result> when the agent exited cleanly
        if termination_reason == "agent_called_exit":
            existing_suspicious = self._collect_sub_agent_suspicious_candidates(conversation)
            if existing_suspicious:
                result = {"suspicious": existing_suspicious}
                result["entry_file"] = entry_file
                result["explored"] = explored
                result["messages"] = conversation
                result["sub_agent_stats"] = sub_tracker.to_dict()
                logger.info(
                    f"[{instance_id}] Sub Agent finalized (reused exit result) for {entry_file}: "
                    f"{len(existing_suspicious)} suspicious, {len(explored)} explored, "
                    f"reason={termination_reason}"
                )
                return result

        finalize_msg = self._make_message(
            "user",
            self.sub_agent_finalize_prompt_template,
            _message_kind="finalize",
        )
        conversation.append(finalize_msg)

        response = self._call_model(conversation, llm=self.sub_agent_llm)
        finalize_text = truncate_at_second_think(self._extract_response_text(response))
        conversation.append(self._make_message("assistant", finalize_text, _message_kind="finalize"))

        result = self._parse_sub_agent_result(finalize_text)
        if result is None:
            result = self._extract_fallback_result(conversation, entry_file)

        result["entry_file"] = entry_file
        result["explored"] = explored
        result["messages"] = conversation
        result["sub_agent_stats"] = sub_tracker.to_dict()

        logger.info(
            f"[{instance_id}] Sub Agent finalized for {entry_file}: "
            f"{len(result.get('suspicious', []))} suspicious, {len(explored)} explored, "
            f"reason={termination_reason}"
        )
        return result

    # -------------------------------------------------------------------
    # Sub Agent stats aggregation
    # -------------------------------------------------------------------

    def _aggregate_sub_agent_stats(
        self, sub_agent_trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate tool-call statistics across all sub-agent trajectories.

        Returns a summary dict kept separate from the main-agent
        ``tool_call_stats`` so that main-agent and sub-agent metrics
        can be compared independently.
        """
        if not sub_agent_trajectories:
            return {}

        total_calls = 0
        total_failed = 0
        total_repeated = 0
        total_distinct = 0
        total_tokens = 0
        total_turns = 0
        total_tool_turns = 0
        total_assistant_turns = 0

        for sa in sub_agent_trajectories:
            stats = sa.get("sub_agent_stats") or {}
            total_calls += int(stats.get("total_tool_calls", 0) or 0)
            total_failed += int(stats.get("failed_tool_calls", 0) or 0)
            total_repeated += int(stats.get("repeated_tool_calls", 0) or 0)
            total_distinct += int(stats.get("distinct_tool_calls", 0) or 0)

            msgs = sa.get("messages") if isinstance(sa.get("messages"), list) else []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                tc = int(m.get("token_count", 0) or 0)
                total_tokens += tc
                total_turns += 1
                role = m.get("role", "")
                if role == "assistant":
                    total_assistant_turns += 1
                elif role == "user" or str(m.get("_message_kind", "")) == "tool_result":
                    total_tool_turns += 1

        count = len(sub_agent_trajectories)
        return {
            "sub_agent_count": count,
            "total_tool_calls": total_calls,
            "failed_tool_calls": total_failed,
            "repeated_tool_calls": total_repeated,
            "distinct_tool_calls": total_distinct,
            "fail_rate": total_failed / total_calls if total_calls > 0 else 0.0,
            "repeat_rate": total_repeated / total_calls if total_calls > 0 else 0.0,
            "total_tokens": total_tokens,
            "avg_tokens_per_sub_agent": total_tokens / count if count > 0 else 0,
            "total_turns": total_turns,
            "total_tool_turns": total_tool_turns,
            "total_assistant_turns": total_assistant_turns,
            "avg_turns_per_sub_agent": total_turns / count if count > 0 else 0,
            "avg_tool_calls_per_sub_agent": total_calls / count if count > 0 else 0,
        }

    # -------------------------------------------------------------------
    # Finalize: ranked output
    # -------------------------------------------------------------------

    def _finalize(
        self,
        *,
        instance_id: str,
        state: GlobalState,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Ask the Main Agent for a final ranked top-5 list in the existing context."""
        if not state.confirmed_suspicious:
            logger.warning(f"[{instance_id}] No suspicious locations to finalize.")
            return {}

        # Build confirmed suspicious summary
        suspicious_summary = "\n".join(
            f"  - {s['location']}: {s['reason']}"
            for s in state.confirmed_suspicious
        )

        finalize_prompt = Template(self.finalize_prompt_template).safe_substitute(
            confirmed_suspicious=suspicious_summary,
        )

        finalize_msg = self._make_message("user", finalize_prompt, _message_kind="finalize")
        messages.append(finalize_msg)

        response = self._call_model(messages)
        finalize_text = self._extract_response_text(response)
        messages.append(self._make_message("assistant", finalize_text, _message_kind="finalize"))

        # Parse <trace_locs> from the response
        found_related_locs = self._parse_trace_locs(finalize_text)

        if not found_related_locs:
            # Fallback: build from confirmed_suspicious
            found_related_locs = self._build_found_related_locs(state)

        return found_related_locs

    def _parse_trace_locs(self, text: str) -> Dict[str, List[str]]:
        """Parse <trace_locs>...</trace_locs> block into the standard format.

        Returns ``{file_path: ["function: A\\nfunction: B"]}`` — each file maps
        to a single-element list whose element is a newline-joined string of all
        ``function:`` lines.  When the same file appears in multiple blocks,
        their functions are merged rather than overwritten.
        """
        trace_match = re.search(r"<trace_locs>(.*?)</trace_locs>", text, re.DOTALL)
        if not trace_match:
            return {}

        content = trace_match.group(1).strip()
        # Collect all function lines per file, preserving order
        file_functions: Dict[str, List[str]] = {}
        current_file = None
        current_functions: List[str] = []

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
                # Save previous file if any, then start new file
                _flush()
                current_file = line

        # Don't forget the last entry
        _flush()

        # Convert to standard format: {file: ["\n".join(functions)]}
        return {f: ["\n".join(funcs)] for f, funcs in file_functions.items()}

    # -------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------

    def _should_terminate(self, state: GlobalState, round_count: int) -> bool:
        """Check if the search should terminate."""
        if round_count >= self.max_rounds:
            return True

        if len(state.confirmed_suspicious) >= TERMINATION_SUSPICIOUS_THRESHOLD:
            return True

        if not state.pending_entries and len(state.confirmed_suspicious) >= 1:
            return True

        return False

    # -------------------------------------------------------------------
    # Build output (fallback)
    # -------------------------------------------------------------------

    def _build_found_related_locs(self, state: GlobalState) -> Dict[str, List[str]]:
        """Convert confirmed_suspicious into the standard trace_locs format."""
        file_locs = {}
        for s in state.confirmed_suspicious:
            location = s["location"]
            if ":" not in location:
                continue
            file_path, func_name = location.split(":", 1)
            if file_path not in file_locs:
                file_locs[file_path] = []
            file_locs[file_path].append(f"function: {func_name}")

        # Format as the standard output
        result = {}
        for file_path, locs in file_locs.items():
            result[file_path] = ["\n".join(locs)]

        return result

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-Agent Issue Localization")
    parser.add_argument("--input_file", type=str, required=True, help="Input data file (parquet or jsonl)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--model_backend", type=str, default="openai", help="Model backend (openai/sglang)")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for API")
    parser.add_argument("--base_output_dir", type=str, default="outputs", help="Base output directory")
    parser.add_argument("--run_name", type=str, default="multi_agent", help="Run name")
    parser.add_argument("--graph_index_dir", type=str, default="graph_index", help="Directory containing dependency graph pickle files")
    parser.add_argument("--max_rounds", type=int, default=5, help="Maximum number of Main Agent rounds")
    parser.add_argument("--max_sub_agent_turns", type=int, default=10, help="Maximum turns per Sub Agent")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Maximum tokens per model call")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for model calls")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for model calls")
    parser.add_argument(
        "--assistant_response_prefill",
        "--assistant_first_token",
        dest="assistant_response_prefill",
        type=str,
        default=DEFAULT_ASSISTANT_RESPONSE_PREFILL,
        help=(
            "Optional text to prefill at the start of every assistant response. "
            "When set, the model call is made with a final assistant message "
            "containing this text, and the saved assistant response is prefixed "
            "with the same text."
        ),
    )
    parser.add_argument("--sub_agent_parallelism", type=int, default=3, help="Number of Sub Agents to run in parallel")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Tokenizer path for token counting (HuggingFace model name or local path)")
    parser.add_argument("--main_agent_token_budget", type=int, default=0, help="Max token budget for Main Agent trajectory (0 = unlimited)")
    parser.add_argument("--sub_agent_token_budget", type=int, default=0, help="Max token budget per Sub Agent trajectory (0 = unlimited)")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default=ABLATION_NONE,
        choices=ABLATION_CHOICES,
        help=(
            "Ablation setting: none, no_independent_context (single agent with "
            "Navigator+Inspector tools), no_dynamic_scheduling (do not skip "
            "explored files), or no_hierarchical_search (dispatch all pending "
            "entries after first dispatch and then finalize)."
        ),
    )
    parser.add_argument("--sub_agent_model_name", type=str, default="", help="Model name for Sub Agent (Inspector). Defaults to --model_name if empty.")
    parser.add_argument("--sub_agent_model_backend", type=str, default="", help="Model backend for Sub Agent (Inspector). Defaults to --model_backend if empty.")
    parser.add_argument("--sub_agent_api_key", type=str, default=None, help="API key for Sub Agent model. Defaults to --api_key if empty.")
    parser.add_argument("--sub_agent_base_url", type=str, default=None, help="Base URL for Sub Agent model API. Defaults to --base_url if empty.")
    parser.add_argument("--instance_ids", type=str, default="", help="Comma-separated list of instance IDs to process")
    parser.add_argument("--instance_ids_file", type=str, default="", help="File containing instance IDs to process")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in data")
    parser.add_argument("--end_index", type=int, default=None, help="End index in data")
    parser.add_argument(
        "--resume_output_dir",
        type=str,
        default="",
        help=(
            "Continue an existing run in-place by reusing this output directory. "
            "When set, inference appends to traj/trajs.jsonl and skips instance_ids "
            "already present there. If --load_config is omitted, the parser will "
            "try to load defaults from <resume_output_dir>/args.json."
        ),
    )
    parser.add_argument("--load_config", type=str, default="", help="Path to a JSON config file to load defaults from")
    return parser


def main():
    parser = _build_arg_parser()

    # Pre-parse to detect --load_config / --resume_output_dir before full parse,
    # so we can load defaults from a config file first.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--load_config", type=str, default="")
    pre_parser.add_argument("--resume_output_dir", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    config_path = ""
    if pre_args.load_config:
        config_path = pre_args.load_config
    elif pre_args.resume_output_dir:
        candidate = osp.join(osp.abspath(pre_args.resume_output_dir), "args.json")
        if osp.exists(candidate):
            config_path = candidate

    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        parser.set_defaults(**config)

    args = parser.parse_args()
    runner = MultiAgentLocalizeRunner(args)
    results = runner.run()
    print(f"\nProcessed {len(results)} issues.")
    for r in results:
        n_suspicious = len(r.get("confirmed_suspicious", []))
        print(f"  {r['instance_id']}: {n_suspicious} suspicious locations found in {r['round_count']} rounds")


if __name__ == "__main__":
    main()
