"""
Shared utilities for multi-agent code localization.

This module contains functions and classes shared between:
- evaluation/multi_agent_inference.py  (inference pipeline)
- verl/experimental/agent_loop/code_localization_loop.py  (RL training loop)

Keeping these in a single place ensures consistency in flow, parameters,
prompt settings, tool-call parsing, and state management.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import re
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
PROMPTS_DIR = osp.join(PROJECT_ROOT, "prompts", "multi_agent")

# Regex patterns
# TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
# TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*\})\s*</tool_call>', re.DOTALL | re.IGNORECASE)
TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{[^<]*\})\s*</tool_call>',  # [^<]* excludes angle brackets
    re.DOTALL | re.IGNORECASE
)

RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"<think>", re.IGNORECASE)

MAIN_AGENT_FIND_FILES_BY_CONTENT = "find_files_by_content"
MAIN_AGENT_FIND_FILES_BY_NAME = "find_files_by_name"


# ---------------------------------------------------------------------------
# Response truncation
# ---------------------------------------------------------------------------

def truncate_at_second_think(text: str) -> str:
    """Discard everything from the second ``<think>`` tag onward.

    Models sometimes produce multiple ``<think>``/``<tool_call>`` blocks in a
    single response.  Only the first block is intentional; the rest is
    hallucinated continuation.  This function keeps the text up to (but not
    including) the second ``<think>`` tag.

    Returns the original text unchanged if fewer than two ``<think>`` tags
    are found.
    """
    matches = list(THINK_TAG_RE.finditer(text))
    if len(matches) < 2:
        return text
    # Truncate right before the second <think>
    return text[:matches[1].start()].rstrip()


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def augment_system_prompt_with_tool_instructions(
    system_prompt: str, tool_schemas: List[Dict[str, Any]]
) -> str:
    """Append tool schemas and <tool_call> format instructions to the system prompt.

    This is the single canonical implementation used by both the inference
    pipeline and the verl RL training loop.
    """
    if not tool_schemas:
        return system_prompt
    content = str(system_prompt or "")
    content += (
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
    )
    for tool in tool_schemas:
        content += f"\n{json.dumps(tool, ensure_ascii=False)}"
    content += (
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
        "wrapped inside <tool_call></tool_call> XML tags.\n"
        "Use the following format:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>\n\n"
        "Examples:\n"
        "<tool_call>\n"
        '{"name": "get_code_of_file_function", "arguments": {"file_name": "pkg/module.py", "func_name": "run"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "get_callers", "arguments": {"file_name": "pkg/module.py", "func_name": "run"}}\n'
        "</tool_call>"
    )
    return content


def build_navigator_messages(
    system_prompt: str,
    user_prompt_template: str,
    tool_schemas: List[Dict[str, Any]],
    problem_statement: str,
    repo_structure: str,
) -> List[Dict[str, str]]:
    """Build the initial Navigator message list.

    Args:
        system_prompt: Raw system prompt text (will be augmented with tool instructions).
        user_prompt_template: Template string with ``$problem_statement`` and ``$repo_structure``.
        tool_schemas: Tool JSON schemas to embed in the system prompt.
        problem_statement: Issue description (will be truncated).
        repo_structure: Repository directory structure (will be truncated).
    """
    system = augment_system_prompt_with_tool_instructions(system_prompt, tool_schemas)
    user = Template(user_prompt_template).safe_substitute(
        problem_statement=problem_statement,
        repo_structure=repo_structure,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_inspector_messages(
    system_prompt: str,
    user_prompt_template: str,
    tool_schemas: List[Dict[str, Any]],
    problem_statement: str,
    entry_file: str,
) -> List[Dict[str, str]]:
    """Build the initial Inspector message list.

    Args:
        system_prompt: Raw system prompt text (will be augmented with tool instructions).
        user_prompt_template: Template string with ``$problem_statement`` and ``$entry_file``.
        tool_schemas: Tool JSON schemas to embed in the system prompt.
        problem_statement: Issue description (will be truncated).
        entry_file: The entry file path for this Inspector to explore.
    """
    system = augment_system_prompt_with_tool_instructions(system_prompt, tool_schemas)
    user = Template(user_prompt_template).safe_substitute(
        problem_statement=problem_statement,
        entry_file=entry_file,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------

def parse_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse tool calls from ``<tool_call>...</tool_call>`` tags in *text*.

    Accepts both ``"arguments"`` and ``"parameters"`` keys for the tool-call
    payload since models sometimes use either form.
    """
    tool_calls: List[Dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            tc = json.loads(match.group(1).strip())
            if isinstance(tc, dict) and "name" in tc:
                # Accept both "arguments" and "parameters" as the payload key
                args = tc.get("arguments") or tc.get("parameters") or {}
                tool_calls.append({
                    "name": tc["name"],
                    "arguments": args,
                })
        except json.JSONDecodeError:
            continue
    return tool_calls


# ---------------------------------------------------------------------------
# Explored-location extraction
# ---------------------------------------------------------------------------

def extract_explored_from_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> List[str]:
    """Programmatically extract explored function locations from sub-agent tool calls.

    Derives explored functions from the *arguments* of code-inspection and
    graph-traversal tools so that the LLM does not need to maintain or report
    a trajectory list.
    """
    explored: List[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if name == "get_code_of_function":
            file_name = args.get("file_name", "")
            func_name = args.get("func_name", "")
            if file_name and func_name:
                explored.append(f"{file_name}:{func_name}")

        elif name == "get_code_of_class_method":
            file_name = args.get("file_name", "")
            class_name = args.get("class_name", "")
            func_name = args.get("func_name", "")
            if file_name and class_name and func_name:
                explored.append(f"{file_name}:{class_name}.{func_name}")

        elif name in ("get_callers", "get_callees"):
            location = args.get("location", "")
            if location:
                explored.append(location)

        elif name == "get_file_functions":
            file_name = args.get("file_name", "")
            if file_name:
                explored.append(f"{file_name}:*")

        elif name == "get_file_classes":
            file_name = args.get("file_name", "")
            if file_name:
                explored.append(f"{file_name}:*")

        elif name == "get_methods_of_class":
            file_name = args.get("file_name", "")
            class_name = args.get("class_name", "")
            if file_name and class_name:
                explored.append(f"{file_name}:{class_name}.*")

    return explored


# ---------------------------------------------------------------------------
# Global State for Main Agent
# ---------------------------------------------------------------------------

@dataclass
class GlobalState:
    """Maintains the global state across all rounds of multi-agent search."""

    confirmed_suspicious: List[Dict[str, str]] = field(default_factory=list)
    explored_entries: Set[str] = field(default_factory=set)
    pending_entries: List[str] = field(default_factory=list)

    def add_suspicious(self, suspicious_list: List[Dict[str, str]]):
        existing_locations = {s["location"] for s in self.confirmed_suspicious}
        for s in suspicious_list:
            loc = s.get("location", "")
            if loc and loc not in existing_locations:
                self.confirmed_suspicious.append(s)
                existing_locations.add(loc)

    def add_explored_entry(self, entry: str):
        """Mark a file-level entry as explored."""
        self.explored_entries.add(entry)

    def add_explored_from_functions(self, functions: List[str]):
        """Extract file parts from explored function locations and add to explored_entries.

        For example, ``"a.py:func"`` or ``"a.py:Class.method"`` will add ``"a.py"``
        to ``explored_entries``.
        """
        for func_loc in functions:
            file_part = func_loc.split(":")[0] if ":" in func_loc else func_loc
            if file_part:
                self.explored_entries.add(file_part)

    def add_pending(self, files: List[str]):
        """Add new files to pending, deduplicating against explored_entries and pending."""
        for f in files:
            if f not in self.pending_entries and f not in self.explored_entries:
                self.pending_entries.append(f)

    def take_pending(self, n: int) -> List[str]:
        taken = self.pending_entries[:n]
        self.pending_entries = self.pending_entries[n:]
        return taken

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirmed_suspicious": self.confirmed_suspicious,
            "explored_entries": sorted(self.explored_entries),
            "pending_entries": self.pending_entries,
        }

    def state_summary(self) -> str:
        return (
            "# Current State\n"
            "# confirmed_suspicious: functions/methods identified as potentially causing the issue.\n"
            "# explored_entries: files already explored by Sub Agents (will be skipped if dispatched again).\n"
            "# pending_entries: files discovered by search tools, waiting for you to dispatch to Sub Agents.\n"
            f"\nconfirmed_suspicious ({len(self.confirmed_suspicious)} items):\n"
            + "\n".join(f"  - {s['location']}: {s.get('reason', '')}" for s in self.confirmed_suspicious)
            + f"\n\nexplored_entries ({len(self.explored_entries)} items):\n"
            + "\n".join(f"  - {f}" for f in sorted(self.explored_entries))
            + f"\n\npending_entries ({len(self.pending_entries)} items):\n"
            + "\n".join(f"  - {f}" for f in self.pending_entries)
        )


# ---------------------------------------------------------------------------
# Sub Agent tool registry builder
# ---------------------------------------------------------------------------

def build_sub_agent_tool_registry(instance_id: str) -> Dict[str, Any]:
    """Build a tool registry mapping tool names to callables for the sub-agent.

    Each callable accepts a single ``args`` dict and routes to the underlying
    ``tools.multi_agent_tools`` function with the bound *instance_id*.
    """
    from tools.multi_agent_tools import (
        get_methods_of_class,
        get_file_functions,
        get_file_classes,
        get_code_of_function,
        get_code_of_class_method,
        get_callers,
        get_callees,
    )

    return {
        "get_methods_of_class": lambda args: get_methods_of_class(
            args.get("file_name", ""), args.get("class_name", ""), instance_id),
        "get_file_functions": lambda args: get_file_functions(args.get("file_name", ""), instance_id),
        "get_file_classes": lambda args: get_file_classes(args.get("file_name", ""), instance_id),
        "get_code_of_function": lambda args: get_code_of_function(
            args.get("file_name", ""), args.get("func_name", ""), instance_id),
        "get_code_of_class_method": lambda args: get_code_of_class_method(
            args.get("file_name", ""), args.get("class_name", ""),
            args.get("func_name", ""), instance_id),
        "get_callers": lambda args: get_callers(args.get("location", ""), instance_id),
        "get_callees": lambda args: get_callees(args.get("location", ""), instance_id),
    }


def build_main_agent_tool_registry(instance_id: str) -> Dict[str, Any]:
    """Build a tool registry mapping tool names to callables for the main agent.

    Each callable accepts a single ``args`` dict and routes to the underlying
    ``tools.multi_agent_tools`` function with the bound *instance_id*.
    """
    from tools.multi_agent_tools import (
        find_files_by_content_repo,
        find_files_by_name_repo,
    )

    return {
        MAIN_AGENT_FIND_FILES_BY_CONTENT: lambda args: find_files_by_content_repo(
            args.get("keywords") or ([args.get("keyword")] if args.get("keyword") else []),
            instance_id,
            args.get("file_pattern", "*.py"),
        ),
        MAIN_AGENT_FIND_FILES_BY_NAME: lambda args: find_files_by_name_repo(
            args.get("patterns") or ([args.get("pattern")] if args.get("pattern") else []),
            instance_id,
        ),
    }
