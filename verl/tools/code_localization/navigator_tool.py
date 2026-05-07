"""
Navigator (Main Agent) tools for code localization RL training.

Wraps the existing tools from tools/multi_agent_tools.py into verl's BaseTool interface.
Tools: find_files_by_content, find_files_by_name, dispatch, exit.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FindFilesByContentTool(BaseTool):
    """Find files by matching keywords or regex patterns in file contents."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instances[instance_id] = {
            "instance_id": create_kwargs.get("instance_id", ""),
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        from tools.multi_agent_tools import find_files_by_content_repo

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        keywords = parameters.get("keywords", [])
        file_pattern = parameters.get("file_pattern", "*.py")

        try:
            result = find_files_by_content_repo(keywords, data_instance_id, file_pattern)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            error_msg = json.dumps({"error": str(e)}, ensure_ascii=False)
            return ToolResponse(text=error_msg), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class FindFilesByNameTool(BaseTool):
    """Find files by matching file name or path patterns in the repository."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        self._instances[instance_id] = {
            "instance_id": create_kwargs.get("instance_id", ""),
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        from tools.multi_agent_tools import find_files_by_name_repo

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        patterns = parameters.get("patterns", [])

        try:
            result = find_files_by_name_repo(patterns, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            error_msg = json.dumps({"error": str(e)}, ensure_ascii=False)
            return ToolResponse(text=error_msg), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class DispatchTool(BaseTool):
    """Dispatch Sub Agents to explore entry files.

    During Navigator training, Sub Agents run via remote vLLM.
    The actual dispatch logic is handled by the agent loop;
    this tool just records the dispatch request and returns a placeholder.
    The agent loop intercepts dispatch calls and injects real sub-agent results.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        entries = parameters.get("entries", [])
        # The actual sub-agent execution is handled by the agent loop.
        # This tool just returns the dispatch request info.
        # The agent loop will intercept "dispatch" tool calls and replace the response.
        return (
            ToolResponse(text=json.dumps({"dispatched": entries, "status": "pending"}, ensure_ascii=False)),
            0.0,
            {"success": True, "entries": entries},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class ExitTool(BaseTool):
    """Signal that the search is complete."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text="Search completed."), 0.0, {"success": True, "exit": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
