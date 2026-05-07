"""
Inspector (Sub Agent) tools for code localization RL training.

Wraps the existing tools from tools/multi_agent_tools.py into verl's BaseTool interface.
Tools: get_methods_of_class, get_file_functions, get_file_classes, get_code_of_function,
       get_code_of_class_method, get_callers, get_callees, exit.
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


class GetMethodsOfClassTool(BaseTool):
    """Get all method signatures of a specific class in a file."""

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
        from tools.multi_agent_tools import get_methods_of_class

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        file_name = parameters.get("file_name", "")
        class_name = parameters.get("class_name", "")
        try:
            result = get_methods_of_class(file_name, class_name, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetFileFunctionsTool(BaseTool):
    """Get all module-level function signatures in a file."""

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
        from tools.multi_agent_tools import get_file_functions

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        file_name = parameters.get("file_name", "")
        try:
            result = get_file_functions(file_name, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetFileClassesTool(BaseTool):
    """Get all class signatures in a file."""

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
        from tools.multi_agent_tools import get_file_classes

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        file_name = parameters.get("file_name", "")
        try:
            result = get_file_classes(file_name, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetCodeOfFunctionTool(BaseTool):
    """Get source code of a module-level function."""

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
        from tools.multi_agent_tools import get_code_of_function

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        file_name = parameters.get("file_name", "")
        func_name = parameters.get("func_name", "")
        try:
            result = get_code_of_function(file_name, func_name, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetCodeOfClassMethodTool(BaseTool):
    """Get source code of a class method."""

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
        from tools.multi_agent_tools import get_code_of_class_method

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        file_name = parameters.get("file_name", "")
        class_name = parameters.get("class_name", "")
        func_name = parameters.get("func_name", "")
        try:
            result = get_code_of_class_method(file_name, class_name, func_name, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetCallersTool(BaseTool):
    """Get functions that call a specified function."""

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
        from tools.multi_agent_tools import get_callers

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        location = parameters.get("location", "")
        try:
            result = get_callers(location, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class GetCalleesTool(BaseTool):
    """Get functions called by a specified function."""

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
        from tools.multi_agent_tools import get_callees

        inst = self._instances.get(instance_id, {})
        data_instance_id = inst.get("instance_id", "")

        location = parameters.get("location", "")
        try:
            result = get_callees(location, data_instance_id)
            return ToolResponse(text=result), 0.0, {"success": True}
        except Exception as e:
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {"success": False}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


class InspectorExitTool(BaseTool):
    """Signal that the Inspector has finished exploring."""

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
        return ToolResponse(text="Exploration completed."), 0.0, {"success": True, "exit": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
