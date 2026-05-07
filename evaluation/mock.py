from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _mock_tool(query: str, **kwargs: Any) -> Dict[str, Any]:
    return {"result": f"mock result for {query}", "extra": kwargs}


def _default_mock_script() -> List[Dict[str, Any]]:
    return [
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "mock_tool",
                        "arguments": json.dumps({"query": "search term"}),
                    },
                }
            ],
            "usage": {"total_tokens": 32},
        },
        {
            "content": "<trace_locs>src/foo.py:Foo.bar</trace_locs>",
            "tool_calls": None,
            "usage": {"total_tokens": 16},
        },
    ]


class MockLLM:
    def __init__(self, scripted: List[Dict[str, Any]]) -> None:
        self.scripted = scripted
        self.index = 0

    def chat_response(self, messages: List[Dict[str, Any]], **params: Any) -> Any:
        if not self.scripted:
            return _MockResponse(_MockMessage(content="<trace_locs></trace_locs>"))
        payload = self.scripted[min(self.index, len(self.scripted) - 1)]
        self.index += 1
        return _MockResponse(
            _MockMessage(
                content=payload["content"],
                tool_calls=payload["tool_calls"],
            ),
            usage=payload["usage"],
        )


class _MockMessage:
    def __init__(self, content: str = "", tool_calls: Any = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _MockChoice:
    def __init__(self, message: _MockMessage) -> None:
        self.message = message


class _MockUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _MockResponse:
    def __init__(self, message: _MockMessage, usage: Optional[Dict[str, Any]] = None) -> None:
        self.choices = [_MockChoice(message)]
        if usage and "total_tokens" in usage:
            self.usage = _MockUsage(int(usage["total_tokens"]))
        else:
            self.usage = None


def build_mock_item() -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "You are a locator."},
            {"role": "user", "content": "Locate the bug in repo foo/bar."},
        ],
        "tools": [],
        "extra_info": {
            "instance_id": "mock-1",
            "repo": "foo/bar",
            "base_commit": "deadbeef",
        },
    }

