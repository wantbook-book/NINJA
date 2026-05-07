
from __future__ import annotations
import asyncio
import atexit
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
except Exception:  # pragma: no cover - openai is an optional runtime dependency
    OpenAI = None  # type: ignore[assignment]
    ChatCompletion = Any  # type: ignore[misc,assignment]

import sglang as sgl
from transformers import AutoTokenizer


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        return base_url
    return f"{base_url}/v1"


def _resolve_openai_base_url() -> Optional[str]:
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def _resolve_sglang_base_url() -> Optional[str]:
    return os.getenv("SGLANG_BASE_URL") or os.getenv("SGLANG_API_BASE")


def _ensure_openai() -> None:
    if OpenAI is None:
        raise RuntimeError("openai package is required. Install it via `pip install openai`.")


def _extract_text_from_response(response: ChatCompletion) -> str:
    if not response or not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    return getattr(message, "content", "") or ""


class _SimpleMessage:
    def __init__(self, content: str, tool_calls: Optional[list[dict]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _SimpleChoice:
    def __init__(self, message: _SimpleMessage) -> None:
        self.message = message


class _SimpleUsage:
    def __init__(self, completion_tokens: int) -> None:
        self.completion_tokens = completion_tokens


class _SimpleChatCompletion:
    def __init__(
        self,
        content: str,
        completion_tokens: Optional[int] = None,
        tool_calls: Optional[list[dict]] = None,
    ) -> None:
        self.choices = [_SimpleChoice(_SimpleMessage(content, tool_calls=tool_calls))]
        self.usage = _SimpleUsage(completion_tokens) if completion_tokens is not None else None

@dataclass
class OpenAIChatModel:
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    max_retries: int = 3
    require_api_key: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_openai()
        base_url = self.base_url if self.base_url is not None else _resolve_openai_base_url()
        self.base_url = _normalize_base_url(base_url) if base_url else None
        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if self.require_api_key and not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI API calls.")
        if not api_key:
            api_key = "EMPTY"
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries)

    def chat_response(self, messages: List[Dict[str, Any]], **params: Any) -> ChatCompletion:
        payload = {**self.default_params, **params}
        return self.client.chat.completions.create(model=self.model, messages=messages, **payload)

    def chat_response_from_text(self, text: str, **params: Any) -> ChatCompletion:
        payload = {**self.default_params, **params}
        return self.client.completions.create(
            model=self.model,
            prompt=text,
            **payload,
        )

    def chat(self, messages: List[Dict[str, Any]], **params: Any) -> str:
        response = self.chat_response(messages, **params)
        return _extract_text_from_response(response)


@dataclass
class SGLangChatModel(OpenAIChatModel):
    server: Optional[SGLangServerProcess] = None

    def __post_init__(self) -> None:
        if self.server:
            self.server.start()
            if not self.base_url:
                self.base_url = self.server.config.base_url
        if not self.base_url:
            self.base_url = _resolve_sglang_base_url()
        if not self.base_url:
            raise ValueError("SGLANG_BASE_URL (or base_url) is required for SGLangChatModel.")
        if not self.api_key:
            self.api_key = os.getenv("SGLANG_API_KEY") or "EMPTY"
        self.require_api_key = False
        super().__post_init__()

@dataclass
class SGLangEngine(OpenAIChatModel):
    
    def __post_init__(self):
        self.llm = sgl.Engine(model_path=self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)
    
    def chat_response(self, messages: List[Dict[str, Any]], **params: Any) -> _SimpleChatCompletion:
        sampling_params = {
            "temperature": params.get("temperature"),
            "top_p": params.get("top_p"),
            "max_new_tokens": params.get("max_tokens"),
        }
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, tools=params.get("tools"))
        created_loop = False
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True
        
        try:
            outputs = self.llm.generate(
                prompt, sampling_params
            )
        finally:
            if created_loop:
                loop.close()
        text = outputs['text']
        tool_calls = []
        completion_tokens = outputs.get("completion_tokens")
        return _SimpleChatCompletion(text, completion_tokens=completion_tokens, tool_calls=tool_calls)


def build_model(backend: str, model: str, **kwargs: Any):
    backend_key = backend.lower().strip()
    if backend_key in {"openai", "openai_api"}:
        return OpenAIChatModel(model=model, **kwargs)
    if backend_key in {"sglang", "sglang_server", "local_api"}:
        return SGLangChatModel(model=model, **kwargs)
    if backend_key in {"sglang_offline", "offline"}:
        return SGLangEngine(model=model)
    raise ValueError(f"Unsupported backend: {backend}")
        
