# src/deepagent/tools/protocol.py
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SafetyLevel(str, Enum):
    READONLY = "readonly"
    WRITE = "write"
    SHELL = "shell"


@runtime_checkable
class ToolProtocol(Protocol):
    """工具必须实现的协议。@tool 装饰器自动生成符合此协议的对象。"""
    tool_name: str
    tool_description: str
    tool_parameters: dict  # JSON Schema
    tool_safety_level: SafetyLevel

    async def __call__(self, **kwargs: Any) -> Any: ...
