# src/deepagent/tools/registry.py
from collections.abc import Callable
import functools
import inspect
import json
from typing import Any, get_type_hints

from deepagent.tools.protocol import SafetyLevel


class ToolRegistry:
    """Registry of tools. Manages ToolProtocol instances and generates JSON Schema lists for the LLM API."""

    def __init__(self):
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        self._tools[tool.tool_name] = tool

    def get(self, name: str):
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict]:
        """Generate an OpenAI-compatible tools parameter list."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.tool_name,
                    "description": tool.tool_description,
                    "parameters": tool.tool_parameters,
                },
            })
        return schemas


def _type_to_json_schema(t: type) -> str:
    """Python type -> JSON Schema type string."""
    mapping = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
    origin = getattr(t, "__origin__", None)
    if origin is not None:
        return "array" if origin is list else "string"
    return mapping.get(t, "string")


def _func_to_json_schema(func: Callable) -> dict:
    """Derive JSON Schema from function signature. Supports str/int/float/bool basic types."""
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
    sig = inspect.signature(func)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = hints.get(param_name, str)
        json_type = _type_to_json_schema(annotation)
        prop = {"type": json_type}

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(
    registry: ToolRegistry | None = None,
    *,
    name: str | None = None,
    description: str = "",
    safety_level: SafetyLevel = SafetyLevel.READONLY,
):
    """Decorator: mark an async function as a tool, auto-derive JSON Schema, and optionally register it."""
    def decorator(func):
        schema = _func_to_json_schema(func)

        @functools.wraps(func)
        async def wrapper(**kwargs):
            return await func(**kwargs)

        wrapper.tool_name = name or func.__name__
        wrapper.tool_description = description
        wrapper.tool_parameters = schema
        wrapper.tool_safety_level = safety_level

        if registry is not None:
            registry.register(wrapper)

        return wrapper

    return decorator
