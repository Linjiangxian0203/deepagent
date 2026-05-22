# tests/test_registry.py
import pytest
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


# Module-level registry for testing
test_registry = ToolRegistry()


@tool(registry=test_registry, description="Add two numbers", safety_level=SafetyLevel.READONLY)
async def add(a: int, b: int) -> str:
    return str(a + b)


@tool(registry=test_registry, name="greet", description="Greet someone", safety_level=SafetyLevel.READONLY)
async def say_hello(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"


def test_registry_has_registered_tools():
    names = test_registry.list_names()
    assert "add" in names
    assert "greet" in names


def test_registry_get_schemas():
    schemas = test_registry.get_schemas()
    assert len(schemas) == 2

    add_schema = next(s for s in schemas if s["function"]["name"] == "add")
    assert add_schema["function"]["description"] == "Add two numbers"
    assert add_schema["function"]["parameters"]["type"] == "object"
    assert "a" in add_schema["function"]["parameters"]["properties"]
    assert "b" in add_schema["function"]["parameters"]["properties"]
    assert add_schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert add_schema["function"]["parameters"]["required"] == ["a", "b"]


def test_registry_get_schemas_uses_custom_name():
    greet_schema = next(s for s in test_registry.get_schemas() if s["function"]["name"] == "greet")
    assert greet_schema["function"]["description"] == "Greet someone"
    assert "name" in greet_schema["function"]["parameters"]["properties"]
    assert "greeting" in greet_schema["function"]["parameters"]["properties"]
    assert "greeting" not in greet_schema["function"]["parameters"]["required"]


def test_registry_get_returns_tool():
    tool = test_registry.get("add")
    assert tool is not None
    assert tool.tool_name == "add"
    assert tool.tool_safety_level == SafetyLevel.READONLY


def test_registry_get_nonexistent():
    assert test_registry.get("nonexistent") is None


@pytest.mark.asyncio
async def test_tool_execution():
    tool = test_registry.get("add")
    result = await tool(a=3, b=4)
    assert result == "7"


@pytest.mark.asyncio
async def test_tool_execution_with_default():
    tool = test_registry.get("greet")
    result = await tool(name="World")
    assert result == "Hello, World!"


def test_tool_decorator_preserves_function():
    """@tool decorator preserves the original function callability."""
    import asyncio
    result = asyncio.run(add(a=10, b=20))
    assert result == "30"
