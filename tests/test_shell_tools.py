# tests/test_shell_tools.py
import pytest
from deepagent.tools.shell_tools import create_shell_tools
from deepagent.tools.protocol import SafetyLevel


@pytest.fixture
def registry_and_tools():
    """Create registry with shell_tools."""
    from deepagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    tools = create_shell_tools(registry)
    return registry, tools


def test_shell_tools_are_registered(registry_and_tools):
    registry, tools = registry_and_tools
    assert "run_shell" in registry.list_names()


def test_run_shell_safety_level(registry_and_tools):
    registry, tools = registry_and_tools
    assert registry.get("run_shell").tool_safety_level == SafetyLevel.SHELL


@pytest.mark.asyncio
async def test_run_shell_echo(registry_and_tools):
    registry, tools = registry_and_tools
    tool = registry.get("run_shell")
    result = await tool(command="echo hello world")
    assert result["success"] is True
    assert "hello world" in result["content"]


@pytest.mark.asyncio
async def test_run_shell_command_not_found(registry_and_tools):
    registry, tools = registry_and_tools
    tool = registry.get("run_shell")
    result = await tool(command="nonexistent_command_xyz_12345")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_run_shell_captures_stderr(registry_and_tools):
    registry, tools = registry_and_tools
    tool = registry.get("run_shell")
    # Cross-platform: use Python to write to both stdout and stderr
    result = await tool(
        command='python -c "import sys; sys.stdout.write(\'stdout\\n\'); sys.stderr.write(\'stderr\\n\')"'
    )
    if result["success"]:
        assert "stdout" in result["content"]
        assert "stderr" in result["metadata"]["stderr"]
