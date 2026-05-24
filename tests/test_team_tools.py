"""Tests for team tools: spawn_teammate, send_message, check_inbox."""
import json
import tempfile
from pathlib import Path

import pytest
from deepagent.core.message_bus import MessageBus
from deepagent.tools.registry import ToolRegistry, tool
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.team_tools import (
    create_team_tools,
    _build_teammate_tool_schemas,
    TEAMMATE_TOOL_NAMES,
)


@pytest.fixture
def bus():
    d = tempfile.TemporaryDirectory()
    b = MessageBus(d.name)
    yield b
    d.cleanup()


@pytest.fixture
def tool_reg():
    reg = ToolRegistry()

    @tool(reg, name="read_file", description="Read a file", safety_level=SafetyLevel.READONLY)
    async def read_file(path: str) -> dict:
        return {"success": True, "content": f"Content of {path}"}

    @tool(reg, name="run_shell", description="Run shell command", safety_level=SafetyLevel.SHELL)
    async def run_shell(command: str) -> dict:
        return {"success": True, "content": f"Ran: {command}"}

    @tool(reg, name="write_file", description="Write a file", safety_level=SafetyLevel.WRITE)
    async def write_file(path: str, content: str) -> dict:
        return {"success": True, "content": f"Wrote {path}"}

    @tool(reg, name="send_message", description="Send to another agent", safety_level=SafetyLevel.READONLY)
    async def send_message(to: str, content: str) -> dict:
        return {"success": True, "content": f"Sent to {to}"}

    return reg


class FakeLLMClient:
    """Fake LLM client that returns preset text + optional tool calls."""
    def __init__(self, response_text="Done.", tool_calls=None):
        self.response_text = response_text
        self.tool_calls = tool_calls
        self.call_count = 0
        self.all_messages = []

    async def stream_chat(self, messages, tools=None):
        from deepagent.core.events import TextDelta, ToolCallEvent
        self.call_count += 1
        self.all_messages.append(messages)
        yield TextDelta(text=self.response_text)
        if self.tool_calls:
            yield ToolCallEvent(tool_calls=self.tool_calls)


def test_tool_registration(bus, tool_reg):
    """All three team tools should be registered."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg, cwd="/tmp")

    assert "spawn_teammate" in reg.list_names()
    assert "send_message" in reg.list_names()
    assert "check_inbox" in reg.list_names()

    spawn_tool = reg.get("spawn_teammate")
    assert spawn_tool.tool_safety_level == SafetyLevel.WRITE

    send_tool = reg.get("send_message")
    assert send_tool.tool_safety_level == SafetyLevel.READONLY


@pytest.mark.asyncio
async def test_send_message_tool(bus, tool_reg):
    """Send a message via the tool."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg)

    tool_fn = reg.get("send_message")
    result = await tool_fn(to="worker", content="Hello worker!")
    assert result["success"] is True
    assert "worker" in result["content"]

    msgs = bus.read_inbox("worker")
    assert len(msgs) == 1
    assert msgs[0]["from"] == "lead"
    assert msgs[0]["content"] == "Hello worker!"


@pytest.mark.asyncio
async def test_send_message_requires_fields(bus, tool_reg):
    """send_message requires to and content."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg)

    tool_fn = reg.get("send_message")
    r1 = await tool_fn(to="", content="msg")
    assert r1["success"] is False

    r2 = await tool_fn(to="worker", content="")
    assert r2["success"] is False


@pytest.mark.asyncio
async def test_check_inbox_empty(bus, tool_reg):
    """Empty inbox returns placeholder."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg)

    tool_fn = reg.get("check_inbox")
    result = await tool_fn()
    assert result["success"] is True
    assert "inbox empty" in result["content"]


@pytest.mark.asyncio
async def test_check_inbox_with_messages(bus, tool_reg):
    """Inbox with messages shows them."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg)

    bus.send("worker-1", "lead", "I finished the task!", "result")
    bus.send("worker-2", "lead", "Still working...", "status")

    tool_fn = reg.get("check_inbox")
    result = await tool_fn()
    assert result["success"] is True
    assert "worker-1" in result["content"]
    assert "worker-2" in result["content"]
    assert "result" in result["content"]
    assert "status" in result["content"]

    # check_inbox consumes messages
    result2 = await tool_fn()
    assert "inbox empty" in result2["content"]


@pytest.mark.asyncio
async def test_spawn_teammate(bus, tool_reg):
    """Spawn a teammate — should not block, return immediately."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg, cwd="/tmp")

    tool_fn = reg.get("spawn_teammate")
    result = await tool_fn(
        name="worker",
        role="tester",
        prompt="Run the test suite and report back.",
    )
    assert result["success"] is True
    assert "worker" in result["content"]

    # Give the teammate asyncio task a moment to run
    import asyncio
    await asyncio.sleep(0.1)

    # Teammate should have sent result to Lead
    lead_inbox = bus.read_inbox("lead")
    assert len(lead_inbox) >= 1
    assert any(m["from"] == "worker" for m in lead_inbox)


@pytest.mark.asyncio
async def test_spawn_teammate_requires_fields(bus, tool_reg):
    """spawn_teammate requires name and prompt."""
    fake_llm = FakeLLMClient()
    reg = ToolRegistry()
    create_team_tools(reg, bus, fake_llm, tool_reg)

    tool_fn = reg.get("spawn_teammate")
    r1 = await tool_fn(name="", role="tester", prompt="do stuff")
    assert r1["success"] is False

    r2 = await tool_fn(name="worker", role="tester", prompt="")
    assert r2["success"] is False


def test_build_teammate_tool_schemas(tool_reg):
    """Should build schemas only for tools in TEAMMATE_TOOL_NAMES."""
    schemas = _build_teammate_tool_schemas(tool_reg)
    assert schemas is not None
    names_in_schemas = {s["function"]["name"] for s in schemas}
    # Only tools in TEAMMATE_TOOL_NAMES should appear
    assert names_in_schemas.issubset(TEAMMATE_TOOL_NAMES)


def test_teammate_tool_names_subset():
    """TEAMMATE_TOOL_NAMES should not include spawn_teammate or delegate."""
    assert "spawn_teammate" not in TEAMMATE_TOOL_NAMES
    assert "delegate" not in TEAMMATE_TOOL_NAMES
    # Should include basic tools
    assert "read_file" in TEAMMATE_TOOL_NAMES
    assert "run_shell" in TEAMMATE_TOOL_NAMES
    assert "send_message" in TEAMMATE_TOOL_NAMES
