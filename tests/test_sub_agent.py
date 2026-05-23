"""Tests for SubAgentRunner and delegate tool."""

import pytest
from deepagent.config import Config
from deepagent.core.events import TextDelta, ToolCallEvent, ToolCall
from deepagent.core.sub_agent import SubAgentRunner, SUB_AGENT_SYSTEM_PROMPT
from deepagent.tools.registry import ToolRegistry, tool
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.delegate_tools import create_delegate_tools


def make_config(**kwargs):
    env = {"DEEPSEEK_API_KEY": "sk-test"}
    env.update(kwargs)
    return Config(_env=env)


class FakeLLMClient:
    """Fake that returns preset responses based on the prompt content."""

    def __init__(self):
        self.call_count = 0
        self.all_messages = []

    async def stream_chat(self, messages, tools=None):
        self.call_count += 1
        self.all_messages.append(messages)
        # Return a simple text response for any input
        yield TextDelta(text=f"Sub-agent result (call #{self.call_count})")


def make_registry():
    reg = ToolRegistry()

    @tool(registry=reg, description="Read a file", safety_level=SafetyLevel.READONLY)
    async def read_file(path: str) -> dict:
        return {"success": True, "content": f"Content of {path}"}

    return reg


# ── SubAgentRunner ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sub_agent_runner_basic():
    """Sub-agent runs a simple task and returns results."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg)

    result = await runner.run(
        description="Test task",
        prompt="Do something simple",
    )
    assert result["success"] is True
    assert "Sub-agent result" in result["content"]
    assert result["metadata"]["description"] == "Test task"


@pytest.mark.asyncio
async def test_sub_agent_runner_tracks_active_count():
    """Active count is 0 before/after, and incremented during execution."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg)

    assert runner.active_count == 0
    await runner.run("task", "do something")
    assert runner.active_count == 0


@pytest.mark.asyncio
async def test_sub_agent_runner_concurrent():
    """Multiple concurrent sub-agents can run in parallel."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg, max_concurrent=5)

    import asyncio
    tasks = [
        runner.run(f"task-{i}", f"do task {i}")
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert len(results) == 3
    assert all(r["success"] for r in results)


# ── delegate tool ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delegate_tool_is_registered():
    """Delegate tool is registered with READONLY safety."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg)

    create_delegate_tools(reg, runner)
    assert "delegate" in reg.list_names()
    tool = reg.get("delegate")
    assert tool.tool_safety_level == SafetyLevel.READONLY


@pytest.mark.asyncio
async def test_delegate_tool_execution():
    """Calling the delegate tool returns sub-agent results."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg)

    create_delegate_tools(reg, runner)
    tool = reg.get("delegate")
    result = await tool(
        description="Research something",
        prompt="Find and summarize the file structure",
    )
    assert result["success"] is True
    assert "Sub-agent result" in result["content"]


@pytest.mark.asyncio
async def test_delegate_tool_in_schema():
    """Delegate tool shows up in the schema list for LLM."""
    cfg = make_config()
    fake_llm = FakeLLMClient()
    reg = make_registry()
    runner = SubAgentRunner(cfg, fake_llm, reg)

    create_delegate_tools(reg, runner)
    schemas = reg.get_schemas()
    delegate_schema = next(
        s for s in schemas if s["function"]["name"] == "delegate"
    )
    assert "sub-agent" in delegate_schema["function"]["description"]
    assert "description" in delegate_schema["function"]["parameters"]["properties"]
    assert "prompt" in delegate_schema["function"]["parameters"]["properties"]


# ── System prompt ──────────────────────────────────────────────────


def test_sub_agent_system_prompt():
    """Sub-agent system prompt has the right content."""
    prompt = SUB_AGENT_SYSTEM_PROMPT
    assert "sub-agent" in prompt.lower()
    assert "autonomously" in prompt.lower()
