import pytest
from deepagent.config import Config
from deepagent.core.loop import AgentLoop, ConfirmationHandler
from deepagent.core.events import (
    TextDelta,
    ToolCallEvent,
    ToolCall,
    ToolCallStartEvent,
    ToolResultEvent,
    ToolResult,
    DoneEvent,
)
from deepagent.tools.registry import ToolRegistry, tool
from deepagent.tools.protocol import SafetyLevel


def make_config(**kwargs):
    env = {"DEEPSEEK_API_KEY": "sk-test"}
    env.update(kwargs)
    return Config(_env=env)


class FakeLLMClient:
    """Mock LLMClient with controllable responses."""

    def __init__(self, events: list):
        self._events = events
        self.last_messages = None
        self.last_tools = None

    async def stream_chat(self, messages, tools=None):
        self.last_messages = messages
        self.last_tools = tools
        for event in self._events:
            yield event


def make_registry():
    reg = ToolRegistry()

    @tool(
        registry=reg,
        description="Echo back the message",
        safety_level=SafetyLevel.READONLY,
    )
    async def echo(message: str) -> dict:
        return {
            "success": True,
            "content": f"Echo: {message}",
            "error": None,
            "metadata": None,
        }

    return reg


@pytest.mark.asyncio
async def test_agent_loop_emits_text_and_done():
    """Text-only reply: yields TextDelta + DoneEvent, no tool execution."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        TextDelta(text="Hello"),
        TextDelta(text=" World"),
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("Say hello"):
        events.append(event)

    texts = [e for e in events if isinstance(e, TextDelta)]
    assert "".join(t.text for t in texts) == "Hello World"
    assert any(isinstance(e, DoneEvent) for e in events)
    assert not any(isinstance(e, ToolCallStartEvent) for e in events)
    assert not any(isinstance(e, ToolResultEvent) for e in events)


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_and_yields_result():
    """Tool call: LLM returns ToolCallEvent -> execute tool -> ToolResultEvent -> DoneEvent."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        TextDelta(text="Let me echo that for you."),
        ToolCallEvent(tool_calls=[
            ToolCall(id="call_1", name="echo", arguments={"message": "hi"})
        ]),
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("Echo hi"):
        events.append(event)

    start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].tool_call.name == "echo"

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is True
    assert "Echo: hi" in result_events[0].result.content

    assert any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.asyncio
async def test_agent_loop_tool_not_found():
    """Unknown tool returns error result."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        ToolCallEvent(tool_calls=[
            ToolCall(id="call_1", name="nonexistent", arguments={})
        ]),
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("Do something"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is False
    assert "not found" in result_events[0].result.error.lower()


@pytest.mark.asyncio
async def test_agent_loop_sends_tool_schemas_to_llm():
    """Verify AgentLoop passes registered tool schemas to LLMClient."""
    cfg = make_config()
    fake_llm = FakeLLMClient([TextDelta(text="ok")])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    async for _ in loop.run("test"):
        pass

    assert fake_llm.last_tools is not None
    assert len(fake_llm.last_tools) == 1
    assert fake_llm.last_tools[0]["function"]["name"] == "echo"


class AlwaysConfirm(ConfirmationHandler):
    """Test handler: always approves."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return True


class AlwaysDeny(ConfirmationHandler):
    """Test handler: always denies."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return False


@pytest.mark.asyncio
async def test_confirmation_handler_approved_executes():
    """Approved confirmation executes tool normally."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        ToolCallEvent(tool_calls=[
            ToolCall(id="call_1", name="echo", arguments={"message": "test"})
        ]),
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg, confirm_handler=AlwaysConfirm())

    events = []
    async for event in loop.run("test"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result_events[0].result.success is True


@pytest.mark.asyncio
async def test_confirmation_handler_denied_skips():
    """Denied confirmation skips execution, returns denied result."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        ToolCallEvent(tool_calls=[
            ToolCall(id="call_1", name="echo", arguments={"message": "test"})
        ]),
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg, confirm_handler=AlwaysDeny())

    events = []
    async for event in loop.run("test"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].result.success is False
    assert "denied" in result_events[0].result.error.lower()
