"""Tests for the multi-turn AgentLoop (P2).

The FakeLLMClient below supports multi-turn responses: it consumes entries from
an iterable, each entry being a list of events to yield for one LLM call.
"""

import pytest
from deepagent.config import Config
from deepagent.core.loop import AgentLoop, ConfirmationHandler
from deepagent.core.context import ContextManager
from deepagent.core.events import (
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolCall,
    ToolCallStartEvent,
    ToolResultEvent,
    ToolResult,
    DoneEvent,
    InterruptedEvent,
    UsageEvent,
)
from deepagent.tools.registry import ToolRegistry, tool
from deepagent.tools.protocol import SafetyLevel


def make_config(**kwargs):
    env = {"DEEPSEEK_API_KEY": "sk-test"}
    env.update(kwargs)
    return Config(_env=env)


class FakeLLMClient:
    """Fake LLM client that returns preset events per call.

    Pass `responses` as a list of lists: each inner list is the events
    for one call to stream_chat(). After all responses are consumed,
    returns an empty text response (no tool calls).
    """

    def __init__(self, responses: list):
        self._responses = responses
        self._call_count = 0
        self.last_messages = None
        self.last_tools = None
        self.all_messages = []
        self.all_tools = []

    async def stream_chat(self, messages, tools=None):
        self.last_messages = messages
        self.last_tools = tools
        self.all_messages.append(messages)
        self.all_tools.append(tools)

        if self._call_count < len(self._responses):
            events = self._responses[self._call_count]
            self._call_count += 1
            for event in events:
                yield event
        else:
            # After preset responses exhausted, return simple text
            yield TextDelta(text="Done.")


def make_registry():
    reg = ToolRegistry()

    @tool(
        registry=reg,
        description="Echo back the message",
        safety_level=SafetyLevel.READONLY,
    )
    async def echo(message: str) -> dict:
        return {"success": True, "content": f"Echo: {message}", "error": None}

    @tool(
        registry=reg,
        name="shell_cmd",
        description="Run a shell command",
        safety_level=SafetyLevel.SHELL,
    )
    async def run_shell(command: str) -> dict:
        return {"success": True, "content": f"Ran: {command}", "error": None}

    return reg


# ── Basic single-turn (no tools) ───────────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_emits_text_and_done():
    """Plain text reply: TextDelta + DoneEvent, no tool execution."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [TextDelta(text="Hello"), TextDelta(text=" World")],
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


# ── Single tool call → final response ─────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_and_yields_result():
    """Tool call → execute → LLM gets feedback → final text reply."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        # Turn 1: LLM decides to call a tool
        [
            TextDelta(text="Let me echo."),
            ToolCallEvent(tool_calls=[
                ToolCall(id="call_1", name="echo", arguments={"message": "hi"})
            ]),
        ],
        # Turn 2: LLM sees tool result, gives final answer
        [TextDelta(text="I echoed: hi")],
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

    # Should have the final text "I echoed: hi"
    final_text = "".join(
        e.text for e in events if isinstance(e, TextDelta)
    )
    assert "I echoed: hi" in final_text
    assert any(isinstance(e, DoneEvent) for e in events)


# ── Tool not found ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_tool_not_found():
    """Nonexistent tool returns error result, LLM sees it."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(tool_calls=[
            ToolCall(id="call_1", name="nonexistent", arguments={})
        ])],
        [TextDelta(text="That tool doesn't exist.")],
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


# ── Tool schemas are passed to LLM ─────────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_sends_tool_schemas_to_llm():
    """Verify tool schemas are passed to LLMClient each turn."""
    cfg = make_config()
    fake_llm = FakeLLMClient([[TextDelta(text="ok")]])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    async for _ in loop.run("test"):
        pass

    assert fake_llm.last_tools is not None
    assert len(fake_llm.last_tools) == 2  # echo + shell_cmd
    names = [t["function"]["name"] for t in fake_llm.last_tools]
    assert "echo" in names
    assert "shell_cmd" in names


# ── Confirmation handler ──────────────────────────────────────────


class AlwaysConfirm(ConfirmationHandler):
    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return True


class AlwaysDeny(ConfirmationHandler):
    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        return False


@pytest.mark.asyncio
async def test_confirmation_handler_approved_executes():
    """Shell tool confirmed → executes normally."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(tool_calls=[
            ToolCall(id="c1", name="shell_cmd", arguments={"command": "ls"})
        ])],
        [TextDelta(text="Ran it.")],
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
    """Shell tool denied → returns ExecutionDenied, LLM sees it."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(tool_calls=[
            ToolCall(id="c1", name="shell_cmd", arguments={"command": "rm -rf /"})
        ])],
        [TextDelta(text="Denied. Won't do that.")],
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


# ── Multi-turn: LLM calls tools twice ─────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_two_rounds_of_tools():
    """LLM calls tool → gets result → calls another tool → final reply."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        # Turn 1: call echo first time
        [
            ToolCallEvent(tool_calls=[
                ToolCall(id="c1", name="echo", arguments={"message": "first"})
            ]),
        ],
        # Turn 2: call echo second time
        [
            ToolCallEvent(tool_calls=[
                ToolCall(id="c2", name="echo", arguments={"message": "second"})
            ]),
        ],
        # Turn 3: final text reply
        [TextDelta(text="All done.")],
    ])
    reg = make_registry()
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("multi"):
        events.append(event)

    # Two tool calls executed
    start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(start_events) == 2
    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(r.result.success for r in result_events)

    # Context should have: user, assistant(tool), tool, assistant(tool), tool, assistant
    assert fake_llm._call_count == 3


# ── ContextManager usage ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_loop_uses_context_manager():
    """Verify ContextManager accumulates messages across turns."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            TextDelta(text="Let me check."),
            ToolCallEvent(tool_calls=[
                ToolCall(id="c1", name="echo", arguments={"message": "test"})
            ]),
        ],
        [TextDelta(text="Got: Echo: test")],
    ])
    reg = make_registry()
    ctx = ContextManager()
    loop = AgentLoop(cfg, fake_llm, reg, context=ctx)

    async for _ in loop.run("query"):
        pass

    # Context should have: user, assistant(with tool_calls), tool, assistant(final)
    assert ctx.message_count == 4
    msgs = ctx.get_messages(with_system=False)
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]


# ── Reasoning content echo-back ───────────────────────────────────


@pytest.mark.asyncio
async def test_reasoning_content_is_echoed():
    """When thinking mode produces reasoning, it's saved in context messages."""
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(
                tool_calls=[
                    ToolCall(id="c1", name="echo", arguments={"message": "hi"})
                ],
                reasoning_content="Let me think about echoing...",
            ),
        ],
        [TextDelta(text="echoed")],
    ])
    reg = make_registry()
    ctx = ContextManager()
    loop = AgentLoop(cfg, fake_llm, reg, context=ctx)

    async for _ in loop.run("echo"):
        pass

    # Check that the assistant message with tool_calls has reasoning_content
    msgs = ctx.get_messages(with_system=False)
    assistant_with_tools = msgs[1]
    assert assistant_with_tools["role"] == "assistant"
    assert "reasoning_content" in assistant_with_tools
    assert assistant_with_tools["reasoning_content"] == "Let me think about echoing..."


# ── Tool execution exception ──────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_execution_exception():
    """Tool that raises an exception → error ToolResult."""
    reg = ToolRegistry()

    @tool(registry=reg, description="Fails always", safety_level=SafetyLevel.READONLY)
    async def bad_tool() -> dict:
        raise RuntimeError("oops")

    cfg = make_config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(tool_calls=[
            ToolCall(id="c1", name="bad_tool", arguments={})
        ])],
        [TextDelta(text="It failed.")],
    ])
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("test"):
        events.append(event)

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert result_events[0].result.success is False
    assert "oops" in result_events[0].result.error


# ── UsageEvent tracking ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_usage_event_updates_context():
    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            TextDelta(text="hi"),
            UsageEvent(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        ],
    ])
    reg = make_registry()
    ctx = ContextManager()
    loop = AgentLoop(cfg, fake_llm, reg, context=ctx)

    async for _ in loop.run("test"):
        pass

    assert ctx.prompt_tokens == 50
    assert ctx.completion_tokens == 10


# ── InterruptedEvent ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_interrupted_mid_tool_execution():
    """Interrupt during tool execution phase yields InterruptedEvent."""
    reg = ToolRegistry()

    @tool(registry=reg, description="Slow tool", safety_level=SafetyLevel.READONLY)
    async def slow_tool() -> dict:
        return {"success": True, "content": "done"}

    cfg = make_config()
    fake_llm = FakeLLMClient([
        [ToolCallEvent(tool_calls=[
            ToolCall(id="c1", name="slow_tool", arguments={})
        ])],
        [TextDelta(text="done")],
    ])
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    interrupted = False
    async for event in loop.run("test"):
        if isinstance(event, ToolCallStartEvent) and not interrupted:
            loop.interrupt()
            interrupted = True
        events.append(event)

    assert any(isinstance(e, InterruptedEvent) for e in events)


# ── Edge: No tools in registry ────────────────────────────────────


@pytest.mark.asyncio
async def test_loop_with_empty_registry():
    """When no tools are registered, LLM is called without tools."""
    cfg = make_config()
    fake_llm = FakeLLMClient([[TextDelta(text="No tools here.")]])
    reg = ToolRegistry()  # empty
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("test"):
        events.append(event)

    assert fake_llm.last_tools is None
    assert any(isinstance(e, DoneEvent) for e in events)


# ── Parallel execution (P4) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_readonly_tools_execute_concurrently():
    """Multiple readonly tools run in parallel (all Start before Result)."""
    reg = ToolRegistry()

    @tool(registry=reg, description="Tool A", safety_level=SafetyLevel.READONLY)
    async def tool_a(x: str) -> dict:
        return {"success": True, "content": f"A:{x}"}

    @tool(registry=reg, description="Tool B", safety_level=SafetyLevel.READONLY)
    async def tool_b(y: str) -> dict:
        return {"success": True, "content": f"B:{y}"}

    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(tool_calls=[
                ToolCall(id="c1", name="tool_a", arguments={"x": "first"}),
                ToolCall(id="c2", name="tool_b", arguments={"y": "second"}),
            ]),
        ],
        [TextDelta(text="done")],
    ])
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("test"):
        events.append(event)

    # Both Start events should appear before both Result events
    start_indices = [
        i for i, e in enumerate(events) if isinstance(e, ToolCallStartEvent)
    ]
    result_indices = [
        i for i, e in enumerate(events) if isinstance(e, ToolResultEvent)
    ]
    assert len(start_indices) == 2
    assert len(result_indices) == 2
    # All starts before all results (parallel execution pattern)
    assert start_indices[-1] < result_indices[0]

    # Both results should be successful
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert all(r.result.success for r in results)
    contents = {r.result.content for r in results}
    assert "A:first" in contents
    assert "B:second" in contents


@pytest.mark.asyncio
async def test_mixed_readonly_and_shell_execute_in_groups():
    """Readonly tools execute in parallel first, then shell tools sequentially."""
    reg = ToolRegistry()

    @tool(registry=reg, description="Read", safety_level=SafetyLevel.READONLY)
    async def read(path: str) -> dict:
        return {"success": True, "content": f"read:{path}"}

    @tool(registry=reg, description="Shell", safety_level=SafetyLevel.SHELL)
    async def cmd(command: str) -> dict:
        return {"success": True, "content": f"ran:{command}"}

    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(tool_calls=[
                ToolCall(id="c1", name="read", arguments={"path": "/a"}),
                ToolCall(id="c2", name="cmd", arguments={"command": "ls"}),
                ToolCall(id="c3", name="read", arguments={"path": "/b"}),
            ]),
        ],
        [TextDelta(text="done")],
    ])
    # AlwaysConfirm for shell tool
    loop = AgentLoop(cfg, fake_llm, reg, confirm_handler=AlwaysConfirm())

    events = []
    async for event in loop.run("test"):
        events.append(event)

    starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
    # 3 tool calls: 2 readonly + 1 shell = 3 starts
    assert len(starts) == 3

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 3
    assert all(r.result.success for r in results)


@pytest.mark.asyncio
async def test_parallel_readonly_one_fails_others_succeed():
    """If one readonly tool fails in parallel, others still complete."""
    reg = ToolRegistry()

    @tool(registry=reg, description="Good", safety_level=SafetyLevel.READONLY)
    async def good() -> dict:
        return {"success": True, "content": "ok"}

    @tool(registry=reg, description="Bad", safety_level=SafetyLevel.READONLY)
    async def bad() -> dict:
        raise RuntimeError("boom")

    cfg = make_config()
    fake_llm = FakeLLMClient([
        [
            ToolCallEvent(tool_calls=[
                ToolCall(id="c1", name="good", arguments={}),
                ToolCall(id="c2", name="bad", arguments={}),
            ]),
        ],
        [TextDelta(text="done")],
    ])
    loop = AgentLoop(cfg, fake_llm, reg)

    events = []
    async for event in loop.run("test"):
        events.append(event)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 2
    successes = [r for r in results if r.result.success]
    failures = [r for r in results if not r.result.success]
    assert len(successes) == 1
    assert len(failures) == 1
    assert "boom" in failures[0].result.error
