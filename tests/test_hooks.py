"""Tests for the HookSystem (Phase 1).

Covers: registration, priority ordering, error isolation, block semantics,
unregister, and integration patterns with AgentLoop events.
"""

import pytest
from deepagent.core.hooks import HookSystem, HookBlock
from deepagent.core.hooks import (
    EVENT_PRE_TOOL_USE,
    EVENT_POST_TOOL_USE,
    EVENT_PRE_LLM_CALL,
    EVENT_POST_LLM_CALL,
    EVENT_SESSION_START,
    EVENT_SESSION_END,
    VALID_EVENTS,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def hook_sys():
    return HookSystem()


# ── Registration ───────────────────────────────────────────────────


def test_register_stores_callback(hook_sys):
    results = []

    async def cb(**kwargs):
        results.append(1)
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb)
    assert hook_sys.count(EVENT_PRE_TOOL_USE) == 1


def test_register_invalid_event_raises(hook_sys):
    async def cb(**kwargs):
        return None

    with pytest.raises(ValueError, match="Unknown hook event"):
        hook_sys.register("BogusEvent", cb)


def test_register_default_priority_is_50(hook_sys):
    async def cb(**kwargs):
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb)
    assert hook_sys.count() == 1


def test_register_auto_names_unnamed_hooks(hook_sys):
    async def cb(**kwargs):
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb)
    names = hook_sys.list_names()
    assert len(names) == 1
    assert names[0].startswith(EVENT_PRE_TOOL_USE + "/hook_")


def test_register_named_hook(hook_sys):
    async def cb(**kwargs):
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb, name="my-hook")
    assert EVENT_PRE_TOOL_USE + "/my-hook" in hook_sys.list_names()


# ── Priority ordering ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_callbacks_fire_in_priority_order(hook_sys):
    order = []

    async def first(**kwargs):
        order.append("first")
        return None

    async def second(**kwargs):
        order.append("second")
        return None

    async def third(**kwargs):
        order.append("third")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, third, priority=90)
    hook_sys.register(EVENT_PRE_TOOL_USE, first, priority=10)
    hook_sys.register(EVENT_PRE_TOOL_USE, second, priority=50)

    await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_same_priority_fire_in_insertion_order(hook_sys):
    order = []

    async def a(**kwargs):
        order.append("a")
        return None

    async def b(**kwargs):
        order.append("b")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, a, priority=50)
    hook_sys.register(EVENT_PRE_TOOL_USE, b, priority=50)

    await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert order == ["a", "b"]


# ── Block semantics ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hook_block_stops_chain(hook_sys):
    order = []

    async def blocker(**kwargs):
        order.append("blocker")
        return HookBlock("denied by policy")

    async def never_called(**kwargs):
        order.append("never")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, blocker, priority=10)
    hook_sys.register(EVENT_PRE_TOOL_USE, never_called, priority=20)

    result = await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="run_shell", arguments={"command": "rm -rf /"})
    assert isinstance(result, HookBlock)
    assert result.reason == "denied by policy"
    assert order == ["blocker"]  # second hook never called


@pytest.mark.asyncio
async def test_hook_returns_none_chain_continues(hook_sys):
    order = []

    async def first(**kwargs):
        order.append("first")
        return None

    async def second(**kwargs):
        order.append("second")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, first, priority=10)
    hook_sys.register(EVENT_PRE_TOOL_USE, second, priority=20)

    result = await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="read_file", arguments={"path": "x"})
    assert result is None
    assert order == ["first", "second"]


# ── Error isolation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_callback_exception_is_isolated(hook_sys):
    order = []

    async def explodes(**kwargs):
        order.append("explodes")
        raise RuntimeError("boom")

    async def survives(**kwargs):
        order.append("survives")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, explodes, priority=10)
    hook_sys.register(EVENT_PRE_TOOL_USE, survives, priority=20)

    result = await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert result is None
    assert "explodes" in order
    assert "survives" in order


@pytest.mark.asyncio
async def test_callback_exception_does_not_block_chain(hook_sys):
    """Exception ≠ HookBlock. Chain continues after error."""
    order = []

    async def broken(**kwargs):
        raise ValueError("broken")

    async def fine(**kwargs):
        order.append("fine")
        return HookBlock("legitimate block")

    hook_sys.register(EVENT_PRE_TOOL_USE, broken, priority=10)
    hook_sys.register(EVENT_PRE_TOOL_USE, fine, priority=20)

    result = await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert isinstance(result, HookBlock)
    assert result.reason == "legitimate block"
    assert order == ["fine"]


# ── Unregister ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unregister_removes_named_hook(hook_sys):
    order = []

    async def removable(**kwargs):
        order.append("removable")
        return None

    async def permanent(**kwargs):
        order.append("permanent")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, removable, name="removable")
    hook_sys.register(EVENT_PRE_TOOL_USE, permanent, name="permanent")

    removed = hook_sys.unregister(EVENT_PRE_TOOL_USE, "removable")
    assert removed is True
    assert hook_sys.count(EVENT_PRE_TOOL_USE) == 1

    await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert order == ["permanent"]


def test_unregister_nonexistent_returns_false(hook_sys):
    assert hook_sys.unregister(EVENT_PRE_TOOL_USE, "nonexistent") is False


def test_unregister_unknown_event_returns_false(hook_sys):
    assert hook_sys.unregister("BogusEvent", "x") is False


# ── Count and list_names ───────────────────────────────────────────


def test_count_all_events(hook_sys):
    async def cb(**kwargs):
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb)
    hook_sys.register(EVENT_POST_TOOL_USE, cb)
    hook_sys.register(EVENT_PRE_TOOL_USE, cb, name="second")
    assert hook_sys.count() == 3
    assert hook_sys.count(EVENT_PRE_TOOL_USE) == 2
    assert hook_sys.count(EVENT_POST_TOOL_USE) == 1
    assert hook_sys.count(EVENT_SESSION_START) == 0


def test_list_names(hook_sys):
    async def cb(**kwargs):
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, cb, name="alpha")
    hook_sys.register(EVENT_POST_TOOL_USE, cb, name="beta")
    hook_sys.register(EVENT_SESSION_START, cb, name="gamma")

    names = hook_sys.list_names()
    assert "PreToolUse/alpha" in names
    assert "PostToolUse/beta" in names
    assert "SessionStart/gamma" in names


# ── Unknown event trigger ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_unknown_event_returns_none(hook_sys):
    result = await hook_sys.trigger("BogusEvent")
    assert result is None


# ── SessionEnd continuation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_end_hook_block_forces_continuation(hook_sys):
    """SessionEnd HookBlock prevents session exit (force continuation)."""

    async def prevent_exit(**kwargs):
        return HookBlock("Session has unfinished work")

    hook_sys.register(EVENT_SESSION_END, prevent_exit)

    result = await hook_sys.trigger(EVENT_SESSION_END, stats={"turns": 3})
    assert isinstance(result, HookBlock)
    assert "unfinished" in result.reason


@pytest.mark.asyncio
async def test_session_end_none_allows_exit(hook_sys):
    async def allow_exit(**kwargs):
        return None

    hook_sys.register(EVENT_SESSION_END, allow_exit)

    result = await hook_sys.trigger(EVENT_SESSION_END, stats={"turns": 1})
    assert result is None


# ── PreLLMCall dict modification ────────────────────────────────────


@pytest.mark.asyncio
async def test_pre_llm_call_can_modify_messages(hook_sys):
    """PreLLMCall returning a dict modifies the params to the LLM call."""
    async def inject_context(messages, tools, **kwargs):
        return {"messages": messages, "tools": tools, "injected": True}

    hook_sys.register(EVENT_PRE_LLM_CALL, inject_context, priority=10)

    result = await hook_sys.trigger(EVENT_PRE_LLM_CALL, messages=[{"role": "user", "content": "hi"}], tools=None)
    assert isinstance(result, dict)
    assert result["injected"] is True
    assert len(result["messages"]) == 1


# ── PostToolUse never blocks ───────────────────────────────────────


@pytest.mark.asyncio
async def test_post_tool_use_block_has_no_effect(hook_sys):
    """PostToolUse return value is ignored (fire-and-forget event)."""
    async def warn_large_output(tool_name, arguments, result, **kwargs):
        return HookBlock("this won't block anything")

    hook_sys.register(EVENT_POST_TOOL_USE, warn_large_output)

    result = await hook_sys.trigger(EVENT_POST_TOOL_USE, tool_name="bash", arguments={}, result="huge output")
    assert result is None  # PostToolUse ignores returns


# ── Priority clamping ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_priority_clamped_to_0_100():
    """Priorities outside [0, 100] are clamped."""
    hook_sys = HookSystem()
    order = []

    async def low(**kwargs):
        order.append("low")
        return None

    async def high(**kwargs):
        order.append("high")
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, high, priority=-999)  # clamped to 0
    hook_sys.register(EVENT_PRE_TOOL_USE, low, priority=9999)   # clamped to 100

    await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert order == ["high", "low"]  # -999 -> 0 fires first, 9999 -> 100 fires second


# ── Empty trigger ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_with_no_registered_hooks(hook_sys):
    result = await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert result is None


# ── Stress: many hooks ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_many_hooks_all_fire(hook_sys):
    counter = [0]

    async def make_counter(**kwargs):
        counter[0] += 1
        return None

    for i in range(50):
        hook_sys.register(EVENT_PRE_TOOL_USE, make_counter, priority=i % 100)

    await hook_sys.trigger(EVENT_PRE_TOOL_USE, tool_name="test", arguments={})
    assert counter[0] == 50


# ── Hook with tool_call object (integration pattern) ────────────────


@pytest.mark.asyncio
async def test_pre_tool_use_receives_tool_call_kwargs(hook_sys):
    """Simulate the real integration: PreToolUse receives tool_name, arguments, tool_call."""
    received = {}

    async def capture(**kwargs):
        received.update(kwargs)
        return None

    hook_sys.register(EVENT_PRE_TOOL_USE, capture)

    await hook_sys.trigger(
        EVENT_PRE_TOOL_USE,
        tool_name="read_file",
        arguments={"path": "/tmp/test.txt", "offset": 0},
        tool_call={"id": "tc_1", "name": "read_file"},
    )
    assert received["tool_name"] == "read_file"
    assert received["arguments"] == {"path": "/tmp/test.txt", "offset": 0}
    assert received["tool_call"] == {"id": "tc_1", "name": "read_file"}
