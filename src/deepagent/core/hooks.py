"""Hook system — extensibility points that don't invade the agent loop.

6 event points cover the complete agent cycle. Extensions register callbacks
with optional priority ordering. The loop calls trigger(); all logic lives
in hook callbacks — the loop body stays clean.

Reference: learn-claude-code s04_hooks. Extended with async, priority
ordering, error isolation, and typed return values.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# ── Event names ────────────────────────────────────────────────────

EVENT_PRE_TOOL_USE = "PreToolUse"
EVENT_POST_TOOL_USE = "PostToolUse"
EVENT_PRE_LLM_CALL = "PreLLMCall"
EVENT_POST_LLM_CALL = "PostLLMCall"
EVENT_SESSION_START = "SessionStart"
EVENT_SESSION_END = "SessionEnd"

VALID_EVENTS = frozenset({
    EVENT_PRE_TOOL_USE,
    EVENT_POST_TOOL_USE,
    EVENT_PRE_LLM_CALL,
    EVENT_POST_LLM_CALL,
    EVENT_SESSION_START,
    EVENT_SESSION_END,
})

# ── Return types ───────────────────────────────────────────────────


@dataclass
class HookBlock:
    """Returned by a hook to deny/prevent an action.

    For PreToolUse: blocks tool execution (the reason is fed to the LLM as a tool result).
    For PreLLMCall: blocks the LLM call.
    For SessionEnd: prevents session exit (agent continues).
    """

    reason: str


# ── Hook callback signatures ────────────────────────────────────────

# PreToolUse(tool_name: str, arguments: dict, tool_call: Any) -> HookBlock | None
# PostToolUse(tool_name: str, arguments: dict, result: Any) -> None
# PreLLMCall(messages: list[dict], tools: list[dict] | None) -> HookBlock | dict | None
# PostLLMCall(messages: list[dict], usage: Any | None) -> None
# SessionStart(cwd: str) -> None
# SessionEnd(stats: dict) -> HookBlock | None  (non-None = force continue)

HookCallback = Callable[..., Awaitable[HookBlock | dict | None]]


@dataclass(order=True)
class _Entry:
    """Ordered hook entry. Sort by (priority, insert_order)."""
    priority: int
    insert_order: int
    name: str = field(compare=False)
    callback: HookCallback = field(compare=False)


class HookSystem:
    """Register, unregister, and trigger hook callbacks.

    Callbacks are called in priority order (lowest first, default 50).
    Errors in one callback are isolated — they log and continue.
    A callback returning HookBlock stops the chain immediately.

    Usage::

        hooks = HookSystem()

        async def my_hook(tool_name: str, **kwargs) -> HookBlock | None:
            if tool_name == "run_shell":
                return HookBlock("shell blocked by policy")
            return None

        hooks.register("PreToolUse", my_hook, priority=10, name="shell-blocker")
        block = await hooks.trigger("PreToolUse", tool_name="bash", arguments={})
        if block:
            print(f"Blocked: {block.reason}")
    """

    def __init__(self):
        self._hooks: dict[str, list[_Entry]] = defaultdict(list)
        self._insert_counter: int = 0

    # ── Registration ─────────────────────────────────────────────

    def register(
        self,
        event: str,
        callback: HookCallback,
        priority: int = 50,
        name: str = "",
    ) -> None:
        """Register a callback for *event*.

        Args:
            event: One of the EVENT_* constants.
            callback: Async callable matching the event's signature.
            priority: Lower values fire first (0–100, default 50).
            name: Optional identifier for ``unregister()``.
        """
        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown hook event: {event!r}. Valid: {sorted(VALID_EVENTS)}"
            )
        self._insert_counter += 1
        entry = _Entry(
            priority=max(0, min(100, priority)),
            insert_order=self._insert_counter,
            name=name if name else f"hook_{self._insert_counter}",
            callback=callback,
        )
        self._hooks[event].append(entry)
        self._hooks[event].sort()

    def unregister(self, event: str, name: str) -> bool:
        """Remove all callbacks matching *name* from *event*.

        Returns True if at least one callback was removed.
        """
        if event not in self._hooks:
            return False
        before = len(self._hooks[event])
        self._hooks[event] = [e for e in self._hooks[event] if e.name != name]
        return len(self._hooks[event]) < before

    # Events where HookBlock return values are meaningful.
    # Fire-and-forget events (PostToolUse, PostLLMCall, SessionStart)
    # always run all callbacks regardless of returns.
    _BLOCKABLE_EVENTS = frozenset({
        EVENT_PRE_TOOL_USE,
        EVENT_PRE_LLM_CALL,
        EVENT_SESSION_END,
    })

    # ── Triggering ───────────────────────────────────────────────

    async def trigger(self, event: str, **kwargs: Any) -> HookBlock | dict | None:
        """Fire all callbacks registered for *event* in priority order.

        Callback semantics by event:
        - PreToolUse, PreLLMCall: return ``HookBlock`` to deny, ``dict`` to modify, ``None`` to allow.
        - PostToolUse, PostLLMCall, SessionStart: return value is ignored (fire-and-forget).
        - SessionEnd: return ``HookBlock`` to force continuation, ``None`` to allow exit.

        The chain stops at the first ``HookBlock`` return for blocking events.
        Errors in individual callbacks are logged and do not prevent remaining
        callbacks from firing.
        """
        if event not in VALID_EVENTS:
            return None

        can_block = event in self._BLOCKABLE_EVENTS

        for entry in self._hooks[event]:
            try:
                result = await entry.callback(**kwargs)
            except Exception:
                logger.exception(
                    "Hook %r (event=%r) raised an exception — skipping.",
                    entry.name,
                    event,
                )
                continue

            if not can_block:
                continue

            if isinstance(result, HookBlock):
                return result
            if isinstance(result, dict) and event == EVENT_PRE_LLM_CALL:
                # dict return modifies params (e.g., inject extra messages)
                return result

        return None

    # ── Introspection ────────────────────────────────────────────

    def count(self, event: str | None = None) -> int:
        """Return total registered callbacks, optionally filtered by event."""
        if event:
            return len(self._hooks.get(event, []))
        return sum(len(v) for v in self._hooks.values())

    def list_names(self, event: str | None = None) -> list[str]:
        """Return list of callback names, optionally filtered by event."""
        events = [event] if event else list(self._hooks)
        result: list[str] = []
        for ev in events:
            for entry in self._hooks.get(ev, []):
                result.append(f"{ev}/{entry.name}")
        return sorted(result)
