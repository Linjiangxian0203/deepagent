from collections.abc import AsyncGenerator
import json
from typing import Protocol

from deepagent.config import Config
from deepagent.core.events import (
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    ToolResult,
    DoneEvent,
    AgentEvent,
)
from deepagent.tools.registry import ToolRegistry


class ConfirmationHandler(Protocol):
    """Confirmation handler protocol. Core defines the interface; CLI layer implements it."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool: ...


class AgentLoop:
    """Main agent loop. P1 implements single-turn: LLM stream -> tool execution -> LLM sees results -> final reply."""

    def __init__(
        self,
        config: Config,
        llm_client,  # LLMClient, duck typing
        tool_registry: ToolRegistry,
        confirm_handler: ConfirmationHandler | None = None,
    ):
        self.config = config
        self._llm = llm_client
        self._tools = tool_registry
        self._confirm = confirm_handler

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Run a single-turn agent interaction.

        Flow:
        1. Call LLM, stream TextDelta / ThinkingDelta / ToolCallEvent
        2. If LLM returns tool calls: execute them, feed results back to LLM, stream final reply
        3. If no tool calls: finish directly
        """
        messages: list[dict] = [{"role": "user", "content": user_input}]
        tool_schemas = self._tools.get_schemas() if self._tools.list_names() else None

        # -- Phase 1: First LLM call --
        response_parts: list[str] = []
        pending_tool_calls: list = []

        async for event in self._llm.stream_chat(
            messages, tools=tool_schemas if tool_schemas else None
        ):
            if isinstance(event, (TextDelta, ThinkingDelta)):
                yield event
                if isinstance(event, TextDelta):
                    response_parts.append(event.text)

            elif isinstance(event, ToolCallEvent):
                pending_tool_calls.extend(event.tool_calls)
                yield event

        # -- Phase 2: No tool calls, finish --
        if not pending_tool_calls:
            if response_parts:
                messages.append(
                    {"role": "assistant", "content": "".join(response_parts)}
                )
            yield DoneEvent()
            return

        # -- Phase 3: Execute tools --
        tool_results: list[dict] = []

        for tc in pending_tool_calls:
            tool = self._tools.get(tc.name)

            # Confirmation check
            if self._confirm is not None:
                approved = await self._confirm.confirm(tc.name, tc.arguments)
                if not approved:
                    denied_result = ToolResult(
                        success=False,
                        content="Execution denied by user.",
                        error="ExecutionDenied",
                    )
                    result_event = ToolResultEvent(
                        tool_call=tc, result=denied_result
                    )
                    yield result_event
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Execution denied by user.",
                        }
                    )
                    continue

            # Tool execution
            yield ToolCallStartEvent(tool_call=tc)

            if tool is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"Tool '{tc.name}' not found",
                )
            else:
                try:
                    raw = await tool(**tc.arguments)
                    if isinstance(raw, dict):
                        result = ToolResult(
                            success=raw.get("success", False),
                            content=raw.get("content", ""),
                            error=raw.get("error"),
                            metadata=raw.get("metadata"),
                        )
                    else:
                        result = ToolResult(success=True, content=str(raw))
                except Exception as e:
                    result = ToolResult(success=False, content="", error=str(e))

            result_event = ToolResultEvent(tool_call=tc, result=result)
            yield result_event

            tool_results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": (
                        result.content
                        if result.success
                        else f"Error: {result.error}"
                    ),
                }
            )

        # -- Phase 4: Feed tool results back to LLM, get final reply --
        assistant_msg = {"role": "assistant", "content": "".join(response_parts)}
        if pending_tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in pending_tool_calls
            ]
        messages.append(assistant_msg)
        messages.extend(tool_results)

        # Second LLM call (without tools, let LLM summarize tool results)
        async for event in self._llm.stream_chat(messages, tools=None):
            yield event

        yield DoneEvent()
