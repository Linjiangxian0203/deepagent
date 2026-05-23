"""AgentLoop — multi-turn ReAct loop.

P2 upgrade: autonomous multi-turn with ContextManager.
Each iteration: LLM stream → collect → commit assistant msg → execute tools → loop.
Breaks when LLM emits no tool calls or max_iterations is reached.
"""

import asyncio
from collections.abc import AsyncGenerator
import json
from typing import Protocol

from deepagent.config import Config
from deepagent.core.context import ContextManager
from deepagent.core.events import (
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    ToolResult,
    ToolLimitEvent,
    InterruptedEvent,
    DoneEvent,
    UsageEvent,
    AgentEvent,
)
from deepagent.tools.registry import ToolRegistry
from deepagent.tools.protocol import SafetyLevel


class ConfirmationHandler(Protocol):
    """Confirmation handler protocol. Core defines the interface; CLI provides it."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool: ...


class AgentLoop:
    """Multi-turn ReAct agent loop.

    Each turn:
    1. Stream LLM response (with tools), yield TextDelta/ThinkingDelta/ToolCallEvent
    2. If no tool calls: commit assistant msg, break (final response)
    3. If tool calls: commit assistant msg (with reasoning_content), execute each tool
    4. Loop continues — LLM sees tool results and decides next action
    """

    def __init__(
        self,
        config: Config,
        llm_client,
        tool_registry: ToolRegistry,
        context: ContextManager | None = None,
        confirm_handler: ConfirmationHandler | None = None,
        memory_store=None,  # MemoryStore, duck typing
    ):
        self.config = config
        self._llm = llm_client
        self._tools = tool_registry
        self._ctx = context or ContextManager()
        self._confirm = confirm_handler
        self._memory = memory_store
        self._interrupted = False

    def interrupt(self) -> None:
        self._interrupted = True

    async def run(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute multi-turn agent interaction for a single user message.

        Loops up to max_iterations, calling the LLM each turn. Breaks when
        the model responds without tool calls (final answer).
        """
        self._interrupted = False

        # L3: refresh long-term memory and inject into system prompt
        if self._memory is not None and self._memory.needs_refresh():
            self._memory.load_index()
            memory_context = self._memory.get_system_context()
            if memory_context:
                # Prepend memory index to existing system prompt
                current_prompt = self._ctx._system_prompt
                if "[Memory context]" not in current_prompt:
                    self._ctx._system_prompt = (
                        f"[Memory context]\n{memory_context}\n\n{current_prompt}"
                    )

        self._ctx.add_user_message(user_input)

        for iteration in range(self.config.max_iterations):
            if self._interrupted:
                yield InterruptedEvent()
                break

            # ── Phase 1: Stream LLM, collect only (no side effects) ──
            response_parts: list[str] = []
            pending_tool_calls: list = []
            reasoning_content: str | None = None

            # Compression: prevent context overflow (rarely triggered with 1M window)
            if self._ctx.is_near_limit():
                boundary = self._ctx.compression_candidates()
                if boundary > 0:
                    savings = self._ctx.estimate_compression_savings()
                    summary = (
                        f"Earlier conversation compressed to save ~{savings} tokens. "
                        "The following is a condensed summary of previous interactions."
                    )
                    self._ctx.compress_to(boundary, summary)
                    yield ThinkingDelta(text=f"\n[Context compressed: saved ~{savings} tokens]\n")

            tool_schemas = (
                self._tools.get_schemas() if self._tools.list_names() else None
            )

            async for event in self._llm.stream_chat(
                self._ctx.get_messages(), tools=tool_schemas
            ):
                if isinstance(event, (TextDelta, ThinkingDelta)):
                    yield event
                    if isinstance(event, TextDelta):
                        response_parts.append(event.text)

                elif isinstance(event, ToolCallEvent):
                    pending_tool_calls.extend(event.tool_calls)
                    reasoning_content = event.reasoning_content
                    if len(pending_tool_calls) > self.config.max_tools_per_turn:
                        yield ToolLimitEvent()
                        pending_tool_calls = pending_tool_calls[
                            : self.config.max_tools_per_turn
                        ]
                    yield event

                elif isinstance(event, UsageEvent):
                    self._ctx.update_usage(event.prompt_tokens, event.completion_tokens)
                    yield event

            if self._interrupted:
                yield InterruptedEvent()
                break

            # ── Phase 2: Commit assistant message ──
            assistant_text = "".join(response_parts)
            if pending_tool_calls:
                self._ctx.add_assistant_message(
                    assistant_text,
                    tool_calls=pending_tool_calls,
                    reasoning_content=reasoning_content,
                )
            else:
                self._ctx.add_assistant_message(assistant_text)
                break  # No tool calls → final response, done

            # ── Phase 3: Execute tools (stream already consumed) ──
            # Split by safety: READONLY tools run in parallel, WRITE/SHELL sequential
            readonly_calls = []
            mutable_calls = []
            for tc in pending_tool_calls:
                tool = self._tools.get(tc.name)
                if tool is not None and tool.tool_safety_level == SafetyLevel.READONLY:
                    readonly_calls.append(tc)
                else:
                    mutable_calls.append(tc)

            # ── 3a: Execute readonly tools in parallel ──
            if readonly_calls:
                for tc in readonly_calls:
                    yield ToolCallStartEvent(tool_call=tc)

                async def _exec_readonly(tc):
                    tool = self._tools.get(tc.name)
                    if tool is None:
                        return ToolResult(False, "", error=f"Tool '{tc.name}' not found")
                    try:
                        raw = await tool(**tc.arguments)
                        if isinstance(raw, dict):
                            return ToolResult(
                                success=raw.get("success", False),
                                content=raw.get("content", ""),
                                error=raw.get("error"),
                                metadata=raw.get("metadata"),
                            )
                        return ToolResult(success=True, content=str(raw))
                    except Exception as e:
                        return ToolResult(success=False, content="", error=str(e))

                results = await asyncio.gather(
                    *[_exec_readonly(tc) for tc in readonly_calls],
                    return_exceptions=True,
                )
                for tc, result in zip(readonly_calls, results):
                    if isinstance(result, BaseException):
                        result = ToolResult(
                            success=False, content="", error=str(result)
                        )
                    yield ToolResultEvent(tool_call=tc, result=result)
                    self._ctx.add_tool_result(tc.id, result)

            # ── 3b: Execute mutable (write/shell) tools sequentially ──
            for tc in mutable_calls:
                if self._interrupted:
                    yield InterruptedEvent()
                    yield DoneEvent()
                    return

                tool = self._tools.get(tc.name)
                if tool is None:
                    result = ToolResult(
                        success=False, content="", error=f"Tool '{tc.name}' not found"
                    )
                    yield ToolResultEvent(tool_call=tc, result=result)
                    self._ctx.add_tool_result(tc.id, result)
                    continue

                # Confirmation for shell-level tools
                if (
                    tool.tool_safety_level == SafetyLevel.SHELL
                    and self._confirm is not None
                ):
                    approved = await self._confirm.confirm(tc.name, tc.arguments)
                    if not approved:
                        denied = ToolResult(
                            success=False,
                            content="Execution denied by user.",
                            error="ExecutionDenied",
                        )
                        yield ToolResultEvent(tool_call=tc, result=denied)
                        self._ctx.add_tool_result(tc.id, denied)
                        continue

                yield ToolCallStartEvent(tool_call=tc)
                result = await self._safe_execute(tc, tool)
                yield ToolResultEvent(tool_call=tc, result=result)
                self._ctx.add_tool_result(tc.id, result)

            # Loop continues → LLM sees tool results, decides next action

        yield DoneEvent()

    async def _safe_execute(self, tc, tool) -> ToolResult:
        """Execute a tool with layered interrupt protection.

        Readonly tools are cancellable. Write/shell tools are shielded
        so they finish before the agent can be interrupted.
        """
        if tool.tool_safety_level == SafetyLevel.READONLY:
            return await self._do_execute(tc, tool)
        else:
            return await asyncio.shield(self._do_execute(tc, tool))

    async def _do_execute(self, tc, tool) -> ToolResult:
        """Execute a single tool call, converting the return value to ToolResult."""
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

        # Enrich failed results with recovery hints for auto-repair
        if not result.success and result.error:
            result = self._enrich_error(tc, result)
        return result

    @staticmethod
    def _enrich_error(tc, result: ToolResult) -> ToolResult:
        """Add recovery hints to failed tool results.

        Classifies common error patterns and injects actionable suggestions
        so the LLM can self-correct in the next turn.
        """
        error = (result.error or "").lower()
        hint = None

        if "not found" in error and "file" in error:
            hint = "The file was not found. Verify the file path exists and try again. Use glob() to search for the correct path."
        elif "not found" in error and "tool" in error:
            hint = "The requested tool is not available. Use only tools listed in the system prompt."
        elif "not found" in error and "path" in error:
            hint = "Directory not found. Check the path exists and try again."
        elif "not found" in error and "command" in error:
            hint = "The command was not found. Verify it's installed or use an alternative."
        elif "not found" in error:
            hint = "Resource not found. Check the name or path and retry."
        elif "not unique" in error or "found 2" in error or "found 3" in error:
            hint = "Duplicate match. Include more surrounding context in old_string to make the match unique."
        elif "outside" in error:
            hint = "The path is outside the allowed directory. Use a path within the project."
        elif "timed out" in error or "timeout" in error:
            hint = "The operation timed out. Consider a smaller scope or shorter timeout."
        elif "invalid" in error and "regex" in error:
            hint = "Invalid regex pattern. Check the syntax and try again."
        elif "permission" in error or "denied" in error:
            hint = "Permission denied. The operation requires different access."
        elif "exit code" in error:
            hint = "The command failed with a non-zero exit code. Check stderr in the output for details."

        if hint:
            result.error = f"{result.error}\n[Recovery hint: {hint}]"
        return result
