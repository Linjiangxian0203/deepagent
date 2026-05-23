"""Sub-agent system for delegating tasks to parallel sub-agents.

A SubAgentRunner manages spawning sub-agents with concurrency control.
Sub-agents are lightweight AgentLoops with their own ContextManager but
sharing the same LLM client and tool registry.

Rate limit awareness:
- deepseek-v4-flash: 2500 concurrent connections (account-level)
- deepseek-v4-pro: 500 concurrent connections
We default to max_concurrent=5 as a conservative limit per session.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from deepagent.config import Config
from deepagent.core.context import ContextManager
from deepagent.core.events import TextDelta, ToolCallStartEvent, ToolResult, AgentEvent


SUB_AGENT_SYSTEM_PROMPT = """You are a sub-agent handling a delegated subtask.
Work autonomously to complete the assigned task. Use tools as needed.
Return a clear, concise result when done. Do not ask clarifying questions —
make reasonable assumptions and proceed."""


class SubAgentRunner:
    """Manages sub-agent spawning with concurrency limiting."""

    def __init__(
        self,
        config: Config,
        llm_client,
        tool_registry,
        max_concurrent: int = 5,
    ):
        self._config = config
        self._llm_client = llm_client
        self._tools = tool_registry
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._max_concurrent = max_concurrent

    @property
    def active_count(self) -> int:
        return self._active_count

    async def run(self, description: str, prompt: str) -> dict:
        """Spawn a sub-agent to handle a delegated task.

        The sub-agent runs with its own ContextManager and the shared tools.
        Concurrency is limited by a semaphore.

        Returns a ToolResult-compatible dict.
        """
        async with self._semaphore:
            self._active_count += 1
            try:
                return await self._run_internal(description, prompt)
            finally:
                self._active_count -= 1

    async def _run_internal(self, description: str, prompt: str) -> dict:
        """Internal sub-agent execution, called under semaphore protection."""
        from deepagent.core.loop import AgentLoop  # late import to avoid cycles

        ctx = ContextManager(system_prompt=SUB_AGENT_SYSTEM_PROMPT)
        loop = AgentLoop(
            config=self._config,
            llm_client=self._llm_client,
            tool_registry=self._tools,
            context=ctx,
            # No confirmation — sub-agents run autonomously
            confirm_handler=None,
        )

        text_parts: list[str] = []
        tool_calls_made = 0

        try:
            async for event in loop.run(prompt):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, ToolCallStartEvent):
                    tool_calls_made += 1
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Sub-agent failed: {e}",
                "metadata": {"description": description, "tool_calls": tool_calls_made},
            }

        result_text = "".join(text_parts).strip()
        return {
            "success": True,
            "content": result_text if result_text else "(no output)",
            "error": None,
            "metadata": {
                "description": description,
                "tool_calls": tool_calls_made,
                "chars": len(result_text),
            },
        }
