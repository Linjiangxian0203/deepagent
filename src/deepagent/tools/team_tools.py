"""Team tools — spawn_teammate, send_message, check_inbox.

Teammates are asyncio-based autonomous agents with restricted tool sets
(no recursive spawn). They communicate with the Lead agent via MessageBus.

Reference: learn-claude-code s15_agent_teams, s16_team_protocols.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from deepagent.core.message_bus import MessageBus
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel

logger = logging.getLogger(__name__)

# Tools available to teammates (no recursive spawn_teammate)
TEAMMATE_TOOL_NAMES = frozenset({
    "read_file", "write_file", "edit_file",
    "glob", "grep", "run_shell",
    "send_message",
    "list_tasks", "get_task", "claim_task", "complete_task",
    "web_search", "web_fetch",
})

TEAMMATE_SYSTEM_PROMPT = """You are a teammate agent working as part of a team led by 'lead'.

## Rules
- Complete the task you were given, then send the result via send_message to 'lead'.
- You have limited tools — focus on what you can do.
- Do not try to spawn other agents.
- Be concise and direct in your results.
- Check your inbox periodically for messages from lead."""


def _build_teammate_tool_schemas(tool_registry: ToolRegistry) -> list[dict] | None:
    """Build schemas for only the tools visible to teammates."""
    schemas = []
    for name in TEAMMATE_TOOL_NAMES:
        t = tool_registry.get(name)
        if t is not None:
            schemas.append({
                "type": "function",
                "function": {
                    "name": t.tool_name,
                    "description": t.tool_description,
                    "parameters": t.tool_parameters,
                },
            })
    return schemas if schemas else None


async def _execute_teammate_tool(
    tool_registry: ToolRegistry,
    tool_name: str,
    arguments: dict,
    bus: MessageBus,
    teammate_name: str,
) -> str:
    """Execute a single tool call on behalf of a teammate. Returns result string.

    send_message is intercepted to use the teammate's name as sender.
    """
    # Intercept send_message to use the teammate's name as sender
    if tool_name == "send_message":
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        bus.send(teammate_name, to, content)
        return "Sent"

    t = tool_registry.get(tool_name)
    if t is None:
        return f"Unknown tool: {tool_name}"
    try:
        raw = await t(**arguments)
        if isinstance(raw, dict):
            return raw.get("content", json.dumps(raw, ensure_ascii=False))
        return str(raw)
    except Exception as e:
        return f"Error: {e}"


async def _teammate_loop(
    name: str,
    role: str,
    prompt: str,
    llm_client,
    tool_registry: ToolRegistry,
    bus: MessageBus,
    cwd: str,
    max_rounds: int = 10,
) -> None:
    """Run a teammate agent loop in an asyncio task.

    The teammate processes one user prompt, loops for up to max_rounds,
    checks its inbox before each LLM call, and sends results via MessageBus.
    """
    system = (
        f"You are '{name}', a {role}. Working directory: {cwd}. "
        f"Use send_message(to='lead', content=...) to report your results. "
        f"Do not spawn other agents."
    )

    messages: list[dict] = [{"role": "user", "content": prompt}]
    tool_schemas = _build_teammate_tool_schemas(tool_registry)

    for _ in range(max_rounds):
        # Check inbox for new instructions from Lead
        inbox_msgs = bus.read_inbox(name)
        for msg in inbox_msgs:
            messages.append({
                "role": "user",
                "content": f"[From {msg['from']}] {msg['content']}",
            })

        try:
            response_parts: list[str] = []
            pending_tool_calls: list = []

            async for event in llm_client.stream_chat(messages, tools=tool_schemas):
                from deepagent.core.events import TextDelta, ToolCallEvent
                if isinstance(event, TextDelta):
                    response_parts.append(event.text)
                elif isinstance(event, ToolCallEvent):
                    pending_tool_calls.extend(event.tool_calls)

            assistant_text = "".join(response_parts)

            if not pending_tool_calls:
                # No tool calls — final response, send result to Lead
                result = assistant_text or "Task completed."
                bus.send(name, "lead", result, "result")
                return

            # Commit assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in pending_tool_calls
                ],
            })

            # Execute each tool call and add results
            for tc in pending_tool_calls:
                output = await _execute_teammate_tool(
                    tool_registry, tc.name, tc.arguments, bus, name
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                })

        except Exception as e:
            logger.warning("Teammate %s error: %s", name, e)
            bus.send(name, "lead", f"Error: {e}", "result")
            return

    # Max rounds reached — send last response
    bus.send(name, "lead", "Task reached max rounds without final answer.", "result")


def create_team_tools(
    reg: ToolRegistry,
    bus: MessageBus,
    llm_client,
    tool_registry: ToolRegistry,
    cwd: str = "",
) -> None:
    """Register all team-related tools into *reg*.

    Args:
        reg: ToolRegistry to register into (Lead's tools).
        bus: MessageBus instance for inter-agent communication.
        llm_client: LLMClient for spawning teammate agents.
        tool_registry: ToolRegistry for tool execution (teammates use a
            restricted subset of these tools).
        cwd: Working directory path for teammates.
    """

    @tool(
        reg,
        name="spawn_teammate",
        description="Spawn a teammate agent to work autonomously on a task. The teammate has access to file, shell, and search tools (but cannot spawn other agents). Results are sent via MessageBus.",
        safety_level=SafetyLevel.WRITE,
    )
    async def spawn_teammate(name: str, role: str, prompt: str) -> dict:
        if not name or not prompt:
            return {"success": False, "content": "", "error": "name and prompt are required"}

        asyncio.create_task(
            _teammate_loop(name, role, prompt, llm_client, tool_registry, bus, cwd)
        )
        return {
            "success": True,
            "content": f"Teammate '{name}' spawned as {role}. Will report results via inbox.",
        }

    @tool(
        reg,
        name="send_message",
        description="Send a message to another agent via MessageBus.",
        safety_level=SafetyLevel.READONLY,
    )
    async def send_message(to: str, content: str) -> dict:
        if not to or not content:
            return {"success": False, "content": "", "error": "to and content are required"}
        bus.send("lead", to, content)
        return {"success": True, "content": f"Sent to {to}"}

    @tool(
        reg,
        name="check_inbox",
        description="Check Lead's inbox for messages from teammates.",
        safety_level=SafetyLevel.READONLY,
    )
    async def check_inbox() -> dict:
        msgs = bus.read_inbox("lead")
        if not msgs:
            return {"success": True, "content": "(inbox empty)"}
        lines = [f"  [{m['from']}] [{m['type']}] {m['content'][:300]}" for m in msgs]
        return {"success": True, "content": "\n".join(lines)}
