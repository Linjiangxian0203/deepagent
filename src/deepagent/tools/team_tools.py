"""Team tools — spawn_teammate, send_message, check_inbox, protocol tools.

Layer 1: MessageBus-based communication.
Layer 2: Teammate spawning with asyncio agent loop.
Layer 3: Request-response protocols (shutdown, plan approval, code review).
Layer 4: Autonomous agents with idle polling and auto-claim.

Reference: learn-claude-code s15_s16_s17.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from deepagent.core.message_bus import MessageBus
from deepagent.core.protocols import ProtocolManager, new_request_id
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel

logger = logging.getLogger(__name__)

# Tools available to teammates (no recursive spawn_teammate, no team mgmt)
TEAMMATE_TOOL_NAMES = frozenset({
    "read_file", "write_file", "edit_file",
    "glob", "grep", "run_shell",
    "send_message", "submit_plan",
    "list_tasks", "get_task", "claim_task", "complete_task",
    "web_search", "web_fetch",
})

TEAMMATE_SYSTEM_PROMPT = """You are a teammate agent working as part of a team led by 'lead'.

## Rules
- Complete the task you were given, then send the result to 'lead' via send_message.
- You have limited tools — focus on what you can do.
- Do not try to spawn other agents.
- Check your inbox each round for protocol messages (shutdown_request, plan_approval_response).
- If you receive a shutdown_request, acknowledge and stop."""


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
    proto: ProtocolManager,
) -> str:
    """Execute a single tool call on behalf of a teammate.

    Special handling:
    - send_message: uses teammate_name as sender
    - submit_plan: creates a plan_approval protocol request
    """
    if tool_name == "send_message":
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        bus.send(teammate_name, to, content)
        return "Sent"

    if tool_name == "submit_plan":
        plan = arguments.get("plan", "")
        req_id = proto.new_request("plan_approval", teammate_name, "lead", plan)
        bus.send(teammate_name, "lead", plan, "plan_approval_request",
                 {"request_id": req_id})
        return f"Plan submitted ({req_id}). Waiting for approval..."

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


# ── Protocol message dispatch ───────────────────────────────────────


def _dispatch_inbox_message(
    name: str,
    msg: dict,
    messages: list[dict],
    bus: MessageBus,
    proto: ProtocolManager,
) -> bool:
    """Dispatch an inbox message by type. Returns True if teammate should stop.

    Protocol messages (shutdown_request, plan_approval_response) are handled
    automatically. Regular messages are returned for the caller to inject.
    """
    msg_type = msg.get("type", "message")
    meta = msg.get("metadata", {})
    req_id = meta.get("request_id", "")

    if msg_type == "shutdown_request":
        bus.send(name, "lead", "Shutting down gracefully.",
                 "shutdown_response",
                 {"request_id": req_id, "approve": True})
        logger.info("Teammate %s approved shutdown (%s)", name, req_id)
        return True  # stop the loop

    if msg_type == "plan_approval_response":
        approve = meta.get("approve", False)
        if approve:
            messages.append({"role": "user",
                "content": "[Plan approved] Proceed with the task."})
        else:
            feedback = msg.get("content", "")
            messages.append({"role": "user",
                "content": f"[Plan rejected] Feedback: {feedback}"})

    return False  # continue


# ── Teammate Agent Loop ─────────────────────────────────────────────


async def _teammate_loop(
    name: str,
    role: str,
    prompt: str,
    llm_client,
    tool_registry: ToolRegistry,
    bus: MessageBus,
    proto: ProtocolManager,
    cwd: str,
    task_mgr=None,        # TaskManager for auto-claim
    max_rounds: int = 10,
    autonomous: bool = False,
) -> None:
    """Run a teammate agent loop in an asyncio task.

    Layer 2: Processes one prompt, loops up to max_rounds, checks inbox.
    Layer 4 (autonomous=True): WORK→IDLE→WORK lifecycle with auto-claim.
        During IDLE: polls inbox (priority), scans unclaimed tasks,
        auto-claims, re-enters WORK. Detects shutdown_request for exit.

    If autonomous=False and the LLM responds without tool calls, the
    teammate sends its result and exits (Layer 2 behavior).
    """
    system = (
        f"You are '{name}', a {role}. Working directory: {cwd}. "
        f"Use send_message(to='lead', content=...) to report results. "
        f"Do not spawn other agents."
    )

    messages: list[dict] = [{"role": "user", "content": prompt}]
    tool_schemas = _build_teammate_tool_schemas(tool_registry)

    rounds = 0
    idle_poll_interval = 5
    idle_timeout = 60

    while rounds < max_rounds:
        # ── Check inbox for protocol messages ──
        inbox_msgs = bus.read_inbox(name)
        should_stop = False
        for msg in inbox_msgs:
            if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                if _dispatch_inbox_message(name, msg, messages, bus, proto):
                    should_stop = True
                    break
            else:
                messages.append({
                    "role": "user",
                    "content": f"[From {msg['from']}] {msg['content']}",
                })

        if should_stop:
            break

        # ── LLM turn ──
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
            rounds += 1

            if not pending_tool_calls:
                if not autonomous:
                    # Layer 2: final response, send result to Lead and exit
                    result = assistant_text or "Task completed."
                    bus.send(name, "lead", result, "result")
                    return

                # Layer 4: Autonomous — enter IDLE mode
                idle_start = asyncio.get_event_loop().time()
                while True:
                    # Check for timeout
                    elapsed = asyncio.get_event_loop().time() - idle_start
                    if elapsed > idle_timeout:
                        bus.send(name, "lead", "Idle timeout — shutting down.", "result")
                        return

                    # Poll inbox
                    await asyncio.sleep(idle_poll_interval)
                    inbox_msgs = bus.read_inbox(name)
                    for msg in inbox_msgs:
                        if msg.get("type") in ("shutdown_request", "plan_approval_response"):
                            if _dispatch_inbox_message(name, msg, messages, bus, proto):
                                return  # shutdown
                        else:
                            # New task message — re-enter WORK
                            messages.append({
                                "role": "user",
                                "content": f"[From {msg['from']}] {msg['content']}",
                            })
                            break
                    else:
                        # No inbox messages — check task board
                        if task_mgr is not None:
                            unclaimed = _scan_unclaimed_tasks(task_mgr)
                            if unclaimed:
                                task = unclaimed[0]
                                claim_result = task_mgr.claim(task.id, owner=name)
                                if claim_result is not None:
                                    messages.append({
                                        "role": "user",
                                        "content": (
                                            f"[Auto-claimed task] ID: {task.id}\n"
                                            f"Subject: {task.subject}\n"
                                            f"Description: {task.description}\n"
                                            f"Complete this task and send result to lead."
                                        ),
                                    })
                                    break  # re-enter WORK with new task
                    # if we get here with no break, loop continues polling
                    continue

                # if we broke out, re-enter WORK
                continue

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

            for tc in pending_tool_calls:
                output = await _execute_teammate_tool(
                    tool_registry, tc.name, tc.arguments, bus, name, proto
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

    if not autonomous:
        bus.send(name, "lead", "Task reached max rounds without final answer.", "result")


# ── Autonomous agent task scanning ──────────────────────────────────


def _scan_unclaimed_tasks(task_mgr) -> list:
    """Find pending, unowned tasks whose dependencies are all completed."""
    candidates = []
    for t in task_mgr.list_all(status="pending"):
        if t.owner is None and task_mgr.can_start(t.id):
            candidates.append(t)
    return candidates


# ── Tool Registration ───────────────────────────────────────────────


def create_team_tools(
    reg: ToolRegistry,
    bus: MessageBus,
    llm_client,
    tool_registry: ToolRegistry,
    cwd: str = "",
    task_mgr=None,  # TaskManager for auto-claim (Layer 4)
) -> ProtocolManager:
    """Register all team-related tools into *reg*.

    Returns the ProtocolManager instance used by the tools.
    """
    proto = ProtocolManager()

    @tool(
        reg,
        name="spawn_teammate",
        description="Spawn a teammate agent to work autonomously on a task. The teammate has access to file, shell, and search tools (but cannot spawn other agents). Use autonomous=true for persistent teammates that poll for tasks. Results are sent via MessageBus.",
        safety_level=SafetyLevel.WRITE,
    )
    async def spawn_teammate(
        name: str,
        role: str,
        prompt: str,
        autonomous: bool = False,
    ) -> dict:
        if not name or not prompt:
            return {"success": False, "content": "", "error": "name and prompt are required"}

        asyncio.create_task(
            _teammate_loop(
                name, role, prompt, llm_client, tool_registry, bus, proto,
                cwd, task_mgr=task_mgr, autonomous=autonomous,
            )
        )
        mode = "autonomous" if autonomous else "task-based"
        return {
            "success": True,
            "content": f"Teammate '{name}' spawned as {role} [{mode}]. Will report results via inbox.",
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
        description="Check Lead's inbox for messages from teammates. Protocol responses are automatically routed.",
        safety_level=SafetyLevel.READONLY,
    )
    async def check_inbox() -> dict:
        msgs = bus.read_inbox("lead")
        if not msgs:
            return {"success": True, "content": "(inbox empty)"}
        lines = []
        for m in msgs:
            meta = m.get("metadata", {})
            req_id = meta.get("request_id", "")
            tag = f" [{m['type']} req:{req_id}]" if req_id else f" [{m['type']}]"
            lines.append(f"  [{m['from']}]{tag} {m['content'][:300]}")
        return {"success": True, "content": "\n".join(lines)}

    # ── Protocol tools (Layer 3) ──

    @tool(
        reg,
        name="request_shutdown",
        description="Request a teammate to shut down gracefully via protocol.",
        safety_level=SafetyLevel.WRITE,
    )
    async def request_shutdown(teammate: str) -> dict:
        req_id = proto.new_request("shutdown", "lead", teammate)
        bus.send("lead", teammate, "Please shut down gracefully.",
                 "shutdown_request", {"request_id": req_id})
        logger.info("shutdown_request → %s (%s)", teammate, req_id)
        return {"success": True, "content": f"Shutdown request sent to {teammate} (req: {req_id})"}

    @tool(
        reg,
        name="request_plan",
        description="Ask a teammate to submit a plan for review via protocol.",
        safety_level=SafetyLevel.WRITE,
    )
    async def request_plan(teammate: str, task: str) -> dict:
        bus.send("lead", teammate, f"Please submit a plan for: {task}")
        return {"success": True, "content": f"Asked {teammate} to submit a plan"}

    @tool(
        reg,
        name="review_plan",
        description="Approve or reject a submitted plan by request_id.",
        safety_level=SafetyLevel.WRITE,
    )
    async def review_plan(request_id: str, approve: bool, feedback: str = "") -> dict:
        state = proto.get_state(request_id)
        if state is None:
            return {"success": False, "content": "", "error": f"Request {request_id} not found"}
        if state.status != "pending":
            return {"success": False, "content": "", "error": f"Request {request_id} already {state.status}"}
        state.status = "approved" if approve else "rejected"
        bus.send("lead", state.sender, feedback or ("Approved" if approve else "Rejected"),
                 "plan_approval_response",
                 {"request_id": request_id, "approve": approve})
        return {"success": True, "content": f"Plan {'approved' if approve else 'rejected'} ({request_id})"}

    return proto
