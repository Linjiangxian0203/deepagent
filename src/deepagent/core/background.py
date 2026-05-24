"""Background task management — asyncio-based parallel tool execution.

Uses asyncio.create_task (not threading) matching deepagent's async architecture.
Detection: explicit run_in_background: true OR heuristic keyword matching.

Reference: learn-claude-code s13_background_tasks. Adapted to asyncio.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


SLOW_KEYWORDS = frozenset([
    "install", "build", "test", "deploy", "compile",
    "docker build", "pip install", "npm install",
    "cargo build", "pytest", "make",
])


@dataclass
class BackgroundTask:
    bg_id: str
    tool_name: str
    arguments: dict
    status: str = "running"
    result: dict | None = None


def is_slow_operation(tool_name: str, arguments: dict) -> bool:
    """Fallback heuristic: commands likely to take > 30s."""
    if tool_name != "run_shell":
        return False
    cmd = arguments.get("command", "").lower()
    return any(kw in cmd for kw in SLOW_KEYWORDS)


def should_run_background(tool_name: str, arguments: dict) -> bool:
    """Check if tool should run in background.
    Explicit run_in_background: true takes priority over heuristic.
    """
    if arguments.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, arguments)


def format_notification(bg_id: str, command: str, summary: str) -> str:
    """Format a background completion notification for LLM context injection."""
    truncated = summary[:200] if len(summary) > 200 else summary
    return (
        f"<task_notification>\n"
        f"  <task_id>{bg_id}</task_id>\n"
        f"  <status>completed</status>\n"
        f"  <command>{command}</command>\n"
        f"  <summary>{truncated}</summary>\n"
        f"</task_notification>"
    )


class BackgroundManager:
    """Manages asyncio background tool execution.

    Usage::

        mgr = BackgroundManager()
        bg_id = mgr.start("run_shell", {"command": "npm install"}, coroutine)
        # ... later, after LLM turn completes:
        notifications = mgr.collect_ready()
    """

    def __init__(self):
        self._counter = 0
        self._tasks: dict[str, BackgroundTask] = {}
        self._futures: dict[str, asyncio.Task] = {}

    def start(
        self,
        tool_name: str,
        arguments: dict,
        coroutine,
    ) -> str:
        """Dispatch a coroutine as a background task. Returns background task ID."""
        self._counter += 1
        bg_id = f"bg_{self._counter:04d}"

        bt = BackgroundTask(
            bg_id=bg_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        self._tasks[bg_id] = bt

        async def _runner():
            try:
                result = await coroutine
                bt.result = result if isinstance(result, dict) else {"success": True, "content": str(result)}
            except Exception as e:
                bt.result = {"success": False, "content": "", "error": str(e)}
            finally:
                bt.status = "completed"

        self._futures[bg_id] = asyncio.create_task(_runner())
        return bg_id

    def collect_ready(self) -> list[dict]:
        """Collect completed background results as notification dicts.
        Removes completed tasks from internal tracking.
        """
        ready = []
        for bg_id, bt in list(self._tasks.items()):
            if bt.status == "completed":
                command = bt.arguments.get("command", bt.tool_name)
                summary = bt.result.get("content", "") if bt.result else ""
                ready.append({
                    "bg_id": bg_id,
                    "command": command,
                    "summary": summary,
                    "notification": format_notification(bg_id, command, summary),
                })
                self._tasks.pop(bg_id, None)
                self._futures.pop(bg_id, None)
        return ready

    def cancel(self, bg_id: str) -> bool:
        """Cancel a running background task. Returns True if cancelled."""
        future = self._futures.pop(bg_id, None)
        bt = self._tasks.pop(bg_id, None)
        if future and not future.done():
            future.cancel()
        return future is not None

    @property
    def pending_count(self) -> int:
        return len(self._tasks)
