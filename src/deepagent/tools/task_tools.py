"""Task tools — todo_write for planning + persistent task management.

Two independent systems:
1. todo_write: Simple list with nag reminders, per-session
2. TaskManager: Persistent tasks with dependencies, cross-session
"""

from __future__ import annotations

import json
from pathlib import Path

from deepagent.core.tasks import TaskManager
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


_TODOS_DIR = ".tasks"


def create_todo_write_tool(reg: ToolRegistry) -> None:
    """Register the todo_write planning tool."""

    @tool(
        reg,
        name="todo_write",
        description="Create and manage a task list for your current coding session. Use to plan and track progress on multi-step tasks.",
        safety_level=SafetyLevel.WRITE,
    )
    async def todo_write(todos: list) -> dict:
        todos_dir = Path(_TODOS_DIR)
        todos_dir.mkdir(exist_ok=True)

        for i, t in enumerate(todos):
            if "content" not in t or "status" not in t:
                return {
                    "success": False, "content": "",
                    "error": f"Todo[{i}] missing 'content' or 'status'",
                }
            if t["status"] not in ("pending", "in_progress", "completed"):
                return {
                    "success": False, "content": "",
                    "error": f"Todo[{i}] invalid status '{t['status']}'",
                }

        todo_file = todos_dir / "current_todos.json"
        todo_file.write_text(json.dumps(todos, indent=2, ensure_ascii=False))

        lines = []
        for t in todos:
            icon = {"pending": " ", "in_progress": ">", "completed": "x"}[t["status"]]
            lines.append(f"  [{icon}] {t['content']}")
        summary = "\n".join(lines)

        return {"success": True, "content": f"Updated {len(todos)} tasks:\n{summary}"}


def create_task_system_tools(reg: ToolRegistry, task_mgr: TaskManager) -> None:
    """Register persistent task system tools."""

    @tool(
        reg,
        name="create_task",
        description="Create a new persistent task with optional blockedBy dependencies.",
        safety_level=SafetyLevel.WRITE,
    )
    async def create_task(
        subject: str,
        description: str = "",
        blocked_by: list | None = None,
    ) -> dict:
        task = task_mgr.create_task(subject, description, blocked_by)
        deps = f" (blockedBy: {', '.join(task.blocked_by)})" if task.blocked_by else ""
        return {"success": True, "content": f"Created {task.id}: {task.subject}{deps}"}

    @tool(
        reg,
        name="list_tasks",
        description="List all tasks with status, owner, and dependencies.",
        safety_level=SafetyLevel.READONLY,
    )
    async def list_tasks() -> dict:
        tasks = task_mgr.list_all()
        if not tasks:
            return {"success": True, "content": "No tasks."}
        lines = []
        for t in tasks:
            icon = {"pending": "o", "in_progress": "*", "completed": "x"}.get(t.status, "?")
            deps = f" (blockedBy: {', '.join(t.blocked_by)})" if t.blocked_by else ""
            owner = f" [{t.owner}]" if t.owner else ""
            lines.append(f"  {icon} {t.id}: {t.subject} [{t.status}]{owner}{deps}")
        return {"success": True, "content": "\n".join(lines)}

    @tool(
        reg,
        name="get_task",
        description="Get full details of a specific task by ID.",
        safety_level=SafetyLevel.READONLY,
    )
    async def get_task(task_id: str) -> dict:
        task = task_mgr.load(task_id)
        if task is None:
            return {"success": False, "content": "", "error": f"Task not found: {task_id}"}
        return {"success": True, "content": json.dumps(task.to_dict(), indent=2)}

    @tool(
        reg,
        name="claim_task",
        description="Claim a pending task. Sets owner and changes status to in_progress.",
        safety_level=SafetyLevel.WRITE,
    )
    async def claim_task(task_id: str) -> dict:
        result = task_mgr.claim(task_id)
        if result is None:
            task = task_mgr.load(task_id)
            if task is None:
                return {"success": False, "content": "", "error": f"Task not found: {task_id}"}
            if task.status != "pending":
                return {"success": False, "content": "", "error": f"Task is {task.status}, not pending"}
            if not task_mgr.can_start(task_id):
                blocked = [d for d in task.blocked_by
                          if not task_mgr.load(d) or task_mgr.load(d).status != "completed"]
                return {"success": False, "content": "", "error": f"Blocked by: {blocked}"}
        return {"success": True, "content": f"Claimed {task_id}"}

    @tool(
        reg,
        name="complete_task",
        description="Complete an in-progress task. Reports unblocked downstream tasks.",
        safety_level=SafetyLevel.WRITE,
    )
    async def complete_task(task_id: str) -> dict:
        result = task_mgr.complete(task_id)
        if result is None:
            task = task_mgr.load(task_id)
            if task is None:
                return {"success": False, "content": "", "error": f"Task not found: {task_id}"}
            return {"success": False, "content": "", "error": f"Task is {task.status}, not in_progress"}
        return {"success": True, "content": f"Completed {result}"}
