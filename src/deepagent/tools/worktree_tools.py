"""Worktree tools — create/remove/keep isolated git worktrees for agent tasks."""

from __future__ import annotations

from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


def create_worktree_tools(reg: ToolRegistry, manager) -> None:
    """Register worktree tools into *reg*. *manager* is a WorktreeManager instance."""

    @tool(
        reg,
        name="create_worktree",
        description="Create an isolated git worktree with its own branch under .worktrees/. The name must be 1-64 characters using only A-Za-z0-9._-",
        safety_level=SafetyLevel.WRITE,
    )
    async def create_worktree(name: str, task_id: str = "") -> dict:
        ok, msg = await manager.create(name, task_id)
        if ok:
            return {"success": True, "content": msg}
        return {"success": False, "content": "", "error": msg}

    @tool(
        reg,
        name="remove_worktree",
        description="Remove a worktree. Refuses if there are uncommitted changes unless discard_changes=true.",
        safety_level=SafetyLevel.WRITE,
    )
    async def remove_worktree(name: str, discard_changes: bool = False) -> dict:
        ok, msg = await manager.remove(name, discard_changes)
        if ok:
            return {"success": True, "content": msg}
        return {"success": False, "content": "", "error": msg}

    @tool(
        reg,
        name="keep_worktree",
        description="Keep a worktree for manual review. The branch wt/{name} will be preserved.",
        safety_level=SafetyLevel.WRITE,
    )
    async def keep_worktree(name: str) -> dict:
        ok, msg = await manager.keep(name)
        if ok:
            return {"success": True, "content": msg}
        return {"success": False, "content": "", "error": msg}
