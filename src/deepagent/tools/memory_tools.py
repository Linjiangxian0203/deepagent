"""Memory tools — expose create/update/delete to the agent for bidirectional memory.

The MemoryStore is passed in at tool creation time.
"""

from __future__ import annotations

from deepagent.memory.store import MemoryStore
from deepagent.memory.models import MemoryEntry
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


def create_memory_tools(reg: ToolRegistry, store: MemoryStore) -> None:
    """Register memory CRUD tools into *reg*."""

    @tool(
        reg,
        name="create_memory",
        description="Create a new persistent memory entry. Use for user preferences, project facts, or reference pointers.",
        safety_level=SafetyLevel.WRITE,
    )
    async def create_memory(
        name: str,
        description: str,
        memory_type: str,
        content: str,
    ) -> dict:
        if memory_type not in ("user", "feedback", "project", "reference"):
            return {
                "success": False, "content": "",
                "error": f"Invalid type '{memory_type}'. Must be: user, feedback, project, reference",
            }
        entry = MemoryEntry(
            name=name,
            description=description,
            memory_type=memory_type,
            content=content,
        )
        store.write_entry(entry)
        store.increment_hits(name)
        return {"success": True, "content": f"Created memory: {name}"}

    @tool(
        reg,
        name="update_memory",
        description="Update an existing memory entry's content.",
        safety_level=SafetyLevel.WRITE,
    )
    async def update_memory(name: str, content: str) -> dict:
        existing = store.read_entry(name)
        if existing is None:
            return {"success": False, "content": "", "error": f"Memory not found: {name}"}
        existing.content = content
        store.write_entry(existing)
        return {"success": True, "content": f"Updated memory: {name}"}

    @tool(
        reg,
        name="delete_memory",
        description="Delete a memory entry by name.",
        safety_level=SafetyLevel.WRITE,
    )
    async def delete_memory(name: str) -> dict:
        existing = store.read_entry(name)
        if existing is None:
            return {"success": False, "content": "", "error": f"Memory not found: {name}"}
        store.delete_entry(name)
        return {"success": True, "content": f"Deleted memory: {name}"}
