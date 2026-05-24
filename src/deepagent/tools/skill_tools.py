"""Skill tools — expose load_skill to the agent for on-demand skill loading.

Uses the existing SkillRegistry from core/skills.py (Phase 1).
The registry is passed in at tool creation time.
"""

from __future__ import annotations
from deepagent.core.skills import SkillRegistry
from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


def create_skill_tools(reg: ToolRegistry, skill_registry: SkillRegistry) -> None:
    """Register skill-related tools into *reg*."""

    @tool(
        reg,
        name="load_skill",
        description="Load the full content of a skill by name. Skills provide specialized instructions for tasks like code review, debugging, or working with specific file formats.",
        safety_level=SafetyLevel.READONLY,
    )
    async def load_skill(name: str) -> dict:
        content = skill_registry.load(name)
        if content is None:
            return {"success": False, "content": "", "error": f"Skill not found: {name}"}
        skill_registry.check_hot_reload()
        return {"success": True, "content": content}
