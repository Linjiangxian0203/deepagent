"""Dynamic system prompt — registry of PromptSections with conditional assembly.

Replaces the hardcoded ~30-line string concatenation in cli/app.py with
a section registry. Each section can have a condition callable; sections
are assembled in priority order. Output is cached until section state changes.

Reference: learn-claude-code s10_system_prompt. Extended with conditional
evaluation, priority ordering, and deterministic caching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any


@dataclass
class PromptSection:
    """A named section of the system prompt with optional activation condition.

    Args:
        name: Unique section identifier (e.g. "platform", "user-claude-md").
        content: The prompt text for this section.
        condition: Optional callable returning bool. If None, always included.
        priority: Lower values appear first in assembled output.

    Usage::

        section = PromptSection(
            name="platform",
            content="OS: Windows | Shell: cmd.exe",
            priority=10,
        )
    """

    name: str
    content: str
    condition: Callable[[], bool] | None = None
    priority: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


# Standard section priorities (lower = earlier in output)
PRIORITY_PLATFORM = 0
PRIORITY_USER_CLAUDE_MD = 10
PRIORITY_PROJECT_CLAUDE_MD = 20
PRIORITY_MEMORY_INDEX = 30
PRIORITY_SKILLS_CATALOG = 40
PRIORITY_PERMISSION_MODE = 50
PRIORITY_TOOL_CATALOG = 60
PRIORITY_MCP_STATUS = 70
PRIORITY_BASE_PROMPT = 100


class SystemPrompt:
    """Registry of PromptSections assembled into the final system prompt.

    Usage::

        sp = SystemPrompt()
        sp.register(PromptSection("base", "You are a coding agent.", priority=100))
        prompt = sp.assemble()  # → "You are a coding agent."
    """

    def __init__(self):
        self._sections: list[PromptSection] = []
        self._cache_key: str | None = None
        self._cache_value: str | None = None

    # ── Registration ─────────────────────────────────────────────

    def register(self, section: PromptSection) -> None:
        """Add a section. Replaces any existing section with the same name."""
        self._sections = [
            s for s in self._sections if s.name != section.name
        ]
        self._sections.append(section)
        self._sections.sort(key=lambda s: (s.priority, s.name))
        self._invalidate_cache()

    def unregister(self, name: str) -> bool:
        """Remove a section by name. Returns True if removed."""
        before = len(self._sections)
        self._sections = [s for s in self._sections if s.name != name]
        if len(self._sections) < before:
            self._invalidate_cache()
            return True
        return False

    def get(self, name: str) -> PromptSection | None:
        """Retrieve a section by name."""
        for s in self._sections:
            if s.name == name:
                return s
        return None

    # ── Assembly ─────────────────────────────────────────────────

    def assemble(self) -> str:
        """Evaluate all conditions, concatenate active sections.

        Sections whose condition returns False are omitted.
        Result is cached until register() or unregister() is called.

        Conditions are evaluated exactly once per call: active states are
        computed for the cache key and reused during assembly.
        """
        # Compute active states once (side-effect: conditions fire exactly once)
        active_map: dict[str, bool] = {}
        for s in self._sections:
            active_map[s.name] = s.condition() if s.condition else True

        cache_key = json.dumps(
            [f"{s.name}:{active_map[s.name]}" for s in self._sections],
            sort_keys=True,
        )
        if cache_key == self._cache_key and self._cache_value is not None:
            return self._cache_value

        parts = [
            s.content
            for s in self._sections
            if active_map.get(s.name, True) and s.content
        ]

        result = "\n\n".join(parts)
        self._cache_key = cache_key
        self._cache_value = result
        return result

    # ── Introspection ────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._sections)

    def list_names(self) -> list[str]:
        return [s.name for s in self._sections]

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self._cache_value = None
