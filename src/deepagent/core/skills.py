"""Skill loading — two-level on-demand SKILL.md registry.

Level 1 (Catalog): scan() at startup parses all frontmatters, extracts
name+description. Injected into system prompt. Cheap.

Level 2 (Content): load(name) returns full SKILL.md body. On-demand, expensive.

Supports hot reload via mtime checking and Claude Code compatible
directory structure.

Reference: learn-claude-code s07_skill_loading.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Skill data ─────────────────────────────────────────────────────


@dataclass
class SkillEntry:
    name: str
    description: str
    file_path: str
    content: str = ""  # populated on Level 2 load
    loaded: bool = False

    @property
    def catalog_line(self) -> str:
        return f"- {self.name}: {self.description}"


# ═══════════════════════════════════════════════════════════════════


class SkillRegistry:
    """Two-level skill registry.

    Usage::

        registry = SkillRegistry()
        count = registry.scan("./skills")
        catalog = registry.get_catalog()        # Level 1: names + descriptions
        content = registry.load("code-review")  # Level 2: full body

    Skills are stored as directories under *skills_dir*:
        skills/
          code-review/
            SKILL.md         ← required
            references/      ← optional
            scripts/         ← optional

    SKILL.md format:
        ---
        name: code-review
        description: Review code changes for quality and security
        ---

        [Skill body content here...]
    """

    def __init__(self):
        self._skills: dict[str, SkillEntry] = {}
        self._skills_dir: str = ""
        self._mtimes: dict[str, float] = {}

    # ── Level 1: Catalog scan ──────────────────────────────────────

    def scan(self, skills_dir: str | Path) -> int:
        """Scan all SKILL.md files under *skills_dir*. Returns count loaded.

        Silently skips invalid or unparseable files. Call at startup,
        then call get_catalog() to inject into the system prompt.
        """
        self._skills_dir = str(skills_dir)
        root = Path(skills_dir)
        if not root.is_dir():
            return 0

        count = 0
        for skill_dir in sorted(root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.is_file():
                continue

            entry = self._parse_frontmatter(skill_md)
            if entry is None:
                logger.warning("Skipping unparseable skill: %s", skill_md)
                continue

            self._skills[entry.name] = entry
            self._mtimes[entry.name] = skill_md.stat().st_mtime
            count += 1

        return count

    def get_catalog(self) -> str:
        """Return a Level 1 catalog string for system prompt injection.

        One line per skill: name + one-line description.
        """
        if not self._skills:
            return ""
        lines = ["Available skills (use load_skill to read full content):"]
        for name in sorted(self._skills):
            entry = self._skills[name]
            lines.append(f"  /{entry.name} — {entry.description}")
        return "\n".join(lines)

    # ── Level 2: Full content load ─────────────────────────────────

    def load(self, name: str) -> str | None:
        """Load full SKILL.md body content for *name*. Returns None if not found.

        Source-tags the content with [Skill: name] prefix so the LLM knows
        which skill provided the instructions.
        """
        entry = self._skills.get(name)
        if entry is None:
            return None

        if not entry.loaded:
            try:
                raw = Path(entry.file_path).read_text("utf-8")
                # Split off frontmatter (between first and second ---)
                body = self._extract_body(raw)
                entry.content = body
                entry.loaded = True
            except OSError as e:
                logger.warning("Failed to load skill %s: %s", name, e)
                return None

        return f"[Skill: {entry.name}]\n{entry.content}"

    # ── Hot reload ──────────────────────────────────────────────────

    def check_hot_reload(self) -> int:
        """Check all scanned skills for mtime changes. Reloads changed ones.

        Returns count of reloaded skills. Call periodically or before
        skill-related operations.
        """
        reloaded = 0
        for name, entry in self._skills.items():
            try:
                mtime = Path(entry.file_path).stat().st_mtime
            except OSError:
                continue
            if mtime > self._mtimes.get(name, 0):
                entry.loaded = False
                entry.content = ""
                self._mtimes[name] = mtime
                reloaded += 1
                logger.info("Hot-reloaded skill: %s", name)
        return reloaded

    # ── Introspection ──────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    # ── Internal helpers ───────────────────────────────────────────

    @staticmethod
    def _parse_frontmatter(path: Path) -> SkillEntry | None:
        """Parse YAML frontmatter from a SKILL.md file. Returns None on failure."""
        raw = path.read_text("utf-8")
        if not raw.startswith("---"):
            return None
        end_idx = raw.find("---", 3)
        if end_idx == -1:
            return None

        frontmatter_text = raw[3:end_idx].strip()
        fields: dict[str, str] = {}
        for line in frontmatter_text.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()

        name = fields.get("name", "")
        if not name:
            return None

        return SkillEntry(
            name=name,
            description=fields.get("description", path.parent.name),
            file_path=str(path),
        )

    @staticmethod
    def _extract_body(raw: str) -> str:
        """Extract markdown body (after frontmatter)."""
        if not raw.startswith("---"):
            return raw
        end_idx = raw.find("---", 3)
        if end_idx == -1:
            return raw
        return raw[end_idx + 3:].strip()
