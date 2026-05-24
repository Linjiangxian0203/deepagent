"""MemoryStore — persistent long-term memory backed by Markdown files.

Format: one .md file per memory entry with YAML frontmatter, plus a MEMORY.md index.
This matches the Claude Code memory system format.
"""

import os
from pathlib import Path

from deepagent.memory.models import MemoryEntry


class MemoryStore:
    """Manages persistent memories stored as .md files in a directory.

    The directory contains:
    - MEMORY.md — index listing all memories (one line per entry)
    - *.md — individual memory files with YAML frontmatter

    Usage:
        store = MemoryStore("/path/to/memory")
        context = store.get_system_context()  # inject into system prompt
        entry = store.read_entry("some-slug")  # read specific memory
    """

    def __init__(self, root: str):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "MEMORY.md"
        self._cache: dict[str, MemoryEntry] = {}
        self._loaded = False

    # ── reading ───────────────────────────────────────────────────

    def load_index(self) -> list[MemoryEntry]:
        """Parse MEMORY.md and return all entries listed in the index."""
        self._loaded = True
        if not self._index_path.exists():
            return []
        entries: list[MemoryEntry] = []
        for line in self._index_path.read_text("utf-8").splitlines():
            line = line.strip()
            if not line.startswith("- ["):
                continue
            # Format: - [Display Name](file.md) — description
            # Extract the file part from the link
            link_start = line.find("](") + 2
            link_end = line.find(")", link_start)
            if link_start < 2 or link_end == -1:
                continue
            filename = line[link_start:link_end]
            entry = self._read_file(filename)
            if entry is not None:
                entries.append(entry)
                self._cache[entry.name] = entry
        return entries

    def read_entry(self, name: str) -> MemoryEntry | None:
        """Read a specific memory entry by its slug name."""
        if name in self._cache:
            return self._cache[name]
        entry = self._read_file(f"{name}.md")
        if entry is not None:
            self._cache[entry.name] = entry
        return entry

    def _read_file(self, filename: str) -> MemoryEntry | None:
        """Parse a single memory .md file."""
        filepath = self._root / filename
        if not filepath.exists():
            return None
        raw = filepath.read_text("utf-8")
        return MemoryEntry.parse_frontmatter(raw, str(filepath))

    # ── writing ───────────────────────────────────────────────────

    def write_entry(self, entry: MemoryEntry) -> None:
        """Write or update a memory entry. Updates both the .md file and MEMORY.md index."""
        filepath = self._root / f"{entry.name}.md"
        filepath.write_text(entry.to_frontmatter(), "utf-8")
        entry.file_path = str(filepath)
        self._cache[entry.name] = entry
        self._rebuild_index()

    def delete_entry(self, name: str) -> bool:
        """Delete a memory entry by name. Returns True if deleted."""
        filepath = self._root / f"{name}.md"
        if filepath.exists():
            filepath.unlink()
        self._cache.pop(name, None)
        self._rebuild_index()
        return True

    def _rebuild_index(self) -> None:
        """Rebuild MEMORY.md from all .md files in the directory."""
        entries: list[MemoryEntry] = []
        for f in sorted(self._root.glob("*.md")):
            if f.name == "MEMORY.md":
                continue
            raw = f.read_text("utf-8")
            entry = MemoryEntry.parse_frontmatter(raw, str(f))
            if entry is not None:
                entries.append(entry)
        lines = [e.index_line for e in entries]
        self._index_path.write_text("\n".join(lines) + "\n", "utf-8")

    # ── system prompt injection ───────────────────────────────────

    def get_system_context(self) -> str:
        """Return MEMORY.md content for injection into the system prompt.

        The agent reads the index and can then call read_entry() to get details.
        """
        if self._index_path.exists():
            return self._index_path.read_text("utf-8").strip()
        return ""

    def get_all_entries(self) -> list[MemoryEntry]:
        """Return all cached entries, loading from disk if needed."""
        if not self._loaded:
            self.load_index()
        return list(self._cache.values())

    def needs_refresh(self) -> bool:
        """Check if the in-memory cache needs a refresh from disk."""
        return not self._loaded

    def refresh(self) -> None:
        """Force reload from disk."""
        self._cache.clear()
        self._loaded = False
        self.load_index()

    # ── hit tracking ───────────────────────────────────────────────

    def increment_hits(self, name: str) -> None:
        """Increment the hit_count on a memory entry."""
        entry = self.read_entry(name)
        if entry is not None:
            entry.hit_count += 1
            self.write_entry(entry)

    # ── consolidation ──────────────────────────────────────────────

    CONSOLIDATION_THRESHOLD = 10

    @property
    def memory_count(self) -> int:
        """Count all .md files (excluding MEMORY.md)."""
        if not self._root.exists():
            return 0
        return len([f for f in self._root.glob("*.md") if f.name != "MEMORY.md"])

    def needs_consolidation(self) -> bool:
        return self.memory_count >= self.CONSOLIDATION_THRESHOLD

    def extract_memories(self, messages: list) -> None:
        """Placeholder for LLM-driven memory extraction.

        Analyzes conversation messages for new facts/preferences/constraints.
        Real implementation calls LLM to determine if new memories should be
        created. For now, this is a no-op — the agent uses create_memory tool
        explicitly.
        """
        pass

    # ── conflict detection ─────────────────────────────────────────

    def detect_conflicts(self) -> list[tuple[str, str]]:
        """Detect potentially conflicting memories by comparing descriptions.

        Returns list of (name_a, name_b) pairs that may conflict.
        Uses simple keyword overlap for now.
        """
        entries = self.get_all_entries()
        if len(entries) < 2:
            return []

        conflicts = []
        for i in range(len(entries)):
            a_words = set(entries[i].description.lower().split())
            for j in range(i + 1, len(entries)):
                b_words = set(entries[j].description.lower().split())
                overlap = len(a_words & b_words) / max(len(a_words | b_words), 1)
                if overlap > 0.5 and entries[i].memory_type == entries[j].memory_type:
                    conflicts.append((entries[i].name, entries[j].name))
        return conflicts
