import tempfile
import os
import pytest
from deepagent.memory.models import MemoryEntry
from deepagent.memory.store import MemoryStore


# ── MemoryEntry parse / serialize ─────────────────────────────────

def test_parse_frontmatter_basic():
    raw = """---
name: user-role
description: User is a senior engineer
metadata:
  type: user
---

Prefers concise answers. Uses Python daily."""
    entry = MemoryEntry.parse_frontmatter(raw)
    assert entry is not None
    assert entry.name == "user-role"
    assert entry.description == "User is a senior engineer"
    assert entry.memory_type == "user"
    assert "Prefers concise answers" in entry.content


def test_parse_frontmatter_feedback():
    raw = """---
name: no-summaries
description: Don't add summaries
metadata:
  type: feedback
---

Stop adding verbose summaries at end of responses.

**Why:** User reads the diff directly.
**How to apply:** End responses with next steps only when needed."""
    entry = MemoryEntry.parse_frontmatter(raw)
    assert entry is not None
    assert entry.name == "no-summaries"
    assert entry.memory_type == "feedback"


def test_parse_frontmatter_project():
    raw = """---
name: merge-freeze
description: Merge freeze active
metadata:
  type: project
---

Merge freeze starts 2026-03-05 for mobile release.

**Why:** Mobile team cutting release branch.
**How to apply:** Flag non-critical PR work after that date."""
    entry = MemoryEntry.parse_frontmatter(raw)
    assert entry is not None
    assert entry.memory_type == "project"


def test_parse_frontmatter_reference():
    raw = """---
name: linear-project
description: Pipeline bugs tracked in Linear
metadata:
  type: reference
---

Pipeline bugs are tracked in Linear project "INGEST"."""
    entry = MemoryEntry.parse_frontmatter(raw)
    assert entry is not None
    assert entry.memory_type == "reference"


def test_parse_frontmatter_no_frontmatter():
    raw = "Just plain text, no frontmatter."
    assert MemoryEntry.parse_frontmatter(raw) is None


def test_parse_frontmatter_no_name():
    raw = """---
description: Missing name field
metadata:
  type: user
---

content here"""
    assert MemoryEntry.parse_frontmatter(raw) is None


def test_to_frontmatter_roundtrip():
    entry = MemoryEntry(
        name="test-entry",
        description="A test memory",
        memory_type="user",
        content="Body content.",
    )
    serialized = entry.to_frontmatter()
    parsed = MemoryEntry.parse_frontmatter(serialized)
    assert parsed is not None
    assert parsed.name == entry.name
    assert parsed.description == entry.description
    assert parsed.memory_type == entry.memory_type
    assert parsed.content == entry.content


def test_index_line():
    entry = MemoryEntry(
        name="my-entry",
        description="Something useful",
        memory_type="user",
        content="Body.",
    )
    line = entry.index_line
    assert "[my-entry]" in line
    assert "(my-entry.md)" in line
    assert "Something useful" in line


# ── MemoryStore ───────────────────────────────────────────────────


@pytest.fixture
def tmp_store():
    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    yield store
    d.cleanup()


def test_store_starts_empty(tmp_store):
    entries = tmp_store.load_index()
    assert entries == []
    assert tmp_store.get_system_context() == ""


def test_write_and_read_entry(tmp_store):
    entry = MemoryEntry(
        name="test-memory",
        description="A test",
        memory_type="user",
        content="Body text.",
    )
    tmp_store.write_entry(entry)

    # Read back
    loaded = tmp_store.read_entry("test-memory")
    assert loaded is not None
    assert loaded.name == "test-memory"
    assert loaded.description == "A test"
    assert loaded.content == "Body text."


def test_write_entry_creates_index(tmp_store):
    entry = MemoryEntry(
        name="first",
        description="First memory",
        memory_type="user",
        content="Content.",
    )
    tmp_store.write_entry(entry)

    context = tmp_store.get_system_context()
    assert "[first]" in context
    assert "(first.md)" in context
    assert "First memory" in context


def test_write_multiple_entries(tmp_store):
    for i in range(3):
        entry = MemoryEntry(
            name=f"memory-{i}",
            description=f"Description {i}",
            memory_type="user",
            content=f"Content {i}.",
        )
        tmp_store.write_entry(entry)

    entries = tmp_store.load_index()
    assert len(entries) == 3
    names = {e.name for e in entries}
    assert names == {"memory-0", "memory-1", "memory-2"}

    context = tmp_store.get_system_context()
    assert "[memory-0]" in context
    assert "[memory-1]" in context
    assert "[memory-2]" in context


def test_delete_entry(tmp_store):
    entry = MemoryEntry(
        name="to-delete",
        description="Will be deleted",
        memory_type="user",
        content="Delete me.",
    )
    tmp_store.write_entry(entry)
    assert tmp_store.read_entry("to-delete") is not None

    tmp_store.delete_entry("to-delete")
    assert tmp_store.read_entry("to-delete") is None

    context = tmp_store.get_system_context()
    assert "to-delete" not in context


def test_read_entry_not_found(tmp_store):
    assert tmp_store.read_entry("nonexistent") is None


def test_needs_refresh_initially_true(tmp_store):
    assert tmp_store.needs_refresh() is True


def test_needs_refresh_after_load(tmp_store):
    tmp_store.load_index()
    assert tmp_store.needs_refresh() is False


def test_refresh_clears_cache(tmp_store):
    entry = MemoryEntry(
        name="cached",
        description="Cached entry",
        memory_type="user",
        content="Cached.",
    )
    tmp_store.write_entry(entry)
    tmp_store.load_index()  # cache it

    # Manually delete the file behind the store's back
    os.remove(os.path.join(str(tmp_store._root), "cached.md"))
    # Rebuild index to reflect deletion
    tmp_store._rebuild_index()

    # Cache still has old entry
    assert tmp_store.read_entry("cached") is not None

    # Refresh should clear
    tmp_store.refresh()
    assert tmp_store.read_entry("cached") is None


def test_get_all_entries(tmp_store):
    for name in ["a", "b", "c"]:
        entry = MemoryEntry(
            name=name, description=f"Desc {name}", memory_type="user", content=f"Body {name}."
        )
        tmp_store.write_entry(entry)

    all_entries = tmp_store.get_all_entries()
    names = {e.name for e in all_entries}
    assert names == {"a", "b", "c"}


def test_store_with_real_format(tmp_store):
    """Verify the store handles the exact Claude Code frontmatter format."""
    raw = """---
name: user-role
description: Senior engineer focused on observability
metadata:
  type: user
---

Prefers concise responses. Uses Python and Go daily."""
    entry = MemoryEntry.parse_frontmatter(raw, "user-role.md")
    assert entry is not None
    tmp_store.write_entry(entry)

    # Check file content on disk
    filepath = tmp_store._root / "user-role.md"
    written = filepath.read_text("utf-8")
    assert "name: user-role" in written
    assert "metadata:" in written
    assert "type: user" in written
    assert "Prefers concise responses" in written
