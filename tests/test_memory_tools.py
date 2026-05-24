"""Tests for bidirectional memory tools."""
import pytest
import tempfile
from pathlib import Path
from deepagent.memory.store import MemoryStore
from deepagent.memory.models import MemoryEntry


@pytest.fixture
def store():
    d = tempfile.TemporaryDirectory()
    s = MemoryStore(d.name)
    yield s
    d.cleanup()


def test_store_write_entry_creates_file(store):
    entry = MemoryEntry(
        name="test-entry", description="A test", memory_type="user",
        content="Body content.", hit_count=0, confidence=1.0,
    )
    store.write_entry(entry)
    filepath = store._root / "test-entry.md"
    assert filepath.exists()
    content = filepath.read_text()
    assert "name: test-entry" in content
    assert "hit_count: 0" in content
    assert "confidence: 1.0" in content


def test_store_increment_hits(store):
    entry = MemoryEntry(
        name="frequent", description="Frequently used", memory_type="user",
        content="Important.", hit_count=0, confidence=1.0,
    )
    store.write_entry(entry)

    store.increment_hits("frequent")
    loaded = store.read_entry("frequent")
    assert loaded is not None
    assert loaded.hit_count == 1


def test_store_increment_hits_unknown_noop(store):
    store.increment_hits("nonexistent")  # should not crash


def test_store_delete_entry_removes_from_index(store):
    entry = MemoryEntry(
        name="temp", description="Temporary", memory_type="user", content="x",
    )
    store.write_entry(entry)
    assert store.read_entry("temp") is not None

    store.delete_entry("temp")
    assert store.read_entry("temp") is None
    assert "temp" not in store.get_system_context()


def test_store_memory_count(store):
    assert store.memory_count == 0
    for i in range(5):
        store.write_entry(MemoryEntry(
            name=f"entry-{i}", description=f"Desc {i}",
            memory_type="user", content=f"Body {i}.",
        ))
    assert store.memory_count == 5


def test_store_needs_consolidation(store):
    assert not store.needs_consolidation()
    for i in range(11):
        store.write_entry(MemoryEntry(
            name=f"entry-{i}", description=f"Desc {i}",
            memory_type="user", content=f"Body {i}.",
        ))
    assert store.needs_consolidation()


def test_store_extract_memories_noop(store):
    store.extract_memories([])


def test_store_conflict_detection(store):
    a = MemoryEntry(
        name="pref-concise", description="User prefers concise answers in chat",
        memory_type="feedback", content="Keep responses short.",
    )
    b = MemoryEntry(
        name="pref-verbose", description="User prefers detailed answers in chat",
        memory_type="feedback", content="Give long explanations.",
    )
    store.write_entry(a)
    store.write_entry(b)

    conflicts = store.detect_conflicts()
    assert len(conflicts) > 0


def test_store_no_conflicts_different_types(store):
    a = MemoryEntry(
        name="pref-style", description="Code style preference for the project",
        memory_type="feedback", content="Use 4-space indents.",
    )
    b = MemoryEntry(
        name="proj-style", description="Code style preference for the project",
        memory_type="project", content="Project uses tabs.",
    )
    store.write_entry(a)
    store.write_entry(b)

    conflicts = store.detect_conflicts()
    # Different types, so no conflict
    assert len(conflicts) == 0


def test_memory_tools_registration():
    """Verify create_memory_tools registers all 3 tools."""
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.memory_tools import create_memory_tools

    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    reg = ToolRegistry()
    create_memory_tools(reg, store)

    assert "create_memory" in reg.list_names()
    assert "update_memory" in reg.list_names()
    assert "delete_memory" in reg.list_names()
    d.cleanup()


@pytest.mark.asyncio
async def test_create_memory_tool():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.memory_tools import create_memory_tools

    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    reg = ToolRegistry()
    create_memory_tools(reg, store)

    tool = reg.get("create_memory")
    result = await tool(
        name="test-memory",
        description="A test memory",
        memory_type="user",
        content="This is the body.",
    )
    assert result["success"] is True
    assert "test-memory" in result["content"]
    d.cleanup()


@pytest.mark.asyncio
async def test_create_memory_invalid_type():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.memory_tools import create_memory_tools

    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    reg = ToolRegistry()
    create_memory_tools(reg, store)

    tool = reg.get("create_memory")
    result = await tool(
        name="bad-type",
        description="Test",
        memory_type="invalid",
        content="Body.",
    )
    assert result["success"] is False
    assert "Invalid type" in result["error"]
    d.cleanup()


@pytest.mark.asyncio
async def test_update_memory_tool():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.memory_tools import create_memory_tools

    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    # Pre-create a memory
    store.write_entry(MemoryEntry(
        name="to-update", description="Original", memory_type="user",
        content="Original content.",
    ))

    reg = ToolRegistry()
    create_memory_tools(reg, store)

    tool = reg.get("update_memory")
    result = await tool(name="to-update", content="Updated content.")
    assert result["success"] is True
    loaded = store.read_entry("to-update")
    assert loaded.content == "Updated content."
    d.cleanup()


@pytest.mark.asyncio
async def test_delete_memory_tool():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.memory_tools import create_memory_tools

    d = tempfile.TemporaryDirectory()
    store = MemoryStore(d.name)
    store.write_entry(MemoryEntry(
        name="to-delete", description="Will be deleted", memory_type="user",
        content="Delete me.",
    ))

    reg = ToolRegistry()
    create_memory_tools(reg, store)

    tool = reg.get("delete_memory")
    result = await tool(name="to-delete")
    assert result["success"] is True
    assert store.read_entry("to-delete") is None
    d.cleanup()
