"""Tests for SystemPrompt (Phase 1).

Covers: section registration, conditional assembly, priority ordering,
deterministic caching, unregister, and integration patterns.
"""

import pytest

from deepagent.core.system_prompt import (
    SystemPrompt,
    PromptSection,
    PRIORITY_PLATFORM,
    PRIORITY_BASE_PROMPT,
)


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def registry():
    return SystemPrompt()


# ── Basic assembly ─────────────────────────────────────────────────


def test_empty_registry_assembles_empty(registry):
    assert registry.assemble() == ""


def test_single_section(registry):
    registry.register(PromptSection("base", "You are a coding agent.", priority=100))
    assert registry.assemble() == "You are a coding agent."


def test_multiple_sections_joined_by_double_newline(registry):
    registry.register(PromptSection("a", "First section", priority=10))
    registry.register(PromptSection("b", "Second section", priority=20))
    result = registry.assemble()
    assert result == "First section\n\nSecond section"


# ── Priority ordering ─────────────────────────────────────────────


def test_sections_sorted_by_priority(registry):
    registry.register(PromptSection("c", "C", priority=90))
    registry.register(PromptSection("a", "A", priority=10))
    registry.register(PromptSection("b", "B", priority=50))
    result = registry.assemble()
    assert result == "A\n\nB\n\nC"


def test_same_priority_sorted_by_name(registry):
    registry.register(PromptSection("banana", "B", priority=50))
    registry.register(PromptSection("apple", "A", priority=50))
    result = registry.assemble()
    assert result == "A\n\nB"


# ── Conditions ────────────────────────────────────────────────────


def test_condition_false_omits_section(registry):
    registry.register(PromptSection("a", "Visible", condition=lambda: True, priority=10))
    registry.register(PromptSection("b", "Hidden", condition=lambda: False, priority=20))
    registry.register(PromptSection("c", "Also visible", priority=30))
    result = registry.assemble()
    assert "Visible" in result
    assert "Hidden" not in result
    assert "Also visible" in result


def test_condition_none_always_included(registry):
    registry.register(PromptSection("x", "Always", condition=None, priority=10))
    result = registry.assemble()
    assert "Always" in result


def test_condition_evaluated_on_each_assemble(registry):
    counter = [0]

    def dynamic():
        counter[0] += 1
        return counter[0] <= 2  # True first 2 calls only

    registry.register(PromptSection("dyn", "Dynamic", condition=dynamic, priority=10))
    assert "Dynamic" in registry.assemble()
    assert "Dynamic" in registry.assemble()
    assert "Dynamic" not in registry.assemble()  # 3rd call returns False


# ── Caching ────────────────────────────────────────────────────────


def test_cache_returns_same_string_object(registry):
    """Consecutive calls with unchanged state return cached result."""
    registry.register(PromptSection("a", "Cached", priority=10))
    a = registry.assemble()
    b = registry.assemble()
    assert a is b  # same object = cache hit


def test_cache_invalidated_on_register(registry):
    registry.register(PromptSection("a", "First", priority=10))
    first = registry.assemble()

    registry.register(PromptSection("a", "Second", priority=10))
    second = registry.assemble()

    assert first != second
    assert second == "Second"


def test_cache_invalidated_on_unregister(registry):
    registry.register(PromptSection("a", "A", priority=10))
    registry.assemble()

    registry.unregister("a")
    assert registry.assemble() == ""


def test_cache_handles_condition_change(registry):
    flag = [True]
    registry.register(PromptSection("x", "Present", condition=lambda: flag[0], priority=10))

    assert "Present" in registry.assemble()
    flag[0] = False
    # Cache key includes condition result; new condition = new cache key
    assert "Present" not in registry.assemble()


# ── Registration / unregistration ─────────────────────────────────


def test_register_replaces_same_name(registry):
    registry.register(PromptSection("a", "First", priority=10))
    registry.register(PromptSection("a", "Second", priority=10))
    assert registry.count == 1
    assert registry.assemble() == "Second"


def test_unregister_removes_section(registry):
    registry.register(PromptSection("a", "A", priority=10))
    registry.register(PromptSection("b", "B", priority=20))
    assert registry.count == 2

    removed = registry.unregister("a")
    assert removed is True
    assert registry.count == 1
    assert registry.assemble() == "B"


def test_unregister_nonexistent_returns_false(registry):
    assert registry.unregister("nonexistent") is False


def test_get_returns_section(registry):
    s = PromptSection("a", "Content", priority=10)
    registry.register(s)
    assert registry.get("a") is s


def test_get_nonexistent_returns_none(registry):
    assert registry.get("nonexistent") is None


def test_list_names(registry):
    registry.register(PromptSection("c", "C", priority=90))
    registry.register(PromptSection("a", "A", priority=10))
    registry.register(PromptSection("b", "B", priority=50))
    assert registry.list_names() == ["a", "b", "c"]


# ── Empty content ──────────────────────────────────────────────────


def test_empty_content_section_is_excluded(registry):
    registry.register(PromptSection("a", "", priority=10))
    registry.register(PromptSection("b", "B", priority=20))
    result = registry.assemble()
    assert result == "B"  # empty "a" omitted


# ── Integration: realistic prompt sections ────────────────────────


def test_realistic_prompt_assembly(registry):
    """Simulate the sections that cli/app.py currently builds via string concat."""
    registry.register(PromptSection(
        "platform",
        "[Platform]\nOS: Windows | Shell: cmd.exe",
        priority=0,
    ))
    registry.register(PromptSection(
        "memory",
        "[Memory index]\n- [Active Project](active-project.md) — CLI coding agent",
        condition=lambda: True,
        priority=30,
    ))
    registry.register(PromptSection(
        "base",
        "You are Deepagent, a CLI coding agent that helps with software engineering tasks.",
        priority=100,
    ))

    result = registry.assemble()
    assert "Platform" in result
    assert "Memory index" in result
    assert "Deepagent" in result

    # Verify ordering: platform → memory → base
    pos_platform = result.index("Platform")
    pos_memory = result.index("Memory")
    pos_deepagent = result.index("Deepagent")
    assert pos_platform < pos_memory < pos_deepagent


def test_priority_constants():
    assert PRIORITY_PLATFORM < PRIORITY_BASE_PROMPT


# ── Metadata ───────────────────────────────────────────────────────


def test_section_metadata():
    s = PromptSection("x", "content", metadata={"source": "test", "version": 1})
    assert s.metadata["source"] == "test"
    assert s.metadata["version"] == 1


def test_section_default_metadata():
    s = PromptSection("x", "content")
    assert s.metadata == {}


def test_section_default_condition():
    s = PromptSection("x", "content")
    assert s.condition is None


def test_section_default_priority():
    s = PromptSection("x", "content")
    assert s.priority == 50
