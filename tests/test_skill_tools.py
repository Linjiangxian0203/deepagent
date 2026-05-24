"""Tests for skill tool registration and execution."""
import pytest
from deepagent.core.skills import SkillRegistry


@pytest.fixture
def skill_registry(tmp_path):
    reg = SkillRegistry()
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: A test skill for checking loading
---

This is the full skill body content.
""")
    reg.scan(str(tmp_path / "skills"))
    return reg


def test_skill_registry_scan_creates_catalog(skill_registry):
    assert skill_registry.count == 1
    assert "test-skill" in skill_registry.names()
    catalog = skill_registry.get_catalog()
    assert "test-skill" in catalog
    assert "A test skill" in catalog


def test_skill_registry_load_returns_sourced_content(skill_registry):
    content = skill_registry.load("test-skill")
    assert content is not None
    assert "[Skill: test-skill]" in content
    assert "full skill body content" in content


def test_skill_registry_load_unknown_returns_none(skill_registry):
    assert skill_registry.load("nonexistent") is None


def test_skill_registry_empty_dir():
    import tempfile
    d = tempfile.TemporaryDirectory()
    reg = SkillRegistry()
    assert reg.scan(d.name) == 0
    assert reg.get_catalog() == ""
    d.cleanup()


def test_skill_tool_creation():
    """Verify create_skill_tools registers load_skill into a ToolRegistry."""
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.skill_tools import create_skill_tools

    reg = ToolRegistry()
    skill_reg = SkillRegistry()
    create_skill_tools(reg, skill_reg)

    assert "load_skill" in reg.list_names()
    tool = reg.get("load_skill")
    assert tool is not None
    assert tool.tool_safety_level.value == "readonly"


@pytest.mark.asyncio
async def test_load_skill_tool_returns_content(tmp_path):
    """End-to-end: register tool and call it."""
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.skill_tools import create_skill_tools

    skill_dir = tmp_path / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("""---
name: demo-skill
description: A demonstration skill
---

Skill instructions here.
""")

    skill_reg = SkillRegistry()
    skill_reg.scan(str(tmp_path / "skills"))

    tool_reg = ToolRegistry()
    create_skill_tools(tool_reg, skill_reg)

    tool = tool_reg.get("load_skill")
    result = await tool(name="demo-skill")
    assert result["success"] is True
    assert "[Skill: demo-skill]" in result["content"]
    assert "Skill instructions here" in result["content"]


@pytest.mark.asyncio
async def test_load_skill_tool_unknown_returns_error(tmp_path):
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.skill_tools import create_skill_tools

    skill_reg = SkillRegistry()
    skill_reg.scan(str(tmp_path))

    tool_reg = ToolRegistry()
    create_skill_tools(tool_reg, skill_reg)

    tool = tool_reg.get("load_skill")
    result = await tool(name="nonexistent")
    assert result["success"] is False
    assert "Skill not found" in result["error"]
