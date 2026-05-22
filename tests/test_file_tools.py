# tests/test_file_tools.py
import os
import tempfile
import pytest
from deepagent.tools.file_tools import create_file_tools
from deepagent.tools.protocol import SafetyLevel


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def registry_and_tools(tmp_dir):
    """Create registry with file_tools, cwd set to temp directory."""
    from deepagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    tools = create_file_tools(registry, safe_root=tmp_dir)
    return registry, tools, tmp_dir


def test_file_tools_are_registered(registry_and_tools):
    registry, tools, _ = registry_and_tools
    names = registry.list_names()
    assert "read_file" in names
    assert "write_file" in names
    assert "edit_file" in names


def test_file_tools_safety_levels(registry_and_tools):
    registry, tools, _ = registry_and_tools
    assert registry.get("read_file").tool_safety_level == SafetyLevel.READONLY
    assert registry.get("write_file").tool_safety_level == SafetyLevel.WRITE
    assert registry.get("edit_file").tool_safety_level == SafetyLevel.WRITE


@pytest.mark.asyncio
async def test_read_file(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "test.txt")
    with open(filepath, "w") as f:
        f.write("line1\nline2\nline3\nline4\nline5")

    tool = registry.get("read_file")
    result = await tool(path=filepath)
    assert result["success"] is True
    assert "line1" in result["content"]
    assert "line5" in result["content"]


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "test.txt")
    with open(filepath, "w") as f:
        f.write("line1\nline2\nline3\nline4\nline5")

    tool = registry.get("read_file")
    result = await tool(path=filepath, offset=2, limit=2)
    assert result["success"] is True
    assert "line2" in result["content"]
    assert "line3" in result["content"]
    assert "line4" not in result["content"]


@pytest.mark.asyncio
async def test_read_file_not_found(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("read_file")
    result = await tool(path=os.path.join(tmp, "nonexistent.txt"))
    assert result["success"] is False
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_read_file_outside_safe_root(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("read_file")
    result = await tool(path="/etc/passwd")
    assert result["success"] is False
    assert "outside" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_write_file(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "output.txt")

    tool = registry.get("write_file")
    result = await tool(path=filepath, content="hello world")
    assert result["success"] is True

    with open(filepath) as f:
        assert f.read() == "hello world"


@pytest.mark.asyncio
async def test_write_file_outside_safe_root(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("write_file")
    result = await tool(path="/etc/malicious.txt", content="bad")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_edit_file_exact_replace(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "edit.txt")
    with open(filepath, "w") as f:
        f.write("hello world\nfoo bar\n")

    tool = registry.get("edit_file")
    result = await tool(path=filepath, old_string="hello world", new_string="hi there")
    assert result["success"] is True

    with open(filepath) as f:
        content = f.read()
    assert "hi there" in content
    assert "hello world" not in content


@pytest.mark.asyncio
async def test_edit_file_old_string_not_found(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "edit.txt")
    with open(filepath, "w") as f:
        f.write("actual content")

    tool = registry.get("edit_file")
    result = await tool(path=filepath, old_string="not in file", new_string="replacement")
    assert result["success"] is False


@pytest.mark.asyncio
async def test_edit_file_duplicate_string(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    filepath = os.path.join(tmp, "dup.txt")
    with open(filepath, "w") as f:
        f.write("hello\nhello\n")

    tool = registry.get("edit_file")
    result = await tool(path=filepath, old_string="hello", new_string="hi")
    assert result["success"] is False
