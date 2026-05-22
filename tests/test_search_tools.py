# tests/test_search_tools.py
import os
import tempfile
import pytest
from deepagent.tools.search_tools import create_search_tools
from deepagent.tools.protocol import SafetyLevel


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        # Create test files
        with open(os.path.join(d, "main.py"), "w") as f:
            f.write("import os\n\ndef hello():\n    print('hello world')\n\ndef goodbye():\n    print('goodbye')\n")
        with open(os.path.join(d, "utils.py"), "w") as f:
            f.write("def helper():\n    return 'hello from helper'\n")
        with open(os.path.join(d, "data.txt"), "w") as f:
            f.write("hello data\nline 2\n")
        os.makedirs(os.path.join(d, "subdir"))
        with open(os.path.join(d, "subdir", "nested.py"), "w") as f:
            f.write("# nested file\nprint('hello nested')\n")
        yield d


@pytest.fixture
def registry_and_tools(tmp_dir):
    from deepagent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    tools = create_search_tools(registry, safe_root=tmp_dir)
    return registry, tools, tmp_dir


def test_search_tools_are_registered(registry_and_tools):
    registry, tools, _ = registry_and_tools
    names = registry.list_names()
    assert "grep" in names
    assert "glob" in names


def test_search_tools_safety_levels(registry_and_tools):
    registry, tools, _ = registry_and_tools
    assert registry.get("grep").tool_safety_level == SafetyLevel.READONLY
    assert registry.get("glob").tool_safety_level == SafetyLevel.READONLY


@pytest.mark.asyncio
async def test_grep_finds_matches(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="hello", path=tmp)
    assert result["success"] is True
    assert "main.py" in result["content"]
    assert "utils.py" in result["content"]
    assert "data.txt" in result["content"]


@pytest.mark.asyncio
async def test_grep_with_glob_filter(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="hello", path=tmp, glob="*.py")
    assert result["success"] is True
    assert "main.py" in result["content"]
    assert "utils.py" in result["content"]
    assert "data.txt" not in result["content"]


@pytest.mark.asyncio
async def test_grep_no_matches(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="nonexistent_pattern_xyz", path=tmp)
    assert result["success"] is True
    assert result["metadata"].get("match_count", -1) == 0


@pytest.mark.asyncio
async def test_glob_finds_files(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("glob")
    result = await tool(pattern="**/*.py", path=tmp)
    assert result["success"] is True
    assert "main.py" in result["content"]
    assert "utils.py" in result["content"]
    assert "nested.py" in result["content"]


@pytest.mark.asyncio
async def test_glob_no_matches(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("glob")
    result = await tool(pattern="*.js", path=tmp)
    assert result["success"] is True
    assert result["metadata"].get("file_count", -1) == 0


@pytest.mark.asyncio
async def test_grep_outside_safe_root(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="test", path="/etc")
    assert result["success"] is False
    assert "outside" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_glob_outside_safe_root(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("glob")
    result = await tool(pattern="*", path="/etc")
    assert result["success"] is False
    assert "outside" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_grep_invalid_regex(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="[unclosed", path=tmp)
    assert result["success"] is False
    assert "regex" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_grep_path_not_found(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("grep")
    result = await tool(pattern="test", path=os.path.join(tmp, "nonexistent_dir"))
    assert result["success"] is False
    assert "not found" in result.get("error", "").lower()


@pytest.mark.asyncio
async def test_glob_path_not_found(registry_and_tools):
    registry, tools, tmp = registry_and_tools
    tool = registry.get("glob")
    result = await tool(pattern="*", path=os.path.join(tmp, "nonexistent_dir"))
    assert result["success"] is False
    assert "not found" in result.get("error", "").lower()
