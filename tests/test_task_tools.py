"""Tests for todo_write and task system tools."""
import pytest
import json
from pathlib import Path
from deepagent.core.tasks import TaskManager


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / ".tasks"
    d.mkdir()
    return str(d)


@pytest.fixture
def task_mgr(tasks_dir):
    return TaskManager(tasks_dir)


def test_task_manager_create_task(task_mgr):
    task = task_mgr.create_task(subject="Test task", description="A test")
    assert task.id.startswith("task_")
    assert task.status == "pending"
    loaded = task_mgr.load(task.id)
    assert loaded is not None
    assert loaded.subject == "Test task"


def test_task_manager_list_all(task_mgr):
    task_mgr.create_task(subject="Task A")
    task_mgr.create_task(subject="Task B")
    all_tasks = task_mgr.list_all()
    assert len(all_tasks) == 2


def test_task_manager_claim_and_complete(task_mgr):
    t = task_mgr.create_task(subject="To complete")
    assert task_mgr.claim(t.id) is not None
    assert task_mgr.load(t.id).status == "in_progress"
    result = task_mgr.complete(t.id)
    assert result is not None
    assert task_mgr.load(t.id).status == "completed"


def test_task_manager_dependency_chain(task_mgr):
    a = task_mgr.create_task(subject="Task A")
    b = task_mgr.create_task(subject="Task B", blocked_by=[a.id])
    c = task_mgr.create_task(subject="Task C", blocked_by=[b.id])
    assert not task_mgr.can_start(b.id)
    assert not task_mgr.can_start(c.id)
    task_mgr.claim(a.id)
    task_mgr.complete(a.id)
    assert task_mgr.can_start(b.id)
    task_mgr.claim(b.id)
    task_mgr.complete(b.id)
    assert task_mgr.can_start(c.id)


def test_task_manager_complete_returns_unblocked_info(task_mgr):
    dep = task_mgr.create_task(subject="Dependency")
    blocked = task_mgr.create_task(subject="Blocked task", blocked_by=[dep.id])
    dep.status = "in_progress"
    task_mgr.save(dep)
    result = task_mgr.complete(dep.id)
    assert result is not None
    assert "Unblocked" in result
    assert "Blocked task" in result


def test_create_task_tool():
    """Verify create_task_system_tools registers all 5 tools."""
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_todo_write_tool, create_task_system_tools

    d = tempfile.TemporaryDirectory()
    mgr = TaskManager(d.name)
    reg = ToolRegistry()
    create_todo_write_tool(reg)
    create_task_system_tools(reg, mgr)

    assert "todo_write" in reg.list_names()
    assert "create_task" in reg.list_names()
    assert "list_tasks" in reg.list_names()
    assert "get_task" in reg.list_names()
    assert "claim_task" in reg.list_names()
    assert "complete_task" in reg.list_names()
    d.cleanup()


@pytest.mark.asyncio
async def test_todo_write_success():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_todo_write_tool
    import os

    d = tempfile.TemporaryDirectory()
    original_cwd = os.getcwd()
    os.chdir(d.name)
    try:
        reg = ToolRegistry()
        create_todo_write_tool(reg)
        tool = reg.get("todo_write")
        result = await tool(todos=[
            {"content": "Write tests", "status": "in_progress"},
            {"content": "Implement code", "status": "pending"},
        ])
        assert result["success"] is True
        assert "2 tasks" in result["content"]
        assert "Write tests" in result["content"]
        assert "Implement code" in result["content"]
    finally:
        os.chdir(original_cwd)
        d.cleanup()


@pytest.mark.asyncio
async def test_todo_write_invalid_status():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_todo_write_tool
    import os

    d = tempfile.TemporaryDirectory()
    original_cwd = os.getcwd()
    os.chdir(d.name)
    try:
        reg = ToolRegistry()
        create_todo_write_tool(reg)
        tool = reg.get("todo_write")
        result = await tool(todos=[
            {"content": "Bad todo", "status": "done"},
        ])
        assert result["success"] is False
        assert "invalid status" in result["error"]
    finally:
        os.chdir(original_cwd)
        d.cleanup()


@pytest.mark.asyncio
async def test_todo_write_missing_fields():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_todo_write_tool
    import os

    d = tempfile.TemporaryDirectory()
    original_cwd = os.getcwd()
    os.chdir(d.name)
    try:
        reg = ToolRegistry()
        create_todo_write_tool(reg)
        tool = reg.get("todo_write")
        result = await tool(todos=[
            {"status": "pending"},
        ])
        assert result["success"] is False
        assert "missing" in result["error"]
    finally:
        os.chdir(original_cwd)
        d.cleanup()


@pytest.mark.asyncio
async def test_create_task_tool_integration():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_task_system_tools

    d = tempfile.TemporaryDirectory()
    mgr = TaskManager(d.name)
    reg = ToolRegistry()
    create_task_system_tools(reg, mgr)

    tool = reg.get("create_task")
    result = await tool(subject="Integration test", description="Testing")
    assert result["success"] is True
    assert "Integration test" in result["content"]
    d.cleanup()


@pytest.mark.asyncio
async def test_get_task_not_found():
    import tempfile
    from deepagent.tools.registry import ToolRegistry
    from deepagent.tools.task_tools import create_task_system_tools

    d = tempfile.TemporaryDirectory()
    mgr = TaskManager(d.name)
    reg = ToolRegistry()
    create_task_system_tools(reg, mgr)

    tool = reg.get("get_task")
    result = await tool(task_id="task_9999999999_0000")
    assert result["success"] is False
    assert "Task not found" in result["error"]
    d.cleanup()
