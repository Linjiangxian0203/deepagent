"""Tests for BackgroundManager."""
import asyncio
import pytest
from deepagent.core.background import (
    BackgroundManager, BackgroundTask,
    is_slow_operation, should_run_background, format_notification,
)


@pytest.fixture
def bg_mgr():
    return BackgroundManager()


def test_background_task_dataclass():
    bt = BackgroundTask(
        bg_id="bg_0001",
        tool_name="run_shell",
        arguments={"command": "sleep 1"},
    )
    assert bt.bg_id == "bg_0001"
    assert bt.status == "running"
    assert bt.result is None


@pytest.mark.asyncio
async def test_background_manager_start_returns_id(bg_mgr):
    async def slow_op():
        await asyncio.sleep(0.01)
        return {"success": True, "content": "done"}

    bg_id = bg_mgr.start("run_shell", {"command": "sleep 1"}, slow_op())
    assert bg_id.startswith("bg_")


@pytest.mark.asyncio
async def test_background_manager_collect_ready(bg_mgr):
    async def fast_op():
        return {"success": True, "content": "fast done"}

    bg_id = bg_mgr.start("run_shell", {"command": "echo hi"}, fast_op())
    await asyncio.sleep(0.05)

    results = bg_mgr.collect_ready()
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_background_manager_collect_empty_when_none_ready(bg_mgr):
    results = bg_mgr.collect_ready()
    assert len(results) == 0


@pytest.mark.asyncio
async def test_background_manager_pending_count(bg_mgr):
    async def slow_op():
        await asyncio.sleep(0.02)
        return {"success": True, "content": "done"}

    bg_mgr.start("run_shell", {"command": "sleep 0.02"}, slow_op())
    assert bg_mgr.pending_count == 1


@pytest.mark.asyncio
async def test_background_manager_cancel(bg_mgr):
    async def slow_op():
        await asyncio.sleep(10)
        return {"success": True, "content": "done"}

    bg_id = bg_mgr.start("run_shell", {"command": "sleep 10"}, slow_op())
    assert bg_mgr.cancel(bg_id) is True
    assert bg_mgr.pending_count == 0


def test_is_slow_operation_keyword_match():
    assert is_slow_operation("run_shell", {"command": "npm install react"})
    assert is_slow_operation("run_shell", {"command": "pip install requests"})
    assert is_slow_operation("run_shell", {"command": "pytest tests/"})
    assert is_slow_operation("run_shell", {"command": "docker build ."})
    assert is_slow_operation("run_shell", {"command": "cargo build --release"})
    assert is_slow_operation("run_shell", {"command": "make all"})
    assert not is_slow_operation("run_shell", {"command": "echo hello"})
    assert not is_slow_operation("read_file", {"path": "foo.py"})


def test_should_run_background_explicit_flag():
    assert should_run_background("run_shell", {"command": "echo hi", "run_in_background": True})
    assert not should_run_background("run_shell", {"command": "echo hi"})
    assert should_run_background("run_shell", {"command": "npm install"})


def test_format_notification():
    notif = format_notification("bg_0001", "npm install react", "Success: installed 42 packages\nand more stuff")
    assert "<task_notification>" in notif
    assert "bg_0001" in notif
    assert "npm install react" in notif
    assert "completed" in notif
    assert "</task_notification>" in notif
    # Should truncate long summaries to 200 chars
    assert len("Success: installed 42 packages\nand more stuff") < 200


def test_format_notification_truncates_long_summary():
    long_output = "x" * 500
    notif = format_notification("bg_0002", "long command", long_output)
    assert "x" * 200 in notif
    assert "x" * 201 not in notif
