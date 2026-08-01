"""Tests for WorktreeManager, validate_worktree_name, worktree tools, and Task worktree field."""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from deepagent.core.tasks import Task
from deepagent.core.worktree import (
    WorktreeInfo,
    WorktreeManager,
    validate_worktree_name,
)
from deepagent.tools.registry import ToolRegistry
from deepagent.tools.worktree_tools import create_worktree_tools


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def manager(tmp_path):
    """A WorktreeManager backed by a temp directory (not a git repo)."""
    return WorktreeManager(tmp_path)


def _init_git_repo(path: Path) -> None:
    """Helper: initialize a git repo with an initial commit so HEAD exists."""
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path,
        capture_output=True,
        check=True,
    )
    (path / "README.md").write_text("# test")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=path, capture_output=True, check=True
    )


@pytest.fixture
def git_repo(tmp_path):
    """A temp directory initialized as a git repo with an initial commit."""
    _init_git_repo(tmp_path)
    return tmp_path


@pytest.fixture
def git_manager(git_repo):
    """A WorktreeManager pointing at an actual git repo."""
    return WorktreeManager(git_repo)


# ==============================================================================
# 1. validate_worktree_name() — 8 tests
# ==============================================================================


def test_validate_valid_names():
    """Valid names return None (no error)."""
    assert validate_worktree_name("auth") is None
    assert validate_worktree_name("feature-branch") is None
    assert validate_worktree_name("release.v2") is None
    assert validate_worktree_name("my_worktree") is None


def test_validate_single_char():
    """A single character is a valid name."""
    assert validate_worktree_name("a") is None


def test_validate_max_64_chars():
    """A 64-character name is valid."""
    name = "a" * 64
    assert len(name) == 64
    assert validate_worktree_name(name) is None


def test_validate_empty_string():
    """Empty string returns an error."""
    err = validate_worktree_name("")
    assert err is not None
    assert "empty" in err.lower() or "cannot be empty" in err


def test_validate_dot():
    """A single dot '.' is not a valid name."""
    err = validate_worktree_name(".")
    assert err is not None
    assert "." in err or "not a valid" in err


def test_validate_double_dot():
    """'..' is not a valid name."""
    err = validate_worktree_name("..")
    assert err is not None
    assert ".." in err or "not a valid" in err


def test_validate_path_traversal():
    """Path traversal strings like '../../etc' are rejected."""
    err = validate_worktree_name("../../etc")
    assert err is not None


def test_validate_special_chars_and_too_long():
    """Names with spaces/special chars are invalid; 65+ chars is invalid."""
    assert validate_worktree_name("my worktree!") is not None
    assert validate_worktree_name("a" * 65) is not None


# ==============================================================================
# 2. WorktreeManager (no git needed) — 8 tests
# ==============================================================================


def test_init_creates_worktrees_dir(tmp_path):
    """__init__ creates the .worktrees/ directory."""
    wt_dir = tmp_path / ".worktrees"
    assert not wt_dir.exists()
    WorktreeManager(tmp_path)
    assert wt_dir.exists()
    assert wt_dir.is_dir()


def test_validate_name_method(manager):
    """validate_name returns True for valid, False for invalid."""
    assert manager.validate_name("auth") is True
    assert manager.validate_name("") is False
    assert manager.validate_name(".") is False


def test_path_for_returns_correct_path(manager):
    """path_for returns the correct path when the directory exists."""
    (manager._worktrees_dir / "my-task").mkdir()
    p = manager.path_for("my-task")
    assert p == manager._worktrees_dir / "my-task"


def test_path_for_returns_none_for_nonexistent(manager):
    """path_for returns None when the directory doesn't exist."""
    assert manager.path_for("nonexistent") is None


def test_list_all_empty_initially(manager):
    """list_all returns an empty list when no worktrees exist."""
    assert manager.list_all() == []


def test_list_all_after_creating_subdir(manager):
    """list_all discovers subdirectories created in .worktrees/."""
    (manager._worktrees_dir / "task-alpha").mkdir()
    (manager._worktrees_dir / "task-beta").mkdir()
    # Also create a file — it should be ignored
    (manager._worktrees_dir / "not-a-dir.txt").write_text("nope")
    results = manager.list_all()
    names = {r.name for r in results}
    assert names == {"task-alpha", "task-beta"}
    for r in results:
        assert r.path == manager._worktrees_dir / r.name
        assert r.branch == f"wt/{r.name}"
        assert r.task_id == ""


def test_bind_task_creates_worktree_field(manager, tmp_path):
    """bind_task writes a 'worktree' field to the task JSON file."""
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    task_data = {
        "id": "task_0001",
        "subject": "Test bind",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "metadata": {},
    }
    task_path = tasks_dir / "task_0001.json"
    task_path.write_text(json.dumps(task_data))

    ok, msg = manager.bind_task("task_0001", "my-worktree", str(tasks_dir))
    assert ok is True
    assert "my-worktree" in msg

    saved = json.loads(task_path.read_text())
    assert saved["worktree"] == "my-worktree"
    assert saved["subject"] == "Test bind"
    assert saved["status"] == "pending"


def test_log_event_appends_to_events_jsonl(manager):
    """_log_event appends a JSON line to events.jsonl."""
    manager._log_event("create", "test-wt", "task_0001")
    events_file = manager._worktrees_dir / "events.jsonl"
    assert events_file.exists()
    lines = events_file.read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "create"
    assert event["worktree"] == "test-wt"
    assert event["task_id"] == "task_0001"
    assert "ts" in event


def test_keep_with_invalid_name_returns_error(manager):
    """keep with an invalid name returns (False, error_string)."""

    async def _test():
        return await manager.keep("")

    ok, msg = asyncio.run(_test())
    assert ok is False
    assert "empty" in msg.lower() or "cannot be empty" in msg


# ==============================================================================
# 3. Git-dependent tests — 6 tests
# ==============================================================================

git_available = shutil.which("git") is not None


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_create_worktree_succeeds(git_manager, git_repo):
    """create() succeeds on a valid git repo and creates the worktree directory."""
    ok, msg = await git_manager.create("test-feature")
    assert ok is True
    assert "test-feature" in msg
    wt_path = git_repo / ".worktrees" / "test-feature"
    assert wt_path.is_dir()
    # Should have a .git file pointing back
    assert (wt_path / ".git").exists()

    # Cleanup
    await git_manager.remove("test-feature", discard_changes=True)


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_create_with_task_binding(git_manager, git_repo):
    """create() with task_id binds the task."""
    tasks_dir = git_repo / ".tasks"
    tasks_dir.mkdir()
    task_path = tasks_dir / "task_bind_test.json"
    task_path.write_text(json.dumps({
        "id": "task_bind_test",
        "subject": "Bind me",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "metadata": {},
    }))

    ok, msg = await git_manager.create("bound-feature", task_id="task_bind_test")
    assert ok is True

    saved = json.loads(task_path.read_text())
    assert saved["worktree"] == "bound-feature"

    # Cleanup
    await git_manager.remove("bound-feature", discard_changes=True)


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_create_invalid_name_returns_error(git_manager):
    """create() with an invalid name returns (False, error)."""
    ok, msg = await git_manager.create("")
    assert ok is False
    assert "Error" in msg or "error" in msg


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_create_duplicate_name_returns_error(git_manager):
    """create() with an already-existing name returns (False, error)."""
    await git_manager.create("dup-test")
    ok, msg = await git_manager.create("dup-test")
    assert ok is False
    assert "already exists" in msg.lower()

    # Cleanup
    await git_manager.remove("dup-test", discard_changes=True)


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_remove_worktree_clean(git_manager):
    """remove() on a clean worktree (no changes) succeeds."""
    await git_manager.create("clean-remove")
    wt_path = git_manager.path_for("clean-remove")
    assert wt_path is not None

    ok, msg = await git_manager.remove("clean-remove")
    assert ok is True
    assert "removed" in msg.lower()
    assert git_manager.path_for("clean-remove") is None


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_keep_worktree(git_manager):
    """keep() succeeds on an existing worktree and logs the event."""
    await git_manager.create("keep-me")
    ok, msg = await git_manager.keep("keep-me")
    assert ok is True
    assert "keep-me" in msg
    # Worktree still exists after keep
    assert git_manager.path_for("keep-me") is not None

    # Verify the event was logged
    events_file = git_manager._worktrees_dir / "events.jsonl"
    lines = events_file.read_text().strip().splitlines()
    keep_events = [json.loads(l) for l in lines if json.loads(l)["type"] == "keep"]
    assert len(keep_events) >= 1
    assert keep_events[-1]["worktree"] == "keep-me"

    # Cleanup
    await git_manager.remove("keep-me", discard_changes=True)


# ==============================================================================
# 4. Safety checks — 3 tests
# ==============================================================================


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_remove_refuses_when_uncommitted_changes(git_manager):
    """remove() refuses when worktree has uncommitted changes."""
    await git_manager.create("dirty-wt")
    wt_path = git_manager.path_for("dirty-wt")

    # Create an uncommitted file in the worktree
    (wt_path / "new-file.txt").write_text("uncommitted content")

    ok, msg = await git_manager.remove("dirty-wt")
    assert ok is False
    assert "uncommitted" in msg.lower() or "discard_changes" in msg.lower()

    # Cleanup — force remove
    await git_manager.remove("dirty-wt", discard_changes=True)


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_remove_discard_changes_succeeds_when_dirty(git_manager):
    """remove() with discard_changes=True succeeds even with dirty worktree."""
    await git_manager.create("force-remove-me")
    wt_path = git_manager.path_for("force-remove-me")

    # Make it dirty
    (wt_path / "temp.txt").write_text("throwaway")

    ok, msg = await git_manager.remove("force-remove-me", discard_changes=True)
    assert ok is True
    assert git_manager.path_for("force-remove-me") is None


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_count_changes_returns_correct_counts(git_manager):
    """_count_changes returns correct (files, commits) counts."""
    await git_manager.create("count-test")
    wt_path = git_manager.path_for("count-test")

    # Should be clean initially
    files, commits = await git_manager._count_changes(wt_path)
    assert files == 0
    assert commits == 0

    # Add an uncommitted file
    (wt_path / "uncommitted.txt").write_text("dirty")
    files, commits = await git_manager._count_changes(wt_path)
    assert files == 1

    # Stage it — still counts as uncommitted change
    subprocess.run(["git", "add", "uncommitted.txt"], cwd=wt_path, capture_output=True)
    files, commits = await git_manager._count_changes(wt_path)
    assert files == 1

    # Cleanup
    await git_manager.remove("count-test", discard_changes=True)


# ==============================================================================
# 5. Tool tests — 6 tests
# ==============================================================================


class MockWorktreeManager:
    """Fake WorktreeManager that records calls and returns preset results."""

    def __init__(self, create_result=(True, "created"), remove_result=(True, "removed"),
                 keep_result=(True, "kept")):
        self.calls: list[dict] = []
        self._create_result = create_result
        self._remove_result = remove_result
        self._keep_result = keep_result

    async def create(self, name: str, task_id: str = "") -> tuple[bool, str]:
        self.calls.append({"method": "create", "name": name, "task_id": task_id})
        return self._create_result

    async def remove(self, name: str, discard_changes: bool = False) -> tuple[bool, str]:
        self.calls.append({"method": "remove", "name": name, "discard_changes": discard_changes})
        return self._remove_result

    async def keep(self, name: str) -> tuple[bool, str]:
        self.calls.append({"method": "keep", "name": name})
        return self._keep_result


@pytest.fixture
def registry_with_tools():
    """ToolRegistry with worktree tools registered against a mock manager."""
    reg = ToolRegistry()
    mock = MockWorktreeManager()
    create_worktree_tools(reg, mock)
    return reg, mock


@pytest.mark.asyncio
async def test_create_worktree_tool_success(registry_with_tools):
    """create_worktree tool calls manager.create and returns success dict."""
    reg, mock = registry_with_tools
    tool = reg.get("create_worktree")
    result = await tool(name="my-feature", task_id="task_0001")
    assert result["success"] is True
    assert result["content"] == "created"
    assert len(mock.calls) == 1
    assert mock.calls[0] == {"method": "create", "name": "my-feature", "task_id": "task_0001"}


@pytest.mark.asyncio
async def test_create_worktree_tool_error(registry_with_tools):
    """create_worktree tool returns error dict on manager failure."""
    reg, mock = registry_with_tools
    mock._create_result = (False, "Name taken")
    tool = reg.get("create_worktree")
    result = await tool(name="dup")
    assert result["success"] is False
    assert result["error"] == "Name taken"
    assert result["content"] == ""


@pytest.mark.asyncio
async def test_remove_worktree_tool_success(registry_with_tools):
    """remove_worktree tool calls manager.remove and returns success dict."""
    reg, mock = registry_with_tools
    tool = reg.get("remove_worktree")
    result = await tool(name="old-feature", discard_changes=True)
    assert result["success"] is True
    assert result["content"] == "removed"
    assert mock.calls[0] == {"method": "remove", "name": "old-feature", "discard_changes": True}


@pytest.mark.asyncio
async def test_remove_worktree_tool_error(registry_with_tools):
    """remove_worktree tool returns error dict on manager failure."""
    reg, mock = registry_with_tools
    mock._remove_result = (False, "Dirty worktree")
    tool = reg.get("remove_worktree")
    result = await tool(name="dirty-wt")
    assert result["success"] is False
    assert result["error"] == "Dirty worktree"


@pytest.mark.asyncio
async def test_keep_worktree_tool_success(registry_with_tools):
    """keep_worktree tool calls manager.keep and returns success dict."""
    reg, mock = registry_with_tools
    tool = reg.get("keep_worktree")
    result = await tool(name="review-me")
    assert result["success"] is True
    assert result["content"] == "kept"
    assert mock.calls[0] == {"method": "keep", "name": "review-me"}


def test_all_three_tools_registered(registry_with_tools):
    """create_worktree_tools registers exactly 3 tools."""
    reg, _ = registry_with_tools
    names = reg.list_names()
    assert "create_worktree" in names
    assert "remove_worktree" in names
    assert "keep_worktree" in names
    assert len(names) == 3


# ==============================================================================
# 6. Task worktree field — 4 tests
# ==============================================================================


def test_task_has_worktree_field_default_none():
    """Task dataclass has a worktree field that defaults to None."""
    task = Task(
        id="task_0001",
        subject="Test",
        description="",
        status="pending",
        owner=None,
        blocked_by=[],
        metadata={},
    )
    assert task.worktree is None


def test_task_to_dict_includes_worktree():
    """Task.to_dict() includes the worktree field."""
    task = Task(
        id="task_0002",
        subject="With worktree",
        description="",
        status="pending",
        owner=None,
        blocked_by=[],
        worktree="my-worktree",
        metadata={},
    )
    d = task.to_dict()
    assert d["worktree"] == "my-worktree"


def test_task_from_dict_handles_worktree_key():
    """Task.from_dict() reads the worktree key."""
    data = {
        "id": "task_0003",
        "subject": "From dict",
        "description": "",
        "status": "in_progress",
        "owner": "agent",
        "blocked_by": [],
        "worktree": "bound-wt",
        "metadata": {},
    }
    task = Task.from_dict(data)
    assert task.worktree == "bound-wt"
    assert task.subject == "From dict"


def test_task_from_dict_missing_worktree_backward_compat():
    """Task.from_dict() handles missing worktree key for backward compatibility."""
    data = {
        "id": "task_0004",
        "subject": "No worktree key",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "metadata": {},
    }
    task = Task.from_dict(data)
    assert task.worktree is None
    assert task.id == "task_0004"


# ==============================================================================
# 7. WorktreeInfo dataclass — 2 tests
# ==============================================================================


def test_worktree_info_defaults():
    """WorktreeInfo has correct default values for task_id and branch."""
    info = WorktreeInfo(name="test", path=Path("/tmp/.worktrees/test"))
    assert info.name == "test"
    assert info.task_id == ""
    assert info.branch == ""


def test_worktree_info_all_fields():
    """WorktreeInfo with all fields explicitly set."""
    p = Path("/tmp/.worktrees/feature-x")
    info = WorktreeInfo(name="feature-x", path=p, task_id="task_0001", branch="wt/feature-x")
    assert info.name == "feature-x"
    assert info.path == p
    assert info.task_id == "task_0001"
    assert info.branch == "wt/feature-x"


# ==============================================================================
# 8. Edge cases and additional coverage — 6 tests
# ==============================================================================


def test_bind_task_nonexistent_task(manager):
    """bind_task returns error when task JSON doesn't exist."""
    ok, msg = manager.bind_task("task_nonexistent", "wt-name")
    assert ok is False
    assert "not found" in msg.lower()


def test_bind_task_default_tasks_dir(manager, tmp_path):
    """bind_task uses project_root/.tasks/ when no tasks_dir is provided."""
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    task_path = tasks_dir / "task_default.json"
    task_path.write_text(json.dumps({
        "id": "task_default",
        "subject": "Default dir",
        "description": "",
        "status": "pending",
        "owner": None,
        "blocked_by": [],
        "metadata": {},
    }))

    ok, msg = manager.bind_task("task_default", "default-wt")
    assert ok is True
    saved = json.loads(task_path.read_text())
    assert saved["worktree"] == "default-wt"


def test_bind_task_corrupted_json(manager, tmp_path):
    """bind_task returns error when task JSON is malformed."""
    tasks_dir = tmp_path / ".tasks"
    tasks_dir.mkdir()
    (tasks_dir / "bad_task.json").write_text("not valid json {{{")

    ok, msg = manager.bind_task("bad_task", "wt-name", str(tasks_dir))
    assert ok is False
    assert "failed" in msg.lower() or "error" in msg.lower()


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_remove_nonexistent_worktree(git_manager):
    """remove() on a non-existent worktree returns error."""
    ok, msg = await git_manager.remove("never-created")
    assert ok is False
    assert "not found" in msg.lower()


@pytest.mark.skipif(not git_available, reason="git not available")
@pytest.mark.asyncio
async def test_keep_nonexistent_worktree(git_manager):
    """keep() on a non-existent worktree returns error."""
    ok, msg = await git_manager.keep("nobody-home")
    assert ok is False
    assert "not found" in msg.lower()


def test_validate_name_method_edge_cases(manager):
    """validate_name rejects '..' and path traversal via the method too."""
    assert manager.validate_name("..") is False
    assert manager.validate_name("../../etc") is False
    assert manager.validate_name("valid-name") is True


def test_log_event_multiple_events(manager):
    """_log_event appends multiple events to the same file."""
    manager._log_event("create", "wt-1", "t1")
    manager._log_event("keep", "wt-1", "t1")
    manager._log_event("remove", "wt-1", "t1")
    events_file = manager._worktrees_dir / "events.jsonl"
    lines = events_file.read_text().strip().splitlines()
    assert len(lines) == 3
    types = [json.loads(l)["type"] for l in lines]
    assert types == ["create", "keep", "remove"]


@pytest.mark.asyncio
async def test_create_with_empty_task_id(git_manager):
    """create() with no task_id (default empty string) skips task binding."""
    if not git_available:
        pytest.skip("git not available")
    ok, msg = await git_manager.create("no-task-wt")
    assert ok is True
    # Cleanup
    await git_manager.remove("no-task-wt", discard_changes=True)


@pytest.mark.asyncio
async def test_remove_with_invalid_name_returns_error(git_manager):
    """remove() with an invalid name returns error tuple."""
    if not git_available:
        pytest.skip("git not available")
    ok, msg = await git_manager.remove("")
    assert ok is False
