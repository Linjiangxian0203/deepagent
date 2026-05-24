"""Tests for task data model and persistence."""
import pytest
from deepagent.core.tasks import Task, TaskManager, TaskStatus


@pytest.fixture
def task_dir(tmp_path):
    return tmp_path / ".tasks"


def test_task_create_generates_id():
    task = Task.create(subject="Test task", description="A test")
    assert task.id.startswith("task_")
    assert len(task.id) > 10
    assert task.status == "pending"
    assert task.owner is None
    assert task.blocked_by == []


def test_task_create_with_blocked_by():
    task = Task.create(
        subject="Depends on other",
        description="Blocked task",
        blocked_by=["task_1234_5678"],
    )
    assert task.blocked_by == ["task_1234_5678"]


def test_task_status_transitions():
    task = Task.create(subject="Lifecycle test")
    assert task.status == "pending"
    task.status = "in_progress"
    assert task.status == "in_progress"
    task.status = "completed"
    assert task.status == "completed"


def test_task_invalid_status_raises():
    task = Task.create(subject="Bad status")
    with pytest.raises(ValueError, match="Invalid status"):
        task.status = "unknown"


def test_task_serialize_roundtrip():
    task = Task.create(
        subject="Roundtrip",
        description="Test serialization",
        blocked_by=["task_other_0001"],
    )
    data = task.to_dict()
    restored = Task.from_dict(data)
    assert restored.id == task.id
    assert restored.subject == task.subject
    assert restored.status == task.status
    assert restored.blocked_by == task.blocked_by


def test_task_manager_save_and_load(task_dir):
    mgr = TaskManager(str(task_dir))
    task = Task.create(subject="Persist me", description="Save to disk")
    mgr.save(task)
    loaded = mgr.load(task.id)
    assert loaded is not None
    assert loaded.subject == "Persist me"
    assert loaded.id == task.id


def test_task_manager_list_empty(task_dir):
    mgr = TaskManager(str(task_dir))
    assert mgr.list_all() == []


def test_task_manager_list_with_filter(task_dir):
    mgr = TaskManager(str(task_dir))
    t1 = mgr.create_task(subject="Task 1")
    t2 = mgr.create_task(subject="Task 2")
    mgr.save(t1)
    mgr.save(t2)
    t1.status = "in_progress"
    mgr.save(t1)
    pending = mgr.list_all(status="pending")
    assert len(pending) == 1
    assert pending[0].subject == "Task 2"


def test_task_manager_can_start(task_dir):
    mgr = TaskManager(str(task_dir))
    t1 = mgr.create_task(subject="Blocking task")
    t2 = mgr.create_task(subject="Blocked task", blocked_by=[t1.id])
    mgr.save(t1)
    mgr.save(t2)
    assert not mgr.can_start(t2.id)
    t1.status = "completed"
    mgr.save(t1)
    assert mgr.can_start(t2.id)


def test_task_manager_can_start_missing_dep(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="Blocked", blocked_by=["task_nonexistent_0000"])
    mgr.save(t)
    assert not mgr.can_start(t.id)


def test_task_manager_claim_task(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="Claimable")
    mgr.save(t)
    result = mgr.claim(t.id, owner="agent-1")
    assert result is not None
    loaded = mgr.load(t.id)
    assert loaded.status == "in_progress"
    assert loaded.owner == "agent-1"


def test_task_manager_claim_already_in_progress(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="Already taken")
    t.status = "in_progress"
    mgr.save(t)
    result = mgr.claim(t.id)
    assert result is None


def test_task_manager_claim_blocked(task_dir):
    mgr = TaskManager(str(task_dir))
    dep = mgr.create_task(subject="Dependency")
    t = mgr.create_task(subject="Still blocked", blocked_by=[dep.id])
    mgr.save(dep)
    mgr.save(t)
    result = mgr.claim(t.id)
    assert result is None


def test_task_manager_complete_reports_unblocked(task_dir):
    mgr = TaskManager(str(task_dir))
    dep = mgr.create_task(subject="Will complete")
    t = mgr.create_task(subject="Waiting", blocked_by=[dep.id])
    mgr.save(dep)
    mgr.save(t)
    dep.status = "in_progress"
    mgr.save(dep)
    result = mgr.complete(dep.id)
    assert result is not None
    assert dep.id in result
    assert mgr.load(dep.id).status == "completed"
    assert mgr.can_start(t.id)


def test_task_manager_complete_wrong_status(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="Not started")
    mgr.save(t)
    result = mgr.complete(t.id)
    assert result is None


def test_task_manager_history_audit_log(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="Audited")
    t.status = "in_progress"
    mgr.save(t)
    t.status = "completed"
    mgr.save(t)
    history = mgr.get_history(t.id)
    assert len(history) == 3
    assert history[0]["status"] == "pending"
    assert history[1]["status"] == "in_progress"
    assert history[2]["status"] == "completed"


def test_task_manager_delete(task_dir):
    mgr = TaskManager(str(task_dir))
    t = mgr.create_task(subject="To delete")
    assert mgr.load(t.id) is not None
    assert mgr.delete(t.id) is True
    assert mgr.load(t.id) is None
    assert mgr.delete("nonexistent") is False


def test_task_status_class():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.IN_PROGRESS == "in_progress"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.validate("pending") == "pending"
    with pytest.raises(ValueError):
        TaskStatus.validate("unknown")


def test_task_from_dict_rejects_invalid_status():
    """C1 regression: corrupted files with invalid status must raise on load."""
    from deepagent.core.tasks import Task
    with pytest.raises(ValueError, match="Invalid status"):
        Task.from_dict({
            "id": "task_1234_5678",
            "subject": "Corrupted",
            "description": "",
            "status": "unknown",
            "owner": None,
            "blocked_by": [],
            "metadata": {},
        })
