"""Task system — file-persisted tasks with blockedBy dependencies.

Matches Claude Code's task system behavior:
- JSON file per task under .tasks/
- State machine: pending -> in_progress -> completed
- Dependency checking: blockedBy all must be completed
- history.jsonl audit log of all status transitions

Reference: learn-claude-code s12_task_system.
"""

from __future__ import annotations

import json
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path


class TaskStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

    VALID = frozenset({PENDING, IN_PROGRESS, COMPLETED})

    @classmethod
    def validate(cls, value: str) -> str:
        if value not in cls.VALID:
            raise ValueError(f"Invalid status: {value!r}. Must be one of {cls.VALID}")
        return value


@dataclass
class Task:
    """A persistent task with dependency tracking."""

    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blocked_by: list[str]
    metadata: dict

    @classmethod
    def create(
        cls,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
    ) -> "Task":
        task_id = f"task_{int(time.time())}_{random.randint(0, 9999):04d}"
        return cls(
            id=task_id,
            subject=subject,
            description=description,
            status=TaskStatus.PENDING,
            owner=None,
            blocked_by=blocked_by or [],
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def __post_init__(self):
        TaskStatus.validate(self.status)

    def __setattr__(self, name, value):
        if name == "status":
            TaskStatus.validate(value)
        super().__setattr__(name, value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(**data)


class TaskManager:
    """Manages persistent tasks stored as JSON files in a directory.

    Each task is stored as task_{id}.json under tasks_dir.
    Status transitions are logged to history.jsonl.
    """

    def __init__(self, tasks_dir: str):
        self._dir = Path(tasks_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._history_path = self._dir / "history.jsonl"

    # -- CRUD --

    def create_task(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
    ) -> Task:
        task = Task.create(subject, description, blocked_by)
        self.save(task)
        return task

    def save(self, task: Task) -> None:
        self._task_path(task.id).write_text(
            json.dumps(task.to_dict(), indent=2, ensure_ascii=False)
        )
        self._log_history(task.id, task.to_dict())

    def load(self, task_id: str) -> Task | None:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        return Task.from_dict(json.loads(path.read_text()))

    def list_all(self, status: str | None = None) -> list[Task]:
        tasks = []
        for p in sorted(self._dir.glob("task_*.json")):
            try:
                t = Task.from_dict(json.loads(p.read_text()))
                if status is None or t.status == status:
                    tasks.append(t)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return tasks

    def delete(self, task_id: str) -> bool:
        path = self._task_path(task_id)
        if path.exists():
            path.unlink()
            return True
        return False

    # -- State machine --

    def can_start(self, task_id: str) -> bool:
        """Check if all blockedBy dependencies are completed.
        Missing dependencies are treated as blocked."""
        task = self.load(task_id)
        if task is None:
            return False
        for dep_id in task.blocked_by:
            dep = self.load(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def claim(self, task_id: str, owner: str = "agent") -> str | None:
        """Claim a pending task. Returns task ID on success, None on failure.
        Guard: status must be pending, all deps must be satisfied."""
        task = self.load(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.PENDING:
            return None
        if not self.can_start(task_id):
            return None
        task.status = TaskStatus.IN_PROGRESS
        task.owner = owner
        task.metadata["claimed_at"] = datetime.now(timezone.utc).isoformat()
        self.save(task)
        return task.id

    def complete(self, task_id: str) -> str | None:
        """Complete an in-progress task. Reports newly unblocked tasks.
        Guard: status must be in_progress."""
        task = self.load(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.IN_PROGRESS:
            return None
        task.status = TaskStatus.COMPLETED
        task.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.save(task)

        unblocked = [
            t.subject for t in self.list_all(status=TaskStatus.PENDING)
            if t.blocked_by and task_id in t.blocked_by and self.can_start(t.id)
        ]
        msg = task.id
        if unblocked:
            msg += f"\nUnblocked: {', '.join(unblocked)}"
        return msg

    # -- History --

    def get_history(self, task_id: str) -> list[dict]:
        if not self._history_path.exists():
            return []
        entries = []
        for line in self._history_path.read_text().splitlines():
            try:
                entry = json.loads(line)
                if entry.get("id") == task_id:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
        return entries

    # -- Internal --

    def _task_path(self, task_id: str) -> Path:
        return self._dir / f"{task_id}.json"

    def _log_history(self, task_id: str, data: dict) -> None:
        entry = {
            "id": task_id,
            "status": data.get("status"),
            "owner": data.get("owner"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._history_path.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
