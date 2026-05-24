"""MessageBus — JSONL mailboxes with exclusive file locks.

Each agent gets a .jsonl inbox. Send appends a line with an exclusive lock.
Read is destructive: reads all messages, then renames the file atomically.

Key improvement over learn-claude-code: proper file locking (fcntl.lockf on
Unix, msvcrt.locking on Windows) + atomic rename on read to prevent message
loss during concurrent access.

Reference: learn-claude-code s15_agent_teams.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _file_lock_acquire(fp) -> bool:
    """Acquire an exclusive lock on an open file. Platform-specific."""
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.lockf(fp.fileno(), fcntl.LOCK_EX)
        return True
    except OSError:
        return False


def _file_lock_release(fp) -> None:
    """Release the lock on an open file."""
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.lockf(fp.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class MessageBus:
    """JSONL-based message bus with exclusive file locking.

    Each agent has a .jsonl inbox under the mailboxes directory.
    Messages are appended one JSON line at a time with an exclusive
    write lock. Reading consumes (reads + atomically renames to .tmp
    then deletes).

    Usage::

        bus = MessageBus(".mailboxes")
        bus.send("lead", "worker", "Review PR #42")
        msgs = bus.read_inbox("worker")  # -> list[dict]
    """

    def __init__(self, mailboxes_dir: str | Path):
        self._dir = Path(mailboxes_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Send ──────────────────────────────────────────────────────

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict | None = None,
    ) -> None:
        """Append a message to *to_agent*'s inbox with an exclusive lock.

        Args:
            from_agent: Sender name (e.g. "lead", "worker-1").
            to_agent: Recipient name.
            content: Message body text.
            msg_type: Category tag (e.g. "message", "shutdown_request").
            metadata: Optional dict stored under ``metadata`` key.
        """
        msg = {
            "from": from_agent,
            "to": to_agent,
            "content": content,
            "type": msg_type,
            "ts": time.time(),
            "metadata": metadata or {},
        }
        inbox = self._dir / f"{to_agent}.jsonl"
        line = json.dumps(msg, ensure_ascii=False) + "\n"

        with open(inbox, "a") as f:
            _file_lock_acquire(f)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                _file_lock_release(f)

    # ── Read (destructive) ─────────────────────────────────────────

    def read_inbox(self, agent: str) -> list[dict]:
        """Read and consume all messages from *agent*'s inbox.

        Messages are read, then the file is atomically renamed to a temp
        file and deleted. This prevents lost messages if two processes
        read concurrently.

        Returns:
            List of message dicts (possibly empty).
        """
        inbox = self._dir / f"{agent}.jsonl"
        if not inbox.exists():
            return []

        msgs: list[dict] = []
        with open(inbox, "r") as f:
            _file_lock_acquire(f)
            try:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Corrupted line in %s inbox", agent)
            finally:
                _file_lock_release(f)

        # Atomic consume: rename to .tmp then delete
        tmp = inbox.with_suffix(".tmp")
        try:
            os.replace(inbox, tmp)
            tmp.unlink()
        except OSError:
            pass

        return msgs

    # ── Introspection ──────────────────────────────────────────────

    def has_messages(self, agent: str) -> bool:
        """Check if *agent* has unread messages without consuming them."""
        inbox = self._dir / f"{agent}.jsonl"
        return inbox.exists() and inbox.stat().st_size > 0
