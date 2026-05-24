"""Tests for MessageBus — send/receive, file locking, atomic operations."""
import json
import os
import tempfile
from pathlib import Path

import pytest
from deepagent.core.message_bus import MessageBus


@pytest.fixture
def bus():
    d = tempfile.TemporaryDirectory()
    b = MessageBus(d.name)
    yield b
    d.cleanup()


def test_send_and_receive_single_message(bus):
    bus.send("lead", "worker", "Hello, Worker!")
    msgs = bus.read_inbox("worker")
    assert len(msgs) == 1
    assert msgs[0]["from"] == "lead"
    assert msgs[0]["to"] == "worker"
    assert msgs[0]["content"] == "Hello, Worker!"
    assert msgs[0]["type"] == "message"
    assert "ts" in msgs[0]
    assert msgs[0]["metadata"] == {}


def test_send_with_msg_type_and_metadata(bus):
    bus.send("lead", "worker", "Shutdown now", msg_type="shutdown_request",
             metadata={"request_id": "req_123", "priority": "high"})
    msgs = bus.read_inbox("worker")
    assert len(msgs) == 1
    assert msgs[0]["type"] == "shutdown_request"
    assert msgs[0]["metadata"]["request_id"] == "req_123"
    assert msgs[0]["metadata"]["priority"] == "high"


def test_send_multiple_messages_preserves_order(bus):
    for i in range(5):
        bus.send("lead", "worker", f"Message {i}")
    msgs = bus.read_inbox("worker")
    assert len(msgs) == 5
    for i, m in enumerate(msgs):
        assert m["content"] == f"Message {i}"


def test_read_is_destructive(bus):
    bus.send("lead", "worker", "One-time message")
    first_read = bus.read_inbox("worker")
    assert len(first_read) == 1

    second_read = bus.read_inbox("worker")
    assert len(second_read) == 0


def test_read_inbox_empty(bus):
    msgs = bus.read_inbox("nonexistent")
    assert msgs == []


def test_send_to_different_agents_independent(bus):
    bus.send("lead", "worker-a", "For A")
    bus.send("lead", "worker-b", "For B")

    a_msgs = bus.read_inbox("worker-a")
    b_msgs = bus.read_inbox("worker-b")
    assert len(a_msgs) == 1
    assert a_msgs[0]["content"] == "For A"
    assert len(b_msgs) == 1
    assert b_msgs[0]["content"] == "For B"


def test_has_messages_true(bus):
    bus.send("lead", "worker", "test")
    assert bus.has_messages("worker") is True


def test_has_messages_false_for_empty_inbox(bus):
    assert bus.has_messages("nonexistent") is False


def test_has_messages_false_after_read(bus):
    bus.send("lead", "worker", "test")
    bus.read_inbox("worker")
    assert bus.has_messages("worker") is False


def test_read_inbox_corrupted_line_is_skipped(bus):
    inbox = bus._dir / "worker.jsonl"
    inbox.write_text('{"valid": "json"}\nnot valid json\n{"also_valid": "yes"}\n', "utf-8")
    msgs = bus.read_inbox("worker")
    # The corrupted line should be skipped; valid lines read
    assert len(msgs) == 2
    assert msgs[0] == {"valid": "json"}
    assert msgs[1] == {"also_valid": "yes"}


def test_send_creates_mailbox_directory_if_missing():
    d = tempfile.TemporaryDirectory()
    mailboxes = Path(d.name) / "nested" / "mailboxes"
    bus = MessageBus(str(mailboxes))
    bus.send("lead", "worker", "test")
    assert (mailboxes / "worker.jsonl").exists()
    d.cleanup()


def test_read_inbox_atomic_rename_on_consume(bus):
    """After read_inbox, the original file should not exist, and .tmp should be gone."""
    bus.send("lead", "worker", "test message")
    inbox = bus._dir / "worker.jsonl"
    assert inbox.exists()

    bus.read_inbox("worker")
    assert not inbox.exists()
    assert not inbox.with_suffix(".tmp").exists()


@pytest.mark.asyncio
async def test_concurrent_sends_to_same_inbox(bus):
    """Multiple sends to the same inbox should all persist."""
    import asyncio

    async def sender(b, msg_id):
        b.send("lead", "shared", f"Message {msg_id}")

    await asyncio.gather(*[sender(bus, i) for i in range(10)])
    msgs = bus.read_inbox("shared")
    assert len(msgs) == 10
    contents = {m["content"] for m in msgs}
    assert contents == {f"Message {i}" for i in range(10)}
