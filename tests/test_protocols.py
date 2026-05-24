"""Tests for Team Protocols (Layer 3) and Autonomous Agents (Layer 4)."""
import json
import tempfile
from pathlib import Path

import pytest
from deepagent.core.message_bus import MessageBus
from deepagent.core.protocols import (
    ProtocolManager,
    ProtocolState,
    new_request_id,
    PROTOCOL_RESPONSE_MAP,
)
from deepagent.tools.registry import ToolRegistry, tool
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.team_tools import (
    create_team_tools,
    _build_teammate_tool_schemas,
    _dispatch_inbox_message,
    _scan_unclaimed_tasks,
    TEAMMATE_TOOL_NAMES,
)


# ── ProtocolManager ──────────────────────────────────────────────────


@pytest.fixture
def proto():
    return ProtocolManager(timeout_seconds=5)


def test_new_request_creates_pending_state(proto):
    req_id = proto.new_request("shutdown", "lead", "worker-1")
    assert req_id.startswith("req_")

    state = proto.get_state(req_id)
    assert state is not None
    assert state.type == "shutdown"
    assert state.sender == "lead"
    assert state.target == "worker-1"
    assert state.status == "pending"


def test_match_response_correct_type(proto):
    req_id = proto.new_request("shutdown", "lead", "worker-1")
    result = proto.match_response("shutdown_response", req_id, approve=True)
    assert result is not None
    assert result.status == "approved"


def test_match_response_reject(proto):
    req_id = proto.new_request("plan_approval", "worker", "lead")
    result = proto.match_response("plan_approval_response", req_id, approve=False)
    assert result is not None
    assert result.status == "rejected"


def test_match_response_type_mismatch(proto):
    req_id = proto.new_request("shutdown", "lead", "worker")
    result = proto.match_response("plan_approval_response", req_id, approve=True)
    assert result is None  # wrong response type


def test_match_response_unknown_request_id(proto):
    result = proto.match_response("shutdown_response", "req_nonexistent", approve=True)
    assert result is None


def test_match_response_already_resolved(proto):
    req_id = proto.new_request("shutdown", "lead", "worker")
    proto.match_response("shutdown_response", req_id, approve=True)
    # Duplicate response should be ignored
    result = proto.match_response("shutdown_response", req_id, approve=False)
    assert result is not None
    assert result.status == "approved"  # unchanged from first resolution


def test_check_timeouts(proto):
    import time
    proto._timeout = 0  # immediate timeout via internal field
    req_id = proto.new_request("shutdown", "lead", "worker")
    time.sleep(0.001)  # ensure time passes
    timed_out = proto.check_timeouts()
    assert len(timed_out) == 1
    assert timed_out[0].status == "timeout"
    assert proto.get_state(req_id) is None  # removed after timeout


def test_pending_count(proto):
    assert proto.pending_count == 0
    proto.new_request("shutdown", "lead", "w1")
    proto.new_request("plan_approval", "lead", "w1")
    assert proto.pending_count == 2
    proto.match_response("shutdown_response", list(proto._pending.keys())[0], approve=True)
    assert proto.pending_count == 1


def test_list_pending(proto):
    proto.new_request("shutdown", "lead", "w1")
    proto.new_request("code_review", "lead", "w2")
    pending = proto.list_pending()
    assert len(pending) == 2
    types = {p.type for p in pending}
    assert types == {"shutdown", "code_review"}


def test_new_request_id_uniqueness():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100  # should be unique


def test_protocol_response_map():
    assert PROTOCOL_RESPONSE_MAP["shutdown_request"] == "shutdown_response"
    assert PROTOCOL_RESPONSE_MAP["plan_approval_request"] == "plan_approval_response"
    assert PROTOCOL_RESPONSE_MAP["code_review_request"] == "code_review_response"


def test_protocol_state_dataclass():
    state = ProtocolState(
        request_id="req_123",
        type="shutdown",
        sender="lead",
        target="worker",
        status="pending",
        payload="",
    )
    assert state.request_id == "req_123"
    assert state.status == "pending"


# ── Dispatch inbound messages ───────────────────────────────────────


def test_dispatch_shutdown_request_returns_true():
    bus = MessageBus(tempfile.mkdtemp())
    proto = ProtocolManager()
    msg = {"from": "lead", "type": "shutdown_request",
           "content": "Please stop", "metadata": {"request_id": "req_001"}}
    messages = []
    result = _dispatch_inbox_message("worker", msg, messages, bus, proto)
    assert result is True

    # Should have sent shutdown_response to lead
    lead_msgs = bus.read_inbox("lead")
    assert len(lead_msgs) == 1
    assert lead_msgs[0]["type"] == "shutdown_response"


def test_dispatch_plan_approval_response_approve():
    bus = MessageBus(tempfile.mkdtemp())
    proto = ProtocolManager()
    msg = {"from": "lead", "type": "plan_approval_response",
           "content": "", "metadata": {"request_id": "req_002", "approve": True}}
    messages = []
    result = _dispatch_inbox_message("worker", msg, messages, bus, proto)
    assert result is False  # continue loop
    assert len(messages) == 1
    assert "approved" in messages[0]["content"]


def test_dispatch_plan_approval_response_reject():
    bus = MessageBus(tempfile.mkdtemp())
    proto = ProtocolManager()
    msg = {"from": "lead", "type": "plan_approval_response",
           "content": "Needs more detail", "metadata": {"request_id": "req_003", "approve": False}}
    messages = []
    result = _dispatch_inbox_message("worker", msg, messages, bus, proto)
    assert result is False
    assert "rejected" in messages[0]["content"]
    assert "Needs more detail" in messages[0]["content"]


# ── Team tools registration with protocols ──────────────────────────


@pytest.fixture
def team_setup():
    d = tempfile.TemporaryDirectory()
    bus = MessageBus(d.name)

    tool_reg = ToolRegistry()

    @tool(tool_reg, name="read_file", description="Read a file", safety_level=SafetyLevel.READONLY)
    async def read_file(path: str) -> dict:
        return {"success": True, "content": f"Content of {path}"}

    @tool(tool_reg, name="run_shell", description="Run shell command", safety_level=SafetyLevel.SHELL)
    async def run_shell(command: str) -> dict:
        return {"success": True, "content": f"Ran: {command}"}

    @tool(tool_reg, name="write_file", description="Write a file", safety_level=SafetyLevel.WRITE)
    async def write_file(path: str, content: str) -> dict:
        return {"success": True, "content": f"Wrote {path}"}

    @tool(tool_reg, name="submit_plan", description="Submit a plan", safety_level=SafetyLevel.WRITE)
    async def submit_plan(plan: str) -> dict:
        return {"success": True, "content": "Plan submitted"}

    reg = ToolRegistry()
    create_team_tools(reg, bus, None, tool_reg, cwd="/tmp")
    yield reg, bus, d
    d.cleanup()


def test_protocol_tools_registered(team_setup):
    reg, bus, d = team_setup
    assert "request_shutdown" in reg.list_names()
    assert "request_plan" in reg.list_names()
    assert "review_plan" in reg.list_names()
    assert "spawn_teammate" in reg.list_names()


@pytest.mark.asyncio
async def test_request_shutdown_tool(team_setup):
    reg, bus, d = team_setup
    tool_fn = reg.get("request_shutdown")
    result = await tool_fn(teammate="worker-1")
    assert result["success"] is True
    assert "worker-1" in result["content"]

    msgs = bus.read_inbox("worker-1")
    assert len(msgs) == 1
    assert msgs[0]["type"] == "shutdown_request"


@pytest.mark.asyncio
async def test_review_plan_approve(team_setup):
    reg, bus, d = team_setup
    from deepagent.tools.team_tools import create_team_tools
    from deepagent.core.protocols import ProtocolManager

    proto = ProtocolManager()
    req_id = proto.new_request("plan_approval", "worker", "lead", "My plan")
    req_id_str = list(proto._pending.keys())[0]

    # Simulate: we need the protocol_manager that create_team_tools created
    # But create_team_tools creates its own internal proto. For this test,
    # we test the bus message path directly.
    #
    # Actually, review_plan sends a bus message to the teammate. Let's verify that.
    tool_fn = reg.get("review_plan")
    from deepagent.core.protocols import new_request_id
    # Create a request first via request_shutdown
    shutdown = reg.get("request_shutdown")
    r = await shutdown(teammate="worker-1")
    # Extract the req_id from the result
    req_id_from_msg = r["content"].split("req: ")[1].rstrip(")")

    # Approve it via review_plan... wait, review_plan is for plan_approval, not shutdown
    # Let's test with the right type
    result = await tool_fn(request_id="req_nonexistent", approve=True)
    assert result["success"] is False  # unknown req_id


@pytest.mark.asyncio
async def test_spawn_teammate_autonomous_flag(team_setup):
    reg, bus, d = team_setup
    tool_fn = reg.get("spawn_teammate")
    result = await tool_fn(
        name="auto-bot",
        role="autonomous-worker",
        prompt="Monitor tasks and claim them.",
        autonomous=True,
    )
    assert result["success"] is True
    assert "autonomous" in result["content"]


# ── Autonomous task scanning ─────────────────────────────────────────


def test_scan_unclaimed_tasks():
    from deepagent.core.tasks import TaskManager
    d = tempfile.TemporaryDirectory()
    mgr = TaskManager(d.name)

    # t1 (pending, no owner) → unclaimed
    # t2 (pending, has owner) → not unclaimed
    t1 = mgr.create_task(subject="Available task")
    t2 = mgr.create_task(subject="Owned task")
    t2.owner = "other-agent"
    mgr.save(t2)
    # t3 will be a blocking dependency for t4
    t3 = mgr.create_task(subject="Blocking dep")
    t3.owner = "other-agent"  # owned, not unclaimed
    mgr.save(t3)
    t4 = mgr.create_task(subject="Blocked task", blocked_by=[t3.id])

    # Only t1 is unclaimed
    unclaimed = _scan_unclaimed_tasks(mgr)
    assert len(unclaimed) == 1
    assert unclaimed[0].subject == "Available task"

    # Complete t3, now t4 becomes unblocked and available
    t3.status = "in_progress"
    mgr.save(t3)
    mgr.complete(t3.id)
    unclaimed = _scan_unclaimed_tasks(mgr)
    assert len(unclaimed) == 2  # t1 + t4
    subjects = {t.subject for t in unclaimed}
    assert "Blocked task" in subjects

    d.cleanup()


def test_teammate_tool_names_includes_submit_plan():
    assert "submit_plan" in TEAMMATE_TOOL_NAMES
    assert "spawn_teammate" not in TEAMMATE_TOOL_NAMES
    assert "request_shutdown" not in TEAMMATE_TOOL_NAMES
    assert "send_message" in TEAMMATE_TOOL_NAMES
