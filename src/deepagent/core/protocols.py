"""Team Protocols — request-response protocol manager with timeout enforcement.

Tracks pending protocol requests (shutdown, plan_approval, code_review),
validates response types match request types, and enforces timeouts on
in-flight requests.

Reference: learn-claude-code s16_team_protocols. Extended with asyncio
timeout support.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Protocol request → response type mapping for validation
PROTOCOL_RESPONSE_MAP: dict[str, str] = {
    "shutdown_request": "shutdown_response",
    "plan_approval_request": "plan_approval_response",
    "code_review_request": "code_review_response",
}

DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class ProtocolState:
    """State of an in-flight protocol request."""

    request_id: str
    type: str          # "shutdown" | "plan_approval" | "code_review"
    sender: str        # who sent the request
    target: str        # who should respond
    status: str        # "pending" | "approved" | "rejected" | "timeout"
    payload: str       # plan text / code content / shutdown reason
    created_at: float = field(default_factory=time.time)


def new_request_id() -> str:
    return f"req_{random.randint(0, 999999):06d}"


class ProtocolManager:
    """Manages in-flight protocol requests with timeout enforcement.

    Usage::

        proto = ProtocolManager(timeout_seconds=60)
        req_id = proto.new_request("shutdown", "lead", "worker-1")
        # ... later, in teammate:
        proto.dispatch("shutdown_request", req_id, approve=True)
        # ... in lead:
        proto.match_response("shutdown_response", req_id, approve=True)
    """

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self._pending: dict[str, ProtocolState] = {}
        self._timeout = timeout_seconds

    # ── Request lifecycle ──────────────────────────────────────────

    def new_request(
        self,
        req_type: str,
        sender: str,
        target: str,
        payload: str = "",
    ) -> str:
        """Create and track a new protocol request. Returns request_id."""
        req_id = new_request_id()
        self._pending[req_id] = ProtocolState(
            request_id=req_id,
            type=req_type,
            sender=sender,
            target=target,
            status="pending",
            payload=payload,
        )
        return req_id

    def match_response(
        self,
        response_type: str,
        request_id: str,
        approve: bool,
    ) -> ProtocolState | None:
        """Correlate a response to the original request.

        Validates that response_type matches the expected type for the request.
        Returns the updated ProtocolState or None on failure.
        """
        state = self._pending.get(request_id)
        if state is None:
            logger.warning("Unknown request_id: %s", request_id)
            return None

        expected = PROTOCOL_RESPONSE_MAP.get(state.type + "_request", "")
        if not response_type.endswith("_response"):
            logger.warning("Not a response type: %s", response_type)
            return None
        expected_response = state.type + "_response"
        if response_type != expected_response:
            logger.warning(
                "Type mismatch: expected %s, got %s", expected_response, response_type
            )
            return None

        if state.status != "pending":
            logger.info("%s already %s, ignoring duplicate", request_id, state.status)
            return state

        state.status = "approved" if approve else "rejected"
        return state

    def get_state(self, request_id: str) -> ProtocolState | None:
        return self._pending.get(request_id)

    def remove(self, request_id: str) -> ProtocolState | None:
        return self._pending.pop(request_id, None)

    # ── Timeout enforcement ────────────────────────────────────────

    def check_timeouts(self) -> list[ProtocolState]:
        """Check for timed-out requests. Returns list of timed-out states.

        Timed-out requests are transitioned to 'timeout' status and
        removed from pending tracking.
        """
        now = time.time()
        timed_out = []
        for req_id, state in list(self._pending.items()):
            if state.status == "pending" and (now - state.created_at) > self._timeout:
                state.status = "timeout"
                timed_out.append(state)
                self._pending.pop(req_id, None)
                logger.warning("Protocol request %s timed out", req_id)
        return timed_out

    async def wait_for_response(
        self,
        request_id: str,
        timeout_seconds: float | None = None,
    ) -> ProtocolState | None:
        """Wait for a response to a specific request, with timeout.

        Returns the resolved ProtocolState, or None on timeout.
        """
        timeout = timeout_seconds or self._timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = self._pending.get(request_id)
            if state is None or state.status != "pending":
                return state
            await asyncio.sleep(0.1)
        # Timeout
        state = self._pending.pop(request_id, None)
        if state is not None:
            state.status = "timeout"
        return state

    # ── Introspection ──────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        return sum(1 for s in self._pending.values() if s.status == "pending")

    def list_pending(self) -> list[ProtocolState]:
        return [s for s in self._pending.values() if s.status == "pending"]
