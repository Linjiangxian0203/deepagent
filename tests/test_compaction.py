"""Tests for context compaction (Phase 1).

Covers: L2 message count trim, L3 tool result summarization,
L4 LLM summary, emergency compact, and transcript saving.
"""

import json
import tempfile
from pathlib import Path

import pytest

from deepagent.core.compaction import (
    Compactor,
    CompactionResult,
    MAX_MESSAGES_L2,
    L2_PLACEHOLDER,
    L3_PLACEHOLDER,
    EMERGENCY_PLACEHOLDER,
)


# ── Helpers ───────────────────────────────────────────────────────


def make_messages(count: int) -> list[dict]:
    """Generate `count` user/assistant message pairs."""
    msgs = []
    for i in range(count):
        msgs.append({"role": "user", "content": f"query {i}"})
        msgs.append({"role": "assistant", "content": f"response {i}"})
    return msgs


def make_tool_messages(tool_count: int) -> list[dict]:
    """Generate messages with tool results."""
    msgs = []
    for i in range(tool_count):
        msgs.append({"role": "user", "content": f"cmd {i}"})
        msgs.append({
            "role": "assistant",
            "content": f"Running tool {i}",
            "tool_calls": [{"id": f"tc_{i}", "type": "function"}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"tc_{i}",
            "content": f"Result {i}: some output data for tool call {i}",
        })
    msgs.append({"role": "assistant", "content": "All done."})
    return msgs


async def fake_summarizer(messages, prompt) -> str:
    return f"Summary of {len(messages)} messages."


# ── L2: Message count threshold ───────────────────────────────────


def test_l2_does_not_trigger_below_threshold():
    compactor = Compactor()
    msgs = make_messages(10)  # 20 messages, < 50
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    assert "L2" not in result.layers_applied
    assert len(result.messages) == 20


def test_l2_triggers_above_threshold():
    compactor = Compactor()
    msgs = make_messages(30)  # 60 messages > 50
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    assert "L2" in result.layers_applied
    # 3 (first) + 1 (placeholder) + 47 (last) = 51
    assert len(result.messages) == 3 + 1 + 47


def test_l2_placeholder_is_injected():
    compactor = Compactor()
    msgs = make_messages(30)
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    placeholder_msg = result.messages[3]  # after first 3
    assert placeholder_msg["role"] == "user"
    assert "Earlier messages trimmed" in placeholder_msg["content"]


def test_l2_keeps_first_and_last():
    compactor = Compactor()
    msgs = make_messages(30)
    result = compactor._compact_l2(CompactionResult(messages=msgs))

    # First message should be "query 0"
    assert result.messages[0]["content"] == "query 0"
    # Last message should be "response 29"
    assert result.messages[-1]["content"] == "response 29"


# ── L3: Old tool result summarization ──────────────────────────────


def test_l3_does_not_trigger_with_few_tool_results():
    compactor = Compactor()
    msgs = make_tool_messages(2)  # 2 tool results, below 3
    result = compactor._compact_l3(CompactionResult(messages=msgs))
    assert "L3" not in result.layers_applied


def test_l3_replaces_old_tool_results():
    compactor = Compactor()
    msgs = make_tool_messages(5)  # 5 tool results, keep last 3
    result = compactor._compact_l3(CompactionResult(messages=msgs))

    assert "L3" in result.layers_applied

    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 5  # still 5 tool messages

    # First 2 are replaced with placeholder
    for i in range(2):
        assert tool_msgs[i]["content"] == L3_PLACEHOLDER

    # Last 3 are verbatim
    for i in range(2, 5):
        assert "Result" in tool_msgs[i]["content"]


def test_l3_preserves_tool_call_ids():
    compactor = Compactor()
    msgs = make_tool_messages(5)
    result = compactor._compact_l3(CompactionResult(messages=msgs))

    tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
    # All 5 have their tool_call_ids preserved
    for i, t in enumerate(tool_msgs):
        assert t["tool_call_id"] == f"tc_{i}"


# ── L4: LLM summary ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l4_requires_summarizer():
    compactor = Compactor(summarizer=None)
    msgs = make_messages(20)
    result = CompactionResult(messages=msgs)
    final = await compactor._compact_l4(result, 100000)
    # Without summarizer, returns unchanged
    assert final is result


@pytest.mark.asyncio
async def test_l4_produces_summary():
    compactor = Compactor(summarizer=fake_summarizer)
    msgs = make_messages(20)  # 40 messages
    result = CompactionResult(messages=msgs)
    final = await compactor._compact_l4(result, 100000)

    assert "L4" in final.layers_applied
    assert final.summary is not None
    assert "Summary of" in final.summary
    # Last 5 messages preserved
    assert len(final.messages) == 1 + 5  # summary + last 5


@pytest.mark.asyncio
async def test_l4_skips_when_too_few_messages():
    compactor = Compactor(summarizer=fake_summarizer)
    msgs = make_messages(3)  # 6 messages, but < EMERGENCY_KEEP_LAST + 3
    result = CompactionResult(messages=msgs)
    final = await compactor._compact_l4(result, 100000)
    assert "L4" not in final.layers_applied


# ── Full compact pipeline ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_applies_multiple_layers():
    compactor = Compactor(summarizer=fake_summarizer)
    msgs = make_messages(30)  # 60 messages triggers L2
    result = await compactor.compact(msgs, token_budget=100000, estimated_tokens=95000)

    assert "L2" in result.layers_applied
    # L3 might not apply (no tool messages), L4 applies (95K > 90K budget)
    assert "L4" in result.layers_applied


@pytest.mark.asyncio
async def test_compact_does_not_apply_l4_below_budget():
    compactor = Compactor(summarizer=fake_summarizer)
    msgs = make_messages(10)  # 20 messages, below L2 threshold
    result = await compactor.compact(msgs, token_budget=100000, estimated_tokens=10000)
    # L2 not triggered, L4 not triggered (10K < 90K budget)
    assert result.layers_applied == [] or "L3" in result.layers_applied


# ── Emergency compact ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_emergency_compact_with_summarizer():
    compactor = Compactor(summarizer=fake_summarizer)
    msgs = make_messages(20)  # 40 messages total
    result = await compactor.emergency_compact(msgs)

    assert "emergency" in result.layers_applied
    assert result.summary is not None
    # 1 summary message + last 5 messages
    assert len(result.messages) == 1 + 5


@pytest.mark.asyncio
async def test_emergency_compact_without_summarizer():
    compactor = Compactor(summarizer=None)
    msgs = make_messages(20)
    result = await compactor.emergency_compact(msgs)

    assert "emergency_truncation" in result.layers_applied
    # Last 5 messages only
    assert len(result.messages) == 5


# ── Transcript saving ──────────────────────────────────────────────


def test_transcript_saving():
    with tempfile.TemporaryDirectory() as tmp:
        compactor = Compactor(transcript_dir=Path(tmp) / ".transcripts")
        msgs = make_messages(5)
        compactor._save_transcript(msgs)

        transcript_dir = Path(tmp) / ".transcripts"
        assert transcript_dir.exists()
        files = list(transcript_dir.glob("*.jsonl"))
        assert len(files) == 1
        content = files[0].read_text()
        assert "query 0" in content


@pytest.mark.asyncio
async def test_session_transcript_saving():
    with tempfile.TemporaryDirectory() as tmp:
        compactor = Compactor(transcript_dir=Path(tmp) / ".transcripts")
        msgs = make_messages(3)
        path_str = await compactor.save_session_transcript(msgs)
        assert path_str is not None
        assert Path(path_str).exists()
        content = Path(path_str).read_text()
        assert "query 0" in content


def test_no_transcript_when_dir_not_set():
    compactor = Compactor(transcript_dir=None)
    msgs = make_messages(5)
    compactor._save_transcript(msgs)  # should not raise


# ── Edge cases ─────────────────────────────────────────────────────


def test_l2_exactly_at_threshold():
    compactor = Compactor()
    msgs = make_messages(25)  # exactly 50 messages
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    # 50 is not > 50, so no trigger
    assert "L2" not in result.layers_applied


def test_l2_just_above_threshold():
    compactor = Compactor()
    msgs = make_messages(26)  # 52 messages > 50
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    assert "L2" in result.layers_applied


def test_compaction_result_tokens_saved():
    compactor = Compactor()
    msgs = make_messages(30)
    result = compactor._compact_l2(CompactionResult(messages=msgs))
    assert result.tokens_saved > 0


def test_compaction_result_layers_accumulate():
    c = Compactor()
    result = CompactionResult(messages=make_tool_messages(5))
    result.layers_applied.append("L2")
    result2 = c._compact_l3(result)
    assert "L2" in result2.layers_applied
    assert "L3" in result2.layers_applied
