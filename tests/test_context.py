import pytest
from deepagent.core.context import ContextManager, TokenBudget
from deepagent.core.events import ToolResult


def test_token_budget_defaults():
    b = TokenBudget()
    assert b.max_tokens == 980_000
    assert b.safety_margin == 20_000
    assert b.effective_limit == 960_000


def test_token_budget_custom():
    b = TokenBudget(max_tokens=100_000, safety_margin=5_000)
    assert b.effective_limit == 95_000


def test_context_manager_starts_empty():
    cm = ContextManager()
    assert cm.get_messages() == []
    assert cm.message_count == 0
    assert cm.prompt_tokens == 0
    assert cm.completion_tokens == 0


def test_add_user_message():
    cm = ContextManager()
    cm.add_user_message("Hello")
    msgs = cm.get_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"


def test_add_assistant_message_plain():
    cm = ContextManager()
    cm.add_assistant_message("Hi there")
    msgs = cm.get_messages()
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Hi there"


def test_add_assistant_message_with_tool_calls():
    from deepagent.core.events import ToolCall

    cm = ContextManager()
    tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/test.txt"})
    cm.add_assistant_message("Let me read that.", tool_calls=[tc])

    msgs = cm.get_messages()
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "Let me read that."
    assert "tool_calls" in msgs[0]
    assert msgs[0]["tool_calls"][0]["id"] == "call_1"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "read_file"


def test_add_assistant_message_with_reasoning():
    cm = ContextManager()
    cm.add_assistant_message("ok", reasoning_content="Let me think...")
    msgs = cm.get_messages()
    assert msgs[0]["reasoning_content"] == "Let me think..."

    # Without reasoning_content
    cm2 = ContextManager()
    cm2.add_assistant_message("ok")
    assert "reasoning_content" not in cm2.get_messages()[0]


def test_add_tool_result_success():
    cm = ContextManager()
    result = ToolResult(success=True, content="file contents here")
    cm.add_tool_result("call_1", result)
    msgs = cm.get_messages()
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "file contents here"


def test_add_tool_result_error():
    cm = ContextManager()
    result = ToolResult(success=False, content="", error="File not found")
    cm.add_tool_result("call_1", result)
    msgs = cm.get_messages()
    assert "Error: File not found" in msgs[0]["content"]


def test_add_tool_result_truncates_long_content():
    cm = ContextManager()
    long_content = "x" * 30_000
    result = ToolResult(success=True, content=long_content)
    cm.add_tool_result("call_1", result)
    msgs = cm.get_messages()
    content = msgs[0]["content"]
    assert len(content) < 30_000
    assert "chars truncated" in content


def test_add_tool_result_does_not_truncate_short_content():
    cm = ContextManager()
    result = ToolResult(success=True, content="short reply")
    cm.add_tool_result("call_1", result)
    msgs = cm.get_messages()
    assert msgs[0]["content"] == "short reply"


def test_system_prompt_prepended():
    cm = ContextManager(system_prompt="You are a helpful assistant.")
    cm.add_user_message("hi")
    msgs = cm.get_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are a helpful assistant."


def test_get_messages_without_system():
    cm = ContextManager(system_prompt="Be helpful.")
    cm.add_user_message("hi")
    msgs = cm.get_messages(with_system=False)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_multiple_messages_ordered():
    cm = ContextManager()
    cm.add_user_message("q1")
    cm.add_assistant_message("a1")
    cm.add_user_message("q2")
    cm.add_assistant_message("a2")
    msgs = cm.get_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_update_usage():
    cm = ContextManager()
    cm.update_usage(100, 50)
    assert cm.prompt_tokens == 100
    assert cm.completion_tokens == 50
    assert cm.total_tokens == 150

    cm.update_usage(200, 100)
    assert cm.prompt_tokens == 300
    assert cm.completion_tokens == 150
    assert cm.total_tokens == 450


def test_clear_resets_state():
    cm = ContextManager()
    cm.add_user_message("hi")
    cm.update_usage(100, 50)
    cm.clear()
    assert cm.message_count == 0
    assert cm.prompt_tokens == 0
    assert cm.completion_tokens == 0


def test_estimated_tokens():
    cm = ContextManager()
    # ~40 chars of content = roughly 10 tokens at 4 chars/token
    cm.add_user_message("hello world " * 3)  # ~36 chars
    assert cm.estimated_tokens > 0
    assert cm.estimated_tokens < 50


def test_is_near_limit_false_initially():
    cm = ContextManager()
    assert not cm.is_near_limit()


def test_is_near_limit_with_huge_content():
    cm = ContextManager(token_budget=TokenBudget(max_tokens=1000, safety_margin=100))
    cm.add_user_message("x" * 10000)  # ~2500 estimated tokens
    assert cm.is_near_limit()


# ── compression ───────────────────────────────────────────────────


def test_compression_candidates_returns_zero_for_few_messages():
    cm = ContextManager()
    cm.add_user_message("hi")
    cm.add_assistant_message("hello")
    assert cm.compression_candidates() == 0


def test_compression_candidates_with_many_messages():
    cm = ContextManager()
    for i in range(12):
        cm.add_user_message(f"q{i}")
        cm.add_assistant_message(f"a{i}")
    # 24 messages, boundary = 24 // 3 = 8
    assert cm.compression_candidates() == 8


def test_compress_to_replaces_old_messages():
    cm = ContextManager()
    for i in range(6):
        cm.add_user_message(f"q{i}")
        cm.add_assistant_message(f"a{i}")
    # 12 messages, boundary = 4
    boundary = cm.compression_candidates()
    assert boundary == 4

    original_count = cm.message_count
    cm.compress_to(boundary, "Summary of first two turns.")

    # Old 4 messages replaced by 1 summary + remaining 8 = 9 total
    assert cm.message_count == original_count - boundary + 1
    msgs = cm.get_messages(with_system=False)
    assert "Earlier conversation summary" in msgs[0]["content"]
    assert "Summary of first two turns." in msgs[0]["content"]
    # Remaining messages should start from index boundary
    assert msgs[1]["content"] == "q2"


def test_compress_to_does_nothing_if_boundary_invalid():
    cm = ContextManager()
    cm.add_user_message("q0")
    cm.add_assistant_message("a0")
    cm.compress_to(0, "summary")
    assert cm.message_count == 2
    cm.compress_to(10, "summary")
    assert cm.message_count == 2


def test_estimate_compression_savings():
    cm = ContextManager()
    for i in range(9):
        cm.add_user_message(f"q{i}")
        cm.add_assistant_message(f"a{i}")
    # 18 messages, boundary = 6
    savings = cm.estimate_compression_savings()
    assert savings > 0


def test_estimate_compression_savings_zero_for_few_messages():
    cm = ContextManager()
    cm.add_user_message("hi")
    assert cm.estimate_compression_savings() == 0
