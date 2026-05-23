# tests/test_llm_client.py
import pytest
from deepagent.config import Config
from deepagent.core.llm_client import LLMClient
from deepagent.core.events import TextDelta, ToolCallEvent, ToolCall


def make_config(**kwargs):
    """Create a Config that bypasses env lookup."""
    env = {"DEEPSEEK_API_KEY": "sk-test"}
    env.update(kwargs)
    return Config(_env=env)


def test_llm_client_initialization():
    cfg = make_config()
    client = LLMClient(cfg)
    assert client.config is cfg


@pytest.mark.asyncio
async def test_stream_chat_yields_text_deltas():
    """Integration test: requires DEEPSEEK_API_KEY in env."""
    cfg = make_config()  # uses test key; real API call requires valid key
    client = LLMClient(cfg)
    messages = [{"role": "user", "content": "Say exactly: hello world"}]

    events = []
    async for event in client.stream_chat(messages):
        events.append(event)

    # Should have at least one TextDelta
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    assert len(text_deltas) > 0, f"Expected TextDelta events, got: {events}"

    # Full response should contain "hello world"
    full_text = "".join(d.text for d in text_deltas)
    assert "hello world" in full_text.lower()


@pytest.mark.asyncio
async def test_stream_chat_with_tool_calling():
    """Integration test: LLM should return tool calls when appropriate."""
    cfg = Config()
    client = LLMClient(cfg)
    messages = [{"role": "user", "content": "What is the weather in Beijing?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    events = []
    async for event in client.stream_chat(messages, tools=tools):
        events.append(event)

    # Should yield at least one ToolCallEvent with get_weather
    tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tool_events) > 0, f"Expected ToolCallEvent, got: {events}"
    all_tool_names = [tc.name for ev in tool_events for tc in ev.tool_calls]
    assert "get_weather" in all_tool_names
