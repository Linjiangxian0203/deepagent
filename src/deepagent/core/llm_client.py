# src/deepagent/core/llm_client.py
from collections.abc import AsyncGenerator
import json

from openai import AsyncOpenAI

from deepagent.config import Config
from deepagent.core.events import TextDelta, ThinkingDelta, ToolCallEvent, ToolCall


class LLMClient:
    """Wraps DeepSeek API (OpenAI-compatible) with streaming + delta-to-ToolCall assembly."""

    def __init__(self, config: Config):
        self.config = config
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[TextDelta | ThinkingDelta | ToolCallEvent, None]:
        """Stream a chat completion, yielding assembled events.

        Tool calls are fully assembled from streaming deltas before yielding
        a single ToolCallEvent. Text and thinking deltas are yielded in real time.
        """
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)

        # Per-tool-call accumulation state
        tool_call_buffers: dict[int, dict] = {}

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Reasoning content (DeepSeek R1)
            if getattr(delta, "reasoning_content", None):
                yield ThinkingDelta(text=delta.reasoning_content)

            # Text content
            if delta.content:
                yield TextDelta(text=delta.content)

            # Tool calls — accumulate across chunks
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    buf = tool_call_buffers[idx]
                    if tc_delta.id:
                        buf["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            buf["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            buf["arguments"] += tc_delta.function.arguments

        # After stream ends, yield assembled tool calls (design decision #2)
        if tool_call_buffers:
            assembled = []
            for idx in sorted(tool_call_buffers.keys()):
                buf = tool_call_buffers[idx]
                assembled.append(ToolCall(
                    id=buf["id"],
                    name=buf["name"],
                    arguments=json.loads(buf["arguments"]),
                ))
            yield ToolCallEvent(tool_calls=assembled)
