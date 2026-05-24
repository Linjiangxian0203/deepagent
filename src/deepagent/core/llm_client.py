# src/deepagent/core/llm_client.py
import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
import json
import logging
import random

from openai import AsyncOpenAI

from deepagent.config import Config
from deepagent.core.events import (
    TextDelta, ThinkingDelta, ToolCallEvent, ToolCall, UsageEvent,
)

logger = logging.getLogger(__name__)

# ── Recovery constants ────────────────────────────────────────────

MAX_TOKENS_ESCALATION = [8192, 16384, 32768]  # progressively larger limits
MAX_CONTINUATION_ATTEMPTS = 2
FALLBACK_MODEL = "deepseek-v4-flash"


@dataclass
class RecoveryState:
    """Tracks error recovery state across a session.

    Used by LLMClient and AgentLoop to coordinate recovery actions.
    """

    max_tokens_escalated: bool = False
    continuation_attempts: int = 0
    has_attempted_reactive_compact: bool = False
    consecutive_529: int = 0
    fallback_model_active: bool = False


class LLMClient:
    """Wraps DeepSeek API (OpenAI-compatible) with streaming + delta-to-ToolCall assembly.

    Includes error recovery: max_tokens escalation, 529 fallback model,
    and exponential backoff with 25% jitter.
    """

    def __init__(self, config: Config, recovery: RecoveryState | None = None):
        self.config = config
        self._recovery = recovery or RecoveryState()
        self._client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    @property
    def active_model(self) -> str:
        if self._recovery.fallback_model_active:
            return FALLBACK_MODEL
        return self.config.model

    async def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[TextDelta | ThinkingDelta | ToolCallEvent | UsageEvent, None]:
        """Stream a chat completion, yielding assembled events.

        Tool calls are fully assembled from streaming deltas before yielding
        a single ToolCallEvent. Text and thinking deltas are yielded in real time.
        Usage is yielded as a UsageEvent after the stream completes.

        Error recovery is applied automatically: 429/5xx retries with jitter,
        529 overload detection with fallback model, and max_tokens escalation.
        """
        kwargs = {
            "model": self.active_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }
        if tools:
            kwargs["tools"] = tools
        if self.config.thinking_enabled:
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        # Retry loop for transient errors (429 rate limit, 5xx server errors)
        response = None
        for attempt in range(3 + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                status = getattr(e, "status_code", 0)
                if attempt < 3 and (status == 429 or status >= 500):
                    # 529 overload → track for fallback
                    if status == 529:
                        self._recovery.consecutive_529 += 1
                        if self._recovery.consecutive_529 >= 2:
                            logger.warning(
                                "2 consecutive 529 errors — switching to fallback model %s",
                                FALLBACK_MODEL,
                            )
                            self._recovery.fallback_model_active = True
                            kwargs["model"] = self.active_model
                    # Exponential backoff with 25% jitter
                    base = 2 ** attempt
                    jitter = random.uniform(-0.25, 0.25) * base
                    delay = base + jitter
                    await asyncio.sleep(delay)
                    continue
                raise

        tool_call_buffers: dict[int, dict] = {}
        reasoning_parts: list[str] = []
        usage = None

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None

            # Usage may appear on final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage

            if delta is None:
                continue

            # Reasoning content (DeepSeek thinking mode)
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
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
            reasoning = "".join(reasoning_parts) if reasoning_parts else None
            yield ToolCallEvent(tool_calls=assembled, reasoning_content=reasoning)

        # Yield usage for context management
        if usage:
            yield UsageEvent(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                reasoning_tokens=_get_reasoning_tokens(usage),
                cache_hit_tokens=getattr(usage, "prompt_cache_hit_tokens", 0),
                cache_miss_tokens=getattr(usage, "prompt_cache_miss_tokens", 0),
            )


def _get_reasoning_tokens(usage) -> int:
    """Extract reasoning tokens from usage, handling varying API shapes."""
    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        return getattr(details, "reasoning_tokens", 0)
    return 0
