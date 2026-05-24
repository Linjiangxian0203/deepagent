"""Context management with token budget and message window.

DeepSeek V4 has a 1M token context window. We reserve ~2% as safety margin.
The ContextManager owns the message list across multi-turn ReAct iterations,
truncates long tool results, and tracks cumulative token usage from UsageEvents.
"""

from dataclasses import dataclass, field
import json

from deepagent.core.events import ToolResult
from deepagent.core.compaction import Compactor, CompactionResult, MAX_MESSAGES_L2


@dataclass
class TokenBudget:
    max_tokens: int = 980_000
    safety_margin: int = 20_000

    @property
    def effective_limit(self) -> int:
        return self.max_tokens - self.safety_margin


class ContextManager:
    """Owns the conversation message list across ReAct iterations.

    Responsibilities:
    - Store and retrieve messages for LLM API calls
    - Append assistant messages with tool_calls and reasoning_content
    - Truncate long tool results (first 10K + last 10K chars)
    - Track cumulative token usage from UsageEvents
    - Inject system prompt at the front of the message list
    """

    def __init__(
        self,
        system_prompt: str = "",
        token_budget: TokenBudget | None = None,
        compactor: Compactor | None = None,
    ):
        self._messages: list[dict] = []
        self._system_prompt = system_prompt
        self._budget = token_budget or TokenBudget()
        self._compactor = compactor
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    # ── message builders ──────────────────────────────────────────

    def add_user_message(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(
        self,
        content: str,
        *,
        tool_calls: list | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ]
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        self._messages.append(msg)

    def add_tool_result(self, call_id: str, result: ToolResult) -> None:
        content = result.content if result.success else f"Error: {result.error}"
        if len(content) > 20_000:
            content = (
                content[:10_000]
                + f"\n... [{len(content) - 20_000} chars truncated] ...\n"
                + content[-10_000:]
            )
        self._messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": content}
        )

    # ── retrieval ─────────────────────────────────────────────────

    def get_messages(self, with_system: bool = True) -> list[dict]:
        msgs: list[dict] = []
        if with_system and self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        return msgs + self._messages

    def clear(self) -> None:
        self._messages.clear()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    # ── token tracking ────────────────────────────────────────────

    def update_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_tokens(self) -> int:
        total_chars = sum(
            len(str(m.get("content", ""))) + len(str(m.get("tool_calls", "")))
            for m in self._messages
        )
        return total_chars // 4

    def is_near_limit(self) -> bool:
        return self.estimated_tokens > self._budget.effective_limit

    @property
    def message_count(self) -> int:
        return len(self._messages)

    # ── L2/L3 compaction (Phase 1) ─────────────────────────────────

    def compact_l2_l3(self) -> bool:
        """Apply L2 and L3 compaction in-place on the message list.

        L2: if message count > 50, keep first 3 + last 47, placeholder middle.
        L3: if > 3 tool results, keep last 3 verbatim, replace older with placeholder.

        Returns True if any compaction was applied. Does NOT require LLM summarizer.
        """
        if self._compactor is None:
            return False

        applied = False

        # L2: message count threshold
        if len(self._messages) > MAX_MESSAGES_L2:
            result = Compactor._compact_l2(CompactionResult(messages=self._messages))
            self._messages = result.messages
            applied = True

        # L3: old tool result summarization
        result = Compactor._compact_l3(CompactionResult(messages=self._messages))
        if "L3" in result.layers_applied:
            self._messages = result.messages
            applied = True

        return applied

    # ── compression ────────────────────────────────────────────────

    def compression_candidates(self) -> int:
        """Return the index boundary for compression.

        Messages at indices [0, boundary) are the oldest ~1/3 and may be
        summarized. Returns 0 if there aren't enough messages to compress
        (fewer than 6 messages means less than one full turn).
        """
        if len(self._messages) < 6:
            return 0
        return len(self._messages) // 3

    def compress_to(self, keep_from_idx: int, summary: str) -> None:
        """Replace messages [0, keep_from_idx) with a single summary message.

        The summary is inserted as a system-level user message that preserves
        context from earlier turns without consuming as many tokens.
        """
        if keep_from_idx <= 0 or keep_from_idx > len(self._messages):
            return
        self._messages = (
            [{"role": "user", "content": f"[Earlier conversation summary]\n{summary}"}]
            + self._messages[keep_from_idx:]
        )

    def estimate_compression_savings(self) -> int:
        """Estimate how many tokens would be saved by compression.

        Returns 0 if no compression is needed.
        """
        boundary = self.compression_candidates()
        if boundary <= 0:
            return 0
        old_chars = sum(
            len(str(self._messages[i].get("content", "")))
            for i in range(boundary)
        )
        return old_chars // 4
