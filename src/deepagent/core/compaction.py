"""Context compaction — multi-layer message pruning and LLM summarization.

L1 (existing in context.py): tool_result > 20K chars → head + tail truncation.
L2 (new): message count > 50 → keep first 3, last 47, placeholder middle.
L3 (new): per-turn — only last 3 tool_results kept verbatim; older summarized.
L4 (improved): tokens > 90% budget → LLM summary, last 5 messages preserved.
Emergency: prompt_too_long error → summarize ALL history, keep last 5.

Transcript saving: .transcripts/{timestamp}.jsonl on compaction or session end.

Reference: learn-claude-code s08_context_compact. Extended with multi-layer
approach and transcript persistence.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Awaitable

logger = logging.getLogger(__name__)

# ── Limits ─────────────────────────────────────────────────────────

MAX_MESSAGES_L2 = 50
KEEP_FIRST_L2 = 3
KEEP_LAST_L2 = 47
KEEP_VERBATIM_TOOL_RESULTS_L3 = 3
EMERGENCY_KEEP_LAST = 5
BUDGET_RATIO_L4 = 0.9  # 90% of budget triggers L4

L2_PLACEHOLDER = "[Earlier messages trimmed — keep first {first} + last {last} of {total} messages]"
L3_PLACEHOLDER = "[Earlier tool result summarized]"
EMERGENCY_PLACEHOLDER = "[All earlier conversation summarized to save context space]"


# ── LLM summarizer type ───────────────────────────────────────────

# Called with messages to summarize and a prompt; returns summary string.
LLMSummarizer = Callable[[list[dict], str], Awaitable[str]]


# ── Result ─────────────────────────────────────────────────────────


@dataclass
class CompactionResult:
    messages: list[dict]
    layers_applied: list[str] = field(default_factory=list)
    tokens_saved: int = 0
    summary: str | None = None  # populated for L4/Emergency


# ═══════════════════════════════════════════════════════════════════


class Compactor:
    """Multi-layer message compactor.

    Usage::

        compactor = Compactor(summarizer=my_llm_summarizer)
        ctx = ContextManager()
        # ... many turns later ...
        if estimated_tokens > budget * 0.9:
            result = await compactor.compact(ctx.get_messages(), budget)
            if result.summary:
                ctx.compress_to(result.messages, result.summary)
    """

    def __init__(
        self,
        summarizer: LLMSummarizer | None = None,
        transcript_dir: str | Path | None = None,
    ):
        self.summarizer = summarizer
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self._compaction_count = 0

    # ── Public API ─────────────────────────────────────────────────

    async def compact(
        self,
        messages: list[dict],
        token_budget: int,
        estimated_tokens: int,
    ) -> CompactionResult:
        """Run compaction with required layers. Returns modified messages.

        Determines which layers to apply based on message count and token usage,
        then applies them in order (L2 → L3 → L4).
        """
        result = CompactionResult(messages=list(messages))

        # L2: message count threshold
        if len(result.messages) > MAX_MESSAGES_L2:
            result = self._compact_l2(result)

        # L3: old tool result summarization
        result = self._compact_l3(result)

        # L4: token budget exceeded
        budget_threshold = int(token_budget * BUDGET_RATIO_L4)
        if estimated_tokens > budget_threshold and self.summarizer is not None:
            result = await self._compact_l4(result, token_budget)

        if result.layers_applied:
            self._compaction_count += 1
            self._save_transcript(result.messages)

        return result

    async def emergency_compact(
        self,
        messages: list[dict],
    ) -> CompactionResult:
        """Prompt-too-long recovery. Summarize ALL history, keep last 5 messages."""
        if self.summarizer is None:
            return CompactionResult(
                messages=messages[-EMERGENCY_KEEP_LAST:],
                layers_applied=["emergency_truncation"],
            )

        old_count = len(messages)
        to_summarize = messages[:-EMERGENCY_KEEP_LAST]
        keep = messages[-EMERGENCY_KEEP_LAST:]

        summary = await self.summarizer(
            to_summarize,
            "Summarize this conversation history concisely. Keep all key facts, "
            "decisions, file paths, and errors. The summary will replace the "
            "original messages to save context space.",
        )

        result = CompactionResult(
            messages=[{"role": "user", "content": EMERGENCY_PLACEHOLDER + "\n" + summary}]
            + keep,
            layers_applied=["emergency"],
            tokens_saved=max(0, old_count - len(keep) - 1),
            summary=summary,
        )
        self._save_transcript(result.messages)
        return result

    # ── Layer implementations ──────────────────────────────────────

    @staticmethod
    def _compact_l2(result: CompactionResult) -> CompactionResult:
        """Keep first KEEP_FIRST_L2 + last KEEP_LAST_L2 messages."""
        msgs = result.messages
        total = len(msgs)
        if total <= MAX_MESSAGES_L2:
            return result

        placeholder = {
            "role": "user",
            "content": L2_PLACEHOLDER.format(
                first=KEEP_FIRST_L2, last=KEEP_LAST_L2, total=total
            ),
        }
        kept = (
            msgs[:KEEP_FIRST_L2]
            + [placeholder]
            + msgs[-KEEP_LAST_L2:]
        )
        return CompactionResult(
            messages=kept,
            layers_applied=result.layers_applied + ["L2"],
            tokens_saved=total - len(kept),
        )

    @staticmethod
    def _compact_l3(result: CompactionResult) -> CompactionResult:
        """Keep only last KEEP_VERBATIM_TOOL_RESULTS_L3 tool results verbatim.

        Older tool results are replaced with a placeholder. Non-tool messages
        are left intact.
        """
        msgs = result.messages
        tool_indices = [
            i for i, m in enumerate(msgs)
            if m.get("role") == "tool"
        ]

        if len(tool_indices) <= KEEP_VERBATIM_TOOL_RESULTS_L3:
            return result

        # Indices to summarize (all except the last N)
        to_summarize = set(tool_indices[:-KEEP_VERBATIM_TOOL_RESULTS_L3])

        new_msgs = []
        for i, m in enumerate(msgs):
            if i in to_summarize:
                new_msgs.append({
                    "role": "tool",
                    "tool_call_id": m.get("tool_call_id", "unknown"),
                    "content": L3_PLACEHOLDER,
                })
            else:
                new_msgs.append(m)

        saved = len(to_summarize)
        return CompactionResult(
            messages=new_msgs,
            layers_applied=result.layers_applied + ["L3"],
            tokens_saved=result.tokens_saved + saved,
        )

    async def _compact_l4(
        self,
        result: CompactionResult,
        token_budget: int,
    ) -> CompactionResult:
        """LLM-generated summary of oldest messages, keeping last 5 verbatim."""
        if self.summarizer is None:
            return result

        msgs = result.messages
        if len(msgs) <= EMERGENCY_KEEP_LAST + 3:
            return result  # not enough messages to summarize

        to_summarize = msgs[:-EMERGENCY_KEEP_LAST]
        keep = msgs[-EMERGENCY_KEEP_LAST:]

        summary = await self.summarizer(
            to_summarize,
            "Summarize this conversation concisely. Preserve: file paths changed, "
            "key decisions, errors encountered, and current task state. Omit: "
            "exact tool output text, verbose thinking, repetitive patterns.",
        )

        new_msgs = [{
            "role": "user",
            "content": (
                f"[Earlier conversation compressed to save ~{len(to_summarize)} turns. "
                "The following is a condensed summary of previous interactions.]\n"
                + summary
            ),
        }] + keep

        return CompactionResult(
            messages=new_msgs,
            layers_applied=result.layers_applied + ["L4"],
            tokens_saved=result.tokens_saved + len(to_summarize) - 1,
            summary=summary,
        )

    # ── Transcript saving ──────────────────────────────────────────

    def _save_transcript(self, messages: list[dict]) -> None:
        if self.transcript_dir is None:
            return
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"{timestamp}_{self._compaction_count:03d}.jsonl"
            path = self.transcript_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            logger.info("Transcript saved: %s (%d messages)", path, len(messages))
        except OSError as e:
            logger.warning("Failed to save transcript: %s", e)

    async def save_session_transcript(self, messages: list[dict]) -> str | None:
        """Save final session transcript. Returns the file path or None."""
        if self.transcript_dir is None:
            return None
        try:
            self.transcript_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"session_{timestamp}.jsonl"
            path = self.transcript_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                for msg in messages:
                    f.write(json.dumps(msg, ensure_ascii=False) + "\n")
            return str(path)
        except OSError as e:
            logger.warning("Failed to save session transcript: %s", e)
            return None
