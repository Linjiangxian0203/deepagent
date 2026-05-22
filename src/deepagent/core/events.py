from dataclasses import dataclass, field


@dataclass
class TextDelta:
    """A chunk of streaming text from the assistant."""
    text: str


@dataclass
class ThinkingDelta:
    """A chunk of reasoning/thinking content (DeepSeek R1 reasoning_content)."""
    text: str


@dataclass
class ToolCall:
    """A complete tool call assembled from streaming deltas."""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolCallEvent:
    """One or more tool calls emitted in a single LLM response."""
    tool_calls: list[ToolCall]


@dataclass
class ToolCallStartEvent:
    """Emitted before a tool starts executing."""
    tool_call: ToolCall


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    content: str
    error: str | None = None
    metadata: dict | None = None


@dataclass
class ToolResultEvent:
    """Emitted after a tool completes execution."""
    tool_call: ToolCall
    result: ToolResult


@dataclass
class ToolLimitEvent:
    """Emitted when too many tool calls are requested."""
    pass


@dataclass
class InterruptedEvent:
    """Emitted when the user interrupts (Ctrl+C)."""
    pass


@dataclass
class DoneEvent:
    """Emitted when the agent loop completes."""
    pass


# Union type for all events
AgentEvent = (
    TextDelta
    | ThinkingDelta
    | ToolCallEvent
    | ToolCallStartEvent
    | ToolResultEvent
    | ToolLimitEvent
    | InterruptedEvent
    | DoneEvent
)
