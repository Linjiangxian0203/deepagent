# src/deepagent/cli/app.py
import asyncio
import sys

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console

from deepagent.config import Config
from deepagent.core.llm_client import LLMClient
from deepagent.core.loop import AgentLoop, ConfirmationHandler
from deepagent.core.events import (
    TextDelta, ThinkingDelta, ToolCallEvent,
    ToolCallStartEvent, ToolResultEvent, DoneEvent,
)
from deepagent.tools.registry import ToolRegistry
from deepagent.tools.file_tools import create_file_tools
from deepagent.tools.shell_tools import create_shell_tools
from deepagent.tools.search_tools import create_search_tools


def _safe_print(console: Console, text: str, **kwargs) -> None:
    """Print text safely, handling Windows encoding issues."""
    try:
        console.print(text, **kwargs)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        console.print(encoded.decode(sys.stdout.encoding or "utf-8"), **kwargs)


class TerminalConfirmationHandler(ConfirmationHandler):
    """Prompt the user for y/N confirmation before executing shell-level tools."""

    def __init__(self, session: PromptSession, console: Console):
        self._session = session
        self._console = console

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        args_summary = " ".join(f"{k}={repr(v)}" for k, v in arguments.items())
        prompt_text = (
            f"[bold yellow]Execute `{tool_name} {args_summary}`? [y/N]: [/bold yellow]"
        )

        try:
            answer = await self._session.prompt_async(prompt_text)
            return answer.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False


async def run_cli(config: Config) -> None:
    """CLI main loop: prompt input -> AgentLoop execution -> render results."""
    console = Console(force_terminal=True)
    llm_client = LLMClient(config)

    # Build tool registry
    tool_registry = ToolRegistry()
    create_file_tools(tool_registry)
    create_shell_tools(tool_registry)
    create_search_tools(tool_registry)

    session = PromptSession()

    # Confirmation handler (triggers only for shell-level tools)
    confirm_handler = TerminalConfirmationHandler(session, console)

    # AgentLoop
    loop = AgentLoop(config, llm_client, tool_registry, confirm_handler=confirm_handler)

    # Key bindings: Ctrl+D to exit
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event):
        event.app.exit(result=None)

    console.print("[bold cyan]Deepagent[/bold cyan] — CLI coding agent")
    console.print(f"Model: {config.model} | Base URL: {config.base_url}")
    console.print(f"Tools: {', '.join(tool_registry.list_names())}")
    console.print("Type /exit or Ctrl+D to quit.\n")

    while True:
        try:
            user_input = await session.prompt_async(
                "> ",
                key_bindings=bindings,
            )
        except (EOFError, KeyboardInterrupt):
            break

        if user_input is None or user_input.strip() == "":
            continue
        if user_input.strip() == "/exit":
            break

        # Run AgentLoop and render events
        in_thinking = False
        tool_phase = False

        try:
            async for event in loop.run(user_input):
                if isinstance(event, ThinkingDelta):
                    if not in_thinking:
                        in_thinking = True
                        console.print()
                    _safe_print(console, f"[dim]{event.text}[/dim]", end="")

                elif isinstance(event, TextDelta):
                    if in_thinking:
                        console.print()
                        in_thinking = False
                    if tool_phase:
                        tool_phase = False
                        console.print()
                    _safe_print(console, event.text, end="")

                elif isinstance(event, ToolCallEvent):
                    if in_thinking:
                        console.print()
                        in_thinking = False
                    tool_phase = True
                    console.print()
                    for tc in event.tool_calls:
                        console.print(
                            f"[bold yellow]🔧 {tc.name}[/bold yellow]", end=""
                        )
                        args = " ".join(
                            f"{k}={repr(v)}" for k, v in tc.arguments.items()
                        )
                        _safe_print(console, f" {args}")

                elif isinstance(event, ToolCallStartEvent):
                    console.print(f"[dim]Running {event.tool_call.name}...[/dim]")

                elif isinstance(event, ToolResultEvent):
                    result = event.result
                    if result.success:
                        content_preview = result.content[:500]
                        if len(result.content) > 500:
                            content_preview += (
                                f"\n[dim]... ({len(result.content)} chars total)[/dim]"
                            )
                        _safe_print(console, content_preview)
                        if result.metadata:
                            meta_str = " ".join(
                                f"{k}={v}" for k, v in result.metadata.items()
                            )
                            _safe_print(console, f"[dim]{meta_str}[/dim]")
                    else:
                        _safe_print(
                            console, f"[bold red]Error:[/bold red] {result.error}"
                        )

                elif isinstance(event, DoneEvent):
                    console.print()

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

    console.print("\n[dim]Goodbye.[/dim]")
