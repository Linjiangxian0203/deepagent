# src/deepagent/cli/app.py
import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from deepagent.config import Config
from deepagent.core.llm_client import LLMClient
from deepagent.core.events import TextDelta, ThinkingDelta, ToolCallEvent


async def run_cli(config: Config) -> None:
    """Main CLI loop: prompt for input, stream response, repeat."""
    console = Console()
    client = LLMClient(config)
    session = PromptSession()

    # Key bindings: Ctrl+D to exit
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event):
        event.app.exit(result=None)

    console.print("[bold cyan]Deepagent[/bold cyan] — CLI coding agent")
    console.print(f"Model: {config.model} | Base URL: {config.base_url}")
    console.print("Type /exit or Ctrl+D to quit.\n")

    messages: list[dict] = []

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

        messages.append({"role": "user", "content": user_input})

        # Stream response
        response_text_parts: list[str] = []
        thinking_parts: list[str] = []
        in_thinking = False

        try:
            async for event in client.stream_chat(messages):
                if isinstance(event, ThinkingDelta):
                    if not in_thinking:
                        in_thinking = True
                        console.print()
                    thinking_parts.append(event.text)
                    # Render thinking in dim style
                    console.print(f"[dim]{event.text}[/dim]", end="")

                elif isinstance(event, TextDelta):
                    if in_thinking:
                        console.print()  # end thinking line
                        in_thinking = False
                    response_text_parts.append(event.text)
                    console.print(event.text, end="")

                elif isinstance(event, ToolCallEvent):
                    if in_thinking:
                        console.print()
                        in_thinking = False
                    for tc in event.tool_calls:
                        console.print(f"\n[bold yellow]Tool: {tc.name}[/bold yellow]")
                        for k, v in tc.arguments.items():
                            console.print(f"  {k}: {v}")

            console.print()  # final newline

            assistant_text = "".join(response_text_parts)
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")

    console.print("\n[dim]Goodbye.[/dim]")
