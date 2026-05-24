# src/deepagent/cli/app.py
import os
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from deepagent import __version__
from deepagent.config import Config
from deepagent.core.context import ContextManager
from deepagent.core.llm_client import LLMClient
from deepagent.core.loop import AgentLoop, ConfirmationHandler
from deepagent.core.sub_agent import SubAgentRunner
from deepagent.core.compaction import Compactor
from deepagent.core.events import (
    TextDelta, ThinkingDelta, ToolCallEvent,
    ToolCallStartEvent, ToolResultEvent, ToolLimitEvent,
    InterruptedEvent, DoneEvent, UsageEvent,
)
from deepagent.memory.store import MemoryStore
from deepagent.tools.registry import ToolRegistry
from deepagent.tools.file_tools import create_file_tools
from deepagent.tools.shell_tools import create_shell_tools
from deepagent.tools.search_tools import create_search_tools
from deepagent.tools.delegate_tools import create_delegate_tools
from deepagent.tools.git_tools import create_git_tools
from deepagent.tools.web_tools import create_web_tools


BASE_SYSTEM_PROMPT = """You are Deepagent, a CLI coding agent that helps with software engineering tasks.

## Core Rules
- Make minimal, surgical edits — don't refactor unrelated code
- Prefer editing existing files over creating new ones
- After making changes, run tests to verify correctness
- Default to no comments in code — only add when the WHY is non-obvious
- Never add error handling for scenarios that can't happen
- Three similar lines is better than a premature abstraction

## Tool Usage (IMPORTANT)
- read_file returns FULL content with metadata (total_lines, shown_start, shown_end). If shown_end == total_lines, you have the complete file — DO NOT retry.
- Tool results may contain truncation markers like "... [X chars truncated, Y lines] ..." or "... truncated (X chars total, Y lines)". These mean the output is genuinely truncated — YOU don't see the full data either. When you see a truncation marker, acknowledge it to the user and, if the missing data matters, re-run the command with a narrower scope (e.g., `git diff --stat` instead of `git diff`, or read a smaller range of a file).
- The user sees only the first 500 chars of each tool result. If a truncation marker lands in those visible chars, tell the user explicitly: "Output was truncated." Don't silently work with partial data.
- If a tool succeeds (success=true), use its result. Don't call the same tool again with different parameters "just to be sure."
- If a tool fails, read the error message carefully. Don't retry the exact same call expecting different results.
- Shell commands: use the syntax and path separator appropriate for your platform. See the [Platform] section above to determine what shell and commands to use.
- The grep tool limits output to 500 matches. The glob tool to 500 entries.

## Response Style
- Be concise. One sentence is often enough.
- Don't just restate tool output the user already sees. Add insight: what stands out, what's missing, what it means.
- After diagnostic commands (git status, ls, test runs), always suggest at least one concrete next step — even if it's just "looks clean, ready to commit."
- For code changes: state what file and what you changed, not a play-by-play.
- If asked a simple question, give a direct answer — not headers and sections.
- When running shell commands, explain in 5 words what it does.
- Do NOT use Markdown formatting (**bold**, `code`, etc.). The CLI renders plain text, so formatting characters appear as literal symbols.
- Use indentation or dashes for lists, not bullet points."""


def _categorize_tools(tool_names: list[str]) -> dict[str, list[str]]:
    """Group tools by category for banner display."""
    _CATEGORIES: dict[str, str] = {
        "read_file": "File", "write_file": "File", "edit_file": "File",
        "run_shell": "Shell",
        "grep": "Search", "glob": "Search", "web_search": "Search", "web_fetch": "Search",
        "git_diff": "Git", "git_log": "Git", "git_status": "Git",
        "delegate": "Agent",
    }
    result: dict[str, list[str]] = {}
    for name in sorted(tool_names):
        cat = _CATEGORIES.get(name, "Other")
        result.setdefault(cat, []).append(name)
    return dict(sorted(result.items()))


_CATEGORY_COLORS: dict[str, str] = {
    "File": "cyan", "Shell": "yellow", "Search": "green",
    "Git": "magenta", "Agent": "blue",
}


def _render_banner(console: Console, config, cwd: str,
                   tool_registry: ToolRegistry, memory_store) -> None:
    """Render the startup banner with structured, colored layout."""
    tool_names = tool_registry.list_names()
    tools_by_cat = _categorize_tools(tool_names)

    lines: list[str] = []
    lines.append(f" [bold cyan]Deepagent[/bold cyan] [dim]v{__version__}[/dim]")
    lines.append("")
    lines.append(f" [bold]Model[/bold]  {config.model}")
    lines.append(f" [bold]CWD  [/bold]  [dim]{cwd}[/dim]")

    thinking_text = "[green]on[/green]" if config.thinking_enabled else "[dim]off[/dim]"
    lines.append(f" [bold]Think[/bold]  {thinking_text}")
    lines.append("")
    lines.append(f" [bold]Tools[/bold] ({len(tool_names)} loaded)")
    for cat, names in tools_by_cat.items():
        color = _CATEGORY_COLORS.get(cat, "dim")
        lines.append(f"   [{color}]{cat:<8}[/{color}] {', '.join(names)}")
    memory_entries = memory_store.get_all_entries()
    if memory_entries:
        lines.append("")
        lines.append(f" [bold]Memory[/bold] {len(memory_entries)} entries")
    lines.append("")
    lines.append(" [dim]/exit  /help    Ctrl+D quit    Ctrl+C interrupt[/dim]")

    panel = Panel("\n".join(lines), padding=(1, 2), expand=False)
    console.print(panel)
    console.print()


def _safe_print(console: Console, text: str, **kwargs) -> None:
    """Print text safely, handling Windows encoding issues.

    Rich markup is enabled by default. Pass markup=False for LLM-generated
    text that may contain [square brackets].
    """
    try:
        console.print(text, **kwargs)
    except UnicodeEncodeError:
        encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
        console.print(encoded.decode(sys.stdout.encoding or "utf-8"), **kwargs)


def _detect_project_root() -> str:
    """Detect the project root by walking up from the installed package location.

    When installed with `pip install -e .`, deepagent.__file__ points to
    the source directory, so we can find the project root from there.
    Falls back to CWD if detection fails.
    """
    try:
        import deepagent
        pkg_dir = Path(deepagent.__file__).resolve().parent
        # Walk up to find pyproject.toml
        candidate = pkg_dir
        for _ in range(5):
            if (candidate / "pyproject.toml").exists():
                return str(candidate)
            candidate = candidate.parent
    except Exception:
        pass
    return os.getcwd()


def _load_text_file(path: str) -> str | None:
    """Load a text file if it exists, returning None otherwise."""
    try:
        p = Path(path)
        if p.exists():
            return p.read_text("utf-8").strip()
    except Exception:
        pass
    return None


def _resolve_project_slug(project_root: str) -> str:
    """Derive a project slug from the root path.

    Uses the directory name with parent initial to reduce collisions.
    Example: G:/Agents -> G--Agents
    """
    p = Path(project_root).resolve()
    drive = p.drive.rstrip(":")
    parts = [drive] + [part for part in p.parts[1:] if part]
    return "--".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def _find_memory_root(project_root: str) -> str:
    """Find the memory directory for a project following Claude Code conventions.

    Returns ~/.claude/projects/<slug>/memory/
    """
    home = Path.home()
    slug = _resolve_project_slug(project_root)
    return str(home / ".claude" / "projects" / slug / "memory")


def _build_system_prompt(memory_context: str) -> str:
    """Build the full system prompt with optional memory and CLAUDE.md content.

    Loads:
    - Platform info (OS + shell availability)
    - User-level CLAUDE.md (~/.claude/CLAUDE.md)
    - Project-level CLAUDE.md (requires knowing project root)
    - Memory index context from MemoryStore

    Layer order: platform → user CLAUDE.md → project CLAUDE.md → memory index → base prompt
    """
    import platform

    home = Path.home()
    parts: list[str] = []

    # Platform info so the model knows what commands to use
    system = platform.system()
    if system == "Windows":
        shell_info = (
            "Windows native shell (cmd.exe). Use Windows commands: "
            "dir (not ls), findstr (not grep), type (not cat), "
            "mkdir/rmdir (not mkdir -p/rm -rf), copy (not cp), move (not mv). "
            "Paths use backslashes (C:\\Users\\...). "
            "Use 'python' or 'python3' for Python, 'pip' for pip."
        )
    else:
        shell_info = "Unix shell — standard POSIX commands with forward-slashed paths"
    parts.append(f"[Platform]\nOS: {system} | Shell: {shell_info}")

    # User-level CLAUDE.md
    user_claude = _load_text_file(str(home / ".claude" / "CLAUDE.md"))
    if user_claude:
        parts.append("[User CLAUDE.md]\n" + user_claude)

    # Project-level CLAUDE.md (current directory)
    project_claude = _load_text_file("CLAUDE.md")
    if project_claude:
        parts.append("[Project CLAUDE.md]\n" + project_claude)

    # Memory index
    if memory_context:
        parts.append("[Memory index]\n" + memory_context)

    parts.append(BASE_SYSTEM_PROMPT)
    return "\n\n".join(parts)


class TerminalConfirmationHandler(ConfirmationHandler):
    """Prompt the user for y/N confirmation before executing shell-level tools."""

    def __init__(self, session: PromptSession):
        self._session = session

    async def confirm(self, tool_name: str, arguments: dict) -> bool:
        args_summary = " ".join(f"{k}={repr(v)}" for k, v in arguments.items())
        prompt_text = f"Execute {tool_name} {args_summary}? [y/N]: "

        try:
            answer = await self._session.prompt_async(prompt_text)
            return answer.strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False


async def run_cli(config: Config) -> None:
    """CLI main loop: prompt input -> AgentLoop execution -> render results."""
    console = Console(force_terminal=True)

    # Auto-detect project root and chdir so file tools work correctly
    project_root = _detect_project_root()
    if project_root != os.getcwd():
        os.chdir(project_root)

    llm_client = LLMClient(config)

    # Build tool registry
    tool_registry = ToolRegistry()
    create_file_tools(tool_registry)
    create_shell_tools(tool_registry)
    create_search_tools(tool_registry)
    create_git_tools(tool_registry)
    create_web_tools(tool_registry)
    # Sub-agent runner + delegate tool (P4)
    sub_agent_runner = SubAgentRunner(config, llm_client, tool_registry)
    create_delegate_tools(tool_registry, sub_agent_runner)

    # Skill loading
    from deepagent.core.skills import SkillRegistry
    from deepagent.tools.skill_tools import create_skill_tools

    skill_registry = SkillRegistry()
    skills_dir = Path(project_root) / "skills"
    if skills_dir.exists():
        skill_registry.scan(str(skills_dir))
    create_skill_tools(tool_registry, skill_registry)

    # Long-term memory
    cwd = os.getcwd()
    memory_root = _find_memory_root(cwd)
    memory_store = MemoryStore(memory_root)
    memory_context = memory_store.get_system_context()

    from deepagent.tools.memory_tools import create_memory_tools
    create_memory_tools(tool_registry, memory_store)

    from deepagent.core.tasks import TaskManager
    from deepagent.tools.task_tools import create_todo_write_tool, create_task_system_tools

    tasks_dir = Path(project_root) / ".tasks"
    task_mgr = TaskManager(str(tasks_dir))
    create_todo_write_tool(tool_registry)
    create_task_system_tools(tool_registry, task_mgr)

    from deepagent.core.message_bus import MessageBus
    from deepagent.tools.team_tools import create_team_tools

    mailboxes_dir = Path(project_root) / ".mailboxes"
    message_bus = MessageBus(str(mailboxes_dir))

    # Register submit_plan for teammate use (intercepted by _execute_teammate_tool)
    from deepagent.tools.registry import tool as register_tool
    @register_tool(tool_registry, name="submit_plan",
                   description="Submit a plan to Lead for approval via protocol.",
                   safety_level=SafetyLevel.WRITE)
    async def submit_plan(plan: str) -> dict:
        return {"success": True, "content": "Plan submitted."}

    create_team_tools(tool_registry, message_bus, llm_client, tool_registry,
                      cwd=project_root, task_mgr=task_mgr)

    session = PromptSession()
    confirm_handler = TerminalConfirmationHandler(session)

    # Build system prompt with CLAUDE.md + memory
    system_prompt = _build_system_prompt(memory_context)

    from deepagent.core.system_prompt import SystemPrompt, PromptSection, PRIORITY_SKILLS_CATALOG

    sp = SystemPrompt()
    sp.register(PromptSection(
        name="base",
        content=system_prompt,
        priority=100,
    ))
    catalog = skill_registry.get_catalog()
    if catalog:
        sp.register(PromptSection(
            name="skills-catalog",
            content=catalog,
            priority=PRIORITY_SKILLS_CATALOG,
        ))
    system_prompt = sp.assemble()

    # ── Startup banner ──
    _render_banner(console, config, cwd, tool_registry, memory_store)

    # Key bindings: Ctrl+D to exit, Ctrl+C to interrupt
    bindings = KeyBindings()

    @bindings.add("c-d")
    def _(event):
        event.app.exit(result=None)

    current_loop: AgentLoop | None = None

    @bindings.add("c-c")
    def _(event):
        if current_loop is not None:
            current_loop.interrupt()

    session_turns = 0
    session_tool_calls = 0
    session_start = time.time()

    # Session-level context: persists across all user turns.
    # Previously a fresh ContextManager was created per turn, causing
    # the agent to forget all prior conversation.
    ctx = ContextManager(system_prompt=system_prompt)

    from deepagent.core.background import BackgroundManager

    background_mgr = BackgroundManager()

    # Transcript saving on compaction events and session exit
    transcript_dir = Path.home() / ".deepagent" / "transcripts"
    compactor = Compactor(transcript_dir=transcript_dir)

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
        if user_input.strip() == "/help":
            console.print()
            console.print("[bold cyan]Commands:[/bold cyan]")
            console.print("  [bold]/exit[/bold]  Quit Deepagent")
            console.print("  [bold]/help[/bold]  Show this help")
            console.print()
            console.print("[bold cyan]Tools:[/bold cyan]")
            for tname in tool_registry.list_names():
                console.print(f"  [bold]{tname}[/bold]")
            console.print()
            console.print("[dim]Ctrl+D to quit. Ctrl+C to interrupt the current agent run.[/dim]")
            console.print()
            continue

        # Reuse session-level context across turns
        current_loop = AgentLoop(
            config, llm_client, tool_registry,
            context=ctx, confirm_handler=confirm_handler,
            memory_store=memory_store,
            background_mgr=background_mgr,
            message_bus=message_bus,
        )

        in_thinking = False
        thinking_chars = 0
        tool_phase = False
        turn_count = 0

        try:
            async for event in current_loop.run(user_input):
                if isinstance(event, ThinkingDelta):
                    thinking_chars += len(event.text)
                    if not in_thinking:
                        in_thinking = True
                        console.print()
                        console.print("[dim]Thinking...[/dim]", end="")

                elif isinstance(event, TextDelta):
                    if in_thinking:
                        if thinking_chars > 0:
                            _safe_print(console, f" [dim]({thinking_chars} chars)[/dim]")
                        console.print()
                        in_thinking = False
                    if tool_phase:
                        tool_phase = False
                        console.print()
                    # markup=False prevents LLM output like [tag] from being
                    # interpreted as Rich markup; Markdown **bold** etc. stay
                    # as readable plain text
                    _safe_print(console, event.text, end="", markup=False)

                elif isinstance(event, ToolCallEvent):
                    session_tool_calls += len(event.tool_calls)
                    if in_thinking:
                        if thinking_chars > 0:
                            _safe_print(console, f" [dim]({thinking_chars} chars)[/dim]")
                        console.print()
                        in_thinking = False
                    tool_phase = True
                    turn_count += 1
                    console.print()
                    _safe_print(console, f"[dim]── Turn {turn_count} ──[/dim]")
                    for tc in event.tool_calls:
                        _safe_print(
                            console, f"[bold cyan]{tc.name}[/bold cyan]", end=""
                        )
                        args = " ".join(
                            f"{k}={repr(v)}" for k, v in tc.arguments.items()
                        )
                        _safe_print(console, f" {args}")

                elif isinstance(event, ToolCallStartEvent):
                    console.print(f"[dim]Running {event.tool_call.name}...[/dim]")

                elif isinstance(event, ToolLimitEvent):
                    _safe_print(
                        console,
                        f"[bold yellow]Too many tool calls, limiting to {config.max_tools_per_turn}[/bold yellow]",
                    )

                elif isinstance(event, ToolResultEvent):
                    result = event.result
                    if result.success:
                        console.print("[dim]─── result ───[/dim]")
                        _safe_print(console, result.content[:500], markup=False)
                        if len(result.content) > 500:
                            console.print(
                                f"[bold yellow]... truncated ({len(result.content)} chars total, "
                                f"{len(result.content.splitlines())} lines)[/bold yellow]"
                            )
                        if result.metadata:
                            meta_str = " ".join(
                                f"{k}={v}" for k, v in result.metadata.items()
                            )
                            _safe_print(console, f"[dim]{meta_str}[/dim]")
                    else:
                        _safe_print(console, "[bold red]Error:[/bold red] ", end="")
                        _safe_print(console, result.error, markup=False)
                        if result.content:
                            _safe_print(console, f"\n[dim]{result.content[:300]}[/dim]")

                elif isinstance(event, InterruptedEvent):
                    _safe_print(console, "\n[dim]Interrupted.[/dim]")

                elif isinstance(event, UsageEvent):
                    pass  # token usage tracked internally

                elif isinstance(event, DoneEvent):
                    console.print()

            session_turns += 1

        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        finally:
            current_loop = None

    elapsed = time.time() - session_start
    if elapsed < 60:
        duration = f"{elapsed:.0f}s"
    elif elapsed < 3600:
        duration = f"{elapsed / 60:.1f}m"
    else:
        duration = f"{elapsed / 3600:.1f}h"
    console.print(f"\n[dim]Session: {session_turns} turns, {session_tool_calls} tool calls, {duration}[/dim]")

    # Save session transcript
    if ctx.message_count > 0:
        transcript_path = await compactor.save_session_transcript(ctx.get_messages())
        if transcript_path:
            console.print(f"[dim]Transcript saved: {transcript_path}[/dim]")

    console.print("[dim]Bye.[/dim]")
