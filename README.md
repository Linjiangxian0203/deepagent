# Deepagent

English | [中文](README.zh-CN.md)

A CLI coding agent powered by the DeepSeek API. Runs in your terminal, reads your codebase, edits files, runs shell commands, and delegates subtasks to sub-agents -- all within a multi-turn ReAct loop with streaming output and thinking mode.

## Features

- **Multi-turn ReAct loop** -- the agent iterates: thinks, calls tools, observes results, and continues until the task is complete or a configurable iteration limit is reached.
- **12 built-in tools** -- `read_file`, `write_file`, `edit_file`, `grep`, `glob`, `run_shell`, `git_diff`, `git_log`, `git_status`, `web_search`, `web_fetch`, and `delegate` (sub-agent spawning with concurrency control).
- **Streaming output** -- LLM text streams to the terminal in real time. DeepSeek thinking mode content is displayed separately with character counts.
- **Token budget management** -- tracks cumulative token usage against a 980K effective limit (of the 1M context window) and can compress older messages into a summary when approaching the limit.
- **Long-term memory** -- filesystem-based persistent memory using Markdown files with YAML frontmatter, compatible with the Claude Code memory format (`~/.claude/projects/<slug>/memory/`).
- **Safety prompts** -- write-level and shell-level tools require interactive `y/N` confirmation before execution. File operations can be sandboxed to a configurable safe root directory.
- **Sub-agent delegation** -- the `delegate` tool spawns parallel sub-agents for independent tasks, with concurrency limited to 5 by default.
- **CLAUDE.md awareness** -- automatically loads `~/.claude/CLAUDE.md` and project-level `CLAUDE.md` files and injects them into the system prompt.
- **Windows native** -- runs natively on Windows with cmd.exe. Uses Windows commands (dir, findstr, type) and backslash paths.

## Quick Start

### Prerequisites

- Python 3.11 or later
- A [DeepSeek API key](https://platform.deepseek.com/)

### Install

```bash
git clone https://github.com/Linjiangxian0203/deepagent.git
cd deepagent
pip install -e .
```

### Run

```bash
# Windows (cmd / PowerShell)
set DEEPSEEK_API_KEY=sk-your-key-here
deepagent

# Linux / macOS
export DEEPSEEK_API_KEY="sk-your-key-here"
deepagent
```

Type your task at the `> ` prompt. Use `/exit` or `Ctrl+D` to quit, `Ctrl+C` to interrupt the current task.

## Configuration

All settings are controlled via environment variables.

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | *(required)* | Your DeepSeek API key. |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | Base URL for the API endpoint. |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Model name to use (e.g., `deepseek-v4-pro`, `deepseek-v4-flash`). |
| `DEEPSEEK_THINKING_ENABLED` | `1` | Enable thinking/reasoning mode. Set to `0` or `false` to disable. |
| `DEEPSEEK_REASONING_EFFORT` | `max` | Reasoning effort level. Valid values: `high`, `max`. |
| `DEEPSEEK_MAX_TOKENS` | `8192` | Maximum completion tokens per LLM call. |
| `DEEPSEEK_TEMPERATURE` | `1.0` | Sampling temperature. |
| `DEEPSEEK_TOP_P` | `1.0` | Nucleus sampling parameter. |
| `DEEPSEEK_MAX_ITERATIONS` | `50` | Maximum ReAct loop iterations per user request. |
| `DEEPSEEK_MAX_TOOLS_PER_TURN` | `10` | Maximum tool calls the model is allowed to emit in a single turn. |

## Available Tools

| Tool | Safety Level | Description |
|---|---|---|
| `read_file` | readonly | Read a file with line-numbered output. Supports offset and limit. |
| `write_file` | write | Create a new file or overwrite an existing one. |
| `edit_file` | write | Replace an exact string in a file with a new string. The match must be unique. |
| `grep` | readonly | Search files with a regex pattern. Supports file filtering with a glob. |
| `glob` | readonly | Find files matching a glob pattern (e.g., `**/*.py` for recursive search). |
| `run_shell` | shell | Execute a shell command and return stdout/stderr. Configurable timeout. |
| `git_diff` | readonly | Show working tree changes via `git diff`. Supports --staged and path filter. |
| `git_log` | readonly | Show recent commit history via `git log --oneline`. Configurable limit. |
| `git_status` | readonly | Show working tree status via `git status --short`. |
| `web_search` | readonly | Search the web using DuckDuckGo. Returns titles, URLs, and snippets. |
| `web_fetch` | readonly | Fetch and extract plain text content from a URL. HTML tags are stripped. |
| `delegate` | readonly | Spawn a sub-agent to work on an independent subtask autonomously. Multiple delegates can run in parallel within a single turn. |

**Safety levels**: `readonly` tools run without confirmation. `write` and `shell` tools prompt `y/N` before execution.

## Project Structure

```
deepagent/
  pyproject.toml                  # Project metadata, dependencies, entry point
  src/deepagent/
    __init__.py                   # Version string
    main.py                       # CLI entry point
    config.py                     # Environment variable configuration
    cli/
      __init__.py
      app.py                      # CLI loop, streaming renderer, system prompt builder
    core/
      __init__.py
      context.py                  # ContextManager with token budget and compression
      events.py                   # Event dataclasses (TextDelta, ToolCallEvent, etc.)
      llm_client.py               # OpenAI-compatible async LLM client
      loop.py                     # AgentLoop -- multi-turn ReAct engine
      sub_agent.py                # SubAgentRunner with concurrency control
    tools/
      __init__.py
      protocol.py                 # SafetyLevel enum, ToolProtocol
      registry.py                 # ToolRegistry and @tool decorator
      file_tools.py               # read_file, write_file, edit_file
      search_tools.py             # grep, glob
      shell_tools.py              # run_shell
      git_tools.py                # git_diff, git_log, git_status
      web_tools.py                # web_search, web_fetch
      delegate_tools.py           # delegate (sub-agent spawning)
    memory/
      __init__.py
      models.py                   # MemoryEntry dataclass with frontmatter parsing
      store.py                    # MemoryStore -- Markdown-backed persistent memory
```

## Development

### Setup

```bash
git clone https://github.com/Linjiangxian0203/deepagent.git
cd deepagent
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -e .
```

### Dependencies

All runtime dependencies are declared in `pyproject.toml`:

- `openai` >= 1.0.0 -- OpenAI-compatible client for the DeepSeek API
- `rich` >= 13.0.0 -- Terminal formatting and colors
- `prompt-toolkit` >= 3.0.0 -- Interactive prompt with key bindings
- `pydantic` >= 2.0.0 -- Data validation (used in LLM client)

### Running Tests

```bash
# From the project root
python -m pytest tests/
```

### Architecture Notes

- **ReAct loop**: Each user message starts a fresh `AgentLoop` with its own `ContextManager`. The loop streams LLM responses, collects tool calls, executes them (respecting safety levels and confirmation), feeds results back, and repeats until the model stops calling tools or the iteration limit is hit.
- **Context compression**: When the estimated token count exceeds the effective budget (980K of the 1M context window), the oldest third of messages can be summarized to free space.
- **Memory system**: Memories are stored as individual `.md` files with YAML frontmatter in `~/.claude/projects/<slug>/memory/`. A `MEMORY.md` index is injected into the system prompt so the agent is aware of available memories.
- **Sub-agents**: The `delegate` tool uses `SubAgentRunner`, which limits concurrent sub-agents to 5 (configurable) to respect DeepSeek API rate limits. Each sub-agent runs its own `AgentLoop` with a lightweight system prompt.
