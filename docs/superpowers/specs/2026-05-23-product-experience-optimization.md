# Deepagent Product Experience Optimization

**Date:** 2026-05-23
**Duration:** ~1 hour
**Approach:** Main agent + 5 parallel sub-agents

## Summary

Optimized the Deepagent CLI coding agent across four dimensions: security, feature completeness, CLI UX, and developer experience. All 120 unit tests pass with zero regressions.

## Changes by Category

### 1. Security — Hardcoded API Key Removed

**Problem:** `src/deepagent/config.py` contained a real DeepSeek API key baked into the default value. This is a security risk for any public distribution.

**Fix:**
- `config.py`: Changed `api_key` default from the hardcoded key to empty string `""`
- Existing `ValueError` with clear error message now fires when `DEEPSEEK_API_KEY` is not set
- Test updated: `test_config_defaults` → `test_config_missing_api_key_raises` to validate the new behavior

**Files:** `src/deepagent/config.py`, `tests/test_config.py`

### 2. Feature Completeness — Web Search & Fetch Tools

**Problem:** Agent had no way to search the web or fetch URL content, limiting its ability to answer questions about documentation, APIs, and current information.

**Fix:** Created `src/deepagent/tools/web_tools.py` with two new READONLY tools:
- **`web_search`**: Searches DuckDuckGo's HTML interface (no API key needed). Parses result titles, URLs, and snippets via regex. Configurable `max_results` (default 5).
- **`web_fetch`**: Fetches a URL, strips HTML tags (including `<script>` and `<style>`), collapses whitespace, truncates to `max_chars` (default 10000). Uses a browser User-Agent header.

Both tools handle network errors, timeouts, and invalid URLs gracefully with descriptive error messages.

**Files:** `src/deepagent/tools/web_tools.py` (new), `src/deepagent/cli/app.py` (import + registration)

### 3. CLI UX Improvements

**Problem:** The CLI had a duplicate code line, tool displays lacked visual clarity, and the overall feel could be more polished.

**Fixes in `src/deepagent/cli/app.py`:**
- **Removed duplicate `in_thinking = False`** — was set twice in the `ToolCallEvent` handler
- **Tool name display** changed from yellow to cyan bold for better readability
- **Added separator line** (`─── result ───`) before each tool result block, visually separating tool execution from output
- **Session duration** shown on exit with smart unit selection (seconds/minutes/hours)
- **Welcome message** enhanced to show tool count with names, model, and available commands
- **`/help` command** shows available commands and registered tools

### 4. Error Recovery Enhancement

**Problem:** The error recovery hints in `AgentLoop._enrich_error()` were good but had gaps in coverage and specificity.

**Fixes in `src/deepagent/core/loop.py`:**
- Added hint for "tool not found" errors → directs LLM to use only listed tools
- Added hint for "path not found" errors (distinct from file-not-found)
- Added hint for "command not found" as a separate case
- Added hint for non-zero exit codes from shell commands → directs LLM to check stderr
- Enhanced file-not-found hint to suggest using `glob()` to search for the correct path

### 5. Better Edit Tool Errors

**Problem:** When `edit_file`'s `old_string` matched multiple times, the error only said "found N times" with no help disambiguating.

**Fix in `src/deepagent/tools/file_tools.py`:**
- Now shows surrounding context (20 chars each side) for each match location
- Includes line numbers for each match
- Capped at 10 matches shown to avoid overwhelming output

### 6. Sub-Agent Bug Fix

**Problem:** `SubAgentRunner` used fragile `event.__class__.__name__ == "ToolCallStartEvent"` string comparison instead of proper `isinstance` check, because `ToolCallStartEvent` wasn't imported.

**Fix in `src/deepagent/core/sub_agent.py`:**
- Added `ToolCallStartEvent` to imports
- Changed to `isinstance(event, ToolCallStartEvent)` for robust type checking

### 7. Documentation — README

**Problem:** Project had no README, making it hard for new users to discover, install, and configure the tool.

**Fix:** Created comprehensive `README.md` covering:
- Project description and feature list (updated to 11 tools)
- Quick start guide (prerequisites, install, run)
- Full configuration reference table (11 environment variables)
- Complete tool table with safety levels
- Project structure diagram
- Development setup and testing instructions

**Files:** `README.md` (new)

## Tool Count: 7 → 11

| Tool | Status |
|------|--------|
| `read_file` | existing |
| `write_file` | existing |
| `edit_file` | existing (enhanced) |
| `grep` | existing |
| `glob` | existing |
| `run_shell` | existing |
| `delegate` | existing |
| `git_diff` | existing (already present) |
| `git_log` | existing (already present) |
| `git_status` | existing (already present) |
| `web_search` | **new** |
| `web_fetch` | **new** |

## Test Results

```
120 passed, 2 deselected in 3.01s
```

- All 120 unit tests pass
- 2 integration tests deselected (require valid API key, expected)
- Zero regressions

## Files Changed

| File | Change |
|------|--------|
| `src/deepagent/config.py` | Remove hardcoded API key |
| `src/deepagent/cli/app.py` | CLI UX improvements + tool registrations |
| `src/deepagent/core/loop.py` | Enhanced error recovery hints |
| `src/deepagent/core/sub_agent.py` | Fix isinstance check |
| `src/deepagent/tools/file_tools.py` | Better edit_file error context |
| `src/deepagent/tools/web_tools.py` | **New** — web_search + web_fetch |
| `tests/test_config.py` | Update for no-hardcoded-key behavior |
| `tests/test_llm_client.py` | Use make_config() helper |
| `README.md` | **New** — comprehensive project documentation |
