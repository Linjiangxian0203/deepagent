"""Worktree isolation — git worktree management with safety guards.

Uses asyncio.create_subprocess_exec for git commands (not sync subprocess),
matching deepagent's async architecture.

Reference: learn-claude-code s18_worktree_isolation. Adapted to asyncio.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_WT_NAME = re.compile(r'^[A-Za-z0-9._-]{1,64}$')


@dataclass
class WorktreeInfo:
    """Metadata about a worktree.

    Attributes:
        name: Worktree name, also used as branch suffix (wt/{name}).
        path: Absolute path to the worktree directory (.worktrees/{name}).
        task_id: Bound task ID, or empty string if unbound.
        branch: Git branch name for this worktree (wt/{name}).
    """

    name: str
    path: Path
    task_id: str = ""
    branch: str = ""


def validate_worktree_name(name: str) -> str | None:
    """Return error string if invalid, None if valid.

    Rules: 1-64 chars, only A-Za-z0-9._-, no "." or "..".
    """
    if not name:
        return "Worktree name cannot be empty"
    if name in (".", ".."):
        return f"'{name}' is not a valid worktree name"
    if not VALID_WT_NAME.match(name):
        return (
            f"Invalid worktree name '{name}': "
            "only letters, digits, dots, underscores, dashes (1-64 chars)"
        )
    return None


class WorktreeManager:
    """Manages git worktrees for task isolation.

    Worktrees live under project_root/.worktrees/.
    Each worktree gets its own branch named wt/{name}.

    Lifecycle events (create, remove, keep) are logged to .worktrees/events.jsonl.
    """

    def __init__(self, project_root: str | Path):
        self._root = Path(project_root).resolve()
        self._worktrees_dir = self._root / ".worktrees"
        self._worktrees_dir.mkdir(exist_ok=True)

    # -- Validation --

    def validate_name(self, name: str) -> bool:
        """Return True if name is valid, False otherwise."""
        return validate_worktree_name(name) is None

    # -- Create --

    async def create(self, name: str, task_id: str = "") -> tuple[bool, str]:
        """Create a git worktree with a dedicated branch.

        Steps:
        1. Validate name
        2. Check path doesn't already exist
        3. git worktree add .worktrees/{name} -b wt/{name} HEAD
        4. If task_id provided, bind task (write worktree field to task JSON)
        5. Log 'create' event to events.jsonl

        Returns (True, path_string) on success, (False, error_string) on failure.
        """
        err = validate_worktree_name(name)
        if err:
            return False, f"Error: {err}"

        wt_path = self._worktrees_dir / name
        if wt_path.exists():
            return False, f"Worktree '{name}' already exists at {wt_path}"

        ok, output = await self._run_git(
            ["worktree", "add", str(wt_path), "-b", f"wt/{name}", "HEAD"]
        )
        if not ok:
            return False, f"Git error: {output}"

        if task_id:
            ok_bind, bind_msg = self.bind_task(task_id, name)
            if not ok_bind:
                logger.warning(
                    "Worktree '%s' created but task binding failed: %s",
                    name,
                    bind_msg,
                )
                self._log_event("create", name, task_id)
                return (
                    True,
                    f"Worktree '{name}' created at {wt_path} "
                    f"(WARNING: task binding failed: {bind_msg})",
                )

        self._log_event("create", name, task_id)
        logger.info("Worktree '%s' created at %s", name, wt_path)
        return True, f"Worktree '{name}' created at {wt_path}"

    # -- Remove --

    async def remove(
        self, name: str, discard_changes: bool = False
    ) -> tuple[bool, str]:
        """Remove a worktree with safety checks.

        Safety guards (unless discard_changes=True):
        1. git status --porcelain in worktree -> count uncommitted files
        2. git log @{push}..HEAD --oneline -> count unpushed commits
        3. If files > 0 or commits > 0, refuse with details

        On success: git worktree remove --force, git branch -D wt/{name},
        then log 'remove' event.

        Returns (True, message) on success, (False, error) on failure.
        """
        err = validate_worktree_name(name)
        if err:
            return False, err

        wt_path = self._worktrees_dir / name
        if not wt_path.exists():
            return False, f"Worktree '{name}' not found"

        if not discard_changes:
            files, commits = await self._count_changes(wt_path)
            if files < 0 or commits < 0:
                return False, (
                    f"Cannot verify worktree '{name}' status. "
                    "Use discard_changes=true to force removal."
                )
            if files > 0 or commits > 0:
                return False, (
                    f"Worktree '{name}' has {files} uncommitted file(s) "
                    f"and {commits} unpushed commit(s). "
                    "Use discard_changes=true to force removal, "
                    "or keep_worktree to preserve for review."
                )

        ok, output = await self._run_git(
            ["worktree", "remove", str(wt_path), "--force"]
        )
        if not ok:
            return False, f"Failed to remove worktree directory for '{name}': {output}"

        await self._run_git(["branch", "-D", f"wt/{name}"])

        self._log_event("remove", name)
        logger.info("Worktree '%s' removed", name)
        return True, f"Worktree '{name}' removed"

    # -- Keep --

    async def keep(self, name: str) -> tuple[bool, str]:
        """Keep worktree for manual review. Branch is preserved.

        Logs a 'keep' event only. This is a no-op on the filesystem.

        Returns (True, message) on success, (False, error) on failure.
        """
        err = validate_worktree_name(name)
        if err:
            return False, err

        wt_path = self._worktrees_dir / name
        if not wt_path.exists():
            return False, f"Worktree '{name}' not found"

        self._log_event("keep", name)
        logger.info("Worktree '%s' kept for review (branch: wt/%s)", name, name)
        return True, f"Worktree '{name}' kept for review (branch: wt/{name})"

    # -- List --

    def list_all(self) -> list[WorktreeInfo]:
        """List all worktrees by scanning .worktrees/ subdirectories.

        Returns one WorktreeInfo per subdirectory found.
        """
        results = []
        for path in sorted(self._worktrees_dir.glob("*")):
            if not path.is_dir():
                continue
            name = path.name
            results.append(
                WorktreeInfo(
                    name=name,
                    path=path,
                    task_id="",
                    branch=f"wt/{name}",
                )
            )
        return results

    # -- Task binding --

    def bind_task(
        self, task_id: str, worktree_name: str, tasks_dir: str | None = None
    ) -> tuple[bool, str]:
        """Write worktree field to a task JSON file.

        Reads the task file, sets the 'worktree' field, and writes it back.
        The task's status and all other fields are preserved unchanged.

        Args:
            task_id: The task ID (e.g. "task_1234567890_0001").
            worktree_name: Worktree name to bind.
            tasks_dir: Directory containing task JSON files.
                       Defaults to project_root/.tasks/.

        Returns (True, message) on success, (False, error) on failure.
        """
        td = Path(tasks_dir) if tasks_dir else self._root / ".tasks"
        task_path = td / f"{task_id}.json"

        if not task_path.exists():
            return False, f"Task '{task_id}' not found at {task_path}"

        try:
            data = json.loads(task_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return False, f"Failed to read task '{task_id}': {e}"

        data["worktree"] = worktree_name

        try:
            task_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except OSError as e:
            return False, f"Failed to write task '{task_id}': {e}"

        logger.info("Bound task '%s' to worktree '%s'", task_id, worktree_name)
        return True, f"Task '{task_id}' bound to worktree '{worktree_name}'"

    # -- Path getter --

    def path_for(self, name: str) -> Path | None:
        """Return the worktree path if it exists, else None."""
        p = self._worktrees_dir / name
        return p if p.is_dir() else None

    # -- Event log --

    def _log_event(
        self, event_type: str, worktree_name: str, task_id: str = ""
    ) -> None:
        """Append a lifecycle event to events.jsonl."""
        event = {
            "type": event_type,
            "worktree": worktree_name,
            "task_id": task_id,
            "ts": time.time(),
        }
        events_file = self._worktrees_dir / "events.jsonl"
        try:
            with events_file.open("a") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("Failed to log worktree event: %s", e)

    # -- Git helper --

    async def _run_git(
        self, args: list[str], cwd: Path | None = None
    ) -> tuple[bool, str]:
        """Run a git command asynchronously.

        Uses asyncio.create_subprocess_exec with a 30-second timeout.
        Captures and combines stdout + stderr, truncated to 5000 chars.

        Returns (True, output) on success, (False, output_or_error) on failure.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                *args,
                cwd=cwd or self._root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30
            )
            output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
            if not output:
                output = "(no output)"
            output = output[:5000]
            return proc.returncode == 0, output
        except asyncio.TimeoutError:
            return False, "Error: git command timed out (30s)"
        except FileNotFoundError:
            return False, "Error: git executable not found"
        except OSError as e:
            return False, f"Error: git command failed: {e}"

    # -- Internal helpers --

    async def _count_changes(self, path: Path) -> tuple[int, int]:
        """Count uncommitted files and unpushed commits in a worktree.

        Returns (files_count, commits_count). Returns (-1, -1) on error.
        """
        try:
            status_ok, status_out = await self._run_git(
                ["status", "--porcelain"], cwd=path
            )
            if not status_ok:
                return -1, -1
            files = len(
                [l for l in status_out.splitlines() if l.strip() and l.strip() != "(no output)"]
            )

            log_ok, log_out = await self._run_git(
                ["log", "@{push}..HEAD", "--oneline"], cwd=path
            )
            if log_ok:
                commits = len(
                    [l for l in log_out.splitlines() if l.strip() and l.strip() != "(no output)"]
                )
            else:
                # No upstream configured for wt/{name} branches. Count commits
                # that exist on this branch but not on the base repo HEAD.
                base_ok, base_out = await self._run_git(
                    ["rev-parse", "--verify", "HEAD"], cwd=self._root
                )
                if base_ok:
                    base_sha = base_out.splitlines()[0].strip() if base_out else ""
                    if base_sha and base_sha != "(no output)":
                        log2_ok, log2_out = await self._run_git(
                            ["log", f"{base_sha}..", "HEAD", "--oneline"], cwd=path
                        )
                        if log2_ok:
                            commits = len(
                                [
                                    l
                                    for l in log2_out.splitlines()
                                    if l.strip() and l.strip() != "(no output)"
                                ]
                            )
                        else:
                            commits = -1
                    else:
                        commits = -1
                else:
                    commits = -1

            return files, commits
        except Exception:
            return -1, -1
