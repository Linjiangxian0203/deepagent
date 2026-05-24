"""Permission system — 3-stage pipeline built on HookSystem.

Stage 1 — DenyList: hard-coded regex patterns, always blocked (no bypass).
Stage 2 — RuleEngine: settings.json allow/deny/ask rules, pattern matching.
Stage 3 — UserApproval: interactive y/N via ConfirmationHandler (existing).

Permission modes: default, accept-edits, plan, bypass.

Reference: learn-claude-code s03_permission. Extended with persistent config,
regex matching, and CLI mode switching.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from deepagent.core.hooks import HookBlock

logger = logging.getLogger(__name__)


# ── Permission modes ──────────────────────────────────────────────


class PermissionMode(str, Enum):
    DEFAULT = "default"          # All 3 stages active
    ACCEPT_EDITS = "accept-edits"  # Auto-approve write_file/edit_file
    PLAN = "plan"                # Deny all writes/shell, readonly only
    BYPASS = "bypass"           # Skip all permission checks


# ── Safety levels for mode enforcement ────────────────────────────

_READONLY_TOOLS = frozenset({
    "read_file", "glob", "grep", "web_search", "web_fetch",
    "git_status", "git_diff", "git_log", "delegate",
})
_EDIT_TOOLS = frozenset({"write_file", "edit_file"})


# ── Hard deny list (regex patterns — never bypassable) ─────────────

_DENY_LIST_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"rm\s+-rf\s+/",
        r"sudo\s+rm",
        r"sudo\s+",
        r"shutdown",
        r"reboot",
        r"mkfs",
        r"dd\s+if=",
        r">\s*/etc/",
        r"chmod\s+777",
        r"chown\s+-R\s+root",
        r"format\s+[cdefgh]:",
        r"diskutil\s+eraseDisk",
        r":\(\)\s*\{\s*:\|:&\s*\};:",  # fork bomb
    ]
]


# ── Confirmation handler protocol ─────────────────────────────────


class ConfirmHandler(Protocol):
    """Performs interactive user confirmation. Same shape as ConfirmationHandler in loop.py."""

    async def confirm(self, tool_name: str, arguments: dict) -> bool: ...


# ── Rule data ─────────────────────────────────────────────────────


@dataclass
class PermissionRule:
    patterns: list[re.Pattern]
    action: str  # "allow", "deny", "ask"


# ── Default settings ──────────────────────────────────────────────

DEFAULT_DENY = [
    r"rm\s+-rf",
    r"git\s+push\s+--force",
    r"git\s+reset\s+--hard",
    r"DROP\s+TABLE",
    r"DELETE\s+FROM",
]
DEFAULT_ALLOW = [
    r"git\s+status",
    r"git\s+diff",
    r"git\s+log",
    r"git\s+branch",
    r"git\s+add",
    r"git\s+commit",
    r"git\s+checkout",
    r"npm\s+test",
    r"npm\s+run",
    r"cargo\s+test",
    r"cargo\s+build",
    r"python\s+-m\s+pytest",
    r"ls\b",
    r"dir\b",
    r"echo\b",
]


# ═══════════════════════════════════════════════════════════════════


class PermissionSystem:
    """3-stage permission pipeline.

    Usage::

        perms = PermissionSystem(mode=PermissionMode.DEFAULT)
        perms.load_project_settings("/path/to/project")

        # Integrate as a PreToolUse hook:
        hooks.register("PreToolUse", perms.as_hook(), priority=10, name="permissions")

        # Or call directly:
        block = await perms.check("run_shell", {"command": "rm -rf /"})
        if block:
            print(f"Blocked: {block.reason}")
    """

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        confirm_handler: ConfirmHandler | None = None,
    ):
        self.mode = mode
        self._confirm = confirm_handler
        self._allow_rules: list[PermissionRule] = []
        self._deny_rules: list[PermissionRule] = []
        self._loaded = False

    # ── Settings loading ──────────────────────────────────────────

    def load_project_settings(self, project_root: str | Path) -> None:
        """Load permissions from a project's .claude/settings.json, if it exists."""
        project_root = Path(project_root)
        settings_path = project_root / ".claude" / "settings.json"
        if settings_path.exists():
            self._load_file(settings_path)

        # Also load user-level settings
        user_settings = Path.home() / ".claude" / "settings.json"
        if user_settings.exists() and user_settings != settings_path:
            self._load_file(user_settings)

    def load_dict(self, permissions: dict) -> None:
        """Load rules from a dict (for testing / programmatic use)."""
        self._load_permissions(permissions)

    def _load_file(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text("utf-8"))
            perms = data.get("permissions", {})
            if isinstance(perms, dict):
                self._load_permissions(perms)
            self._loaded = True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load permission settings from %s: %s", path, e)

    def _load_permissions(self, perms: dict) -> None:
        mode_str = perms.get("defaultMode")
        if mode_str:
            try:
                self.mode = PermissionMode(mode_str)
            except ValueError:
                logger.warning("Unknown permission mode: %r", mode_str)

        for pattern in perms.get("allow", []):
            self._allow_rules.append(PermissionRule(
                patterns=[re.compile(pattern, re.IGNORECASE)],
                action="allow",
            ))

        for pattern in perms.get("deny", []):
            self._deny_rules.append(PermissionRule(
                patterns=[re.compile(pattern, re.IGNORECASE)],
                action="deny",
            ))

    # ── Default rules ─────────────────────────────────────────────

    def load_defaults(self) -> None:
        """Load sensible default rules. Called if no settings.json is found."""
        for pattern in DEFAULT_ALLOW:
            self._allow_rules.append(PermissionRule(
                patterns=[re.compile(pattern, re.IGNORECASE)],
                action="allow",
            ))
        for pattern in DEFAULT_DENY:
            self._deny_rules.append(PermissionRule(
                patterns=[re.compile(pattern, re.IGNORECASE)],
                action="deny",
            ))

    # ── Main check ────────────────────────────────────────────────

    async def check(
        self,
        tool_name: str,
        arguments: dict,
    ) -> HookBlock | None:
        """Run the 3-stage pipeline. Returns HookBlock if denied, None if allowed.

        Only applies to shell-level tools (run_shell). Read-only tools are always allowed.
        Write tools are checked in non-bypass modes.
        """

        # Bypass mode: allow everything
        if self.mode == PermissionMode.BYPASS:
            return None

        # Plan mode: only readonly allowed
        if self.mode == PermissionMode.PLAN:
            if tool_name not in _READONLY_TOOLS:
                return HookBlock(
                    f"Permission denied: plan mode. Tool '{tool_name}' is read-only protected."
                )

        # Stage 1: Hard deny list (applies to run_shell only)
        if tool_name == "run_shell":
            command = arguments.get("command", "")
            if not isinstance(command, str):
                command = " ".join(command) if isinstance(command, list) else str(command)
            for pattern in _DENY_LIST_PATTERNS:
                if pattern.search(command):
                    return HookBlock(
                        f"Permission denied by security policy: command matches "
                        f"blocked pattern '{pattern.pattern}'"
                    )

        # Stage 2: Rule engine
        # Check deny rules first (more specific)
        if tool_name == "run_shell":
            command = arguments.get("command", "")
            if not isinstance(command, str):
                command = " ".join(command) if isinstance(command, list) else str(command)

            for rule in self._deny_rules:
                for pat in rule.patterns:
                    if pat.search(command):
                        return HookBlock(
                            f"Permission denied by rules: command matches '{pat.pattern}'"
                        )

            for rule in self._allow_rules:
                for pat in rule.patterns:
                    if pat.search(command):
                        return None  # explicitly allowed

        # Accept-edits mode: auto-approve file edits
        if self.mode == PermissionMode.ACCEPT_EDITS:
            if tool_name in _EDIT_TOOLS:
                return None

        # Stage 3: User approval (for shell or write tools)
        if tool_name == "run_shell" or tool_name not in _READONLY_TOOLS:
            if self._confirm is not None:
                approved = await self._confirm.confirm(tool_name, arguments)
                if not approved:
                    return HookBlock("Execution denied by user.")
                return None

            # No confirm handler available — deny by default for safety
            if tool_name == "run_shell":
                return HookBlock(
                    "No confirmation handler available. Shell commands require user approval."
                )

        return None

    # ── Hook integration ──────────────────────────────────────────

    def as_hook(self):
        """Return an async callback suitable for HookSystem.register().

        Usage::

            perms = PermissionSystem(...)
            hooks.register("PreToolUse", perms.as_hook(), priority=10, name="permissions")
        """

        async def _hook(tool_name: str, arguments: dict, **kwargs) -> HookBlock | None:
            return await self.check(tool_name, arguments)

        return _hook
