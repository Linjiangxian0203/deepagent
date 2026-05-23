# src/deepagent/tools/git_tools.py
import subprocess
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


def create_git_tools(registry: ToolRegistry) -> list:
    """Create and register git tools (diff, log, status)."""

    def _run_git(args: list[str]) -> dict:
        """Run a git command and return a result dict. Handles git not found / not a repo."""
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return {
                    "success": False,
                    "content": "",
                    "error": result.stderr.strip() or f"git exited with code {result.returncode}",
                    "metadata": None,
                }
            output = result.stdout.strip()
            return {
                "success": True,
                "content": output if output else "(no output)",
                "error": None,
                "metadata": None,
            }
        except FileNotFoundError:
            return {
                "success": False,
                "content": "",
                "error": "git is not installed or not found on PATH",
                "metadata": None,
            }
        except Exception as e:
            return {
                "success": False,
                "content": "",
                "error": f"Error running git: {e}",
                "metadata": None,
            }

    @tool(registry=registry, description="Show working tree changes via 'git diff'. Use staged=True for staged changes, and path to filter by file.", safety_level=SafetyLevel.READONLY)
    async def git_diff(staged: bool = False, path: str = "") -> dict:
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            args.extend(["--", path])
        return _run_git(args)

    @tool(registry=registry, description="Show recent commit history via 'git log --oneline'. Use n to limit results (default 10).", safety_level=SafetyLevel.READONLY)
    async def git_log(n: int = 10) -> dict:
        return _run_git(["log", "--oneline", "-n", str(n)])

    @tool(registry=registry, description="Show working tree status via 'git status --short'.", safety_level=SafetyLevel.READONLY)
    async def git_status() -> dict:
        return _run_git(["status", "--short"])

    return [git_diff, git_log, git_status]
