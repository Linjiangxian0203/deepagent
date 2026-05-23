# src/deepagent/tools/shell_tools.py
import asyncio
import os

from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


def _shell_executable() -> str | None:
    """Return the shell executable to use, or None for the system default.

    On Windows the native shell (cmd.exe via create_subprocess_shell) is used
    so that commands and paths match the Windows environment the user sees.
    """
    return None


def create_shell_tools(registry: ToolRegistry) -> list:
    """Create and register all shell tools. Returns the tool list."""

    @tool(
        registry=registry,
        description="Execute a shell command and return stdout/stderr. Use for running tests, installing packages, git operations, etc.",
        safety_level=SafetyLevel.SHELL,
    )
    async def run_shell(command: str, cwd: str = "", timeout: int = 120) -> dict:
        proc = None
        try:
            working_dir = cwd if cwd else os.getcwd()
            shell_exe = _shell_executable()

            if shell_exe:
                proc = await asyncio.create_subprocess_exec(
                    shell_exe, "-c", command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            content_parts = []
            if stdout_str:
                content_parts.append(stdout_str)
            if stderr_str:
                content_parts.append(f"[stderr]\n{stderr_str}")

            content = "\n".join(content_parts) if content_parts else "(no output)"

            return {
                "success": proc.returncode == 0,
                "content": content,
                "error": None if proc.returncode == 0 else f"Exit code: {proc.returncode}",
                "metadata": {
                    "exit_code": proc.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                },
            }
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return {
                "success": False,
                "content": "",
                "error": f"Command timed out after {timeout}s",
                "metadata": None,
            }
        except Exception as e:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return {
                "success": False,
                "content": "",
                "error": f"Error running command: {e}",
                "metadata": None,
            }

    return [run_shell]
