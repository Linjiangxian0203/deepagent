# src/deepagent/tools/search_tools.py
import os
import re
from pathlib import Path
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


def create_search_tools(registry: ToolRegistry, safe_root: str | None = None) -> list:
    """Create and register search tools (grep, glob)."""
    root = os.path.abspath(safe_root) if safe_root else None

    def _is_safe(target: str) -> bool:
        if root is None:
            return True
        abs_target = os.path.abspath(target)
        return abs_target.startswith(root + os.sep) or abs_target == root

    @tool(registry=registry, description="Search for a regex pattern in files. Returns matching file paths and line content.", safety_level=SafetyLevel.READONLY)
    async def grep(pattern: str, path: str = ".", glob: str = "") -> dict:
        try:
            if not _is_safe(path):
                return {"success": False, "content": "", "error": f"Path '{path}' is outside safe root", "metadata": None}

            search_dir = Path(path)
            if not search_dir.exists():
                return {"success": False, "content": "", "error": f"Path not found: {path}", "metadata": None}

            compiled = re.compile(pattern)
            results = []
            file_count = 0

            pattern_glob = glob if glob else "*"
            files_to_search = search_dir.rglob(pattern_glob) if "**" in pattern_glob else search_dir.glob(pattern_glob)

            for filepath in files_to_search:
                if not filepath.is_file():
                    continue
                if not _is_safe(str(filepath)):
                    continue
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            if compiled.search(line):
                                results.append(f"{filepath}:{line_no}: {line.rstrip()[:200]}")
                except (OSError, UnicodeDecodeError):
                    continue
                file_count += 1

            if not results:
                return {
                    "success": True,
                    "content": f"No matches found for pattern '{pattern}' in {file_count} files",
                    "error": None,
                    "metadata": {"match_count": 0, "files_searched": file_count},
                }

            content = "\n".join(results[:500])
            if len(results) > 500:
                content += f"\n... and {len(results) - 500} more matches"

            return {
                "success": True,
                "content": content,
                "error": None,
                "metadata": {"match_count": len(results), "files_searched": file_count},
            }
        except re.error as e:
            return {"success": False, "content": "", "error": f"Invalid regex pattern: {e}", "metadata": None}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Error in grep: {e}", "metadata": None}

    @tool(registry=registry, description="Find files matching a glob pattern. Use ** for recursive search.", safety_level=SafetyLevel.READONLY)
    async def glob(pattern: str, path: str = ".") -> dict:
        try:
            if not _is_safe(path):
                return {"success": False, "content": "", "error": f"Path '{path}' is outside safe root", "metadata": None}

            search_dir = Path(path)
            if not search_dir.exists():
                return {"success": False, "content": "", "error": f"Path not found: {path}", "metadata": None}

            matches = list(search_dir.glob(pattern))
            files = [str(m) for m in matches if m.is_file()]
            dirs = [str(m) for m in matches if m.is_dir()]

            lines = []
            if files:
                for f in sorted(files):
                    lines.append(f)
            if dirs:
                for d in sorted(dirs):
                    lines.append(f"{d}/")

            if not lines:
                return {
                    "success": True,
                    "content": f"No files found matching '{pattern}'",
                    "error": None,
                    "metadata": {"file_count": 0, "dir_count": 0},
                }

            return {
                "success": True,
                "content": "\n".join(lines[:500]),
                "error": None,
                "metadata": {"file_count": len(files), "dir_count": len(dirs)},
            }
        except Exception as e:
            return {"success": False, "content": "", "error": f"Error in glob: {e}", "metadata": None}

    return [grep, glob]
