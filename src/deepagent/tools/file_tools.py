# src/deepagent/tools/file_tools.py
import os
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry, tool


def create_file_tools(registry: ToolRegistry, safe_root: str | None = None) -> list:
    """Create and register all file operation tools. Returns the tool list."""
    root = os.path.abspath(safe_root) if safe_root else None

    def _resolve(path: str) -> str:
        """Resolve path and check it is within safe_root."""
        abs_path = os.path.abspath(path)
        if root is not None:
            if not abs_path.startswith(root + os.sep) and abs_path != root:
                raise ValueError(f"Path '{path}' is outside safe root '{root}'")
        return abs_path

    @tool(registry=registry, description="Read a file from the filesystem. Returns content with line numbers.", safety_level=SafetyLevel.READONLY)
    async def read_file(path: str, offset: int = 1, limit: int = 200) -> dict:
        try:
            resolved = _resolve(path)
            if not os.path.isfile(resolved):
                return {"success": False, "content": "", "error": f"File not found: {path}", "metadata": None}

            with open(resolved, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            start = max(0, offset - 1)
            end = min(total, start + limit)
            selected = lines[start:end]

            output_lines = []
            for i, line in enumerate(selected, start=start + 1):
                output_lines.append(f"{i:>6}\t{line.rstrip()}")

            content = "\n".join(output_lines)
            metadata = {"total_lines": total, "shown_start": start + 1, "shown_end": end}

            return {"success": True, "content": content, "error": None, "metadata": metadata}
        except ValueError as e:
            return {"success": False, "content": "", "error": str(e), "metadata": None}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Error reading file: {e}", "metadata": None}

    @tool(registry=registry, description="Create a new file or overwrite an existing file with the given content.", safety_level=SafetyLevel.WRITE)
    async def write_file(path: str, content: str) -> dict:
        try:
            resolved = _resolve(path)
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(content)
            file_size = os.path.getsize(resolved)
            return {"success": True, "content": f"File written: {path} ({file_size} bytes)", "error": None, "metadata": {"size": file_size}}
        except ValueError as e:
            return {"success": False, "content": "", "error": str(e), "metadata": None}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Error writing file: {e}", "metadata": None}

    @tool(registry=registry, description="Replace a specific string in a file with a new string. The old_string must match exactly once.", safety_level=SafetyLevel.WRITE)
    async def edit_file(path: str, old_string: str, new_string: str) -> dict:
        try:
            resolved = _resolve(path)
            if not os.path.isfile(resolved):
                return {"success": False, "content": "", "error": f"File not found: {path}", "metadata": None}

            with open(resolved, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_string)
            if count == 0:
                return {"success": False, "content": "", "error": f"old_string not found in file: {repr(old_string)}", "metadata": None}
            if count > 1:
                # Show context around each match to help craft a unique old_string
                contexts = []
                idx = 0
                for i in range(count):
                    idx = content.find(old_string, idx)
                    line_num = content[:idx].count('\n') + 1
                    contexts.append(f"  Line {line_num}: ...{repr(content[max(0,idx-20):idx+len(old_string)+20])}...")
                    idx += len(old_string)
                ctx_block = "\n".join(contexts[:10])
                return {"success": False, "content": "", "error": f"old_string found {count} times. Must be unique. Matches at:\n{ctx_block}", "metadata": None}

            new_content = content.replace(old_string, new_string, 1)
            with open(resolved, "w", encoding="utf-8") as f:
                f.write(new_content)

            return {"success": True, "content": f"File edited: {path}", "error": None, "metadata": None}
        except ValueError as e:
            return {"success": False, "content": "", "error": str(e), "metadata": None}
        except Exception as e:
            return {"success": False, "content": "", "error": f"Error editing file: {e}", "metadata": None}

    return [read_file, write_file, edit_file]
