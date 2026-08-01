"""MCP (Model Context Protocol) client system for deepagent.

Manages MCP server processes via JSON-RPC 2.0 over stdio transport.
Each MCP server runs as a subprocess; tools are discovered and registered
into the ToolRegistry with the naming convention mcp__{server}__{tool}.

Reference: docs/mcp-interface-contract.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from deepagent.tools.protocol import SafetyLevel

logger = logging.getLogger(__name__)

_DISALLOWED_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

# Timeouts in seconds
_INIT_TIMEOUT = 30.0
_TOOL_CALL_TIMEOUT = 60.0


def normalize_mcp_name(name: str) -> str:
    """Replace any character NOT in [a-zA-Z0-9_-] with underscore."""
    return _DISALLOWED_CHARS.sub("_", name)


@dataclass
class MCPToolDef:
    """A tool discovered from an MCP server via tools/list."""

    name: str
    description: str
    inputSchema: dict


class MCPClient:
    """Manages a single MCP server connection via stdio transport.

    Lifecycle: connect() -> call_tool() -> disconnect()

    JSON-RPC 2.0 messages are single lines of JSON over stdin/stdout.
    Responses are matched to requests by ``id`` via a pending futures dict.
    """

    def __init__(self, name: str, command: str, args: list[str] | None = None):
        self.name = name
        self.tools: list[MCPToolDef] = []

        self._command = command
        self._args: list[str] = args or []

        self._proc: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

        self._stdin_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Launch the subprocess, send initialize, and discover tools.

        Raises RuntimeError if the subprocess fails to start or the
        initialize / tools/list handshake times out.
        """
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise RuntimeError(
                f"MCP server '{self.name}' failed to start: {self._command} "
                f"{' '.join(self._args)} — {exc}"
            ) from exc

        # Start the background reader *before* sending any requests so we
        # never miss a response line.
        self._reader_task = asyncio.create_task(self._read_loop())

        try:
            # 1. Initialize
            init_id = await self._send("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "deepagent", "version": "0.1.0"},
            })
            await asyncio.wait_for(
                self._pending[init_id], timeout=_INIT_TIMEOUT
            )

            # 2. Discover tools
            tools_id = await self._send("tools/list", {})
            result = await asyncio.wait_for(
                self._pending[tools_id], timeout=_INIT_TIMEOUT
            )
            raw_tools = result.get("tools", [])
            self.tools = [
                MCPToolDef(
                    name=t["name"],
                    description=t.get("description", ""),
                    inputSchema=t.get("inputSchema", {}),
                )
                for t in raw_tools
            ]

            logger.info(
                "MCP server '%s' connected — discovered %d tools: %s",
                self.name,
                len(self.tools),
                [t.name for t in self.tools],
            )

        except asyncio.TimeoutError:
            await self.disconnect()
            raise RuntimeError(
                f"MCP server '{self.name}' timed out during initialization"
            )
        except Exception:
            await self.disconnect()
            raise

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke a tool on the MCP server and return the result text.

        Args:
            tool_name: The MCP tool name (as reported by the server).
            arguments: Keyword arguments for the tool.

        Returns:
            The ``content[0].text`` value from the response.

        Raises RuntimeError on timeout, protocol error, or when the
        server reports ``isError: true``.
        """
        if self._proc is None or self._proc.returncode is not None:
            raise RuntimeError(
                f"MCP server '{self.name}' is not connected"
            )

        req_id = await self._send("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        try:
            result = await asyncio.wait_for(
                self._pending[req_id], timeout=_TOOL_CALL_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"MCP tool '{tool_name}' on server '{self.name}' timed out"
            )

        content = result.get("content", [])
        is_error = result.get("isError", False)

        text = ""
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if text:
                    break
            elif isinstance(item, str):
                text = item
                break

        if is_error:
            raise RuntimeError(
                f"MCP tool '{tool_name}' on server '{self.name}' "
                f"returned an error: {text}"
            )

        return text

    async def disconnect(self) -> None:
        """Terminate the subprocess and cancel the reader task."""
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass  # reader may have crashed; proceed with cleanup
            self._reader_task = None

        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            except ProcessLookupError:
                pass  # Already exited
            self._proc = None

        # Reject any outstanding futures so callers don't hang forever.
        for future in self._pending.values():
            if not future.done():
                future.set_exception(
                    RuntimeError(f"MCP server '{self.name}' disconnected")
                )
        self._pending.clear()

        logger.info("MCP server '%s' disconnected", self.name)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _send(self, method: str, params: dict | None = None) -> int:
        """Send a JSON-RPC request over stdin. Returns the request id.

        Creates a Future stored in ``_pending``; the reader loop resolves it
        when the matching response arrives.
        """
        self._next_id += 1
        req_id = self._next_id

        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        line = json.dumps(message, ensure_ascii=False) + "\n"

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future

        try:
            async with self._stdin_lock:
                self._proc.stdin.write(line.encode("utf-8"))
                await self._proc.stdin.drain()
        except (BrokenPipeError, OSError, AttributeError):
            self._pending.pop(req_id, None)
            raise

        return req_id

    async def _read_loop(self):
        """Background task: read stdout line by line, resolve futures.

        Lines with an ``id`` field are JSON-RPC responses — the matching
        future is resolved.  Lines without an ``id`` are notifications
        (e.g. progress updates) and are logged at DEBUG level.
        """
        try:
            while True:
                line_bytes = await self._proc.stdout.readline()
                if not line_bytes:
                    # EOF — subprocess exited
                    break

                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "MCP server '%s' sent non-JSON line: %s",
                        self.name, line[:200],
                    )
                    continue

                msg_id = message.get("id")
                if msg_id is not None and msg_id in self._pending:
                    future = self._pending.pop(msg_id)

                    if "error" in message:
                        err = message["error"]
                        if not future.done():
                            future.set_exception(
                                RuntimeError(
                                    f"MCP server '{self.name}' JSON-RPC error "
                                    f"(code={err.get('code')}): {err.get('message', '')}"
                                )
                            )
                    else:
                        if not future.done():
                            future.set_result(message.get("result", {}))
                else:
                    # Notification (no id) or unrecognized response
                    logger.debug(
                        "MCP server '%s' notification: method=%s",
                        self.name, message.get("method", "<none>"),
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("MCP server '%s' read loop crashed", self.name)
            # Reject all pending futures so callers don't hang until timeout
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(
                        RuntimeError(f"MCP server '{self.name}' read loop crashed")
                    )
            self._pending.clear()


# ------------------------------------------------------------------
# Tool wrapper helpers
# ------------------------------------------------------------------

def _infer_safety_level(description: str) -> SafetyLevel:
    """Guess the safety level of an MCP tool from its description.

    Looks for keywords suggesting mutation. Defaults to READONLY.
    """
    desc_lower = description.lower()
    write_keywords = (
        "destructive", "write", "delete", "deploy",
        "create", "modify", "update",
    )
    if any(kw in desc_lower for kw in write_keywords):
        return SafetyLevel.WRITE
    return SafetyLevel.READONLY


async def _make_mcp_wrapper(
    client: MCPClient, tool_def: MCPToolDef, full_name: str,
):
    """Build a ToolProtocol-compatible wrapper for a single MCP tool."""

    async def handler(**kwargs):
        return await client.call_tool(tool_def.name, kwargs)

    handler.tool_name = full_name
    handler.tool_description = f"[MCP:{client.name}] {tool_def.description}"
    handler.tool_parameters = _build_input_schema(tool_def.inputSchema)
    handler.tool_safety_level = _infer_safety_level(tool_def.description)

    return handler


def _build_input_schema(raw_schema: dict) -> dict:
    """Ensure the inputSchema has required JSON Schema fields."""
    schema = raw_schema.copy()
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    schema.setdefault("required", [])
    return schema


# ------------------------------------------------------------------
# MCPManager — orchestrates multiple MCPClient instances
# ------------------------------------------------------------------

class MCPManager:
    """Orchestrates multiple :class:`MCPClient` connections.

    Handles connect, disconnect, tool discovery, and dynamic registration
    into the shared :class:`ToolRegistry`.
    """

    def __init__(self, tool_registry):
        self._clients: dict[str, MCPClient] = {}
        self._registry = tool_registry
        self._server_tools: dict[str, list[str]] = {}

    async def connect_server(
        self, name: str, command: str, args: list[str] | None = None,
    ) -> str:
        """Create an MCPClient, connect, discover tools, and register them.

        Returns a status message describing success or failure.
        """
        safe_name = normalize_mcp_name(name)

        if safe_name in self._clients:
            return f"MCP server '{name}' is already connected"

        client = MCPClient(name=safe_name, command=command, args=args)

        try:
            await client.connect()
        except RuntimeError as exc:
            return f"Failed to connect to MCP server '{name}': {exc}"

        self._clients[safe_name] = client
        registered: list[str] = []

        for tool_def in client.tools:
            safe_tool = normalize_mcp_name(tool_def.name)
            full_name = f"mcp__{safe_name}__{safe_tool}"

            wrapper = await _make_mcp_wrapper(client, tool_def, full_name)
            self._registry.register(wrapper)
            registered.append(full_name)

        self._server_tools[safe_name] = registered

        tool_list = ", ".join(registered) if registered else "(none)"
        logger.info(
            "MCP server '%s' registered %d tools: %s",
            name, len(registered), tool_list,
        )
        return (
            f"Connected to MCP server '{name}' — "
            f"{len(registered)} tool(s) registered: {tool_list}"
        )

    async def disconnect_server(self, name: str) -> str:
        """Disconnect an MCP server and unregister its tools.

        Returns a status message.
        """
        safe_name = normalize_mcp_name(name)
        client = self._clients.get(safe_name)

        if client is None:
            return f"MCP server '{name}' is not connected"

        await client.disconnect()

        # Remove registered tool wrappers from the registry.
        tool_names = self._server_tools.pop(safe_name, [])
        for tool_name in tool_names:
            self._registry._tools.pop(tool_name, None)

        del self._clients[safe_name]

        logger.info(
            "MCP server '%s' disconnected — %d tools unregistered",
            name, len(tool_names),
        )
        return (
            f"Disconnected MCP server '{name}' — "
            f"{len(tool_names)} tool(s) unregistered"
        )

    def list_servers(self) -> list[str]:
        """Return names of currently connected MCP servers."""
        return list(self._clients.keys())

    def get_client(self, name: str) -> MCPClient | None:
        """Return the :class:`MCPClient` for *name*, or ``None``."""
        return self._clients.get(normalize_mcp_name(name))

    def is_connected(self, name: str) -> bool:
        """Return ``True`` if *name* is a connected MCP server."""
        return normalize_mcp_name(name) in self._clients
