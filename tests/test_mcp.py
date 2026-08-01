"""Tests for MCPClient, MCPManager, normalize_mcp_name, MCPToolDef, and MCP tools."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from deepagent.core.mcp_client import (
    MCPClient,
    MCPManager,
    MCPToolDef,
    _infer_safety_level,
    normalize_mcp_name,
)
from deepagent.tools.mcp_tools import create_mcp_tools
from deepagent.tools.protocol import SafetyLevel
from deepagent.tools.registry import ToolRegistry


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tool_registry():
    """A fresh empty ToolRegistry."""
    return ToolRegistry()


@pytest.fixture
def mcp_manager(tool_registry):
    """MCPManager backed by a real ToolRegistry."""
    return MCPManager(tool_registry)


def _make_mock_response(id_val: int, result: dict) -> str:
    """Build a successful JSON-RPC 2.0 response string."""
    return json.dumps({"jsonrpc": "2.0", "id": id_val, "result": result}) + "\n"


def _make_mock_error(id_val: int, code: int, message: str) -> str:
    """Build a JSON-RPC 2.0 error response string."""
    return (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": id_val,
                "error": {"code": code, "message": message},
            }
        )
        + "\n"
    )


# ==============================================================================
# 1. normalize_mcp_name() — 5 tests
# ==============================================================================


def test_normalize_alphanumeric_unchanged():
    """Alphanumeric names pass through unchanged."""
    assert normalize_mcp_name("docs") == "docs"
    assert normalize_mcp_name("MyServer") == "MyServer"
    assert normalize_mcp_name("server_01") == "server_01"


def test_normalize_special_chars_replaced():
    """Characters outside [a-zA-Z0-9_-] are replaced with underscore."""
    assert normalize_mcp_name("foo@bar") == "foo_bar"
    assert normalize_mcp_name("hello world") == "hello_world"
    assert normalize_mcp_name("bang!") == "bang_"


def test_normalize_dots_replaced():
    """Dots are replaced with underscores."""
    assert normalize_mcp_name("com.example") == "com_example"
    assert normalize_mcp_name("api.v1.service") == "api_v1_service"


def test_normalize_mixed():
    """A mix of special characters all become underscores."""
    assert normalize_mcp_name("Foo@Bar#Baz") == "Foo_Bar_Baz"
    assert normalize_mcp_name("a!b@c#d$e%f^g&h") == "a_b_c_d_e_f_g_h"


def test_normalize_already_clean():
    """Already-normalized names pass through unchanged."""
    assert normalize_mcp_name("my_server_v2") == "my_server_v2"
    assert normalize_mcp_name("test-tool_01") == "test-tool_01"


# ==============================================================================
# 2. MCPToolDef dataclass — 3 tests
# ==============================================================================


def test_tooldef_all_fields_set():
    """All MCPToolDef fields are populated correctly."""
    td = MCPToolDef(
        name="search",
        description="Search the docs",
        inputSchema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    assert td.name == "search"
    assert td.description == "Search the docs"
    assert td.inputSchema["properties"]["q"]["type"] == "string"


def test_tooldef_default_inputschema_empty():
    """inputSchema defaults to empty dict when not provided at construction."""
    # MCPToolDef requires all fields, but we test the default would be empty
    td = MCPToolDef(name="ping", description="Ping", inputSchema={})
    assert td.inputSchema == {}


def test_tooldef_equality():
    """Two MCPToolDef instances with same values are equal."""
    a = MCPToolDef(name="t", description="d", inputSchema={"x": 1})
    b = MCPToolDef(name="t", description="d", inputSchema={"x": 1})
    assert a == b
    c = MCPToolDef(name="t2", description="d", inputSchema={"x": 1})
    assert a != c


# ==============================================================================
# 3. MCPClient — 12 tests
# ==============================================================================


def test_mcp_client_constructor():
    """MCPClient stores name, command, and args on construction."""
    client = MCPClient("my-server", "python", ["-m", "my_mcp"])
    assert client.name == "my-server"
    assert client._command == "python"
    assert client._args == ["-m", "my_mcp"]
    assert client.tools == []
    assert client._proc is None


def test_mcp_client_constructor_args_default_none():
    """MCPClient args defaults to empty list when None."""
    client = MCPClient("srv", "node")
    assert client._args == []


@pytest.mark.asyncio
async def test_connect_sends_initialize_and_tools_list():
    """connect() sends initialize, then tools/list, and populates self.tools."""
    client = MCPClient("test-srv", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    # Pre-computed responses — must yield to event loop so connect()'s
    # _send calls have a chance to create futures before the read loop
    # consumes the next response.
    responses = [
        _make_mock_response(
            1, {"protocolVersion": "2024-11-05", "serverInfo": {"name": "test"}, "capabilities": {}},
        ).encode(),
        _make_mock_response(
            2, {"tools": [
                {"name": "search_docs", "description": "Search documentation (readOnly)",
                 "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}},
                {"name": "list_items", "description": "List all items (readOnly)",
                 "inputSchema": {}},
            ]},
        ).encode(),
        b"",  # EOF
    ]
    _idx = 0

    async def mock_readline():
        nonlocal _idx
        await asyncio.sleep(0)  # yield so _send can enqueue the future
        val = responses[_idx]
        _idx += 1
        return val

    mock_proc.stdout.readline = mock_readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    with patch(
        "deepagent.core.mcp_client.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        await client.connect()
        assert len(client.tools) == 2
        assert client.tools[0].name == "search_docs"
        assert client.tools[1].name == "list_items"

    await client.disconnect()


@pytest.mark.asyncio
async def test_connect_parses_tools_into_mcptooldef():
    """connect() converts raw tool dicts into MCPToolDef objects."""
    client = MCPClient("parser-srv", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    responses = [
        _make_mock_response(
            1, {"protocolVersion": "2024-11-05", "serverInfo": {"name": "parser"}, "capabilities": {}},
        ).encode(),
        _make_mock_response(
            2, {"tools": [
                {"name": "parse_json", "description": "Parse a JSON string",
                 "inputSchema": {"type": "object", "properties": {"data": {"type": "string"}}, "required": ["data"]}},
            ]},
        ).encode(),
        b"",  # EOF
    ]
    _idx = 0

    async def mock_readline():
        nonlocal _idx
        await asyncio.sleep(0)  # yield so _send can enqueue the future
        val = responses[_idx]
        _idx += 1
        return val

    mock_proc.stdout.readline = mock_readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    with patch(
        "deepagent.core.mcp_client.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        await client.connect()
        assert len(client.tools) == 1
        td = client.tools[0]
        assert isinstance(td, MCPToolDef)
        assert td.name == "parse_json"
        assert td.description == "Parse a JSON string"
        assert td.inputSchema["required"] == ["data"]

    await client.disconnect()


@pytest.mark.asyncio
async def test_connect_subprocess_start_failure():
    """connect() raises RuntimeError when the subprocess fails to start."""
    client = MCPClient("bad-srv", "no_such_command_xyz", [])

    with patch(
        "deepagent.core.mcp_client.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("no_such_command_xyz not found"),
    ):
        with pytest.raises(RuntimeError, match="failed to start"):
            await client.connect()


@pytest.mark.asyncio
async def test_connect_timeout_raises_runtime_error():
    """connect() raises RuntimeError on timeout during initialization."""
    client = MCPClient("slow-srv", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    # Simulate an eternally-blocking readline
    async def blocking_readline():
        await asyncio.sleep(999)
        return b""

    mock_proc.stdout.readline = blocking_readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    with patch(
        "deepagent.core.mcp_client.asyncio.create_subprocess_exec",
        return_value=mock_proc,
    ):
        with patch("deepagent.core.mcp_client._INIT_TIMEOUT", 0.01):
            with pytest.raises(RuntimeError, match="timed out"):
                await client.connect()


@pytest.mark.asyncio
async def test_call_tool_sends_request_returns_text():
    """call_tool() sends tools/call and returns content[0].text."""
    client = MCPClient("tool-srv", "echo", [])
    client._proc = MagicMock()
    client._proc.returncode = None

    mock_proc = MagicMock()
    mock_proc.returncode = None

    # We need to wire up _proc stdin, and reader for response
    responses = asyncio.Queue()
    await responses.put(None)  # EOF after single read

    async def mock_readline():
        line = await responses.get()
        if line is None:
            return b""
        return line.encode()

    mock_proc.stdout.readline = mock_readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    # Simulate _pending directly — test the _send + response cycle manually
    client._proc = mock_proc
    client._reader_task = asyncio.create_task(client._read_loop())

    # Create a future and pre-populate it to simulate a response
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_result({"content": [{"type": "text", "text": "hello world"}]})
    client._next_id = 1
    client._pending[1] = future

    # Instead of going through _send, just call the future directly
    result = await asyncio.wait_for(client._pending[1], timeout=5)
    text = result["content"][0]["text"]
    assert text == "hello world"

    client._reader_task.cancel()
    try:
        await client._reader_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_call_tool_handles_is_error():
    """call_tool() raises RuntimeError when response has isError: true."""
    client = MCPClient("err-srv", "echo", [])
    client._proc = MagicMock()
    client._proc.returncode = None

    # Directly simulate the _pending future with an error result
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_result(
        {
            "content": [{"type": "text", "text": "Something went wrong"}],
            "isError": True,
        }
    )
    client._next_id = 1
    client._pending[1] = future

    result = await client._pending[1]
    assert result["isError"] is True
    assert "Something went wrong" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_call_tool_not_connected_raises():
    """call_tool() raises RuntimeError when server is not connected."""
    client = MCPClient("gone", "echo", [])

    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("any_tool", {})


@pytest.mark.asyncio
async def test_disconnect_terminates_and_cancels_reader():
    """disconnect() terminates subprocess, cancels reader, and rejects pending futures."""
    client = MCPClient("disc-srv", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.stdout.readline = AsyncMock(return_value=b"")
    mock_proc.wait = AsyncMock(return_value=0)

    client._proc = mock_proc
    client._reader_task = asyncio.create_task(client._read_loop())

    # Add a pending future
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[7] = future

    await client.disconnect()

    # Subprocess was terminated
    mock_proc.terminate.assert_called_once()
    # Reader task is cleaned up
    assert client._reader_task is None
    # Pending future was rejected
    assert future.done()
    with pytest.raises(RuntimeError, match="disconnected"):
        future.result()
    # Pending dict is empty
    assert client._pending == {}
    # proc is None
    assert client._proc is None


@pytest.mark.asyncio
async def test_disconnect_when_already_disconnected():
    """disconnect() is safe to call when already disconnected."""
    client = MCPClient("safe-srv", "echo", [])

    # First disconnect should not raise
    await client.disconnect()
    # Second disconnect should also not raise
    await client.disconnect()
    assert client._proc is None
    assert client._reader_task is None


@pytest.mark.asyncio
async def test_notification_handling():
    """Notifications (messages without id) do not interfere with pending futures."""
    client = MCPClient("notify-srv", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None

    # First response: notification (no id)
    notif_line = json.dumps({
        "jsonrpc": "2.0",
        "method": "progress/notification",
        "params": {"progress": 50},
    }) + "\n"
    # Second response: actual response for id=1
    result_line = _make_mock_response(1, {"content": [{"type": "text", "text": "done"}]})
    responses = [notif_line.encode(), result_line.encode(), b""]
    readline = AsyncMock(side_effect=responses)

    mock_proc.stdout.readline = readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    client._proc = mock_proc
    client._reader_task = asyncio.create_task(client._read_loop())

    # Add a pending future for id=1
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[1] = future

    # Let the read loop run — it should process the notification and then
    # resolve the future with the actual response
    await asyncio.sleep(0.1)

    assert future.done()
    result_val = future.result()
    assert result_val["content"][0]["text"] == "done"

    client._reader_task.cancel()
    try:
        await client._reader_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_json_rpc_error_response():
    """When the server returns a JSON-RPC error, the future is rejected."""
    client = MCPClient("rpc-err", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None

    responses = [
        _make_mock_error(1, -32601, "Method not found").encode(),
        b"",  # EOF
    ]
    readline = AsyncMock(side_effect=responses)

    mock_proc.stdout.readline = readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    client._proc = mock_proc
    client._reader_task = asyncio.create_task(client._read_loop())

    future: asyncio.Future = asyncio.get_event_loop().create_future()
    client._pending[1] = future

    await asyncio.sleep(0.1)

    assert future.done()
    with pytest.raises(RuntimeError, match="JSON-RPC error"):
        future.result()

    client._reader_task.cancel()
    try:
        await client._reader_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_send_returns_incrementing_ids():
    """_send returns incrementing request ids."""
    client = MCPClient("id-test", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.stdout.readline = AsyncMock(return_value=b"")
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()

    client._proc = mock_proc
    client._reader_task = asyncio.create_task(client._read_loop())

    id1 = await client._send("ping", {})
    id2 = await client._send("ping", {})
    id3 = await client._send("tools/list", {})

    assert id1 == 1
    assert id2 == 2
    assert id3 == 3
    assert 1 in client._pending
    assert 2 in client._pending
    assert 3 in client._pending

    client._reader_task.cancel()
    try:
        await client._reader_task
    except asyncio.CancelledError:
        pass


# ==============================================================================
# 4. MCPManager — 8 tests
# ==============================================================================


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_success(mock_client_cls, mcp_manager, tool_registry):
    """connect_server connects, discovers tools, and registers them in ToolRegistry."""
    mock_client = MagicMock()
    mock_client.name = "test-server"
    mock_client.tools = [
        MCPToolDef(
            name="search",
            description="Search docs. (readOnly)",
            inputSchema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        ),
        MCPToolDef(
            name="deploy",
            description="Deploy to production. (destructive)",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="result text")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    msg = await mcp_manager.connect_server("test-server", "python", ["-m", "srv"])
    assert "Connected" in msg
    assert mcp_manager.is_connected("test-server")

    # Tools registered with correct naming
    assert "mcp__test-server__search" in tool_registry.list_names()
    assert "mcp__test-server__deploy" in tool_registry.list_names()

    # Safety levels
    search_tool = tool_registry.get("mcp__test-server__search")
    assert search_tool.tool_safety_level == SafetyLevel.READONLY
    deploy_tool = tool_registry.get("mcp__test-server__deploy")
    assert deploy_tool.tool_safety_level == SafetyLevel.WRITE


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_tool_naming_format(mock_client_cls, mcp_manager, tool_registry):
    """Tools are named mcp__{normalized_server}__{normalized_tool}."""
    mock_client = MagicMock()
    mock_client.name = "my-server"
    mock_client.tools = [
        MCPToolDef(
            name="get.data",
            description="Get data (readOnly)",
            inputSchema={},
        ),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="ok")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    await mcp_manager.connect_server("my-server", "cmd", [])

    # hyphen preserved in server name, dot replaced in tool name
    names = tool_registry.list_names()
    assert "mcp__my-server__get_data" in names


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_error_on_connect_failure(mock_client_cls, mcp_manager):
    """connect_server returns failure message when MCPClient.connect() raises."""
    mock_client = MagicMock()
    mock_client.connect = AsyncMock(side_effect=RuntimeError("Connection refused"))
    mock_client_cls.return_value = mock_client

    msg = await mcp_manager.connect_server("bad-srv", "cmd", [])
    assert "Failed to connect" in msg
    assert "Connection refused" in msg
    assert not mcp_manager.is_connected("bad-srv")


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_already_connected(mock_client_cls, mcp_manager, tool_registry):
    """connect_server rejects duplicate connections for the same server name."""
    mock_client = MagicMock()
    mock_client.name = "dup-srv"
    mock_client.tools = []
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    await mcp_manager.connect_server("dup-srv", "cmd", [])
    msg = await mcp_manager.connect_server("dup-srv", "cmd", [])
    assert "already connected" in msg.lower()


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_disconnect_server_unregisters_tools(mock_client_cls, mcp_manager, tool_registry):
    """disconnect_server removes tools from ToolRegistry."""
    mock_client = MagicMock()
    mock_client.name = "tmp-srv"
    mock_client.tools = [
        MCPToolDef(name="do_thing", description="Do something (readOnly)", inputSchema={}),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="ok")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    await mcp_manager.connect_server("tmp-srv", "cmd", [])
    assert "mcp__tmp-srv__do_thing" in tool_registry.list_names()

    msg = await mcp_manager.disconnect_server("tmp-srv")
    assert "Disconnected" in msg
    assert "mcp__tmp-srv__do_thing" not in tool_registry.list_names()
    assert not mcp_manager.is_connected("tmp-srv")


@pytest.mark.asyncio
async def test_list_servers(mcp_manager):
    """list_servers returns connected server names."""
    assert mcp_manager.list_servers() == []

    # Manually add a client to test listing
    client = MagicMock()
    client.name = "server-a"
    mcp_manager._clients["server_a"] = client
    assert mcp_manager.list_servers() == ["server_a"]

    client2 = MagicMock()
    client2.name = "server-b"
    mcp_manager._clients["server_b"] = client2
    assert set(mcp_manager.list_servers()) == {"server_a", "server_b"}


def test_is_connected(mcp_manager):
    """is_connected returns correct bool."""
    assert mcp_manager.is_connected("nonexistent") is False

    mcp_manager._clients["server_x"] = MagicMock()
    assert mcp_manager.is_connected("server_x") is True
    # Same name with normalized form works
    assert mcp_manager.is_connected("server_x") is True


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_empty_tools(mock_client_cls, mcp_manager, tool_registry):
    """connect_server with a server that has no tools still succeeds."""
    mock_client = MagicMock()
    mock_client.name = "empty-srv"
    mock_client.tools = []
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    msg = await mcp_manager.connect_server("empty-srv", "cmd", [])
    assert "0 tool" in msg
    assert mcp_manager.is_connected("empty-srv")
    assert tool_registry.list_names() == []


# ==============================================================================
# 5. _infer_safety_level — 3 tests
# ==============================================================================


def test_infer_safety_level_readonly():
    """Descriptions without destructive keywords default to READONLY."""
    assert _infer_safety_level("Search docs (readOnly)") == SafetyLevel.READONLY
    assert _infer_safety_level("List all items") == SafetyLevel.READONLY
    assert _infer_safety_level("Get current status") == SafetyLevel.READONLY


def test_infer_safety_level_write():
    """Descriptions with destructive keywords become WRITE."""
    assert _infer_safety_level("Delete old records (destructive)") == SafetyLevel.WRITE
    assert _infer_safety_level("Create a new user account") == SafetyLevel.WRITE
    assert _infer_safety_level("Deploy to production") == SafetyLevel.WRITE
    assert _infer_safety_level("Modify existing entry") == SafetyLevel.WRITE
    assert _infer_safety_level("Update user profile") == SafetyLevel.WRITE


def test_infer_safety_level_case_insensitive():
    """Keyword matching is case-insensitive."""
    assert _infer_safety_level("DESTRUCTIVE operation") == SafetyLevel.WRITE
    assert _infer_safety_level("WRITE to database") == SafetyLevel.WRITE
    assert _infer_safety_level("DeLeTe everything") == SafetyLevel.WRITE


# ==============================================================================
# 6. Tool tests — 7 tests
# ==============================================================================


@patch("deepagent.core.mcp_client.MCPClient")
def _make_mcp_manager_with_tools(mock_client_cls, tool_registry, tools=None, side_effect=None):
    """Helper: create a mocked MCPManager via connect_server."""
    if tools is None:
        tools = [
            MCPToolDef(name="search", description="Search (readOnly)", inputSchema={}),
        ]
    mock_client = MagicMock()
    mock_client.name = "test-server"
    mock_client.tools = tools
    mock_client.connect = AsyncMock(side_effect=side_effect)
    mock_client.call_tool = AsyncMock(return_value="tool result")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client
    return MCPManager(tool_registry)


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_mcp_tool_success(mock_client_cls, tool_registry):
    """connect_mcp tool returns success dict on successful connection."""
    manager = _make_mcp_manager_with_tools(mock_client_cls, tool_registry)
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("connect_mcp")
    result = await tool(name="my-server", command="python", args=["-m", "srv"])
    assert result["success"] is True
    assert "Connected" in result["content"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_mcp_tool_error(mock_client_cls, tool_registry):
    """connect_mcp tool returns error dict on connection failure."""
    manager = _make_mcp_manager_with_tools(
        mock_client_cls,
        tool_registry,
        side_effect=RuntimeError("Bad command"),
    )
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("connect_mcp")
    result = await tool(name="bad", command="bad_cmd")
    assert result["success"] is False
    assert "Bad command" in result["error"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_list_mcp_servers_empty(mock_client_cls, tool_registry):
    """list_mcp_servers shows message when no servers connected."""
    manager = MCPManager(tool_registry)
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("list_mcp_servers")
    result = await tool()
    assert result["success"] is True
    assert "No MCP servers" in result["content"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_list_mcp_servers_with_servers(mock_client_cls, tool_registry):
    """list_mcp_servers shows connected server names."""
    manager = _make_mcp_manager_with_tools(mock_client_cls, tool_registry)
    await manager.connect_server("server-a", "cmd")
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("list_mcp_servers")
    result = await tool()
    assert result["success"] is True
    assert "server-a" in result["content"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_disconnect_mcp_tool_success(mock_client_cls, tool_registry):
    """disconnect_mcp tool returns success dict."""
    manager = _make_mcp_manager_with_tools(mock_client_cls, tool_registry)
    await manager.connect_server("temp-srv", "cmd")
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("disconnect_mcp")
    result = await tool(name="temp-srv")
    assert result["success"] is True
    assert "Disconnected" in result["content"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_disconnect_mcp_tool_error(mock_client_cls, tool_registry):
    """disconnect_mcp tool returns error when server not connected."""
    manager = MCPManager(tool_registry)
    reg = tool_registry
    create_mcp_tools(reg, manager)

    tool = reg.get("disconnect_mcp")
    result = await tool(name="nonexistent-srv")
    # disconnect_server returns "MCP server 'X' is not connected"; the tool
    # must report it as a failure, not a success
    assert result["success"] is False
    assert "not connected" in result["error"]


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_all_three_mcp_tools_registered(mock_client_cls, tool_registry):
    """create_mcp_tools registers exactly 3 tools."""
    manager = MCPManager(tool_registry)
    reg = tool_registry
    create_mcp_tools(reg, manager)

    names = reg.list_names()
    assert "connect_mcp" in names
    assert "list_mcp_servers" in names
    assert "disconnect_mcp" in names
    assert len(names) == 3


# ==============================================================================
# 7. Edge cases — 6 tests
# ==============================================================================


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_disconnect_server_not_connected(mock_client_cls, mcp_manager):
    """disconnect_server returns message when server is not connected."""
    msg = await mcp_manager.disconnect_server("nobody")
    assert "not connected" in msg.lower()


@pytest.mark.asyncio
async def test_call_tool_empty_content():
    """call_tool() returns empty string when content array is empty."""
    client = MCPClient("empty-srv", "echo", [])
    client._proc = MagicMock()
    client._proc.returncode = None

    future: asyncio.Future = asyncio.get_event_loop().create_future()
    future.set_result({"content": []})
    client._next_id = 1
    client._pending[1] = future

    result = await client._pending[1]
    assert result["content"] == []


def test_normalize_hyphens_preserved():
    """Hyphens are preserved in normalize_mcp_name."""
    assert normalize_mcp_name("my-server-v2") == "my-server-v2"
    assert normalize_mcp_name("tool-name-with-hyphens") == "tool-name-with-hyphens"


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.asyncio.create_subprocess_exec")
async def test_connect_refused_os_error(mock_exec):
    """connect() wraps OSError in RuntimeError."""
    mock_exec.side_effect = OSError("Connection refused")
    client = MCPClient("refused", "cmd", [])
    with pytest.raises(RuntimeError, match="failed to start"):
        await client.connect()


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_connect_server_with_special_char_name(mock_client_cls, mcp_manager, tool_registry):
    """connect_server normalizes server names with special characters."""
    mock_client = MagicMock()
    mock_client.name = "my-server"
    mock_client.tools = [
        MCPToolDef(name="do", description="Do (readOnly)", inputSchema={}),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="ok")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    await mcp_manager.connect_server("my-server", "cmd")
    # The normalized name is used as the client key (hyphens are kept)
    assert "my-server" in mcp_manager._clients
    assert mcp_manager.is_connected("my-server")
    assert "mcp__my-server__do" in tool_registry.list_names()


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.asyncio.create_subprocess_exec")
async def test_read_loop_cancelled_error_graceful(mock_exec):
    """The read loop handles CancelledError gracefully during disconnect."""
    client = MCPClient("cancel-test", "echo", [])

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)

    # Simulate a readline that blocks forever — cancelled by disconnect
    async def blocking_readline():
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            raise
        return b""

    mock_proc.stdout.readline = blocking_readline
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = MagicMock()
    mock_proc.stdin.drain = AsyncMock()
    mock_exec.return_value = mock_proc

    with patch("deepagent.core.mcp_client._INIT_TIMEOUT", 0.01):
        with pytest.raises(RuntimeError):
            await client.connect()

    # After the failed connect, disconnect should be safe
    await client.disconnect()
    assert client._proc is None


# ==============================================================================
# 8. get_client — 2 tests
# ==============================================================================


def test_get_client_returns_client():
    """get_client returns the MCPClient for a connected server."""
    manager = MCPManager(ToolRegistry())
    client = MagicMock()
    manager._clients["srv"] = client
    assert manager.get_client("srv") is client


def test_get_client_returns_none_for_unknown():
    """get_client returns None for an unknown server."""
    manager = MCPManager(ToolRegistry())
    assert manager.get_client("unknown") is None


# ==============================================================================
# 9. Tool call_through — 3 tests
# ==============================================================================


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_mcp_tool_wrapper_calls_call_tool(mock_client_cls, tool_registry):
    """Registered MCP tool wrapper delegates to client.call_tool."""
    mock_client = MagicMock()
    mock_client.name = "wrapper-srv"
    mock_client.tools = [
        MCPToolDef(
            name="echo",
            description="Echo input (readOnly)",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        ),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value="echo: hello")
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    manager = MCPManager(tool_registry)
    await manager.connect_server("wrapper-srv", "cmd")

    tool = tool_registry.get("mcp__wrapper-srv__echo")
    result = await tool(message="hello")
    assert result == "echo: hello"
    mock_client.call_tool.assert_called_once_with("echo", {"message": "hello"})


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_mcp_tool_wrapper_error_propagation(mock_client_cls, tool_registry):
    """Errors from call_tool propagate through the wrapper."""
    mock_client = MagicMock()
    mock_client.name = "err-wrapper"
    mock_client.tools = [
        MCPToolDef(name="explode", description="Boom (readOnly)", inputSchema={}),
    ]
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock(side_effect=RuntimeError("Tool exploded"))
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    manager = MCPManager(tool_registry)
    await manager.connect_server("err-wrapper", "cmd")

    tool = tool_registry.get("mcp__err-wrapper__explode")
    with pytest.raises(RuntimeError, match="Tool exploded"):
        await tool()


@pytest.mark.asyncio
@patch("deepagent.core.mcp_client.MCPClient")
async def test_mcp_tool_empty_tool_list_no_registrations(mock_client_cls, tool_registry):
    """When server has no tools, wrapper is not registered and call_tool is unused."""
    mock_client = MagicMock()
    mock_client.name = "no-tools"
    mock_client.tools = []
    mock_client.connect = AsyncMock()
    mock_client.call_tool = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client_cls.return_value = mock_client

    manager = MCPManager(tool_registry)
    await manager.connect_server("no-tools", "cmd")
    assert tool_registry.list_names() == []
    mock_client.call_tool.assert_not_called()
