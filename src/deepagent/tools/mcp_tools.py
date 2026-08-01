"""MCP tools — connect/disconnect/list MCP (Model Context Protocol) servers."""

from __future__ import annotations

from deepagent.tools.registry import tool, ToolRegistry
from deepagent.tools.protocol import SafetyLevel


def create_mcp_tools(reg: ToolRegistry, manager) -> None:
    """Register MCP-related tools into *reg*. *manager* is an MCPManager instance."""

    @tool(
        reg,
        name="connect_mcp",
        description="Connect to an MCP (Model Context Protocol) server via stdio transport. The server's tools are discovered and registered dynamically. Example: connect_mcp(name='docs', command='python', args=['-m', 'my_mcp_server'])",
        safety_level=SafetyLevel.WRITE,
    )
    async def connect_mcp(name: str, command: str, args: list | None = None) -> dict:
        msg = await manager.connect_server(name, command, args)
        if msg.lower().startswith("error") or msg.lower().startswith("failed"):
            return {"success": False, "content": "", "error": msg}
        return {"success": True, "content": msg}

    @tool(
        reg,
        name="list_mcp_servers",
        description="List all connected MCP servers.",
        safety_level=SafetyLevel.READONLY,
    )
    async def list_mcp_servers() -> dict:
        servers = manager.list_servers()
        if not servers:
            return {"success": True, "content": "No MCP servers connected."}
        lines = [f"  {name} (connected)" for name in servers]
        return {"success": True, "content": "\n".join(lines)}

    @tool(
        reg,
        name="disconnect_mcp",
        description="Disconnect from an MCP server and unregister its tools.",
        safety_level=SafetyLevel.WRITE,
    )
    async def disconnect_mcp(name: str) -> dict:
        msg = await manager.disconnect_server(name)
        lowered = msg.lower()
        if lowered.startswith("error") or "not connected" in lowered:
            return {"success": False, "content": "", "error": msg}
        return {"success": True, "content": msg}
