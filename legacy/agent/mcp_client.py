"""Pont synchrone vers le serveur MCP (une connexion stdio gardée ouverte)."""
import asyncio
import os
import sys
import threading
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "mcp_server", "server.py")


class McpBridge:
    def __init__(self, server_path: str = SERVER):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.params = StdioServerParameters(command=sys.executable, args=[server_path])
        self.tools = asyncio.run_coroutine_threadsafe(
            self._start(), self.loop).result(timeout=90)

    async def _start(self):
        self.stack = AsyncExitStack()
        read, write = await self.stack.enter_async_context(stdio_client(self.params))
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return (await self.session.list_tools()).tools

    def call(self, name: str, args: dict) -> str:
        fut = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(name, args), self.loop)
        res = fut.result(timeout=180)
        text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
        return text if not res.isError else f"Erreur MCP : {text}"


_bridge = None


def get_bridge() -> McpBridge:
    global _bridge
    if _bridge is None:
        _bridge = McpBridge()
    return _bridge
