"""Client transports for upstream MCP tool servers.

``uds_transport`` speaks MCP streamable HTTP over a Unix domain socket. It lets the gateway
reach tool servers that run with ``network_mode: none`` (e.g. the shell sandbox): the only
channel is a socket file on a dedicated volume.

This module must stay self-contained (imports: stdlib, httpx2, mcp only) — the integration
probe executes this exact file inside the sandbox image to verify it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx2
from mcp.client._transport import TransportStreams
from mcp.client.streamable_http import streamable_http_client

MCP_PATH = "/mcp"


@asynccontextmanager
async def uds_transport(
    socket_path: str, *, host: str, timeout_s: float = 30.0
) -> AsyncGenerator[TransportStreams, None]:
    """Transport usable as ``mcp.Client(uds_transport(path, host=...))``.

    ``host`` is sent as the HTTP Host header and must be on the server's DNS-rebinding
    allowlist (servers keep that protection ON; e.g. the shell server allows ``shell.uds``).
    Owns its HTTP client: the SDK does not close a caller-provided client, so we do.
    """
    base_url = f"http://{host}"
    async with (
        httpx2.AsyncClient(
            transport=httpx2.AsyncHTTPTransport(uds=socket_path),
            base_url=base_url,
            timeout=httpx2.Timeout(timeout_s),
        ) as http_client,
        streamable_http_client(f"{base_url}{MCP_PATH}", http_client=http_client) as streams,
    ):
        yield streams
