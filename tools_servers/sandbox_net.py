"""In-process stand-in for the internal sandbox network (mockweb + exfil-sink).

Used by in-process gateways (unit tests, dataset capture): http_fetch requests to the
sandbox hosts are served by the real ASGI apps without any network."""

from __future__ import annotations

import httpx

from tools_servers.exfil_sink.app import create_app as create_sink_app
from tools_servers.mockweb.app import create_app as create_mockweb


class SandboxWeb(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.sink = create_sink_app()
        self.routes = {
            "mockweb": httpx.ASGITransport(app=create_mockweb()),
            "exfil-sink": httpx.ASGITransport(app=self.sink),
        }

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        route = self.routes.get(request.url.host)
        if route is None:
            raise httpx.ConnectError("no route to host", request=request)
        return await route.handle_async_request(request)
