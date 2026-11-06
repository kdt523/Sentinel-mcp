"""Mock HTTP fetch MCP server restricted to sandbox hosts (mockweb, exfil-sink).

Network confinement is a sandbox property, not a security judgement: any URL on an allowed
host is fetched (including exfil-sink, so exfiltration attempts really "succeed" when the
gateway is off). Redirects are followed manually and every hop is re-checked.

HTML is converted to text the way typical agent fetch tools do: script/style and HTML
comments are dropped, but CSS-hidden text is kept (no CSS is evaluated). Pass ``raw=True``
to get the untouched HTML, comments included.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tools_servers.content_gen import strip_url_tail

DEFAULT_ALLOWED_HOSTS = ("mockweb", "exfil-sink")
TIMEOUT_S = 5.0
MAX_BYTES = 1_000_000
MAX_REDIRECTS = 3
MAX_URL_CHARS = 8_000
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml+xml")
_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "section", "article", "header", "footer", "blockquote", "title",
}  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        joined = "".join(self.parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in joined.split("\n"))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


class HostPolicy:
    """Allowlist of ``host`` (any port) or ``host:port`` entries."""

    def __init__(self, entries: tuple[str, ...]) -> None:
        self.entries = {e.strip().lower() for e in entries if e.strip()}

    def check(self, url: str) -> None:
        if len(url) > MAX_URL_CHARS:
            raise ToolError("URL too long")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"}:
            raise ToolError(f"unsupported scheme: {parts.scheme or '(none)'}")
        if parts.username is not None or parts.password is not None:
            raise ToolError("credentials in URL are not allowed")
        host = (parts.hostname or "").lower()
        if not host:
            raise ToolError("URL has no host")
        try:
            port = parts.port
        except ValueError as exc:
            raise ToolError("invalid port") from exc
        if host in self.entries or (port is not None and f"{host}:{port}" in self.entries):
            return
        raise ToolError(f"host not reachable from the sandbox: {host}")


async def _get_capped(client: httpx.AsyncClient, url: str) -> tuple[httpx.Response, bytes]:
    async with client.stream("GET", url) as resp:
        buf = bytearray()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > MAX_BYTES:
                raise ToolError(f"response too large (> {MAX_BYTES} bytes)")
        return resp, bytes(buf)


def build_server(
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> MCPServer:
    policy = HostPolicy(allowed_hosts)
    server = MCPServer(name="http_fetch", instructions="Fetch web pages and HTTP APIs.")

    @server.tool(
        description="Fetch a URL with HTTP GET and return its text. Set raw=true for raw HTML."
    )
    async def fetch(url: str, raw: bool = False) -> str:
        current = strip_url_tail(url)
        async with httpx.AsyncClient(
            transport=transport,
            timeout=TIMEOUT_S,
            follow_redirects=False,
            headers={"User-Agent": "sentinel-sandbox-fetch/0.1"},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                policy.check(current)
                try:
                    resp, body = await _get_capped(client, current)
                except httpx.TimeoutException as exc:
                    raise ToolError("request timed out") from exc
                except httpx.HTTPError as exc:
                    raise ToolError(f"request failed: {type(exc).__name__}") from exc
                if resp.is_redirect and "location" in resp.headers:
                    current = urljoin(current, resp.headers["location"])
                    continue
                break
            else:
                raise ToolError(f"too many redirects (> {MAX_REDIRECTS})")

        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        if not ctype.startswith(_TEXT_TYPES):
            content = f"[non-text content: {ctype or 'unknown'}, {len(body)} bytes]"
        else:
            text = body.decode(resp.encoding or "utf-8", errors="replace")
            is_html = ctype in {"text/html", "application/xhtml+xml"}
            content = html_to_text(text) if is_html and not raw else text
        return f"URL: {current}\nStatus: {resp.status_code}\nContent-Type: {ctype}\n\n{content}"

    return server


def main() -> None:
    env = os.environ.get("SANDBOX_FETCH_ALLOWED_HOSTS")
    hosts = tuple(env.split(",")) if env else DEFAULT_ALLOWED_HOSTS
    build_server(hosts).run("stdio")


if __name__ == "__main__":
    main()
