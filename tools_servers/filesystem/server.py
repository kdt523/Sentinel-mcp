"""Mock filesystem MCP server confined to a sandbox root seeded with FAKE data.

Tools: ``read_file``, ``list_dir``, ``write_file``. Paths are relative to the sandbox root.
This server performs NO security judgement beyond confinement — deciding whether reading
``.env`` is acceptable is the gateway's job.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from tools_servers.sandbox import Sandbox, SandboxViolation, prepare_sandbox

DEFAULT_ROOT = Path("sandbox_runtime/fs")
MAX_READ_BYTES = 1_000_000
MAX_WRITE_BYTES = 1_000_000


def _resolve(sandbox: Sandbox, path: str) -> Path:
    try:
        return sandbox.resolve(path)
    except SandboxViolation as exc:
        raise ToolError(f"access denied: {exc}") from exc


def build_server(root: Path) -> MCPServer:
    sandbox = Sandbox(root)
    server = MCPServer(
        name="filesystem",
        instructions="Read, list and write files inside the workspace sandbox.",
    )

    @server.tool(description="Read a UTF-8 text file. `path` is relative to the workspace root.")
    def read_file(path: str) -> str:
        target = _resolve(sandbox, path)
        if not target.is_file():
            raise ToolError(f"not a file: {path}")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(f"file too large ({size} bytes > {MAX_READ_BYTES})")
        return target.read_text(encoding="utf-8", errors="replace")

    @server.tool(description="List a directory relative to the workspace root ('.' for root).")
    def list_dir(path: str = ".") -> str:
        target = _resolve(sandbox, path)
        if not target.is_dir():
            raise ToolError(f"not a directory: {path}")
        lines = []
        for entry in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}\t{entry.stat().st_size} B")
        return "\n".join(lines) if lines else "(empty)"

    @server.tool(description="Write UTF-8 text to a workspace file (created or overwritten).")
    def write_file(path: str, content: str) -> str:
        target = _resolve(sandbox, path)
        if target == sandbox.root or target.is_dir():
            raise ToolError(f"not a file path: {path}")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"content too large ({len(data)} bytes > {MAX_WRITE_BYTES})")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return f"wrote {len(data)} bytes to {sandbox.relative(target)}"

    return server


def main() -> None:
    root = Path(os.environ.get("SANDBOX_FS_ROOT", str(DEFAULT_ROOT)))
    reset = os.environ.get("SANDBOX_RESET", "0") == "1"
    build_server(prepare_sandbox(root, reset=reset)).run("stdio")


if __name__ == "__main__":
    main()
