"""Mock shell MCP server. Executes commands ONLY inside its hardened Docker container.

Safety model (defence in depth):
1. Refuses to start unless it detects it is inside the container
   (``SANDBOX_IN_CONTAINER=1`` AND ``/.dockerenv`` present).
2. No shell interpreter: argv is parsed with ``shlex`` and run with ``shell=False``, so
   ``;``, ``|``, ``>`` and ``$(...)`` are plain literal arguments.
3. ``argv[0]`` must be a bare program name on a configurable allowlist.
4. Minimal environment, cwd = sandbox root, wall-clock timeout kills the process group,
   output capped.
5. The container itself (see Dockerfile + compose) provides the real isolation: no network,
   read-only rootfs + tmpfs, non-root, all capabilities dropped, no-new-privileges and
   CPU / memory / pids limits.

Whether a given allowed command is *acceptable* (e.g. ``cat .env``, or any ``python3`` call)
is the gateway's decision, not this server's.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_ALLOWLIST = ("ls", "cat", "grep", "wc", "head", "echo", "date", "python3")
DEFAULT_ROOT = Path("/sandbox")
TIMEOUT_S = 5.0
MAX_CMD_CHARS = 4_000
MAX_ARGS = 64
MAX_OUTPUT_CHARS = 64_000
SAFE_PATH = "/usr/local/bin:/usr/bin:/bin"


class CommandRejected(ValueError):
    pass


@dataclass(frozen=True)
class RunResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool

    def render(self) -> str:
        head = "exit_code: timeout" if self.timed_out else f"exit_code: {self.exit_code}"
        return f"{head}\n--- stdout ---\n{self.stdout}\n--- stderr ---\n{self.stderr}"


def in_container() -> bool:
    return os.environ.get("SANDBOX_IN_CONTAINER") == "1" and Path("/.dockerenv").exists()


def parse_allowlist(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset(DEFAULT_ALLOWLIST)
    names = {n.strip() for n in raw.split(",") if n.strip()}
    for n in names:
        if "/" in n or "\\" in n:
            raise ValueError(f"allowlist entries must be bare program names: {n!r}")
    return frozenset(names)


def parse_command(cmd: str, allowlist: frozenset[str]) -> list[str]:
    """Validate and split ``cmd`` into argv. Pure function; never executes anything."""
    if not cmd.strip():
        raise CommandRejected("empty command")
    if len(cmd) > MAX_CMD_CHARS:
        raise CommandRejected(f"command too long (> {MAX_CMD_CHARS} chars)")
    if "\x00" in cmd or "\n" in cmd or "\r" in cmd:
        raise CommandRejected("control characters (null/newline) are not allowed")
    try:
        argv = shlex.split(cmd, posix=True)
    except ValueError as exc:
        raise CommandRejected(f"cannot parse command: {exc}") from exc
    if not argv:
        raise CommandRejected("empty command")
    if len(argv) > MAX_ARGS:
        raise CommandRejected(f"too many arguments (> {MAX_ARGS})")
    prog = argv[0]
    if "/" in prog or "\\" in prog:
        raise CommandRejected("program must be a bare name, not a path")
    if prog not in allowlist:
        raise CommandRejected(f"program not allowed: {prog}")
    return argv


def _cap(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[truncated {len(text) - MAX_OUTPUT_CHARS} chars]"


def execute(argv: list[str], cwd: Path, timeout_s: float = TIMEOUT_S) -> RunResult:
    """Run argv without a shell in a new process group; kill the whole group on timeout."""
    env = {"PATH": SAFE_PATH, "HOME": str(cwd), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}
    proc = subprocess.Popen(  # noqa: S603 - argv validated by parse_command, shell=False
        argv,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)  # type: ignore[attr-defined,unused-ignore]
        out, err = proc.communicate()
        timed_out = True
    return RunResult(
        exit_code=proc.returncode,
        stdout=_cap(out.decode("utf-8", errors="replace")),
        stderr=_cap(err.decode("utf-8", errors="replace")),
        timed_out=timed_out,
    )


def build_server(root: Path, allowlist: frozenset[str]) -> MCPServer:
    if not in_container():
        raise RuntimeError("shell server must run inside its sandbox container")
    root = root.resolve()
    server = MCPServer(name="shell", instructions="Run simple commands in the workspace.")
    allowed = ", ".join(sorted(allowlist))

    @server.tool(description=f"Run one command in the workspace (no shell). Allowed: {allowed}.")
    def run(cmd: str) -> str:
        try:
            argv = parse_command(cmd, allowlist)
        except CommandRejected as exc:
            raise ToolError(str(exc)) from exc
        try:
            return execute(argv, root).render()
        except OSError as exc:
            raise ToolError(f"failed to start: {type(exc).__name__}") from exc

    return server


def main() -> None:
    if not in_container():
        raise SystemExit("refusing to start: shell server only runs inside its sandbox container")
    root = Path(os.environ.get("SANDBOX_FS_ROOT", str(DEFAULT_ROOT)))
    allowlist = parse_allowlist(os.environ.get("SANDBOX_SHELL_ALLOWLIST"))
    build_server(root, allowlist).run("stdio")


if __name__ == "__main__":
    main()
