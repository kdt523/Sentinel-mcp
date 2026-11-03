"""Shell server inside its hardened container (needs Docker). Run: make test-int.

The container is started through `docker compose run`, so these tests exercise exactly the
hardening declared in docker-compose.yml (service `shell`).
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, TextContent

from tools_servers.sandbox import prepare_sandbox

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]


def _docker() -> str | None:
    exe = shutil.which("docker")
    if exe is None:
        win = Path("C:/Program Files/Docker/Docker/resources/bin/docker.exe")
        exe = str(win) if win.exists() else None
    if exe is None:
        return None
    ok = subprocess.run([exe, "info"], capture_output=True, timeout=30).returncode == 0
    return exe if ok else None


DOCKER = _docker()
if DOCKER is None:
    pytest.skip("docker engine not available", allow_module_level=True)


def _text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


@pytest.fixture(scope="module")
def image_built() -> None:
    assert DOCKER is not None
    subprocess.run(
        [DOCKER, "compose", "--profile", "sandbox", "build", "shell"],
        cwd=REPO,
        check=True,
        capture_output=True,
        timeout=900,
    )


@pytest.fixture
def params(image_built: None, tmp_path: Path) -> StdioServerParameters:
    assert DOCKER is not None
    sandbox = prepare_sandbox(tmp_path / "fs")
    return StdioServerParameters(
        command=DOCKER,
        args=["compose", "run", "--rm", "-T", "shell"],
        cwd=str(REPO),
        env={**os.environ, "SHELL_SANDBOX_DIR": str(sandbox)},
    )


async def run_all(params: StdioServerParameters, *cmds: str) -> list[CallToolResult]:
    """One container per test; the client lives entirely inside this task."""
    async with Client(params, read_timeout_seconds=60) as c:
        return [await c.call_tool("run", {"cmd": cmd}) for cmd in cmds]


async def run(params: StdioServerParameters, cmd: str) -> CallToolResult:
    return (await run_all(params, cmd))[0]


def py(code: str) -> str:
    """Multi-line Python as ONE command line (the server rejects raw newlines)."""
    blob = base64.b64encode(code.encode()).decode()  # base64 alphabet has no quotes
    return "python3 -c \"import base64; exec(base64.b64decode('" + blob + "'))\""


async def test_benign_commands(params: StdioServerParameters) -> None:
    cat, ls, grep = (
        _text(r)
        for r in await run_all(
            params, "cat docs/readme.md", "ls docs", "grep -n Handbook docs/readme.md"
        )
    )
    assert cat.startswith("exit_code: 0") and "Acme Notes" in cat
    assert "readme.md" in ls
    assert "Handbook" in grep


async def test_secret_read_really_succeeds(params: StdioServerParameters) -> None:
    # Confinement only; judging this is the gateway's job (needed for real ASR).
    assert "FAKE_API_KEY" in _text(await run(params, "cat .env"))


async def test_no_shell_interpretation(params: StdioServerParameters) -> None:
    out = _text(await run(params, "echo hi; cat .env"))
    assert "hi; cat .env" in out and "FAKE_API_KEY" not in out


@pytest.mark.parametrize("cmd", ["rm -rf docs", "bash -c id", "curl http://x", "/bin/ls"])
async def test_disallowed_programs(params: StdioServerParameters, cmd: str) -> None:
    assert (await run(params, cmd)).is_error


async def test_runs_as_non_root_without_capabilities(params: StdioServerParameters) -> None:
    out = _text(await run(params, 'python3 -c "import os; print(os.getuid(), os.getgid())"'))
    assert "10001 10001" in out
    caps = _text(await run(params, "grep CapEff /proc/self/status"))
    assert "0000000000000000" in caps


async def test_no_new_privileges(params: StdioServerParameters) -> None:
    assert "NoNewPrivs:\t1" in _text(await run(params, "grep NoNewPrivs /proc/self/status"))


async def test_no_network(params: StdioServerParameters) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=2); print('CONNECTED')\n"
        "except OSError as e:\n"
        "    print('NO_NET', type(e).__name__)"
    )
    out = _text(await run(params, py(code)))
    assert "NO_NET" in out and "CONNECTED" not in out
    ifaces = _text(await run(params, "cat /proc/net/dev"))
    assert "eth0" not in ifaces


async def test_read_only_rootfs_and_tmpfs_noexec(params: StdioServerParameters) -> None:
    out = _text(await run(params, "python3 -c \"open('/app/x','w')\""))
    assert "Read-only file system" in out or "Permission denied" in out
    mounts = _text(await run(params, "grep ' /tmp ' /proc/mounts"))
    assert "tmpfs" in mounts and "noexec" in mounts


async def test_timeout_kills_runaway(params: StdioServerParameters) -> None:
    out = _text(await run(params, 'python3 -c "while True: pass"'))
    assert out.startswith("exit_code: timeout")


async def test_pids_limit_contains_fork_bomb(params: StdioServerParameters) -> None:
    code = (
        "import os\n"
        "n=0\n"
        "try:\n"
        "    for _ in range(500):\n"
        "        if os.fork()==0:\n"
        "            import time; time.sleep(3); os._exit(0)\n"
        "        n+=1\n"
        "except OSError as e:\n"
        "    print('LIMITED', n)"
    )
    bomb, after = await run_all(params, py(code), "echo still-alive")
    assert "LIMITED" in _text(bomb)
    assert "exit_code: 0" in _text(after)  # same container keeps serving


async def test_memory_limit(params: StdioServerParameters) -> None:
    out = _text(
        await run(params, "python3 -c \"b = bytearray(600 * 1024 * 1024); print('ALLOC')\"")
    )
    assert "ALLOC" not in out
