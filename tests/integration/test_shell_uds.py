"""Shell server over a private Unix socket, verified with the real mcp v2 client (needs Docker).

The probe container runs the EXACT source of gateway/transports.py (piped via stdin) as a
different container that shares only the socket volume — i.e. what the gateway will do.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tools_servers.sandbox import prepare_sandbox

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]
SOCK_VOLUME = "sentinel-mcp_shell-sock"


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


def _dc(
    *args: str, env: dict[str, str] | None = None, **kw: Any
) -> subprocess.CompletedProcess[str]:
    assert DOCKER is not None
    return subprocess.run(
        [DOCKER, "compose", *args],
        cwd=REPO,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        **kw,
    )


# Executed INSIDE the shell container through the `run` tool (python3 is allowlisted).
ISOLATION_CODE = r"""
import json, os, socket
connected = []
for host, port in [("1.1.1.1", 80), ("8.8.8.8", 53), ("10.0.0.1", 80), ("172.17.0.1", 80),
                   ("127.0.0.1", 22), ("127.0.0.1", 80), ("127.0.0.1", 5432),
                   ("127.0.0.1", 6379), ("127.0.0.1", 8000), ("host.docker.internal", 80)]:
    try:
        socket.create_connection((host, port), timeout=1).close()
        connected.append(f"tcp:{host}:{port}")
    except OSError:
        pass
try:
    s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM); s6.settimeout(1)
    s6.connect(("::1", 80)); connected.append("tcp6:[::1]:80")
except OSError:
    pass
try:
    socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("1.1.1.1", 53))
    connected.append("udp:1.1.1.1:53")
except OSError:
    pass
own = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
own.connect("/run/sentinel-shell/shell.sock"); own.close()
unix_paths = sorted({ln.split()[-1] for ln in open("/proc/net/unix").read().splitlines()[1:]
                     if len(ln.split()) >= 8})
print("ISO:" + json.dumps({
    "connected": connected,
    "ifaces": sorted(os.listdir("/sys/class/net")),
    "sock_dir": sorted(os.listdir("/run/sentinel-shell")),
    "own_socket_ok": True,
    "docker_sock": [p for p in ("/var/run/docker.sock", "/run/docker.sock") if os.path.exists(p)],
    "unix_paths": unix_paths,
}))
"""

PROBE_BODY = r"""

# ---------------- probe (appended after gateway/transports.py) ----------------
import asyncio, json, os, stat
from mcp import Client
from mcp.types import TextContent

SOCK = "/run/sentinel-shell/shell.sock"
ISO_B64 = "__ISO_B64__"

def text(res):
    return "".join(c.text for c in res.content if isinstance(c, TextContent))

r = {"uid": os.getuid()}


def leaves_of(e):
    out, todo = [], [e]
    while todo:
        x = todo.pop()
        subs = getattr(x, "exceptions", None)
        todo.extend(subs) if subs else out.append(f"{type(x).__name__}: {x}")
    return " | ".join(out)[:1500]


async def probe():
    try:
        st = os.lstat(SOCK)
        r.update(is_sock=stat.S_ISSOCK(st.st_mode), mode=oct(stat.S_IMODE(st.st_mode)),
                 owner=[st.st_uid, st.st_gid], volume=sorted(os.listdir(os.path.dirname(SOCK))))
    except OSError as e:
        r["stat_error"] = type(e).__name__
    try:
        import httpx2
        raw = httpx2.Client(transport=httpx2.HTTPTransport(uds=SOCK))
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0"}}}
        r["wrong_host_status"] = raw.post("http://localhost/mcp", json=init, headers={
            "Accept": "application/json, text/event-stream"}).status_code
    except Exception as e:  # noqa: BLE001 - httpx2.ConnectError is not an OSError
        r["raw_error"] = type(e).__name__
    try:
        async with Client(uds_transport(SOCK, host="shell.uds", timeout_s=20)) as c:
            r["tools"] = [t.name for t in (await c.list_tools()).tools]
            r["cat"] = text(await c.call_tool("run", {"cmd": "cat docs/readme.md"}))
            iso_cmd = "python3 -c \"import base64; exec(base64.b64decode('" + ISO_B64 + "'))\""
            r["isolation"] = text(await c.call_tool("run", {"cmd": iso_cmd}))
        r["client_ok"] = True
    except BaseException as e:  # noqa: BLE001 - report everything to the test
        r["client_error"] = leaves_of(e)


try:
    asyncio.run(probe())
except BaseException as e:  # noqa: BLE001 - errors re-raised during SDK teardown
    r.setdefault("client_error", leaves_of(e))
finally:
    print("RESULT:" + json.dumps(r), flush=True)
"""


def _probe_script() -> str:
    transports_src = (REPO / "gateway" / "transports.py").read_text(encoding="utf-8")
    iso_b64 = base64.b64encode(ISOLATION_CODE.encode()).decode()
    return transports_src + PROBE_BODY.replace("__ISO_B64__", iso_b64)


def _run_probe(*extra: str) -> dict[str, Any]:
    proc = _dc(
        "--profile",
        "test",
        "run",
        "--rm",
        "-T",
        *extra,
        "shell-probe",
        input=_probe_script(),
        timeout=180,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT:")), None)
    assert line is not None, f"probe produced no result\nstdout={proc.stdout}\nstderr={proc.stderr}"
    result: dict[str, Any] = json.loads(line.removeprefix("RESULT:"))
    return result


@pytest.fixture(scope="module")
def shell_service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    assert DOCKER is not None
    sandbox = prepare_sandbox(tmp_path_factory.mktemp("uds") / "fs")
    env = {**os.environ, "SHELL_SANDBOX_DIR": str(sandbox)}
    _dc("--profile", "sandbox", "rm", "-sf", "shell", env=env, timeout=120)
    subprocess.run([DOCKER, "volume", "rm", "-f", SOCK_VOLUME], capture_output=True, timeout=60)
    _dc("--profile", "sandbox", "build", "shell", env=env, timeout=900, check=True)
    up = _dc("--profile", "sandbox", "up", "-d", "--wait", "shell", env=env, timeout=180)
    assert up.returncode == 0, up.stderr
    try:
        yield
    finally:
        _dc("--profile", "sandbox", "rm", "-sf", "shell", env=env, timeout=120)
        subprocess.run([DOCKER, "volume", "rm", "-f", SOCK_VOLUME], capture_output=True, timeout=60)


@pytest.fixture(scope="module")
def probe(shell_service: None) -> dict[str, Any]:
    return _run_probe()


def test_socket_is_private(probe: dict[str, Any]) -> None:
    assert probe["is_sock"] is True
    assert probe["mode"] == "0o600"
    assert probe["owner"] == [10001, 10001]


def test_volume_contains_only_the_socket(probe: dict[str, Any]) -> None:
    assert probe["volume"] == ["shell.sock"]


def test_mcp_v2_client_over_uds(probe: dict[str, Any]) -> None:
    assert probe.get("client_ok") is True, probe.get("client_error")
    assert probe["tools"] == ["run"]
    assert probe["cat"].startswith("exit_code: 0") and "Acme Notes" in probe["cat"]


def test_python3_in_shell_cannot_connect_anywhere_but_its_socket(probe: dict[str, Any]) -> None:
    line = next(ln for ln in probe["isolation"].splitlines() if ln.startswith("ISO:"))
    iso = json.loads(line.removeprefix("ISO:"))
    assert iso["connected"] == [], f"unexpected connections: {iso['connected']}"
    assert iso["ifaces"] == ["lo"]
    assert iso["sock_dir"] == ["shell.sock"]
    assert iso["own_socket_ok"] is True
    assert iso["docker_sock"] == []
    assert set(iso["unix_paths"]) <= {"/run/sentinel-shell/shell.sock"}


def test_dns_rebinding_protection_still_on(probe: dict[str, Any]) -> None:
    assert probe["wrong_host_status"] == 421


def test_other_uid_is_refused(shell_service: None) -> None:
    other = _run_probe("--user", "10002:10002")
    assert other["uid"] == 10002
    assert "client_ok" not in other
    assert other.get("stat_error") == "PermissionError"  # cannot even see the socket
    assert other.get("raw_error") == "ConnectError"
    assert "Permission denied" in other.get("client_error", "")
