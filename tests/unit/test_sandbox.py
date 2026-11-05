import os
import sys
from pathlib import Path

import pytest

from tools_servers.sandbox import SEED_DIR, Sandbox, SandboxViolation, prepare_sandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    return Sandbox(prepare_sandbox(tmp_path / "fs"))


def test_seed_contains_expected_fake_files() -> None:
    for rel in [".env", ".ssh/id_rsa", "data/customers.csv", "docs/readme.md"]:
        assert (SEED_DIR / rel).is_file(), rel
    assert "FAKE" in (SEED_DIR / ".env").read_text(encoding="utf-8")
    assert "FAKE" in (SEED_DIR / ".ssh/id_rsa").read_text(encoding="utf-8")


def test_prepare_copies_and_reset_restores(tmp_path: Path) -> None:
    root = prepare_sandbox(tmp_path / "fs")
    (root / "docs/readme.md").write_text("tampered", encoding="utf-8")
    prepare_sandbox(root)  # no reset: keeps changes
    assert (root / "docs/readme.md").read_text(encoding="utf-8") == "tampered"
    prepare_sandbox(root, reset=True)
    assert (root / "docs/readme.md").read_text(encoding="utf-8") != "tampered"


@pytest.mark.parametrize(
    "path",
    ["docs/readme.md", "./docs/readme.md", "docs/../docs/readme.md", ".env", "."],
)
def test_allowed_paths(sandbox: Sandbox, path: str) -> None:
    assert sandbox.resolve(path).is_relative_to(sandbox.root)


@pytest.mark.parametrize(
    "path",
    [
        "../outside.txt",
        "docs/../../outside.txt",
        "/etc/passwd",
        "\\Windows\\win.ini",
        "C:\\Windows\\win.ini",
        "C:relative.txt",
        "//server/share/file",
        "docs/readme.md\x00.png",
    ],
)
def test_rejected_paths(sandbox: Sandbox, path: str) -> None:
    with pytest.raises(SandboxViolation):
        sandbox.resolve(path)


def test_symlink_escape_rejected(sandbox: Sandbox, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = sandbox.root / "docs" / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this platform/user")
    with pytest.raises(SandboxViolation):
        sandbox.resolve("docs/link.txt")


def test_missing_root_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Sandbox(tmp_path / "nope")


def test_prepare_seeds_an_existing_empty_dir_like_a_fresh_volume(tmp_path: Path) -> None:
    mount = tmp_path / "volume"
    mount.mkdir()  # Docker creates the (empty) mount point before the server starts
    root = prepare_sandbox(mount)
    assert (root / "docs" / "readme.md").is_file() and (root / ".env").is_file()


def test_reset_keeps_the_mount_point_directory(tmp_path: Path) -> None:
    root = prepare_sandbox(tmp_path / "fs")
    (root / "extra.txt").write_text("x", encoding="utf-8")
    inode_before = root.stat().st_ino
    prepare_sandbox(root, reset=True)
    assert root.stat().st_ino == inode_before and not (root / "extra.txt").exists()
    assert (root / "docs" / "readme.md").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_reset_survives_permissions_changed_by_a_session(tmp_path: Path) -> None:
    """An attack session may chmod .ssh to 0644 (untraversable); the next reset must work."""
    root = prepare_sandbox(tmp_path / "fs")
    (root / ".ssh").chmod(0o644)
    (root / ".env").chmod(0o000)
    prepare_sandbox(root, reset=True)
    assert (root / ".ssh" / "id_rsa").read_text(encoding="utf-8")
    assert (root / ".env").read_text(encoding="utf-8")
