"""Sandbox filesystem root shared by the mock tool servers.

Every path an agent supplies is resolved against a fixed root. Anything that resolves
outside the root (``..``, absolute paths, drive letters, symlinks pointing out) is
rejected. The runtime root is (re)created from a version-controlled seed directory that
contains only FAKE data.
"""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath

SEED_DIR = Path(__file__).resolve().parent / "seed"


class SandboxViolation(ValueError):
    """A path resolved outside the sandbox root or was otherwise disallowed."""


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"sandbox root does not exist: {self.root}")

    def resolve(self, user_path: str) -> Path:
        """Map an agent-supplied relative path to an absolute path inside the root."""
        if "\x00" in user_path:
            raise SandboxViolation("path contains a null byte")
        # Reject anything anchored under EITHER convention so behaviour is identical on
        # Linux and Windows hosts: "/x", "\\x", "C:\\x", "C:x", "//server/share".
        if PurePosixPath(user_path).anchor or PureWindowsPath(user_path).anchor:
            raise SandboxViolation("absolute or drive/root-anchored paths are not allowed")
        candidate = (self.root / user_path).resolve()  # follows symlinks
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise SandboxViolation("path escapes the sandbox root")
        return candidate

    def relative(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel or "."


def _restore_owner_access(path: Path) -> None:
    """Give the owner rwx on every directory and rw on every file below ``path`` (parents
    before children, so directories made untraversable can be entered again)."""
    if path.is_symlink():
        return  # never follow links out of the sandbox
    is_dir = path.is_dir()
    try:
        path.chmod(path.lstat().st_mode | (0o700 if is_dir else 0o600))
    except OSError:
        return
    if is_dir:
        for child in path.iterdir():
            _restore_owner_access(child)


def prepare_sandbox(runtime_root: Path, seed_dir: Path = SEED_DIR, *, reset: bool = False) -> Path:
    """Seed ``runtime_root`` from ``seed_dir`` if it is missing or EMPTY (a freshly created
    Docker volume is an existing empty directory). With ``reset``, wipe its contents first
    (the directory itself is kept: it may be a mount point)."""
    runtime_root = runtime_root.resolve()
    if reset and runtime_root.exists():
        # A session may have changed permissions (e.g. an attack chmod-ing .ssh to 0644
        # makes it untraversable); restore owner access first so the reset never fails.
        _restore_owner_access(runtime_root)
        for child in runtime_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    if not runtime_root.exists() or not any(runtime_root.iterdir()):
        runtime_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, runtime_root, dirs_exist_ok=True)
    return runtime_root
