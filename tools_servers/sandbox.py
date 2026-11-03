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
        if PurePosixPath(user_path).is_absolute() or PureWindowsPath(user_path).is_absolute():
            raise SandboxViolation("absolute paths are not allowed")
        if PureWindowsPath(user_path).drive:
            raise SandboxViolation("drive-qualified paths are not allowed")
        candidate = (self.root / user_path).resolve()  # follows symlinks
        if candidate != self.root and not candidate.is_relative_to(self.root):
            raise SandboxViolation("path escapes the sandbox root")
        return candidate

    def relative(self, path: Path) -> str:
        rel = path.relative_to(self.root).as_posix()
        return rel or "."


def prepare_sandbox(runtime_root: Path, seed_dir: Path = SEED_DIR, *, reset: bool = False) -> Path:
    """Create ``runtime_root`` from ``seed_dir``. With ``reset`` wipe it first."""
    runtime_root = runtime_root.resolve()
    if reset and runtime_root.exists():
        shutil.rmtree(runtime_root)
    if not runtime_root.exists():
        runtime_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed_dir, runtime_root)
    return runtime_root
