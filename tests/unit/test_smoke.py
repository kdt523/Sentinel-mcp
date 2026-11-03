"""Smoke test: the project installs and core modules import on CPU."""

import importlib

import pytest

PACKAGES = [
    "gateway",
    "gateway.config",
    "gateway.db",
    "tools_servers",
    "agents",
    "data",
    "baselines",
    "training",
    "fast_tier",
    "serving",
    "eval",
    "dashboard",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(name)


def test_settings_importable() -> None:
    from gateway.config import Settings, get_settings

    assert issubclass(Settings, object)
    assert callable(get_settings)
