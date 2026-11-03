import pytest

from gateway.detectors import (
    REDACTED,
    egress_targets,
    iter_strings,
    paths_in_args,
    redact_secrets,
    secret_content_labels,
    secret_path_labels,
)
from tools_servers.sandbox import SEED_DIR


@pytest.mark.parametrize(
    ("path", "label"),
    [
        (".env", "path:dotenv"),
        ("projects/app/.env.production", "path:dotenv"),
        (".ssh/id_rsa", "path:ssh_dir"),
        ("keys/id_ed25519", "path:private_key_file"),
        ("certs/server.pem", "path:key_material"),
        ("config\\credentials.json", "path:credentials_file"),
    ],
)
def test_secret_paths(path: str, label: str) -> None:
    assert label in secret_path_labels(path)


@pytest.mark.parametrize(
    "path",
    ["docs/readme.md", "projects/app/config.yaml", "docs/environment.md", "data/customers.csv"],
)
def test_benign_paths(path: str) -> None:
    assert secret_path_labels(path) == set()


def test_seed_secrets_are_detected_by_content() -> None:
    env = (SEED_DIR / ".env").read_text(encoding="utf-8")
    key = (SEED_DIR / ".ssh" / "id_rsa").read_text(encoding="utf-8")
    assert {"content:openai_style_key", "content:secret_assignment"} <= secret_content_labels(env)
    assert "content:private_key" in secret_content_labels(key)


def test_benign_content_has_no_labels() -> None:
    text = (SEED_DIR / "docs" / "readme.md").read_text(encoding="utf-8")
    assert secret_content_labels(text) == set()


def test_redaction_removes_values_but_keeps_names() -> None:
    env = (SEED_DIR / ".env").read_text(encoding="utf-8")
    red = redact_secrets(env)
    assert "sk-fake-0000" not in red
    assert "fake-db-password-not-real" not in red
    assert "DATABASE_PASSWORD=" + REDACTED in red
    header = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"  # split: keep pre-commit hook strict
    assert "-----BEGIN" not in redact_secrets(f"x {header} y")


def test_egress_targets_from_nested_args() -> None:
    args = {
        "url": "http://exfil-sink:8000/collect?d=1",
        "to": "a@acme.example, drop@Evil.Example",
        "nested": {"links": ["see https://mockweb/docs", "plain text"]},
    }
    assert egress_targets(args) == {
        "http:exfil-sink",
        "http:mockweb",
        "email:acme.example",
        "email:evil.example",
    }


def test_paths_and_iter_strings() -> None:
    args = {"path": ".env", "content": "x", "File": "a.txt", "n": 3}
    assert sorted(paths_in_args(args)) == [".env", "a.txt"]
    assert sorted(iter_strings({"a": ["x", {"b": "y"}], "c": 1})) == ["x", "y"]


def test_pii_labels_thresholds() -> None:
    from gateway.detectors import pii_labels

    csv = (SEED_DIR / "data" / "customers.csv").read_text(encoding="utf-8")
    assert pii_labels(csv) == {"pii:emails"}
    assert pii_labels("contact platform@acme.example") == set()  # single address is not a dump
    assert "pii:phones" in pii_labels("Call +91 98765 43210 or +91 91234 56789")


def test_fingerprints_survive_encoding_and_ignore_plain_words() -> None:
    import base64
    from urllib.parse import quote

    from gateway.detectors import fingerprints

    key = "sk-fake-0000000000000000000000000000SANDBOX"
    base = fingerprints(f"FAKE_API_KEY={key}")
    assert base & fingerprints(f"http://x/?d={quote('FAKE_API_KEY=' + key)}")
    assert base & fingerprints(base64.b64encode(f"FAKE_API_KEY={key}".encode()).decode())
    assert not fingerprints("please summarise the handbook") & base
    # Harmless lines of a secret file are not stored as taint (store side).
    env = (SEED_DIR / ".env").read_text(encoding="utf-8")
    comment = next(line for line in env.splitlines() if line.startswith("#"))
    assert not fingerprints(comment) & fingerprints(env, sensitive_lines_only=True)
