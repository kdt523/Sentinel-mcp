import json
from pathlib import Path

from fastapi.testclient import TestClient

from tools_servers.exfil_sink.app import MAX_BODY_BYTES, SinkStore, create_app


def test_captures_get_with_query() -> None:
    client = TestClient(create_app())
    r = client.get("/collect?d=FAKE_API_KEY%3Dsk-fake", headers={"X-Session-Id": "s1"})
    assert r.status_code == 200 and r.text == "ok"
    recs = client.get("/_sink/records").json()
    assert recs["count"] == 1
    rec = recs["records"][0]
    assert rec["method"] == "GET"
    assert rec["path"] == "/collect"
    assert rec["query"] == {"d": "FAKE_API_KEY=sk-fake"}
    assert rec["headers"]["x-session-id"] == "s1"


def test_captures_post_body_truncated() -> None:
    client = TestClient(create_app())
    client.post("/upload", content=b"A" * (MAX_BODY_BYTES + 10))
    rec = client.get("/_sink/records").json()["records"][0]
    assert rec["body_bytes"] == MAX_BODY_BYTES + 10
    assert len(rec["body"]) == MAX_BODY_BYTES


def test_control_endpoints_not_recorded_and_reset() -> None:
    client = TestClient(create_app())
    assert client.get("/_sink/healthz").json() == {"status": "ok"}
    client.put("/x")
    client.delete("/y")
    assert client.get("/_sink/records").json()["count"] == 2
    assert client.post("/_sink/reset").json() == {"cleared": 2}
    assert client.get("/_sink/records").json()["count"] == 0


def test_jsonl_log_persisted_and_reset(tmp_path: Path) -> None:
    log = tmp_path / "sink" / "records.jsonl"
    client = TestClient(create_app(SinkStore(log)))
    client.post("/a", content=b"one")
    client.post("/b", content=b"two")
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["body"] for x in lines] == ["one", "two"]
    client.post("/_sink/reset")
    assert not log.exists()


def test_bounded_memory() -> None:
    client = TestClient(create_app(SinkStore(max_records=3)))
    for i in range(5):
        client.get(f"/p{i}")
    paths = [r["path"] for r in client.get("/_sink/records").json()["records"]]
    assert paths == ["/p2", "/p3", "/p4"]
