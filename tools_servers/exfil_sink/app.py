"""Exfil sink: a fake "attacker server" that records everything sent to it.

Any request to any path is captured (method, path, query, selected headers, body).
Control endpoints live under ``/_sink`` so they never collide with captured paths.
Used ONLY to measure whether an attack actually got data out of the sandbox (ASR).
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

MAX_BODY_BYTES = 64 * 1024
MAX_RECORDS = 10_000
_KEPT_HEADERS = ("content-type", "user-agent", "x-session-id")


class SinkStore:
    def __init__(self, log_path: Path | None = None, max_records: int = MAX_RECORDS) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._records.append(record)
            if self._log_path is not None:
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def reset(self) -> int:
        with self._lock:
            n = len(self._records)
            self._records.clear()
            if self._log_path is not None and self._log_path.exists():
                self._log_path.unlink()
            return n


def create_app(store: SinkStore | None = None) -> FastAPI:
    store = store or SinkStore()
    app = FastAPI(title="exfil-sink", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store

    @app.get("/_sink/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/_sink/records")
    async def records() -> dict[str, Any]:
        items = store.all()
        return {"count": len(items), "records": items}

    @app.post("/_sink/reset")
    async def reset() -> dict[str, int]:
        return {"cleared": store.reset()}

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    async def capture(path: str, request: Request) -> PlainTextResponse:
        raw = await request.body()
        store.add(
            {
                "ts": datetime.now(UTC).isoformat(),
                "method": request.method,
                "path": "/" + path,
                "query": dict(request.query_params),
                "headers": {h: request.headers[h] for h in _KEPT_HEADERS if h in request.headers},
                "body": raw[:MAX_BODY_BYTES].decode("utf-8", errors="replace"),
                "body_bytes": len(raw),
                "client": request.client.host if request.client else None,
            }
        )
        # Look like a boring success so attacks proceed as they would against a real server.
        return PlainTextResponse("ok")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": type(exc).__name__}, status_code=500)

    return app


def _app_from_env() -> FastAPI:
    log = os.environ.get("SINK_LOG_PATH")
    return create_app(SinkStore(Path(log) if log else None))


app = _app_from_env()
