"""Decision recorder: asynchronous, bounded, never blocks the request path.

``submit()`` is synchronous and O(1): it builds a redacted row and puts it on a bounded
queue. A background worker writes batches to the database. If the queue is full the row
is dropped and counted (``dropped``); if the database fails the batch is retried once and
then dropped and counted (``failed``). Deciding never waits for Postgres: database calls
are blocking and run in a worker thread (see gateway/db/sink.py for why).

Optional raw capture (Phase 2 dataset building) appends FULL, UNREDACTED events to
``capture_dir/events-YYYYMMDD.jsonl``. It is refused outside the sandbox (see Settings)
and announced with a warning at startup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import structlog

from gateway.config import Settings
from gateway.db.sink import DecisionSink, create_sink
from gateway.detectors import redact_deep, redact_secrets
from gateway.models import Decision, Event

log = structlog.get_logger("sentinel.recorder")
PREVIEW_CHARS = 500
RETRY_DELAY_S = 0.5
_DROP_LOG_EVERY = 100
# (kind, row, raw-capture payload or None)
_Item = tuple[Literal["decision", "approval"], dict[str, Any], dict[str, Any] | None]


def decision_row(event: Event, decision: Decision, *, enforced: bool) -> dict[str, Any]:
    """Row for ``decisions`` — redacted args, redacted output preview, hash of full output."""
    preview = None
    if event.output is not None:
        preview = redact_secrets(event.output[:PREVIEW_CHARS])
    return {
        "session_id": event.session_id,
        "event_id": event.event_id,
        "event_ts": event.ts,
        "kind": event.kind.value,
        "direction": event.direction.value,
        "server": event.server,
        "tool": event.tool,
        "qualified_tool": event.qualified_tool,
        "args_redacted": redact_deep(event.args) if event.args is not None else None,
        "output_preview": preview,
        "output_sha256": event.output_sha256,
        "output_chars": event.output_chars,
        "output_truncated": event.output_truncated,
        "upstream_ms": event.upstream_ms,
        "action": decision.action.value,
        "risk": decision.risk,
        "attack_family": decision.attack_family,
        "reason": redact_secrets(decision.reason),
        "tier": decision.tier.value,
        "rule_id": decision.rule_id,
        "cached": decision.cached,
        "escalated": decision.escalated,
        "latency_ms": decision.latency_ms,
        "model_version": decision.model_version,
        "enforced": enforced,
    }


class DecisionRecorder:
    def __init__(
        self,
        sink: DecisionSink,
        *,
        queue_size: int = 10_000,
        batch_size: int = 200,
        capture_dir: Path | None = None,
    ) -> None:
        self.sink = sink
        self._queue: asyncio.Queue[_Item] = asyncio.Queue(maxsize=queue_size)
        self._batch_size = batch_size
        self._capture_dir = capture_dir
        self._worker: asyncio.Task[None] | None = None
        self.submitted = 0
        self.written = 0
        self.dropped = 0  # queue full / unflushed at shutdown
        self.failed = 0  # database errors after retry

    @classmethod
    def from_settings(
        cls, settings: Settings, sink: DecisionSink | None = None
    ) -> DecisionRecorder:
        if sink is None:
            sink = create_sink(
                settings.database_url,
                max_size=settings.db_pool_size,
                timeout_s=settings.db_connect_timeout_s,
            )
        return cls(
            sink,
            queue_size=settings.decision_log_queue_size,
            batch_size=settings.decision_log_batch_size,
            capture_dir=settings.capture_dir if settings.capture_mode else None,
        )

    @property
    def capturing(self) -> bool:
        return self._capture_dir is not None

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def ping(self) -> None:
        """Database health check (runs in a worker thread)."""
        await asyncio.to_thread(self.sink.ping)

    # ----------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._worker is not None:
            return
        if self._capture_dir is not None:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            log.warning(
                "raw_capture_enabled",
                capture_dir=str(self._capture_dir),
                detail="FULL UNREDACTED tool arguments and outputs are written to disk "
                "(sandbox only, for dataset building).",
            )
        self._worker = asyncio.create_task(self._run(), name="decision-recorder")

    async def stop(self, timeout_s: float = 10.0, *, close_sink: bool = True) -> None:
        """Flush what is queued (bounded by ``timeout_s``), stop the worker, close the sink."""
        if self._worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout_s)
            except TimeoutError:
                lost = self._queue.qsize()
                self.dropped += lost
                log.error("decision_log_flush_timeout", unflushed=lost, timeout_s=timeout_s)
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
            log.info(
                "decision_recorder_stopped",
                written=self.written,
                dropped=self.dropped,
                failed=self.failed,
            )
        if close_sink:
            await asyncio.to_thread(self.sink.close)

    # ----------------------------------------------------------------- submit

    def submit(
        self,
        event: Event,
        decision: Decision,
        *,
        enforced: bool,
        raw_output: str | None = None,
    ) -> bool:
        """Queue one decision for logging. Never blocks; returns False if dropped."""
        self.submitted += 1
        raw = None
        if self._capture_dir is not None:
            raw = {
                "event": event.model_dump(mode="json"),
                "raw_output": raw_output if raw_output is not None else event.output,
                "decision": decision.model_dump(mode="json"),
                "enforced": enforced,
            }
        return self._put(("decision", decision_row(event, decision, enforced=enforced), raw))

    def submit_approval(self, row: dict[str, Any]) -> bool:
        """Queue an approval record (upserted by approval_id). Never blocks."""
        self.submitted += 1
        return self._put(("approval", row, None))

    def _put(self, item: _Item) -> bool:
        try:
            self._queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % _DROP_LOG_EVERY == 0:
                log.error(
                    "decision_log_queue_full",
                    dropped_total=self.dropped,
                    queue_size=self._queue.maxsize,
                )
            return False

    # ----------------------------------------------------------------- worker

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < self._batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            try:
                await self._write(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def _write(self, batch: list[_Item]) -> None:
        rows = [row for kind, row, _ in batch if kind == "decision"]
        if rows and not await self._try(self.sink.write_decisions, rows, len(rows), attempts=2):
            if len(rows) > 1:
                # Isolate bad rows (e.g. duplicate event_id) so one row cannot sink a batch.
                for row in rows:
                    if not await self._try(self.sink.write_decisions, [row], 1, attempts=1):
                        self.failed += 1
            else:
                self.failed += 1
        for kind, row, _ in batch:  # in queue order, so a resolution lands after its request
            if kind == "approval" and not await self._try(
                self.sink.upsert_approval, row, 1, attempts=2
            ):
                self.failed += 1
        raws = [raw for _, _, raw in batch if raw is not None]
        if raws and self._capture_dir is not None:
            try:
                await asyncio.to_thread(self._append_capture, raws)
            except OSError as exc:
                log.error("raw_capture_write_failed", error=str(exc)[:300])

    async def _try(self, fn: Any, arg: Any, n_rows: int, *, attempts: int) -> bool:
        """Run a blocking sink call in a worker thread; count rows written on success."""
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.to_thread(fn, arg)
            except Exception as exc:  # noqa: BLE001 - logging must never crash the gateway
                if attempt < attempts:
                    await asyncio.sleep(RETRY_DELAY_S)
                    continue
                log.error(
                    "decision_log_write_failed",
                    rows=n_rows,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                )
                return False
            self.written += n_rows
            return True
        return False

    def _append_capture(self, raws: list[dict[str, Any]]) -> None:
        assert self._capture_dir is not None
        day = datetime.now(UTC).strftime("%Y%m%d")
        path = self._capture_dir / f"events-{day}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for raw in raws:
                fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
