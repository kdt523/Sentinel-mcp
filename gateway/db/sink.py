"""Synchronous database sinks for the decision log, called from a worker thread.

Why synchronous: on Windows, psycopg's async mode needs a SelectorEventLoop, but the
gateway's stdio tool servers need the ProactorEventLoop (only it can spawn subprocesses).
Running blocking writes in a worker thread (``asyncio.to_thread``) works on every OS and
event loop, and writes are already batched off the request path.

- ``PostgresPoolSink``: psycopg (sync) + ``psycopg_pool.ConnectionPool``; statements are
  built with SQLAlchemy Core and compiled for PostgreSQL.
- ``SqlAlchemySink``: synchronous SQLAlchemy engine (SQLite for tests / local dev).

Both build their statements from the same functions, so behaviour cannot diverge.
"""

from __future__ import annotations

from typing import Any, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from sqlalchemy import Insert, bindparam, create_engine, insert, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine, make_url

from gateway.db.models import ApprovalRow, DecisionRow, SessionRow
from gateway.models import Action

_PG_DIALECT = postgresql.psycopg.dialect()  # type: ignore[no-untyped-call]
_JSON_COLUMNS = ("args_redacted",)


class DecisionSink(Protocol):
    def write_decisions(self, rows: list[dict[str, Any]]) -> None:
        """Insert decision rows and roll up their sessions in ONE transaction."""
        ...

    def upsert_approval(self, row: dict[str, Any]) -> None: ...

    def ping(self) -> None: ...

    def close(self) -> None: ...


# ----------------------------------------------------------------------------- statements


def session_rollups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = sessions.setdefault(r["session_id"], {
            "session_id": r["session_id"], "last_seen_at": r["event_ts"], "event_count": 0,
            "terminated": False, "terminated_reason": None,
        })  # fmt: skip
        s["last_seen_at"] = max(s["last_seen_at"], r["event_ts"])
        s["event_count"] += 1
        if r["action"] == Action.TERMINATE_SESSION.value and r["enforced"]:
            s["terminated"], s["terminated_reason"] = True, r["reason"]
    return list(sessions.values())


def session_upsert(dialect: str, s: dict[str, Any]) -> Insert:
    ins = pg_insert if dialect == "postgresql" else sqlite_insert
    stmt = ins(SessionRow).values(**s)
    return stmt.on_conflict_do_update(
        index_elements=[SessionRow.session_id],
        set_={
            "last_seen_at": stmt.excluded.last_seen_at,
            "event_count": SessionRow.event_count + stmt.excluded.event_count,
            "terminated": SessionRow.terminated | stmt.excluded.terminated,
            "terminated_reason": stmt.excluded.terminated_reason
            if s["terminated"]
            else SessionRow.terminated_reason,
        },
    )


def approval_upsert(dialect: str, row: dict[str, Any]) -> Insert:
    ins = pg_insert if dialect == "postgresql" else sqlite_insert
    stmt = ins(ApprovalRow).values(**row)
    return stmt.on_conflict_do_update(
        index_elements=[ApprovalRow.approval_id],
        set_={k: stmt.excluded[k] for k in ("status", "resolved_at", "resolved_by", "note")},
    )


# ----------------------------------------------------------------------------- sinks


class SqlAlchemySink:
    """Synchronous SQLAlchemy engine (used for SQLite)."""

    def __init__(self, url: str) -> None:
        self.engine: Engine = create_engine(url)
        self.dialect = self.engine.dialect.name

    def write_decisions(self, rows: list[dict[str, Any]]) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(DecisionRow), rows)
            for s in session_rollups(rows):
                conn.execute(session_upsert(self.dialect, s))

    def upsert_approval(self, row: dict[str, Any]) -> None:
        with self.engine.begin() as conn:
            conn.execute(approval_upsert(self.dialect, row))

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def close(self) -> None:
        self.engine.dispose()


def _compile(stmt: Any) -> tuple[str, dict[str, Any]]:
    compiled = stmt.compile(dialect=_PG_DIALECT)
    return str(compiled), dict(compiled.params)


class PostgresPoolSink:
    """psycopg (sync) with a psycopg_pool connection pool."""

    dialect = "postgresql"

    def __init__(self, conninfo: str, *, max_size: int = 5, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s
        self.pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=max_size,
            timeout=timeout_s,  # max wait for a pooled connection
            kwargs={"connect_timeout": max(1, int(timeout_s))},
            open=False,
        )
        # Open without waiting: an unreachable database must not block gateway startup.
        self.pool.open(wait=False)

    @staticmethod
    def _adapt(row: dict[str, Any]) -> dict[str, Any]:
        return {k: (Jsonb(v) if k in _JSON_COLUMNS and v is not None else v)
                for k, v in row.items()}  # fmt: skip

    def write_decisions(self, rows: list[dict[str, Any]]) -> None:
        columns = list(rows[0])
        sql, _ = _compile(insert(DecisionRow).values({c: bindparam(c) for c in columns}))
        with self.pool.connection(timeout=self.timeout_s) as conn, conn.cursor() as cur:
            cur.executemany(sql, [self._adapt(r) for r in rows])
            for s in session_rollups(rows):
                cur.execute(*_compile(session_upsert(self.dialect, s)))
        # leaving pool.connection() commits (or rolls back on error)

    def upsert_approval(self, row: dict[str, Any]) -> None:
        with self.pool.connection(timeout=self.timeout_s) as conn:
            conn.execute(*_compile(approval_upsert(self.dialect, row)))

    def ping(self) -> None:
        with self.pool.connection(timeout=self.timeout_s) as conn:
            conn.execute("SELECT 1")

    def close(self) -> None:
        self.pool.close()


def create_sink(database_url: str, *, max_size: int = 5, timeout_s: float = 5.0) -> DecisionSink:
    url = make_url(database_url)
    if url.get_backend_name() == "postgresql":
        conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
        return PostgresPoolSink(conninfo, max_size=max_size, timeout_s=timeout_s)
    if url.drivername.startswith("sqlite"):
        return SqlAlchemySink(url.set(drivername="sqlite").render_as_string(hide_password=False))
    return SqlAlchemySink(database_url)
