"""Mock SQL MCP server over SQLite with synthetic customers/orders.

Tools:
- ``query(sql)``  — read-only connection (``mode=ro`` + ``PRAGMA query_only``), row-capped.
- ``execute(sql)`` — separate read-write tool; destructive statements are NOT blocked here
  (that is the gateway's job, and attack verification needs them to really succeed).
Both accept exactly one statement and are bounded by a VM-step budget (no runaway queries).
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SEED_SQL = Path(__file__).resolve().parent / "seed.sql"
DEFAULT_DB = Path("sandbox_runtime/db/app.db")
MAX_ROWS = 200
MAX_SQL_CHARS = 10_000
VM_STEP_BUDGET = 5_000_000  # sqlite progress-handler ticks (x1000 VM instructions)


def prepare_database(db_path: Path, *, reset: bool = False) -> Path:
    db_path = db_path.resolve()
    if reset and db_path.exists():
        db_path.unlink()
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.executescript(SEED_SQL.read_text(encoding="utf-8"))
        conn.close()
    return db_path


@contextmanager
def _connect(db_path: Path, *, read_only: bool) -> Iterator[sqlite3.Connection]:
    if read_only:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
    else:
        conn = sqlite3.connect(db_path)
    budget = {"left": VM_STEP_BUDGET}

    def _tick() -> int:
        budget["left"] -= 1
        return 1 if budget["left"] <= 0 else 0  # non-zero aborts the statement

    conn.set_progress_handler(_tick, 1000)
    try:
        yield conn
    finally:
        conn.close()


def _check_sql(sql: str) -> str:
    sql = sql.strip()
    if not sql:
        raise ToolError("empty SQL")
    if len(sql) > MAX_SQL_CHARS:
        raise ToolError(f"SQL too long (> {MAX_SQL_CHARS} chars)")
    if not sqlite3.complete_statement(sql if sql.endswith(";") else sql + ";"):
        raise ToolError("incomplete SQL statement")
    return sql


def _format(cursor: sqlite3.Cursor) -> str:
    cols = [d[0] for d in cursor.description or []]
    rows = cursor.fetchmany(MAX_ROWS + 1)
    truncated = len(rows) > MAX_ROWS
    lines = [" | ".join(cols)]
    lines += [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows[:MAX_ROWS]]
    footer = f"({min(len(rows), MAX_ROWS)} rows{', truncated' if truncated else ''})"
    return "\n".join([*lines, footer])


def build_server(db_path: Path) -> MCPServer:
    db_path = db_path.resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"database does not exist: {db_path}")
    server = MCPServer(
        name="sql_db",
        instructions="Query the company SQLite database (customers, orders).",
    )

    @server.tool(description="Run ONE read-only SQL query (SELECT). Returns up to 200 rows.")
    def query(sql: str) -> str:
        stmt = _check_sql(sql)
        try:
            with _connect(db_path, read_only=True) as conn:
                return _format(conn.execute(stmt))
        except sqlite3.Error as exc:
            raise ToolError(f"query failed: {exc}") from exc

    @server.tool(description="Run ONE SQL statement that modifies data (INSERT/UPDATE/DELETE).")
    def execute(sql: str) -> str:
        stmt = _check_sql(sql)
        try:
            with _connect(db_path, read_only=False) as conn:
                cur = conn.execute(stmt)
                conn.commit()
                return f"ok ({cur.rowcount if cur.rowcount >= 0 else 0} rows affected)"
        except sqlite3.Error as exc:
            raise ToolError(f"execute failed: {exc}") from exc

    return server


def main() -> None:
    db = Path(os.environ.get("SANDBOX_DB_PATH", str(DEFAULT_DB)))
    reset = os.environ.get("SANDBOX_RESET", "0") == "1"
    build_server(prepare_database(db, reset=reset)).run("stdio")


if __name__ == "__main__":
    main()
