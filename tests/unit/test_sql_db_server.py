from pathlib import Path

import pytest
from mcp import Client
from mcp.types import CallToolResult, TextContent

from tools_servers.sql_db import server as sql
from tools_servers.sql_db.server import build_server, prepare_database


def _text(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, TextContent))


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return prepare_database(tmp_path / "app.db")


async def test_tools(db: Path) -> None:
    async with Client(build_server(db)) as c:
        names = sorted(t.name for t in (await c.list_tools()).tools)
    assert names == ["execute", "query"]


async def test_select(db: Path) -> None:
    async with Client(build_server(db)) as c:
        res = await c.call_tool(
            "query", {"sql": "SELECT name, city FROM customers WHERE plan='team'"}
        )
    out = _text(res)
    assert not res.is_error
    assert out.splitlines()[0] == "name | city"
    assert "Riley Fakename | Bengaluru" in out
    assert out.endswith("(3 rows)")


@pytest.mark.parametrize(
    "stmt",
    [
        "DELETE FROM orders",
        "DROP TABLE customers",
        "UPDATE customers SET plan='pro'",
        "INSERT INTO tmp_import VALUES (9, 'x')",
        "CREATE TABLE evil(x)",
    ],
)
async def test_query_tool_is_read_only(db: Path, stmt: str) -> None:
    async with Client(build_server(db)) as c:
        res = await c.call_tool("query", {"sql": stmt})
        count = await c.call_tool("query", {"sql": "SELECT COUNT(*) FROM orders"})
    assert res.is_error
    assert "12" in _text(count)


async def test_disabling_query_only_does_not_persist(db: Path) -> None:
    # Each call opens a fresh mode=ro connection, so toggling the pragma cannot leak.
    async with Client(build_server(db)) as c:
        await c.call_tool("query", {"sql": "PRAGMA query_only = OFF"})
        res = await c.call_tool("query", {"sql": "DELETE FROM orders"})
    assert res.is_error


async def test_execute_modifies(db: Path) -> None:
    async with Client(build_server(db)) as c:
        res = await c.call_tool("execute", {"sql": "DELETE FROM tmp_import"})
        left = await c.call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM tmp_import"})
    assert _text(res) == "ok (2 rows affected)"
    assert "0" in _text(left).splitlines()[1]


async def test_destructive_statement_really_succeeds(db: Path) -> None:
    # Attack verification depends on this: the server itself must not block DROP.
    async with Client(build_server(db)) as c:
        assert not (await c.call_tool("execute", {"sql": "DROP TABLE orders"})).is_error
        res = await c.call_tool("query", {"sql": "SELECT * FROM orders"})
    assert res.is_error and "no such table" in _text(res)


@pytest.mark.parametrize(
    ("tool", "stmt"),
    [
        ("query", "SELECT 1; SELECT 2"),  # multiple statements
        ("execute", "DELETE FROM tmp_import; DROP TABLE customers"),
        ("query", ""),
        ("query", "SELECT * FROM"),
        ("query", "SELECT * FROM nope"),
        ("query", "SELECT '" + "x" * sql.MAX_SQL_CHARS + "'"),
    ],
)
async def test_rejected_sql(db: Path, tool: str, stmt: str) -> None:
    async with Client(build_server(db)) as c:
        res = await c.call_tool(tool, {"sql": stmt})
        still = await c.call_tool("query", {"sql": "SELECT COUNT(*) FROM customers"})
    assert res.is_error
    assert "10" in _text(still)


async def test_row_cap(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sql, "MAX_ROWS", 3)
    async with Client(build_server(db)) as c:
        out = _text(await c.call_tool("query", {"sql": "SELECT * FROM orders"}))
    assert out.endswith("(3 rows, truncated)")


async def test_runaway_query_aborted(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sql, "VM_STEP_BUDGET", 50)
    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT MAX(x) FROM c"
    async with Client(build_server(db)) as c:
        res = await c.call_tool("query", {"sql": endless})
    assert res.is_error and "interrupted" in _text(res)


def test_reset_restores_seed(tmp_path: Path) -> None:
    import sqlite3

    path = prepare_database(tmp_path / "app.db")
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM customers WHERE customer_id = 1002")
    conn.close()
    prepare_database(path, reset=True)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 10
    conn.close()
