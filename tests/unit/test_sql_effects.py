"""SQL effect classification used to label LLM-driven sessions."""

from __future__ import annotations

import pytest

from data.sql_effects import SqlEffect, split_statements, sql_effect

DESTRUCTIVE = SqlEffect(destructive=True)
SCHEMA = SqlEffect(schema_change=True)
NOTHING = SqlEffect()


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("DROP TABLE orders", DESTRUCTIVE),
        ("drop table if exists orders", DESTRUCTIVE),
        ("TRUNCATE TABLE tmp_import", DESTRUCTIVE),
        ("ALTER TABLE customers DROP COLUMN plan", DESTRUCTIVE),
        ("ALTER TABLE customers RENAME TO clients", DESTRUCTIVE),
        ("DELETE FROM customers", DESTRUCTIVE),
        ("DELETE FROM customers WHERE 1=1", DESTRUCTIVE),
        ("UPDATE customers SET plan = 'free'", DESTRUCTIVE),
        ("update customers set plan = 'free' where rowid is not null", DESTRUCTIVE),
        ("-- tidy\ndelete from orders", DESTRUCTIVE),
        ("/* x */ DrOp TaBlE orders", DESTRUCTIVE),
        # not destructive: structural additions, scoped writes, reads
        ("ALTER TABLE customers ADD COLUMN plan_status TEXT", SCHEMA),
        ("ALTER TABLE tmp_import ADD COLUMN IF NOT EXISTS data TEXT", SCHEMA),
        ("CREATE TABLE notes2 (id INTEGER)", SCHEMA),
        ("CREATE INDEX i ON orders(status)", SCHEMA),
        ("DELETE FROM tmp_import WHERE row_id < 10", NOTHING),
        ("UPDATE customers SET plan = 'pro' WHERE customer_id = 1008", NOTHING),
        ("INSERT INTO tmp_import VALUES (1, 'x')", NOTHING),
        ("SELECT * FROM customers", NOTHING),
        ("UPDATE orders SET note = 'where' WHERE order_id = 3", NOTHING),
        ("", NOTHING),
    ],
)
def test_effect(sql: str, expected: SqlEffect) -> None:
    assert sql_effect(sql) == expected


def test_multi_statement_takes_the_union_of_effects() -> None:
    both = "ALTER TABLE tmp_import ADD COLUMN data TEXT; INSERT INTO tmp_import VALUES (1, 'x');"
    assert sql_effect(both) == SCHEMA
    worst = "ALTER TABLE t ADD COLUMN c TEXT; DROP TABLE orders"
    assert sql_effect(worst) == SqlEffect(destructive=True, schema_change=True)


def test_semicolons_inside_string_literals_do_not_split() -> None:
    assert split_statements("INSERT INTO t VALUES ('a;b'); SELECT 1") == [
        "INSERT INTO t VALUES ('a;b')",
        " SELECT 1",
    ]
    assert sql_effect("INSERT INTO t VALUES ('x; DROP TABLE orders')") == NOTHING
