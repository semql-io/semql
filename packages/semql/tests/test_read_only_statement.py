"""Tests for ``semql.safe.is_read_only_statement`` — belt-and-braces guard.

The compiler emits a single ``SELECT`` by construction (the sqlglot AST
path doesn't expose a way to interleave DDL/DML). ``is_read_only_statement``
runs as a post-hoc check the caller can apply to ``CompiledQuery.sql`` for
defence-in-depth: if a future refactor ever lets through a non-SELECT
statement, this catches it before the SQL hits the database.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.safe import is_read_only_statement

# ---------------------------------------------------------------------------
# Happy path — every SemanticQuery the compiler emits is a safe SELECT.
# ---------------------------------------------------------------------------


def test_simple_select_is_safe() -> None:
    assert is_read_only_statement("SELECT 1") is True


def test_select_with_where_group_order_limit_is_safe() -> None:
    assert (
        is_read_only_statement(
            "SELECT region, SUM(amount) AS revenue FROM orders "
            "WHERE deleted_at IS NULL GROUP BY region "
            "ORDER BY revenue DESC LIMIT 10"
        )
        is True
    )


def test_select_with_union_is_safe() -> None:
    """UNION of two SELECTs is still a SELECT statement."""
    assert is_read_only_statement("SELECT 1 UNION ALL SELECT 2") is True


def test_select_with_cte_is_safe() -> None:
    assert is_read_only_statement("WITH t AS (SELECT 1 AS x) SELECT * FROM t") is True


def test_compiled_sql_round_trip_is_safe() -> None:
    """Every shape ``compile_query`` emits is a safe SELECT — pin it
    for a representative query so a future regression in the compiler
    surfaces here, not in production logs."""
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    out = Catalog([orders]).compile(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    )
    assert is_read_only_statement(out.sql) is True


# ---------------------------------------------------------------------------
# Rejected shapes — DDL, DML, multi-statement.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET x = 1 WHERE y = 2",
        "DELETE FROM t WHERE y = 2",
        "CREATE TABLE t (x INT)",
        "DROP TABLE t",
        "ALTER TABLE t ADD COLUMN x INT",
        "TRUNCATE TABLE t",
        "GRANT SELECT ON t TO u",
    ],
)
def test_ddl_and_dml_rejected(sql: str) -> None:
    assert is_read_only_statement(sql) is False


def test_multi_statement_rejected() -> None:
    """Even when both statements are SELECTs, a multi-statement payload
    is suspicious — the compiler emits exactly one."""
    assert is_read_only_statement("SELECT 1; SELECT 2") is False


def test_statement_with_trailing_semicolon_is_safe() -> None:
    """A single SELECT terminated by a semicolon is conventional — the
    multi-statement guard should not false-positive here."""
    assert is_read_only_statement("SELECT 1;") is True


def test_empty_string_rejected() -> None:
    assert is_read_only_statement("") is False


def test_whitespace_only_rejected() -> None:
    assert is_read_only_statement("   \n  ") is False


def test_malformed_sql_rejected() -> None:
    assert is_read_only_statement("SELECT FROM WHERE") is False


def test_dialect_kwarg_threads_to_parser() -> None:
    """ClickHouse-only syntax (typed placeholders) must parse cleanly
    under the CH dialect even though they aren't valid PG."""
    assert (
        is_read_only_statement("SELECT count(*) FROM t WHERE x = {p0:String}", dialect="clickhouse")
        is True
    )
