"""Tests for AsyncDBAPIAdapter (PG / SF target)."""

# mypy: disable-error-code=var-annotated

# DB-API is a sync protocol - every driver (psycopg2, psycopg,
# snowflake-connector-python, pymysql, ...) implements the same PEP-249
# shape. AsyncDBAPIAdapter wraps a sync DB-API connection and
# dispatches each execute to a worker thread via asyncio.to_thread
# so async-first user code can register a connection it already owns
# without first wrapping it in to_async_adapter.
#
# We use an in-memory DuckDB as a stand-in for the real driver - the
# shape under test is the async dispatch + dict-param handoff, not the
# wire protocol. A real PG/SF connection would behave identically given
# a connection that the same SQL string works against.

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Mapping
from typing import Any

import duckdb
import pytest
from semql import (
    Backend,
    Cube,
    Dimension,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql_engine import AsyncDBAPIAdapter, AsyncEngine


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _DBAPICursor:
    def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
        self._raw = raw
        self._description: list[tuple[str, ...]] | None = None
        self._result: Any = None

    def execute(self, sql: str, params: Any = None) -> _DBAPICursor:
        if params:
            self._result = self._raw.execute(sql, dict(params))
        else:
            self._result = self._raw.execute(sql)
        desc = self._result.description
        self._description = list(desc) if desc else None
        return self

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._result.fetchall())

    def close(self) -> None:
        pass

    @property
    def description(self) -> list[tuple[str, ...]] | None:
        return self._description


class _DBAPIConn:
    """Minimal PEP-249 wrapper around an in-memory DuckDB.

    Production users pass a real ``psycopg2.connection`` /
    ``snowflake.connector.connection`` / ``pymysql.connections.Connection``
    - all conform to the same duck-typed contract this class exposes."""

    def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
        self._raw = raw

    def cursor(self) -> _DBAPICursor:
        return _DBAPICursor(self._raw)


class _SlowCursor(_DBAPICursor):
    def execute(self, sql: str, params: Any = None) -> _SlowCursor:
        time.sleep(0.05)
        result = super().execute(sql, params)
        return result  # type: ignore[return-value]


class _SlowConn(_DBAPIConn):
    """A DB-API conn whose ``execute`` blocks for 50ms; long enough
    for a sibling coroutine to make progress if the loop is free."""

    def cursor(self) -> _SlowCursor:
        return _SlowCursor(self._raw)


@pytest.fixture()
def duckdb_as_pg() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES (1, 'paid', 100.0), (2, 'paid', 200.0), (3, 'pending', 25.0)"
    )
    return con


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


class _AsyncDialectTranslatingAdapter:
    """Test-only: rewrite ``%(name)s`` to ``$name`` so DuckDB can run PG SQL."""

    def __init__(self, conn: _DBAPIConn) -> None:
        self._inner = AsyncDBAPIAdapter(conn)

    async def execute(self, sql: str, params: Mapping[str, Any]) -> Any:
        sql = re.sub(r"%\((\w+)\)s", r"$\1", sql)
        return await self._inner.execute(sql, params)


def test_async_dbapi_adapter_returns_columns_and_rows() -> None:
    raw = duckdb.connect(":memory:")
    raw.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    raw.execute("INSERT INTO t VALUES (1, 'x'), (2, 'y')")
    conn = _DBAPIConn(raw)
    adapter = AsyncDBAPIAdapter(conn)

    result = _run(adapter.execute("SELECT a, b FROM t ORDER BY a", {}))
    assert result.columns == ["a", "b"]
    assert list(result.rows) == [(1, "x"), (2, "y")]


def test_async_dbapi_adapter_passes_params_as_dict() -> None:
    raw = duckdb.connect(":memory:")
    raw.execute("CREATE TABLE t (status TEXT, amount DOUBLE)")
    raw.execute("INSERT INTO t VALUES ('paid', 100.0), ('pending', 25.0)")
    conn = _DBAPIConn(raw)
    adapter = AsyncDBAPIAdapter(conn)

    result = _run(
        adapter.execute(
            "SELECT status, amount FROM t WHERE status = $status",
            {"status": "paid"},
        )
    )
    assert list(result.rows) == [("paid", 100.0)]


def test_async_dbapi_adapter_releases_event_loop() -> None:
    """Each ``execute`` should release the loop (via ``to_thread``). A
    pure sync block would block the loop and prevent any sibling
    coroutine from making progress - verify the wrapping is actually
    async by interleaving a sibling coroutine that ticks a counter
    during a 50ms dispatch."""
    raw = duckdb.connect(":memory:")
    raw.execute("CREATE TABLE t (a INTEGER)")
    raw.execute("INSERT INTO t VALUES (1)")
    conn = _SlowConn(raw)
    adapter = AsyncDBAPIAdapter(conn)
    progress: list[int] = []

    async def sibling() -> int:
        for i in range(10):
            await asyncio.sleep(0.01)
            progress.append(i)
        return len(progress)

    async def run() -> tuple[Any, int]:
        result_coro = adapter.execute("SELECT a FROM t", {})
        sibling_coro = sibling()
        result, ticks = await asyncio.gather(result_coro, sibling_coro)
        return result, ticks

    result, ticks = _run(run())
    assert list(result.rows) == [(1,)]
    # If ``adapter.execute`` blocked the loop, ticks would be 0.
    assert ticks >= 4, f"event loop was blocked; only {ticks} sibling ticks"


def test_async_dbapi_adapter_runs_through_async_engine(
    duckdb_as_pg: duckdb.DuckDBPyConnection,
) -> None:
    """End-to-end: ``AsyncDBAPIAdapter`` plugs into ``AsyncEngine`` and
    runs a federated plan against an in-memory DuckDB standing in for
    a real Postgres DB-API connection."""
    catalog = {c.name: c for c in [_orders_cube()]}
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(
        Backend.POSTGRES,
        _AsyncDialectTranslatingAdapter(_DBAPIConn(duckdb_as_pg)),
    )
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 300.0, "pending": 25.0}


def test_async_dbapi_adapter_empty_params_executes_without_dict() -> None:
    """No params to driver ``execute(sql)`` signature, not ``execute(sql, {})``.

    Some drivers reject the empty-dict call (e.g. ``TypeError``); we
    detect the empty case and call the no-params form."""
    raw = duckdb.connect(":memory:")
    raw.execute("CREATE TABLE t (a INTEGER)")
    raw.execute("INSERT INTO t VALUES (1), (2)")
    conn = _DBAPIConn(raw)
    adapter = AsyncDBAPIAdapter(conn)
    result = _run(adapter.execute("SELECT COUNT(*) AS n FROM t", {}))
    assert list(result.rows) == [(2,)]
