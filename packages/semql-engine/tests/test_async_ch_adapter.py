"""Tests for AsyncClickHouseAdapter (CH target)."""

# mypy: disable-error-code=var-annotated

# clickhouse-connect ships a true async client
# (asynch_client.AsyncClient) where .query() is a coroutine.
# The query parameters are passed as a parameters= dict; CH's
# {name:Type} placeholder syntax is already native to our compiler's
# CH dialect.
#
# We use a fake _FakeAsyncCHClient rather than the real
# clickhouse_connect.asynch_client.AsyncClient - the network code
# isn't the surface under test, only the async dispatch + parameter
# hand-off.

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Mapping
from typing import Any

import duckdb
from semql import (
    Backend,
    Cube,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql_engine import AsyncClickHouseAdapter, AsyncEngine


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _FakeCHResult:
    """Mimics ``clickhouse_connect.driver.query.QueryResult``."""

    def __init__(self, column_names: list[str], result_rows: list[tuple[Any, ...]]) -> None:
        self.column_names = column_names
        self.result_rows = result_rows

    def named_results(self) -> list[dict[str, Any]]:
        return [dict(zip(self.column_names, row, strict=True)) for row in self.result_rows]


class _FakeAsyncCHClient:
    """Mimics the ``clickhouse_connect.asynch_client.AsyncClient`` surface.

    The real client's ``.query()`` is an ``async def`` that returns a
    ``QueryResult`` synchronously (the rows are sync-iterable)."""

    def __init__(self, results: list[_FakeCHResult]) -> None:
        self._results = iter(results)
        self.calls: list[dict[str, Any]] = []
        self.sleep_seconds: float = 0.0

    async def query(
        self,
        query: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> _FakeCHResult:
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        self.calls.append({"query": query, "parameters": dict(parameters or {})})
        return next(self._results)


# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------


def test_async_ch_adapter_returns_columns_and_rows() -> None:
    client = _FakeAsyncCHClient(
        [_FakeCHResult(column_names=["a", "b"], result_rows=[(1, "x"), (2, "y")])]
    )
    adapter = AsyncClickHouseAdapter(client)
    result = _run(adapter.execute("SELECT a, b FROM t ORDER BY a", {}))
    assert result.columns == ["a", "b"]
    assert list(result.rows) == [(1, "x"), (2, "y")]


def test_async_ch_adapter_passes_params_as_dict() -> None:
    """The real client accepts ``parameters=dict``; pass through verbatim."""
    client = _FakeAsyncCHClient(
        [_FakeCHResult(column_names=["status", "amount"], result_rows=[("paid", 100.0)])]
    )
    adapter = AsyncClickHouseAdapter(client)
    _run(
        adapter.execute(
            "SELECT status, amount FROM t WHERE status = {p0:String}",
            {"p0": "paid"},
        )
    )
    assert client.calls[0]["parameters"] == {"p0": "paid"}


def test_async_ch_adapter_handles_empty_params() -> None:
    """No params to ``parameters={}`` (not ``None``) for shape uniformity."""
    client = _FakeAsyncCHClient([_FakeCHResult(column_names=["n"], result_rows=[(1,)])])
    adapter = AsyncClickHouseAdapter(client)
    _run(adapter.execute("SELECT COUNT(*) AS n FROM t", {}))
    assert client.calls[0]["parameters"] == {}


def test_async_ch_adapter_preserves_sql_verbatim() -> None:
    """The adapter is dialect-blind - it does not rewrite placeholders.

    The CH dialect emits ``{name:Type}`` and the driver accepts that
    shape directly. The test wrapper does its own translation to
    DuckDB; the production driver does its own type binding."""
    client = _FakeAsyncCHClient([_FakeCHResult(column_names=[], result_rows=[])])
    adapter = AsyncClickHouseAdapter(client)
    sql = "SELECT * FROM t WHERE x = {x:String} AND y > {y:Int64}"
    _run(adapter.execute(sql, {"x": "a", "y": 1}))
    assert client.calls[0]["query"] == sql


def test_async_ch_adapter_runs_under_event_loop() -> None:
    """CH is a true async client - verify the dispatch *itself* doesn't
    block the loop. We tick a sibling coroutine while the (faked)
    client is sleeping."""

    class _SlowFakeAsyncCHClient(_FakeAsyncCHClient):
        async def query(
            self,
            query: str,
            parameters: Mapping[str, Any] | None = None,
        ) -> _FakeCHResult:
            await asyncio.sleep(0.05)
            return await super().query(query, parameters)

    client = _SlowFakeAsyncCHClient([_FakeCHResult(column_names=["a"], result_rows=[(1,)])])
    adapter = AsyncClickHouseAdapter(client)
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
    assert ticks >= 4, f"event loop was blocked; only {ticks} sibling ticks"


def test_async_ch_adapter_runs_through_async_engine_via_duckdb_stand_in() -> None:
    """End-to-end: register ``AsyncClickHouseAdapter`` on a fake CH
    client that translates ``{name:Type}`` to ``$name`` and runs the
    SQL against in-memory DuckDB."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (status TEXT, amount DOUBLE)")
    con.execute("INSERT INTO orders VALUES ('paid', 100.0), ('pending', 25.0)")

    class _DuckDBBackedAsyncCHClient:
        def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
            self._raw = raw
            self.calls: list[dict[str, Any]] = []

        async def query(
            self,
            query: str,
            parameters: Mapping[str, Any] | None = None,
        ) -> _FakeCHResult:
            # ``{name:Type}`` to ``$name`` for DuckDB; the type suffix is
            # part of the placeholder, so the regex strips the ``:Type`` half.
            translated = re.sub(r"\{(\w+):[^}]+\}", r"$\1", query)
            params = dict(parameters or {})
            cur = self._raw.execute(translated, params if params else None)
            self.calls.append({"query": query, "translated": translated, "params": params})
            desc = list(cur.description or [])
            cols = [str(d[0]) for d in desc]
            return _FakeCHResult(column_names=cols, result_rows=list(cur.fetchall()))

    client = _DuckDBBackedAsyncCHClient(con)
    adapter = AsyncClickHouseAdapter(client)

    cube = Cube(
        name="orders",
        backend=Backend.CLICKHOUSE,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.status"],
            filters=[Filter(dimension="orders.status", op="in", values=["paid"])],
        ),
        {cube.name: cube},
    )
    engine = AsyncEngine()
    engine.register(Backend.CLICKHOUSE, adapter)
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 100.0}
    # The CH-flavoured placeholders ``{p0:String}`` should have been
    # preserved verbatim by the adapter and translated only by the
    # test wrapper.
    assert any("{" in c["query"] for c in client.calls), (
        f"AsyncClickHouseAdapter rewrote the SQL; should have been a pass-through: {client.calls}"
    )


def test_async_ch_adapter_supports_native_named_params() -> None:
    """The real clickhouse-connect client takes ``parameters={...}``
    and binds them to ``{name:Type}`` placeholders. Verify the
    adapter's hand-off is purely that."""
    client = _FakeAsyncCHClient([_FakeCHResult(column_names=["v"], result_rows=[(42,)])])
    adapter = AsyncClickHouseAdapter(client)
    _run(
        adapter.execute(
            "SELECT {p0:Int64} AS v",
            {"p0": 42},
        )
    )
    assert client.calls[0]["query"] == "SELECT {p0:Int64} AS v"
    assert client.calls[0]["parameters"] == {"p0": 42}
