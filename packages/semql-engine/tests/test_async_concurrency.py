"""Concurrent AsyncEngine calls must not corrupt each other.

The merge step materialises fragments into fixed ``frag_<i>`` tables. A
single shared DuckDB connection across in-flight ``run()`` / ``iter_run()``
coroutines (the normal FastAPI fan-out) would race on those tables —
one call's reset/load clobbering another's mid-stream. The engine gives
each call its own isolated connection; these tests pin that invariant by
running many overlapping calls and checking every result is correct.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from typing import Any

import duckdb
import pytest
from semql import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.federate import FederatedPlan
from semql.spec import Filter
from semql_engine import AdapterResult, AsyncEngine


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _SlowAdapter:
    """DuckDB-backed adapter that sleeps before returning, forcing the
    two concurrent calls' fragment fetches + merges to interleave."""

    def __init__(self, con: duckdb.DuckDBPyConnection, delay: float) -> None:
        self._con = con
        self._delay = delay

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        await asyncio.sleep(self._delay)
        cur = self._con.execute(sql, dict(params))
        return AdapterResult(columns=[d[0] for d in cur.description], rows=cur.fetchall())


def _orders(dialect: Dialect = Dialect.DUCKDB) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(dialect: Dialect = Dialect.DUCKDB) -> Cube:
    return Cube(
        name="customers",
        dialect=dialect,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


@pytest.fixture()
def con() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect(":memory:")
    c.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DOUBLE)")
    c.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 100.0), (2, 10, 200.0), (3, 11, 50.0), (4, 12, 300.0), (5, 12, 25.0)"
    )
    c.execute("CREATE TABLE customers (id INTEGER, region TEXT)")
    c.execute("INSERT INTO customers VALUES (10, 'EU'), (11, 'US'), (12, 'EU')")
    return c


def _plan_for_customer(customer_id: int) -> FederatedPlan:
    catalog = {c.name: c for c in (_orders(), _customers())}
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        filters=[Filter(dimension="orders.customer_id", op="eq", values=[customer_id])],
    )
    # Force a real two-fragment DuckDB merge (not the single-fragment fast path).
    return compile_federated_query(q, catalog)


# Revenue per customer (single region each): 10→EU 300, 11→US 50, 12→EU 325.
_EXPECTED = {10: [("EU", 300.0)], 11: [("US", 50.0)], 12: [("EU", 325.0)]}


def test_concurrent_run_calls_are_isolated(con: duckdb.DuckDBPyConnection) -> None:
    engine = AsyncEngine()
    engine.register(Dialect.DUCKDB, _SlowAdapter(con, delay=0.02))

    async def drive() -> list[tuple[int, list[tuple[Any, ...]]]]:
        cids = [10, 11, 12] * 4  # 12 overlapping calls
        plans = [_plan_for_customer(cid) for cid in cids]
        results = await asyncio.gather(*(engine.run(p) for p in plans))
        return [(cid, [tuple(r) for r in res.rows]) for cid, res in zip(cids, results, strict=True)]

    for cid, rows in _run(drive()):
        assert sorted(rows) == sorted(_EXPECTED[cid]), f"customer {cid} got {rows}"


def test_concurrent_iter_run_streams_are_isolated(con: duckdb.DuckDBPyConnection) -> None:
    """iter_run holds a merge cursor across await points; concurrent
    streams must not see each other's fragment tables."""
    engine = AsyncEngine()
    engine.register(Dialect.DUCKDB, _SlowAdapter(con, delay=0.02))

    async def collect(cid: int) -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(_plan_for_customer(cid), chunk_rows=1):
            out.extend(chunk)
        return out

    async def drive() -> list[tuple[int, list[tuple[Any, ...]]]]:
        cids = [10, 11, 12] * 4
        results = await asyncio.gather(*(collect(cid) for cid in cids))
        return list(zip(cids, results, strict=True))

    for cid, rows in _run(drive()):
        assert sorted(rows) == sorted(_EXPECTED[cid]), f"customer {cid} got {rows}"


def test_async_fragment_success_keeps_plan_order_despite_reverse_completion(
    con: duckdb.DuckDBPyConnection,
) -> None:
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        {cube.name: cube for cube in (_orders(Dialect.POSTGRES), _customers(Dialect.BIGQUERY))},
    )
    assert len(plan.fragments) == 2
    customers_completed = asyncio.Event()
    completion: list[str] = []

    class ReverseCompletionAdapter:
        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            source = "orders" if "orders" in sql else "customers"
            if source == "orders":
                await asyncio.wait_for(customers_completed.wait(), timeout=1)
            result = await _SlowAdapter(con, delay=0).execute(sql, params)
            completion.append(source)
            if source == "customers":
                customers_completed.set()
            return result

    adapter = ReverseCompletionAdapter()
    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, adapter)
    engine.register(Dialect.BIGQUERY, adapter)

    result = _run(engine.run(plan))

    assert completion == ["customers", "orders"]
    assert {region: total for region, total in result.rows} == {"EU": 625.0, "US": 50.0}


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "nonfast-stream"])
def test_fragment_failure_cancels_and_drains_sibling(streaming: bool) -> None:
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        {cube.name: cube for cube in (_orders(Dialect.POSTGRES), _customers(Dialect.BIGQUERY))},
    )
    assert len(plan.fragments) == 2
    sibling_started = asyncio.Event()
    sibling_finalized = asyncio.Event()
    failure = RuntimeError("fragment failed")

    class CoordinatedAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            self.calls += 1
            if self.calls == 1:
                await asyncio.wait_for(sibling_started.wait(), timeout=1)
                raise failure
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_finalized.set()
            raise AssertionError("blocking sibling unexpectedly completed")

    adapter = CoordinatedAdapter()
    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, adapter)
    engine.register(Dialect.BIGQUERY, adapter)

    async def drive() -> None:
        if streaming:
            with pytest.raises(RuntimeError) as raised:
                async for _ in engine.iter_run(plan):
                    pass
        else:
            with pytest.raises(RuntimeError) as raised:
                await engine.run(plan)
        assert raised.value is failure
        assert sibling_finalized.is_set()

    _run(drive())


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "nonfast-stream"])
def test_materialization_failure_closes_all_acquired_row_iterators(
    streaming: bool,
) -> None:
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        {cube.name: cube for cube in (_orders(Dialect.POSTGRES), _customers(Dialect.BIGQUERY))},
    )
    assert len(plan.fragments) == 2

    class CloseableRows:
        def __init__(self, fails: bool) -> None:
            self.fails = fails
            self.closed = False

        def __iter__(self) -> CloseableRows:
            return self

        def __next__(self) -> tuple[Any, ...]:
            if self.fails:
                raise RuntimeError("row iteration failed")
            raise StopIteration

        def close(self) -> None:
            self.closed = True

    row_iters = [CloseableRows(fails=index == 0) for index in range(2)]

    class Adapter:
        def __init__(self, columns: list[str], rows: CloseableRows) -> None:
            self.columns = columns
            self.rows = rows

        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            return AdapterResult(columns=self.columns, rows=self.rows)

    engine = AsyncEngine()
    for fragment, rows in zip(plan.fragments, row_iters, strict=True):
        engine.register(fragment.dialect, Adapter(list(fragment.columns), rows))

    async def drive() -> None:
        with pytest.raises(RuntimeError, match="row iteration failed"):
            if streaming:
                stream = engine.iter_run(plan)
                await stream.__anext__()
            else:
                await engine.run(plan)

    _run(drive())
    assert [rows.closed for rows in row_iters] == [True, True]


@pytest.mark.parametrize("streaming", [False, True], ids=["run", "nonfast-stream"])
def test_caller_cancellation_drains_fragment_siblings(streaming: bool) -> None:
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        {cube.name: cube for cube in (_orders(Dialect.POSTGRES), _customers(Dialect.BIGQUERY))},
    )
    assert len(plan.fragments) == 2
    both_started = asyncio.Event()
    finalized: list[int] = []

    class BlockingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            self.calls += 1
            if self.calls == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finalized.append(self.calls)
            raise AssertionError("cancelled adapter unexpectedly completed")

    engine = AsyncEngine()
    adapter = BlockingAdapter()
    engine.register(Dialect.POSTGRES, adapter)
    engine.register(Dialect.BIGQUERY, adapter)

    async def drive() -> None:
        async def execute() -> Any:
            if streaming:
                async for chunk in engine.iter_run(plan):
                    return chunk
            return await engine.run(plan)

        task = asyncio.create_task(execute())
        await asyncio.wait_for(both_started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(finalized) == 2

    _run(drive())
