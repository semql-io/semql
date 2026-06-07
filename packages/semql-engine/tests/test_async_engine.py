"""Tests for AsyncEngine + AsyncAdapter (P2).

Mirrors test_engine.py but exercises the async surface: per-fragment
adapters run in parallel via ``asyncio.gather``, the merge step still
happens in DuckDB, and ``iter_run`` streams merge-cursor rows in
chunks via ``fetchmany``.

We use ``asyncio.run`` directly (no pytest-asyncio dependency) — the
test surface is small enough that the boilerplate is worth less than
the extra dep.
"""

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
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql_engine import (
    AdapterResult,
    AsyncDuckDBAdapter,
    AsyncEngine,
    DuckDBAdapter,
    EngineError,
    to_async_adapter,
)


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test-only adapters: stand in for non-DuckDB async backends.
# ---------------------------------------------------------------------------


class _AsyncDialectTranslatingAdapter:
    """Rewrites Postgres ``%(name)s`` and BigQuery ``@name`` placeholders
    to DuckDB ``$name`` so a single in-memory DuckDB can stand in for
    multiple backends in tests — async variant of the helper in
    test_engine.py."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._inner = AsyncDuckDBAdapter(connection)

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        sql = re.sub(r"%\((\w+)\)s", r"$\1", sql)
        sql = re.sub(r"@(\w+)", r"$\1", sql)
        return await self._inner.execute(sql, params)


# ---------------------------------------------------------------------------
# Cube fixtures — shared shape with the sync engine tests.
# ---------------------------------------------------------------------------


def _orders_cube(backend: Backend = Backend.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="order_count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )


def _customers_cube(backend: Backend = Backend.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        backend=backend,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), "
        "(2, 10, 'paid', 200.0), "
        "(3, 11, 'paid', 50.0), "
        "(4, 11, 'pending', 25.0), "
        "(5, 12, 'paid', 300.0)"
    )
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE customers (id INTEGER, region TEXT, tier TEXT)")
    con.execute(
        "INSERT INTO customers VALUES (10, 'EU', 'gold'), (11, 'US', 'silver'), (12, 'EU', 'gold')"
    )
    return con


def _catalog(*cubes: Cube) -> dict[str, Cube]:
    return {c.name: c for c in cubes}


# ---------------------------------------------------------------------------
# AsyncEngine: two-fragment federation + merge
# ---------------------------------------------------------------------------


def test_async_engine_runs_two_fragment_plan(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

    result = _run(engine.run(plan))
    assert result.columns == ["region", "revenue"]
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"EU": 600.0, "US": 75.0}


def test_async_engine_single_fragment_plan(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.status"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, AsyncDuckDBAdapter(pg_con))
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 650.0, "pending": 25.0}


def test_async_engine_refuses_unregistered_backend(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, AsyncDuckDBAdapter(pg_con))
    with pytest.raises(EngineError, match="No adapter registered"):
        _run(engine.run(plan))


# ---------------------------------------------------------------------------
# Parallelism: fragments run concurrently via asyncio.gather
# ---------------------------------------------------------------------------


class _SleepyAdapter:
    """Adapter that sleeps for ``delay`` seconds before delegating, so
    we can observe whether two fragments actually overlap on the event
    loop."""

    def __init__(self, inner: AsyncDuckDBAdapter, delay: float) -> None:
        self._inner = inner
        self._delay = delay

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        await asyncio.sleep(self._delay)
        return await self._inner.execute(sql, params)


def test_async_engine_runs_fragments_in_parallel(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Two fragments each sleeping 100ms should finish in ~100ms total,
    not ~200ms — proving ``asyncio.gather`` overlaps them."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    pg_inner = _AsyncDialectTranslatingAdapter(pg_con)
    bq_inner = _AsyncDialectTranslatingAdapter(bq_con)
    engine.register(Backend.POSTGRES, _SleepyAdapter(pg_inner, delay=0.1))  # type: ignore[arg-type]
    engine.register(Backend.BIGQUERY, _SleepyAdapter(bq_inner, delay=0.1))  # type: ignore[arg-type]

    t0 = time.perf_counter()
    _run(engine.run(plan))
    elapsed = time.perf_counter() - t0
    # Serial would be ~0.2s; parallel ~0.1s. Add slack for CI jitter
    # but stay well under the serial cost.
    assert elapsed < 0.18, f"fragments didn't overlap: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# iter_run: chunked streaming over the merge cursor
# ---------------------------------------------------------------------------


def test_iter_run_yields_chunks_summing_to_full_result(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=1):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert {r[0] for r in rows} == {"EU", "US"}


def test_iter_run_chunk_size_one_yields_one_row_per_chunk(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

    async def collect() -> list[int]:
        sizes: list[int] = []
        async for chunk in engine.iter_run(plan, chunk_rows=1):
            sizes.append(len(chunk))
        return sizes

    sizes = _run(collect())
    # Each chunk should have exactly one row.
    assert all(s == 1 for s in sizes)
    # Total chunks == row count == 2 regions.
    assert sum(sizes) == 2


def test_iter_run_rejects_non_positive_chunk_rows(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Backend.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def call() -> None:
        async for _ in engine.iter_run(plan, chunk_rows=0):
            pass

    with pytest.raises(EngineError, match="chunk_rows must be positive"):
        _run(call())


# ---------------------------------------------------------------------------
# to_async_adapter: bridge a sync adapter into the async engine
# ---------------------------------------------------------------------------


def test_to_async_adapter_bridges_sync_into_async_engine(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = AsyncEngine()
    # DuckDBAdapter is sync; wrap it for AsyncEngine.
    engine.register(Backend.POSTGRES, to_async_adapter(DuckDBAdapter(pg_con)))
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 650.0, "pending": 25.0}
