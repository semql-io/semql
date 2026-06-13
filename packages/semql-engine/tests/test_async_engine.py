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
from dataclasses import replace
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
from semql_engine import (
    AdapterResult,
    AsyncDuckDBAdapter,
    AsyncEngine,
    AsyncMergeEngine,
    DuckDBAdapter,
    EngineError,
    to_async_adapter,
    to_async_merge_engine,
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


def _orders_cube(backend: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="order_count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
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


def _customers_cube(backend: Dialect = Dialect.BIGQUERY) -> Cube:
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
    engine.register(Dialect.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

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
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))
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
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))
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
    engine.register(Dialect.POSTGRES, _SleepyAdapter(pg_inner, delay=0.1))  # type: ignore[arg-type]
    engine.register(Dialect.BIGQUERY, _SleepyAdapter(bq_inner, delay=0.1))  # type: ignore[arg-type]

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
    engine.register(Dialect.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

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
    engine.register(Dialect.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

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
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

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
    engine.register(Dialect.POSTGRES, to_async_adapter(DuckDBAdapter(pg_con)))
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 650.0, "pending": 25.0}


def test_to_async_merge_engine_bridges_sync_merge_engine() -> None:
    from semql import MergeSpec

    class RecordingMergeEngine:
        def merge(self, fragment_results: list[AdapterResult], spec: MergeSpec) -> AdapterResult:
            return AdapterResult(columns=["count"], rows=[(len(fragment_results),)])

    wrapped = to_async_merge_engine(RecordingMergeEngine())
    assert isinstance(wrapped, AsyncMergeEngine)


# ---------------------------------------------------------------------------
# iter_run single-fragment fast path
# ---------------------------------------------------------------------------
#
# For one-fragment plans, iter_run skips the DuckDB CREATE TABLE +
# INSERT roundtrip and runs the merge (column rename + identity-SUM +
# optional ORDER / LIMIT + AVG decomposition) in Python directly
# against the adapter rows. The fast path is opt-in by shape: the
# parser bails on anything it doesn't immediately recognise (HAVING,
# unusual expressions) and falls through to the DuckDB merge.
#
# ``last_iter_run_used_fast_path`` records which path the most recent
# ``iter_run`` call took — tests assert on it directly.


def test_iter_run_single_fragment_takes_fast_path(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """A single-cube SUM aggregation by a dimension is the canonical
    fast-path shape: 1 fragment, merge SQL has only identity SUMs."""
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    assert len(plan.fragments) == 1

    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=100):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert engine.last_iter_run_used_fast_path
    assert {r[0]: r[1] for r in rows} == {"paid": 650.0, "pending": 25.0}


def test_iter_run_single_fragment_fast_path_uses_merge_spec(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], order=[("revenue", "desc")], limit=1),
        catalog,
    )
    plan = replace(plan, merge=replace(plan.merge, sql="not valid sql"))

    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=100):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert rows == [(675.0,)]
    assert engine.last_iter_run_used_fast_path


def test_iter_run_multi_fragment_skips_fast_path(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Two fragments → the merge needs a JOIN; can only run in DuckDB."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    assert len(plan.fragments) == 2

    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, _AsyncDialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _AsyncDialectTranslatingAdapter(bq_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=100):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert not engine.last_iter_run_used_fast_path
    assert {r[0] for r in rows} == {"EU", "US"}


def test_iter_run_fast_path_applies_order_by_and_limit(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """The Python-side merge handles ORDER BY + LIMIT correctly —
    so a fast-path result is identical to what DuckDB would return."""
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.status"],
            order=[("revenue", "desc")],
            limit=1,
        ),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=100):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert engine.last_iter_run_used_fast_path
    assert len(rows) == 1
    # paid (650) outranks pending (25); LIMIT 1 keeps only paid.
    assert rows[0] == ("paid", 650.0)


def test_iter_run_fast_path_handles_avg_decomposition(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """An AVG measure decomposes into stored sum + count columns at
    the fragment; the merge composes them via SUM(sum)/NULLIF(SUM(count),
    0). The fast path recognises this shape and computes the division
    in Python."""
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.avg_amount"],
            dimensions=["orders.status"],
        ),
        catalog,
    )
    assert len(plan.fragments) == 1
    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def collect() -> list[tuple[Any, ...]]:
        out: list[tuple[Any, ...]] = []
        async for chunk in engine.iter_run(plan, chunk_rows=100):
            out.extend(chunk)
        return out

    rows = _run(collect())
    assert engine.last_iter_run_used_fast_path
    by_status = {r[0]: r[1] for r in rows}
    # paid: (100+200+50+300)/4 = 162.5
    # pending: 25/1 = 25.0
    assert abs(by_status["paid"] - 162.5) < 1e-9
    assert abs(by_status["pending"] - 25.0) < 1e-9


def test_iter_run_fast_path_streams_in_requested_chunks(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """``chunk_rows=1`` yields one row per chunk even on the fast path."""
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = AsyncEngine()
    engine.register(Dialect.POSTGRES, AsyncDuckDBAdapter(pg_con))

    async def collect() -> list[int]:
        sizes: list[int] = []
        async for chunk in engine.iter_run(plan, chunk_rows=1):
            sizes.append(len(chunk))
        return sizes

    sizes = _run(collect())
    assert engine.last_iter_run_used_fast_path
    assert sizes == [1, 1]  # two status groups, one row each
