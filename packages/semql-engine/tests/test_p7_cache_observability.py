"""Tests for P7: result cache + observability hooks on the engine.

P7 is a paired feature:

- **Result cache** on ``Engine.run``: a plan-keyed memoisation layer.
  The cache key is the *plan's emitted shape* — merge SQL, merge
  params, per-fragment SQL, per-fragment params, column list. The
  cached value is the final ``ExecutionResult``. Re-running the
  same plan skips both the per-fragment adapter calls and the
  DuckDB materialise-and-merge.

- **Observability hook** on ``Engine.run``: an ``on_execute`` callback
  fires after every run with timing information. A hit returns
  ``cache_hit=True``; a miss returns ``cache_hit=False`` along with
  the wall-clock duration of the underlying work.

The cache is *in-process* — P7 explicitly defers distributed cache
to a follow-up. The cache invalidates when the key shape changes;
callers that mutate the catalog must reset the engine (or the
cache) themselves. This is by design: invalidation is the hard
part of any cache, and the engine can't know which catalog
mutation the caller is making.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
from semql import Backend, Cube, Dimension, Measure, SemanticQuery, compile_federated_query
from semql_engine import (
    AdapterResult,
    DuckDBAdapter,
    Engine,
)


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _catalog() -> dict[str, Cube]:
    return {"orders": _orders()}


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES (1, 'paid', 100.0), (2, 'paid', 200.0), (3, 'pending', 25.0)"
    )
    return con


def test_engine_without_cache_runs_normally() -> None:
    """The default engine has no cache; behaviour is identical to
    before. (Backwards compat assertion.)"""
    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine()
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    result = engine.run(plan)
    assert result.columns == ["status", "revenue"]


def pg_con_for_test() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES (1, 'paid', 100.0), (2, 'paid', 200.0), (3, 'pending', 25.0)"
    )
    return con


def test_engine_result_cache_misses_first_time() -> None:
    """First run with the same plan is a miss."""
    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan)
    assert engine.cache_hits == 0
    assert engine.cache_misses == 1


def test_engine_result_cache_hits_second_time() -> None:
    """Second run of the same plan is a hit."""
    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan)
    engine.run(plan)
    assert engine.cache_hits == 1
    assert engine.cache_misses == 1


def test_engine_result_cache_skips_adapter_on_hit() -> None:
    """On a cache hit, the adapter must NOT be called. We use a
    counting adapter to assert this."""

    class CountingAdapter:
        def __init__(self, inner: DuckDBAdapter) -> None:
            self.inner = inner
            self.calls = 0

        def execute(self, sql: str, params: Any) -> AdapterResult:
            self.calls += 1
            return self.inner.execute(sql, params)

    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8)
    adapter = CountingAdapter(DuckDBAdapter(pg_con_for_test()))
    engine.register(Backend.POSTGRES, adapter)
    engine.run(plan)  # miss
    engine.run(plan)  # hit
    assert adapter.calls == 1


def test_engine_result_cache_different_plans_different_keys() -> None:
    """Two distinct plans don't collide in the cache."""
    catalog = _catalog()
    plan1 = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    plan2 = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"]),
        catalog,
    )
    engine = Engine(cache_size=8)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan1)
    engine.run(plan2)
    assert engine.cache_misses == 2
    assert engine.cache_hits == 0


def test_engine_result_cache_clear() -> None:
    """Engine.clear_cache() drops all cached results; the next run
    is a miss."""
    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan)
    engine.run(plan)
    assert engine.cache_hits == 1
    engine.clear_cache()
    engine.run(plan)
    assert engine.cache_hits == 1
    assert engine.cache_misses == 2


def test_engine_cache_size_zero_disables_cache() -> None:
    """cache_size=0 means no caching — every run is a miss."""
    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=0)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan)
    engine.run(plan)
    assert engine.cache_misses == 2
    assert engine.cache_hits == 0


def test_engine_observability_hook_fires_on_run() -> None:
    """Engine(on_execute=...) gets a callback for every run, with
    timing + hit/miss info."""
    events: list[dict[str, Any]] = []

    def on_execute(plan: Any, elapsed_ms: float, *, cache_hit: bool) -> None:
        events.append(
            {"cache_hit": cache_hit, "elapsed_ms": elapsed_ms, "fragments": len(plan.fragments)}
        )

    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8, on_execute=on_execute)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    engine.run(plan)  # miss
    engine.run(plan)  # hit
    assert len(events) == 2
    assert events[0]["cache_hit"] is False
    assert events[1]["cache_hit"] is True
    # Each event carries the fragment count + a non-negative elapsed.
    for e in events:
        assert e["fragments"] == 1
        assert e["elapsed_ms"] >= 0


def test_engine_observability_hook_swallows_exceptions() -> None:
    """A buggy hook must not break the engine. The hook's exception
    is logged (or just dropped) — the engine still returns the result."""

    def bad_hook(plan: Any, elapsed_ms: float, *, cache_hit: bool) -> None:
        raise RuntimeError("hook boom")

    catalog = _catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    engine = Engine(cache_size=8, on_execute=bad_hook)
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con_for_test()))
    # Should not raise.
    result = engine.run(plan)
    assert result.columns == ["status", "revenue"]


def test_engine_cache_size_negative_rejected() -> None:
    """cache_size must be non-negative."""
    with pytest.raises(ValueError):
        Engine(cache_size=-1)
