"""A5 — Engine result-cache hazards (architecture review 2026-06).

Three defects in the P7 result cache (``engine.py``):

1. **Aliased result.** A cache hit (and the first miss) returned the
   *stored* ``ExecutionResult`` by reference. A caller that mutated
   ``result.rows`` / ``result.columns`` (sort in place, append, pop)
   silently corrupted every later hit on the same plan.

2. **Unhashable key.** The cache key embedded raw param values. A
   list-valued param — exactly what a BigQuery ``ArrayQueryParameter``
   binds for an ``IN (...)`` filter — made the key tuple unhashable, so
   the ``key in self._cache`` membership test raised ``TypeError``
   before the query could run at all.

3. **No expiry.** Entries lived until evicted by LRU pressure or an
   explicit ``clear_cache``; there was no time-to-live for callers who
   want "cache for N seconds then re-check the source".

These tests pin: cache reads/writes hand out isolated copies; the key
tolerates list/dict/tuple-valued params; and ``cache_ttl`` expires
entries on a clock the test controls.
"""

from __future__ import annotations

from typing import Any

import duckdb
import pytest
from semql import Cube, Dialect, Dimension, Measure, SemanticQuery, compile_federated_query
from semql_engine import AdapterResult, DuckDBAdapter, Engine


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _plan() -> Any:
    return compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        {"orders": _orders()},
    )


def _pg_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES (1, 'paid', 100.0), (2, 'paid', 200.0), (3, 'pending', 25.0)"
    )
    return con


class _FixedAdapter:
    """Adapter that ignores sql/params and returns a canned shape.

    Lets a test put an awkward (e.g. list-valued) param on a fragment
    without a real backend needing to bind it."""

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.calls = 0

    def execute(self, sql: str, params: Any) -> AdapterResult:
        self.calls += 1
        return AdapterResult(columns=list(self.columns), rows=[])


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# Hazard 1 — cache must hand out isolated copies
# ---------------------------------------------------------------------------


def test_cache_hit_is_isolated_from_caller_mutation() -> None:
    """Mutating a returned result must not poison later cache hits."""
    engine = Engine(cache_size=8)
    engine.register(Dialect.POSTGRES, DuckDBAdapter(_pg_con()))
    plan = _plan()

    first = engine.run(plan)  # miss
    baseline_rows = list(first.rows)
    baseline_cols = list(first.columns)

    # Caller abuses the result it was handed.
    first.rows.append(("CORRUPT", -1))
    first.columns.append("injected")

    second = engine.run(plan)  # hit
    assert engine.cache_hits == 1
    assert second is not first
    assert second.rows == baseline_rows
    assert second.columns == baseline_cols
    assert "injected" not in second.columns

    # And mutating the hit result must not poison a third read either.
    second.rows.clear()
    third = engine.run(plan)
    assert third.rows == baseline_rows


def test_cache_hit_result_meta_is_isolated() -> None:
    """column_meta is a list of mutable dataclasses; a hit must not share
    element identity with a prior return."""
    engine = Engine(cache_size=8)
    engine.register(Dialect.POSTGRES, DuckDBAdapter(_pg_con()))
    plan = _plan()

    first = engine.run(plan)
    second = engine.run(plan)
    assert first.column_meta == second.column_meta
    assert all(a is not b for a, b in zip(first.column_meta, second.column_meta, strict=True))


# ---------------------------------------------------------------------------
# Hazard 2 — list/dict-valued params must not break the key
# ---------------------------------------------------------------------------


def test_cache_key_hashable_with_list_valued_params() -> None:
    """A list-valued param (BQ ArrayQueryParameter) must not make the
    cache key unhashable."""
    engine = Engine(cache_size=8)
    plan = _plan()
    plan.fragments[0].params["ids"] = [1, 2, 3]
    plan.merge.params["tags"] = ["a", "b"]
    # The bug raised TypeError here, before the query ever ran.
    assert isinstance(hash(engine._cache_key(plan)), int)


def test_cache_list_param_round_trips_to_a_hit() -> None:
    """End to end: a plan with a list-valued fragment param caches and
    hits on the second run instead of raising."""
    plan = _plan()
    plan.fragments[0].params["ids"] = [1, 2, 3]
    adapter = _FixedAdapter(plan.fragments[0].columns)
    engine = Engine(cache_size=8)
    engine.register(Dialect.POSTGRES, adapter)

    engine.run(plan)  # miss
    engine.run(plan)  # hit
    assert engine.cache_hits == 1
    assert engine.cache_misses == 1
    assert adapter.calls == 1  # the hit skipped the adapter


def test_cache_distinguishes_different_list_params() -> None:
    """Different list-param values must land in different cache slots."""
    engine = Engine(cache_size=8)
    plan_a = _plan()
    plan_a.fragments[0].params["ids"] = [1, 2, 3]
    plan_b = _plan()
    plan_b.fragments[0].params["ids"] = [1, 2, 4]
    assert engine._cache_key(plan_a) != engine._cache_key(plan_b)


# ---------------------------------------------------------------------------
# Hazard 3 — TTL expiry
# ---------------------------------------------------------------------------


def test_cache_entry_expires_after_ttl() -> None:
    """An entry older than cache_ttl is a miss, and re-runs the plan."""
    clock = _FakeClock()
    engine = Engine(cache_size=8, cache_ttl=30.0)
    engine._clock = clock
    engine.register(Dialect.POSTGRES, DuckDBAdapter(_pg_con()))
    plan = _plan()

    engine.run(plan)  # miss, stored with deadline = 1030
    clock.advance(10.0)
    engine.run(plan)  # still fresh -> hit
    assert engine.cache_hits == 1
    assert engine.cache_misses == 1

    clock.advance(25.0)  # now 1035 > 1030 deadline
    engine.run(plan)  # expired -> miss
    assert engine.cache_hits == 1
    assert engine.cache_misses == 2


def test_cache_ttl_none_never_expires() -> None:
    """Without a TTL, a far-future clock still hits."""
    clock = _FakeClock()
    engine = Engine(cache_size=8)  # cache_ttl defaults to None
    engine._clock = clock
    engine.register(Dialect.POSTGRES, DuckDBAdapter(_pg_con()))
    plan = _plan()

    engine.run(plan)
    clock.advance(10_000_000.0)
    engine.run(plan)
    assert engine.cache_hits == 1


def test_cache_ttl_must_be_positive() -> None:
    with pytest.raises(ValueError, match=r"(?i)cache_ttl"):
        Engine(cache_size=8, cache_ttl=0)
    with pytest.raises(ValueError, match=r"(?i)cache_ttl"):
        Engine(cache_size=8, cache_ttl=-5.0)
