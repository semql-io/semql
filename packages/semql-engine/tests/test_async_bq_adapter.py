"""Tests for AsyncBigQueryAdapter (BQ target)."""

# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# The BQ param translator is duck-typed; pyright can't see into
# the fake _BQ_FAKE_TYPE dict (keyed on `type`).

# mypy: disable-error-code=var-annotated

# google-cloud-bigquery ships a sync client - the .query() call
# returns a RowIterator that's sync-iterable. The async adapter
# dispatches the job's .query() to a worker thread (which both runs
# the query and materialises the rows) and converts the typed
# @name-style parameters into BQ's ScalarQueryParameter /
# ArrayQueryParameter structured-parameter form.
#
# We use a fake BQClient rather than the real google.cloud.bigquery.Client
# to avoid the heavyweight dependency - the shape under test is the
# async dispatch + param translation, both of which are independent of
# the real driver's network code. The default translator imports
# google.cloud.bigquery lazily; tests pass a stub translator that
# builds duck-typed _ScalarParam objects.

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
    Filter,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql_engine import AsyncBigQueryAdapter, AsyncEngine


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class _FakeBQRow:
    def __init__(self, values: tuple[Any, ...]) -> None:
        self._values = values

    def __getitem__(self, i: int) -> Any:
        return self._values[i]

    def __iter__(self) -> Any:
        return iter(self._values)


class _FakeBQSchema:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    @property
    def names(self) -> list[str]:
        return list(self._names)


class _FakeBQResult:
    """Mimics ``google.cloud.bigquery.table.RowIterator``.

    ``.schema`` exposes the column names; iteration yields ``_FakeBQRow``
    objects indexable by position. The fake also records the most
    recent query + job_config so tests can assert on what was sent."""

    def __init__(self, rows: list[tuple[Any, ...]], schema: list[str]) -> None:
        self._rows = rows
        self._schema = _FakeBQSchema(schema)

    @property
    def schema(self) -> _FakeBQSchema:
        return self._schema

    def __iter__(self) -> Any:
        return iter(_FakeBQRow(r) for r in self._rows)

    def to_dataframe(self) -> Any:  # pragma: no cover - not exercised in tests
        raise NotImplementedError


class _FakeBQClient:
    def __init__(self, rows: list[tuple[Any, ...]], columns: list[str]) -> None:
        self._rows = rows
        self._columns = columns
        self.calls: list[dict[str, Any]] = []

    def query(self, sql: str, job_config: Any = None) -> _FakeBQResult:
        params: list[Any] = []
        if job_config is not None:
            params = list(getattr(job_config, "query_parameters", []) or [])
        self.calls.append({"sql": sql, "params": params})
        return _FakeBQResult(self._rows, self._columns)


class _ScalarParam:
    """Duck-typed stand-in for ``google.cloud.bigquery.ScalarQueryParameter``.

    Exposes the same ``.name`` / ``._type`` / ``.value`` attributes."""

    def __init__(self, name: str, type_: str, value: Any) -> None:
        self.name = name
        self._type = type_
        self.value = value


_BQ_FAKE_TYPE: dict[type, str] = {
    bool: "BOOL",
    int: "INT64",
    float: "FLOAT64",
    str: "STRING",
}


def _fake_translator(params: Mapping[str, Any]) -> Any:
    """Build a ``job_config`` with one ``_ScalarParam`` per name.

    The real BQ ``ScalarQueryParameter`` constructor is
    ``(name, type_, value)``; we mirror that signature."""

    class _FakeJobConfig:
        def __init__(self) -> None:
            self.query_parameters: list[_ScalarParam] = []

    def _type_for(v: object) -> str:
        return _BQ_FAKE_TYPE.get(type(v), "STRING")

    job_config = _FakeJobConfig()
    for name, value in params.items():
        if isinstance(value, list):
            element_type = _type_for(value[0]) if value else "STRING"
            job_config.query_parameters.append(
                _ScalarParam(name=name, type_=f"ARRAY<{element_type}>", value=value)
            )
        else:
            job_config.query_parameters.append(
                _ScalarParam(name=name, type_=_type_for(value), value=value)
            )
    return job_config


# -----------------------------------------------------------------
# Tests
# -----------------------------------------------------------------


def test_async_bq_adapter_returns_columns_and_rows() -> None:
    client = _FakeBQClient(rows=[(1, "x"), (2, "y")], columns=["a", "b"])
    adapter = AsyncBigQueryAdapter(client, translator=_fake_translator)
    result = _run(adapter.execute("SELECT a, b FROM t ORDER BY a", {}))
    assert result.columns == ["a", "b"]
    assert list(result.rows) == [(1, "x"), (2, "y")]


def test_async_bq_adapter_translates_named_params_to_structured() -> None:
    client = _FakeBQClient(rows=[("paid", 100.0)], columns=["status", "amount"])
    adapter = AsyncBigQueryAdapter(client, translator=_fake_translator)
    _run(
        adapter.execute(
            "SELECT status, amount FROM t WHERE status = @status",
            {"status": "paid"},
        )
    )
    assert len(client.calls) == 1
    params = client.calls[0]["params"]
    assert len(params) == 1
    p = params[0]
    assert p.name == "status"
    assert p.value == "paid"


def test_async_bq_adapter_handles_empty_params() -> None:
    client = _FakeBQClient(rows=[(1,)], columns=["n"])
    adapter = AsyncBigQueryAdapter(client, translator=_fake_translator)
    _run(adapter.execute("SELECT COUNT(*) AS n FROM t", {}))
    assert client.calls[0]["params"] == []


def test_async_bq_adapter_converts_param_types() -> None:
    client = _FakeBQClient(rows=[], columns=[])
    adapter = AsyncBigQueryAdapter(client, translator=_fake_translator)
    _run(
        adapter.execute(
            "SELECT * FROM t WHERE a = @a AND b = @b AND c = @c AND d = @d AND e = @e",
            {
                "a": 1,
                "b": "hello",
                "c": 1.5,
                "d": True,
                "e": ["a", "b"],
            },
        )
    )
    param_by_name = {p.name: p for p in client.calls[0]["params"]}
    assert param_by_name["a"]._type == "INT64"
    assert param_by_name["b"]._type == "STRING"
    assert param_by_name["c"]._type == "FLOAT64"
    assert param_by_name["d"]._type == "BOOL"
    assert param_by_name["e"]._type == "ARRAY<STRING>"


def test_async_bq_adapter_preserves_sql_verbatim() -> None:
    client = _FakeBQClient(rows=[], columns=[])
    adapter = AsyncBigQueryAdapter(client, translator=_fake_translator)
    sql = "SELECT * FROM t WHERE x = @x AND y > @y"
    _run(adapter.execute(sql, {"x": "a", "y": 1}))
    assert client.calls[0]["sql"] == sql


def test_async_bq_adapter_runs_under_event_loop() -> None:
    """Dispatch should release the loop. We verify by interleaving a
    sibling coroutine that ticks a counter during a 50ms sync block."""

    class _SlowClient:
        def __init__(self) -> None:
            self.calls = 0

        def query(self, sql: str, job_config: Any = None) -> _FakeBQResult:
            self.calls += 1
            time.sleep(0.05)
            return _FakeBQResult(rows=[(1,)], schema=["a"])

    adapter = AsyncBigQueryAdapter(_SlowClient(), translator=_fake_translator)
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


def test_async_bq_adapter_runs_through_async_engine_via_duckdb_stand_in() -> None:
    """End-to-end: register ``AsyncBigQueryAdapter`` on a fake BQ client
    that translates ``@name`` to ``$name`` and runs the SQL against
    in-memory DuckDB. The adapter itself must not perform the
    translation (that's the test wrapper's job) - the SQL it hands to
    the BQ client still has ``@name``."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (status TEXT, amount DOUBLE)")
    con.execute("INSERT INTO orders VALUES ('paid', 100.0), ('pending', 25.0)")

    class _DuckDBBackedBQClient:
        def __init__(self, raw: duckdb.DuckDBPyConnection) -> None:
            self._raw = raw
            self.calls: list[dict[str, Any]] = []

        def query(self, sql: str, job_config: Any = None) -> _FakeBQResult:
            translated = re.sub(r"@(\w+)", r"$\1", sql)
            params: dict[str, Any] = {}
            if job_config is not None:
                params = {p.name: p.value for p in getattr(job_config, "query_parameters", [])}
            cur = self._raw.execute(translated, params if params else None)
            self.calls.append({"sql": sql, "translated": translated, "params": params})
            desc = list(cur.description or [])
            cols = [str(d[0]) for d in desc]
            return _FakeBQResult(rows=list(cur.fetchall()), schema=cols)

    client = _DuckDBBackedBQClient(con)

    def _duckback_translator(params: Mapping[str, Any]) -> Any:
        # Same translator as the test fake - the engine doesn't see the
        # implementation, only the job_config shape.
        return _fake_translator(params)

    adapter = AsyncBigQueryAdapter(client, translator=_duckback_translator)

    cube = Cube(
        name="orders",
        backend=Backend.BIGQUERY,
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
    engine.register(Backend.BIGQUERY, adapter)
    result = _run(engine.run(plan))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"paid": 100.0}
    assert any("@" in c["sql"] for c in client.calls), (
        f"AsyncBigQueryAdapter rewrote the SQL; should have been a pass-through: {client.calls}"
    )


def test_async_bq_adapter_default_translator_raises_without_google_cloud() -> None:
    """If the user doesn't pass ``translator=`` and the real BQ package
    isn't installed, the default translator raises a helpful error at
    execute time (not at adapter construction time)."""
    client = _FakeBQClient(rows=[(1,)], columns=["a"])
    adapter = AsyncBigQueryAdapter(client)  # no translator override

    # Force the default translator path. ``google.cloud.bigquery`` is
    # not a dev dep, so the import will fail and the helper raises.
    with pytest.raises(RuntimeError, match="google-cloud-bigquery"):
        _run(adapter.execute("SELECT 1 AS a", {}))
