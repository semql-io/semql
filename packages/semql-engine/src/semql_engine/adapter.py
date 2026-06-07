"""Adapter protocols for the in-process executor.

An ``Adapter`` is the glue between a backend's connection and the
semql executor. Given a ``(sql, params)`` pair, an adapter runs the
SQL on its backend and yields the result as an :class:`AdapterResult`
— a typed pair of ``columns: list[str]`` and ``rows: Iterable[Sequence
[Any]]`` (positional, matching ``columns`` order).

Two parallel protocols ship:

- :class:`Adapter` — sync ``execute``. Wired up to :class:`semql_engine.Engine`.
- :class:`AsyncAdapter` — same shape, ``async def execute``. Wired up
  to :class:`semql_engine.AsyncEngine`. Production deployments running
  on asyncio (FastAPI / Litestar / aiohttp) avoid the per-call
  ``asyncio.to_thread`` boilerplate.

:func:`to_async_adapter` wraps any sync ``Adapter`` so it satisfies the
async protocol — useful when only a sync driver is available and the
caller is otherwise async-first.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class AdapterResult:
    """A query result as a column list + row iterator.

    ``rows`` is positional — each row is a tuple/list aligned to
    ``columns``. The executor zips them into dicts when materialising
    into DuckDB temp tables. Iterators are fine; the engine consumes
    each row exactly once.
    """

    columns: list[str]
    rows: Iterable[Sequence[Any]]


class Adapter(Protocol):
    """Minimal contract for a backend connection.

    Implementations execute a SQL string against their backend and
    return rows + column metadata. Parameter binding is the adapter's
    responsibility (sqlglot-emitted placeholders match the dialect, so
    a Postgres adapter sees ``$1``-style or named placeholders as
    appropriate)."""

    def execute(
        self,
        sql: str,
        params: Mapping[str, Any],
    ) -> AdapterResult: ...


class DBAPIAdapter:
    """PEP-249 adapter — wraps any DB-API 2.0 connection.

    Uses a fresh cursor per call, passes ``params`` as the second
    argument to ``cursor.execute`` (the named-parameter form is
    driver-specific; this assumes the driver matches the dialect the
    compiler emitted for the backend).

    Read-only by design: we never commit or close the connection; the
    caller owns its lifecycle.
    """

    def __init__(self, connection: Any) -> None:  # noqa: ANN401 — any PEP-249 conn
        self._conn = connection

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        cursor = self._conn.cursor()
        try:
            if params:
                cursor.execute(sql, dict(params))
            else:
                cursor.execute(sql)
            description: list[Any] = list(cursor.description or [])
            columns: list[str] = [str(d[0]) for d in description]
            rows = list(cursor.fetchall())
        finally:
            cursor.close()
        return AdapterResult(columns=columns, rows=rows)


class DuckDBAdapter:
    """DuckDB adapter — wraps an existing ``duckdb.DuckDBPyConnection``.

    DuckDB uses ``$name`` placeholders for named parameters; the
    compiler emits them already for DuckDB targets, so we pass
    ``params`` through unchanged.

    Useful for: local CSV / Parquet enrichment cubes (point a Cube at a
    file path and DuckDB reads it natively), in-memory test fixtures,
    and as a unified backend for users who don't want to manage
    multiple connections.
    """

    def __init__(self, connection: Any) -> None:  # noqa: ANN401 — duckdb conn
        self._conn = connection

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        cursor = self._conn.execute(sql, dict(params) if params else None)
        description: list[Any] = list(cursor.description or [])
        columns: list[str] = [str(d[0]) for d in description]
        rows = cursor.fetchall()
        return AdapterResult(columns=columns, rows=rows)


class AsyncAdapter(Protocol):
    """Async counterpart of :class:`Adapter`.

    ``execute`` is an awaitable that returns the same
    :class:`AdapterResult` shape. Implementations should be safe to
    call concurrently — :class:`semql_engine.AsyncEngine` runs all the
    fragments of a federated plan in parallel via ``asyncio.gather``.
    """

    async def execute(
        self,
        sql: str,
        params: Mapping[str, Any],
    ) -> AdapterResult: ...


class _SyncAsAsyncAdapter:
    """Internal wrapper produced by :func:`to_async_adapter`."""

    def __init__(self, inner: Adapter) -> None:
        self._inner = inner

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        # ``asyncio.to_thread`` releases the event loop so other
        # fragments registered on the AsyncEngine can run in parallel
        # even when this adapter is pure-Python sync.
        return await asyncio.to_thread(self._inner.execute, sql, params)


def to_async_adapter(adapter: Adapter) -> AsyncAdapter:
    """Wrap a sync :class:`Adapter` so it satisfies :class:`AsyncAdapter`.

    The wrapped adapter dispatches each ``execute`` to a worker thread
    via ``asyncio.to_thread`` — fragments scheduled on
    :class:`semql_engine.AsyncEngine` still run concurrently because
    the event loop is freed up while the thread blocks on I/O. Prefer a
    native async adapter when the underlying driver supports one; the
    bridge is the right answer for drivers that only ship sync APIs.
    """
    return _SyncAsAsyncAdapter(adapter)


class AsyncDuckDBAdapter:
    """Async DuckDB adapter — wraps an existing ``duckdb.DuckDBPyConnection``.

    DuckDB has no native async API; this adapter dispatches each
    ``execute`` to a worker thread via ``asyncio.to_thread``. Useful
    for single-fragment async plans, in-memory test fixtures, and for
    keeping async-first user code from inheriting a sync ``Engine``.
    """

    def __init__(self, connection: Any) -> None:  # noqa: ANN401 — duckdb conn
        self._inner = DuckDBAdapter(connection)

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        return await asyncio.to_thread(self._inner.execute, sql, params)


__all__ = [
    "Adapter",
    "AdapterResult",
    "AsyncAdapter",
    "AsyncDuckDBAdapter",
    "DBAPIAdapter",
    "DuckDBAdapter",
    "to_async_adapter",
]
