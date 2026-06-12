# pyright: reportAttributeAccessIssue=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# The BigQuery adapter reaches into the google.cloud.bigquery module
# at runtime (lazy import) so pyright can't see the types.
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
from collections.abc import Callable, Iterable, Mapping, Sequence
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


class AsyncDBAPIAdapter:
    """Async wrapper for any PEP-249 (DB-API 2.0) connection.

    The whole PEP-249 family — ``psycopg2``, ``psycopg`` (sync mode),
    ``pymysql``, ``mysql.connector``, ``snowflake-connector-python`` —
    is synchronous. Async drivers (asyncpg, ``psycopg`` async mode)
    are *not* DB-API; they implement :class:`AsyncAdapter` directly.

    This adapter dispatches each ``execute`` to a worker thread via
    :func:`asyncio.to_thread` so the event loop is free to schedule
    sibling coroutines while the DB-API call blocks. Parameters are
    passed as a dict — the named-parameter form is the only one
    guaranteed portable across the DB-API family; the caller is
    responsible for matching the compiler's emitted placeholder
    syntax to the driver (Postgres / Snowflake both use
    ``%(name)s``; the ``AsyncDBAPIAdapter`` itself is dialect-blind
    and passes the SQL through verbatim).

    Read-only by design: we never commit or close the connection;
    the caller owns its lifecycle.
    """

    def __init__(self, connection: Any) -> None:  # noqa: ANN401 — any PEP-249 conn
        self._conn = connection

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        return await asyncio.to_thread(self._sync_execute, sql, params)

    def _sync_execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
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


_BQ_TYPE_HINTS: tuple[tuple[str, tuple[type, ...]], ...] = (
    ("INT64", (int,)),
    ("FLOAT64", (float,)),
    ("BOOL", (bool,)),
    ("STRING", (str,)),
)


def _bq_type_for(value: Any) -> str:  # noqa: ANN401 — duck-typed on BQ param shape
    """Map a Python value to BQ's parameter ``type_`` string.

    Boolean must be checked before int (``bool`` is a subclass of
    ``int`` in Python — ``isinstance(True, int)`` is ``True``)."""
    for type_name, py_types in _BQ_TYPE_HINTS:
        if isinstance(value, py_types):
            return type_name
    return "STRING"


def _bq_array_type_for(value: Any) -> str:  # noqa: ANN401 — duck-typed on BQ param shape
    """Map a Python list's element type to BQ's ``ARRAY<T>`` form.

    Falls back to ``ARRAY<STRING>`` if the list is empty (BQ needs a
    non-null element type) or contains mixed types."""
    if not isinstance(value, list) or not value:
        return "ARRAY<STRING>"
    element = value[0]
    return f"ARRAY<{_bq_type_for(element)}>"


class AsyncBigQueryAdapter:
    """Async adapter for ``google.cloud.bigquery.Client``.

    The BQ client is synchronous: ``.query(sql, job_config=...)``
    returns a ``RowIterator`` that's sync-iterable. This adapter:

    1. Maps the SemQL params ``dict`` to BQ's structured-parameter
       form (``ScalarQueryParameter`` / ``ArrayQueryParameter``) so the
       job is *parameterised* — not f-string interpolated. This is
       the only safe way to run user-supplied values against BQ
       without exposing a SQL-injection surface.
    2. Dispatches the entire ``.query()`` call to a worker thread via
       :func:`asyncio.to_thread` so the event loop is free to schedule
       sibling coroutines during the round-trip.

    The translator is a function ``(params) -> job_config``. The
    default attempts to import ``google.cloud.bigquery`` and use the
    real ``QueryJobConfig`` / ``ScalarQueryParameter`` constructors.
    Tests inject a stub translator that builds their duck-typed
    fake objects. Pass ``translator=...`` to ``__init__`` to override.

    The ``google-cloud-bigquery`` package is *not* a hard import-time
    dependency — we try-import inside the default translator on first
    use, and raise a helpful error if it's missing.
    """

    def __init__(
        self,
        client: Any,  # noqa: ANN401 — google-cloud-bigquery
        translator: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self._client = client
        self._translator = translator or _default_bq_translator

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        return await asyncio.to_thread(self._sync_execute, sql, params)

    def _sync_execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        job_config = self._translator(params)
        result = self._client.query(sql, job_config=job_config)
        schema = getattr(result, "schema", None)
        column_names: list[str] = list(getattr(schema, "names", []) or [])
        if not column_names:
            first = next(iter(result), None)
            if first is not None:
                column_names = [str(i) for i in range(len(first))]
        rows: list[Sequence[Any]] = [tuple(r) for r in result]
        return AdapterResult(columns=column_names, rows=rows)


def _default_bq_translator(params: Mapping[str, Any]) -> Any:  # noqa: ANN401 — BQ types are duck-typed
    """Translate the params ``dict`` into a ``QueryJobConfig`` with one
    ``ScalarQueryParameter`` / ``ArrayQueryParameter`` per name.

    Imports ``google.cloud.bigquery`` lazily; raises a helpful error
    if it isn't installed."""
    try:
        from google.cloud import bigquery as _bq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "AsyncBigQueryAdapter needs `google-cloud-bigquery` to translate "
            "parameters into structured-query-parameter form. Install with "
            "`uv add google-cloud-bigquery`."
        ) from exc
    job_config = _bq.QueryJobConfig()
    for name, value in params.items():
        if isinstance(value, list):
            job_config.query_parameters.append(
                _bq.ArrayQueryParameter(name, _bq_array_type_for(value), value)
            )
        else:
            job_config.query_parameters.append(
                _bq.ScalarQueryParameter(name, _bq_type_for(value), value)
            )
    return job_config


class AsyncClickHouseAdapter:
    """Async adapter for ``clickhouse_connect.asynch_client.AsyncClient``.

    Unlike the BQ client, the clickhouse-connect async client has a
    true coroutine ``.query()`` method — it returns immediately when
    the row set starts streaming rather than blocking the event loop.
    The named-parameter form ``{name:Type}`` is native to the CH wire
    protocol, so the adapter just hands the params dict through as
    the ``parameters=`` keyword argument.

    The ``clickhouse-connect`` package is *not* a hard import-time
    dependency — we duck-type against its surface (``query(query,
    parameters=...)``) and let the user pass a real or fake client
    object.

    Returns rows via the same :class:`AdapterResult` shape the sync
    side uses: ``columns`` from ``QueryResult.column_names``, ``rows``
    as positional tuples aligned to that column order.
    """

    def __init__(self, client: Any) -> None:  # noqa: ANN401 — clickhouse-connect
        self._client = client

    async def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        result = await self._client.query(sql, parameters=dict(params))
        column_names: list[str] = list(getattr(result, "column_names", []) or [])
        # ``QueryResult.result_rows`` is a list of positional tuples;
        # ``named_results()`` returns dicts which we don't need.
        raw_rows: list[Sequence[Any]] = [tuple(r) for r in getattr(result, "result_rows", []) or []]
        return AdapterResult(columns=column_names, rows=raw_rows)


__all__ = [
    "Adapter",
    "AdapterResult",
    "AsyncAdapter",
    "AsyncDuckDBAdapter",
    "DBAPIAdapter",
    "DuckDBAdapter",
    "to_async_adapter",
]
