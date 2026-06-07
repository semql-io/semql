"""Adapter protocol for the in-process executor.

An ``Adapter`` is the glue between a backend's connection and the
semql executor. Given a ``(sql, params)`` pair, an adapter runs the
SQL on its backend and yields the result as an :class:`AdapterResult`
— a typed pair of ``columns: list[str]`` and ``rows: Iterable[Sequence
[Any]]`` (positional, matching ``columns`` order).

We keep the protocol tiny on purpose. Most backends already speak
PEP-249 (cursors with ``description`` + ``fetchall``); :class:`DBAPIAdapter`
wraps them. DuckDB is special-cased only because the executor runs the
merge step in DuckDB itself; :class:`DuckDBAdapter` reuses an existing
connection so adapters and the merge engine can share temp tables when
the user wants.
"""

from __future__ import annotations

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


__all__ = ["Adapter", "AdapterResult", "DBAPIAdapter", "DuckDBAdapter"]
