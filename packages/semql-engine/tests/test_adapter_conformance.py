"""Behavioral adapter contract checks for concrete driver surfaces and stand-ins."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Generator, Mapping
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import duckdb
import pytest
from adapter_conformance import (
    SOURCE_ROWS,
    AdapterCase,
    adapter_cases,
    assert_adapter,
    assert_async_adapter,
    assert_result,
)
from semql import Dialect
from semql_engine import (
    Adapter,
    AdapterResult,
    AsyncAdapter,
    AsyncBigQueryAdapter,
    AsyncClickHouseAdapter,
    AsyncDBAPIAdapter,
    AsyncDuckDBAdapter,
    DBAPIAdapter,
    DuckDBAdapter,
    to_async_adapter,
)


@pytest.fixture()
def values_connection() -> Generator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE adapter_values(id INTEGER, label VARCHAR, amount DECIMAL(12,4))"
    )
    connection.executemany("INSERT INTO adapter_values VALUES (?, ?, ?)", list(SOURCE_ROWS))
    try:
        yield connection
    finally:
        connection.close()


@pytest.mark.parametrize("adapter_kind", ["duckdb-driver", "dbapi-duckdb-driver"])
def test_shared_pack_actual_duckdb_driver(
    values_connection: duckdb.DuckDBPyConnection, adapter_kind: str
) -> None:
    if adapter_kind == "duckdb-driver":
        adapter: Adapter = DuckDBAdapter(values_connection)
    else:
        adapter = DBAPIAdapter(_DuckDBDBAPIConnection(values_connection))
    cases = adapter_cases(Dialect.DUCKDB)
    for case in cases:
        assert_adapter(adapter, case)


class _DuckDBDBAPICursor:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection
        self.result: Any = None
        self.description: Any = None

    def execute(self, sql: str, params: Mapping[str, Any] | None = None) -> None:
        self.result = self.connection.execute(sql, dict(params) if params else None)
        self.description = self.result.description

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self.result.fetchall())

    def close(self) -> None:
        return None


class _DuckDBDBAPIConnection:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    def cursor(self) -> _DuckDBDBAPICursor:
        return _DuckDBDBAPICursor(self.connection)


@pytest.mark.parametrize(
    "fault", ["changed", "missing", "duplicated", "reordered", "labels", "schema", "coerced"]
)
def test_shared_assertions_reject_consumer_visible_faults(fault: str) -> None:
    case = AdapterCase(
        "fault",
        "SELECT",
        {},
        ("id", "amount"),
        ((1, Decimal("1.25")), (1, Decimal("1.25")), (2, None)),
        ("INTEGER", "DECIMAL(12,2)"),
        ordered=fault == "reordered",
    )
    rows = list(case.rows)
    result = AdapterResult(columns=list(case.columns), rows=rows, column_types=case.column_types)
    if fault == "changed":
        rows[0] = (1, Decimal("1.26"))
    elif fault == "missing":
        rows.pop()
    elif fault == "duplicated":
        rows.append((2, None))
    elif fault == "reordered":
        rows.reverse()
    elif fault == "labels":
        result.columns[0] = "wrong"
    elif fault == "coerced":
        # 1.25 is exactly representable as float; numeric equality alone misses this.
        rows[0] = (1, 1.25)
    else:
        result.column_types = ("INTEGER", "DOUBLE")
    with pytest.raises(AssertionError):
        assert_result(result, case)


def test_repeated_rows_are_compared_as_a_multiset() -> None:
    case = AdapterCase("duplicates", "", {}, ("id",), ((1,), (1,), (2,)))
    assert_result(AdapterResult(["id"], [(2,), (1,), (1,)]), case)


@pytest.mark.parametrize(
    "adapter_kind", ["async-duckdb-driver", "async-dbapi-duckdb-driver", "wrapped-sync-driver"]
)
def test_async_shared_pack_actual_duckdb_driver(
    values_connection: duckdb.DuckDBPyConnection, adapter_kind: str
) -> None:
    adapter: AsyncAdapter
    if adapter_kind == "async-duckdb-driver":
        adapter = AsyncDuckDBAdapter(values_connection)
    elif adapter_kind == "async-dbapi-duckdb-driver":
        adapter = AsyncDBAPIAdapter(_DuckDBDBAPIConnection(values_connection))
    else:
        adapter = to_async_adapter(DuckDBAdapter(values_connection))

    async def run() -> None:
        for case in adapter_cases(Dialect.DUCKDB):
            await assert_async_adapter(adapter, case)

    asyncio.run(run())


class _BigQueryRows(list[tuple[Any, ...]]):
    def __init__(self, cursor: duckdb.DuckDBPyConnection) -> None:
        self.schema = [
            SimpleNamespace(name=column[0], field_type=str(column[1]))
            for column in cursor.description
        ]
        super().__init__(cursor.fetchall())


@pytest.mark.parametrize("dialect", [Dialect.BIGQUERY, Dialect.CLICKHOUSE])
def test_async_shared_pack_sdk_boundary_with_duckdb_stand_in(
    values_connection: duckdb.DuckDBPyConnection, dialect: Dialect
) -> None:
    """Exercise the real adapters; DuckDB substitutes for the remote SQL service."""
    adapter: AsyncAdapter
    if dialect == Dialect.BIGQUERY:

        def translator(params: Mapping[str, Any]) -> SimpleNamespace:
            return SimpleNamespace(
                query_parameters=[
                    SimpleNamespace(name=name, value=value) for name, value in params.items()
                ]
            )

        def query(sql: str, job_config: Any) -> _BigQueryRows:
            params = {parameter.name: parameter.value for parameter in job_config.query_parameters}
            cursor = values_connection.execute(re.sub(r"@(\w+)", r"$\1", sql), params)
            return _BigQueryRows(cursor)

        adapter = AsyncBigQueryAdapter(SimpleNamespace(query=query), translator=translator)
    else:

        async def query_clickhouse(sql: str, parameters: Mapping[str, Any]) -> SimpleNamespace:
            translated = re.sub(r"\{(\w+):[^}]+\}", r"$\1", sql)
            cursor = values_connection.execute(translated, dict(parameters))
            return SimpleNamespace(
                column_names=[column[0] for column in cursor.description],
                column_types=[str(column[1]) for column in cursor.description],
                result_rows=cursor.fetchall(),
            )

        adapter = AsyncClickHouseAdapter(SimpleNamespace(query=query_clickhouse))

    async def run() -> None:
        for case in adapter_cases(dialect):
            await assert_async_adapter(adapter, case)

    asyncio.run(run())
