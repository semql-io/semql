"""Reusable observable SQL-adapter checks, not backend capability certification.

Cases contain an independent expected result, not another adapter's output.
Unordered SQL compares bags (duplicates matter); only an explicit order promise
compares sequences. Each execution consumes its result iterator exactly once.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from semql import Dialect
from semql_engine import Adapter, AdapterResult, AsyncAdapter


@dataclass(frozen=True)
class AdapterCase:
    name: str
    sql: str
    params: Mapping[str, Any]
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    column_types: tuple[str | None, ...] | None = None
    ordered: bool = False
    deterministic: bool = True


def _cell(value: object) -> tuple[type[object], object]:
    # Decimal equality with float/int would otherwise hide driver coercion.
    return type(value), value


def assert_result(result: AdapterResult, case: AdapterCase) -> None:
    """Assert labels, optional physical schema, row width, values and multiplicity."""
    assert tuple(result.columns) == case.columns, f"{case.name}: labels"
    if result.column_types is not None:
        assert len(result.column_types) == len(case.columns), f"{case.name}: schema width"
    if case.column_types is not None:
        assert result.column_types is not None, f"{case.name}: missing schema"
        assert tuple(result.column_types) == case.column_types, f"{case.name}: schema"
    rows = tuple(tuple(row) for row in result.rows)
    assert all(len(row) == len(case.columns) for row in rows), f"{case.name}: row width"
    actual = tuple(tuple(_cell(value) for value in row) for row in rows)
    expected = tuple(tuple(_cell(value) for value in row) for row in case.rows)
    assert Counter(actual) == Counter(expected), f"{case.name}: row multiset"
    if case.ordered:
        assert actual == expected, f"{case.name}: promised order"


def assert_adapter(adapter: Adapter, case: AdapterCase) -> None:
    """Run deterministic cases twice, checking each result against the oracle."""
    for _ in range(2 if case.deterministic else 1):
        assert_result(adapter.execute(case.sql, case.params), case)


async def assert_async_adapter(adapter: AsyncAdapter, case: AdapterCase) -> None:
    for _ in range(2 if case.deterministic else 1):
        assert_result(await adapter.execute(case.sql, case.params), case)


# Deliberately repeated rows and values too precise for a float round-trip.
SOURCE_ROWS = (
    (1, "O'Reilly", Decimal("12.3401")),
    (1, "O'Reilly", Decimal("12.3401")),
    (2, "other", None),
    (3, "other", Decimal("-0.0002")),
)


def adapter_cases(
    dialect: Dialect,
    *,
    table: str = "adapter_values",
    column_types: tuple[str | None, ...] | None = None,
) -> tuple[AdapterCase, ...]:
    """Portable shapes with native placeholders and exact numeric values."""
    placeholders = {
        Dialect.DUCKDB: "$label",
        Dialect.POSTGRES: "%(label)s",
        Dialect.BIGQUERY: "@label",
        Dialect.CLICKHOUSE: "{label:String}",
    }
    source = SOURCE_ROWS
    ordered_source = tuple(reversed(source))
    columns = ("key", "label", "amount")
    sql = f"SELECT id AS key, label, amount FROM {table}"

    def case(
        name: str,
        suffix: str,
        rows: tuple[tuple[Any, ...], ...],
        *,
        params: Mapping[str, Any] | None = None,
        ordered: bool = False,
    ) -> AdapterCase:
        return AdapterCase(name, sql + suffix, params or {}, columns, rows, column_types, ordered)

    # Aggregate oracle is Python Decimal arithmetic over the input fixture,
    # independent of generated SQL or a second execution strategy.
    total = sum((row[2] for row in SOURCE_ROWS if row[2] is not None), Decimal(0))
    return (
        case("bag", "", source),
        case("ordered", " ORDER BY id DESC", ordered_source, ordered=True),
        case(
            "bound",
            f" WHERE label = {placeholders[dialect]}",
            source[:2],
            params={"label": "O'Reilly"},
        ),
        case(
            "rebound",
            f" WHERE label = {placeholders[dialect]}",
            source[2:],
            params={"label": "other"},
        ),
        case("decimal-null", " WHERE id IN (2, 3)", source[2:]),
        case("all-null", " WHERE id = 2", source[2:3]),
        case("empty", " WHERE id < 0", ()),
        AdapterCase(
            "aggregate",
            f"SELECT SUM(amount) AS total FROM {table}",
            {},
            ("total",),
            ((total,),),
        ),
    )
