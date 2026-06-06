"""Per-strategy unit tests for the BackendStrategy Protocol.

Strategies are stateless, string-in/string-out — trivially testable.
The compiler's job is param-name allocation, resolution, joins,
GROUP BY, ordering; the strategy's job is dialect-specific SQL shape
(placeholders, truncation, contains predicate) and value transforms
tied to that shape (e.g. PG baking ``%v%``).
"""

from __future__ import annotations

from typing import Any

import pytest
from semql.backend import (
    BackendStrategy,
    ClickHouseStrategy,
    MetaStrategy,
    PostgresStrategy,
    strategy_for,
)
from semql.introspect import META_CUBES
from semql.model import Backend, Cube, Dimension, Measure

# ---------------------------------------------------------------------------
# placeholder()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim_type", ["string", "number", "time", "bool", "uuid"])
def test_postgres_placeholder_always_percent_paren(dim_type: str) -> None:
    s = PostgresStrategy()
    assert s.placeholder("p0", dim_type) == "%(p0)s"


def test_postgres_placeholder_uses_given_name() -> None:
    s = PostgresStrategy()
    assert s.placeholder("p7", "string") == "%(p7)s"
    assert s.placeholder("start_ts", "time") == "%(start_ts)s"


def test_clickhouse_placeholder_carries_typed_suffix() -> None:
    s = ClickHouseStrategy()
    assert s.placeholder("p0", "string") == "{p0:String}"
    assert s.placeholder("p1", "number") == "{p1:Float64}"
    assert s.placeholder("p2", "time") == "{p2:DateTime}"
    assert s.placeholder("p3", "bool") == "{p3:UInt8}"
    # uuid binds as String — CH UUIDs are quoted strings on the wire.
    assert s.placeholder("p4", "uuid") == "{p4:String}"


def test_meta_strategy_placeholder_matches_postgres() -> None:
    """META cubes are materialised as VALUES literals — when params do
    appear (rare), they should follow the Postgres convention."""
    s = MetaStrategy()
    assert s.placeholder("p0", "string") == "%(p0)s"


# ---------------------------------------------------------------------------
# trunc()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", ["hour", "day", "week", "month"])
def test_postgres_trunc_uses_date_trunc(granularity: str) -> None:
    s = PostgresStrategy()
    assert s.trunc(granularity, "x.ts") == f"date_trunc('{granularity}', x.ts)"


def test_clickhouse_trunc_uses_toStartOf_family() -> None:
    s = ClickHouseStrategy()
    assert s.trunc("hour", "x.ts") == "toStartOfHour(x.ts)"
    assert s.trunc("day", "x.ts") == "toStartOfDay(x.ts)"
    assert s.trunc("week", "x.ts") == "toStartOfWeek(x.ts)"
    assert s.trunc("month", "x.ts") == "toStartOfMonth(x.ts)"


# ---------------------------------------------------------------------------
# emit_contains() — proves the PG-bakes-%v%, CH-passes-literal contract
# ---------------------------------------------------------------------------


def test_postgres_emit_contains_bakes_percent_wildcards() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> str:  # noqa: ANN401
        bound.append((value, dim_type))
        return f"%(p{len(bound) - 1})s"

    s = PostgresStrategy()
    sql = s.emit_contains("x.email", "@acme.com", bind)
    assert sql == "x.email ILIKE %(p0)s"
    # PG bakes the wildcards INTO the bound value, so the placeholder
    # body is just the user's input wrapped.
    assert bound == [("%@acme.com%", "string")]


def test_clickhouse_emit_contains_passes_value_literally() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> str:  # noqa: ANN401
        bound.append((value, dim_type))
        return f"{{p{len(bound) - 1}:String}}"

    s = ClickHouseStrategy()
    sql = s.emit_contains("x.email", "@acme.com", bind)
    assert sql == "positionCaseInsensitive(x.email, {p0:String}) > 0"
    # CH expects the raw substring — no wildcard wrapping.
    assert bound == [("@acme.com", "string")]


# ---------------------------------------------------------------------------
# emit_source() — vanilla cube vs META cube
# ---------------------------------------------------------------------------


def test_postgres_emit_source_vanilla_cube_returns_table_as_alias() -> None:
    cube = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )

    def resolve(s: str) -> str:
        return s  # nothing to resolve in this fixture

    out = PostgresStrategy().emit_source(cube, {"orders": cube}, resolve)
    assert out == "public.orders AS o"


def test_postgres_emit_source_threads_resolver_over_table_name() -> None:
    """``emit_source`` should send the cube's ``table`` through the
    resolver so ``{schema}`` (and other context placeholders) get
    substituted before the table is wrapped with ``AS alias``."""
    cube = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )

    def resolve(s: str) -> str:
        return s.replace("{schema}", "prod")

    out = PostgresStrategy().emit_source(cube, {"orders": cube}, resolve)
    assert out == "prod.orders AS o"


def test_meta_emit_source_materialises_values_literal() -> None:
    """META cubes don't have a backing table; their source is a VALUES
    subquery the strategy builds from the catalog snapshot."""
    catalog = {c.name: c for c in META_CUBES}
    s = MetaStrategy()
    out = s.emit_source(META_CUBES[0], catalog, lambda x: x)
    # Should produce a SELECT-over-VALUES wrapped subquery aliased to
    # the META cube's alias.
    assert "VALUES" in out
    assert f"AS {META_CUBES[0].alias}" in out


# ---------------------------------------------------------------------------
# Protocol conformance — runtime_checkable
# ---------------------------------------------------------------------------


def test_strategies_satisfy_protocol_at_runtime() -> None:
    for s in (PostgresStrategy(), ClickHouseStrategy(), MetaStrategy()):
        assert isinstance(s, BackendStrategy)


# ---------------------------------------------------------------------------
# strategy_for() — registry + DI hybrid
# ---------------------------------------------------------------------------


def test_strategy_for_returns_default_for_known_backend() -> None:
    assert isinstance(strategy_for(Backend.POSTGRES), PostgresStrategy)
    assert isinstance(strategy_for(Backend.CLICKHOUSE), ClickHouseStrategy)
    assert isinstance(strategy_for(Backend.META), MetaStrategy)


def test_strategy_for_accepts_overrides() -> None:
    fake = PostgresStrategy()  # any concrete impl
    result = strategy_for(Backend.POSTGRES, overrides={Backend.POSTGRES: fake})
    assert result is fake


def test_strategy_for_duckdb_uses_postgres_default() -> None:
    """The TODO notes this is an open question; pin the current behavior
    so a future revisit shows up as a test diff, not a silent change."""
    assert isinstance(strategy_for(Backend.DUCKDB), PostgresStrategy)
