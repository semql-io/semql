# pyright: reportPrivateImportUsage=false
"""Per-strategy unit tests for the DialectStrategy Protocol.

Strategies return sqlglot AST nodes; tests render them via ``.sql()``
under the dialect we'd emit at, and assert against the exact bytes the
compiler weaves into ``compile_query``'s output.
"""

from __future__ import annotations

from typing import Any

import pytest
from semql.backend import (
    ClickHouseDialect,
    DialectStrategy,
    MetaDialect,
    PostgresDialect,
    dialect_for,
    render,
)
from semql.introspect import META_CUBES
from semql.model import Cube, Dialect, Dimension, Measure
from sqlglot import exp

# ---------------------------------------------------------------------------
# placeholder()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim_type", ["string", "number", "time", "bool", "uuid"])
def test_postgres_placeholder_always_percent_paren(dim_type: str) -> None:
    s = PostgresDialect()
    assert render(s.placeholder("p0", dim_type), Dialect.POSTGRES) == "%(p0)s"


def test_postgres_placeholder_uses_given_name() -> None:
    s = PostgresDialect()
    assert render(s.placeholder("p7", "string"), Dialect.POSTGRES) == "%(p7)s"
    assert render(s.placeholder("start_ts", "time"), Dialect.POSTGRES) == "%(start_ts)s"


def test_clickhouse_placeholder_carries_typed_suffix() -> None:
    s = ClickHouseDialect()
    assert render(s.placeholder("p0", "string"), Dialect.CLICKHOUSE) == "{p0:String}"
    assert render(s.placeholder("p1", "number"), Dialect.CLICKHOUSE) == "{p1:Float64}"
    assert render(s.placeholder("p2", "time"), Dialect.CLICKHOUSE) == "{p2:DateTime}"
    assert render(s.placeholder("p3", "bool"), Dialect.CLICKHOUSE) == "{p3:UInt8}"
    # uuid binds as String — CH UUIDs are quoted strings on the wire.
    assert render(s.placeholder("p4", "uuid"), Dialect.CLICKHOUSE) == "{p4:String}"


def test_meta_strategy_placeholder_matches_postgres() -> None:
    """META cubes are materialised as VALUES literals — when params do
    appear (rare), they should follow the Postgres convention."""
    s = MetaDialect()
    assert render(s.placeholder("p0", "string"), Dialect.META) == "%(p0)s"


# ---------------------------------------------------------------------------
# trunc()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", ["hour", "day", "week", "month"])
def test_postgres_trunc_uses_date_trunc(granularity: str) -> None:
    s = PostgresDialect()
    expr = exp.column("ts", table="x")
    out = render(s.trunc(granularity, expr), Dialect.POSTGRES)
    assert out == f"date_trunc('{granularity}', x.ts)"


def test_clickhouse_trunc_uses_toStartOf_family() -> None:
    s = ClickHouseDialect()
    expr = exp.column("ts", table="x")
    assert render(s.trunc("hour", expr), Dialect.CLICKHOUSE) == "toStartOfHour(x.ts)"
    assert render(s.trunc("day", expr), Dialect.CLICKHOUSE) == "toStartOfDay(x.ts)"
    assert render(s.trunc("week", expr), Dialect.CLICKHOUSE) == "toStartOfWeek(x.ts)"
    assert render(s.trunc("month", expr), Dialect.CLICKHOUSE) == "toStartOfMonth(x.ts)"


# ---------------------------------------------------------------------------
# emit_contains() — proves the PG-bakes-%v%, CH-passes-literal contract
# ---------------------------------------------------------------------------


def test_postgres_emit_contains_bakes_percent_wildcards() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        bound.append((value, dim_type))
        return exp.Placeholder(this=f"p{len(bound) - 1}")

    s = PostgresDialect()
    field = exp.column("email", table="x")
    sql = render(s.emit_contains(field, "@acme.com", bind), Dialect.POSTGRES)
    assert sql == "x.email ILIKE %(p0)s"
    # PG bakes the wildcards INTO the bound value, so the placeholder
    # body is just the user's input wrapped.
    assert bound == [("%@acme.com%", "string")]


def test_clickhouse_emit_contains_passes_value_literally() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        bound.append((value, dim_type))
        ph = exp.Placeholder(this=f"p{len(bound) - 1}")
        ph.set("kind", "String")
        return ph

    s = ClickHouseDialect()
    field = exp.column("email", table="x")
    sql = render(s.emit_contains(field, "@acme.com", bind), Dialect.CLICKHOUSE)
    assert sql == "positionCaseInsensitive(x.email, {p0:String}) > 0"
    # CH expects the raw substring — no wildcard wrapping.
    assert bound == [("@acme.com", "string")]


# ---------------------------------------------------------------------------
# emit_source() — vanilla cube vs META cube
# ---------------------------------------------------------------------------


def test_postgres_emit_source_vanilla_cube_returns_table_as_alias() -> None:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )

    def resolve(s: str) -> str:
        return s  # nothing to resolve in this fixture

    out = render(PostgresDialect().emit_source(cube, {"orders": cube}, resolve), Dialect.POSTGRES)
    assert out == "public.orders AS o"


def test_postgres_emit_source_threads_resolver_over_table_name() -> None:
    """``emit_source`` should send the cube's ``table`` through the
    resolver so ``{schema}`` (and other context placeholders) get
    substituted before the table is wrapped with ``AS alias``."""
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )

    def resolve(s: str) -> str:
        return s.replace("{schema}", "prod")

    out = render(PostgresDialect().emit_source(cube, {"orders": cube}, resolve), Dialect.POSTGRES)
    assert out == "prod.orders AS o"


def test_postgres_emit_source_quotes_reserved_word_table_name() -> None:
    """Table names that are SQL reserved words (e.g. ``using``) must be
    double-quoted so the generated SQL parses as a valid SELECT."""
    cube = Cube(
        name="evt",
        backend=Dialect.POSTGRES,
        table="using",
        alias="u",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[],
    )
    out = render(PostgresDialect().emit_source(cube, {"evt": cube}, lambda x: x), Dialect.POSTGRES)
    assert out == '"using" AS u'


def test_meta_emit_source_materialises_values_literal() -> None:
    """META cubes don't have a backing table; their source is a VALUES
    subquery the strategy builds from the catalog snapshot."""
    catalog = {c.name: c for c in META_CUBES}
    s = MetaDialect()
    out = render(s.emit_source(META_CUBES[0], catalog, lambda x: x), Dialect.META)
    # Should produce a SELECT-over-VALUES wrapped subquery aliased to
    # the META cube's alias.
    assert "VALUES" in out
    assert f"AS {META_CUBES[0].alias}" in out


# ---------------------------------------------------------------------------
# Protocol conformance — runtime_checkable
# ---------------------------------------------------------------------------


def test_strategies_satisfy_protocol_at_runtime() -> None:
    for s in (PostgresDialect(), ClickHouseDialect(), MetaDialect()):
        assert isinstance(s, DialectStrategy)


# ---------------------------------------------------------------------------
# dialect_for() — registry + DI hybrid
# ---------------------------------------------------------------------------


def test_dialect_for_returns_default_for_known_backend() -> None:
    assert isinstance(dialect_for(Dialect.POSTGRES), PostgresDialect)
    assert isinstance(dialect_for(Dialect.CLICKHOUSE), ClickHouseDialect)
    assert isinstance(dialect_for(Dialect.META), MetaDialect)


def test_dialect_for_accepts_overrides() -> None:
    fake = PostgresDialect()
    result = dialect_for(Dialect.POSTGRES, overrides={Dialect.POSTGRES: fake})
    assert result is fake


def test_dialect_for_duckdb_uses_dedicated_strategy() -> None:
    """DuckDB has its own strategy (``$name`` placeholders) — the
    Postgres-aliased default was a stopgap until the dedicated class
    landed."""
    from semql.backend import DuckDBDialect

    assert isinstance(dialect_for(Dialect.DUCKDB), DuckDBDialect)
