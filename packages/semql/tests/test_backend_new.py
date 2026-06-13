# pyright: reportPrivateImportUsage=false
"""Dialect strategies for BigQuery, Snowflake, and DuckDB.

Each emits the same sqlglot AST shape as the Postgres strategy
(``date_trunc`` via ``exp.Anonymous``, ``ILIKE`` with ``%value%``
binding, plain aliased table source) — sqlglot's dialect renderer
takes care of the per-backend differences (``@p0`` for BigQuery,
``:p0`` for Snowflake, ``$p0`` for DuckDB; ``ILIKE`` → ``LOWER LIKE``
for BigQuery).

These tests pin the rendered SQL so a future sqlglot upgrade can't
silently change the wire format under us.
"""

from __future__ import annotations

from typing import Any

import pytest
from semql.backend import (
    BigQueryDialect,
    DialectStrategy,
    DuckDBDialect,
    SnowflakeDialect,
    SqlResolver,
    dialect_for,
    render,
)
from semql.model import Cube, Dialect, Dimension, Measure
from sqlglot import exp


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,  # backend on cube doesn't affect the strategy under test
        table="public.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


# ---------------------------------------------------------------------------
# Placeholder shape — sqlglot's stock renderer picks the right syntax.
# ---------------------------------------------------------------------------


def test_bigquery_placeholder_renders_at_prefix() -> None:
    s = BigQueryDialect()
    assert render(s.placeholder("p0", "string"), Dialect.BIGQUERY) == "@p0"


def test_snowflake_placeholder_renders_colon_prefix() -> None:
    s = SnowflakeDialect()
    assert render(s.placeholder("p0", "string"), Dialect.SNOWFLAKE) == ":p0"


def test_duckdb_placeholder_renders_dollar_prefix() -> None:
    s = DuckDBDialect()
    assert render(s.placeholder("p0", "string"), Dialect.DUCKDB) == "$p0"


# ---------------------------------------------------------------------------
# trunc() — same date_trunc shape everywhere; dialect picks output.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy_cls", "backend"),
    [
        (BigQueryDialect, Dialect.BIGQUERY),
        (SnowflakeDialect, Dialect.SNOWFLAKE),
        (DuckDBDialect, Dialect.DUCKDB),
    ],
)
def test_trunc_emits_date_trunc(strategy_cls: type, backend: Dialect) -> None:
    s = strategy_cls()
    out = render(s.trunc("day", exp.column("ts", table="x")), backend)
    assert "date_trunc" in out.lower()
    assert "day" in out.lower()


# ---------------------------------------------------------------------------
# emit_contains() — % wildcards baked into bound value, ILIKE on AST.
# BigQuery transpiles ILIKE to LOWER LIKE LOWER; the bound %v% still
# applies (LIKE understands %).
# ---------------------------------------------------------------------------


def test_bigquery_emit_contains_transpiles_ilike_to_lower_like() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        bound.append((value, dim_type))
        return exp.Placeholder(this=f"p{len(bound) - 1}")

    s = BigQueryDialect()
    node = s.emit_contains(exp.column("email", table="x"), "@acme.com", bind)
    out = render(node, Dialect.BIGQUERY)
    # BigQuery has no native ILIKE; sqlglot rewrites to LOWER LIKE LOWER.
    assert "LOWER" in out and "LIKE" in out
    assert bound == [("%@acme.com%", "string")]


def test_snowflake_emit_contains_uses_native_ilike() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        bound.append((value, dim_type))
        return exp.Placeholder(this=f"p{len(bound) - 1}")

    s = SnowflakeDialect()
    node = s.emit_contains(exp.column("email", table="x"), "@acme.com", bind)
    out = render(node, Dialect.SNOWFLAKE)
    assert out == "x.email ILIKE :p0"
    assert bound == [("%@acme.com%", "string")]


def test_duckdb_emit_contains_uses_native_ilike() -> None:
    bound: list[tuple[Any, str]] = []

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        bound.append((value, dim_type))
        return exp.Placeholder(this=f"p{len(bound) - 1}")

    s = DuckDBDialect()
    node = s.emit_contains(exp.column("email", table="x"), "@acme.com", bind)
    out = render(node, Dialect.DUCKDB)
    assert out == "x.email ILIKE $p0"
    assert bound == [("%@acme.com%", "string")]


# ---------------------------------------------------------------------------
# emit_source() — aliased table for vanilla cubes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy_cls", "backend"),
    [
        (BigQueryDialect, Dialect.BIGQUERY),
        (SnowflakeDialect, Dialect.SNOWFLAKE),
        (DuckDBDialect, Dialect.DUCKDB),
    ],
)
def test_emit_source_renders_aliased_table(strategy_cls: type, backend: Dialect) -> None:
    cube = _orders()
    identity: SqlResolver = lambda x: x  # noqa: E731 -- inline identity resolver; named def is overkill for a one-line test fixture
    out = render(strategy_cls().emit_source(cube, {"orders": cube}, identity), backend)
    # All three dialects emit ``public.orders AS o`` (possibly quoted on BQ).
    assert "orders" in out and "o" in out


# ---------------------------------------------------------------------------
# Protocol conformance + registry default.
# ---------------------------------------------------------------------------


def test_new_strategies_satisfy_protocol() -> None:
    for s in (BigQueryDialect(), SnowflakeDialect(), DuckDBDialect()):
        assert isinstance(s, DialectStrategy)


def test_dialect_for_returns_new_defaults() -> None:
    assert isinstance(dialect_for(Dialect.BIGQUERY), BigQueryDialect)
    assert isinstance(dialect_for(Dialect.SNOWFLAKE), SnowflakeDialect)
    assert isinstance(dialect_for(Dialect.DUCKDB), DuckDBDialect)


# ---------------------------------------------------------------------------
# End-to-end compile — each new backend produces dialect-correct SQL.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected_placeholder"),
    [
        (Dialect.BIGQUERY, "@p0"),
        (Dialect.SNOWFLAKE, ":p0"),
        (Dialect.DUCKDB, "$p0"),
    ],
)
def test_compile_against_new_backend_uses_dialect_placeholder(
    backend: Dialect, expected_placeholder: str
) -> None:
    from semql import Catalog, Filter, SemanticQuery

    cube = Cube(
        name="orders",
        backend=backend,
        table="public.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    out = Catalog([cube]).compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
        )
    )
    assert out.backend is backend
    assert expected_placeholder in out.sql
    assert out.params == {"p0": "us"}
