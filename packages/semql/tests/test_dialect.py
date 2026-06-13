"""Tests for the sqlglot dialect module.

Commit 1 of the sqlglot migration is *additive*: the module exists,
``compile_query`` does not yet use it. These tests pin the dialect's
shape so the eventual switch in Commit 2 is a no-behaviour-change diff.
"""

from __future__ import annotations

import pytest
import sqlglot
from semql.dialect import dialect_for, placeholder_for
from semql.introspect import META_CUBES
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension
from sqlglot import exp

# ---------------------------------------------------------------------------
# dialect_for() — string the sqlglot parser/renderer wants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (Dialect.POSTGRES, "postgres"),
        (Dialect.CLICKHOUSE, "clickhouse"),
        (Dialect.DUCKDB, "duckdb"),
        (Dialect.BIGQUERY, "bigquery"),
        (Dialect.SNOWFLAKE, "snowflake"),
        # META cubes are rendered as portable VALUES literals — pick a
        # neutral dialect rather than invent a new one.
        (Dialect.META, "postgres"),
    ],
)
def test_dialect_for_returns_canonical_string(backend: Dialect, expected: str) -> None:
    assert dialect_for(backend) == expected


# ---------------------------------------------------------------------------
# placeholder_for() — preserves the existing dialect conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim_type", ["string", "number", "time", "bool", "uuid"])
def test_postgres_placeholder_matches_current_convention(dim_type: str) -> None:
    p = placeholder_for("p0", dim_type, Dialect.POSTGRES)
    assert isinstance(p, exp.Placeholder)
    assert p.sql(dialect="postgres") == "%(p0)s"


def test_clickhouse_placeholder_typed_suffix() -> None:
    assert (
        placeholder_for("p0", "string", Dialect.CLICKHOUSE).sql(dialect="clickhouse")
        == "{p0:String}"
    )
    assert (
        placeholder_for("p1", "number", Dialect.CLICKHOUSE).sql(dialect="clickhouse")
        == "{p1:Float64}"
    )
    assert (
        placeholder_for("p2", "time", Dialect.CLICKHOUSE).sql(dialect="clickhouse")
        == "{p2:DateTime}"
    )
    assert (
        placeholder_for("p3", "bool", Dialect.CLICKHOUSE).sql(dialect="clickhouse") == "{p3:UInt8}"
    )
    assert (
        placeholder_for("p4", "uuid", Dialect.CLICKHOUSE).sql(dialect="clickhouse") == "{p4:String}"
    )


# ---------------------------------------------------------------------------
# Round-trip parse — the canary that catches anything sqlglot's parser
# can't handle from our catalog SQL conventions.
# ---------------------------------------------------------------------------


def _resolve_braces(s: str, alias: str) -> str:
    """Substitute the cube's alias for any ``{...}`` placeholder so the
    fragment is parseable SQL. The point is the parser, not the resolver."""
    out = s
    for tok in ("{o}", "{c}", "{p}", "{s}", "{r}", "{u}", "{cc}", "{cm}", "{cd}"):
        out = out.replace(tok, alias)
    return out


def _make_test_cubes() -> list[Cube]:
    """A mirror of conftest's user cubes — copied here so this test
    module stands alone."""
    return [
        Cube(
            name="orders",
            backend=Dialect.POSTGRES,
            table="orders",
            alias="o",
            measures=[
                Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
                Measure(name="count", sql="*", agg="count", unit="count"),
            ],
            dimensions=[
                Dimension(name="region", sql="{o}.region", type="string"),
                Dimension(name="amount", sql="{o}.amount", type="number"),
            ],
            time_dimensions=[
                TimeDimension(name="created_at", sql="{o}.created_at"),
            ],
        ),
        Cube(
            name="sessions",
            backend=Dialect.CLICKHOUSE,
            table="sessions",
            alias="s",
            measures=[
                Measure(name="duration", sql="{s}.duration_sec", agg="sum", unit="duration"),
            ],
            dimensions=[Dimension(name="app_name", sql="{s}.app_name", type="string")],
        ),
    ]


@pytest.mark.parametrize("cube", _make_test_cubes())
def test_catalog_fragments_round_trip_through_sqlglot(cube: Cube) -> None:
    """Parse → render → parse every dim / measure / time_dim SQL fragment
    under the cube's declared backend. Failures here mean a future
    ``compile_query`` switch onto the sqlglot AST path can't form the
    same SQL we emit today."""
    dialect = dialect_for(cube.backend)
    fragments = (
        [d.sql for d in cube.dimensions]
        + [m.sql for m in cube.measures if m.sql != "*"]
        + [td.sql for td in cube.time_dimensions]
    )
    for raw in fragments:
        resolved = _resolve_braces(raw, cube.alias)
        first = sqlglot.parse_one(resolved, dialect=dialect)
        rendered = first.sql(dialect=dialect)
        # If sqlglot can parse + render the catalog's SQL once, it must
        # parse the rendered output too — otherwise the migration would
        # silently turn parseable SQL into unparseable SQL.
        sqlglot.parse_one(rendered, dialect=dialect)


def test_meta_cube_fragments_parse_under_postgres() -> None:
    """META cubes use Postgres VALUES syntax; ensure sqlglot's PG parser
    handles their dim SQL fragments."""
    for cube in META_CUBES:
        for d in cube.dimensions:
            resolved = _resolve_braces(d.sql, cube.alias)
            sqlglot.parse_one(resolved, dialect="postgres")
