"""Tests for derived-table cubes — ``Cube(source=DerivedTable(sql=...))``.

A derived-table cube exposes a SQL expression as its physical source
(e.g. a layered CTE preamble producing derived columns) instead of a
real table. The compiler wraps the resolved SQL in a subquery aliased
to the cube's ``alias`` and threads tenancy / security_sql / scope
wrappers around it just like a plain-table cube.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Catalog,
    Cube,
    DerivedTable,
    Dialect,
    Dimension,
    Measure,
    NamedCTE,
    SemanticQuery,
)
from semql.model import PhysicalTable

# ---------------------------------------------------------------------------
# Model: Cube.source / Cube.table shorthand consistency
# ---------------------------------------------------------------------------


def test_cube_table_shorthand_resolves_to_tableref() -> None:
    cube = Cube(name="c", dialect=Dialect.POSTGRES, table="schema.t", alias="c")
    assert isinstance(cube.resolved_source, PhysicalTable)
    assert cube.resolved_source.table == "schema.t"


def test_cube_explicit_source_takes_effect() -> None:
    src = DerivedTable(sql="SELECT id, value FROM raw WHERE active")
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        source=src,
        alias="c",
        dimensions=[Dimension(name="id", sql="{c}.id", type="string")],
    )
    assert cube.resolved_source is src


def test_cube_rejects_neither_table_nor_source() -> None:
    with pytest.raises(ValidationError, match=r"(?i)must declare one source"):
        Cube(name="c", dialect=Dialect.POSTGRES, alias="c")


def test_cube_rejects_table_plus_derived_source() -> None:
    with pytest.raises(ValidationError, match=r"(?i)cannot set both"):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="foo",
            source=DerivedTable(sql="SELECT * FROM bar"),
            alias="c",
        )


def test_cube_accepts_redundant_tableref_when_equal() -> None:
    Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="schema.t",
        source=PhysicalTable(table="schema.t"),
        alias="c",
    )


def test_cube_rejects_conflicting_table_and_tableref() -> None:
    with pytest.raises(ValidationError, match=r"(?i)disagrees"):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="foo",
            source=PhysicalTable(table="bar"),
            alias="c",
        )


def test_cube_discriminator_rejects_tenant_schema_in_derived_sql() -> None:
    with pytest.raises(ValidationError, match=r"(?i)tenant_schema"):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            source=DerivedTable(sql="SELECT * FROM {tenant_schema}.raw"),
            alias="c",
            tenancy="discriminator",
            tenancy_columns=["tenant_id"],
        )


# ---------------------------------------------------------------------------
# Compile: DerivedTable cubes emit a subquery as the FROM source
# ---------------------------------------------------------------------------


def _make_derived_cube() -> Cube:
    """Cube whose source is a 2-CTE preamble surfacing derived columns
    (the zenai shape: layered CTEs handing the compiler `productive_time`
    + friends as dimensions / measures)."""
    return Cube(
        name="user_events",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            sql=(
                "SELECT user_id, "
                "CASE WHEN in_shift THEN duration ELSE 0 END AS active_time, "
                "tenant_schema_col "
                "FROM raw_events WHERE deleted_at IS NULL"
            )
        ),
        alias="ue",
        measures=[Measure(name="total_active", sql="{ue}.active_time", agg="sum")],
        dimensions=[Dimension(name="user_id", sql="{ue}.user_id", type="string")],
    )


def test_derived_cube_compiles_with_subquery_from() -> None:
    cat = Catalog([_make_derived_cube()])
    q = SemanticQuery(measures=["user_events.total_active"], dimensions=["user_events.user_id"])
    c = cat.compile(q)
    # Aliased subquery, not a plain Table reference.
    assert "FROM (SELECT" in c.sql
    assert ") AS ue" in c.sql
    assert "SUM(ue.active_time)" in c.sql


def test_compiled_surfaces_derived_sources() -> None:
    cat = Catalog([_make_derived_cube()])
    q = SemanticQuery(measures=["user_events.total_active"], dimensions=["user_events.user_id"])
    c = cat.compile(q)
    assert len(c.derived_sources) == 1
    assert "raw_events" in c.derived_sources[0]
    # Order matches touched_cube_names: derived cube is the only entry.
    assert c.touched_cube_names == ["user_events"]


def test_plain_cube_has_empty_derived_sources() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    c = cat.compile(q)
    assert c.derived_sources == []


# ---------------------------------------------------------------------------
# Tenancy + derived-table interactions
# ---------------------------------------------------------------------------


def test_derived_cube_resolves_tenant_schema_placeholder() -> None:
    cube = Cube(
        name="rollup",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(sql="SELECT * FROM {tenant_schema}.raw_rollup"),
        alias="r",
        measures=[Measure(name="cnt", sql="*", agg="count")],
        dimensions=[Dimension(name="bucket", sql="{r}.bucket", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["rollup.cnt"], dimensions=["rollup.bucket"])
    c = cat.compile(q, context={"tenant_schema": "tenant42"})
    assert "tenant42.raw_rollup" in c.sql
    assert "{tenant_schema}" not in c.sql
    # derived_sources channel sees the substituted SQL too.
    assert "tenant42.raw_rollup" in c.derived_sources[0]


def test_discriminator_wraps_derived_source() -> None:
    """``tenancy="discriminator"`` must AND a ``tenant_col = $tenant``
    predicate INSIDE the alias subquery — even for derived sources —
    so a malformed outer ``OR`` can't bypass it."""
    cube = Cube(
        name="events",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(sql="SELECT id, tenant_id, kind FROM events_raw"),
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["tenant_id"],
        measures=[Measure(name="n", sql="*", agg="count")],
        dimensions=[Dimension(name="kind", sql="{e}.kind", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["events.n"], dimensions=["events.kind"])
    c = cat.compile(q, context={"tenant": "tenant42"})
    # The derived source got wrapped: outer alias `e` wraps a SELECT *
    # FROM (<derived>) WHERE e.tenant_id = $p.
    assert "e.tenant_id =" in c.sql
    assert "tenant42" in c.params.values()


def test_security_sql_wraps_derived_source() -> None:
    cube = Cube(
        name="rows",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(sql="SELECT id, owner_id, value FROM raw_rows"),
        alias="r",
        security_sql="{r}.owner_id = {ctx.viewer_id}",
        measures=[Measure(name="total", sql="{r}.value", agg="sum")],
        dimensions=[Dimension(name="id", sql="{r}.id", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["rows.total"], dimensions=["rows.id"])
    c = cat.compile(q, context={"ctx.viewer_id": "u99"})
    assert "r.owner_id" in c.sql
    assert "u99" in c.params.values()


# ---------------------------------------------------------------------------
# Joins: plain-table → derived-table cube
# ---------------------------------------------------------------------------


def test_join_from_plain_cube_to_derived_cube() -> None:
    facts = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="user_id", sql="{o}.user_id", type="string")],
        joins=[
            {  # type: ignore[list-item]
                "to": "user_segments",
                "relationship": "many_to_one",
                "on": "{o}.user_id = {us}.user_id",
            }
        ],
    )
    segments = Cube(
        name="user_segments",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            sql=(
                "SELECT user_id, "
                "CASE WHEN spend > 1000 THEN 'whale' ELSE 'minnow' END AS tier "
                "FROM spend_rollup"
            )
        ),
        alias="us",
        dimensions=[
            Dimension(name="user_id", sql="{us}.user_id", type="string"),
            Dimension(name="tier", sql="{us}.tier", type="string"),
        ],
    )
    cat = Catalog([facts, segments])
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["user_segments.tier"])
    c = cat.compile(q)
    assert ") AS us" in c.sql
    assert "spend_rollup" in c.sql
    assert "o.user_id = us.user_id" in c.sql
    # derived_sources surfaces the joined cube's derived SQL.
    assert len(c.derived_sources) == 1
    assert "spend_rollup" in c.derived_sources[0]


# ---------------------------------------------------------------------------
# DerivedTable.with_ctes — hoisted preamble
# ---------------------------------------------------------------------------


def _layered_cube() -> Cube:
    """A two-CTE preamble feeding the main cube body — the zenai shape."""
    return Cube(
        name="user_activity",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            with_ctes=[
                NamedCTE(
                    name="raw_active",
                    sql="SELECT user_id, duration FROM events WHERE in_shift",
                ),
                NamedCTE(
                    name="bucketed",
                    sql="SELECT user_id, SUM(duration) AS total FROM raw_active GROUP BY user_id",
                ),
            ],
            sql="SELECT user_id, total AS active_time FROM bucketed",
        ),
        alias="ua",
        measures=[Measure(name="total_active", sql="{ua}.active_time", agg="sum")],
        dimensions=[Dimension(name="user_id", sql="{ua}.user_id", type="string")],
    )


def test_with_ctes_emits_outer_with_clause() -> None:
    cat = Catalog([_layered_cube()])
    q = SemanticQuery(measures=["user_activity.total_active"], dimensions=["user_activity.user_id"])
    c = cat.compile(q)
    # WITH clause sits at the top; both CTEs appear before the SELECT.
    assert c.sql.startswith("WITH ")
    assert "raw_active AS" in c.sql
    assert "bucketed AS" in c.sql
    # Main FROM still wraps the body sql aliased to the cube alias.
    assert ") AS ua" in c.sql
    # CTE order is preserved (raw_active references events; bucketed
    # references raw_active — emit the prerequisite first).
    assert c.sql.index("raw_active AS") < c.sql.index("bucketed AS")


def test_with_ctes_appear_in_derived_sources() -> None:
    cat = Catalog([_layered_cube()])
    q = SemanticQuery(measures=["user_activity.total_active"], dimensions=["user_activity.user_id"])
    c = cat.compile(q)
    # Three entries: cte1, cte2, then the main sql.
    assert len(c.derived_sources) == 3
    assert "raw_active" in c.derived_sources[0] or "events" in c.derived_sources[0]
    assert "bucketed" in c.derived_sources[1] or "raw_active" in c.derived_sources[1]
    assert "active_time" in c.derived_sources[2]


def test_with_ctes_rejects_duplicate_names_within_cube() -> None:
    with pytest.raises(ValidationError, match=r"(?i)duplicate CTE name"):
        DerivedTable(
            with_ctes=[
                NamedCTE(name="a", sql="SELECT 1"),
                NamedCTE(name="a", sql="SELECT 2"),
            ],
            sql="SELECT * FROM a",
        )


def test_with_ctes_rejects_cross_cube_name_collision() -> None:
    c1 = Cube(
        name="cube_a",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            with_ctes=[NamedCTE(name="shared", sql="SELECT 1 AS x")],
            sql="SELECT * FROM shared",
        ),
        alias="a",
        dimensions=[Dimension(name="x", sql="{a}.x", type="string")],
    )
    c2 = Cube(
        name="cube_b",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            with_ctes=[NamedCTE(name="shared", sql="SELECT 2 AS y")],
            sql="SELECT * FROM shared",
        ),
        alias="b",
        dimensions=[Dimension(name="y", sql="{b}.y", type="string")],
    )
    with pytest.raises(ValueError, match=r"(?i)CTE name 'shared'.*collides"):
        Catalog([c1, c2])


def test_with_ctes_resolves_tenant_schema_placeholder() -> None:
    cube = Cube(
        name="rollup",
        dialect=Dialect.POSTGRES,
        source=DerivedTable(
            with_ctes=[
                NamedCTE(
                    name="raw_rows",
                    sql="SELECT * FROM {tenant_schema}.raw_table",
                )
            ],
            sql="SELECT * FROM raw_rows",
        ),
        alias="r",
        measures=[Measure(name="cnt", sql="*", agg="count")],
        dimensions=[Dimension(name="bucket", sql="{r}.bucket", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["rollup.cnt"], dimensions=["rollup.bucket"])
    c = cat.compile(q, context={"tenant_schema": "tenant42"})
    assert "tenant42.raw_table" in c.sql
    assert "{tenant_schema}" not in c.sql
    # CTE body is also in derived_sources with the substituted schema.
    assert any("tenant42.raw_table" in src for src in c.derived_sources)


def test_with_ctes_rejected_with_tenant_schema_under_discriminator() -> None:
    with pytest.raises(ValidationError, match=r"(?i)tenant_schema"):
        Cube(
            name="bad",
            dialect=Dialect.POSTGRES,
            source=DerivedTable(
                with_ctes=[NamedCTE(name="raw", sql="SELECT * FROM {tenant_schema}.raw")],
                sql="SELECT * FROM raw",
            ),
            alias="b",
            tenancy="discriminator",
            tenancy_columns=["tenant_id"],
        )


def test_plain_cube_emits_no_with_clause() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    c = cat.compile(q)
    assert not c.sql.startswith("WITH ")


def test_with_ctes_join_path_plain_to_layered() -> None:
    """Joining a plain-table cube against a layered-CTE cube emits ONE
    outer WITH followed by the join. Both cubes' sources are inside the
    same FROM ... JOIN clause."""
    facts = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="user_id", sql="{o}.user_id", type="string")],
        joins=[
            {  # type: ignore[list-item]
                "to": "user_activity",
                "relationship": "many_to_one",
                "on": "{o}.user_id = {ua}.user_id",
            }
        ],
    )
    cat = Catalog([facts, _layered_cube()])
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["user_activity.user_id"])
    c = cat.compile(q)
    assert c.sql.startswith("WITH ")
    assert "raw_active AS" in c.sql
    assert "bucketed AS" in c.sql
    assert "o.user_id = ua.user_id" in c.sql
