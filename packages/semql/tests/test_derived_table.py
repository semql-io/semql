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
    Backend,
    Catalog,
    Cube,
    DerivedTable,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.model import TableRef

# ---------------------------------------------------------------------------
# Model: Cube.source / Cube.table shorthand consistency
# ---------------------------------------------------------------------------


def test_cube_table_shorthand_resolves_to_tableref() -> None:
    cube = Cube(name="c", backend=Backend.POSTGRES, table="schema.t", alias="c")
    assert isinstance(cube.resolved_source, TableRef)
    assert cube.resolved_source.table == "schema.t"


def test_cube_explicit_source_takes_effect() -> None:
    src = DerivedTable(sql="SELECT id, value FROM raw WHERE active")
    cube = Cube(
        name="c",
        backend=Backend.POSTGRES,
        source=src,
        alias="c",
        dimensions=[Dimension(name="id", sql="{c}.id", type="string")],
    )
    assert cube.resolved_source is src


def test_cube_rejects_neither_table_nor_source() -> None:
    with pytest.raises(ValidationError, match=r"(?i)either ``table=``"):
        Cube(name="c", backend=Backend.POSTGRES, alias="c")


def test_cube_rejects_table_plus_derived_source() -> None:
    with pytest.raises(ValidationError, match=r"(?i)cannot set both"):
        Cube(
            name="c",
            backend=Backend.POSTGRES,
            table="foo",
            source=DerivedTable(sql="SELECT * FROM bar"),
            alias="c",
        )


def test_cube_accepts_redundant_tableref_when_equal() -> None:
    Cube(
        name="c",
        backend=Backend.POSTGRES,
        table="schema.t",
        source=TableRef(table="schema.t"),
        alias="c",
    )


def test_cube_rejects_conflicting_table_and_tableref() -> None:
    with pytest.raises(ValidationError, match=r"(?i)disagrees"):
        Cube(
            name="c",
            backend=Backend.POSTGRES,
            table="foo",
            source=TableRef(table="bar"),
            alias="c",
        )


def test_cube_discriminator_rejects_tenant_schema_in_derived_sql() -> None:
    with pytest.raises(ValidationError, match=r"(?i)tenant_schema"):
        Cube(
            name="c",
            backend=Backend.POSTGRES,
            source=DerivedTable(sql="SELECT * FROM {tenant_schema}.raw"),
            alias="c",
            tenancy="discriminator",
            tenancy_column="tenant_id",
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
        backend=Backend.POSTGRES,
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
        backend=Backend.POSTGRES,
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
        backend=Backend.POSTGRES,
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
        backend=Backend.POSTGRES,
        source=DerivedTable(sql="SELECT id, tenant_id, kind FROM events_raw"),
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
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
        backend=Backend.POSTGRES,
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
        backend=Backend.POSTGRES,
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
        backend=Backend.POSTGRES,
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
