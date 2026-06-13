"""Tests for ``Cube.extends`` — LookML-style cube inheritance.

A child cube inherits the parent's measures / dimensions /
time_dimensions / segments by name; the child's own items
override the parent's of the same name and may add new ones.
Other cube-level settings (backend, table, alias, base_predicate,
tenancy, joins) are not inherited — they're cube-specific.

Inheritance is resolved once at ``Catalog`` construction. By the
time the compiler sees the catalog, each cube carries its full
merged field list.
"""

from __future__ import annotations

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery, TimeDimension


def _parent() -> Cube:
    return Cube(
        name="all_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )


# ---------------------------------------------------------------------------
# Model — extends field.
# ---------------------------------------------------------------------------


def test_cube_extends_defaults_to_none() -> None:
    cube = Cube(name="x", backend=Dialect.POSTGRES, table="x", alias="x")
    assert cube.extends is None


def test_cube_accepts_extends_field() -> None:
    cube = Cube(
        name="vip",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        base_predicate="{o}.vip = TRUE",
    )
    assert cube.extends == "all_orders"


# ---------------------------------------------------------------------------
# Catalog — flatten inheritance at construction time.
# ---------------------------------------------------------------------------


def test_child_inherits_parent_measures() -> None:
    child = Cube(
        name="vip_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        base_predicate="{o}.vip = TRUE",
    )
    cat = Catalog([_parent(), child])
    merged = cat.as_dict()["vip_orders"]
    assert {m.name for m in merged.measures} == {"count", "revenue"}


def test_child_inherits_parent_dimensions_and_time_dimensions() -> None:
    child = Cube(
        name="vip_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
    )
    cat = Catalog([_parent(), child])
    merged = cat.as_dict()["vip_orders"]
    assert {d.name for d in merged.dimensions} == {"region", "status"}
    assert {td.name for td in merged.time_dimensions} == {"created_at"}


def test_child_overrides_parent_measure_by_name() -> None:
    """A child redeclaring a same-name measure wins — typical case is
    swapping the SQL or agg for a filtered context."""
    child = Cube(
        name="vip_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        measures=[
            Measure(name="revenue", sql="{o}.vip_amount", agg="sum", description="VIP-only"),
        ],
    )
    cat = Catalog([_parent(), child])
    merged = cat.as_dict()["vip_orders"]
    revenue = next(m for m in merged.measures if m.name == "revenue")
    assert revenue.sql == "{o}.vip_amount"
    assert revenue.description == "VIP-only"
    # count is still inherited.
    assert any(m.name == "count" for m in merged.measures)


def test_child_can_add_new_measures() -> None:
    child = Cube(
        name="vip_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        measures=[Measure(name="vip_count", sql="*", agg="count")],
    )
    cat = Catalog([_parent(), child])
    merged = cat.as_dict()["vip_orders"]
    assert {m.name for m in merged.measures} == {"count", "revenue", "vip_count"}


def test_child_inherits_segments() -> None:
    from semql import Segment

    parent = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
    )
    child = Cube(
        name="vip",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="orders",
    )
    cat = Catalog([parent, child])
    assert {s.name for s in cat.as_dict()["vip"].segments} == {"paid"}


# ---------------------------------------------------------------------------
# Multi-level + cycle detection.
# ---------------------------------------------------------------------------


def test_extends_chain_flattens_grandparent_fields() -> None:
    grandparent = _parent()
    parent = Cube(
        name="paid_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        measures=[Measure(name="paid_count", sql="*", agg="count")],
    )
    child = Cube(
        name="vip_paid_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="paid_orders",
    )
    cat = Catalog([grandparent, parent, child])
    names = {m.name for m in cat.as_dict()["vip_paid_orders"].measures}
    assert names == {"count", "revenue", "paid_count"}


def test_extends_cycle_raises() -> None:
    a = Cube(name="a", backend=Dialect.POSTGRES, table="a", alias="a", extends="b")
    b = Cube(name="b", backend=Dialect.POSTGRES, table="b", alias="b", extends="a")
    with pytest.raises(ValueError, match=r"(?i)cycle|extends"):
        Catalog([a, b])


def test_unknown_extends_target_raises() -> None:
    bad = Cube(
        name="vip",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="ghost",
    )
    with pytest.raises(ValueError, match=r"(?i)extends|ghost"):
        Catalog([bad])


# ---------------------------------------------------------------------------
# Compile path — inherited measures resolve in queries against the child.
# ---------------------------------------------------------------------------


def test_compile_uses_inherited_measure() -> None:
    child = Cube(
        name="vip_orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        extends="all_orders",
        base_predicate="{o}.vip = TRUE",
    )
    cat = Catalog([_parent(), child])
    out = cat.compile(SemanticQuery(measures=["vip_orders.revenue"]))
    # Inherited measure compiles via the child cube's alias.
    assert "SUM(o.amount)" in out.sql
    # Child's base_predicate applies.
    assert "vip" in out.sql.lower()
