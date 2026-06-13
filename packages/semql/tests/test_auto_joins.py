"""Tests for auto-inferred joins from primary_key + foreign_key
declarations.

Catalog authors who already model entities (orders.customer_id →
customers.id) shouldn't have to repeat the join. Declaring
``Cube(primary_key="id")`` and ``Dimension(foreign_key="customers")``
gives the Catalog enough to derive the Join edge.

Explicit ``Join`` entries always win — auto-inference fills only
the gaps. The inferred relationship is ``many_to_one`` from the
foreign-key side (one customer, many orders).
"""

from __future__ import annotations

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, SemanticQuery


def _customers() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="uuid"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


def _orders_with_fk() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[
            Dimension(name="amount", sql="{o}.amount", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="uuid",
                foreign_key="customers",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Model shape — new fields default to None.
# ---------------------------------------------------------------------------


def test_cube_primary_key_defaults_to_none() -> None:
    cube = Cube(name="x", backend=Dialect.POSTGRES, table="x", alias="x")
    assert cube.primary_key is None


def test_dimension_foreign_key_defaults_to_none() -> None:
    d = Dimension(name="id", sql="{c}.id", type="uuid")
    assert d.foreign_key is None


def test_cube_accepts_primary_key() -> None:
    cube = _customers()
    assert cube.primary_key == "id"


def test_dimension_accepts_foreign_key() -> None:
    cube = _orders_with_fk()
    fk_dim = next(d for d in cube.dimensions if d.name == "customer_id")
    assert fk_dim.foreign_key == "customers"


# ---------------------------------------------------------------------------
# Catalog — auto-derives a Join from the FK declaration.
# ---------------------------------------------------------------------------


def test_catalog_auto_derives_join_from_foreign_key() -> None:
    cat = Catalog([_orders_with_fk(), _customers()])
    orders = cat.as_dict()["orders"]
    assert any(j.to == "customers" for j in orders.joins), (
        f"Expected auto-inferred join orders → customers; got {orders.joins}"
    )


def test_auto_derived_join_has_many_to_one_relationship() -> None:
    cat = Catalog([_orders_with_fk(), _customers()])
    j = next(j for j in cat.as_dict()["orders"].joins if j.to == "customers")
    assert j.relationship == "many_to_one"


def test_auto_derived_join_predicate_matches_fk_to_pk() -> None:
    cat = Catalog([_orders_with_fk(), _customers()])
    j = next(j for j in cat.as_dict()["orders"].joins if j.to == "customers")
    assert "{o}.customer_id" in j.on
    assert "{c}.id" in j.on


# ---------------------------------------------------------------------------
# Compile path — the auto-derived join works end-to-end.
# ---------------------------------------------------------------------------


def test_compile_can_join_via_auto_inferred_edge() -> None:
    cat = Catalog([_orders_with_fk(), _customers()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            dimensions=["customers.region"],
        )
    )
    assert "orders AS o" in out.sql
    assert "customers AS c" in out.sql
    assert "o.customer_id" in out.sql
    assert "c.id" in out.sql


# ---------------------------------------------------------------------------
# Explicit joins win over auto-inference.
# ---------------------------------------------------------------------------


def test_explicit_join_suppresses_auto_inference() -> None:
    """If the cube already declares a Join.to=customers, don't add a
    second auto-inferred edge — we'd end up with two edges to the
    same target and surprising join behaviour."""
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="uuid",
                foreign_key="customers",
            ),
        ],
        joins=[
            Join(
                to="customers",
                relationship="many_to_one",
                on="{o}.customer_id = {c}.id",
            ),
        ],
    )
    cat = Catalog([orders, _customers()])
    edges = [j for j in cat.as_dict()["orders"].joins if j.to == "customers"]
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# Validation — bad FK references fail at Catalog construction.
# ---------------------------------------------------------------------------


def test_unknown_foreign_key_target_raises_at_catalog_build() -> None:
    bad = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="uuid",
                foreign_key="ghost",
            ),
        ],
    )
    with pytest.raises(ValueError, match=r"(?i)foreign_key|ghost"):
        Catalog([bad])


def test_foreign_key_target_without_primary_key_raises() -> None:
    no_pk = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        # primary_key intentionally omitted.
        dimensions=[Dimension(name="id", sql="{c}.id", type="uuid")],
    )
    orders = _orders_with_fk()
    with pytest.raises(ValueError, match=r"(?i)primary_key"):
        Catalog([orders, no_pk])


def test_primary_key_must_be_a_dimension_on_the_cube() -> None:
    bad = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="not_a_dim",
        dimensions=[Dimension(name="id", sql="{c}.id", type="uuid")],
    )
    with pytest.raises(ValueError, match=r"(?i)primary_key|not_a_dim"):
        Catalog([bad])
