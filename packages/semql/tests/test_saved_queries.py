"""Tests for ``SavedQuery`` model + ``Catalog.saved_queries`` wiring (S4).

The model is small (a name, a SemanticQuery, optional metadata) so
most of the surface to test is the Catalog's construction-time
validation — duplicate names, namespace collisions with cubes/views,
shape constraints on the tool-name slot.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SavedQuery,
    SemanticQuery,
)
from semql.model import View


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _saved_paid_revenue() -> SavedQuery:
    return SavedQuery(
        name="paid_revenue_by_region",
        description="Revenue from paid orders, broken down by region.",
        query=SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
    )


# ---------------------------------------------------------------------------
# Model + Catalog wiring
# ---------------------------------------------------------------------------


def test_catalog_exposes_saved_queries() -> None:
    sq = _saved_paid_revenue()
    cat = Catalog([_orders_cube()], saved_queries=[sq])
    assert cat.saved_queries == {sq.name: sq}


def test_catalog_with_no_saved_queries_has_empty_dict() -> None:
    cat = Catalog([_orders_cube()])
    assert cat.saved_queries == {}


def test_saved_query_can_be_compiled() -> None:
    """Round-trip: register a saved query, then compile its underlying
    SemanticQuery against the catalog the way the MCP tool will."""
    sq = _saved_paid_revenue()
    cat = Catalog([_orders_cube()], saved_queries=[sq])
    c = cat.compile(sq.query)
    assert "SUM" in c.sql.upper()
    assert "region" in c.columns


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_duplicate_saved_query_names_rejected() -> None:
    sq = _saved_paid_revenue()
    with pytest.raises(ValueError, match=r"(?i)duplicate SavedQuery"):
        Catalog([_orders_cube()], saved_queries=[sq, sq])


def test_saved_query_name_collides_with_cube_name() -> None:
    """The name slot becomes part of an MCP tool name — sharing it
    with a cube name would mean two different shapes register tools
    under colliding identifiers."""
    sq = SavedQuery(
        name="orders",  # same as the cube
        query=SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
    )
    with pytest.raises(ValueError, match=r"(?i)collides with a cube"):
        Catalog([_orders_cube()], saved_queries=[sq])


def test_saved_query_name_collides_with_view_name() -> None:
    view = View(name="paid_view", fields={"revenue": "orders.revenue"})
    sq = SavedQuery(
        name="paid_view",  # same as the view
        query=SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
    )
    with pytest.raises(ValueError, match=r"(?i)collides with a cube"):
        Catalog([_orders_cube()], views=[view], saved_queries=[sq])


def test_saved_query_name_must_be_non_empty() -> None:
    sq = SavedQuery(
        name="",
        query=SemanticQuery(measures=["orders.revenue"]),
    )
    with pytest.raises(ValueError, match=r"(?i)invalid name"):
        Catalog([_orders_cube()], saved_queries=[sq])


def test_saved_query_name_must_not_contain_dot() -> None:
    """Names become MCP tool names; dots collide with the qualified
    ``cube.field`` reference syntax."""
    sq = SavedQuery(
        name="orders.paid_revenue",
        query=SemanticQuery(measures=["orders.revenue"]),
    )
    with pytest.raises(ValueError, match=r"(?i)invalid name"):
        Catalog([_orders_cube()], saved_queries=[sq])


def test_saved_query_name_must_not_contain_space() -> None:
    sq = SavedQuery(
        name="paid revenue",
        query=SemanticQuery(measures=["orders.revenue"]),
    )
    with pytest.raises(ValueError, match=r"(?i)invalid name"):
        Catalog([_orders_cube()], saved_queries=[sq])


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------


def test_saved_query_defaults_are_empty() -> None:
    sq = SavedQuery(
        name="example",
        query=SemanticQuery(measures=["orders.revenue"]),
    )
    assert sq.description == ""
    assert sq.owner is None
    assert sq.required_roles == []


def test_saved_query_carries_required_roles() -> None:
    sq = SavedQuery(
        name="admin_only",
        query=SemanticQuery(measures=["orders.revenue"]),
        required_roles=["admin"],
    )
    cat = Catalog([_orders_cube()], saved_queries=[sq])
    assert cat.saved_queries["admin_only"].required_roles == ["admin"]
