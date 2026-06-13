"""Tests for the Python-side catalog-walking primitives in
``semql.introspect``: ``iter_cubes``, ``iter_fields``, ``iter_joins``,
``resolve_field``. These primitives are the seam every downstream
tool (prompt, MCP, ERD, validate-db) calls into — pinning their
contracts here keeps the consumers from each reinventing
``only_exposed`` and ``include_meta`` filtering.
"""

from __future__ import annotations

import pytest
from semql.errors import ResolveError
from semql.introspect import (
    META_CUBES,
    ResolvedQuery,
    iter_cubes,
    iter_fields,
    iter_joins,
    resolve_field,
    resolve_query,
)
from semql.model import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    Segment,
    TimeDimension,
    View,
)
from semql.spec import SemanticQuery, TimeWindow


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id")],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        expose_in_prompt=False,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in [_orders(), _customers(), *META_CUBES]}


# ---------------------------------------------------------------------------
# iter_cubes
# ---------------------------------------------------------------------------


def test_iter_cubes_excludes_meta_by_default() -> None:
    names = [c.name for c in iter_cubes(_catalog())]
    assert "orders" in names
    assert "customers" in names
    assert "catalog_cubes" not in names


def test_iter_cubes_include_meta_yields_them() -> None:
    names = [c.name for c in iter_cubes(_catalog(), include_meta=True)]
    assert "catalog_cubes" in names
    assert "catalog_measures" in names


def test_iter_cubes_only_exposed_skips_hidden() -> None:
    names = [c.name for c in iter_cubes(_catalog(), only_exposed=True)]
    assert "orders" in names
    assert "customers" not in names  # expose_in_prompt=False on _customers


def test_iter_cubes_accepts_iterable_of_cubes() -> None:
    """Callers with a plain list shouldn't have to build a dict."""
    cubes = list(iter_cubes([_orders(), _customers()]))
    assert {c.name for c in cubes} == {"orders", "customers"}


# ---------------------------------------------------------------------------
# iter_fields
# ---------------------------------------------------------------------------


def test_iter_fields_yields_all_addressable_fields_in_order() -> None:
    fields = list(iter_fields(_orders()))
    names = [f.name for f in fields]
    # Order: measures, dimensions, time_dimensions, segments
    assert names == ["revenue", "region", "created_at", "paid"]


def test_iter_fields_preserves_concrete_types() -> None:
    """``isinstance`` narrowing must still work — the BaseField supertype
    is structural, not a discriminator."""
    by_name = {f.name: f for f in iter_fields(_orders())}
    assert isinstance(by_name["revenue"], Measure)
    assert isinstance(by_name["region"], Dimension)
    assert isinstance(by_name["created_at"], TimeDimension)
    assert isinstance(by_name["paid"], Segment)


# ---------------------------------------------------------------------------
# iter_joins
# ---------------------------------------------------------------------------


def test_iter_joins_yields_source_edge_target_triples() -> None:
    triples = list(iter_joins(_catalog()))
    assert len(triples) == 1
    src, edge, tgt = triples[0]
    assert src.name == "orders"
    assert tgt.name == "customers"
    assert edge.relationship == "many_to_one"


def test_iter_joins_excludes_meta_cubes_by_default() -> None:
    """META cubes have no joins so this is a structural check — the
    triple iteration shouldn't walk them at all."""
    triples = list(iter_joins(_catalog()))
    assert all(src.backend is not Dialect.META for src, _, _ in triples)
    assert all(tgt.backend is not Dialect.META for _, _, tgt in triples)


# ---------------------------------------------------------------------------
# resolve_field
# ---------------------------------------------------------------------------


def test_resolve_field_returns_cube_and_field() -> None:
    cube, fld = resolve_field("orders.revenue", _catalog())
    assert cube.name == "orders"
    assert fld.name == "revenue"
    assert isinstance(fld, Measure)


def test_resolve_field_handles_time_dimension() -> None:
    _cube, fld = resolve_field("orders.created_at", _catalog())
    assert isinstance(fld, TimeDimension)


def test_resolve_field_unknown_raises() -> None:
    with pytest.raises(ResolveError):
        resolve_field("orders.no_such_field", _catalog())


def test_resolve_field_with_view_carries_local_name() -> None:
    """When a view exposes a renamed field, the resolver returns the
    underlying Cube + a Field whose ``name`` matches the view's local
    alias — so callers building SELECT output columns get the user-
    facing name without extra plumbing."""
    view = View(name="rev", fields={"net": "orders.revenue"})
    cube, fld = resolve_field("rev.net", _catalog(), views={"rev": view})
    assert cube.name == "orders"
    assert fld.name == "net"  # renamed via the view


def test_resolve_field_with_view_unknown_local_raises() -> None:
    view = View(name="rev", fields={"net": "orders.revenue"})
    with pytest.raises(ResolveError):
        resolve_field("rev.not_in_view", _catalog(), views={"rev": view})


# ---------------------------------------------------------------------------
# resolve_query
# ---------------------------------------------------------------------------


def test_resolve_query_fills_typed_buckets() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2024-01-01", "2024-02-01"),
        ),
    )
    resolved = resolve_query(q, _catalog())
    assert isinstance(resolved, ResolvedQuery)
    assert [m.name for _, m in resolved.measures] == ["revenue"]
    assert [d.name for _, d in resolved.dimensions] == ["region"]
    assert resolved.time_dimension is not None
    _, td = resolved.time_dimension
    assert td.name == "created_at"


def test_resolve_query_touched_cubes_dedups_in_order() -> None:
    """Same cube referenced by multiple fields must only appear once
    in ``touched_cubes`` — that's what the compiler / visualize layer
    use to build the FROM/JOIN graph."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    resolved = resolve_query(q, _catalog())
    assert [c.name for c in resolved.touched_cubes] == ["orders"]


def test_resolve_query_wrong_kind_raises_value_error() -> None:
    q = SemanticQuery(dimensions=["orders.revenue"])  # revenue is a Measure
    with pytest.raises(ValueError, match="non-Dimension"):
        resolve_query(q, _catalog())


def test_resolve_query_supports_views() -> None:
    view = View(name="rev", fields={"net": "orders.revenue"})
    q = SemanticQuery(measures=["rev.net"])
    resolved = resolve_query(q, _catalog(), views={"rev": view})
    cube, m = resolved.measures[0]
    assert cube.name == "orders"
    assert m.name == "net"  # view local name carried through
