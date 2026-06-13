# pyright: reportPrivateUsage=false
# (Deliberately unit-tests the private ``_edge_weight`` fan-out cost model.)
"""Weighted, bidirectional Dijkstra join-path resolution (ktx M2 / C6).

``find_join_path`` replaced an unweighted, forward-only BFS. The three
properties that matter:

1. **Bidirectional reachability** — a catalog join declared on one cube
   resolves a path from either side, so a child-cube measure can be
   aggregated by a parent-cube dimension even though only the parent (or
   only the child) declares the edge.
2. **Fan-out weighting** — ``one_to_many`` traversals cost 10×, so when
   two paths connect the cubes the planner prefers the one that doesn't
   multiply rows, even if it has more hops.
3. **Transit prohibition** — a LEFT-joined cube may be a path *endpoint*
   but never a *through* node (the chasm-trap guard); see
   ``test_left_joins.py`` for the end-to-end refusal.
"""

from __future__ import annotations

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, SemanticQuery
from semql.errors import JoinPathError
from semql.logical import _edge_weight, find_join_path


def _cube(name: str, alias: str, *, joins: list[Join] | None = None) -> Cube:
    return Cube(
        name=name,
        dialect=Dialect.POSTGRES,
        table=name,
        alias=alias,
        measures=[Measure(name="n", sql="*", agg="count")],
        dimensions=[Dimension(name="k", sql=f"{{{alias}}}.k", type="string")],
        joins=joins or [],
    )


# ---------------------------------------------------------------------------
# _edge_weight — the shared fan-out cost model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relationship", "forward", "expected"),
    [
        ("many_to_one", True, 1),  # toward the "one" side — safe
        ("many_to_one", False, 10),  # reverse → toward "many" — fans out
        ("one_to_many", True, 10),  # toward the "many" side — fans out
        ("one_to_many", False, 1),  # reverse → toward "one" — safe
        ("one_to_one", True, 1),
        ("one_to_one", False, 1),
    ],
)
def test_edge_weight_penalises_fan_out(relationship: str, forward: bool, expected: int) -> None:
    assert _edge_weight(relationship, forward=forward) == expected


# ---------------------------------------------------------------------------
# Bidirectional reachability
# ---------------------------------------------------------------------------


def test_path_resolves_via_reverse_edge() -> None:
    """Only the child declares the join (child --many_to_one--> parent),
    yet a path parent → child resolves via the reverse edge."""
    parent = _cube("parent", "p")
    child = _cube("child", "ch", joins=[Join(to="parent", relationship="many_to_one", on="x")])
    cat = Catalog([parent, child]).as_dict()
    path = find_join_path("parent", "child", cat)
    assert [c for c, _ in path] == ["child"]


def test_child_measure_by_parent_dimension_compiles() -> None:
    """Regression: aggregate a child-cube measure grouped by a parent-cube
    dimension. The join is declared only on the parent; the reverse edge
    makes it resolvable (previously a JoinPathError)."""
    parent = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[Join(to="order_items", relationship="one_to_many", on="{o}.id = {i}.order_id")],
    )
    child = Cube(
        name="order_items",
        dialect=Dialect.POSTGRES,
        table="order_items",
        alias="i",
        measures=[Measure(name="qty", sql="{i}.quantity", agg="sum")],
        dimensions=[Dimension(name="sku", sql="{i}.sku", type="string")],
    )
    cat = Catalog([parent, child])
    q = SemanticQuery(measures=["order_items.qty"], dimensions=["orders.region"])
    sql = cat.compile(q, context={"schema": "prod"}).sql
    assert "SUM(i.quantity)" in sql
    assert "JOIN" in sql.upper()


# ---------------------------------------------------------------------------
# Fan-out weighting — prefer the safe path even when it is longer
# ---------------------------------------------------------------------------


def test_dijkstra_prefers_fan_out_free_path_over_shorter_one() -> None:
    """``a`` reaches ``d`` two ways: a direct one_to_many edge (cost 10) or
    a two-hop many_to_one chain through ``b`` (cost 1 + 1 = 2). Dijkstra
    takes the cheaper, fan-out-free chain despite the extra hop."""
    a = _cube(
        "a",
        "a",
        joins=[
            Join(to="d", relationship="one_to_many", on="x"),
            Join(to="b", relationship="many_to_one", on="x"),
        ],
    )
    b = _cube("b", "b", joins=[Join(to="d", relationship="many_to_one", on="x")])
    d = _cube("d", "d")
    cat = Catalog([a, b, d]).as_dict()
    path = find_join_path("a", "d", cat)
    assert [c for c, _ in path] == ["b", "d"], path


# ---------------------------------------------------------------------------
# Transit prohibition + genuine no-path
# ---------------------------------------------------------------------------


def test_forbidden_cube_may_be_endpoint_but_not_transit() -> None:
    """``mid`` is forbidden as a transit node: a path may END at it, but a
    path to ``far`` *through* it does not exist."""
    a = _cube("a", "a", joins=[Join(to="mid", relationship="many_to_one", on="x")])
    mid = _cube("mid", "m", joins=[Join(to="far", relationship="many_to_one", on="x")])
    far = _cube("far", "f")
    cat = Catalog([a, mid, far]).as_dict()
    forbid = frozenset({"mid"})
    # Endpoint is fine.
    assert [c for c, _ in find_join_path("a", "mid", cat, forbid_transit=forbid)] == ["mid"]
    # Transit through it is not.
    with pytest.raises(JoinPathError):
        find_join_path("a", "far", cat, forbid_transit=forbid)


def test_disconnected_cubes_raise_join_path_error() -> None:
    a = _cube("a", "a")
    b = _cube("b", "b")
    cat = Catalog([a, b]).as_dict()
    with pytest.raises(JoinPathError, match=r"(?i)no join path"):
        find_join_path("a", "b", cat)
