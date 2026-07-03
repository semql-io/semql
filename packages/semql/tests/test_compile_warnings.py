"""Compile-path warnings channel (W3/ktx-C4).

``CompiledQuery.warnings`` carries soft, non-fatal advisories the caller
may surface but need not act on — distinct from the exception path, which
is for refusals. Its first two callers are the ambiguous-join-path
warning and the soft fan-out warning (both derived from the plan's
``join_diagnostics``). Errors stay exceptions; a clean query warns nothing.
"""

from __future__ import annotations

from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, SemanticQuery


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


def test_clean_query_has_no_warnings() -> None:
    """A single-cube aggregate touches no join graph — nothing to warn."""
    a = _cube("a", "a")
    cat = Catalog([a])
    out = cat.compile(SemanticQuery(measures=["a.n"], dimensions=["a.k"]))
    assert out.warnings == ()


def test_safe_join_has_no_warnings() -> None:
    """Aggregating a child measure by a parent dimension traverses toward
    the 'one' side — safe, unambiguous, no warning."""
    parent = _cube("parent", "p")
    child = _cube("child", "ch", joins=[Join(to="parent", relationship="many_to_one", on="x")])
    cat = Catalog([parent, child])
    out = cat.compile(SemanticQuery(measures=["child.n"], dimensions=["parent.k"]))
    assert out.warnings == ()


def test_ambiguous_join_path_warns() -> None:
    """``a`` reaches ``d`` two equal-cost ways (a→b→d, a→c→d). The compiler
    resolves one by tie-break and warns the catalog is under-specified."""
    a = _cube(
        "a",
        "a",
        joins=[
            Join(to="b", relationship="many_to_one", on="x"),
            Join(to="c", relationship="many_to_one", on="x"),
        ],
    )
    b = _cube("b", "b", joins=[Join(to="d", relationship="many_to_one", on="x")])
    c = _cube("c", "c", joins=[Join(to="d", relationship="many_to_one", on="x")])
    d = _cube("d", "d")
    cat = Catalog([a, b, c, d])
    out = cat.compile(SemanticQuery(measures=["a.n"], dimensions=["d.k"]))
    assert len(out.warnings) == 1
    w = out.warnings[0]
    assert "ambiguous" in w.lower()
    assert "'d'" in w


def test_fan_out_row_listing_warns() -> None:
    """A row-listing query that fans out the root cube (one→many) survives
    the hard fan-out refusal (no additive measure) but earns a soft warning:
    a SUM added later would over-count across the join."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        dimensions=[Dimension(name="id", sql="{o}.id", type="string")],
        joins=[Join(to="items", relationship="one_to_many", on="{o}.id = {i}.order_id")],
    )
    items = Cube(
        name="items",
        dialect=Dialect.POSTGRES,
        table="items",
        alias="i",
        dimensions=[Dimension(name="sku", sql="{i}.sku", type="string")],
    )
    cat = Catalog([orders, items])
    q = SemanticQuery(dimensions=["orders.id", "items.sku"], ungrouped=True, limit=100)
    out = cat.compile(q)
    assert len(out.warnings) == 1
    assert "fans out" in out.warnings[0]
    assert "'items'" in out.warnings[0]


def test_warnings_survive_model_dump_round_trip() -> None:
    from semql.compile import CompiledQuery

    a = _cube(
        "a",
        "a",
        joins=[
            Join(to="b", relationship="many_to_one", on="x"),
            Join(to="c", relationship="many_to_one", on="x"),
        ],
    )
    b = _cube("b", "b", joins=[Join(to="d", relationship="many_to_one", on="x")])
    c = _cube("c", "c", joins=[Join(to="d", relationship="many_to_one", on="x")])
    d = _cube("d", "d")
    cat = Catalog([a, b, c, d])
    out = cat.compile(SemanticQuery(measures=["a.n"], dimensions=["d.k"]))
    assert out.warnings
    assert CompiledQuery.model_validate(out.model_dump()).warnings == out.warnings
