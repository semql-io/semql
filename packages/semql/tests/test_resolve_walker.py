"""Tests for the shared query-resolution walker.

The walker (``semql._resolve.walk_query_fields``) is the single
implementation that backs both ``compile_query`` (re-raises as
``CompileError``) and ``validate`` (translates to ``ValidationError``).
These tests pin the diagnostic codes the walker emits — if they drift,
both downstream layers drift in lockstep."""

from __future__ import annotations

import pytest
from semql import (
    BoolExpr,
    Catalog,
    Cube,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from semql._resolve import walk_query_fields
from semql.errors import FilterTypeError, UnknownIdentifierError
from semql.model import Dialect, Segment


def _cat() -> dict[str, Cube]:
    orders = Cube(
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
            Dimension(name="is_paid", sql="{o}.is_paid", type="bool"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
        segments=[Segment(name="paid", sql="o.is_paid = true")],
    )
    return Catalog([orders]).as_dict()


def test_resolves_clean_query_with_no_diagnostics() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    resolved, diagnostics = walk_query_fields(q, _cat())
    assert diagnostics == []
    assert len(resolved.measure_fields) == 1
    assert len(resolved.dim_fields) == 1
    assert resolved.time_cube is not None
    assert [c.name for c in resolved.touched] == ["orders"]


def test_unknown_field_diagnostic_carries_source_exception() -> None:
    """Single-error compile-path preservation depends on ``source``
    being the original typed exception. Pin the shape."""
    q = SemanticQuery(measures=["orders.no_such"])
    _, diagnostics = walk_query_fields(q, _cat())
    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.code == "unknown_field"
    assert isinstance(d.source, UnknownIdentifierError)
    assert d.source.name == "no_such"


def test_filter_type_mismatch_source_is_typed_filter_error() -> None:
    """A single filter-type diagnostic must carry FilterTypeError as
    source so the compile path re-raises the typed leaf standalone."""
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    _, diagnostics = walk_query_fields(q, _cat())
    assert len(diagnostics) == 1
    assert diagnostics[0].code == "filter_type_mismatch"
    assert isinstance(diagnostics[0].source, FilterTypeError)


def test_walks_where_tree_leaves() -> None:
    q = SemanticQuery(
        measures=["orders.count"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.no_such", op="eq", values=["x"]),
            ],
        ),
    )
    resolved, diagnostics = walk_query_fields(q, _cat())
    codes = [d.code for d in diagnostics]
    assert codes == ["unknown_field"]
    # The resolved leaf is keyed by id(leaf); the resolved-OK leaf
    # made it into the map.
    assert len(resolved.where_leaf_resolutions) == 1


@pytest.mark.parametrize(
    ("seg_ref", "expected_code"),
    [
        ("bareword", "segment_unqualified"),
        ("nope.paid", "segment_unknown_cube"),
        ("orders.no_such_segment", "segment_unknown_segment"),
    ],
)
def test_segment_diagnostics(seg_ref: str, expected_code: str) -> None:
    q = SemanticQuery(measures=["orders.count"], segments=[seg_ref])
    _, diagnostics = walk_query_fields(q, _cat())
    codes = [d.code for d in diagnostics]
    assert expected_code in codes


def test_partial_resolution_still_collects_touched_cubes() -> None:
    """Even when a query has bad references, the cubes that did
    resolve land in ``touched`` so downstream auth / lifecycle checks
    still see the partial set."""
    q = SemanticQuery(
        measures=["orders.revenue", "nope.boom"],
        dimensions=["orders.region"],
    )
    resolved, diagnostics = walk_query_fields(q, _cat())
    assert any(d.code == "unknown_cube" for d in diagnostics)
    assert [c.name for c in resolved.touched] == ["orders"]


def test_accumulates_all_diagnostics_not_just_first() -> None:
    q = SemanticQuery(
        measures=["orders.bogus", "nope.boom"],
        dimensions=["orders.region", "orders.also_bogus"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    _, diagnostics = walk_query_fields(q, _cat())
    codes = [d.code for d in diagnostics]
    # 1× unknown_field on orders.bogus, 1× unknown_cube on nope,
    # 1× unknown_field on orders.also_bogus, 1× filter_type_mismatch.
    assert codes.count("unknown_field") == 2
    assert codes.count("unknown_cube") == 1
    assert codes.count("filter_type_mismatch") == 1
