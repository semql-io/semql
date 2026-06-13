"""Tests for ``Cube.segments`` — named, reusable filter sets.

A segment is a predicate the planner can name instead of re-deriving.
``Cube(segments=[Segment("active_orders", "...")])`` declares one;
``SemanticQuery(segments=["orders.active_orders"])`` references it.
The compiler ANDs each named segment's predicate into the WHERE
clause alongside ``filters``.

Centralising the predicate kills "the planner forgot the status
filter" errors and documents the business definition in one place.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    CompileError,
    Cube,
    Dialect,
    Dimension,
    Measure,
    Segment,
    SemanticQuery,
)
from semql_prompt import planner_prompt


def _orders_with_segments() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        segments=[
            Segment(
                name="paid",
                sql="{o}.status = 'paid'",
                description="Only orders with payment confirmed.",
            ),
            Segment(
                name="recent",
                sql="{o}.created_at >= now() - INTERVAL '30 days'",
                description="Orders from the last 30 days.",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Model — Segment + Cube.segments + SemanticQuery.segments
# ---------------------------------------------------------------------------


def test_segment_constructs_with_name_sql_description() -> None:
    s = Segment(name="paid", sql="{o}.status = 'paid'", description="Confirmed.")
    assert s.name == "paid"
    assert s.sql == "{o}.status = 'paid'"
    assert s.description == "Confirmed."


def test_segment_description_defaults_to_empty() -> None:
    s = Segment(name="x", sql="x = 1")
    assert s.description == ""


def test_cube_segments_default_to_empty_list() -> None:
    cube = Cube(name="c", backend=Dialect.POSTGRES, table="c", alias="c")
    assert cube.segments == []


def test_cube_accepts_segments_list() -> None:
    cube = _orders_with_segments()
    assert {s.name for s in cube.segments} == {"paid", "recent"}


def test_semantic_query_segments_default_empty() -> None:
    q = SemanticQuery(measures=["orders.count"])
    assert q.segments == []


def test_semantic_query_accepts_segments_list() -> None:
    q = SemanticQuery(measures=["orders.count"], segments=["orders.paid"])
    assert q.segments == ["orders.paid"]


# ---------------------------------------------------------------------------
# Compiler — segment predicates AND into the WHERE clause.
# ---------------------------------------------------------------------------


def test_segment_predicate_appears_in_compiled_where() -> None:
    cat = Catalog([_orders_with_segments()])
    out = cat.compile(SemanticQuery(measures=["orders.count"], segments=["orders.paid"]))
    # The segment's SQL fragment shows up resolved against the alias.
    assert "o.status" in out.sql
    assert "'paid'" in out.sql


def test_multiple_segments_and_compose() -> None:
    cat = Catalog([_orders_with_segments()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            segments=["orders.paid", "orders.recent"],
        )
    )
    assert "o.status" in out.sql
    assert "30 days" in out.sql or "INTERVAL" in out.sql


def test_segments_and_filters_compose_via_and() -> None:
    """Segments AND with the existing filter list; no precedence
    surprises."""
    cat = Catalog([_orders_with_segments()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            dimensions=["orders.region"],
            segments=["orders.paid"],
            filters=[__import__("semql").Filter(dimension="orders.region", op="eq", values=["us"])],
        )
    )
    # Segment predicate AND filter predicate AND (any base_predicate) — all present.
    assert "o.status" in out.sql
    assert "o.region" in out.sql
    assert "us" in out.params.values()


def test_segment_qualified_reference_required() -> None:
    """Segments reference as ``cube.segment`` — bare names without
    a cube prefix don't resolve."""
    cat = Catalog([_orders_with_segments()])
    with pytest.raises(CompileError, match=r"(?i)segment"):
        cat.compile(SemanticQuery(measures=["orders.count"], segments=["paid"]))


def test_segment_unknown_name_rejected() -> None:
    cat = Catalog([_orders_with_segments()])
    with pytest.raises(CompileError, match=r"(?i)segment"):
        cat.compile(SemanticQuery(measures=["orders.count"], segments=["orders.nonexistent"]))


def test_segment_unknown_cube_rejected() -> None:
    cat = Catalog([_orders_with_segments()])
    with pytest.raises(CompileError, match=r"(?i)segment"):
        cat.compile(SemanticQuery(measures=["orders.count"], segments=["ghost.paid"]))


# ---------------------------------------------------------------------------
# Prompt fragment surfaces the segment menu.
# ---------------------------------------------------------------------------


def test_prompt_lists_segments_for_each_cube() -> None:
    rendered = planner_prompt(Catalog([_orders_with_segments()]))
    assert "**Segments:**" in rendered
    assert "`orders.paid`" in rendered
    assert "`orders.recent`" in rendered


def test_prompt_includes_segment_descriptions() -> None:
    rendered = planner_prompt(Catalog([_orders_with_segments()]))
    assert "payment confirmed" in rendered.lower()


def test_prompt_omits_segments_section_when_none() -> None:
    cube = Cube(
        name="bare",
        backend=Dialect.POSTGRES,
        table="bare",
        alias="b",
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    rendered = planner_prompt(Catalog([cube]))
    assert "**Segments:**" not in rendered
