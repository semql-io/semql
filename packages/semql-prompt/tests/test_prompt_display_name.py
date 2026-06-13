"""Tests for ``display_name`` rendering in prompt fragments.

The catalog block keeps the machine identifier as the primary label
(LLMs reference it as ``cube.field`` in the SemanticQuery). The
human-readable ``display_name`` rides alongside as a short suffix so
the planner can pick up domain language without the identifier
disappearing from view.
"""

from __future__ import annotations

from semql import Cube, Dimension, Measure, TimeDimension
from semql.model import Dialect
from semql_prompt import (
    build_planner_prompt_fragment,
    build_router_prompt_fragment,
    render_catalog_block,
)


def _cat() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        display_name="Customer Orders",
        description="Each row is one order.",
        measures=[
            Measure(
                name="revenue",
                sql="{o}.amount",
                agg="sum",
                unit="currency",
                display_name="Total Revenue",
            ),
            # Measure without display_name — should NOT get a suffix.
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(
                name="region",
                sql="{o}.region",
                type="string",
                display_name="Sales Region",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                display_name="Order Date",
            ),
        ],
    )
    return {"orders": orders}


def test_cube_display_name_surfaces_in_catalog_block() -> None:
    rendered = render_catalog_block(_cat())
    assert "Customer Orders" in rendered


def test_measure_display_name_surfaces() -> None:
    rendered = render_catalog_block(_cat())
    assert "Total Revenue" in rendered


def test_dimension_display_name_surfaces() -> None:
    rendered = render_catalog_block(_cat())
    assert "Sales Region" in rendered


def test_time_dimension_display_name_surfaces() -> None:
    rendered = render_catalog_block(_cat())
    assert "Order Date" in rendered


def test_identifier_still_primary_when_display_name_set() -> None:
    """The machine identifier must still appear — display_name is a
    suffix, not a replacement, so the LLM still knows what to reference."""
    rendered = render_catalog_block(_cat())
    assert "`orders.revenue`" in rendered
    assert "`orders.region`" in rendered


def test_field_without_display_name_renders_without_suffix() -> None:
    """No display_name → no `(human: ...)`-style noise on that line."""
    rendered = render_catalog_block(_cat())
    # 'count' has no display_name — make sure that line stays clean.
    count_lines = [line for line in rendered.splitlines() if "`orders.count`" in line]
    assert count_lines, "expected to find the orders.count line"
    for line in count_lines:
        assert "human" not in line.lower()


def test_planner_fragment_includes_display_names() -> None:
    rendered = build_planner_prompt_fragment(_cat())
    assert "Total Revenue" in rendered


def test_router_topic_summary_uses_display_name() -> None:
    """When a cube has a display_name, the router topic summary should
    show it (machine ident stays as the code-spanned label)."""
    rendered = build_router_prompt_fragment(_cat())
    assert "Customer Orders" in rendered
    assert "`orders`" in rendered
