"""Tests for ``semql_erd.render_dot``.

The DOT path is pure Python — no graphviz binary needed — so these
tests pin the output shape across every supported relationship,
filtering option, and edge case (joins to filtered-out cubes,
META cubes, empty catalogs).
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    TimeDimension,
)
from semql_erd import render_dot


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        display_name="Customer Orders",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id"),
        ],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )


# ---------------------------------------------------------------------------
# Graph skeleton — header, rankdir, node shape, edge presence.
# ---------------------------------------------------------------------------


def test_render_dot_returns_digraph_header() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert out.startswith("digraph catalog {")
    assert out.rstrip().endswith("}")


def test_render_dot_default_rankdir_is_LR() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert 'rankdir="LR"' in out


def test_render_dot_respects_rankdir_kwarg() -> None:
    out = render_dot(Catalog([_orders(), _customers()]), rankdir="TB")
    assert 'rankdir="TB"' in out


def test_render_dot_sets_record_node_shape() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "node [shape=record" in out


def test_render_dot_optional_title_emitted_as_label() -> None:
    out = render_dot(Catalog([_orders(), _customers()]), title="My Catalog")
    assert "label=" in out
    assert "My Catalog" in out
    assert "labelloc=t" in out


def test_render_dot_omits_label_when_no_title() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    # No graph-level label= line.
    for line in out.splitlines():
        s = line.strip()
        assert not s.startswith('label="')


# ---------------------------------------------------------------------------
# Node labels — cube name, backend, sections per field kind.
# ---------------------------------------------------------------------------


def test_node_label_includes_cube_name() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "orders [label=" in out


def test_node_label_includes_display_name_when_present() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "Customer Orders" in out


def test_node_label_includes_backend() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "postgres" in out


def test_node_label_lists_measures() -> None:
    """Measures are listed with their storage unit in square brackets so
    a reader (human or LLM) sees the dimension at a glance."""
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "revenue [currency]" in out
    assert "count [count]" in out


def test_node_label_renders_display_unit_when_set() -> None:
    """When a measure declares both ``unit`` and ``display_unit``, the
    ERD shows the conversion arrow ``unit → display_unit``."""
    cube = Cube(
        name="sessions",
        backend=Dialect.POSTGRES,
        table="sessions",
        alias="s",
        measures=[
            Measure(
                name="watch_time",
                sql="{s}.duration",
                agg="sum",
                unit="seconds",
                display_unit="hours",
            ),
        ],
    )
    out = render_dot(Catalog([cube]))
    assert "watch_time [seconds → hours]" in out


def test_node_label_lists_dimensions() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "dimensions: region, status" in out


def test_node_label_lists_time_dimensions() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "time: created_at" in out


def test_node_label_omits_empty_sections() -> None:
    cube = Cube(
        name="lonely",
        backend=Dialect.POSTGRES,
        table="lonely",
        alias="l",
        dimensions=[Dimension(name="x", sql="{l}.x", type="string")],
    )
    out = render_dot(Catalog([cube]))
    # No measures / time sections when the cube has none.
    assert "measures:" not in out
    assert "time:" not in out
    assert "dimensions: x" in out


# ---------------------------------------------------------------------------
# Edges — crow's-foot arrowheads per relationship.
# ---------------------------------------------------------------------------


def test_many_to_one_uses_crow_tail_tee_head() -> None:
    out = render_dot(Catalog([_orders(), _customers()]))
    assert 'arrowtail="crow"' in out
    assert 'arrowhead="tee"' in out
    assert 'dir="both"' in out


def test_one_to_many_mirrors_many_to_one() -> None:
    a = Cube(
        name="a",
        backend=Dialect.POSTGRES,
        table="a",
        alias="a",
        joins=[Join(to="b", relationship="one_to_many", on="{a}.id = {b}.aid")],
        dimensions=[Dimension(name="x", sql="{a}.x", type="string")],
    )
    b = Cube(
        name="b",
        backend=Dialect.POSTGRES,
        table="b",
        alias="b",
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    out = render_dot(Catalog([a, b]))
    assert 'arrowtail="tee"' in out
    assert 'arrowhead="crow"' in out


def test_one_to_one_uses_tee_on_both_sides() -> None:
    a = Cube(
        name="a",
        backend=Dialect.POSTGRES,
        table="a",
        alias="a",
        joins=[Join(to="b", relationship="one_to_one", on="{a}.id = {b}.id")],
        dimensions=[Dimension(name="x", sql="{a}.x", type="string")],
    )
    b = Cube(
        name="b",
        backend=Dialect.POSTGRES,
        table="b",
        alias="b",
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    out = render_dot(Catalog([a, b]))
    # Both arrowtail and arrowhead are tee — count to make sure.
    edge_lines = [line for line in out.splitlines() if " -> " in line]
    assert len(edge_lines) == 1
    edge = edge_lines[0]
    assert 'arrowtail="tee"' in edge
    assert 'arrowhead="tee"' in edge


def test_edge_uses_safe_node_ids() -> None:
    """Cube names already conform to ``[a-z_][a-z0-9_]*`` by the
    resolver regex, so they need no escaping. Pin that we don't quote
    them either."""
    out = render_dot(Catalog([_orders(), _customers()]))
    assert "  orders -> customers " in out


# ---------------------------------------------------------------------------
# Filtering — only_exposed, META exclusion, dangling-edge skip.
# ---------------------------------------------------------------------------


def test_only_exposed_default_hides_internal_cubes() -> None:
    hidden = Cube(
        name="hidden",
        backend=Dialect.POSTGRES,
        table="hidden",
        alias="h",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{h}.x", type="string")],
    )
    out = render_dot(Catalog([_orders(), _customers(), hidden]))
    assert "hidden [label=" not in out


def test_only_exposed_false_shows_hidden_cubes() -> None:
    hidden = Cube(
        name="hidden",
        backend=Dialect.POSTGRES,
        table="hidden",
        alias="h",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{h}.x", type="string")],
    )
    out = render_dot(Catalog([_orders(), _customers(), hidden]), only_exposed=False)
    assert "hidden [label=" in out


def test_meta_cubes_are_always_excluded() -> None:
    """META reflection cubes are an introspection mechanism; they
    aren't part of the data-model graph and would clutter every
    diagram."""
    out = render_dot(Catalog([_orders(), _customers()]), only_exposed=False)
    for meta in ("catalog_cubes", "catalog_measures", "catalog_dimensions"):
        assert f"{meta} [label=" not in out


def test_edges_to_filtered_out_cubes_are_dropped() -> None:
    """A Join into a hidden cube shouldn't render as a dangling edge."""
    hidden = Cube(
        name="hidden",
        backend=Dialect.POSTGRES,
        table="hidden",
        alias="h",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{h}.x", type="string")],
    )
    orders_with_hidden_join = _orders().model_copy(
        update={
            "joins": [
                Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id"),
                Join(to="hidden", relationship="many_to_one", on="{o}.hid = {h}.id"),
            ]
        }
    )
    out = render_dot(Catalog([orders_with_hidden_join, _customers(), hidden]))
    # ``customers`` edge present; ``hidden`` edge dropped.
    assert "orders -> customers" in out
    assert "orders -> hidden" not in out


# ---------------------------------------------------------------------------
# Empty / single-cube catalogs
# ---------------------------------------------------------------------------


def test_single_cube_no_edges_renders_clean() -> None:
    out = render_dot(Catalog([_customers()]))
    # No edges at all.
    assert " -> " not in out
    # But the node is there.
    assert "customers [label=" in out


def test_catalog_with_only_hidden_cubes_renders_empty_graph() -> None:
    hidden = Cube(
        name="hidden",
        backend=Dialect.POSTGRES,
        table="hidden",
        alias="h",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{h}.x", type="string")],
    )
    out = render_dot(Catalog([hidden]))
    assert " -> " not in out
    assert "[label=" not in out


# ---------------------------------------------------------------------------
# Escaping — record label specials don't break the graph.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weird",
    [
        "Has | pipes",
        'Quotes "inside"',
        "Curly {braces} too",
        "Angle <brackets>",
        "Mix | of < weird > | things",
    ],
)
def test_record_label_escapes_special_characters(weird: str) -> None:
    cube = Cube(
        name="ugly",
        backend=Dialect.POSTGRES,
        table="ugly",
        alias="u",
        display_name=weird,
        dimensions=[Dimension(name="x", sql="{u}.x", type="string")],
    )
    out = render_dot(Catalog([cube]))
    # The escape function should have neutralised pipes / braces / quotes.
    # Cheap proof: the output is still valid as text and doesn't have an
    # *unescaped* pipe inside the label (only the section separators).
    for ch in ("|", "{", "}"):
        if ch in weird:
            assert f"\\{ch}" in out
