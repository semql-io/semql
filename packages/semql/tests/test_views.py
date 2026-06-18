"""Tests for ``View`` — curated catalog facades.

A view exposes a *renamed* subset of measures / dimensions drawn
from one or more underlying cubes. The planner addresses the view
by name and uses the renamed fields; the compiler rewrites the
references back to the underlying cube fields before resolution.

Two practical benefits:

1. **Prompt trimming.** When a catalog has 30 cubes but only
   five matter for a given question shape, expose a view that
   names just those fields. The planner prompt shrinks; the
   planner stays inside what's modelled.

2. **Join-ambiguity resolution.** If both ``users`` and ``orders``
   carry an ``identity_id`` dimension, ``view.identity_id`` picks
   one explicitly — the planner can't ask the wrong one.
"""

from __future__ import annotations

import pytest
from semql import AuthContext, Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery, View
from semql.errors import UnknownIdentifierError
from semql_prompt import planner_prompt


def _orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="count", sql="*", agg="count"),
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _checkout_view() -> View:
    return View(
        name="checkout",
        description="Curated facade — orders revenue + region only.",
        fields={
            "revenue": "orders.revenue",
            "region": "orders.region",
        },
    )


# ---------------------------------------------------------------------------
# Model — View shape.
# ---------------------------------------------------------------------------


def test_view_constructs_with_name_and_field_map() -> None:
    v = _checkout_view()
    assert v.name == "checkout"
    assert v.fields == {"revenue": "orders.revenue", "region": "orders.region"}


def test_view_rejects_empty_field_map() -> None:
    with pytest.raises(ValueError, match=r"(?i)view|fields|empty"):
        View(name="empty", fields={})


def test_view_rejects_unqualified_target() -> None:
    with pytest.raises(ValueError, match=r"(?i)cube\.field|qualified"):
        View(name="bad", fields={"x": "no_dot"})


def test_view_is_frozen() -> None:
    v = _checkout_view()
    with pytest.raises(Exception):  # noqa: B017, BLE001 — Pydantic raises ValidationError.
        v.name = "renamed"


# ---------------------------------------------------------------------------
# Catalog — accepts views, validates targets.
# ---------------------------------------------------------------------------


def test_catalog_accepts_views() -> None:
    cat = Catalog([_orders()], views=[_checkout_view()])
    assert "checkout" in cat.views
    assert cat.views["checkout"].name == "checkout"


def test_view_with_unknown_cube_target_raises() -> None:
    bad = View(name="bad", fields={"x": "ghost.field"})
    with pytest.raises(ValueError, match=r"(?i)view|ghost|cube"):
        Catalog([_orders()], views=[bad])


def test_view_with_unknown_field_target_raises() -> None:
    bad = View(name="bad", fields={"x": "orders.nonexistent"})
    with pytest.raises(ValueError, match=r"(?i)view|field|nonexistent"):
        Catalog([_orders()], views=[bad])


def test_duplicate_view_names_rejected() -> None:
    a = View(name="dup", fields={"x": "orders.revenue"})
    b = View(name="dup", fields={"y": "orders.count"})
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        Catalog([_orders()], views=[a, b])


def test_view_name_collides_with_cube_name() -> None:
    """A view named ``orders`` would shadow the cube — reject up front."""
    bad = View(name="orders", fields={"x": "orders.revenue"})
    with pytest.raises(ValueError, match=r"(?i)view|cube|name"):
        Catalog([_orders()], views=[bad])


# ---------------------------------------------------------------------------
# Compile — references via view.X resolve to underlying cube.field.
# ---------------------------------------------------------------------------


def test_compile_view_measure_resolves_to_underlying_cube() -> None:
    cat = Catalog([_orders()], views=[_checkout_view()])
    out = cat.compile(SemanticQuery(measures=["checkout.revenue"]))
    # Underlying cube SQL is what shows in the output.
    assert "SUM(o.amount)" in out.sql
    # The view's renamed alias is the output column.
    assert "AS revenue" in out.sql


def test_compile_view_dimension_groupby() -> None:
    cat = Catalog([_orders()], views=[_checkout_view()])
    out = cat.compile(
        SemanticQuery(
            measures=["checkout.revenue"],
            dimensions=["checkout.region"],
        )
    )
    assert "o.region" in out.sql
    assert "GROUP BY" in out.sql.upper()


def test_view_can_rename_fields() -> None:
    """``"net_revenue": "orders.revenue"`` exposes the underlying
    ``revenue`` measure under a different name in the view."""
    v = View(
        name="finance",
        fields={"net_revenue": "orders.revenue"},
    )
    cat = Catalog([_orders()], views=[v])
    out = cat.compile(SemanticQuery(measures=["finance.net_revenue"]))
    assert "SUM(o.amount)" in out.sql
    assert "AS net_revenue" in out.sql


# ---------------------------------------------------------------------------
# Prompt — views appear in the catalog block.
# ---------------------------------------------------------------------------


def test_prompt_includes_view_section() -> None:
    rendered = planner_prompt(Catalog([_orders()], views=[_checkout_view()]))
    assert "checkout" in rendered
    # The view's exposed names are what the planner sees.
    assert "`checkout.revenue`" in rendered or "checkout.revenue" in rendered


# ---------------------------------------------------------------------------
# Field-level role enforcement via views (SEMQL-VIEW-FIELD-ROLES-002):
# a view alias that maps to a required_roles-protected underlying field
# must be blocked for a viewer who lacks the required role.
# ---------------------------------------------------------------------------


def _orders_with_secret_revenue() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="count", sql="*", agg="count"),
            Measure(
                name="secret_revenue",
                sql="{o}.amount",
                agg="sum",
                required_roles=["finance"],
            ),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def _public_report_view() -> View:
    """View that exposes the role-protected measure under a bland name."""
    return View(
        name="public_report",
        fields={
            "revenue": "orders.secret_revenue",
            "region": "orders.region",
        },
    )


def test_view_field_role_blocked_for_low_role_viewer() -> None:
    """A viewer without the 'finance' role must not read secret_revenue
    even when it's accessed via a view alias (SEMQL-VIEW-FIELD-ROLES-002)."""
    cat = Catalog([_orders_with_secret_revenue()], views=[_public_report_view()])
    low_viewer = AuthContext(viewer_id="u1", roles=["viewer"])
    with pytest.raises(UnknownIdentifierError):
        cat.compile(
            SemanticQuery(measures=["public_report.revenue"]),
            viewer=low_viewer,
        )


def test_view_field_role_allowed_for_finance_viewer() -> None:
    """A viewer with the 'finance' role can access the field via the view."""
    cat = Catalog([_orders_with_secret_revenue()], views=[_public_report_view()])
    finance_viewer = AuthContext(viewer_id="u2", roles=["finance"])
    out = cat.compile(
        SemanticQuery(measures=["public_report.revenue"]),
        viewer=finance_viewer,
    )
    assert "SUM(o.amount)" in out.sql


def test_direct_field_role_still_blocked() -> None:
    """Baseline: direct access to the protected field is also blocked."""
    cat = Catalog([_orders_with_secret_revenue()])
    low_viewer = AuthContext(viewer_id="u1", roles=["viewer"])
    with pytest.raises(UnknownIdentifierError):
        cat.compile(
            SemanticQuery(measures=["orders.secret_revenue"]),
            viewer=low_viewer,
        )
