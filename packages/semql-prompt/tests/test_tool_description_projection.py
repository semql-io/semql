"""Tests for ``project_tool_descriptions`` — the cache-friendly
per-cube MCP tool-description layout.

Mirrors :class:`CatalogPrompt` at the tool-schema layer: the static
``invariant`` segment holds descriptions for cubes with no
``required_roles`` (identical for every viewer — cache them); the
``viewer_gated`` segment holds the role-gated cubes the viewer is
authorised to see beyond the public set.

Auth invariant: a viewer never sees the description of a cube they
can't access. The ``viewer_gated`` map omits anything ``viewer_sees``
rejects.
"""

from __future__ import annotations

from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    TimeDimension,
)
from semql_prompt import (
    ToolDescriptionProjection,
    project_tool_descriptions,
    render_tool_description,
)


def _public_orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        description="Customer orders.",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(
                name="watch_time",
                sql="{o}.watch_seconds",
                agg="sum",
                unit="seconds",
                display_unit="hours",
            ),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="placed_at", sql="{o}.placed_at")],
    )


def _admin_audit() -> Cube:
    return Cube(
        name="audit_events",
        dialect=Dialect.POSTGRES,
        table="public.audit",
        alias="a",
        required_roles=["admin"],
        description="Audit log entries — admin only.",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="actor", sql="{a}.actor", type="string")],
    )


def _support_tickets() -> Cube:
    return Cube(
        name="tickets",
        dialect=Dialect.POSTGRES,
        table="public.tickets",
        alias="t",
        required_roles=["support"],
        measures=[Measure(name="open", sql="*", agg="count")],
        dimensions=[Dimension(name="status", sql="{t}.status", type="string")],
    )


def _mixed_catalog() -> Catalog:
    return Catalog([_public_orders(), _admin_audit(), _support_tickets()])


# ---------------------------------------------------------------------------
# Invariant vs viewer-gated partitioning
# ---------------------------------------------------------------------------


def test_returns_projection_dataclass() -> None:
    cat = _mixed_catalog()
    proj = project_tool_descriptions(cat.as_dict())
    assert isinstance(proj, ToolDescriptionProjection)
    assert isinstance(proj.invariant, dict)
    assert isinstance(proj.viewer_gated, dict)


def test_invariant_segment_contains_only_public_cubes() -> None:
    cat = _mixed_catalog()
    proj = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u", roles=["admin"]),
    )
    assert "orders" in proj.invariant
    # role-gated cubes never appear in the invariant set, even when
    # the viewer holds the matching role.
    assert "audit_events" not in proj.invariant
    assert "tickets" not in proj.invariant


def test_viewer_gated_holds_role_visible_cubes_only() -> None:
    cat = _mixed_catalog()
    proj = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u", roles=["admin"]),
    )
    # Viewer is admin → audit_events visible. tickets requires
    # "support" which the viewer lacks → not visible.
    assert "audit_events" in proj.viewer_gated
    assert "tickets" not in proj.viewer_gated


def test_no_viewer_means_empty_overlay() -> None:
    """Without a viewer, the gated segment is empty — role-gated cubes
    aren't disclosed to anonymous callers."""
    cat = _mixed_catalog()
    proj = project_tool_descriptions(cat.as_dict())
    assert proj.viewer_gated == {}


def test_invariant_set_is_viewer_independent() -> None:
    """Same catalog → same invariant set regardless of viewer. The
    whole point of the segment is cacheability."""
    cat = _mixed_catalog()
    p_admin = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u1", roles=["admin"]),
    )
    p_support = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u2", roles=["support"]),
    )
    p_none = project_tool_descriptions(cat.as_dict())
    assert p_admin.invariant == p_support.invariant == p_none.invariant


def test_unauthorised_viewer_doesnt_see_gated_cube_descriptions() -> None:
    """Auth invariant: a viewer with no matching role gets an empty
    gated set, never the description of a cube they can't access."""
    cat = _mixed_catalog()
    proj = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u", roles=["nobody"]),
    )
    assert proj.viewer_gated == {}
    # And the invariant still has the public cube.
    assert "orders" in proj.invariant


# ---------------------------------------------------------------------------
# Description content
# ---------------------------------------------------------------------------


def test_description_includes_measures_with_unit_annotations() -> None:
    """The ``watch_time`` measure has unit=seconds + display_unit=hours.
    The rendered description surfaces that as ``[seconds → hours]`` so
    an LLM picking tools doesn't reinvent the conversion."""
    cat = _mixed_catalog()
    proj = project_tool_descriptions(cat.as_dict())
    desc = proj.invariant["orders"]
    assert "watch_time [seconds → hours]" in desc
    assert "revenue [currency]" in desc


def test_description_includes_dimensions_and_time_dims() -> None:
    cat = _mixed_catalog()
    proj = project_tool_descriptions(cat.as_dict())
    desc = proj.invariant["orders"]
    assert "Dimensions: region" in desc
    assert "Time dimensions: placed_at" in desc


def test_description_uses_default_when_cube_lacks_description() -> None:
    cube = Cube(
        name="bare",
        dialect=Dialect.POSTGRES,
        table="bare",
        alias="b",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="id", sql="{b}.id", type="number")],
    )
    cat = Catalog([cube])
    proj = project_tool_descriptions(cat.as_dict())
    assert "Query the bare cube." in proj.invariant["bare"]


def test_render_tool_description_is_the_source_of_truth() -> None:
    """The MCP server's per-cube tool docstring goes through the same
    ``render_tool_description`` function. Calling it directly should
    produce the same string the projection returns — so the prompt
    surface and the MCP surface can't drift."""
    cube = _public_orders()
    cat = Catalog([cube])
    proj = project_tool_descriptions(cat.as_dict())
    assert proj.invariant["orders"] == render_tool_description(cube)


# ---------------------------------------------------------------------------
# ``all()`` convenience
# ---------------------------------------------------------------------------


def test_all_merges_segments_for_full_visible_set() -> None:
    cat = _mixed_catalog()
    proj = project_tool_descriptions(
        cat.as_dict(),
        viewer=AuthContext(viewer_id="u", roles=["admin"]),
    )
    merged = proj.all()
    assert set(merged) == {"orders", "audit_events"}


def test_protected_fields_are_viewer_specific_not_invariant() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(
                name="secret",
                sql="{o}.secret",
                agg="sum",
                required_roles=["admin"],
            ),
        ],
    )
    catalog = Catalog([cube])
    anonymous = project_tool_descriptions(catalog.as_dict())
    admin = project_tool_descriptions(
        catalog.as_dict(),
        viewer=AuthContext(viewer_id="admin", roles=["admin"]),
    )
    guest = project_tool_descriptions(
        catalog.as_dict(),
        viewer=AuthContext(viewer_id="guest", roles=[]),
    )

    assert anonymous.invariant == admin.invariant == guest.invariant
    assert "secret" not in str(admin.invariant)
    assert "secret" in str(admin.all())
    assert "secret" not in str(guest.all())


def test_all_prefers_authorized_overlay_for_duplicate_cube_key() -> None:
    projection = ToolDescriptionProjection(
        invariant={"orders": "public fields only"},
        viewer_gated={"orders": "authorized fields including secret"},
    )

    assert projection.all()["orders"] == "authorized fields including secret"


def test_dynamic_policy_is_authoritative_at_projection_api() -> None:
    catalog = Catalog(
        [_public_orders()],
        policy=lambda _cube, viewer: viewer.viewer_id == "admin",
    )
    no_viewer = project_tool_descriptions(
        catalog.as_dict(),
        policy=catalog.policy,
    )
    admin = project_tool_descriptions(
        catalog.as_dict(),
        viewer=AuthContext(viewer_id="admin"),
        policy=catalog.policy,
    )
    guest = project_tool_descriptions(
        catalog.as_dict(),
        viewer=AuthContext(viewer_id="guest"),
        policy=catalog.policy,
    )

    assert no_viewer.invariant == admin.invariant == guest.invariant == {}
    assert "orders" in no_viewer.viewer_gated
    assert "orders" in no_viewer.all()
    assert "orders" in admin.viewer_gated
    assert "orders" in admin.all()
    assert guest.viewer_gated == {}
    assert "orders" not in guest.all()
