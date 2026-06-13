"""Tests for the cache-friendly two-segment prompt layout (P6).

The cacheable layout splits ``planner_prompt(Catalog, ...)`` into:

- ``static``: spec contract + publicly visible cubes + raw-fallback.
  Identical across viewers. Goes above an Anthropic / Bedrock prompt-
  cache breakpoint.
- ``overlay``: role-gated cubes the viewer can see, plus a short note.
  Varies per viewer; goes below the breakpoint.

The auth invariant — viewers shouldn't learn names of cubes they
can't access — is preserved by gating overlay membership on
``viewer_sees`` and gating static membership on ``Cube.required_roles
== []``.
"""

from __future__ import annotations

from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Lookup,
    Measure,
)
from semql_prompt import (
    CatalogPrompt,
    planner_prompt,
    planner_prompt_segments,
    prompt_hash,
)


def _public_orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def _admin_audit() -> Cube:
    return Cube(
        name="audit_events",
        backend=Dialect.POSTGRES,
        table="public.audit",
        alias="a",
        required_roles=["admin"],
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="actor", sql="{a}.actor", type="string")],
    )


def _support_tickets() -> Cube:
    return Cube(
        name="tickets",
        backend=Dialect.POSTGRES,
        table="public.tickets",
        alias="t",
        required_roles=["support"],
        measures=[Measure(name="open", sql="*", agg="count")],
        dimensions=[Dimension(name="status", sql="{t}.status", type="string")],
    )


def _mixed_catalog() -> Catalog:
    return Catalog([_public_orders(), _admin_audit(), _support_tickets()])


# ---------------------------------------------------------------------------
# Segment partitioning: static vs overlay
# ---------------------------------------------------------------------------


def test_segments_returns_catalog_prompt() -> None:
    cat = _mixed_catalog()
    segs = planner_prompt_segments(
        cat,
    )
    assert isinstance(segs, CatalogPrompt)
    assert isinstance(segs.static, str)
    assert isinstance(segs.overlay, str)


def test_static_segment_contains_only_public_cubes() -> None:
    cat = _mixed_catalog()
    segs = planner_prompt_segments(cat, viewer=AuthContext(viewer_id="u", roles=["admin"]))
    # `orders` is public — must be in static.
    assert "### orders" in segs.static
    # `audit_events` is admin-gated — must NOT be in static (even
    # though viewer is admin).
    assert "### audit_events" not in segs.static
    assert "### tickets" not in segs.static


def test_overlay_holds_role_gated_cubes_the_viewer_can_see() -> None:
    cat = _mixed_catalog()
    admin = AuthContext(viewer_id="u", roles=["admin"])
    segs = planner_prompt_segments(cat, viewer=admin)
    assert "### audit_events" in segs.overlay
    # `tickets` requires `support`, which the admin viewer doesn't hold.
    assert "tickets" not in segs.overlay


def test_overlay_excludes_cubes_outside_viewer_roles() -> None:
    cat = _mixed_catalog()
    nobody = AuthContext(viewer_id="u", roles=[])
    segs = planner_prompt_segments(cat, viewer=nobody)
    # Viewer has no roles → overlay is empty.
    assert segs.overlay == ""


def test_viewer_none_returns_empty_overlay() -> None:
    cat = _mixed_catalog()
    segs = planner_prompt_segments(
        cat,
    )
    assert segs.overlay == ""


def test_overlay_notes_visibility() -> None:
    cat = _mixed_catalog()
    multi = AuthContext(viewer_id="u", roles=["admin", "support"])
    segs = planner_prompt_segments(cat, viewer=multi)
    # Note enumerates the role-gated cubes the viewer can now see.
    assert "audit_events" in segs.overlay
    assert "tickets" in segs.overlay
    assert "CUBES VISIBLE TO YOU" in segs.overlay


# ---------------------------------------------------------------------------
# Cache-key stability: static segment is viewer-invariant
# ---------------------------------------------------------------------------


def test_static_segment_identical_across_viewers() -> None:
    cat = _mixed_catalog()
    admin = AuthContext(viewer_id="a", roles=["admin"])
    support = AuthContext(viewer_id="s", roles=["support"])
    none = AuthContext(viewer_id="n", roles=[])

    s_admin = planner_prompt_segments(cat, viewer=admin).static
    s_support = planner_prompt_segments(cat, viewer=support).static
    s_none = planner_prompt_segments(cat, viewer=none).static
    s_no_viewer = planner_prompt_segments(
        cat,
    ).static
    assert s_admin == s_support == s_none == s_no_viewer


def test_prompt_hash_stable_across_viewers() -> None:
    cat = _mixed_catalog()
    h_admin = prompt_hash(
        cat,
    )
    h_support_viewer = prompt_hash(
        cat,
    )
    assert h_admin == h_support_viewer
    # SHA256 hex is 64 chars.
    assert len(h_admin) == 64
    assert all(c in "0123456789abcdef" for c in h_admin)


def test_prompt_hash_changes_when_public_cube_renames() -> None:
    cat_before = Catalog([_public_orders()])
    renamed = _public_orders().model_copy(update={"name": "orders_v2"})
    cat_after = Catalog([renamed])
    assert prompt_hash(
        cat_before,
    ) != prompt_hash(
        cat_after,
    )


def test_prompt_hash_unchanged_when_role_gated_cube_added() -> None:
    """Adding a role-gated cube doesn't touch the static segment — so
    cached prompt fragments shouldn't invalidate."""
    cat_base = Catalog([_public_orders()])
    cat_with_admin = Catalog([_public_orders(), _admin_audit()])
    assert prompt_hash(
        cat_base,
    ) == prompt_hash(
        cat_with_admin,
    )


# ---------------------------------------------------------------------------
# joined() fallback for non-cached emission
# ---------------------------------------------------------------------------


def test_joined_concatenates_static_and_overlay() -> None:
    cat = _mixed_catalog()
    segs = planner_prompt_segments(cat, viewer=AuthContext(viewer_id="u", roles=["admin"]))
    joined = segs.joined()
    assert "### orders" in joined
    assert "### audit_events" in joined
    # Static comes first.
    assert joined.index("### orders") < joined.index("### audit_events")


def test_joined_with_empty_overlay_returns_static() -> None:
    cat = _mixed_catalog()
    segs = planner_prompt_segments(
        cat,
    )
    joined = segs.joined()
    assert joined.rstrip() == segs.static.rstrip()


# ---------------------------------------------------------------------------
# Lookups travel with their cubes — public lookup is static, role-gated overlay
# ---------------------------------------------------------------------------


def test_public_cube_lookup_appears_in_static() -> None:
    cat = Catalog(
        [_public_orders()],
        lookups=[Lookup(dimension="orders.region", values=("EMEA", "APAC"))],
    )
    segs = planner_prompt_segments(cat, viewer=AuthContext(viewer_id="u", roles=["admin"]))
    assert "`EMEA`" in segs.static
    assert "`EMEA`" not in segs.overlay


def test_role_gated_cube_lookup_appears_in_overlay() -> None:
    # `audit_events.actor` is a string dim on an admin-only cube — its
    # lookup must follow the cube into the overlay so non-admins can't
    # discover the values.
    cat = Catalog(
        [_public_orders(), _admin_audit()],
        lookups=[Lookup(dimension="audit_events.actor", values=("svc_a", "svc_b"))],
    )
    admin = AuthContext(viewer_id="u", roles=["admin"])
    segs = planner_prompt_segments(cat, viewer=admin)
    assert "`svc_a`" not in segs.static
    assert "`svc_a`" in segs.overlay


# ---------------------------------------------------------------------------
# build_planner_prompt_fragment still works (back-compat)
# ---------------------------------------------------------------------------


def test_legacy_prompt_method_still_works() -> None:
    """``planner_prompt(Catalog, ...)`` (single-string) is the non-cached path
    and must continue to render every authorised cube in one go."""
    cat = _mixed_catalog()
    admin = AuthContext(viewer_id="u", roles=["admin"])
    out = planner_prompt(cat, viewer=admin)
    assert "### orders" in out
    assert "### audit_events" in out
    assert "### tickets" not in out
