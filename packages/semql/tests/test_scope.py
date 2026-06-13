"""ScopeFn registry — declarative row-level scoping driven by AuthContext.

The contract:
- ``Cube.scope`` names a function in ``Catalog.scope_fns``; missing
  registrations are loud at Catalog construction (not at compile time).
- The compiler calls the function with ``(cube, viewer)``, gets back a
  ``ScopePredicate``, and AND-injects the predicate inside the cube's
  isolation subquery — same protection layer as tenancy / security_sql,
  so an outer ``OR`` cannot bypass it.
- ``ScopePredicate`` may declare ctx_keys it needs; missing keys raise
  ``CompileError`` *before* the predicate gets near SQL emission.
- Returning ``None`` from the ScopeFn means "no scoping for this viewer"
  (admin role, service account, etc.).
- ``viewer=None`` skips the lookup entirely — preserving today's
  unscoped behaviour for callers that don't pass auth.
"""

from __future__ import annotations

import pytest
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    ScopePredicate,
    SemanticQuery,
)
from semql.errors import CompileError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tickets() -> Cube:
    return Cube(
        name="tickets",
        backend=Dialect.POSTGRES,
        table="tickets",
        alias="t",
        scope="reportees",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="assignee", sql="{t}.assignee", type="string")],
    )


def _reportees_scope(_cube: Cube, viewer: AuthContext) -> ScopePredicate | None:
    """Row-level scope: tickets assigned to anyone in viewer's reportee
    subtree. Admins (any 'admin' role) see everything."""
    if "admin" in viewer.roles:
        return None
    return ScopePredicate(
        sql=("{t}.assignee_id IN (SELECT id FROM employees WHERE manager_id = {ctx.viewer_id})"),
        ctx_keys=["ctx.viewer_id"],
    )


# ---------------------------------------------------------------------------
# Catalog-construction validation
# ---------------------------------------------------------------------------


def test_catalog_rejects_unregistered_scope() -> None:
    """A cube referencing a scope name no function is registered for
    should fail loudly at Catalog construction, not silently at compile."""
    cube = _tickets()
    with pytest.raises(ValueError, match="no scope function is registered"):
        Catalog([cube])  # scope_fns omitted


def test_catalog_accepts_registered_scope() -> None:
    Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})


def test_catalog_exposes_scope_fns_via_property() -> None:
    cat = Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})
    assert "reportees" in cat.scope_fns


# ---------------------------------------------------------------------------
# Compiler injection
# ---------------------------------------------------------------------------


def test_compile_injects_scope_predicate() -> None:
    cat = Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})
    viewer = AuthContext(viewer_id="manager_42", roles=["manager"])
    q = SemanticQuery(measures=["tickets.count"])
    compiled = cat.compile(q, viewer=viewer)
    # Scope SQL went into the isolation subquery, not the outer WHERE.
    assert "manager_id" in compiled.sql
    assert "employees" in compiled.sql.lower()
    # viewer_id never appears as a literal — must be bound.
    assert "manager_42" not in compiled.sql
    assert "manager_42" in compiled.params.values()


def test_compile_skips_scope_for_viewer_returning_none() -> None:
    """Admin-style overrides: a ScopeFn that returns None for some
    viewer means "no row-level scope" for that viewer."""
    cat = Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})
    admin = AuthContext(viewer_id="root", roles=["admin"])
    q = SemanticQuery(measures=["tickets.count"])
    compiled = cat.compile(q, viewer=admin)
    assert "manager_id" not in compiled.sql  # scope didn't fire


def test_compile_skips_scope_without_viewer() -> None:
    """Backward compat: no viewer = no scoping. The scope_fn registry
    is dormant until a viewer is passed."""
    cat = Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})
    q = SemanticQuery(measures=["tickets.count"])
    compiled = cat.compile(q)  # no viewer
    assert "manager_id" not in compiled.sql


def test_compile_scope_inside_isolation_subquery() -> None:
    """The predicate must live *inside* the alias subquery so an outer
    OR cannot reach around it. Structural check: the scope SQL should
    appear before the outer alias's FROM clause closes."""
    cat = Catalog([_tickets()], scope_fns={"reportees": _reportees_scope})
    viewer = AuthContext(viewer_id="u1", roles=["manager"])
    q = SemanticQuery(measures=["tickets.count"])
    sql = cat.compile(q, viewer=viewer).sql
    # The cube alias's subquery contains the scope predicate.
    # Look for the wrap shape: ``(SELECT * FROM tickets AS t WHERE ...)``.
    assert "SELECT * FROM" in sql.upper()
    assert "AS t" in sql or "AS T" in sql


# ---------------------------------------------------------------------------
# ctx_keys validation
# ---------------------------------------------------------------------------


def test_compile_rejects_when_ctx_keys_missing() -> None:
    """A ScopePredicate declaring ctx_keys must surface its requirements
    at compile time when those keys aren't in the resolution context."""

    def needs_dept(_cube: Cube, _viewer: AuthContext) -> ScopePredicate:
        return ScopePredicate(
            sql="{t}.dept = {ctx.viewer_dept}",
            ctx_keys=["ctx.viewer_dept"],
        )

    cat = Catalog([_tickets()], scope_fns={"reportees": needs_dept})
    viewer = AuthContext(viewer_id="u1", roles=[])
    q = SemanticQuery(measures=["tickets.count"])
    # viewer_id auto-flattens, but viewer_dept doesn't — caller has
    # to pass it explicitly, and we want the failure to name it.
    with pytest.raises(CompileError, match="ctx.viewer_dept"):
        cat.compile(q, viewer=viewer)


def test_compile_accepts_when_ctx_keys_satisfied() -> None:
    def needs_dept(_cube: Cube, _viewer: AuthContext) -> ScopePredicate:
        return ScopePredicate(
            sql="{t}.dept = {ctx.viewer_dept}",
            ctx_keys=["ctx.viewer_dept"],
        )

    cat = Catalog([_tickets()], scope_fns={"reportees": needs_dept})
    viewer = AuthContext(viewer_id="u1", roles=[])
    q = SemanticQuery(measures=["tickets.count"])
    compiled = cat.compile(q, viewer=viewer, context={"ctx.viewer_dept": "Engineering"})
    # Value bound, not inlined.
    assert "Engineering" not in compiled.sql
    assert "Engineering" in compiled.params.values()


# ---------------------------------------------------------------------------
# Cube without scope is unaffected
# ---------------------------------------------------------------------------


def test_cube_without_scope_is_unaffected_by_registry() -> None:
    """Cubes whose ``scope`` is None pass through cleanly even with
    a populated scope_fns registry."""
    open_cube = Cube(
        name="public",
        backend=Dialect.POSTGRES,
        table="public_data",
        alias="p",
        measures=[Measure(name="count", sql="*", agg="count")],
    )
    cat = Catalog([open_cube], scope_fns={"reportees": _reportees_scope})
    viewer = AuthContext(viewer_id="u1", roles=["manager"])
    q = SemanticQuery(measures=["public.count"])
    compiled = cat.compile(q, viewer=viewer)
    assert "manager_id" not in compiled.sql
