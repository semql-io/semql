"""Authorisation primitives — AuthContext + Cube.required_roles + viewer
threading through Catalog.compile / semql_prompt.planner_prompt / iter_cubes / the
prompt fragments.

The contract this file pins:
- viewer=None preserves today's behaviour (no auth filtering anywhere).
- required_roles is ANY-match: at least one role overlap.
- The compiler refuses queries that touch unauthorised cubes (loud
  error, not silent filtering).
- iter_cubes / prompt fragments hide unauthorised cubes (silent filtering
  is the right behaviour for *discovery*, since the consumer's whole
  point is "what can this viewer see?").
- A Catalog policy override AND-composes with required_roles.
- viewer.viewer_id auto-binds to ctx.viewer_id so security_sql can
  reference it without manual context plumbing.
"""

from __future__ import annotations

import pytest
from semql import (
    AuthContext,
    Backend,
    Catalog,
    Cube,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.errors import CompileError
from semql.introspect import iter_cubes, viewer_sees
from semql_prompt import (
    build_planner_prompt_fragment,
    build_router_prompt_fragment,
    planner_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _public_cube() -> Cube:
    return Cube(
        name="public",
        backend=Backend.POSTGRES,
        table="public_data",
        alias="p",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="region", sql="{p}.region", type="string")],
    )


def _finance_cube() -> Cube:
    return Cube(
        name="finance",
        backend=Backend.POSTGRES,
        table="finance_data",
        alias="f",
        required_roles=["finance"],
        measures=[Measure(name="revenue", sql="{f}.amount", agg="sum")],
        dimensions=[Dimension(name="month", sql="{f}.month", type="string")],
    )


def _hr_cube() -> Cube:
    return Cube(
        name="hr",
        backend=Backend.POSTGRES,
        table="hr_data",
        alias="h",
        required_roles=["hr", "admin"],  # ANY of these
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="dept", sql="{h}.dept", type="string")],
    )


def _catalog() -> Catalog:
    return Catalog([_public_cube(), _finance_cube(), _hr_cube()])


# ---------------------------------------------------------------------------
# viewer_sees
# ---------------------------------------------------------------------------


def test_viewer_none_short_circuits_to_visible() -> None:
    """No viewer = no auth filtering, today's default."""
    assert viewer_sees(_finance_cube(), viewer=None, policy=None) is True


def test_empty_required_roles_means_open_cube() -> None:
    """A cube with no required_roles is visible to any authenticated viewer."""
    viewer = AuthContext(viewer_id="u1", roles=[])
    assert viewer_sees(_public_cube(), viewer=viewer, policy=None) is True


def test_any_match_grants_visibility() -> None:
    """ANY-match: holding one of the listed roles is enough."""
    viewer = AuthContext(viewer_id="u1", roles=["hr"])
    assert viewer_sees(_hr_cube(), viewer=viewer, policy=None) is True


def test_no_role_overlap_hides_cube() -> None:
    viewer = AuthContext(viewer_id="u1", roles=["analyst"])
    assert viewer_sees(_finance_cube(), viewer=viewer, policy=None) is False


def test_policy_override_can_deny_even_when_roles_match() -> None:
    """Policy AND-composes with required_roles — both must pass."""

    def deny_all(_cube: Cube, _viewer: AuthContext) -> bool:
        return False

    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    assert viewer_sees(_finance_cube(), viewer=viewer, policy=deny_all) is False


def test_policy_override_cannot_grant_when_roles_fail() -> None:
    """required_roles is a hard gate; policy can only further restrict."""

    def always_allow(_cube: Cube, _viewer: AuthContext) -> bool:
        return True

    viewer = AuthContext(viewer_id="u1", roles=["analyst"])
    # roles fail first; policy never gets to vote yes for it.
    assert viewer_sees(_finance_cube(), viewer=viewer, policy=always_allow) is False


# ---------------------------------------------------------------------------
# iter_cubes filtering
# ---------------------------------------------------------------------------


def test_iter_cubes_without_viewer_yields_all() -> None:
    cat = _catalog()
    names = {c.name for c in iter_cubes(cat)}
    assert {"public", "finance", "hr"}.issubset(names)


def test_iter_cubes_with_viewer_filters_unauthorised() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    names = {c.name for c in iter_cubes(cat, viewer=viewer)}
    assert names == {"public", "finance"}  # hr hidden


def test_iter_cubes_with_policy_further_restricts() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance", "hr"])

    def hide_hr(cube: Cube, _viewer: AuthContext) -> bool:
        return cube.name != "hr"

    names = {c.name for c in iter_cubes(cat, viewer=viewer, policy=hide_hr)}
    assert names == {"public", "finance"}


# ---------------------------------------------------------------------------
# Compiler enforcement
# ---------------------------------------------------------------------------


def test_compile_refuses_query_against_unauthorised_cube() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["analyst"])  # no finance
    q = SemanticQuery(measures=["finance.revenue"])
    with pytest.raises(CompileError, match="not authorised"):
        cat.compile(q, viewer=viewer)


def test_compile_allows_authorised_query() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    q = SemanticQuery(measures=["finance.revenue"], dimensions=["finance.month"])
    compiled = cat.compile(q, viewer=viewer)
    assert "amount" in compiled.sql.lower()


def test_compile_without_viewer_preserves_today_behaviour() -> None:
    """Backward compat: no viewer = no auth check, queries that would
    fail under auth still succeed without it."""
    cat = _catalog()
    q = SemanticQuery(measures=["finance.revenue"])
    compiled = cat.compile(q)  # no viewer
    assert compiled.sql  # didn't raise


def test_compile_auto_binds_viewer_id_to_ctx() -> None:
    """``security_sql`` referencing ``{ctx.viewer_id}`` should resolve
    from ``viewer.viewer_id`` without the caller passing context."""
    scoped_cube = Cube(
        name="my_tickets",
        backend=Backend.POSTGRES,
        table="tickets",
        alias="t",
        security_sql="{t}.assignee = {ctx.viewer_id}",
        measures=[Measure(name="count", sql="*", agg="count")],
    )
    cat = Catalog([scoped_cube])
    viewer = AuthContext(viewer_id="alice@example.com", roles=[])
    q = SemanticQuery(measures=["my_tickets.count"])
    compiled = cat.compile(q, viewer=viewer)
    # viewer_id never inlined as a SQL literal — it must round-trip
    # through the bound params dict.
    assert "alice@example.com" not in compiled.sql
    assert "alice@example.com" in compiled.params.values()


def test_compile_explicit_context_viewer_id_wins_over_auto() -> None:
    """When the caller passes ``context={'ctx.viewer_id': X}`` explicitly,
    that wins over the viewer's auto-flatten — the caller knows best."""
    scoped_cube = Cube(
        name="my_tickets",
        backend=Backend.POSTGRES,
        table="tickets",
        alias="t",
        security_sql="{t}.assignee = {ctx.viewer_id}",
        measures=[Measure(name="count", sql="*", agg="count")],
    )
    cat = Catalog([scoped_cube])
    viewer = AuthContext(viewer_id="alice@example.com", roles=[])
    q = SemanticQuery(measures=["my_tickets.count"])
    compiled = cat.compile(q, viewer=viewer, context={"ctx.viewer_id": "impersonated@example.com"})
    assert "impersonated@example.com" in compiled.params.values()
    assert "alice@example.com" not in compiled.params.values()


# ---------------------------------------------------------------------------
# Catalog.policy override hook
# ---------------------------------------------------------------------------


def test_catalog_policy_filters_in_iter_cubes() -> None:
    def deny_finance(cube: Cube, _viewer: AuthContext) -> bool:
        return cube.name != "finance"

    cat = Catalog(
        [_public_cube(), _finance_cube(), _hr_cube()],
        policy=deny_finance,
    )
    viewer = AuthContext(viewer_id="u1", roles=["finance", "hr"])
    # Need to pass viewer for the policy to apply.
    names = {c.name for c in iter_cubes(cat, viewer=viewer, policy=cat.policy)}
    assert "finance" not in names


def test_catalog_compile_uses_registered_policy() -> None:
    def deny_finance(cube: Cube, _viewer: AuthContext) -> bool:
        return cube.name != "finance"

    cat = Catalog(
        [_public_cube(), _finance_cube(), _hr_cube()],
        policy=deny_finance,
    )
    viewer = AuthContext(viewer_id="u1", roles=["finance", "hr"])
    q = SemanticQuery(measures=["finance.revenue"])
    with pytest.raises(CompileError, match="not authorised"):
        cat.compile(q, viewer=viewer)


# ---------------------------------------------------------------------------
# Prompt fragments filter by viewer
# ---------------------------------------------------------------------------


def test_planner_fragment_hides_unauthorised_cubes() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    rendered = build_planner_prompt_fragment(cat.as_dict(), viewer=viewer)
    assert "public" in rendered
    assert "finance" in rendered
    assert "### hr " not in rendered  # hr is hidden


def test_router_fragment_hides_unauthorised_topics() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    rendered = build_router_prompt_fragment(cat.as_dict(), viewer=viewer)
    assert "`finance`" in rendered
    assert "`hr`" not in rendered


def test_catalog_prompt_method_threads_viewer() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["finance"])
    rendered = planner_prompt(cat, viewer=viewer)
    assert "`hr`" not in rendered  # never proposed to a non-hr viewer
