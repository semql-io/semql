"""Unit tests for ``semql.prompt`` — catalog, planner, and router fragments.

The fragments are the LLM-facing contract: an unexpected change here
shows up as a planner that suddenly emits worse SemanticQuery specs.
Pinning the shape (sections present, exposed-vs-hidden cubes, the
required-filters callout, the spec contract block) catches drift
before it ships.
"""

from __future__ import annotations

from semql.model import Cube, Dialect, Dimension, Join, Measure, TimeDimension, View
from semql_prompt import (
    build_planner_prompt_fragment,
    build_router_prompt_fragment,
    render_catalog_block,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _orders(*, expose: bool = True, required: list[str] | None = None) -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        expose_in_prompt=expose,
        description="One row per order.",
        display_name="Customer Orders",
        required_filters=required or [],
        measures=[
            Measure(
                name="revenue",
                sql="{o}.amount",
                agg="sum",
                unit="currency",
                description="Total revenue",
                display_name="Total Revenue",
            ),
        ],
        dimensions=[
            Dimension(
                name="region",
                sql="{o}.region",
                type="string",
                description="ISO region code",
                display_name="Sales Region",
            ),
        ],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id"),
        ],
    )


def _hidden() -> Cube:
    return Cube(
        name="internal",
        backend=Dialect.POSTGRES,
        table="internal",
        alias="i",
        expose_in_prompt=False,
        description="Reserved for internal joins.",
        dimensions=[Dimension(name="x", sql="{i}.x", type="string")],
    )


# ---------------------------------------------------------------------------
# render_catalog_block — exposed / hidden, sections
# ---------------------------------------------------------------------------


def test_render_only_exposed_default() -> None:
    rendered = render_catalog_block({"orders": _orders(), "internal": _hidden()})
    assert "### orders" in rendered
    assert "### internal" not in rendered


def test_render_full_with_only_exposed_false() -> None:
    rendered = render_catalog_block(
        {"orders": _orders(), "internal": _hidden()}, only_exposed=False
    )
    assert "### orders" in rendered
    assert "### internal" in rendered


def test_render_empty_catalog_returns_empty_string() -> None:
    assert render_catalog_block({}) == ""


def test_render_only_hidden_cubes_returns_empty_string() -> None:
    assert render_catalog_block({"internal": _hidden()}) == ""


def test_render_includes_section_headers() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "## SEMANTIC CATALOG" in rendered
    assert "**Measures:**" in rendered
    assert "**Dimensions:**" in rendered
    assert "**Time dimensions:**" in rendered
    assert "**Joins:**" in rendered


def test_render_cube_description_shown() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "One row per order." in rendered


def test_render_measure_lists_unit_agg_description() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "`orders.revenue`" in rendered
    assert "[currency]" in rendered
    assert "`agg=sum`" in rendered
    assert "Total revenue" in rendered


def test_render_dimension_lists_type() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "`orders.region`" in rendered
    assert "`type=string`" in rendered


def test_render_time_dimension_lists_granularities() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "`orders.created_at`" in rendered
    assert "`granularities=hour|day|week|month`" in rendered


def test_render_join_includes_relationship() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "→ `customers`" in rendered
    assert "(many_to_one)" in rendered


# ---------------------------------------------------------------------------
# Required-filters callout
# ---------------------------------------------------------------------------


def test_required_filters_callout_present_when_declared() -> None:
    cube = _orders(required=["region"])
    rendered = render_catalog_block({"orders": cube})
    assert "**Required filters:**" in rendered
    assert "`orders.region`" in rendered
    assert "compile fails" in rendered


def test_required_filters_callout_absent_when_none() -> None:
    rendered = render_catalog_block({"orders": _orders(required=[])})
    assert "**Required filters:**" not in rendered


# ---------------------------------------------------------------------------
# display_name suffix
# ---------------------------------------------------------------------------


def test_display_name_suffix_on_cube() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "(human: Customer Orders)" in rendered


def test_display_name_suffix_on_measure() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "(human: Total Revenue)" in rendered


def test_display_name_suffix_on_dimension() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert "(human: Sales Region)" in rendered


# ---------------------------------------------------------------------------
# build_planner_prompt_fragment — sections present
# ---------------------------------------------------------------------------


def test_planner_fragment_contains_spec_contract() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert "## Semantic path" in rendered
    assert "`SemanticQuery`" in rendered
    assert "measures:" in rendered
    assert "dimensions:" in rendered
    assert "time_dimension:" in rendered
    assert "filters:" in rendered
    assert "having:" in rendered
    assert "order:" in rendered


def test_planner_fragment_contains_catalog() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert "## SEMANTIC CATALOG" in rendered
    assert "`orders.revenue`" in rendered


def test_planner_fragment_contains_raw_fallback() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert "## When to fall back to raw SQL" in rendered


def test_planner_fragment_omits_introspection_by_default() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert "## Introspecting the catalog" not in rendered


def test_planner_fragment_with_introspection_includes_meta_cubes() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()}, include_introspection=True)
    assert "## Introspecting the catalog" in rendered
    assert "`catalog_cubes`" in rendered
    assert "`catalog_measures`" in rendered
    assert "`catalog_dimensions`" in rendered


def test_planner_fragment_only_exposed_default() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders(), "internal": _hidden()})
    assert "### orders" in rendered
    assert "### internal" not in rendered


def test_planner_fragment_only_exposed_false_includes_hidden() -> None:
    rendered = build_planner_prompt_fragment(
        {"orders": _orders(), "internal": _hidden()}, only_exposed=False
    )
    assert "### internal" in rendered


def test_planner_fragment_documents_qualified_or_bare_having() -> None:
    """The HAVING contract advertises both bare and qualified
    measure references — pin so the prompt stays in sync with the
    compiler that accepts both."""
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert "revenue" in rendered
    assert "orders.revenue" in rendered


# ---------------------------------------------------------------------------
# build_router_prompt_fragment
# ---------------------------------------------------------------------------


def test_router_fragment_header_present() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()})
    assert "## Path routing — semantic vs raw SQL" in rendered


def test_router_fragment_topic_summary_default_on() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()})
    assert "## Catalog topics" in rendered
    assert "`orders`" in rendered
    assert "Customer Orders" in rendered  # display_name suffix


def test_router_fragment_topic_summary_can_be_disabled() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()}, include_topic_summary=False)
    assert "## Catalog topics" not in rendered


def test_router_fragment_only_exposed_default() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders(), "internal": _hidden()})
    # `orders` shows; `internal` is hidden by default.
    assert "`orders`" in rendered
    assert "`internal`" not in rendered


def test_router_fragment_only_exposed_false_includes_hidden() -> None:
    rendered = build_router_prompt_fragment(
        {"orders": _orders(), "internal": _hidden()}, only_exposed=False
    )
    assert "`internal`" in rendered


def test_router_fragment_lists_raw_triggers() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()})
    # Some signature triggers from _RAW_TRIGGERS — pin a couple so a
    # rewrite that drops the bulleted list surfaces here.
    assert "Window" in rendered or "window" in rendered
    assert "Recursive" in rendered or "recursive" in rendered


def test_router_fragment_omits_views_section_when_views_none() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()})
    assert "## Views" not in rendered


def test_router_fragment_includes_views_section_when_views_provided() -> None:
    views = {
        "revenue_overview": View(
            name="revenue_overview",
            display_name="Revenue Overview",
            description="Cross-cube revenue and region rollup.",
            fields={
                "revenue": "orders.revenue",
                "region": "orders.region",
            },
        ),
    }
    rendered = build_router_prompt_fragment({"orders": _orders()}, views=views)
    assert "## Views" in rendered
    assert "`revenue_overview`" in rendered
    assert "Revenue Overview" in rendered
    assert "Cross-cube revenue and region rollup" in rendered


def test_router_fragment_views_section_follows_topic_summary_toggle() -> None:
    views = {
        "revenue_overview": View(
            name="revenue_overview",
            fields={"revenue": "orders.revenue"},
        ),
    }
    rendered = build_router_prompt_fragment(
        {"orders": _orders()},
        include_topic_summary=False,
        views=views,
    )
    # When the topic summary is off, the view list (same catalog-content
    # signal) also disappears — the router stays pure routing rules.
    assert "## Catalog topics" not in rendered
    assert "## Views" not in rendered


def test_router_fragment_view_without_description_renders_cleanly() -> None:
    views = {
        "bare_view": View(
            name="bare_view",
            fields={"revenue": "orders.revenue"},
        ),
    }
    rendered = build_router_prompt_fragment({"orders": _orders()}, views=views)
    assert "`bare_view`" in rendered
    assert "— ." not in rendered  # no double-em-dash with empty body


# ---------------------------------------------------------------------------
# Empty-catalog edge cases
# ---------------------------------------------------------------------------


def test_planner_fragment_with_empty_catalog_still_renders_contract() -> None:
    rendered = build_planner_prompt_fragment({})
    assert "## Semantic path" in rendered
    # Catalog block is empty but the surrounding sections remain.
    assert "## When to fall back to raw SQL" in rendered


def test_router_fragment_with_empty_catalog_has_no_topic_lines() -> None:
    rendered = build_router_prompt_fragment({})
    # The header section still renders; topic summary is empty
    # (only the section heading).
    assert "## Path routing" in rendered
    # No bullet rows under topics — we asserted earlier that orders shows up.
    assert "`orders`" not in rendered


# ---------------------------------------------------------------------------
# Trailing newline contract — fragments end with a single newline so
# they splice cleanly into a larger system prompt.
# ---------------------------------------------------------------------------


def test_render_catalog_block_ends_with_single_newline() -> None:
    rendered = render_catalog_block({"orders": _orders()})
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_planner_fragment_ends_with_single_newline() -> None:
    rendered = build_planner_prompt_fragment({"orders": _orders()})
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")


def test_router_fragment_ends_with_single_newline() -> None:
    rendered = build_router_prompt_fragment({"orders": _orders()})
    assert rendered.endswith("\n")
    assert not rendered.endswith("\n\n")
