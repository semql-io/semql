"""Smoke tests for the four prompt-pipeline fragment builders.

Each builder owns one role in the question-answering pipeline. The
contracts pinned here are *shape* contracts — what sections must
appear, what schema gets embedded — so a refactor that drops a
section or breaks the structured-output contract surfaces here.
"""

from __future__ import annotations

from semql import (
    AuthContext,
    Cube,
    Dialect,
    Dimension,
    Measure,
    TimeDimension,
    View,
)
from semql_prompt import (
    build_drilldown_prompt_fragment,
    build_presenter_prompt_fragment,
    build_query_generator_prompt_fragment,
    build_router_prompt_fragment,
)


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        description="One row per order.",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="city", sql="{o}.city", type="string"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        drill_paths=[["region", "city"]],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )


def _catalog() -> dict[str, Cube]:
    return {"orders": _orders(), "customers": _customers()}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_fragment_emits_router_decision_schema() -> None:
    rendered = build_router_prompt_fragment(_catalog())
    # Schema section is present so a typed-output LLM client can parse.
    assert "RouterDecision" in rendered
    assert '"semantic"' in rendered or "semantic" in rendered
    assert '"raw"' in rendered or "raw" in rendered


# ---------------------------------------------------------------------------
# Query Generator
# ---------------------------------------------------------------------------


def test_generator_fragment_includes_spec_and_schema() -> None:
    rendered = build_query_generator_prompt_fragment(_catalog())
    assert "Semantic path" in rendered  # spec contract present
    assert "QueryPlan" in rendered  # output schema present
    assert "intent" in rendered
    # Intent vocabulary is documented for the model.
    assert "headline" in rendered
    assert "breakdown" in rendered


def test_generator_scope_to_trims_catalog() -> None:
    rendered = build_query_generator_prompt_fragment(_catalog(), scope_to=["orders"])
    assert "### orders" in rendered
    assert "### customers" not in rendered  # excluded by scope_to


def test_generator_scope_to_trims_views_too() -> None:
    views = {
        "rev_view": View(name="rev_view", fields={"r": "orders.revenue"}),
        "other_view": View(name="other_view", fields={"n": "customers.name"}),
    }
    rendered = build_query_generator_prompt_fragment(
        _catalog(), scope_to=["orders", "rev_view"], views=views
    )
    assert "rev_view" in rendered
    assert "other_view" not in rendered


def test_generator_filters_by_viewer() -> None:
    restricted_cube = _orders().model_copy(update={"required_roles": ["finance"]})
    cat = {"orders": restricted_cube, "customers": _customers()}
    viewer = AuthContext(viewer_id="u1", roles=["analyst"])  # no finance
    rendered = build_query_generator_prompt_fragment(cat, viewer=viewer)
    assert "### orders" not in rendered
    assert "### customers" in rendered


# ---------------------------------------------------------------------------
# Presenter
# ---------------------------------------------------------------------------


def test_presenter_fragment_emits_presentation_schema() -> None:
    rendered = build_presenter_prompt_fragment()
    assert "Presentation" in rendered
    assert "summary" in rendered
    assert "highlights" in rendered
    assert "caveats" in rendered


def test_presenter_includes_query_labels_when_provided() -> None:
    rendered = build_presenter_prompt_fragment(query_labels=["Q4 revenue", "Q3 revenue (compare)"])
    assert "Q4 revenue" in rendered
    assert "Q3 revenue" in rendered


def test_presenter_includes_result_summary_when_provided() -> None:
    rendered = build_presenter_prompt_fragment(result_summary="3 rows; max=12_400; min=8_200")
    assert "12_400" in rendered


# ---------------------------------------------------------------------------
# Drilldown
# ---------------------------------------------------------------------------


def test_drilldown_fragment_emits_suggestions_schema() -> None:
    rendered = build_drilldown_prompt_fragment(_orders())
    assert "DrilldownSuggestions" in rendered
    assert "label" in rendered
    assert "query" in rendered


def test_drilldown_lists_cube_measures_and_dimensions() -> None:
    rendered = build_drilldown_prompt_fragment(_orders())
    assert "orders.revenue" in rendered
    assert "orders.region" in rendered
    assert "orders.created_at" in rendered


def test_drilldown_renders_declared_drill_paths() -> None:
    rendered = build_drilldown_prompt_fragment(_orders())
    # ``[["region", "city"]]`` should appear as "region → city".
    assert "region" in rendered and "city" in rendered
    assert "→" in rendered  # drill-path arrow


def test_drilldown_carries_focused_row_context() -> None:
    rendered = build_drilldown_prompt_fragment(
        _orders(), focused_row={"region": "EMEA", "month": "2024-04"}
    )
    assert "EMEA" in rendered
    assert "2024-04" in rendered


def test_drilldown_paths_hint_can_be_disabled() -> None:
    cube = _orders()
    rendered = build_drilldown_prompt_fragment(cube, drill_paths_hint=False)
    assert "Declared drill paths" not in rendered
