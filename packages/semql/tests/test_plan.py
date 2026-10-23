"""Smoke tests for the prompt-pipeline output models.

These models are the contract between the four prompt roles and
their callers. Each test pins one shape so a planner emitting valid
JSON gets parsed correctly, and an invalid shape (wrong literal,
missing required field) fails at the validator boundary.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    DrilldownSuggestion,
    DrilldownSuggestions,
    Presentation,
    QueryPlan,
    QueryStep,
    RouterDecision,
    SemanticQuery,
)


def test_router_decision_semantic_route_carries_cubes_views() -> None:
    d = RouterDecision(
        route_to="semantic",
        cubes=["orders"],
        views=["revenue_overview"],
        reasoning="The question asks for revenue rollup.",
    )
    assert d.route_to == "semantic"
    assert d.cubes == ["orders"]
    assert d.views == ["revenue_overview"]


def test_router_decision_raw_route_has_empty_surfaces() -> None:
    d = RouterDecision(route_to="raw")
    assert d.cubes == []
    assert d.views == []


def test_router_decision_rejects_unknown_route() -> None:
    with pytest.raises(ValidationError):
        RouterDecision(route_to="hybrid")  # type: ignore[arg-type]


def test_router_decision_is_frozen() -> None:
    d = RouterDecision(route_to="semantic")
    with pytest.raises(ValidationError):
        d.route_to = "raw"


def test_query_step_intent_is_closed_literal() -> None:
    q = SemanticQuery(measures=["orders.revenue"])
    QueryStep(query=q, intent="headline")
    QueryStep(query=q, intent="breakdown")
    QueryStep(query=q, intent="compare")
    QueryStep(query=q, intent="context")
    with pytest.raises(ValidationError):
        QueryStep(query=q, intent="bespoke")  # type: ignore[arg-type]


def test_query_plan_carries_ordered_steps() -> None:
    q = SemanticQuery(measures=["orders.revenue"])
    plan = QueryPlan(
        steps=[
            QueryStep(query=q, intent="headline", label="Total"),
            QueryStep(query=q, intent="breakdown", label="By region"),
        ],
    )
    assert len(plan.steps) == 2
    assert plan.steps[0].intent == "headline"


def test_query_plan_deeply_freezes_steps_and_rejects_unknown_fields() -> None:
    steps = [{"query": {"measures": ["orders.revenue"]}, "intent": "headline"}]
    plan = QueryPlan.model_validate({"steps": steps})
    steps.clear()
    assert len(plan.steps) == 1
    with pytest.raises((AttributeError, TypeError)):
        plan.steps.append(plan.steps[0])  # type: ignore[attr-defined]
    with pytest.raises(ValidationError):
        QueryPlan.model_validate({"steps": [], "sql": "select 1"})


def test_query_step_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        QueryStep(
            query=SemanticQuery(measures=["orders.revenue"]),
            intent="headline",
            sql="select 1",  # type: ignore[call-arg]
        )


def test_presentation_summary_required_others_optional() -> None:
    p = Presentation(summary="Revenue up 12%.")
    assert p.highlights == []
    assert p.caveats == []


def test_drilldown_suggestion_carries_ready_to_run_query() -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    s = DrilldownSuggestion(label="Break down by region", query=q)
    assert s.query is q
    assert s.rationale is None


def test_drilldown_suggestions_collection_holds_suggestions() -> None:
    q = SemanticQuery(measures=["orders.revenue"])
    coll = DrilldownSuggestions(
        suggestions=[
            DrilldownSuggestion(label="By region", query=q),
            DrilldownSuggestion(label="By product", query=q),
        ],
        focus="Q4 revenue spike",
    )
    assert len(coll.suggestions) == 2
    assert coll.focus == "Q4 revenue spike"
