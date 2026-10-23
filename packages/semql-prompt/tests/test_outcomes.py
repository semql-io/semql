from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import QueryPlan
from semql_prompt.outcomes import (
    CannotPlan,
    ContextRequest,
    NeedContext,
    NeedUserClarification,
    PlannerResponse,
    PlanOutcome,
)


def test_plan_outcome_requires_nonempty_plan() -> None:
    with pytest.raises(ValidationError):
        PlanOutcome(plan=QueryPlan())
    result = PlanOutcome(
        plan=QueryPlan.model_validate(
            {
                "steps": [
                    {
                        "query": {"measures": ["orders.revenue"]},
                        "intent": "headline",
                    }
                ]
            }
        )
    )
    assert len(result.plan.steps) == 1


def test_outcomes_are_discriminated_and_reject_extra_fields() -> None:
    response = PlannerResponse.model_validate(
        {
            "outcome": {
                "kind": "need_context",
                "requests": [
                    {
                        "request_id": "r1",
                        "decision_id": "d1",
                        "kind": "definition",
                        "target": "orders.revenue",
                    }
                ],
            }
        }
    )
    assert isinstance(response.outcome, NeedContext)
    with pytest.raises(ValidationError):
        PlannerResponse.model_validate(
            {
                "outcome": {
                    "kind": "plan",
                    "plan": {"steps": []},
                    "sql": "select 1",
                }
            }
        )


def test_need_context_has_bounded_typed_requests() -> None:
    request = ContextRequest(
        request_id="r1", decision_id="d1", kind="definition", target="orders.revenue"
    )
    outcome = NeedContext(requests=(request,))
    assert outcome.requests[0].target == "orders.revenue"
    with pytest.raises(ValidationError):
        ContextRequest.model_validate(
            {
                "request_id": "r1",
                "decision_id": "d1",
                "kind": "tool",
                "target": "https://example.test",
            }
        )


def test_context_result_requires_truthful_authority_and_exact_correlation() -> None:
    with pytest.raises(ValidationError):
        NeedContext.model_validate(
            {
                "requests": [
                    {
                        "request_id": "r1",
                        "decision_id": "d1",
                        "kind": "definition",
                        "target": "orders.revenue",
                    }
                ],
                "results": [
                    {
                        "request_id": "r1",
                        "decision_id": "other",
                        "status": "resolved",
                        "candidates": [
                            {
                                "id": "x",
                                "label": "X",
                                "value": "x",
                                "match_kind": "fuzzy",
                                "authoritative": True,
                            }
                        ],
                        "complete": True,
                    }
                ],
            }
        )


def test_clarification_choices_require_stable_ids() -> None:
    outcome = NeedUserClarification.model_validate(
        {
            "decision_id": "d1",
            "prompt": "Which region?",
            "choices": [{"id": "emea", "label": "Europe", "value": "EMEA"}],
        }
    )
    assert outcome.choices[0].id == "emea"
    with pytest.raises(ValidationError):
        NeedUserClarification.model_validate(
            {
                "decision_id": "d1",
                "prompt": "Which?",
                "choices": [{"label": "Europe", "value": "EMEA"}],
            }
        )


def test_cannot_plan_requires_safe_nonempty_diagnostics() -> None:
    with pytest.raises(ValidationError):
        CannotPlan(diagnostics=())
    result = CannotPlan.model_validate(
        {"diagnostics": [{"code": "invalid", "reason": "select * from customers"}]}
    )
    assert result.diagnostics[0].reason == "contract_violation"
