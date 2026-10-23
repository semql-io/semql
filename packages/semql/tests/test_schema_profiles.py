from __future__ import annotations

from typing import Any, cast

from semql.plan import QueryPlan
from semql.schema import SchemaProfile, SchemaProjection, project_schema


def _query_schema(projection: SchemaProjection) -> dict[str, Any]:
    schema = projection.schema
    step = schema["$defs"]["QueryStep"]
    query_ref = step["properties"]["query"]["$ref"]
    return cast("dict[str, Any]", schema["$defs"][query_ref.rsplit("/", 1)[-1]])


def test_tool_profile_keeps_recursive_query_features_and_reports_tuple_weakening() -> None:
    projection = project_schema(QueryPlan, SchemaProfile.TOOL)
    query = _query_schema(projection)
    assert "where" in query["properties"]
    assert "semi_joins" in query["properties"]
    assert projection.weakened_constraints == ()


def test_strict_profile_removes_recursive_query_features_and_preserves_query_requirement() -> None:
    projection = project_schema(QueryPlan, SchemaProfile.STRICT)
    query = _query_schema(projection)
    assert "where" not in query["properties"]
    assert "semi_joins" not in query["properties"]
    assert "query" in projection.schema["$defs"]["QueryStep"]["required"]
    assert {item.feature for item in projection.removed_features} >= {"where", "semi_joins"}
    assert projection.weakened_constraints


def test_strict_profile_reports_fixed_tuple_positional_weakening() -> None:
    projection = project_schema(QueryPlan, SchemaProfile.STRICT)
    assert any(
        item.constraint == "tuple positional validation" for item in projection.weakened_constraints
    )


def test_canonical_query_payload_rejects_unknown_nested_fields() -> None:
    projection = project_schema(QueryPlan, SchemaProfile.TOOL)
    payload: dict[str, Any] = {
        "steps": [{"intent": "headline", "query": {"measures": [], "unknown": True}}]
    }
    try:
        projection.validate_payload(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown nested semantic input was accepted")


def test_strict_profile_rejects_payload_that_smuggles_removed_operation() -> None:
    payload: dict[str, Any] = {
        "steps": [{"intent": "headline", "query": {"where": {"field": "orders.region"}}}]
    }
    try:
        project_schema(QueryPlan, SchemaProfile.STRICT).validate_payload(payload)
    except ValueError:
        pass
    else:
        raise AssertionError("strict profile accepted a removed operation")


def test_projection_validation_rejects_missing_nested_query() -> None:
    projection = project_schema(QueryPlan, SchemaProfile.TOOL)
    assert "query" in projection.schema["$defs"]["QueryStep"]["required"]
    try:
        projection.validate_payload({"steps": [{"intent": "headline"}]})
    except ValueError:
        pass
    else:
        raise AssertionError("missing QueryStep.query was accepted")
