from __future__ import annotations

from typing import Any, cast

# jsonschema supplies no inline typing; validate the actual provider schema.
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from semql import AuthContext, Catalog, Cube, Dialect, Dimension
from semql.schema import SchemaProfile, SemanticFeature
from semql_prompt.bedrock import to_bedrock_converse_tools
from semql_prompt.catalog_tools import planner_prompt_segments, prompt_hash, to_openai_tools
from semql_prompt.generation import build_generation_contract
from semql_prompt.outcomes import PlannerResponse
from semql_prompt.prompt import build_query_generator_prompt_fragment


def _catalog() -> Catalog:
    return Catalog(
        [
            Cube(
                name="orders",
                dialect=Dialect.POSTGRES,
                table="orders",
                alias="o",
                dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            )
        ]
    )


def _semantic_query_schema(schema: dict[str, Any]) -> dict[str, Any]:
    definitions_value: Any = schema.get("$defs", {})
    if isinstance(definitions_value, dict):
        definitions = cast(dict[str, Any], definitions_value)
        for definition_value in definitions.values():
            if isinstance(definition_value, dict):
                definition = cast(dict[str, Any], definition_value)
                if definition.get("title") == "SemanticQuery":
                    return definition
    return schema


def test_generation_contract_pairs_profile_schema_guidance_and_canonical_parser() -> None:
    contract = build_generation_contract(
        _catalog().as_dict(),
        profile=SchemaProfile.STRICT,
        disabled_features=(SemanticFeature.RANK,),
    )
    assert contract.output_model is PlannerResponse
    assert "PlannerResponse" in contract.text
    assert "where" in contract.text and "semi_joins" in contract.text
    semantic = _semantic_query_schema(contract.schema)
    assert "where" not in semantic["properties"]
    assert "semi_joins" not in semantic["properties"]
    assert "rank" not in semantic["properties"]
    assert {item.feature for item in contract.report.removed_features} >= {
        "where",
        "semi_joins",
        "rank",
    }
    try:
        contract.validate_payload(
            {
                "outcome": {
                    "kind": "plan",
                    "plan": {
                        "steps": [
                            {
                                "intent": "headline",
                                "query": {"where": None},
                            }
                        ]
                    },
                }
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("strict generation contract accepted removed input")


def test_tool_exports_and_bedrock_use_the_explicit_schema_profile() -> None:
    catalog = _catalog()
    tool = to_openai_tools(catalog, profile=SchemaProfile.STRICT)[0]["function"]
    bedrock = to_bedrock_converse_tools(catalog, profile=SchemaProfile.STRICT)[0]["toolSpec"]
    assert tool["parameters"] == bedrock["inputSchema"]["json"]
    assert "where" not in _semantic_query_schema(tool["parameters"])["properties"]
    assert (
        "where"
        in _semantic_query_schema(
            to_openai_tools(catalog, profile=SchemaProfile.TOOL)[0]["function"]["parameters"]
        )["properties"]
    )


def test_profile_static_prompt_and_cache_hash_are_separated() -> None:
    catalog = _catalog()
    rich = planner_prompt_segments(catalog, profile=SchemaProfile.TOOL)
    strict = planner_prompt_segments(catalog, profile=SchemaProfile.STRICT)
    assert rich.static != strict.static
    assert prompt_hash(catalog, profile=SchemaProfile.TOOL) != prompt_hash(
        catalog, profile=SchemaProfile.STRICT
    )
    assert prompt_hash(catalog) != prompt_hash(catalog, disabled_features=(SemanticFeature.RANK,))


def test_viewer_specific_catalog_overlay_stays_out_of_shared_cache_segment() -> None:
    protected = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        dimensions=[
            Dimension(
                name="salary",
                sql="{o}.salary",
                type="string",
                required_roles=["finance"],
            )
        ],
    )
    catalog = Catalog([protected])
    reader = planner_prompt_segments(catalog, viewer=AuthContext(viewer_id="reader", roles=[]))
    finance = planner_prompt_segments(
        catalog, viewer=AuthContext(viewer_id="finance", roles=["finance"])
    )
    assert reader.static == finance.static
    assert reader.overlay != finance.overlay
    assert "salary" not in reader.overlay
    assert "salary" in finance.overlay


def test_disabled_rank_is_not_guided_or_advertised() -> None:
    text = build_query_generator_prompt_fragment(
        _catalog().as_dict(), disabled_features=(SemanticFeature.RANK,)
    )
    assert "Ranking is disabled" in text
    projection = build_generation_contract(
        _catalog().as_dict(), disabled_features=(SemanticFeature.RANK,)
    )
    assert "rank" not in _semantic_query_schema(projection.schema)["properties"]


def test_generation_schema_copy_cannot_mutate_contract_or_leak_hidden_fields() -> None:
    hidden = Cube(
        name="private",
        dialect=Dialect.POSTGRES,
        table="private",
        alias="p",
        dimensions=[
            Dimension(
                name="payroll",
                sql="{p}.salary",
                type="string",
                required_roles=["finance"],
            )
        ],
    )
    catalog = Catalog([*_catalog(), hidden])
    from semql_prompt.adaptive import AdaptivePromptRequest, build_adaptive_prompt

    result = build_adaptive_prompt(
        catalog,
        AdaptivePromptRequest(question="orders region"),
        viewer=AuthContext(viewer_id="reader", roles=[]),
        count_tokens=len,
    )
    assert result.token_count == len(result.text)
    assert "payroll" not in result.text
    original_schema = result.generation_contract.schema
    schema = result.generation_contract.schema
    schema["$defs"].clear()
    assert result.generation_contract.schema == original_schema


def test_continuation_is_required_context_not_budget_trimmable() -> None:
    from semql.plan import QueryPlan, QueryStep
    from semql.spec import SemanticQuery
    from semql_prompt.adaptive import (
        AdaptivePromptPolicy,
        AdaptivePromptRequest,
        build_adaptive_prompt,
    )
    from semql_prompt.continuation import ContinuationState

    catalog = _catalog()
    state = ContinuationState(
        request_id="req-1",
        revision=1,
        baseline=QueryPlan(steps=[QueryStep(query=SemanticQuery(), intent="headline")]),
        authorized_cubes=("orders",),
        visibility_revision="vis-1",
    )
    result = build_adaptive_prompt(
        catalog,
        AdaptivePromptRequest(question="orders"),
        profile=SchemaProfile.STRICT,
        policy=AdaptivePromptPolicy(max_tokens=1),
        count_tokens=len,
        continuation=state,
        current_authorized_cubes=("orders",),
        current_visibility_revision="vis-1",
    )
    assert not result.fits
    assert "## REQUIRED CONTINUATION" in result.text
    assert result.token_count == len(result.text)


def test_profiled_langchain_schema_generation_preserves_registered_refs_and_shape() -> None:
    # Exercise provider argument models even without the optional LangChain runtime.
    from semql_prompt.catalog_tools import (
        _profiled_query_model,  # pyright: ignore[reportPrivateUsage]
    )

    strict_model = _profiled_query_model(SchemaProfile.STRICT, ())
    strict_schema = strict_model.model_json_schema()
    Draft202012Validator.check_schema(strict_schema)
    semantic = _semantic_query_schema(strict_schema)
    assert "where" not in semantic["properties"]
    assert "semi_joins" not in semantic["properties"]
    assert strict_schema["type"] == "object"
    assert "measures" in semantic["properties"]
    try:
        strict_model.model_validate({"where": None})
    except ValueError:
        pass
    else:
        raise AssertionError("strict profiled model accepted an explicitly omitted feature")


def test_disabled_rank_profiled_schema_keeps_recursive_tool_shape() -> None:
    # Exercise provider argument models even without the optional LangChain runtime.
    from semql_prompt.catalog_tools import (
        _profiled_query_model,  # pyright: ignore[reportPrivateUsage]
    )

    model = _profiled_query_model(SchemaProfile.TOOL, (SemanticFeature.RANK,))
    schema = model.model_json_schema()
    Draft202012Validator.check_schema(schema)
    semantic = _semantic_query_schema(schema)
    assert "rank" not in semantic["properties"]
    assert "semi_joins" in semantic["properties"]
    assert "where" in semantic["properties"]
    assert schema["$defs"]["SemiJoin"]["properties"]["source"]["$ref"].endswith("/SemanticQuery")

    try:
        model.model_validate(
            {
                "rank": {
                    "partition_by": ["orders.region"],
                    "order_by": [{"field": "orders.revenue", "direction": "desc"}],
                    "count": 1,
                }
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("rank-disabled profiled model accepted rank")
