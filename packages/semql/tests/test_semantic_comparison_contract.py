"""Semantic comparison follows definitions, never artifact-local spellings."""

from __future__ import annotations

import copy

import pytest
from semql.analysis import (
    CatalogContext,
    Coverage,
    SemanticAnalysis,
    SemanticBinding,
    SemanticNode,
    SemanticOutput,
    SemanticPopulation,
    SemanticResult,
    compare_analysis,
)
from semql.errors import ContractError


def _artifact(*, suffix: str = "", filtered: bool = False) -> SemanticAnalysis:
    population = "population" + suffix
    dimension = "dimension" + suffix
    amount = "amount" + suffix
    predicate = "predicate" + suffix
    binding = "binding" + suffix
    nodes = [
        SemanticNode(node_id=dimension, kind="dimension", catalog_ref="events.region"),
        SemanticNode(
            node_id=amount,
            kind="aggregate",
            catalog_ref="events.amount",
            aggregate="sum",
            logical_stage="event_to_group",
            population_ref=population,
            null_policy="ignore_null_inputs",
            unit="currency",
            unit_state="declared",
        ),
    ]
    if filtered:
        nodes.append(
            SemanticNode(
                node_id=predicate,
                kind="predicate",
                catalog_ref="events.region",
                aggregate="eq",
                logical_stage="where",
                binding_refs=(binding,),
            )
        )
    return SemanticAnalysis(
        catalog_context=CatalogContext(namespace="reporting", semantic_revision="r17"),
        nodes=tuple(nodes),
        outputs=(
            SemanticOutput(
                output_id="group" + suffix, node_id=dimension, sql_alias="region" + suffix
            ),
            SemanticOutput(output_id="metric" + suffix, node_id=amount, sql_alias="net" + suffix),
        ),
        populations=(
            SemanticPopulation(
                population_id=population,
                inclusion="observed_facts",
                source_refs=("events",),
                predicate_ids=(predicate,) if filtered else (),
            ),
        ),
        binding_dependencies=(
            SemanticBinding(
                slot=binding,
                kind="filter",
                reference="events.region",
                operator="eq",
                stage="where",
                arity=1,
            ),
        )
        if filtered
        else (),
        result=SemanticResult(grain_node_ids=(dimension,), population_ref=population),
        coverage=Coverage(status="complete"),
    )


def test_all_local_reference_domains_and_aliases_can_be_renamed() -> None:
    left, right = _artifact(filtered=True), _artifact(suffix="_other", filtered=True)
    right = right.model_copy(update={"outputs": tuple(reversed(right.outputs))})
    assert compare_analysis(left, right).outcome == "equivalent"
    assert compare_analysis(left, right, scope="result").outcome == "not_established"
    assert (
        compare_analysis(
            left,
            right,
            scope="result",
            left_context={"binding": ["A"]},
            right_context={"binding_other": ["A"]},
        ).outcome
        == "equivalent"
    )
    assert (
        compare_analysis(
            left,
            right,
            scope="result",
            left_context={"binding": ["A"]},
            right_context={"binding_other": ["B"]},
        ).outcome
        == "different"
    )


def test_empty_private_context_cannot_attest_nonempty_dependencies() -> None:
    analysis = _artifact(filtered=True)
    assert (
        compare_analysis(
            analysis, analysis, scope="result", left_context={}, right_context={}
        ).outcome
        == "not_established"
    )
    assert (
        compare_analysis(analysis, analysis, scope="result", bound_context_equal=True).outcome
        == "equivalent"
    )


def test_output_expression_and_result_selection_are_distinct_scopes() -> None:
    left = _artifact()
    assert left.result is not None
    right = left.model_copy(update={"result": left.result.model_copy(update={"limit": 3})})
    assert compare_analysis(left, right).outcome == "equivalent"
    assert compare_analysis(left, right, scope="result").outcome == "different"


def test_grouping_is_unordered_but_selection_order_is_not() -> None:
    left = _artifact()
    assert left.result is not None
    result = left.result.model_copy(
        update={
            "grain_node_ids": ("dimension", "amount"),
            "order": (("dimension", "asc"), ("amount", "desc")),
        }
    )
    left = left.model_copy(update={"result": result})
    reordered = left.model_copy(
        update={
            "result": result.model_copy(
                update={
                    "grain_node_ids": ("amount", "dimension"),
                }
            )
        }
    )
    assert compare_analysis(left, reordered, scope="result").outcome == "equivalent"
    reordered = left.model_copy(
        update={
            "result": result.model_copy(
                update={
                    "order": (("amount", "desc"), ("dimension", "asc")),
                }
            )
        }
    )
    assert compare_analysis(left, reordered, scope="result").outcome == "different"


def test_metric_stage_unit_and_period_are_semantic_not_presentation() -> None:
    left = _artifact()
    for change in (
        {"catalog_ref": "events.bookings"},
        {"logical_stage": "entity_total_to_group"},
        {"unit": "seconds"},
        {"period_role": "prior"},
        {"null_policy": "zero_fill"},
    ):
        right = left.model_copy(
            update={
                "nodes": (
                    left.nodes[0],
                    left.nodes[1].model_copy(update=change),
                )
            }
        )
        assert compare_analysis(left, right).outcome == "different"


def test_optional_public_id_annotation_does_not_change_meaning() -> None:
    left = _artifact()
    payload = copy.deepcopy(left.model_dump())
    payload["semantic_id"] = "future-result-id"
    payload["nodes"][1]["semantic_id"] = "future-metric-id"
    right = SemanticAnalysis.model_validate(payload)
    assert compare_analysis(left, right, scope="result").outcome == "equivalent"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 999},
        {"coverage": Coverage(status="not_established")},
        {"catalog_context": CatalogContext(namespace="reporting", semantic_revision="r18")},
        {"catalog_context": CatalogContext(namespace="reporting")},
        {"required_semantics": ("future_operator",)},
    ],
)
def test_unknown_coverage_or_revision_never_proves_equivalence(change: dict[str, object]) -> None:
    left = _artifact()
    right = left.model_copy(update=change)
    assert compare_analysis(left, right).outcome == "not_established"


def test_missing_revision_requires_explicit_shared_snapshot_attestation() -> None:
    left = _artifact().model_copy(update={"catalog_context": CatalogContext(namespace="reporting")})
    right = _artifact(suffix="_new").model_copy(update={"catalog_context": left.catalog_context})
    assert compare_analysis(left, right).outcome == "not_established"
    assert compare_analysis(left, right, shared_catalog_snapshot=True).outcome == "equivalent"


@pytest.mark.parametrize(
    "defect", ["dangling_operand", "dangling_population", "cycle", "duplicate_id"]
)
def test_invalid_graphs_raise_typed_failures_including_hidden_nodes(defect: str) -> None:
    left = _artifact()
    hidden = SemanticNode(
        node_id="hidden", kind="arithmetic", aggregate="ratio", operand_ids=("amount",)
    )
    if defect == "dangling_operand":
        hidden = hidden.model_copy(update={"operand_ids": ("missing",)})
    elif defect == "dangling_population":
        hidden = hidden.model_copy(update={"population_ref": "missing"})
    elif defect == "cycle":
        hidden = hidden.model_copy(update={"operand_ids": ("hidden",)})
    else:
        hidden = hidden.model_copy(update={"node_id": "amount"})
    invalid = left.model_copy(update={"nodes": (*left.nodes, hidden)})
    with pytest.raises(ContractError) as exc:
        compare_analysis(left, invalid)
    assert exc.value.stage == "analysis"


def test_unknown_node_kind_is_not_silently_compared() -> None:
    left = _artifact()
    right = left.model_copy(
        update={
            "nodes": (
                left.nodes[0],
                left.nodes[1].model_copy(update={"kind": "future_metric"}),
            )
        }
    )
    assert compare_analysis(left, right).outcome == "not_established"


def test_private_values_keep_boolean_branch_correlations() -> None:
    analysis = _artifact()
    leaves = tuple(
        SemanticNode(
            node_id=f"leaf{index}",
            kind="predicate",
            aggregate="eq",
            catalog_ref=reference,
            binding_refs=(f"b{index}",),
        )
        for index, reference in enumerate(
            ("events.region", "events.id", "events.region", "events.id")
        )
    )
    bindings = tuple(
        SemanticBinding(
            slot=f"b{index}",
            kind="filter",
            reference=node.catalog_ref,
            operator="eq",
            arity=1,
        )
        for index, node in enumerate(leaves)
    )
    branches = (
        SemanticNode(
            node_id="and1", kind="predicate", aggregate="and", operand_ids=("leaf0", "leaf1")
        ),
        SemanticNode(
            node_id="and2", kind="predicate", aggregate="and", operand_ids=("leaf2", "leaf3")
        ),
        SemanticNode(node_id="or", kind="predicate", aggregate="or", operand_ids=("and1", "and2")),
    )
    analysis = analysis.model_copy(
        update={
            "nodes": (*analysis.nodes, *leaves, *branches),
            "binding_dependencies": bindings,
            "populations": (analysis.populations[0].model_copy(update={"predicate_ids": ("or",)}),),
        }
    )
    comparison = compare_analysis(
        analysis,
        analysis,
        scope="result",
        left_context={"b0": "A", "b1": 1, "b2": "B", "b3": 2},
        right_context={"b0": "A", "b1": 2, "b2": "B", "b3": 1},
    )
    assert comparison.outcome == "different"


def test_hidden_operand_graph_renaming_preserves_ordered_ratio_meaning() -> None:
    def ratio(suffix: str, *, reverse: bool = False) -> SemanticAnalysis:
        analysis = _artifact(suffix=suffix)
        denominator = SemanticNode(
            node_id="views" + suffix,
            kind="aggregate",
            catalog_ref="events.views",
            aggregate="sum",
            logical_stage="event_to_group",
            population_ref="population" + suffix,
        )
        operands: tuple[str, ...] = ("amount" + suffix, denominator.node_id)
        if reverse:
            operands = tuple(reversed(operands))
        expression = SemanticNode(
            node_id="ratio" + suffix,
            kind="arithmetic",
            aggregate="ratio",
            operand_ids=operands,
            zero_denominator="null",
        )
        return analysis.model_copy(
            update={
                "nodes": (*analysis.nodes, denominator, expression),
                "outputs": (
                    analysis.outputs[0],
                    SemanticOutput(
                        output_id="rate" + suffix, node_id=expression.node_id, sql_alias="rate"
                    ),
                ),
            }
        )

    assert compare_analysis(ratio(""), ratio("_renamed")).outcome == "equivalent"
    assert compare_analysis(ratio(""), ratio("_renamed", reverse=True)).outcome == "different"


def test_compiled_join_kind_and_having_change_result_semantics() -> None:
    from semql import Cube, Dialect, Dimension, Filter, Join, Measure, SemanticQuery, compile_query

    events = Cube(
        name="events",
        table="events",
        alias="e",
        dialect=Dialect.DUCKDB,
        measures=[Measure(name="amount", sql="{e}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
        joins=[Join(to="regions", on="{e}.region = {r}.id", relationship="many_to_one")],
    )
    regions = Cube(
        name="regions",
        table="regions",
        alias="r",
        dialect=Dialect.DUCKDB,
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{r}.id", type="string"),
            Dimension(name="label", sql="{r}.label", type="string"),
        ],
    )
    catalog = {cube.name: cube for cube in (events, regions)}
    context = CatalogContext(namespace="reporting", semantic_revision="r17")
    query = SemanticQuery(
        dimensions=["events.region"],
        measures=["events.amount"],
        filters=[Filter(dimension="regions.id", op="is_null")],
    )
    inner = compile_query(query, catalog, catalog_context=context).analysis
    outer = compile_query(
        query.model_copy(update={"left_joins": ["regions"]}), catalog, catalog_context=context
    ).analysis
    assert compare_analysis(inner, outer, scope="result").outcome == "different"
    filtered = compile_query(
        query.model_copy(
            update={"having": [Filter(dimension="events.amount", op="gt", values=[10])]}
        ),
        catalog,
        catalog_context=context,
    ).analysis
    assert compare_analysis(inner, filtered, scope="result").outcome == "different"


def test_row_listing_is_not_equivalent_to_grouped_distinct_dimension() -> None:
    from semql import Cube, Dialect, Dimension, SemanticQuery, compile_query

    cube = Cube(
        name="events",
        table="events",
        alias="e",
        dialect=Dialect.DUCKDB,
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )
    context = CatalogContext(namespace="reporting", semantic_revision="r17")
    query = SemanticQuery(dimensions=["events.region"], limit=10)
    grouped = compile_query(query, {"events": cube}, catalog_context=context).analysis
    rows = compile_query(
        query.model_copy(update={"ungrouped": True}), {"events": cube}, catalog_context=context
    ).analysis
    assert compare_analysis(grouped, rows, scope="result").outcome == "different"


def test_security_context_requires_private_evidence_for_result_equivalence() -> None:
    from semql import AuthContext, Cube, Dialect, Measure, SemanticQuery, compile_query

    cube = Cube(
        name="events",
        table="events",
        alias="e",
        dialect=Dialect.DUCKDB,
        security_sql="{e}.owner = {ctx.viewer_id}",
        measures=[Measure(name="amount", sql="{e}.amount", agg="sum")],
    )
    context = CatalogContext(namespace="reporting", semantic_revision="r17")
    query = SemanticQuery(measures=["events.amount"])
    left = compile_query(
        query,
        {"events": cube},
        viewer=AuthContext(viewer_id="private-A"),
        catalog_context=context,
    ).analysis
    right = compile_query(
        query,
        {"events": cube},
        viewer=AuthContext(viewer_id="private-B"),
        catalog_context=context,
    ).analysis
    assert compare_analysis(left, right, scope="result").outcome == "not_established"
    different = compare_analysis(
        left,
        right,
        scope="result",
        left_context={item.slot: "private-A" for item in left.binding_dependencies},
        right_context={item.slot: "private-B" for item in right.binding_dependencies},
    )
    assert different.outcome == "different"
    assert "private-A" not in different.model_dump_json()
    assert "private-B" not in different.model_dump_json()
    assert (
        compare_analysis(left, left, scope="result", bound_context_equal=True).outcome
        == "equivalent"
    )


@pytest.mark.parametrize("op", ["is_null", "not_null", "eq", "in"])
def test_filter_dependencies_describe_only_executed_values(op: str) -> None:
    from semql import Cube, Dialect, Dimension, Filter, SemanticQuery, compile_query

    cube = Cube(
        name="events",
        table="events",
        alias="e",
        dialect=Dialect.DUCKDB,
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )
    context = CatalogContext(namespace="reporting", semantic_revision="r17")
    left = compile_query(
        SemanticQuery(
            dimensions=["events.region"],
            filters=[
                Filter.model_validate(
                    {"dimension": "events.region", "op": op, "values": ["A", "B"]}
                )
            ],
        ),
        {"events": cube},
        catalog_context=context,
    )
    right = compile_query(
        SemanticQuery(
            dimensions=["events.region"],
            filters=[
                Filter.model_validate(
                    {"dimension": "events.region", "op": op, "values": ["A", "C"]}
                )
            ],
        ),
        {"events": cube},
        catalog_context=context,
    )
    if op in {"is_null", "not_null"}:
        assert left.analysis.binding_dependencies == ()
        assert (
            compare_analysis(left.analysis, right.analysis, scope="result").outcome == "equivalent"
        )
    else:
        assert left.analysis.binding_dependencies[0].arity == (2 if op == "in" else 1)
        left_values = tuple(left.params.values())
        right_values = tuple(right.params.values())
        comparison = compare_analysis(
            left.analysis,
            right.analysis,
            scope="result",
            left_context={left.analysis.binding_dependencies[0].slot: left_values},
            right_context={right.analysis.binding_dependencies[0].slot: right_values},
        )
        assert comparison.outcome == ("different" if op == "in" else "equivalent")


def test_logical_assumptions_constrain_result_but_not_physical_recipe() -> None:
    from semql.analysis import SemanticAssumption

    left = _artifact().model_copy(
        update={
            "assumptions": (
                SemanticAssumption(
                    reference="join:events->regions",
                    kind="relationship",
                    declared_value="one_to_one",
                ),
            )
        }
    )
    changed = left.model_copy(
        update={
            "assumptions": (
                SemanticAssumption(
                    reference="join:events->regions",
                    kind="relationship",
                    declared_value="many_to_many",
                ),
            )
        }
    )
    assert compare_analysis(left, changed, scope="result").outcome == "different"
    physical = left.model_copy(
        update={
            "assumptions": (
                *left.assumptions,
                SemanticAssumption(
                    reference="fragment:0.id", kind="unique_merge_key", declared_value="unique"
                ),
            )
        }
    )
    assert compare_analysis(left, physical, scope="result").outcome == "equivalent"
