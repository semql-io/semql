from __future__ import annotations

from typing import Any

import pytest
from semql.analysis import (
    CatalogContext,
    ComparisonResult,
    ComparisonScope,
    Coverage,
    SemanticAnalysis,
    SemanticBinding,
    SemanticNode,
    SemanticOutput,
    SemanticPopulation,
    SemanticResult,
    compare_analysis,
)
from semql.compile import CompiledQuery, CompileError, compile_query
from semql.model import Cube, Dialect, Dimension, Join, Measure, TimeDimension, View
from semql.spec import CompareWindow, Filter, InlineDerived, SemanticQuery, TimeWindow


def _analysis(
    *, node_id: str, output_id: str, alias: str, revision: str | None = "r1"
) -> SemanticAnalysis:
    return SemanticAnalysis(
        nodes=(
            SemanticNode(
                node_id=node_id,
                kind="aggregate",
                catalog_ref="orders.amount",
                aggregate="sum",
                logical_stage="aggregate",
                population_ref="p1",
                null_policy="ignore_null_inputs",
            ),
        ),
        outputs=(SemanticOutput(output_id=output_id, node_id=node_id, sql_alias=alias),),
        populations=(SemanticPopulation(population_id="p1", inclusion="query_rows"),),
        binding_dependencies=(
            SemanticBinding(
                slot="b1",
                kind="filter_value",
                reference="orders.region",
                operator="eq",
                stage="where",
                arity=1,
            ),
        ),
        result=SemanticResult(population_ref="p1", completeness="complete"),
        catalog_context=CatalogContext(namespace="tenant-a", semantic_revision=revision),
        coverage=Coverage(status="complete", established=("requested_outputs",)),
    )


def test_comparison_follows_renamed_local_references_and_ignores_aliases() -> None:
    left = _analysis(node_id="n1", output_id="o1", alias="amount")
    right = _analysis(node_id="new-node", output_id="new-output", alias="gross")
    result = compare_analysis(left, right, scope=ComparisonScope.EXPRESSION)
    assert result.outcome is ComparisonResult.EQUIVALENT


def test_comparison_requires_revision_and_result_context_attestation() -> None:
    left = _analysis(node_id="n1", output_id="o1", alias="amount", revision=None)
    right = _analysis(node_id="n2", output_id="o2", alias="amount", revision=None)
    assert compare_analysis(left, right).outcome is ComparisonResult.NOT_ESTABLISHED
    rev_left = _analysis(node_id="n1", output_id="o1", alias="amount")
    rev_right = _analysis(node_id="n2", output_id="o2", alias="amount")
    assert (
        compare_analysis(rev_left, rev_right, scope=ComparisonScope.RESULT).outcome
        is ComparisonResult.NOT_ESTABLISHED
    )
    assert (
        compare_analysis(
            rev_left, rev_right, scope=ComparisonScope.RESULT, bound_context_equal=False
        ).outcome
        is ComparisonResult.DIFFERENT
    )


def test_compiled_artifact_round_trip_preserves_semantic_references() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        measures=[Measure(name="amount", sql="{o}.amount", agg="sum")],
    )
    query = SemanticQuery(measures=["orders.amount"], dimensions=["orders.region"])
    compiled = compile_query(
        query,
        {"orders": cube},
        catalog_context=CatalogContext(namespace="tenant-a", semantic_revision="r1"),
    )
    restored = CompiledQuery.model_validate(compiled.model_dump())
    assert restored.analysis == compiled.analysis
    assert [out.output_id for out in restored.analysis.outputs] == [
        out.output_id for out in compiled.analysis.outputs
    ]
    assert restored.version == compiled.version
    assert restored.binding_requirements == compiled.binding_requirements


def test_legacy_artifact_analysis_is_unavailable_and_future_annotations_are_ignored() -> None:
    payload: dict[str, Any] = {
        "dialect": "duckdb",
        "sql": "SELECT 1",
        "params": {},
        "columns": ["x"],
    }
    legacy = CompiledQuery.model_validate(payload)
    assert legacy.analysis.coverage.status == "unavailable"
    annotated = _analysis(node_id="n1", output_id="o1", alias="amount").model_dump()
    annotated["future_optional_annotation"] = {"opaque": True}
    assert SemanticAnalysis.model_validate(annotated).nodes[0].node_id == "n1"


def test_unknown_required_semantic_kind_is_not_equivalent() -> None:
    known = _analysis(node_id="n1", output_id="o1", alias="amount")
    unknown = known.model_copy(
        update={
            "nodes": (
                SemanticNode(
                    node_id="renamed", kind="future_required_operator", catalog_ref="orders.amount"
                ),
            ),
            "outputs": (
                SemanticOutput(output_id="renamed-output", node_id="renamed", sql_alias="amount"),
            ),
        }
    )
    result = compare_analysis(known, unknown)
    assert result.outcome is ComparisonResult.NOT_ESTABLISHED


def test_inline_derived_output_links_hidden_measure_operands() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[
            Measure(name="amount", sql="{o}.amount", agg="sum"),
            Measure(name="views", sql="{o}.views", agg="sum"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    query = SemanticQuery(
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="rate",
                op="ratio",
                operands=["orders.amount", "orders.views"],
            )
        ],
    )
    analysis = compile_query(query, {"orders": cube}).analysis
    rate = next(node for node in analysis.nodes if node.kind == "arithmetic")
    assert len(rate.operand_ids) == 2
    assert {node.logical_stage for node in analysis.nodes if node.node_id in rate.operand_ids} == {
        "event_to_group"
    }
    assert [output.sql_alias for output in analysis.outputs] == ["region", "rate"]


def test_result_records_grouping_and_declared_join_semantics() -> None:
    customers = Cube(
        name="customers",
        dialect=Dialect.DUCKDB,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
        joins=[Join(to="orders", relationship="one_to_many", on="{c}.id = {o}.customer_id")],
    )
    orders = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        primary_key="id",
        dimensions=[Dimension(name="customer_id", sql="{o}.customer_id", type="number")],
        measures=[Measure(name="amount", sql="{o}.amount", agg="sum")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )

    analysis = compile_query(
        SemanticQuery(measures=["orders.amount"], dimensions=["customers.region"]),
        {"customers": customers, "orders": orders},
    ).analysis

    assert analysis.result is not None
    assert analysis.result.grain_kind == "grouped"
    assert analysis.result.completeness == "not_established"
    assert analysis.result.predicate_ids == ()
    assert len(analysis.joins) == 1
    assert analysis.joins[0].relationship in {"one_to_many", "many_to_one"}
    assert len(analysis.assumptions) == 1


def test_duplicate_semantic_projection_is_rejected() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="amount", sql="{o}.amount", agg="sum")],
    )
    query = SemanticQuery(measures=["orders.amount", "orders.amount"])
    with pytest.raises(CompileError, match="Duplicate semantic projection"):
        compile_query(query, {"orders": cube})


def test_duplicate_inline_semantic_expression_is_rejected() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[
            Measure(name="amount", sql="{o}.amount", agg="sum"),
            Measure(name="views", sql="{o}.views", agg="sum"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    query = SemanticQuery(
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="rate_a", op="ratio", operands=["orders.amount", "orders.views"]),
            InlineDerived(name="rate_b", op="ratio", operands=["orders.amount", "orders.views"]),
        ],
    )
    with pytest.raises(CompileError, match="Duplicate semantic projection"):
        compile_query(query, {"orders": cube})


def _graph_references_are_local(analysis: SemanticAnalysis) -> None:
    node_ids = {node.node_id for node in analysis.nodes}
    output_ids = {output.output_id for output in analysis.outputs}
    population_ids = {population.population_id for population in analysis.populations}
    binding_ids = {binding.slot for binding in analysis.binding_dependencies}
    assert len(node_ids) == len(analysis.nodes)
    assert len(output_ids) == len(analysis.outputs)
    assert all(output.node_id in node_ids for output in analysis.outputs)
    for node in analysis.nodes:
        assert set(node.operand_ids) <= node_ids
        assert set(node.binding_refs) <= binding_ids
        assert node.population_ref is None or node.population_ref in population_ids
    for population in analysis.populations:
        assert set(population.predicate_ids) <= node_ids
    assert analysis.result is not None
    assert set(analysis.result.grain_node_ids) <= node_ids
    assert set(analysis.result.predicate_ids) <= node_ids
    assert all(node_id in node_ids for node_id, _ in analysis.result.order)


def test_resolved_input_alias_and_view_fields_form_result_grain() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        dimensions=[
            Dimension(name="region", aliases=["territory"], sql="{o}.region", type="string")
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
    )
    for query, views, grain_ref in (
        (
            SemanticQuery(measures=["orders.revenue"], dimensions=["orders.territory"]),
            None,
            "orders.region",
        ),
        (
            SemanticQuery(measures=["checkout.revenue"], dimensions=["checkout.region"]),
            {
                "checkout": View(
                    name="checkout", fields={"revenue": "orders.revenue", "region": "orders.region"}
                )
            },
            "orders.region",
        ),
    ):
        analysis = compile_query(query, {"orders": cube}, views=views).analysis
        assert analysis.coverage.status == "complete"
        assert analysis.result is not None
        assert analysis.result.grain_kind == "grouped"
        grain_node = next(
            node for node in analysis.nodes if node.node_id == analysis.result.grain_node_ids[0]
        )
        assert grain_node.catalog_ref == grain_ref
        _graph_references_are_local(analysis)


@pytest.mark.parametrize(
    ("having_ref", "derived"),
    [
        ("revenue", False),
        ("net", False),
        ("aov", True),
    ],
)
def test_having_references_target_resolved_output_nodes(having_ref: str, derived: bool) -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
        ],
    )
    query = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        aliases={"net": "orders.revenue"} if having_ref == "net" else {},
        derived_measures=(
            [InlineDerived(name="aov", op="ratio", operands=["orders.revenue", "orders.count"])]
            if derived
            else []
        ),
        having=[Filter(dimension=having_ref, op="gt", values=[1])],
    )
    analysis = compile_query(query, {"orders": cube}).analysis
    assert analysis.coverage.status == "complete"
    assert analysis.result is not None
    predicate = next(
        node for node in analysis.nodes if node.node_id in analysis.result.predicate_ids
    )
    target = next(node for node in analysis.nodes if node.node_id == predicate.operand_ids[0])
    assert predicate.logical_stage == "having"
    assert target.kind == ("arithmetic" if derived else "aggregate")
    _graph_references_are_local(analysis)


def test_compare_synthetic_having_and_public_string_scope() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at", granularities=("day", "week"))
        ],
    )
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-01-03"),
        ),
        compare=CompareWindow(mode="previous_period"),
        having=[Filter(dimension="compare.revenue.delta", op="gt", values=[1])],
    )
    analysis = compile_query(query, {"orders": cube}).analysis
    assert analysis.result is not None
    predicate = next(
        node for node in analysis.nodes if node.node_id in analysis.result.predicate_ids
    )
    target = next(node for node in analysis.nodes if node.node_id == predicate.operand_ids[0])
    assert target.kind == "comparison"
    assert target.aggregate == "delta"
    assert (
        compare_analysis(
            _analysis(node_id="n1", output_id="o1", alias="amount"),
            _analysis(node_id="n2", output_id="o2", alias="amount"),
            scope="expression",
        ).outcome
        is ComparisonResult.EQUIVALENT
    )
    _graph_references_are_local(analysis)


def test_time_policy_omits_inapplicable_timezone_and_week_start() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        timezone="America/Los_Angeles",
        week_start="monday",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[
            TimeDimension(name="business_day", sql="{o}.business_day", type="date"),
            TimeDimension(name="created_at", sql="{o}.created_at", granularities=("day", "week")),
        ],
    )
    date_analysis = compile_query(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.business_day",
                granularity="day",
                range=("2026-01-01", "2026-01-03"),
            ),
        ),
        {"orders": cube},
    ).analysis
    assert date_analysis.result is not None
    day_policy = date_analysis.result.time_policy
    assert day_policy is not None
    assert day_policy.timezone is None
    assert day_policy.week_start is None
    week_analysis = compile_query(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="week",
                range=("2026-01-01", "2026-01-15"),
            ),
        ),
        {"orders": cube},
    ).analysis
    assert week_analysis.result is not None
    week_policy = week_analysis.result.time_policy
    assert week_policy is not None
    assert week_policy.week_start == "monday"
