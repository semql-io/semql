"""Shared local-reference validation across semantic checking and comparison."""

from typing import Literal

import pytest
from semql import (
    Cube,
    Dialect,
    Measure,
    MetricContract,
    ProofStatus,
    SemanticAnalysis,
    SemanticNode,
    SemanticQuery,
    SemanticRanking,
    check_semantics,
    compare_analysis,
    compile_query,
)
from semql.errors import ContractError


def _analysis() -> SemanticAnalysis:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="amount", sql="{o}.amount", agg="sum")],
    )
    return compile_query(SemanticQuery(measures=["orders.amount"]), {"orders": cube}).analysis


@pytest.mark.parametrize("edge", ["order", "ranking", "population_cycle"])
def test_malformed_local_edges_cannot_establish_semantic_validity(
    edge: Literal["order", "ranking", "population_cycle"],
) -> None:
    original = _analysis()
    assert original.result is not None
    if edge == "order":
        invalid = original.model_copy(
            update={"result": original.result.model_copy(update={"order": (("absent", "asc"),)})}
        )
    elif edge == "ranking":
        invalid = original.model_copy(
            update={
                "result": original.result.model_copy(
                    update={
                        "ranking": SemanticRanking(
                            count=1, ties="exact", order=(("absent", "desc", "last"),)
                        )
                    }
                )
            }
        )
    else:
        population = original.populations[0]
        cyclic = SemanticNode(
            node_id="cycle",
            kind="predicate",
            aggregate="and",
            population_ref=population.population_id,
        )
        invalid = original.model_copy(
            update={
                "nodes": (*original.nodes, cyclic),
                "populations": (
                    population.model_copy(
                        update={
                            "predicate_ids": (*population.predicate_ids, cyclic.node_id),
                        }
                    ),
                ),
            }
        )
    report = check_semantics(
        invalid, {"orders.amount": MetricContract(reference="orders.amount", revision="r1")}
    )
    assert report.structural_validity == ProofStatus.DISPROVED
    assert not report.accepted
    with pytest.raises(ContractError):
        compare_analysis(invalid, invalid)


def test_reference_validation_handles_deep_artifacts_without_recursive_expansion() -> None:
    original = _analysis()
    chain: list[SemanticNode] = []
    previous = original.outputs[0].node_id
    for index in range(2000):
        node = SemanticNode(
            node_id=f"predicate_{index}", kind="predicate", aggregate="and", operand_ids=(previous,)
        )
        chain.append(node)
        previous = node.node_id
    deep = original.model_copy(update={"nodes": (*original.nodes, *chain)})
    report = check_semantics(
        deep, {"orders.amount": MetricContract(reference="orders.amount", revision="r1")}
    )
    assert report.structural_validity == ProofStatus.PROVED
    assert report.accepted
