import pytest
from pydantic import ValidationError
from semql.analysis import SemanticAnalysis, SemanticNode, attach_semantic_check
from semql.compile import compile_query
from semql.contracts import (
    FunctionalDependency,
    MetricContract,
    ProofStatus,
    SemanticExpectations,
    SourceGrain,
)
from semql.model import (
    Cube,
    Dialect,
    Dimension,
    GranularityLiteral,
    Measure,
    Segment,
    TimeDimension,
)
from semql.spec import SemanticQuery, TimeWindow
from semql.validate import check_semantics


def analysis(
    *,
    axis: str | None = None,
    granularity: GranularityLiteral | None = None,
    dimensions: tuple[str, ...] = (),
    metric: str = "amount",
    segments: tuple[str, ...] = (),
) -> SemanticAnalysis:
    cube = Cube(
        name="orders",
        table="orders",
        alias="o",
        dialect=Dialect.DUCKDB,
        measures=(
            Measure(name="amount", sql="{o}.amount", agg="sum"),
            Measure(name="items", sql="{o}.items", agg="sum"),
            Measure(name="rate", sql="", agg="ratio", numerator="amount", denominator="items"),
        ),
        dimensions=(
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
        ),
        time_dimensions=(
            TimeDimension(name="created", sql="{o}.created", type="date"),
            TimeDimension(name="shipped", sql="{o}.shipped", type="date"),
        ),
        segments=(Segment(name="paid", sql="{o}.paid"),),
    )
    time = (
        None
        if axis is None
        else TimeWindow(
            dimension=f"orders.{axis}", range=("2026-01-01", "2026-02-01"), granularity=granularity
        )
    )
    return compile_query(
        SemanticQuery(
            measures=(f"orders.{metric}",),
            dimensions=dimensions,
            segments=segments,
            time_dimension=time,
        ),
        {"orders": cube},
    ).analysis


def contract(**changes: object) -> MetricContract:
    return MetricContract.model_validate(
        {"reference": "orders.amount", "revision": "r1", **changes}
    )


def test_contract_values_are_frozen_and_references_are_validated() -> None:
    keys = ["orders.tenant", "orders.id"]
    grain = SourceGrain.model_validate({"keys": keys})
    keys.append("orders.region")
    assert grain.keys == ("orders.tenant", "orders.id")
    with pytest.raises(ValidationError):
        MetricContract(reference="amount", revision="r1")
    with pytest.raises(ValidationError):
        FunctionalDependency(determinant=("orders.region",), dependent=("unqualified",))
    with pytest.raises(ValidationError):
        contract(required_segments=("orders.paid",), forbidden_segments=("orders.paid",))
    with pytest.raises(ValidationError):
        MetricContract.model_validate(
            {"reference": "orders.amount", "revision": "r1", "schema_version": 2}
        )


def test_result_descriptor_and_unknown_grain_are_not_collapsed_to_empty() -> None:
    base = analysis()
    missing_result = base.model_copy(update={"result": None})
    report = check_semantics(
        missing_result,
        {"orders.amount": contract(source_grain=SourceGrain(keys=("orders.id",)))},
    )
    assert not report.accepted
    assert (
        next(o for o in report.obligations if o.code == "result_semantics").status
        == ProofStatus.NOT_ESTABLISHED
    )

    unknown_grain = SemanticNode(node_id="opaque-grain", kind="dimension")
    assert base.result is not None
    changed_result = base.result.model_copy(update={"grain_node_ids": ("opaque-grain",)})
    changed = base.model_copy(
        update={"nodes": (*base.nodes, unknown_grain), "result": changed_result}
    )
    requirement = {"orders.amount": contract(required_dimensions=("orders.region",))}
    report = check_semantics(changed, requirement)
    assert (
        next(o for o in report.obligations if o.code == "required_grain").status
        == ProofStatus.NOT_ESTABLISHED
    )


def test_empty_permitted_axis_set_is_distinct_from_unrestricted_none() -> None:
    report = check_semantics(analysis(), {"orders.amount": contract(permitted_time_axes=())})
    assert report.accepted
    assert next(o for o in report.obligations if o.code == "time_axis").status == ProofStatus.PROVED


def test_source_grain_fd_closure_is_conditional_and_unknown_stays_unknown() -> None:
    grain = SourceGrain(
        keys=("orders.tenant", "orders.id"),
        dependencies=(
            FunctionalDependency(determinant=("orders.id",), dependent=("orders.region",)),
            FunctionalDependency(determinant=("orders.region",), dependent=("orders.country",)),
        ),
    )
    assert grain.determines(("orders.country",)) == ProofStatus.PROVED
    assert grain.determines(("orders.unknown",)) == ProofStatus.NOT_ESTABLISHED


def test_overlapping_fd_fields_are_valid_and_empty_target_is_trivial() -> None:
    grain = SourceGrain(
        keys=("orders.tenant", "orders.id"),
        dependencies=(
            FunctionalDependency(
                determinant=("orders.tenant", "orders.id"),
                dependent=("orders.id", "orders.region"),
            ),
        ),
    )
    assert grain.determines(()) == ProofStatus.PROVED
    assert grain.determines(("orders.region",)) == ProofStatus.PROVED


def test_registry_is_canonical_and_frozen_inputs_cannot_mutate() -> None:
    item = contract()
    registry: dict[str, MetricContract] = {item.reference: item}
    report = check_semantics(analysis(), registry)
    registry.clear()
    assert report.applied_contracts == ("contract:orders.amount@r1",)


def test_source_grain_is_conditional_and_not_physical_verification() -> None:
    report = check_semantics(
        analysis(),
        {"orders.amount": contract(source_grain=SourceGrain(keys=("orders.tenant", "orders.id")))},
    )
    assert report.accepted
    assert (
        next(x for x in report.obligations if x.code == "source_grain").status == ProofStatus.PROVED
    )
    uniqueness = next(x for x in report.obligations if x.code == "physical_uniqueness")
    assert uniqueness.status == ProofStatus.NOT_ESTABLISHED and not uniqueness.mandatory


def test_required_dimension_and_named_segment_fail_closed() -> None:
    requirements = {
        "orders.amount": contract(
            required_dimensions=("orders.region",), required_segments=("orders.paid",)
        )
    }
    assert not check_semantics(analysis(), requirements).accepted
    assert check_semantics(
        analysis(dimensions=("orders.region",), segments=("orders.paid",)), requirements
    ).accepted


def test_hidden_ratio_operands_are_checked_without_becoming_projected_intent() -> None:
    report = check_semantics(
        analysis(metric="rate", axis="created"),
        {
            "orders.amount": contract(permitted_time_axes=("orders.shipped",)),
            "orders.rate": MetricContract(reference="orders.rate", revision="r1"),
        },
        SemanticExpectations(metrics=("orders.rate",)),
    )
    assert report.alignment[0].status == ProofStatus.PROVED
    assert report.semantic_validity == ProofStatus.DISPROVED


def test_repeated_time_axis_reference_does_not_create_ambiguous_false_proof() -> None:
    base = analysis(axis="created")
    # Two explicit range predicates with different refs are not reduced to an arbitrary axis.
    from semql.analysis import SemanticBinding, SemanticNode

    nodes = (
        *base.nodes,
        SemanticNode(node_id="extra-field", kind="time", catalog_ref="orders.shipped"),
        SemanticNode(
            node_id="extra-range",
            kind="predicate",
            aggregate="half_open_range",
            catalog_ref="orders.shipped",
            operand_ids=("extra-field",),
            binding_refs=("extra-binding",),
        ),
    )
    bindings = (
        *base.binding_dependencies,
        SemanticBinding(
            slot="extra-binding",
            kind="time_range",
            reference="orders.shipped",
            operator="half_open",
            stage="where",
            arity=2,
        ),
    )
    populations = tuple(
        p.model_copy(update={"predicate_ids": (*p.predicate_ids, "extra-range")})
        for p in base.populations
    )
    changed = base.model_copy(
        update={"nodes": nodes, "binding_dependencies": bindings, "populations": populations}
    )
    report = check_semantics(
        changed,
        {
            "orders.amount": contract(
                required_time_axis=True, permitted_time_axes=("orders.created",)
            )
        },
    )
    assert any(
        o.code == "time_axis_required" and o.status == ProofStatus.NOT_ESTABLISHED
        for o in report.obligations
    )
    assert any(
        o.code == "time_axis" and o.status == ProofStatus.NOT_ESTABLISHED
        for o in report.obligations
    )


def test_snapshot_bucket_is_not_an_as_of_selector_and_calendar_is_unknown() -> None:
    report = check_semantics(
        analysis(axis="created", granularity="month"),
        {
            "orders.amount": contract(
                temporal_behavior="snapshot",
                snapshot_axis="orders.created",
                calendar_ref="fiscal:v2",
            )
        },
    )
    assert not report.accepted
    assert (
        next(o for o in report.obligations if o.code == "snapshot_aggregation").status
        == ProofStatus.DISPROVED
    )
    assert (
        next(o for o in report.obligations if o.code == "calendar").status
        == ProofStatus.NOT_ESTABLISHED
    )


def test_fixed_cohort_comparison_is_not_inferred_from_local_population_id() -> None:
    report = check_semantics(
        analysis(),
        {"orders.amount": contract()},
        SemanticExpectations(
            comparison_basis="fixed_cohort",
            origin="independently_confirmed",
            evidence_ref="request:12",
        ),
        require_confirmed_intent=True,
    )
    assert report.confirmed_intent_alignment == ProofStatus.NOT_ESTABLISHED
    assert not report.accepted


def test_analysis_annotations_are_explicit_immutable_and_serializable() -> None:
    original = analysis()
    report = check_semantics(
        original,
        {
            "orders.amount": contract(
                source_grain=SourceGrain(keys=("orders.id",)), calendar_ref="fiscal:2026"
            )
        },
        SemanticExpectations(
            metrics=("orders.amount",), origin="independently_confirmed", evidence_ref="request:12"
        ),
        require_confirmed_intent=True,
    )
    attached = attach_semantic_check(original, report)
    assert original.check_annotations == ()
    restored = SemanticAnalysis.model_validate_json(attached.model_dump_json())
    evidence = restored.check_annotations[0]
    assert evidence.node_ids == (original.outputs[0].node_id,)
    assert evidence.report.intent_evidence_ref == "request:12"
    assert evidence.report.confirmed_intent_alignment == ProofStatus.PROVED
    assert not evidence.report.accepted
    assert (
        next(
            obligation
            for obligation in evidence.report.obligations
            if obligation.code == "calendar"
        ).status
        == ProofStatus.NOT_ESTABLISHED
    )
    assert not next(
        obligation
        for obligation in evidence.report.obligations
        if obligation.code == "physical_uniqueness"
    ).mandatory


def test_any_masked_occurrence_blocks_numeric_contract_claims() -> None:
    base = analysis()
    amount = next(node for node in base.nodes if node.catalog_ref == "orders.amount")
    masked = amount.model_copy(update={"node_id": "masked-occurrence", "masked": True})
    changed = base.model_copy(update={"nodes": (*base.nodes, masked)})
    report = check_semantics(changed, {"orders.amount": contract()})
    assert (
        next(o for o in report.obligations if o.code == "masked_value").status
        == ProofStatus.NOT_ESTABLISHED
    )


def test_unknown_analysis_operator_fails_closed() -> None:
    base = analysis()
    amount = next(node for node in base.nodes if node.catalog_ref == "orders.amount")
    unknown = amount.model_copy(update={"aggregate": "future_operator"})
    nodes = tuple(unknown if node.node_id == amount.node_id else node for node in base.nodes)
    report = check_semantics(
        base.model_copy(update={"nodes": nodes}), {"orders.amount": contract()}
    )
    assert report.semantic_validity == ProofStatus.NOT_ESTABLISHED


def test_unknown_analysis_schema_and_execution_support_fail_closed() -> None:
    unknown = analysis().model_copy(update={"schema_version": 99})
    report = check_semantics(unknown, {"orders.amount": contract()})
    assert report.semantic_validity == ProofStatus.NOT_ESTABLISHED
    assert report.execution_support == ProofStatus.NOT_ESTABLISHED
    assert report.execution_explanation
