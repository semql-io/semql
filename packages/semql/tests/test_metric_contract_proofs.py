from semql.analysis import SemanticAnalysis, SemanticPopulation
from semql.compile import compile_query
from semql.contracts import MetricContract, ProofStatus, SemanticCheck, SemanticExpectations
from semql.model import AuthContext, Cube, Dialect, Measure, Segment, TimeDimension
from semql.spec import SemanticQuery, TimeWindow
from semql.validate import check_semantics


def _catalog(*, masked: bool = False) -> Cube:
    return Cube(
        name="orders",
        table="orders",
        alias="o",
        dialect=Dialect.DUCKDB,
        measures=(
            Measure(
                name="amount",
                sql="{o}.amount",
                agg="sum",
                required_roles=("finance",) if masked else (),
                mask_roles=("finance",) if masked else (),
            ),
            Measure(name="items", sql="{o}.items", agg="sum"),
            Measure(name="rate", sql="", agg="ratio", numerator="amount", denominator="items"),
        ),
        dimensions=(),
        time_dimensions=(TimeDimension(name="created", sql="{o}.created", type="date"),),
        segments=(Segment(name="paid", sql="{o}.paid"),),
    )


def _compiled(
    *,
    metric: str = "rate",
    masked: bool = False,
    segments: tuple[str, ...] = (),
    time: bool = False,
) -> SemanticAnalysis:
    return compile_query(
        SemanticQuery(
            measures=(f"orders.{metric}",),
            segments=segments,
            time_dimension=(
                TimeWindow(
                    dimension="orders.created",
                    range=("2026-01-01", "2026-02-01"),
                    granularity="month",
                )
                if time
                else None
            ),
        ),
        {"orders": _catalog(masked=masked)},
        viewer=AuthContext(viewer_id="viewer", roles=("finance",) if masked else ()),
    ).analysis


def _contract(reference: str, **changes: object) -> MetricContract:
    return MetricContract.model_validate({"reference": reference, "revision": "r1", **changes})


def _status(report: SemanticCheck, code: str) -> ProofStatus:
    return next(
        item.status for item in (*report.obligations, *report.alignment) if item.code == code
    )


def test_ratio_contract_observes_masked_uncontracted_operand_from_compiler() -> None:
    contract = _contract("orders.rate")
    masked = _compiled(masked=True)
    assert any(node.catalog_ref == "orders.amount" and node.masked for node in masked.nodes)
    assert not check_semantics(masked, {contract.reference: contract}).accepted
    assert (
        _status(check_semantics(masked, {contract.reference: contract}), "masked_value")
        == ProofStatus.NOT_ESTABLISHED
    )

    unmasked = _compiled()
    assert check_semantics(unmasked, {contract.reference: contract}).accepted


def test_each_metric_occurrence_needs_its_own_required_population() -> None:
    base = _compiled(metric="amount", segments=("orders.paid",))
    assert base.result is not None
    amount = next(node for node in base.nodes if node.catalog_ref == "orders.amount")
    prior = amount.model_copy(
        update={"node_id": "prior-amount", "population_ref": "prior-population"}
    )
    result = base.result.model_copy(update={"population_ref": "p1"})
    changed = base.model_copy(
        update={
            "nodes": (*base.nodes, prior),
            "populations": (
                *base.populations,
                SemanticPopulation(population_id="prior-population", inclusion="query_rows"),
            ),
            "result": result,
        }
    )
    contract = _contract("orders.amount", required_segments=("orders.paid",))
    assert not check_semantics(changed, {contract.reference: contract}).accepted
    assert (
        _status(check_semantics(changed, {contract.reference: contract}), "required_population")
        == ProofStatus.DISPROVED
    )
    assert check_semantics(base, {contract.reference: contract}).accepted
    complete = changed.model_copy(
        update={
            "populations": (
                *base.populations,
                SemanticPopulation(
                    population_id="prior-population",
                    inclusion="query_rows",
                    predicate_ids=base.populations[0].predicate_ids,
                ),
            ),
        }
    )
    assert check_semantics(complete, {contract.reference: contract}).accepted


def test_missing_population_does_not_prove_forbidden_contract_or_confirmed_intent() -> None:
    base = _compiled(metric="amount")
    contract = _contract("orders.amount", forbidden_segments=("orders.paid",))
    expectations = SemanticExpectations(
        forbidden_segments=("orders.paid",),
        origin="independently_confirmed",
        evidence_ref="request:1",
    )
    assert base.result is not None
    changed = base.model_copy(
        update={
            "populations": (),
            "nodes": tuple(node.model_copy(update={"population_ref": None}) for node in base.nodes),
            "result": base.result.model_copy(update={"population_ref": None}),
        }
    )
    report = check_semantics(
        changed, {contract.reference: contract}, expectations, require_confirmed_intent=True
    )
    assert not report.accepted
    assert _status(report, "forbidden_population") == ProofStatus.NOT_ESTABLISHED
    assert report.confirmed_intent_alignment == ProofStatus.NOT_ESTABLISHED
    inferred = check_semantics(
        changed,
        {contract.reference: contract},
        expectations.model_copy(update={"origin": "model_inferred", "evidence_ref": None}),
        require_confirmed_intent=True,
    )
    assert not inferred.accepted
    assert inferred.confirmed_intent_alignment == ProofStatus.NOT_ESTABLISHED


def test_time_policy_requires_consistent_known_granularity_and_snapshot_control() -> None:
    monthly = _compiled(metric="amount", time=True)
    contract = _contract("orders.amount", permitted_granularities=(None,))
    snapshot = _contract(
        "orders.amount", temporal_behavior="snapshot", snapshot_axis="orders.created"
    )
    assert monthly.result is not None and monthly.result.time_policy is not None
    # An unbucketed semantic artifact need not claim compiler lowering support.
    actual_unbucketed = monthly.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"time_granularity": None}) if node.kind == "time" else node
                for node in monthly.nodes
            ),
            "result": monthly.result.model_copy(
                update={
                    "time_policy": monthly.result.time_policy.model_copy(
                        update={"granularity": None}
                    ),
                }
            ),
        }
    )
    assert (
        _status(
            check_semantics(actual_unbucketed, {snapshot.reference: snapshot}),
            "snapshot_aggregation",
        )
        == ProofStatus.PROVED
    )
    assert (
        _status(check_semantics(monthly, {snapshot.reference: snapshot}), "snapshot_aggregation")
        == ProofStatus.DISPROVED
    )
    no_time_policy_query = _compiled(metric="amount")
    assert check_semantics(no_time_policy_query, {contract.reference: contract}).accepted


def test_missing_or_contradictory_time_policy_cannot_prove_unbucketed_result() -> None:
    monthly = _compiled(metric="amount", time=True)
    assert monthly.result is not None and monthly.result.time_policy is not None
    unbucketed_contract = _contract("orders.amount", permitted_granularities=(None,))
    missing = monthly.model_copy(
        update={
            "result": monthly.result.model_copy(update={"time_policy": None}),
        }
    )
    contradictory = monthly.model_copy(
        update={
            "result": monthly.result.model_copy(
                update={
                    "time_policy": monthly.result.time_policy.model_copy(
                        update={"granularity": None}
                    ),
                }
            ),
        }
    )
    for incomplete in (missing, contradictory):
        report = check_semantics(incomplete, {unbucketed_contract.reference: unbucketed_contract})
        assert _status(report, "time_granularity") == ProofStatus.NOT_ESTABLISHED
        assert not report.accepted
    monthly_contract = _contract("orders.amount", permitted_granularities=("month",))
    assert check_semantics(monthly, {monthly_contract.reference: monthly_contract}).accepted


def test_confirmed_granularity_intent_requires_consistent_temporal_evidence() -> None:
    monthly = _compiled(metric="amount", time=True)
    assert monthly.result is not None and monthly.result.time_policy is not None
    contract = _contract("orders.amount")
    registry = {contract.reference: contract}
    expected = SemanticExpectations(
        granularity="day", origin="independently_confirmed", evidence_ref="request:daily-report"
    )
    wrong_policy = monthly.result.time_policy.model_copy(update={"granularity": "day"})
    for policy in (wrong_policy, None):
        incomplete = monthly.model_copy(
            update={
                "result": monthly.result.model_copy(update={"time_policy": policy}),
            }
        )
        report = check_semantics(
            incomplete,
            registry,
            expected,
            require_confirmed_intent=True,
        )
        assert report.semantic_validity == ProofStatus.PROVED
        assert _status(report, "intent_granularity") == ProofStatus.NOT_ESTABLISHED
        assert report.confirmed_intent_alignment == ProofStatus.NOT_ESTABLISHED
        assert not report.accepted
    assert check_semantics(
        monthly,
        registry,
        expected.model_copy(update={"granularity": "month"}),
        require_confirmed_intent=True,
    ).accepted
