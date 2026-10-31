"""Fail-closed semantic contract checks over compiler analysis evidence."""

from __future__ import annotations

from collections.abc import Mapping

from semql.analysis import SemanticAnalysis, SemanticNode
from semql.contracts import (
    Advice,
    MetricContract,
    Obligation,
    ProofStatus,
    SemanticCheck,
    SemanticExpectations,
    conjunction,
    validate_registry,
)

_AGGREGATES = {
    "sum",
    "count",
    "count_distinct",
    "avg",
    "min",
    "max",
    "ratio",
    "median",
    "p75",
    "p90",
    "p95",
}
_PREDICATES = {
    "eq",
    "neq",
    "in",
    "not_in",
    "gt",
    "lt",
    "gte",
    "lte",
    "contains",
    "is_null",
    "not_null",
    "and",
    "or",
    "not",
    "segment",
    "scope",
    "half_open_range",
}


def _known_node(node: SemanticNode) -> bool:
    if node.kind == "aggregate":
        return node.aggregate in _AGGREGATES
    if node.kind == "comparison":
        return node.aggregate in {"current", "prior", "delta", "pct_change"}
    if node.kind == "arithmetic":
        return node.aggregate in {"ratio", "sum", "diff"}
    if node.kind == "predicate":
        return node.aggregate in _PREDICATES
    return node.kind in {"dimension", "time"} and node.aggregate is None


_KNOWN_KINDS = {"aggregate", "comparison", "dimension", "time", "arithmetic", "predicate"}


def _structure(analysis: SemanticAnalysis) -> ProofStatus:
    from semql._compare import validate_analysis_references
    from semql.errors import ContractError

    try:
        validate_analysis_references(analysis)
    except ContractError:
        return ProofStatus.DISPROVED
    return ProofStatus.PROVED


def _projected_metrics(analysis: SemanticAnalysis) -> tuple[str, ...] | None:
    nodes = {node.node_id: node for node in analysis.nodes}
    pending = [output.node_id for output in analysis.outputs]
    visited: set[str] = set()
    refs: set[str] = set()
    while pending:
        ref = pending.pop()
        if ref in visited:
            continue
        visited.add(ref)
        node = nodes[ref]
        if node.kind == "comparison":
            pending.extend(node.operand_ids)
        elif node.kind == "aggregate" and node.catalog_ref is not None:
            refs.add(node.catalog_ref)
        elif node.kind not in {"dimension", "time"}:
            return None
    return tuple(sorted(refs))


def check_semantics(
    analysis: SemanticAnalysis,
    contracts: Mapping[str, MetricContract],
    expectations: SemanticExpectations | None = None,
    *,
    require_confirmed_intent: bool = False,
) -> SemanticCheck:
    """Check contract claims without SQL evaluation or lowering assumptions."""
    registry = dict(validate_registry(contracts))
    structural = _structure(analysis)
    obligations: list[Obligation] = []
    alignment: list[Obligation] = []
    advice: list[Advice] = []
    applied: list[str] = []

    def add(
        code: str,
        status: ProofStatus,
        reference: str,
        explanation: str,
        assumptions: tuple[str, ...] = (),
        *,
        mandatory: bool = True,
        intent: bool = False,
    ) -> None:
        (alignment if intent else obligations).append(
            Obligation(
                code=code,
                status=status,
                reference=reference,
                explanation=explanation,
                assumptions=assumptions,
                mandatory=mandatory,
            )
        )

    if structural != ProofStatus.PROVED:
        add(
            "reference_integrity",
            ProofStatus.DISPROVED,
            "analysis",
            "Broken or cyclic artifact-local references.",
        )
        return SemanticCheck(structural_validity=structural, obligations=tuple(obligations))

    complete = (
        analysis.schema_version == 1
        and analysis.coverage.status == "complete"
        and set(analysis.required_semantics) <= {"partition_ranking"}
        and all(node.kind in _KNOWN_KINDS and _known_node(node) for node in analysis.nodes)
    )
    add(
        "analysis_coverage",
        ProofStatus.PROVED if complete else ProofStatus.NOT_ESTABLISHED,
        "analysis",
        "Rules require schema v1, known operators, and complete analysis coverage.",
    )
    if analysis.result is None:
        add(
            "result_semantics",
            ProofStatus.NOT_ESTABLISHED,
            "analysis",
            "No result descriptor is available.",
        )
    result = analysis.result
    available = complete and result is not None
    nodes = {node.node_id: node for node in analysis.nodes}
    grain_refs: list[str] = []
    grain_known = result is not None
    if result is not None:
        for ref in result.grain_node_ids:
            catalog_ref = nodes[ref].catalog_ref
            if catalog_ref is None:
                grain_known = False
            else:
                grain_refs.append(catalog_ref)
    grain = tuple(sorted(grain_refs)) if grain_known else None
    result_population = next(
        (p for p in analysis.populations if result and p.population_id == result.population_ref),
        None,
    )
    predicate_ids = result_population.predicate_ids if result_population is not None else ()
    range_refs = {
        nodes[ref].catalog_ref
        for ref in predicate_ids
        if nodes[ref].kind == "predicate"
        and nodes[ref].aggregate == "half_open_range"
        and nodes[ref].catalog_ref is not None
    }
    selected_time_refs = {
        nodes[ref].catalog_ref
        for ref in (result.grain_node_ids if result else ())
        if nodes[ref].kind == "time" and nodes[ref].catalog_ref is not None
    }
    selected_time_refs.update(
        nodes[output.node_id].catalog_ref
        for output in analysis.outputs
        if nodes[output.node_id].kind == "time" and nodes[output.node_id].catalog_ref is not None
    )
    result_time_refs = range_refs | selected_time_refs
    result_axis = next(iter(result_time_refs)) if len(result_time_refs) == 1 else None
    result_axis_known = (
        available
        and len(result_time_refs) <= 1
        and (bool(result_time_refs) or result_population is not None)
    )
    result_granularity = result.time_policy.granularity if result and result.time_policy else None
    observed_time_nodes = [
        nodes[node_id]
        for node_id in dict.fromkeys(
            (
                *(result.grain_node_ids if result else ()),
                *(output.node_id for output in analysis.outputs),
            )
        )
        if nodes[node_id].kind == "time"
    ]
    result_granularity_known = (
        available
        and result is not None
        and (
            (
                result.time_policy is None
                and (
                    (not result_time_refs and result_population is not None)
                    or (
                        bool(observed_time_nodes)
                        and all(node.time_granularity is None for node in observed_time_nodes)
                    )
                )
            )
            or (
                result.time_policy is not None
                and all(
                    node.time_granularity == result.time_policy.granularity
                    for node in observed_time_nodes
                )
            )
        )
    )
    result_segment_refs = {
        nodes[ref].catalog_ref
        for ref in predicate_ids
        if nodes[ref].kind == "predicate"
        and nodes[ref].aggregate == "segment"
        and nodes[ref].catalog_ref is not None
    }
    axis = result_axis
    axis_known = result_axis_known
    granularity = result_granularity
    segment_refs = result_segment_refs
    population = result_population

    def population_segments(population_ref: str | None) -> tuple[set[str], bool]:
        matched = next(
            (p for p in analysis.populations if p.population_id == population_ref),
            None,
        )
        if matched is None:
            return set(), False
        return (
            {
                catalog_ref
                for ref in matched.predicate_ids
                if nodes[ref].kind == "predicate"
                and nodes[ref].aggregate == "segment"
                and (catalog_ref := nodes[ref].catalog_ref) is not None
            },
            True,
        )

    def occurrence_axis(metric: SemanticNode) -> tuple[str | None, bool]:
        matched = next(
            (p for p in analysis.populations if p.population_id == metric.population_ref),
            None,
        )
        if matched is None:
            return None, False
        refs = {
            nodes[ref].catalog_ref
            for ref in matched.predicate_ids
            if nodes[ref].kind == "predicate"
            and nodes[ref].aggregate == "half_open_range"
            and nodes[ref].catalog_ref is not None
        }
        refs.update(
            nodes[ref].catalog_ref
            for ref in (result.grain_node_ids if result else ())
            if nodes[ref].kind == "time" and nodes[ref].catalog_ref is not None
        )
        refs.update(
            nodes[output.node_id].catalog_ref
            for output in analysis.outputs
            if nodes[output.node_id].kind == "time"
            and nodes[output.node_id].catalog_ref is not None
        )
        return (next(iter(refs)) if len(refs) == 1 else None), available and len(refs) <= 1

    def check(
        code: str,
        truth: bool,
        ref: str,
        explanation: str,
        assumptions: tuple[str, ...],
        *,
        known: bool = True,
    ) -> None:
        status = (
            ProofStatus.NOT_ESTABLISHED
            if not available or not known
            else (ProofStatus.PROVED if truth else ProofStatus.DISPROVED)
        )
        add(code, status, ref, explanation, assumptions)

    metric_nodes: dict[str, list[SemanticNode]] = {}
    for item in analysis.nodes:
        if item.kind == "aggregate" and item.catalog_ref is not None:
            metric_nodes.setdefault(item.catalog_ref, []).append(item)
    for ref, matching_nodes in metric_nodes.items():
        contract = registry.get(ref)
        if contract is None:
            continue
        declaration = f"contract:{ref}@{contract.revision}"
        applied.append(declaration)
        assumptions = (declaration,)
        add(
            "contract_applied",
            ProofStatus.PROVED,
            ref,
            "Checks trusted declarations; does not verify their physical truth.",
            assumptions,
        )
        pending = [item.node_id for item in matching_nodes]
        contributing: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in contributing:
                continue
            contributing.add(node_id)
            pending.extend(nodes[node_id].operand_ids)
        if any(nodes[node_id].masked for node_id in contributing):
            add(
                "masked_value",
                ProofStatus.NOT_ESTABLISHED,
                ref,
                "Masked metric values or contributing operands do not establish numeric semantics.",
                assumptions,
            )
        node = matching_nodes[0]
        if contract.source_grain is not None:
            add(
                "source_grain",
                ProofStatus.PROVED,
                ref,
                f"Declared composite source grain {contract.source_grain.keys!r}.",
                assumptions,
            )
            add(
                "physical_uniqueness",
                ProofStatus.NOT_ESTABLISHED,
                ref,
                "Catalog grain and dependencies do not verify physical rows.",
                assumptions,
                mandatory=False,
            )
        if contract.required_time_axis:
            statuses: list[ProofStatus] = []
            for item in matching_nodes:
                occurrence_axis_ref, known = occurrence_axis(item)
                statuses.append(
                    ProofStatus.PROVED
                    if occurrence_axis_ref is not None
                    else ProofStatus.DISPROVED
                    if known
                    else ProofStatus.NOT_ESTABLISHED
                )
            add(
                "time_axis_required",
                conjunction(tuple(statuses)),
                ref,
                "Metric requires one explicit time axis in each occurrence population.",
                assumptions,
            )
        if contract.permitted_time_axes is not None:
            occurrence_axes = [occurrence_axis(item) for item in matching_nodes]
            status = (
                ProofStatus.NOT_ESTABLISHED
                if any(not known for _axis, known in occurrence_axes)
                else ProofStatus.DISPROVED
                if any(
                    axis_ref is not None and axis_ref not in contract.permitted_time_axes
                    for axis_ref, _ in occurrence_axes
                )
                else ProofStatus.PROVED
            )
            add(
                "time_axis",
                status,
                ref,
                "Every occurrence uses a permitted time axis.",
                assumptions,
            )
        if contract.permitted_granularities is not None:
            check(
                "time_granularity",
                result_granularity in contract.permitted_granularities,
                ref,
                f"Observed bucket granularity {result_granularity!r}.",
                assumptions,
                known=result_granularity_known,
            )
        if contract.calendar_ref is not None:
            add(
                "calendar",
                ProofStatus.NOT_ESTABLISHED,
                ref,
                f"Calendar {contract.calendar_ref!r} is not represented by compiler evidence.",
                assumptions,
            )
        if contract.required_segments:
            statuses = []
            for item in matching_nodes:
                segments, known = population_segments(item.population_ref)
                statuses.append(
                    ProofStatus.NOT_ESTABLISHED
                    if not available or not known
                    else ProofStatus.PROVED
                    if set(contract.required_segments) <= segments
                    else ProofStatus.DISPROVED
                )
            add(
                "required_population",
                conjunction(tuple(statuses)),
                ref,
                "Required named segment predicates must be applied to each occurrence population.",
                assumptions,
            )
        if contract.forbidden_segments:
            statuses = []
            for item in matching_nodes:
                segments, known = population_segments(item.population_ref)
                statuses.append(
                    ProofStatus.NOT_ESTABLISHED
                    if not available or not known
                    else ProofStatus.DISPROVED
                    if set(contract.forbidden_segments) & segments
                    else ProofStatus.PROVED
                )
            add(
                "forbidden_population",
                conjunction(tuple(statuses)),
                ref,
                "Forbidden named segment predicates must be absent from each "
                "occurrence population.",
                assumptions,
            )
        if contract.required_dimensions:
            check(
                "required_grain",
                set(contract.required_dimensions) <= set(grain or ()),
                ref,
                "Required result dimensions must be retained.",
                assumptions,
                known=grain is not None,
            )
        if contract.temporal_behavior == "snapshot":
            preserved = contract.snapshot_axis in (grain or ()) and granularity is None
            check(
                "snapshot_aggregation",
                preserved,
                ref,
                "Snapshot axis must remain unbucketed; analysis has no as-of selector evidence.",
                assumptions,
                known=(
                    node.aggregate in {"sum", "count"}
                    and grain is not None
                    and result_granularity_known
                ),
            )
            advice.append(
                Advice(
                    code="snapshot_selector",
                    reference=ref,
                    recommendation="Select an explicit as-of snapshot before aggregation.",
                )
            )
        if contract.non_additive_dimensions:
            add(
                "rollup_state",
                ProofStatus.NOT_ESTABLISHED,
                ref,
                "Analysis does not establish preserved sufficient state for reaggregation.",
                assumptions,
            )
        if contract.preferred_time_axis and axis != contract.preferred_time_axis:
            advice.append(
                Advice(
                    code="preferred_time_axis",
                    reference=ref,
                    recommendation=f"Consider axis {contract.preferred_time_axis}.",
                    preconditions=("only if compatible with confirmed intent",),
                )
            )
        if contract.preferred_granularity and granularity != contract.preferred_granularity:
            advice.append(
                Advice(
                    code="preferred_granularity",
                    reference=ref,
                    recommendation=f"Consider {contract.preferred_granularity} buckets.",
                    preconditions=("only if compatible with confirmed intent",),
                )
            )
        if node.aggregate in {"avg", "ratio", "count_distinct"}:
            advice.append(
                Advice(
                    code="preserve_state",
                    reference=ref,
                    recommendation="Roll up sufficient state, not finalized values.",
                    preconditions=("reaggregation is requested",),
                )
            )
    if not applied:
        add(
            "contract_scope",
            ProofStatus.NOT_ESTABLISHED,
            "contracts",
            "No applicable contract establishes metric-specific validity.",
        )

    if expectations is not None:

        def compare(code: str, expected: object, actual: object, *, known: bool = True) -> None:
            if expected is None:
                return
            status = (
                ProofStatus.NOT_ESTABLISHED
                if not available or not known
                else (ProofStatus.PROVED if expected == actual else ProofStatus.DISPROVED)
            )
            add(
                code,
                status,
                "expectations",
                "Compared requested expectation with compiler evidence.",
                intent=True,
            )

        metrics = _projected_metrics(analysis) if available else None
        compare(
            "intent_metrics",
            None if expectations.metrics is None else tuple(sorted(expectations.metrics)),
            metrics,
            known=metrics is not None,
        )
        compare(
            "intent_grain",
            None if expectations.grain is None else tuple(sorted(expectations.grain)),
            grain,
            known=grain is not None,
        )
        compare(
            "intent_population",
            expectations.population_inclusion,
            population.inclusion if population is not None else None,
            known=population is not None,
        )
        compare("intent_time_axis", expectations.time_axis, axis, known=axis_known)
        compare("intent_granularity", expectations.granularity, granularity, known=available)
        if expectations.required_segments is not None:
            compare(
                "intent_required_segments",
                True,
                set(expectations.required_segments) <= segment_refs,
                known=result_population is not None,
            )
        if expectations.forbidden_segments is not None:
            compare(
                "intent_forbidden_segments",
                True,
                not (set(expectations.forbidden_segments) & segment_refs),
                known=result_population is not None,
            )
        compare("intent_calendar", expectations.calendar_ref, None, known=False)
        compare("comparison_basis", expectations.comparison_basis, None, known=False)
    confirmed = ProofStatus.NOT_ESTABLISHED
    if expectations is not None and expectations.origin == "independently_confirmed" and alignment:
        confirmed = conjunction(tuple(item.status for item in alignment))
    return SemanticCheck(
        structural_validity=structural,
        obligations=tuple(obligations),
        advice=tuple(advice),
        alignment=tuple(alignment),
        intent_origin=expectations.origin if expectations else None,
        intent_evidence_ref=expectations.evidence_ref if expectations else None,
        confirmed_intent_alignment=confirmed,
        execution_explanation=(
            "No lowering evidence supplied; checking does not establish execution success."
        ),
        applied_contracts=tuple(applied),
        compiler_assumptions=tuple(
            f"{a.reference}={a.declared_value} ({a.verification})" for a in analysis.assumptions
        ),
        require_confirmed_intent=require_confirmed_intent,
    )
