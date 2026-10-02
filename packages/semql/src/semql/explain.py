"""Safe human rendering for existing compiled semantic analysis.

Unlike :meth:`semql.catalog.Catalog.explain`, this module does not resolve a
query or render the privileged logical-plan representation.  It only reads an
artifact's already-built :class:`~semql.analysis.SemanticAnalysis` graph.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from semql.analysis import SemanticAnalysis, SemanticAssumption, SemanticNode

_MAX_LABEL_LENGTH = 120

_KINDS = {
    "aggregate": "aggregate",
    "derived": "derived measure",
    "dimension": "dimension",
    "predicate": "predicate",
    "time": "time dimension",
}
_AGGREGATES = {
    "avg",
    "count",
    "count_distinct",
    "max",
    "median",
    "min",
    "p75",
    "p90",
    "p95",
    "ratio",
    "stddev",
    "sum",
}
_JOIN_KINDS = {
    "full": "full",
    "inner": "inner",
    "left": "left",
    "right": "right",
}
_RELATIONSHIPS = {
    "many_to_many": "many-to-many",
    "many_to_one": "many-to-one",
    "one_to_many": "one-to-many",
    "one_to_one": "one-to-one",
}

_RECIPE_DESCRIPTIONS = {
    "federated_projection": "projects a dimension from backend fragments",
    "fragment_avg": "merges fragment averages",
    "fragment_count": "merges fragment counts",
    "fragment_max": "merges fragment maxima",
    "fragment_min": "merges fragment minima",
    "fragment_sum": "merges fragment sums",
    "merged_numerator_denominator": "recomposes a ratio from merged operands",
    "stored_measure_reaggregation": "re-aggregates stored rollup measures",
    "sum_count_recomposition": "recomposes an average from fragment sums and counts",
    "sum_partial_aggregates": "sums partial aggregates",
    "sum_stored_partial_counts": "sums stored partial counts",
}
_COVERAGE_DESCRIPTIONS = {
    "analysis_unavailable": "semantic analysis is unavailable",
    "effective_scope_not_established": "effective population coverage is not established",
    "no_projected_outputs": "projected output semantics are not established",
    "order_bindings_not_established": "ordering semantics are not established",
    "quantile_method_not_established": "quantile method semantics are not established",
    "semi_join_semantics_not_established": "semi-join semantics are not established",
}


def _display(value: str) -> str:
    """Return one bounded JSON string literal, escaping controls and newlines."""
    shortened = value[:_MAX_LABEL_LENGTH]
    if len(value) > _MAX_LABEL_LENGTH:
        shortened += "…"
    return json.dumps(shortened, ensure_ascii=True)


def _safe_token(value: str | None, known: dict[str, str] | set[str]) -> str:
    if value is None:
        return "unknown"
    if isinstance(known, set):
        return value if value in known else "unknown"
    return known.get(value, "unknown")


def _node_description(node: SemanticNode) -> str:
    parts = [_KINDS.get(node.kind, "semantic value")]
    if node.aggregate is not None:
        parts.append(f"aggregation {_safe_token(node.aggregate, _AGGREGATES)}")
    if node.unit is not None:
        state = "declared" if node.unit_state == "declared" else "not established"
        parts.append(f"unit {_display(node.unit)} ({state})")
    elif node.kind in {"aggregate", "derived"}:
        parts.append("unit unknown")
    if node.period_role is not None:
        role = node.period_role if node.period_role in {"current", "prior"} else "unknown"
        parts.append(f"period {role}")
    if node.time_granularity is not None:
        parts.append(f"time grain {_display(node.time_granularity)}")
    return "; ".join(parts)


def _assumption_evidence(assumption: SemanticAssumption | None) -> str:
    if assumption is None:
        return "declared in join descriptor; runtime evidence unavailable"
    return "declared; runtime evidence unavailable"


def _hidden_operand_ids(
    output_ids: set[str], nodes: dict[str, SemanticNode]
) -> tuple[set[str], bool]:
    hidden: set[str] = set()
    dangling = False
    pending = [
        operand
        for node_id in output_ids
        if (node := nodes.get(node_id)) is not None
        for operand in node.operand_ids
    ]
    while pending:
        node_id = pending.pop()
        if node_id in output_ids or node_id in hidden:
            continue
        node = nodes.get(node_id)
        if node is None:
            dangling = True
            continue
        hidden.add(node_id)
        pending.extend(node.operand_ids)
    return hidden, dangling


def _physical_strategy(analysis: SemanticAnalysis) -> tuple[str, tuple[str, ...]]:
    recipes = {derivation.recipe for derivation in analysis.derivations}
    federated = any(
        derivation.operation in {"dimension", "join"}
        or derivation.recipe.startswith(("fragment_", "raw_rows_"))
        or derivation.recipe
        in {
            "federated_projection",
            "merged_numerator_denominator",
            "sum_count_recomposition",
            "sum_partial_aggregates",
        }
        or any(ref.startswith("fragment:") for ref in derivation.physical_refs)
        for derivation in analysis.derivations
    )
    strategy = (
        "federated merge"
        if federated
        else "native query"
        if recipes & {"stored_measure_reaggregation", "sum_stored_partial_counts"}
        else "unknown"
    )
    details: list[str] = []
    for recipe in sorted(recipes):
        description = _RECIPE_DESCRIPTIONS.get(recipe)
        if description is None and recipe.startswith("raw_rows_"):
            description = "aggregates merged raw rows"
        if description is not None and description not in details:
            details.append(description)
    return strategy, tuple(details)


def _append_unknowns(lines: list[str], unknowns: Iterable[str]) -> None:
    unique = tuple(dict.fromkeys(unknowns))
    lines.append("Unknowns:")
    if not unique:
        lines.append("- None reported by the semantic analysis.")
        return
    lines.extend(f"- {unknown}." for unknown in unique)


def render_analysis(analysis: SemanticAnalysis) -> str:
    """Render an existing semantic graph as bounded, safe human-readable text.

    The renderer performs no catalog lookup, query re-resolution, SQL parsing,
    or engine work.  It deliberately omits catalog/source references, physical
    reference names, bindings, predicate values, policy inputs, mask values,
    and raw SQL.  Labels are bounded and JSON-escaped so an untrusted alias
    cannot inject lines or terminal controls.
    """
    lines = [f"Semantic analysis (schema {analysis.schema_version})"]
    coverage = analysis.coverage.status.replace("_", " ")
    lines.append(f"Coverage: {coverage}.")

    nodes: dict[str, SemanticNode] = {}
    duplicate_node_ids = False
    node: SemanticNode | None
    for node in analysis.nodes:
        if node.node_id in nodes:
            duplicate_node_ids = True
            continue
        nodes[node.node_id] = node

    lines.append("Outputs:")
    output_node_ids: set[str] = set()
    dangling_outputs = False
    if not analysis.outputs:
        lines.append("- No outputs are described.")
    for output in analysis.outputs:
        node = nodes.get(output.node_id)
        if node is None:
            dangling_outputs = True
            lines.append(f"- {_display(output.sql_alias)}: semantics unavailable.")
            continue
        output_node_ids.add(node.node_id)
        lines.append(f"- {_display(output.sql_alias)}: {_node_description(node)}.")

    hidden_ids, dangling_operands = _hidden_operand_ids(output_node_ids, nodes)
    lines.append("Hidden operands:")
    if not hidden_ids:
        lines.append("- None described.")
    else:
        for index, node_id in enumerate(sorted(hidden_ids), start=1):
            lines.append(f"- Operand {index}: {_node_description(nodes[node_id])}.")

    lines.append("Result grain:")
    result = analysis.result
    dangling_grain = False
    if result is None:
        lines.append("- Unavailable.")
    else:
        lines.append(f"- Kind: {result.grain_kind}.")
        aliases_by_node = {output.node_id: output.sql_alias for output in analysis.outputs}
        if not result.grain_node_ids:
            lines.append("- Components: none (scalar result).")
        for node_id in result.grain_node_ids:
            node = nodes.get(node_id)
            if node is None:
                dangling_grain = True
                lines.append("- Component: unavailable.")
            elif node_id in aliases_by_node:
                lines.append(f"- Component: {_display(aliases_by_node[node_id])}.")
            else:
                lines.append(f"- Component: hidden {_KINDS.get(node.kind, 'semantic value')}.")
        if result.time_policy is not None:
            policy = result.time_policy
            if policy.granularity is not None:
                lines.append(f"- Period grain: {_display(policy.granularity)}.")
            if policy.alignment is not None:
                alignment = (
                    policy.alignment
                    if policy.alignment in {"explicit", "half_open", "previous_period"}
                    else "unknown"
                )
                lines.append(f"- Period alignment: {alignment.replace('_', ' ')}.")

    relationship_assumptions = {
        assumption.reference: assumption
        for assumption in analysis.assumptions
        if assumption.kind == "relationship"
    }
    lines.append("Joins and cardinality:")
    if not analysis.joins:
        lines.append("- No joins are described.")
    for index, join in enumerate(analysis.joins):
        assumption_ref = f"join:{join.source_ref}->{join.target_ref}"
        assumption = relationship_assumptions.get(assumption_ref)
        kind = _safe_token(join.kind, _JOIN_KINDS)
        relationship = _safe_token(join.relationship, _RELATIONSHIPS)
        lines.append(
            f"- Join {index + 1}: {kind}; cardinality {relationship} "
            f"({_assumption_evidence(assumption)})."
        )
    merge_key_assumptions = [
        assumption for assumption in analysis.assumptions if assumption.kind == "unique_merge_key"
    ]
    for index, assumption in enumerate(merge_key_assumptions, start=1):
        lines.append(f"- Merge key {index}: uniqueness {_assumption_evidence(assumption)}.")

    strategy, details = _physical_strategy(analysis)
    strategy_label = (
        "Physical strategy: not established."
        if strategy == "unknown"
        else f"Physical strategy: {strategy}."
    )
    lines.append(strategy_label)
    lines.extend(f"- {detail}." for detail in details)

    unknowns: list[str] = []
    if strategy == "unknown":
        unknowns.append("Physical execution strategy is not established")
    if analysis.coverage.status != "complete":
        unknowns.append("Semantic coverage is not complete")
    unknown_codes_seen: set[str] = set()
    for code in analysis.coverage.not_established:
        description = _COVERAGE_DESCRIPTIONS.get(code)
        if description is not None:
            unknowns.append(description.capitalize())
        elif code not in unknown_codes_seen:
            unknowns.append("Additional semantic coverage is unavailable")
            unknown_codes_seen.add(code)
    relevant_nodes = [
        nodes[node_id] for node_id in output_node_ids | hidden_ids if node_id in nodes
    ]
    if any(
        node.kind in {"aggregate", "derived"} and node.unit_state != "declared"
        for node in relevant_nodes
    ):
        unknowns.append("One or more output units are not established")
    if result is None:
        unknowns.append("Result grain and population are unavailable")
    if duplicate_node_ids:
        unknowns.append("Duplicate semantic node identifiers make part of the graph unavailable")
    if dangling_outputs or dangling_operands or dangling_grain:
        unknowns.append("Dangling semantic graph references make part of the graph unavailable")
    if analysis.assumptions:
        unknowns.append("Runtime evidence for one or more declared assumptions is unavailable")
    _append_unknowns(lines, unknowns)
    return "\n".join(lines)


__all__ = ["render_analysis"]
