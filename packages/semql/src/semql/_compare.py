"""Structured comparison of semantic descriptors, not executable SQL or local IDs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, cast

from pydantic import BaseModel

from semql.errors import ContractError

if TYPE_CHECKING:
    from semql.analysis import Comparison, ComparisonScope, SemanticAnalysis


class _UnknownSemantics(Exception):
    """A well-formed artifact uses a semantic vocabulary this reader cannot prove."""


_KINDS = frozenset({"aggregate", "dimension", "time", "arithmetic", "comparison", "predicate"})
_AGGREGATES = frozenset(
    {
        "sum",
        "count",
        "count_distinct",
        "avg",
        "min",
        "max",
        "median",
        "p75",
        "p90",
        "p95",
        "ratio",
    }
)
_ARITHMETIC = frozenset({"sum", "diff", "ratio"})
_PREDICATES = frozenset(
    {
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
        "tenancy",
        "security",
        "base_predicate",
        "half_open_range",
    }
)


def _invalid(reason: str) -> ContractError:
    return ContractError(
        "Semantic artifact reference graph is invalid.",
        reason=reason,
        operation="compare",
        stage="analysis",
    )


def _value(value: object) -> object:
    """Normalize typed descriptor values while ignoring optional annotations."""
    if isinstance(value, BaseModel):
        return tuple((name, _value(getattr(value, name))) for name in type(value).model_fields)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        return tuple(sorted((str(key), _value(item)) for key, item in mapping.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_value(item) for item in cast(Sequence[object], value))
    if isinstance(value, (set, frozenset)):
        members = cast(set[object] | frozenset[object], value)
        return tuple(sorted((_value(item) for item in members), key=repr))
    return value


class _Graph:
    def __init__(
        self, analysis: SemanticAnalysis, context: Mapping[str, object] | None = None
    ) -> None:
        self.analysis = analysis
        self.private_context = context
        self.nodes = {node.node_id: node for node in analysis.nodes}
        self.populations = {
            population.population_id: population for population in analysis.populations
        }
        self.bindings = {binding.slot: binding for binding in analysis.binding_dependencies}
        if len(self.nodes) != len(analysis.nodes) or any(not key for key in self.nodes):
            raise _invalid("invalid_node_ids")
        if len(self.populations) != len(analysis.populations) or any(
            not key for key in self.populations
        ):
            raise _invalid("invalid_population_ids")
        if len(self.bindings) != len(analysis.binding_dependencies) or any(
            not key for key in self.bindings
        ):
            raise _invalid("invalid_binding_ids")
        output_ids = {output.output_id for output in analysis.outputs}
        aliases = {output.sql_alias for output in analysis.outputs}
        if len(output_ids) != len(analysis.outputs) or "" in output_ids:
            raise _invalid("invalid_output_ids")
        if len(aliases) != len(analysis.outputs):
            raise _invalid("duplicate_output_alias")
        self.active: set[tuple[str, str]] = set()
        self.memo: dict[tuple[str, str], object] = {}
        for output in analysis.outputs:
            if output.node_id not in self.nodes:
                raise _invalid("dangling_node_reference")
        for provenance in analysis.enrichments:
            if provenance.output_id not in output_ids:
                raise _invalid("dangling_enrichment_reference")

    def binding(self, slot: str) -> object:
        binding = self.bindings.get(slot)
        if binding is None:
            raise _invalid("dangling_binding_reference")
        definition = tuple(
            (name, _value(getattr(binding, name)))
            for name in type(binding).model_fields
            if name != "slot"
        )
        if self.private_context is not None:
            return (*definition, ("private_value", _value(self.private_context[slot])))
        return definition

    def node(self, node_id: str) -> object:
        return self._visit("node", node_id)

    def population(self, population_id: str) -> object:
        return self._visit("population", population_id)

    def _visit(self, domain: str, local_id: str) -> object:
        key = (domain, local_id)
        if key in self.active:
            raise _invalid("cyclic_semantic_graph")
        if key in self.memo:
            return self.memo[key]
        model = self.nodes.get(local_id) if domain == "node" else self.populations.get(local_id)
        if model is None:
            raise _invalid(f"dangling_{domain}_reference")
        self.active.add(key)
        try:
            descriptor = self.descriptor(model)
        finally:
            self.active.remove(key)
        self.memo[key] = descriptor
        return descriptor

    def descriptor(self, model: BaseModel) -> object:
        fields: list[tuple[str, object]] = []
        for name in type(model).model_fields:
            if name in {"node_id", "population_id", "slot", "evaluated_scopes"}:
                continue
            value = getattr(model, name)
            if value is None:
                normalized: object = None
            elif name in {"operand_ids", "predicate_ids"}:
                normalized = tuple(self.node(item) for item in value)
                if name == "predicate_ids":
                    normalized = tuple(sorted(normalized, key=repr))
            elif name == "population_ref":
                normalized = self.population(value)
            elif name == "binding_refs":
                normalized = tuple(sorted((self.binding(item) for item in value), key=repr))
            elif name.endswith("binding_ref"):
                normalized = self.binding(value)
            elif name == "grain_node_ids":
                normalized = tuple(sorted((self.node(item) for item in value), key=repr))
            elif name == "order":
                normalized = tuple((self.node(item[0]), item[1]) for item in value)
            elif name in {"grain_refs", "source_refs", "policy_refs", "mask_roles"}:
                normalized = tuple(sorted(value))
            elif isinstance(value, BaseModel):
                normalized = self.descriptor(value)
            else:
                normalized = _value(value)
            fields.append((name, normalized))
        return tuple(fields)

    def validate(self) -> None:
        # Validate the complete artifact, including hidden/unprojected nodes.
        for node_id in self.nodes:
            self.node(node_id)
        for population_id in self.populations:
            self.population(population_id)
        if self.analysis.result is not None:
            self.descriptor(self.analysis.result)
        for derivation in self.analysis.derivations:
            for node_id in derivation.logical_node_ids:
                self.node(node_id)
        for node in self.nodes.values():
            if node.kind not in _KINDS:
                raise _UnknownSemantics
            if node.kind == "aggregate" and node.aggregate not in _AGGREGATES:
                raise _UnknownSemantics
            if node.kind == "arithmetic" and node.aggregate not in _ARITHMETIC:
                raise _UnknownSemantics
            if node.kind == "predicate" and node.aggregate not in _PREDICATES:
                raise _UnknownSemantics
            if node.kind == "comparison" and node.aggregate not in {
                "current",
                "prior",
                "delta",
                "pct_change",
            }:
                raise _UnknownSemantics
        descriptors = self.outputs()
        if len(set(descriptors)) != len(descriptors):
            raise _invalid("duplicate_semantic_output")

    def outputs(self) -> tuple[object, ...]:
        return tuple(
            sorted((self.node(output.node_id) for output in self.analysis.outputs), key=repr)
        )

    def context(self, values: Mapping[str, object]) -> object | None:
        if not self.bindings.keys() <= values.keys():
            return None
        # Retain the binding's position in the expression graph: comparing a
        # bag of values would lose correlations across Boolean branches.
        bound = _Graph(self.analysis, values)
        return (
            bound.outputs(),
            bound.descriptor(self.analysis.result) if self.analysis.result is not None else None,
            tuple(sorted((bound.binding(slot) for slot in self.bindings), key=repr)),
        )


def compare_analysis(
    left: SemanticAnalysis,
    right: SemanticAnalysis,
    *,
    scope: ComparisonScope | Literal["expression", "result"] = "expression",
    left_context: Mapping[str, object] | None = None,
    right_context: Mapping[str, object] | None = None,
    shared_catalog_snapshot: bool = False,
    bound_context_equal: bool | None = None,
) -> Comparison:
    """Follow local references and compare their declared logical meanings.

    Private context maps use each artifact's binding-dependency slots, not SQL
    parameter names. The trusted equality attestation is an alternative to
    supplying those values; neither values nor hashes of them are returned.
    """
    from semql.analysis import Comparison, ComparisonResult, ComparisonScope

    selected_scope = ComparisonScope(scope)

    def result(outcome: ComparisonResult, *reasons: str) -> Comparison:
        return Comparison(outcome=outcome, scope=selected_scope, reasons=reasons)

    if left.schema_version != 1 or right.schema_version != 1:
        return result(ComparisonResult.NOT_ESTABLISHED, "analysis_version_unsupported")
    if left.required_semantics or right.required_semantics:
        return result(ComparisonResult.NOT_ESTABLISHED, "required_semantics_unsupported")
    lg, rg = _Graph(left), _Graph(right)
    try:
        lg.validate()
        rg.validate()
    except _UnknownSemantics:
        return result(ComparisonResult.NOT_ESTABLISHED, "unknown_semantic_vocabulary")
    if left.coverage.status != "complete" or right.coverage.status != "complete":
        return result(ComparisonResult.NOT_ESTABLISHED, "analysis_coverage_incomplete")
    lc, rc = left.catalog_context, right.catalog_context
    if lc is None or rc is None or not lc.namespace or not rc.namespace:
        if not shared_catalog_snapshot:
            return result(ComparisonResult.NOT_ESTABLISHED, "catalog_context_missing")
    elif lc.namespace != rc.namespace:
        return result(ComparisonResult.NOT_ESTABLISHED, "catalog_namespace_incompatible")
    if lc is None or rc is None or lc.semantic_revision is None or rc.semantic_revision is None:
        if not shared_catalog_snapshot:
            return result(ComparisonResult.NOT_ESTABLISHED, "catalog_revision_missing")
    elif lc.semantic_revision != rc.semantic_revision:
        return result(ComparisonResult.NOT_ESTABLISHED, "catalog_revision_incompatible")
    if lg.outputs() != rg.outputs():
        return result(ComparisonResult.DIFFERENT, "semantic_descriptors_differ")
    if selected_scope is ComparisonScope.EXPRESSION:
        return result(ComparisonResult.EQUIVALENT)
    if left.result is None or right.result is None:
        return result(ComparisonResult.NOT_ESTABLISHED, "result_contract_missing")
    if lg.descriptor(left.result) != rg.descriptor(right.result):
        return result(ComparisonResult.DIFFERENT, "result_contract_differs")
    if tuple(lg.descriptor(join) for join in left.joins) != tuple(
        rg.descriptor(join) for join in right.joins
    ):
        return result(ComparisonResult.DIFFERENT, "join_semantics_differ")
    # Materialized merge-key obligations describe a physical strategy.
    # All other declared preconditions belong to logical result equivalence.
    left_assumptions = tuple(
        sorted(
            (lg.descriptor(item) for item in left.assumptions if item.kind != "unique_merge_key"),
            key=repr,
        )
    )
    right_assumptions = tuple(
        sorted(
            (rg.descriptor(item) for item in right.assumptions if item.kind != "unique_merge_key"),
            key=repr,
        )
    )
    if left_assumptions != right_assumptions:
        return result(ComparisonResult.DIFFERENT, "logical_assumptions_differ")
    if left_context is not None and right_context is not None:
        lcontext, rcontext = lg.context(left_context), rg.context(right_context)
        if lcontext is None or rcontext is None:
            return result(ComparisonResult.NOT_ESTABLISHED, "bound_context_incomplete")
        if lcontext != rcontext:
            return result(ComparisonResult.DIFFERENT, "bound_context_differs")
        bound_context_equal = True
    if bound_context_equal is False:
        return result(ComparisonResult.DIFFERENT, "bound_context_differs")
    if (lg.bindings or rg.bindings) and bound_context_equal is None:
        return result(ComparisonResult.NOT_ESTABLISHED, "bound_context_not_attested")
    return result(ComparisonResult.EQUIVALENT)
