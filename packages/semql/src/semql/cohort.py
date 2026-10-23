"""Typed single-hop cohort specifications and compiler."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from semql.analysis import CatalogContext
from semql.backend import DialectStrategy
from semql.compile import ColumnMeta
from semql.errors import FederationError, ResolveError
from semql.federate import FederatedPlan, compile_federated_query
from semql.introspect import PolicyFn, ScopeFn, resolve_field
from semql.model import (
    AuthContext,
    Cube,
    Dialect,
    Dimension,
    FrozenModel,
    Measure,
    StorageType,
    View,
)
from semql.spec import BoolExpr, DecimalLiteral, Filter, InlineDerived, SemanticQuery
from semql.units import DEFAULT_REGISTRY

COHORT_PLAN_VERSION = 1
_MAX_COHORT_MEMBERS = 1_000


class _CohortSpec(FrozenModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Cohort(_CohortSpec):
    source: SemanticQuery
    member_keys: Sequence[str] = Field(min_length=1)
    attributes: Sequence[str] = ()
    max_members: int = Field(default=1000, gt=0, le=_MAX_COHORT_MEMBERS, strict=True)

    @model_validator(mode="after")
    def _shape(self) -> Cohort:
        refs = (*self.member_keys, *self.attributes)
        if len(set(refs)) != len(refs):
            raise ValueError("cohort keys and attributes must be distinct")
        if self.source.ungrouped:
            raise ValueError("cohort source must remain grouped")
        if self.source.aliases:
            raise ValueError("cohort source aliases are not supported")
        if self.source.semi_joins:
            raise ValueError("cohort source cannot contain staged semi-joins")
        if self.source.time_dimension is not None and (
            self.source.time_dimension.granularity is not None
            or self.source.time_dimension.fill_nulls_with is not None
        ):
            raise ValueError("cohort source cannot use a grouped or filled time dimension")
        if set(self.source.dimensions) != set(refs):
            raise ValueError("source dimensions must exactly equal declared keys and attributes")
        return self


class CohortMeasurement(_CohortSpec):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    query: SemanticQuery
    key_mapping: Mapping[str, str]
    attributes: Sequence[str] = ()

    @model_validator(mode="after")
    def _shape(self) -> CohortMeasurement:
        if not self.key_mapping:
            raise ValueError("key_mapping must not be empty")
        if len(set(self.key_mapping.values())) != len(self.key_mapping):
            raise ValueError("each cohort key must map to a distinct measurement dimension")
        if set(self.key_mapping.values()) & set(self.attributes):
            raise ValueError("measurement key and attribute dimensions must be distinct")
        if (
            self.query.rank is not None
            or self.query.limit is not None
            or self.query.offset is not None
        ):
            raise ValueError("measurement cannot use rank, limit, or offset")
        if self.query.ungrouped or self.query.semi_joins:
            raise ValueError("measurement must be a grouped query without staged operators")
        if self.query.aliases:
            raise ValueError("measurement aliases are not supported")
        if self.query.time_dimension is not None and (
            self.query.time_dimension.granularity is not None
            or self.query.time_dimension.fill_nulls_with is not None
        ):
            raise ValueError("measurement cannot use a grouped or filled time dimension")
        if self.query.compare is not None:
            raise ValueError("measurement cannot use query comparison mode")
        if set(self.query.dimensions) != set(self.key_mapping.values()) | set(self.attributes):
            raise ValueError("measurement dimensions must exactly equal mapped keys and attributes")
        return self


class CohortMetric(_CohortSpec):
    measurement: str = Field(min_length=1)
    column: str = Field(min_length=1)


class CohortComparison(_CohortSpec):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    left: CohortMetric
    right: CohortMetric
    op: Literal["difference", "ratio", "percent_change"] = "difference"


class CohortQuery(_CohortSpec):
    cohort: Cohort
    measurements: Sequence[CohortMeasurement] = Field(min_length=1)
    comparisons: Sequence[CohortComparison] = ()

    @model_validator(mode="after")
    def _names(self) -> CohortQuery:
        names = [x.name for x in self.measurements]
        comparisons = [x.name for x in self.comparisons]
        if len(set(names)) != len(names) or len(set(comparisons)) != len(comparisons):
            raise ValueError("measurement and comparison names must be unique")
        if set(names) & set(comparisons) or set(names) & {"selection", "comparison"}:
            raise ValueError("measurement names collide with reserved output namespaces")
        if any(
            metric.measurement not in names
            for comparison in self.comparisons
            for metric in (comparison.left, comparison.right)
        ):
            raise ValueError("comparison references unknown measurement")
        return self


@dataclass(frozen=True)
class CohortColumn:
    name: str
    kind: str
    storage_type: str | None
    source_column: str
    child_meta: ColumnMeta | None = None


@dataclass(frozen=True)
class CohortStage:
    name: str
    plan: FederatedPlan
    columns: tuple[str, ...]
    column_meta: tuple[ColumnMeta, ...]
    key_columns: tuple[str, ...]
    attribute_columns: tuple[str, ...]
    metric_columns: tuple[str, ...]
    key_types: tuple[str, ...]
    query: SemanticQuery
    bind_members: Callable[[Sequence[tuple[Any, ...]]], FederatedPlan] | None = None


@dataclass(frozen=True)
class CohortPlan:
    cohort: CohortQuery
    source: CohortStage
    measurements: tuple[CohortStage, ...]
    columns: tuple[CohortColumn, ...]
    catalog_context: CatalogContext | None = None
    version: int = COHORT_PLAN_VERSION


def compile_cohort_query(
    query: CohortQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    dialects: dict[Dialect, DialectStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    catalog_context: CatalogContext | None = None,
) -> CohortPlan:
    """Compile and authorize all fixed stage shapes without adapter I/O."""
    compile_catalog = dict(catalog)
    compile_context = dict(context) if context is not None else None
    compile_dialects = dict(dialects) if dialects is not None else None
    compile_views = dict(views) if views is not None else None
    compile_scope_fns = dict(scope_fns) if scope_fns is not None else None

    def compile_stage(stage_query: SemanticQuery) -> FederatedPlan:
        return compile_federated_query(
            stage_query,
            compile_catalog,
            context=compile_context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=compile_dialects,
            views=compile_views,
            viewer=viewer,
            policy=policy,
            scope_fns=compile_scope_fns,
            catalog_context=catalog_context,
        )

    def dimension_type(ref: str) -> str:
        try:
            _cube, field = resolve_field(ref, compile_catalog, views=compile_views)
        except ResolveError:
            raise FederationError(
                "Cohort identity reference must resolve to an authorized dimension.",
                reason="cohort_key_reference",
            ) from None
        if not isinstance(field, Dimension):
            raise FederationError(
                "Cohort identity reference must resolve to a dimension.",
                reason="cohort_key_reference",
            )
        return field.type

    def output_column(ref: str, plan: FederatedPlan, *, role: str) -> str:
        try:
            cube, field = resolve_field(ref, compile_catalog, views=compile_views)
        except ResolveError:
            raise FederationError(
                "Cohort output reference must resolve to an authorized field.",
                reason=f"cohort_{role}_reference",
            ) from None
        canonical_ref = f"{cube.name}.{field.name}"
        nodes = [node.node_id for node in plan.analysis.nodes if node.catalog_ref == canonical_ref]
        aliases = {
            output.sql_alias
            for output in plan.analysis.outputs
            if output.node_id in nodes and output.sql_alias in plan.columns
        }
        if len(aliases) != 1:
            raise FederationError(
                "Cohort output reference does not identify one compiled column.",
                reason=f"cohort_{role}_shape",
            )
        return next(iter(aliases))

    def _bound_value(value: object) -> str | int | float | bool | DecimalLiteral:
        if isinstance(value, Decimal):
            return DecimalLiteral(value=str(value))
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, (str, int, float, bool)):
            return value
        raise FederationError(
            "Member key cannot be represented as a bound filter value.",
            reason="cohort_member_key_type",
        )

    def measure_unit_signature(
        ref: str, visiting: frozenset[str] = frozenset()
    ) -> tuple[tuple[Any, ...], str | None]:
        try:
            cube, field = resolve_field(ref, compile_catalog, views=compile_views)
        except ResolveError:
            raise FederationError(
                "Comparison operand must resolve to a declared measure.",
                reason="cohort_comparison_column",
            ) from None
        if not isinstance(field, Measure):
            raise FederationError(
                "Comparison operand must resolve to a declared measure.",
                reason="cohort_comparison_column",
            )
        canonical_ref = f"{cube.name}.{field.name}"
        if canonical_ref in visiting:
            raise FederationError(
                "Cyclic comparison unit declarations are unsupported.",
                reason="cohort_comparison_unit_unknown",
            )
        if field.unit is not None:
            return ("unit", DEFAULT_REGISTRY.canonical(field.unit)), field.unit
        if field.agg == "ratio":
            children = tuple(
                measure_unit_signature(f"{cube.name}.{name}", visiting | {canonical_ref})
                for name in (field.numerator, field.denominator)
                if name is not None
            )
            if children[0][0] == children[1][0]:
                return ("dimensionless",), None
            return ("ratio", children[0][0], children[1][0]), None
        return ("unproven", canonical_ref), None

    def derived_unit_signature(
        derived: InlineDerived,
    ) -> tuple[tuple[Any, ...], str | None]:
        operands = tuple(measure_unit_signature(ref) for ref in derived.operands)
        if derived.op in {"sum", "diff"}:
            if any(signature != operands[0][0] for signature, _ in operands[1:]):
                raise FederationError(
                    "Inline comparison arithmetic has incompatible units.",
                    reason="cohort_comparison_unit_mismatch",
                )
            return operands[0]
        if operands[0][0] == operands[1][0]:
            return ("dimensionless",), None
        return ("ratio", operands[0][0], operands[1][0]), None

    source_limit = query.cohort.source.limit
    effective_limit = query.cohort.max_members + 1
    if source_limit is not None:
        effective_limit = min(source_limit, effective_limit)
    source_query = query.cohort.source.model_copy(update={"limit": effective_limit})
    source_plan = compile_stage(source_query)
    source_types = {ref: dimension_type(ref) for ref in query.cohort.member_keys}
    keys = tuple(output_column(ref, source_plan, role="source") for ref in query.cohort.member_keys)
    attrs = tuple(output_column(ref, source_plan, role="source") for ref in query.cohort.attributes)
    source_meta = tuple(source_plan.column_meta)
    source_meta_map = {meta.name: meta for meta in source_meta}
    if any(source_meta_map[name].masked for name in (*keys, *attrs)):
        raise FederationError(
            "Masked fields cannot define identity or attribution.", reason="cohort_masked_identity"
        )
    source = CohortStage(
        "selection",
        source_plan,
        tuple(source_plan.columns),
        source_meta,
        keys,
        attrs,
        (),
        tuple(dimension_type(ref) for ref in query.cohort.member_keys),
        source_query,
    )
    stages: list[CohortStage] = []
    columns = [
        CohortColumn(
            f"selection.{name}",
            "dimension",
            source_meta_map[name].storage_type,
            name,
            source_meta_map[name],
        )
        for name in (*keys, *attrs)
    ]
    stage_unit_signatures: dict[str, dict[str, tuple[tuple[Any, ...], str | None]]] = {}
    for item in query.measurements:
        if set(item.key_mapping) != set(query.cohort.member_keys):
            raise FederationError(
                "Measurement key mapping must cover every cohort key exactly.",
                reason="cohort_key_mapping",
            )
        plan = compile_stage(item.query)
        targets = tuple(item.key_mapping[ref] for ref in query.cohort.member_keys)
        for source_ref, target_ref in zip(query.cohort.member_keys, targets, strict=True):
            if source_types[source_ref] != dimension_type(target_ref):
                raise FederationError(
                    "Cohort and measurement key types are incompatible.",
                    reason="cohort_key_type_mismatch",
                )
        key_columns = tuple(output_column(ref, plan, role="measurement") for ref in targets)
        attr_columns = tuple(
            output_column(ref, plan, role="measurement") for ref in item.attributes
        )
        metrics = tuple(
            meta.name for meta in plan.column_meta if meta.kind in ("measure", "computed")
        )
        if not metrics or any(
            plan.columns.count(name) != 1 for name in (*key_columns, *attr_columns)
        ):
            raise FederationError(
                "Measurement projection is ambiguous or has no metric.",
                reason="cohort_measurement_shape",
            )
        if any(
            plan.column_meta[plan.columns.index(name)].masked
            for name in (*key_columns, *attr_columns)
        ):
            raise FederationError(
                "Masked fields cannot define measurement identity or attribution.",
                reason="cohort_masked_identity",
            )

        def bind_members(
            members: Sequence[tuple[Any, ...]],
            *,
            template: SemanticQuery = item.query,
            target_refs: tuple[str, ...] = targets,
        ) -> FederatedPlan:
            if not members:
                raise ValueError("empty cohorts must skip measurement compilation and execution")

            def trie(depth: int, tuples: Sequence[tuple[Any, ...]]) -> BoolExpr | Filter:
                grouped: dict[Any, list[tuple[Any, ...]]] = {}
                for member in tuples:
                    grouped.setdefault(member[depth], []).append(member)
                condition = Filter(
                    dimension=target_refs[depth],
                    op="in",
                    values=tuple(_bound_value(value) for value in grouped),
                )
                if depth == len(target_refs) - 1:
                    return condition
                clauses: list[BoolExpr | Filter] = [condition]
                for value, suffixes in grouped.items():
                    not_this_prefix = Filter(
                        dimension=target_refs[depth],
                        op="neq",
                        values=[_bound_value(value)],
                    )
                    clauses.append(
                        BoolExpr(
                            op="or",
                            children=(not_this_prefix, trie(depth + 1, suffixes)),
                        )
                    )
                return BoolExpr(op="and", children=tuple(clauses))

            member_predicate = trie(0, members)
            if isinstance(member_predicate, Filter):
                bound = template.model_copy(
                    update={
                        "filters": [*template.filters, member_predicate],
                        "limit": len(members) + 1,
                    }
                )
            else:
                predicates = ([template.where] if template.where is not None else []) + [
                    member_predicate
                ]
                where = (
                    predicates[0]
                    if len(predicates) == 1
                    else BoolExpr(op="and", children=tuple(predicates))
                )
                bound = template.model_copy(update={"where": where, "limit": len(members) + 1})
            return compile_stage(bound)

        stage = CohortStage(
            item.name,
            plan,
            tuple(plan.columns),
            tuple(plan.column_meta),
            key_columns,
            attr_columns,
            metrics,
            tuple(dimension_type(ref) for ref in targets),
            item.query,
            bind_members,
        )
        stages.append(stage)
        metadata = {meta.name: meta for meta in plan.column_meta}
        requested_columns = {
            operand.column
            for comparison in query.comparisons
            for operand in (comparison.left, comparison.right)
            if operand.measurement == item.name
        }
        unit_signatures = {
            name: measure_unit_signature(ref)
            for ref in item.query.measures
            if (name := output_column(ref, plan, role="measurement")) in requested_columns
        }
        unit_signatures.update(
            {
                derived.name: derived_unit_signature(derived)
                for derived in item.query.derived_measures
                if derived.name in requested_columns
            }
        )
        stage_unit_signatures[item.name] = unit_signatures
        columns.extend(
            CohortColumn(
                f"{item.name}.{name}",
                "dimension" if name in attr_columns else "measure",
                metadata[name].storage_type,
                name,
                metadata[name],
            )
            for name in (*attr_columns, *metrics)
        )
    stage_by_name = {stage.name: stage for stage in stages}
    for comparison in query.comparisons:
        for operand in (comparison.left, comparison.right):
            stage = stage_by_name[operand.measurement]
            if operand.column not in stage.metric_columns:
                raise FederationError(
                    "Comparison must reference a numeric measurement output.",
                    reason="cohort_comparison_column",
                )
            meta = stage.column_meta[stage.columns.index(operand.column)]
            if meta.storage_type not in {"integer", "float", "number"} and not (
                meta.storage_type is None
                and operand.column in stage_unit_signatures[operand.measurement]
            ):
                raise FederationError(
                    "Comparison must reference a numeric measurement output.",
                    reason="cohort_comparison_non_numeric",
                )
        left = stage_by_name[comparison.left.measurement]
        right = stage_by_name[comparison.right.measurement]
        left_meta = left.column_meta[left.columns.index(comparison.left.column)]
        right_meta = right.column_meta[right.columns.index(comparison.right.column)]
        left_signature = stage_unit_signatures[left.name].get(comparison.left.column)
        right_signature = stage_unit_signatures[right.name].get(comparison.right.column)
        if left_signature is None or right_signature is None:
            raise FederationError(
                "Comparison unit semantics are not established.",
                reason="cohort_comparison_unit_unknown",
            )
        if left_signature[0] != right_signature[0]:
            raise FederationError(
                "Comparison measurement units are incompatible.",
                reason="cohort_comparison_unit_mismatch",
            )
        if (
            left_meta.storage_type == "float"
            and right_meta.storage_type == "number"
            or right_meta.storage_type == "float"
            and left_meta.storage_type == "number"
        ):
            raise FederationError(
                "Comparison numeric precision classes are incompatible.",
                reason="cohort_comparison_type_mismatch",
            )
        result_type: StorageType
        if comparison.op == "difference":
            result_type = (
                "float"
                if "float" in {left_meta.storage_type, right_meta.storage_type}
                else (
                    "integer"
                    if left_meta.storage_type == right_meta.storage_type == "integer"
                    else "number"
                )
            )
        else:
            result_type = (
                "number"
                if "number" in {left_meta.storage_type, right_meta.storage_type}
                else "float"
            )
        result_unit = (
            left_signature[1]
            if comparison.op == "difference" and left_signature[0][0] == "unit"
            else None
        )
        result_format = (
            "percent"
            if comparison.op == "percent_change"
            else left_meta.format
            if comparison.op == "difference" and left_meta.format == right_meta.format
            else None
        )
        result_meta = ColumnMeta(
            name=comparison.name,
            kind="computed",
            display_name=comparison.name,
            unit=result_unit,
            display_unit=(
                left_meta.display_unit
                if comparison.op == "difference"
                and result_unit is not None
                and left_meta.display_unit == right_meta.display_unit
                else None
            ),
            format=result_format,
            storage_type=result_type,
            masked=left_meta.masked or right_meta.masked,
            additive=False,
        )
        columns.append(
            CohortColumn(
                f"comparison.{comparison.name}",
                "computed",
                result_type,
                comparison.name,
                result_meta,
            )
        )
    if len({column.name for column in columns}) != len(columns):
        raise FederationError("Cohort output names collide.", reason="cohort_output_collision")
    return CohortPlan(query, source, tuple(stages), tuple(columns), catalog_context)


__all__ = [
    "COHORT_PLAN_VERSION",
    "Cohort",
    "CohortColumn",
    "CohortComparison",
    "CohortMetric",
    "CohortMeasurement",
    "CohortPlan",
    "CohortQuery",
    "compile_cohort_query",
]
