"""Artifact-local semantic descriptors and structured comparison.

The graph records logical meaning only; physical SQL, parameter values, and
policy predicates are intentionally absent. IDs are local references preserved
by serialization, not stable identifiers across compilations.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from semql._resolve import _ResolvedFields
from semql.logical import LogicalPlan
from semql.model import BaseField, Cube, Dimension, Measure, TimeDimension
from semql.spec import BoolExpr, Filter, SemanticQuery


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class CatalogContext(_FrozenModel):
    namespace: str | None = None
    semantic_revision: str | None = None


class SemanticNode(_FrozenModel):
    node_id: str
    kind: str
    catalog_ref: str | None = None
    aggregate: str | None = None
    logical_stage: str | None = None
    operand_ids: tuple[str, ...] = ()
    population_ref: str | None = None
    period_role: str | None = None
    unit: str | None = None
    unit_state: Literal["declared", "unproven"] = "unproven"
    quantile_method: str | None = None
    null_policy: str | None = None
    zero_denominator: str | None = None
    applicability: str | None = None
    numeric_scale: float | None = None
    binding_refs: tuple[str, ...] = ()
    masked: bool = False
    mask_roles: tuple[str, ...] = ()
    source_ref: str | None = None
    grain_refs: tuple[str, ...] = ()
    time_granularity: str | None = None
    time_fill_value: int | None = None
    time_bounds: str | None = None


class SemanticOutput(_FrozenModel):
    output_id: str
    node_id: str
    sql_alias: str


class EnrichmentProvenance(_FrozenModel):
    output_id: str
    sql_alias: str
    lookup_ref: str
    field: str


class SemanticPopulation(_FrozenModel):
    population_id: str
    inclusion: str
    predicate_ids: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()
    policy_refs: tuple[str, ...] = ()
    evaluated_scopes: tuple[str, ...] = ()


class SemanticJoin(_FrozenModel):
    source_ref: str
    target_ref: str
    kind: str
    relationship: str


class SemanticAssumption(_FrozenModel):
    reference: str
    kind: str
    declared_value: str
    verification: str = "catalog_trust"


class SemanticDerivation(_FrozenModel):
    operation: str
    recipe: str
    logical_node_ids: tuple[str, ...] = ()
    physical_refs: tuple[str, ...] = ()


class SemanticBinding(_FrozenModel):
    slot: str
    kind: str
    reference: str | None = None
    operator: str | None = None
    stage: str | None = None
    arity: int | None = None


class SemanticTimePolicy(_FrozenModel):
    alignment: str | None = None
    timezone: str | None = None
    week_start: str | None = None
    granularity: str | None = None
    fill_value: int | None = None
    current_binding_ref: str | None = None
    prior_binding_ref: str | None = None
    null_policy: str | None = None


class SemanticResult(_FrozenModel):
    grain_node_ids: tuple[str, ...] = ()
    population_ref: str | None = None
    order: tuple[tuple[str, str], ...] = ()
    limit: int | None = None
    offset: int | None = None
    time_policy: SemanticTimePolicy | None = None
    completeness: str = "not_established"
    grain_kind: Literal["rows", "grouped", "scalar"] = "scalar"
    predicate_ids: tuple[str, ...] = ()


class Coverage(_FrozenModel):
    status: Literal["complete", "partial", "not_established", "unavailable"] = "complete"
    established: tuple[str, ...] = ()
    not_established: tuple[str, ...] = ()


class SemanticAnalysis(_FrozenModel):
    schema_version: int = 1
    nodes: tuple[SemanticNode, ...] = ()
    outputs: tuple[SemanticOutput, ...] = ()
    populations: tuple[SemanticPopulation, ...] = ()
    catalog_context: CatalogContext | None = None
    coverage: Coverage = Field(default_factory=Coverage)
    enrichments: tuple[EnrichmentProvenance, ...] = ()
    binding_dependencies: tuple[SemanticBinding, ...] = ()
    result: SemanticResult | None = None
    required_semantics: tuple[str, ...] = ()
    joins: tuple[SemanticJoin, ...] = ()
    assumptions: tuple[SemanticAssumption, ...] = ()
    derivations: tuple[SemanticDerivation, ...] = ()

    @classmethod
    def unavailable(cls) -> SemanticAnalysis:
        return cls(
            coverage=Coverage(status="unavailable", not_established=("analysis_unavailable",))
        )


class ComparisonResult(StrEnum):
    EQUIVALENT = "equivalent"
    DIFFERENT = "different"
    NOT_ESTABLISHED = "not_established"


class ComparisonScope(StrEnum):
    EXPRESSION = "expression"
    RESULT = "result"


class Comparison(_FrozenModel):
    outcome: ComparisonResult
    scope: ComparisonScope
    reasons: tuple[str, ...] = ()


def _quantile_method(dialect: str) -> str:
    return {
        "postgres": "percentile_cont_exact",
        "duckdb": "percentile_cont_exact",
        "snowflake": "percentile_cont_exact",
        "redshift": "percentile_cont",
        "bigquery": "approx_quantiles",
        "clickhouse": "quantile",
        "trino": "approx_percentile",
        "databricks": "percentile_approx",
    }.get(dialect, "unverified_backend_quantile")


def build_analysis(
    query: SemanticQuery,
    plan: LogicalPlan,
    resolved_fields: _ResolvedFields,
    catalog_context: CatalogContext | None = None,
    *,
    viewer_roles: frozenset[str] = frozenset(),
    effective_scope_refs: tuple[str, ...] = (),
    evaluated_scopes: tuple[str, ...] = (),
    scope_coverage: Literal["complete", "not_established"] = "complete",
    population_inclusion: Literal["observed_facts", "query_rows"] | None = None,
    assumptions: tuple[SemanticAssumption, ...] = (),
    derivations: tuple[SemanticDerivation, ...] = (),
    applied_rollup: str | None = None,
    physical_sources_hit: tuple[str, ...] = (),
) -> SemanticAnalysis:
    """Project resolved logical origins to an artifact-local semantic graph."""
    next_node = 1
    bindings: list[SemanticBinding] = []

    def allocate_node() -> str:
        nonlocal next_node
        node_id = f"n{next_node}"
        next_node += 1
        return node_id

    nodes: list[SemanticNode] = []
    outputs: list[SemanticOutput] = []
    ids: dict[str, str] = {}
    fields: dict[str, tuple[Cube, BaseField]] = {}

    def register_field(ref: str, pair: tuple[Cube, BaseField]) -> None:
        fields[ref] = pair
        cube, field = pair
        fields[f"{cube.name}.{field.name}"] = pair

    pair: tuple[Cube, BaseField]
    field: BaseField
    for ref, pair in zip(query.measures, resolved_fields.measure_fields, strict=True):
        register_field(ref, pair)
    for ref, pair in zip(query.dimensions, resolved_fields.dim_fields, strict=True):
        register_field(ref, pair)
    derived_refs = [operand for derived in query.derived_measures for operand in derived.operands]
    for ref, pair in zip(derived_refs, resolved_fields.derived_operand_fields, strict=True):
        register_field(ref, pair)
    for predicate, cube, field in resolved_fields.filter_resolutions:
        register_field(predicate.dimension, (cube, field))
    for ref, pair in resolved_fields.where_leaf_resolutions.items():
        register_field(ref, pair)
    for cube in resolved_fields.touched:
        for field in (*cube.measures, *cube.dimensions, *cube.time_dimensions):
            register_field(f"{cube.name}.{field.name}", (cube, field))
    if (
        query.time_dimension is not None
        and resolved_fields.time_cube is not None
        and resolved_fields.time_dim is not None
    ):
        time_pair = (resolved_fields.time_cube, resolved_fields.time_dim)
        register_field(query.time_dimension.dimension, time_pair)

    def add_field(ref: str, stage: str) -> str:
        if ref in ids:
            return ids[ref]
        pair = fields.get(ref)
        if pair is None:
            raise ValueError(f"Semantic graph has no resolved field for {ref!r}.")
        cube, field = pair
        canonical_ref = f"{cube.name}.{field.name}"
        if canonical_ref in ids:
            ids[ref] = ids[canonical_ref]
            return ids[canonical_ref]
        node_id = allocate_node()
        ids[canonical_ref] = node_id
        ids[ref] = node_id
        operands: tuple[str, ...] = ()
        if isinstance(field, Measure) and field.agg == "ratio":
            refs = tuple(
                f"{cube.name}.{name}"
                for name in (field.numerator, field.denominator)
                if name is not None
            )
            operands = tuple(add_field(child, "event_to_group") for child in refs)
        roles = tuple(field.mask_roles) if isinstance(field, (Measure, Dimension)) else ()
        is_ratio = isinstance(field, Measure) and field.agg == "ratio"
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="aggregate"
                if isinstance(field, Measure)
                else ("time" if isinstance(field, TimeDimension) else "dimension"),
                catalog_ref=canonical_ref,
                aggregate=field.agg if isinstance(field, Measure) else None,
                logical_stage=(
                    "event_to_group"
                    if isinstance(field, Measure)
                    else "row"
                    if query.ungrouped
                    else "time_bucket"
                    if isinstance(field, TimeDimension)
                    and query.time_dimension
                    and query.time_dimension.granularity
                    else "group_by"
                ),
                operand_ids=operands,
                population_ref="p1",
                unit=getattr(field, "unit", None),
                unit_state="declared" if getattr(field, "unit", None) is not None else "unproven",
                quantile_method=(
                    _quantile_method(cube.dialect.value)
                    if isinstance(field, Measure) and field.agg in {"median", "p75", "p90", "p95"}
                    else None
                ),
                null_policy=(
                    "null_propagates"
                    if is_ratio
                    else "backend_defined"
                    if isinstance(field, Measure) and field.agg in {"median", "p75", "p90", "p95"}
                    else "counts_rows"
                    if isinstance(field, Measure) and field.agg == "count" and field.sql == "*"
                    else "ignore_null_inputs"
                    if isinstance(field, Measure)
                    else None
                ),
                zero_denominator="null" if is_ratio else None,
                masked=bool(set(roles) & set(viewer_roles)),
                mask_roles=roles,
                grain_refs=tuple(f"{c.name}.{f.name}" for c, f in resolved_fields.dim_fields),
                time_granularity=(
                    query.time_dimension.granularity
                    if isinstance(field, TimeDimension) and query.time_dimension is not None
                    else None
                ),
                time_fill_value=(
                    query.time_dimension.fill_nulls_with
                    if isinstance(field, TimeDimension) and query.time_dimension is not None
                    else None
                ),
                time_bounds=(
                    "half_open"
                    if isinstance(field, TimeDimension) and query.time_dimension is not None
                    else None
                ),
            )
        )
        return node_id

    for cube, field in resolved_fields.dim_fields:
        add_field(f"{cube.name}.{field.name}", "group")
    for cube, field in resolved_fields.measure_fields:
        add_field(f"{cube.name}.{field.name}", "aggregate")
    for cube, field in resolved_fields.derived_operand_fields:
        add_field(f"{cube.name}.{field.name}", "hidden_operand")
    if resolved_fields.time_cube is not None and resolved_fields.time_dim is not None:
        add_field(
            f"{resolved_fields.time_cube.name}.{resolved_fields.time_dim.name}",
            "time_bucket" if query.time_dimension and query.time_dimension.granularity else "time",
        )
    predicate_ids: list[str] = []

    def add_predicate_field(cube: Cube, field: BaseField) -> str:
        # Predicates read source rows, not the filtered/grouped population
        # they define. Keeping that input separate makes the graph acyclic.
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="time" if isinstance(field, TimeDimension) else "dimension",
                catalog_ref=f"{cube.name}.{field.name}",
                logical_stage="row",
                source_ref=cube.name,
            )
        )
        return node_id

    def add_filter(predicate: Filter, stage: str = "where") -> str:
        ref = predicate.dimension
        pair: tuple[Cube, BaseField] | None = resolved_fields.where_leaf_resolutions.get(ref)
        if pair is None:
            match = next(
                (
                    entry
                    for entry in resolved_fields.filter_resolutions
                    if entry[0].dimension == ref
                ),
                None,
            )
            pair = (match[1], match[2]) if match is not None else None
        if pair is None:
            pair = fields.get(ref)
        if pair is None:
            raise ValueError(f"Semantic graph has no resolved predicate field for {ref!r}.")
        cube, field = pair
        canonical_ref = f"{cube.name}.{field.name}"
        field_id = add_predicate_field(cube, field)
        return add_target_filter(
            predicate,
            stage=stage,
            target_id=field_id,
            reference=canonical_ref,
            source_ref=cube.name,
        )

    def add_target_filter(
        predicate: Filter,
        *,
        stage: str,
        target_id: str,
        reference: str | None,
        source_ref: str | None,
    ) -> str:
        node_id = allocate_node()
        arity = (
            0
            if predicate.op in {"is_null", "not_null"}
            else len(predicate.values)
            if predicate.op in {"in", "not_in"}
            else 1
        )
        slot = f"b{len(bindings) + 1}"
        if arity:
            bindings.append(
                SemanticBinding(
                    slot=slot,
                    kind="filter_value",
                    reference=reference,
                    operator=predicate.op,
                    stage=stage,
                    arity=arity,
                )
            )
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="predicate",
                catalog_ref=reference,
                aggregate=predicate.op,
                logical_stage=stage,
                operand_ids=(target_id,),
                binding_refs=(slot,) if arity else (),
                source_ref=source_ref,
            )
        )
        return node_id

    def add_bool(node: BoolExpr | Filter, stage: str = "where") -> str:
        if isinstance(node, Filter):
            return add_filter(node, stage)
        children = tuple(add_bool(child, stage) for child in node.children)
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="predicate",
                aggregate=node.op,
                logical_stage=stage,
                operand_ids=children,
            )
        )
        return node_id

    filter_roots = [add_filter(predicate) for predicate in query.filters]
    if query.where is not None:
        filter_roots.append(add_bool(query.where))
    if len(filter_roots) == 1:
        predicate_ids.append(filter_roots[0])
    elif filter_roots:
        root_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=root_id,
                kind="predicate",
                aggregate="and",
                logical_stage="where",
                operand_ids=tuple(filter_roots),
            )
        )
        predicate_ids.append(root_id)
    for segment_ref in query.segments:
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="predicate",
                catalog_ref=segment_ref,
                aggregate="segment",
                logical_stage="where",
            )
        )
        predicate_ids.append(node_id)
    time_range_binding = None
    if query.time_dimension is not None:
        ref = query.time_dimension.dimension
        field_id = add_predicate_field(*fields[ref])
        time_range_binding = f"b{len(bindings) + 1}"
        bindings.append(
            SemanticBinding(
                slot=time_range_binding,
                kind="time_range",
                reference=ref,
                operator="current_range" if query.compare is not None else "half_open",
                stage="where",
                arity=2,
            )
        )
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="predicate",
                catalog_ref=ref,
                aggregate="half_open_range",
                logical_stage="where",
                operand_ids=(field_id,),
                binding_refs=(time_range_binding,),
            )
        )
        predicate_ids.append(node_id)
    policy_refs: set[str] = set(effective_scope_refs)
    for cube in resolved_fields.touched:
        if cube.base_predicate:
            policy_refs.add(f"catalog:{cube.name}:base_predicate")
        if cube.security_sql:
            policy_refs.add(f"catalog:{cube.name}:security_sql")
        if cube.tenancy != "none":
            policy_refs.add(f"catalog:{cube.name}:tenancy")
    # Catalog revision identifies policy code, not its request-scoped inputs.
    # Opaque private context slots let the host attest those inputs without
    # publishing values or evaluating scope functions a second time.
    context_refs = policy_refs | {f"scope:{name}" for name in evaluated_scopes}
    for reference in sorted(context_refs):
        slot = f"b{len(bindings) + 1}"
        bindings.append(
            SemanticBinding(slot=slot, kind="policy_context", reference=reference, stage="where")
        )
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="predicate",
                aggregate="scope",
                catalog_ref=reference,
                logical_stage="where",
                binding_refs=(slot,),
            )
        )
        predicate_ids.append(node_id)
    joins = tuple(
        SemanticJoin(
            source_ref=join.left.name,
            target_ref=join.right.name,
            kind=join.kind,
            relationship=join.model.relationship,
        )
        for join in plan.joins
    )
    declared_assumptions = list(assumptions)
    declared_assumptions.extend(
        SemanticAssumption(
            reference=f"join:{join.source_ref}->{join.target_ref}",
            kind="relationship",
            declared_value=join.relationship,
        )
        for join in joins
    )
    population = SemanticPopulation(
        population_id="p1",
        inclusion=population_inclusion
        or ("observed_facts" if plan.symmetric is not None else "query_rows"),
        predicate_ids=tuple(predicate_ids),
        source_refs=tuple(cube.name for cube in resolved_fields.touched),
        policy_refs=tuple(sorted(policy_refs)),
        evaluated_scopes=tuple(sorted(evaluated_scopes)),
    )

    aliases = dict(query.aliases)
    seen: set[str] = set()
    having_targets: dict[str, str] = {}
    measure_names_by_node: dict[str, set[str]] = {}

    def add_output(ref: str, alias: str, node_id: str, identity: str) -> None:
        from semql.errors import CompileError

        if identity in seen:
            raise CompileError(f"Duplicate semantic projection for {ref!r}.")
        seen.add(identity)
        alias = next((key for key, target in aliases.items() if target == ref), alias)
        outputs.append(
            SemanticOutput(
                output_id=f"o{len(outputs) + 1}",
                node_id=node_id,
                sql_alias=alias,
            )
        )

    for col in plan.project.columns:
        ref = f"{col.cube.name}.{col.field_name}"
        node_id = add_field(ref, "output")
        add_output(ref, col.alias, node_id, node_id)
        if col.kind == "measure":
            names = measure_names_by_node.setdefault(node_id, set())
            names.update((ref, col.field_name, col.alias))
            having_targets.update({name: node_id for name in names})
    for requested_ref, (cube, field) in zip(
        query.measures, resolved_fields.measure_fields, strict=True
    ):
        node_id = add_field(requested_ref, "output")
        names = measure_names_by_node.setdefault(node_id, set())
        names.update((requested_ref, field.name, f"{cube.name}.{field.name}"))
        having_targets.update({name: node_id for name in names})
    for alias, target in aliases.items():
        target_id = having_targets.get(target)
        if target_id is not None:
            having_targets[alias] = target_id
            measure_names_by_node.setdefault(target_id, set()).add(alias)
    # Inline expressions are appended by emission, not Project.
    for inline in query.derived_measures:
        operand_ids = tuple(add_field(ref, "hidden_operand") for ref in inline.operands)
        node_id = allocate_node()
        nodes.append(
            SemanticNode(
                node_id=node_id,
                kind="arithmetic",
                aggregate=inline.op,
                logical_stage="post_aggregate",
                operand_ids=operand_ids,
                population_ref="p1",
                null_policy="null_propagates",
                zero_denominator="null" if inline.op == "ratio" else None,
            )
        )
        add_output(
            inline.name,
            inline.name,
            node_id,
            f"inline:{inline.op}:{','.join(inline.operands)}",
        )
        having_targets[inline.name] = node_id
    if plan.compare is not None:
        base = tuple(outputs)
        outputs.clear()
        by_id = {node.node_id: node for node in nodes}
        for output in base:
            node = by_id[output.node_id]
            if node.kind != "aggregate":
                outputs.append(output)
                continue
            period_nodes: dict[str, str] = {}
            for role in ("current", "prior"):
                node_id = allocate_node()
                nodes.append(
                    SemanticNode(
                        node_id=node_id,
                        kind="comparison",
                        aggregate=role,
                        logical_stage="compare",
                        operand_ids=(node.node_id,),
                        population_ref="p1",
                        period_role=role,
                        null_policy="preserve_null",
                    )
                )
                period_nodes[role] = node_id
                add_output(
                    output.sql_alias,
                    f"{output.sql_alias}_{role}",
                    node_id,
                    f"{output.node_id}:{role}",
                )
                for name in measure_names_by_node.get(output.node_id, {output.sql_alias}):
                    having_targets[f"{name}_{role}"] = node_id
                    having_targets[f"compare.{name}.{role}"] = node_id
            for role in ("delta", "pct_change"):
                node_id = allocate_node()
                nodes.append(
                    SemanticNode(
                        node_id=node_id,
                        kind="comparison",
                        aggregate=role,
                        logical_stage="compare",
                        operand_ids=(period_nodes["current"], period_nodes["prior"]),
                        population_ref="p1",
                        period_role=role,
                        null_policy="coalesce_zero" if role == "delta" else "preserve_null",
                        applicability="always" if role == "delta" else "prior_gt_zero",
                        numeric_scale=100.0 if role == "pct_change" else None,
                    )
                )
                add_output(
                    output.sql_alias,
                    f"{output.sql_alias}_{role}",
                    node_id,
                    f"{output.node_id}:{role}",
                )
                for name in measure_names_by_node.get(output.node_id, {output.sql_alias}):
                    having_targets[f"{name}_{role}"] = node_id
                    having_targets[f"compare.{name}.{role}"] = node_id

    having_predicate_ids: list[str] = []
    nodes_by_id = {node.node_id: node for node in nodes}
    for predicate in query.having:
        target_id = having_targets.get(predicate.dimension)
        if target_id is None:
            raise ValueError(
                f"Semantic graph has no resolved predicate field for {predicate.dimension!r}."
            )
        target_node = nodes_by_id[target_id]
        having_predicate_ids.append(
            add_target_filter(
                predicate,
                stage="having",
                target_id=target_id,
                reference=target_node.catalog_ref,
                source_ref=target_node.source_ref,
            )
        )

    grain_ids: list[str] = [
        add_field(f"{cube.name}.{field.name}", "group")
        for cube, field in resolved_fields.dim_fields
    ]
    if (
        query.time_dimension is not None
        and query.time_dimension.granularity is not None
        and resolved_fields.time_cube is not None
        and resolved_fields.time_dim is not None
    ):
        grain_ids.append(
            add_field(
                f"{resolved_fields.time_cube.name}.{resolved_fields.time_dim.name}",
                "time_bucket",
            )
        )
    alias_to_node = {output.sql_alias: output.node_id for output in outputs}
    ref_to_node = {
        node.catalog_ref: node.node_id
        for node in nodes
        if node.catalog_ref is not None and node.kind in {"aggregate", "dimension", "time"}
    }
    order: list[tuple[str, str]] = []
    unresolved_order = False
    for name, direction in query.order:
        ref = query.aliases.get(name, name)
        order_node_id = alias_to_node.get(name) or ref_to_node.get(ref)
        if order_node_id is None:
            unresolved_order = True
            break
        order.append((order_node_id, direction))
    prior_binding_ref = None
    time_policy = None
    if query.time_dimension is not None:
        if query.compare is not None:
            prior_binding_ref = f"b{len(bindings) + 1}"
            bindings.append(
                SemanticBinding(
                    slot=prior_binding_ref,
                    kind="time_window",
                    reference=query.time_dimension.dimension,
                    operator=query.compare.mode,
                    stage="compare",
                    arity=2,
                )
            )
        time_cube = resolved_fields.time_cube
        time_dim = resolved_fields.time_dim
        granularity = query.time_dimension.granularity
        time_policy = SemanticTimePolicy(
            alignment=query.compare.mode if query.compare is not None else "half_open",
            timezone=(
                time_cube.timezone
                if time_cube is not None and (time_dim is None or time_dim.type != "date")
                else None
            ),
            week_start=(
                time_cube.week_start if time_cube is not None and granularity == "week" else None
            ),
            granularity=granularity,
            fill_value=query.time_dimension.fill_nulls_with,
            current_binding_ref=time_range_binding,
            prior_binding_ref=prior_binding_ref,
            null_policy=(
                "delta_coalesce_zero_pct_prior_gt_zero"
                if query.compare is not None
                else "fill_missing"
                if query.time_dimension.fill_nulls_with is not None
                else "preserve_null"
            ),
        )
    unsupported_semijoins = bool(query.semi_joins)
    unsupported_quantiles = any(
        node.quantile_method == "unverified_backend_quantile" for node in nodes
    )
    result_complete = (
        bool(outputs)
        and not unresolved_order
        and not unsupported_semijoins
        and not unsupported_quantiles
    )
    grain_kind: Literal["rows", "grouped", "scalar"] = (
        "rows" if query.ungrouped else "grouped" if grain_ids else "scalar"
    )
    incomplete: list[str] = []
    if not outputs:
        incomplete.append("no_projected_outputs")
    if scope_coverage != "complete":
        incomplete.append("effective_scope_not_established")
    if unresolved_order:
        incomplete.append("order_binding_not_established")
    if unsupported_semijoins:
        incomplete.append("semi_join_semantics_not_established")
    if unsupported_quantiles:
        incomplete.append("quantile_method_not_established")
    complete = result_complete and scope_coverage == "complete"
    result = SemanticResult(
        grain_node_ids=tuple(grain_ids),
        population_ref="p1",
        predicate_ids=tuple(having_predicate_ids),
        order=tuple(order),
        limit=query.limit,
        offset=query.offset,
        time_policy=time_policy,
        grain_kind=grain_kind,
        completeness="not_established",
    )
    coverage = Coverage(
        status="complete" if complete else "not_established",
        established=(
            "requested_outputs",
            "logical_lineage",
            "effective_population",
            "result_grain",
            "ordering_pagination",
        )
        if complete
        else (),
        not_established=tuple(incomplete),
    )
    physical_derivations = list(derivations)
    if applied_rollup is not None:
        measure_nodes = tuple(
            node.node_id
            for node in nodes
            if node.kind == "aggregate" and node.aggregate is not None
        )
        count_nodes = tuple(
            node.node_id for node in nodes if node.kind == "aggregate" and node.aggregate == "count"
        )
        if count_nodes:
            physical_derivations.append(
                SemanticDerivation(
                    operation="count",
                    recipe="sum_stored_partial_counts",
                    logical_node_ids=count_nodes,
                    physical_refs=(applied_rollup, *physical_sources_hit),
                )
            )
        other_nodes = tuple(node_id for node_id in measure_nodes if node_id not in count_nodes)
        if other_nodes:
            physical_derivations.append(
                SemanticDerivation(
                    operation="rollup",
                    recipe="stored_measure_reaggregation",
                    logical_node_ids=other_nodes,
                    physical_refs=(applied_rollup, *physical_sources_hit),
                )
            )
    return SemanticAnalysis(
        nodes=tuple(nodes),
        outputs=tuple(outputs),
        populations=(population,),
        catalog_context=catalog_context,
        coverage=coverage,
        binding_dependencies=tuple(bindings),
        result=result,
        joins=joins,
        assumptions=tuple(declared_assumptions),
        derivations=tuple(physical_derivations),
    )


def compare_analysis(
    left: SemanticAnalysis,
    right: SemanticAnalysis,
    *,
    scope: ComparisonScope | Literal["expression", "result"] = ComparisonScope.EXPRESSION,
    left_context: Mapping[str, object] | None = None,
    right_context: Mapping[str, object] | None = None,
    shared_catalog_snapshot: bool = False,
    bound_context_equal: bool | None = None,
) -> Comparison:
    """Compare semantic descriptors through the strict artifact comparator."""
    from semql._compare import compare_analysis as compare

    return compare(
        left,
        right,
        scope=scope,
        left_context=left_context,
        right_context=right_context,
        shared_catalog_snapshot=shared_catalog_snapshot,
        bound_context_equal=bound_context_equal,
    )


__all__ = [
    "CatalogContext",
    "SemanticNode",
    "SemanticOutput",
    "EnrichmentProvenance",
    "SemanticPopulation",
    "SemanticBinding",
    "SemanticTimePolicy",
    "SemanticResult",
    "Coverage",
    "SemanticAnalysis",
    "SemanticJoin",
    "SemanticAssumption",
    "SemanticDerivation",
    "ComparisonResult",
    "ComparisonScope",
    "Comparison",
    "build_analysis",
    "compare_analysis",
]
