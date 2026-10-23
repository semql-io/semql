"""Execution and alignment for bounded, single-hop cohort plans."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, replace
from decimal import Context, Decimal, localcontext
from math import isfinite
from typing import Any

from pydantic import ValidationError
from semql.analysis import CatalogContext, SemanticAnalysis
from semql.bindings import validate_bindings
from semql.cohort import COHORT_PLAN_VERSION, CohortPlan, CohortQuery, CohortStage
from semql.compile import COMPILED_QUERY_VERSION, ColumnMeta
from semql.errors import ContractError
from semql.federate import FEDERATED_PLAN_VERSION
from semql.spec import SemanticQuery

from semql_engine.engine import Engine, ExecutionContractError, ExecutionResult


@dataclass(frozen=True)
class CohortStageProvenance:
    stage: str
    cohort_query: CohortQuery
    query: SemanticQuery
    analysis: SemanticAnalysis
    catalog_context: CatalogContext | None
    rows: int
    identity_validated: bool


@dataclass(frozen=True)
class CohortCoverage:
    measurement: str
    member_count: int
    matched_members: int


@dataclass(frozen=True)
class CohortResult:
    columns: tuple[str, ...]
    column_meta: tuple[ColumnMeta, ...]
    rows: tuple[tuple[Any, ...], ...]
    presence: dict[str, tuple[bool, ...]]
    coverage: tuple[CohortCoverage, ...]
    provenance: tuple[CohortStageProvenance, ...]


def _fail(reason: str, stage: str) -> ExecutionContractError:
    return ExecutionContractError(
        "Cohort stage output violates its declared contract.",
        reason=reason,
        artifact="cohort_plan",
        operation="execute",
        stage=stage,
    )


def _shape(result: ExecutionResult, stage: CohortStage) -> None:
    if tuple(result.columns) != stage.columns or len(stage.column_meta) != len(stage.columns):
        raise _fail("cohort_columns_mismatch", "output_validation")
    if any(len(row) != len(stage.columns) for row in result.rows):
        raise _fail("cohort_row_width_mismatch", "output_validation")


def _typed_key(values: tuple[Any, ...], types: tuple[str, ...]) -> tuple[Any, ...]:
    normalized: list[Any] = []
    for value, logical_type in zip(values, types, strict=True):
        if value is None:
            raise _fail("cohort_null_member_key", "materialized_validation")
        if logical_type in {"integer", "float", "number"}:
            if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation")
            if isinstance(value, float) and not isfinite(value):
                raise _fail("cohort_nonfinite_member_key", "materialized_validation")
            if isinstance(value, Decimal) and not value.is_finite():
                raise _fail("cohort_nonfinite_member_key", "materialized_validation")
            normalized.append(("number", Decimal(str(value))))
        elif logical_type == "uuid":
            if not isinstance(value, (str, uuid.UUID)):
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation")
            try:
                normalized.append(("uuid", uuid.UUID(value) if isinstance(value, str) else value))
            except ValueError:
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation") from None
        elif logical_type == "date":
            try:
                date = datetime.date.fromisoformat(value) if isinstance(value, str) else value
                if not isinstance(date, datetime.date) or isinstance(date, datetime.datetime):
                    raise ValueError("not a calendar date")
                normalized.append(("date", date))
            except ValueError:
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation") from None
        elif logical_type == "time":
            try:
                instant = (
                    datetime.datetime.fromisoformat(value) if isinstance(value, str) else value
                )
                if not isinstance(instant, datetime.datetime):
                    raise ValueError("not a timestamp")
                if instant.utcoffset() is not None:
                    instant = instant.astimezone(datetime.UTC)
                normalized.append(("time", instant))
            except ValueError:
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation") from None
        elif logical_type == "bool":
            if not isinstance(value, bool):
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation")
            normalized.append(("bool", value))
        elif logical_type == "string":
            if not isinstance(value, str):
                raise _fail("cohort_member_key_type_mismatch", "materialized_validation")
            normalized.append(("string", value))
        else:
            raise _fail("cohort_member_key_type_mismatch", "materialized_validation")
    return tuple(normalized)


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _compare(op: str, left: object, right: object) -> int | float | Decimal | None:
    if left is None or right is None:
        return None
    if not _numeric(left) or not _numeric(right):
        raise _fail("cohort_comparison_non_numeric", "comparison")
    if isinstance(left, Decimal) or isinstance(right, Decimal):
        if isinstance(left, float) or isinstance(right, float):
            raise _fail("cohort_comparison_numeric_incompatible", "comparison")
        if not isinstance(left, (int, Decimal)) or not isinstance(right, (int, Decimal)):
            raise _fail("cohort_comparison_non_numeric", "comparison")
        decimal_left, decimal_right = Decimal(left), Decimal(right)
        if not decimal_left.is_finite() or not decimal_right.is_finite():
            raise _fail("cohort_comparison_nonfinite", "comparison")
        scale = max(
            0, -int(decimal_left.as_tuple().exponent), -int(decimal_right.as_tuple().exponent)
        )
        precision = max(38, max(decimal_left.adjusted(), decimal_right.adjusted(), 0) + scale + 3)
        # Preserve full finite driver precision for subtraction. Division uses
        # at least 38 significant digits, independent of the host Decimal context.
        with localcontext(Context(prec=precision)):
            if op == "difference":
                return decimal_left - decimal_right
            if decimal_right == 0:
                return None
            if op == "ratio":
                return decimal_left / decimal_right
            return (decimal_left - decimal_right) / decimal_right * 100
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise _fail("cohort_comparison_non_numeric", "comparison")
    if op == "difference":
        return left - right
    if right == 0:
        return None
    if op == "ratio":
        return left / right
    return (left - right) / right * 100


def run_cohort(
    plan: CohortPlan,
    engine: Engine,
    *,
    cache_namespace: str | None = None,
) -> CohortResult:
    """Run selection, validate it, then restrict and align each measurement."""
    if plan.version != COHORT_PLAN_VERSION:
        raise _fail("artifact_version_unsupported", "preflight")
    if not plan.measurements or len(plan.measurements) != len(plan.cohort.measurements):
        raise _fail("cohort_plan_invalid", "preflight")
    if any(stage.bind_members is None for stage in plan.measurements):
        raise _fail("cohort_binding_missing", "preflight")
    expected_names: tuple[str, ...] = tuple(column.name for column in plan.columns)
    if len(expected_names) != len(set(expected_names)):
        raise _fail("cohort_plan_invalid", "preflight")
    for stage in (plan.source, *plan.measurements):
        if (
            stage.plan.version != FEDERATED_PLAN_VERSION
            or stage.plan.columns != list(stage.columns)
            or len(stage.columns) != len(stage.column_meta)
            or tuple(stage.plan.column_meta) != stage.column_meta
            or len(stage.key_columns) != len(stage.key_types)
            or not set(stage.key_columns).issubset(stage.columns)
            or not set(stage.attribute_columns).issubset(stage.columns)
            or not set(stage.metric_columns).issubset(stage.columns)
            or not stage.plan.fragments
        ):
            raise _fail("cohort_plan_invalid", "preflight")
        for index, fragment in enumerate(stage.plan.fragments):
            if fragment.version != COMPILED_QUERY_VERSION or fragment.binding_requirements is None:
                raise _fail("artifact_version_unsupported", "preflight")
            try:
                validate_bindings(
                    fragment.sql,
                    fragment.dialect,
                    fragment.params,
                    fragment.binding_requirements,
                    artifact=f"cohort_{stage.name}_{index}",
                )
            except ContractError as error:
                raise ExecutionContractError.from_contract_error(error) from error
    expected_projection = [
        (f"selection.{name}", "dimension", name)
        for name in (*plan.source.key_columns, *plan.source.attribute_columns)
    ]
    for stage in plan.measurements:
        expected_projection.extend(
            (f"{stage.name}.{name}", "dimension", name) for name in stage.attribute_columns
        )
        expected_projection.extend(
            (f"{stage.name}.{name}", "measure", name) for name in stage.metric_columns
        )
    expected_projection.extend(
        (f"comparison.{comparison.name}", "computed", comparison.name)
        for comparison in plan.cohort.comparisons
    )
    if tuple(expected_projection) != tuple(
        (column.name, column.kind, column.source_column) for column in plan.columns
    ):
        raise _fail("cohort_projection_mismatch", "preflight")
    if plan.source.name != "selection" or tuple(stage.name for stage in plan.measurements) != tuple(
        measurement.name for measurement in plan.cohort.measurements
    ):
        raise _fail("cohort_plan_invalid", "preflight")
    output_metadata: list[ColumnMeta] = []
    for column in plan.columns:
        if column.child_meta is None:
            raise _fail("cohort_projection_metadata_missing", "preflight")
        output_metadata.append(replace(column.child_meta, name=column.name))
    source_result = engine.run(plan.source.plan, cache_namespace=cache_namespace)
    _shape(source_result, plan.source)
    max_members = plan.cohort.cohort.max_members
    if len(source_result.rows) > max_members:
        raise _fail("cohort_member_limit_exceeded", "source_validation")

    key_positions = tuple(source_result.columns.index(name) for name in plan.source.key_columns)
    attribute_positions = tuple(
        source_result.columns.index(name) for name in plan.source.attribute_columns
    )
    members: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    member_values: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    member_order: list[tuple[Any, ...]] = []
    for row in source_result.rows:
        values = tuple(row[position] for position in key_positions)
        if any(value is None for value in values):
            raise _fail("cohort_null_member_key", "source_validation")
        key = _typed_key(values, plan.source.key_types)
        if key in members:
            raise _fail("cohort_duplicate_member_key", "source_validation")
        members[key] = tuple(row[position] for position in attribute_positions)
        member_values[key] = values
        member_order.append(key)

    provenance = [
        CohortStageProvenance(
            "selection",
            plan.cohort,
            plan.source.query,
            plan.source.plan.analysis,
            plan.catalog_context,
            len(source_result.rows),
            True,
        )
    ]
    if not member_order:
        return CohortResult(
            expected_names,
            tuple(output_metadata),
            (),
            {stage.name: () for stage in plan.measurements},
            tuple(CohortCoverage(stage.name, 0, 0) for stage in plan.measurements),
            tuple(provenance),
        )

    aligned_by_stage: dict[str, dict[tuple[Any, ...], tuple[Any, ...]]] = {}
    presence: dict[str, tuple[bool, ...]] = {}
    coverage: list[CohortCoverage] = []
    input_keys = tuple(member_values[key] for key in member_order)
    for stage in plan.measurements:
        # Missing binders were rejected in preflight before selection I/O.
        assert stage.bind_members is not None
        try:
            bound_plan = stage.bind_members(input_keys)
        except ValidationError:
            raise _fail("cohort_key_binding_failed", "preflight") from None
        if bound_plan.columns != list(stage.columns):
            raise _fail("cohort_bound_plan_shape", "preflight")
        result = engine.run(bound_plan, cache_namespace=cache_namespace)
        _shape(result, stage)
        if len(result.rows) > len(member_order):
            raise _fail("cohort_measurement_overflow", "measurement_validation")
        key_indices = tuple(result.columns.index(name) for name in stage.key_columns)
        attribute_indices = tuple(result.columns.index(name) for name in stage.attribute_columns)
        metric_indices = tuple(result.columns.index(name) for name in stage.metric_columns)
        aligned: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        for row in result.rows:
            values = tuple(row[index] for index in key_indices)
            if any(value is None for value in values):
                raise _fail("cohort_null_measurement_key", "measurement_validation")
            key = _typed_key(values, stage.key_types)
            if key not in members:
                raise _fail("cohort_unexpected_measurement_key", "measurement_validation")
            if key in aligned:
                raise _fail("cohort_duplicate_measurement_key", "measurement_validation")
            aligned[key] = tuple(row[index] for index in (*attribute_indices, *metric_indices))
        aligned_by_stage[stage.name] = aligned
        presence[stage.name] = tuple(key in aligned for key in member_order)
        coverage.append(CohortCoverage(stage.name, len(member_order), len(aligned)))
        provenance.append(
            CohortStageProvenance(
                stage.name,
                plan.cohort,
                stage.query,
                stage.plan.analysis,
                plan.catalog_context,
                len(result.rows),
                True,
            )
        )

    stage_by_name = {stage.name: stage for stage in plan.measurements}
    comparison_values: dict[str, tuple[Any, ...]] = {}
    for comparison in plan.cohort.comparisons:
        left_stage = stage_by_name[comparison.left.measurement]
        right_stage = stage_by_name[comparison.right.measurement]
        if (
            comparison.left.column not in left_stage.metric_columns
            or comparison.right.column not in right_stage.metric_columns
        ):
            raise _fail("cohort_comparison_column_invalid", "preflight")
        left_index = len(left_stage.attribute_columns) + left_stage.metric_columns.index(
            comparison.left.column
        )
        right_index = len(right_stage.attribute_columns) + right_stage.metric_columns.index(
            comparison.right.column
        )
        compared_values: list[Any] = []
        for key in member_order:
            left = aligned_by_stage[left_stage.name].get(key)
            right = aligned_by_stage[right_stage.name].get(key)
            compared_values.append(
                _compare(
                    comparison.op,
                    left[left_index] if left is not None else None,
                    right[right_index] if right is not None else None,
                )
            )
        comparison_values[comparison.name] = tuple(compared_values)

    output: list[tuple[Any, ...]] = []
    for index, key in enumerate(member_order):
        output_row: list[Any] = [*member_values[key], *members[key]]
        for stage in plan.measurements:
            aligned_values = aligned_by_stage[stage.name].get(key)
            output_row.extend(
                aligned_values
                if aligned_values is not None
                else (None,) * (len(stage.attribute_columns) + len(stage.metric_columns))
            )
        output_row.extend(
            comparison_values[comparison.name][index] for comparison in plan.cohort.comparisons
        )
        output.append(tuple(output_row))
    return CohortResult(
        expected_names,
        tuple(output_metadata),
        tuple(output),
        presence,
        tuple(coverage),
        tuple(provenance),
    )


__all__ = ["CohortCoverage", "CohortResult", "CohortStageProvenance", "run_cohort"]
