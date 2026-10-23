"""Cohort spec and compiler contract regressions."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError
from semql import (
    AuthContext,
    Cohort,
    CohortComparison,
    CohortMeasurement,
    CohortMetric,
    CohortQuery,
    Cube,
    DecimalLiteral,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Measure,
    SemanticQuery,
    TimeWindow,
    View,
    compile_cohort_query,
    compile_query,
)


def _catalog() -> dict[str, Cube]:
    cube = Cube(
        name="activity",
        dialect=Dialect.DUCKDB,
        table="activity",
        alias="a",
        measures=[Measure(name="amount", sql="{a}.amount", agg="sum")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(name="period", sql="{a}.period", type="string"),
            Dimension(name="region", sql="{a}.region", type="string"),
        ],
    )
    return {cube.name: cube}


def _query() -> CohortQuery:
    return CohortQuery(
        cohort=Cohort(
            source=SemanticQuery(
                dimensions=["activity.id", "activity.region"],
                filters=[Filter(dimension="activity.period", op="eq", values=["prior"])],
                order=[("activity.id", "asc")],
            ),
            member_keys=["activity.id"],
            attributes=["activity.region"],
            max_members=10,
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["activity.id", "activity.region"],
                    measures=["activity.amount"],
                    filters=[Filter(dimension="activity.period", op="eq", values=["current"])],
                ),
                key_mapping={"activity.id": "activity.id"},
                attributes=["activity.region"],
            )
        ],
        comparisons=[
            CohortComparison(
                name="change",
                left=CohortMetric(measurement="current", column="amount"),
                right=CohortMetric(measurement="current", column="amount"),
                op="difference",
            )
        ],
    )


def test_cohort_specs_are_strict_and_immutable() -> None:
    query = _query()
    with pytest.raises(ValidationError):
        query.model_copy(update={"unexpected": True})
    with pytest.raises(ValidationError):
        Cohort(source=query.cohort.source, member_keys=["activity.id"], max_members=0)
    with pytest.raises(ValidationError):
        query.cohort.max_members = 2


def test_cohort_query_serialization_round_trips() -> None:
    query = _query()
    restored = CohortQuery.model_validate(query.model_dump(mode="python"))
    assert restored == query
    assert restored.cohort.source.filters == query.cohort.source.filters


def test_selection_metrics_are_available_but_not_cohort_outputs() -> None:
    query = _query()
    source = query.cohort.source.model_copy(update={"measures": ["activity.amount"]})
    query = query.model_copy(
        update={
            "cohort": query.cohort.model_copy(update={"source": source}),
        }
    )
    plan = compile_cohort_query(query, _catalog())
    assert "amount" in plan.source.columns
    assert all(column.name != "selection.amount" for column in plan.columns)


def test_decimal_filter_literal_round_trips_and_binds_exactly() -> None:
    exact = "0.10000000000000000001"
    filter_spec = Filter(
        dimension="activity.id",
        op="eq",
        values=[DecimalLiteral(value=exact)],
    )
    restored = Filter.model_validate(filter_spec.model_dump(mode="json"))
    assert restored == filter_spec
    decimal_value = restored.values[0]
    assert isinstance(decimal_value, DecimalLiteral)
    assert Decimal(decimal_value.value) == Decimal(exact)

    compiled = compile_query(
        SemanticQuery(dimensions=["activity.id"], filters=[filter_spec]),
        _catalog(),
    )
    assert next(iter(compiled.params.values())) == Decimal(exact)

    with pytest.raises(ValidationError):
        DecimalLiteral(value="NaN")


def test_cohort_projection_uses_compiled_metadata_and_rejects_hidden_time_grain() -> None:
    plan = compile_cohort_query(_query(), _catalog())
    selection_id = next(column for column in plan.columns if column.name == "selection.id")
    selection_meta = next(meta for meta in plan.source.column_meta if meta.name == "id")
    assert selection_id.child_meta == selection_meta
    current_amount = next(column for column in plan.columns if column.name == "current.amount")
    current_meta = next(meta for meta in plan.measurements[0].column_meta if meta.name == "amount")
    assert current_amount.child_meta == current_meta

    source = plan.cohort.cohort.source.model_copy(
        update={
            "time_dimension": TimeWindow(
                dimension="activity.period",
                range=("2026-01-01", "2026-02-01"),
                granularity="day",
            )
        }
    )
    with pytest.raises(ValidationError, match="grouped or filled time dimension"):
        Cohort(
            source=source,
            member_keys=["activity.id"],
            attributes=["activity.region"],
        )


def test_comparison_metadata_tracks_inline_measure_units() -> None:
    catalog = _catalog()
    catalog["activity"] = catalog["activity"].model_copy(
        update={"measures": [Measure(name="amount", sql="{a}.amount", agg="sum", unit="seconds")]}
    )
    query = _query()
    measurement = query.measurements[0].model_copy(
        update={
            "query": query.measurements[0].query.model_copy(
                update={
                    "derived_measures": [
                        InlineDerived(
                            name="double_amount",
                            op="sum",
                            operands=["activity.amount", "activity.amount"],
                        )
                    ]
                }
            )
        }
    )
    query = query.model_copy(
        update={
            "measurements": [measurement],
            "comparisons": [
                CohortComparison(
                    name="derived_delta",
                    left=CohortMetric(measurement="current", column="double_amount"),
                    right=CohortMetric(measurement="current", column="amount"),
                )
            ],
        }
    )
    plan = compile_cohort_query(query, catalog)
    metadata = next(
        column.child_meta for column in plan.columns if column.name == "comparison.derived_delta"
    )
    assert metadata is not None
    assert metadata.unit == "seconds"

    for operation in ("ratio", "percent_change"):
        comparison_query = query.model_copy(
            update={
                "comparisons": [
                    query.comparisons[0].model_copy(update={"name": operation, "op": operation})
                ]
            }
        )
        comparison_plan = compile_cohort_query(comparison_query, catalog)
        comparison_meta = next(
            column.child_meta
            for column in comparison_plan.columns
            if column.name == f"comparison.{operation}"
        )
        assert comparison_meta is not None
        assert comparison_meta.unit is None
        assert comparison_meta.display_unit is None
        if operation == "percent_change":
            assert comparison_meta.format == "percent"


def test_comparison_metadata_preserves_operand_masking() -> None:
    catalog = _catalog()
    catalog["activity"] = catalog["activity"].model_copy(
        update={
            "measures": [
                Measure(
                    name="amount",
                    sql="{a}.amount",
                    agg="sum",
                    required_roles=["finance"],
                    mask_roles=["finance"],
                    mask_value="0",
                )
            ]
        }
    )
    plan = compile_cohort_query(
        _query(),
        catalog,
        viewer=AuthContext(viewer_id="viewer", roles=["finance"]),
    )
    metadata = next(
        column.child_meta for column in plan.columns if column.name == "comparison.change"
    )
    assert metadata is not None
    assert metadata.masked


def test_view_fields_bind_to_compiler_resolved_output_columns() -> None:
    view = View(
        name="activity_view",
        fields={"person": "activity.id", "selected_region": "activity.region"},
    )
    source = SemanticQuery(dimensions=["activity_view.person", "activity_view.selected_region"])
    query = CohortQuery(
        cohort=Cohort(
            source=source,
            member_keys=["activity_view.person"],
            attributes=["activity_view.selected_region"],
        ),
        measurements=[
            CohortMeasurement(
                name="current",
                query=SemanticQuery(
                    dimensions=["activity_view.person", "activity_view.selected_region"],
                    measures=["activity.amount"],
                ),
                key_mapping={"activity_view.person": "activity_view.person"},
                attributes=["activity_view.selected_region"],
            )
        ],
    )

    plan = compile_cohort_query(query, _catalog(), views={view.name: view})
    assert plan.source.key_columns == ("person",)
    assert plan.source.attribute_columns == ("selected_region",)
    assert plan.measurements[0].key_columns == ("person",)
