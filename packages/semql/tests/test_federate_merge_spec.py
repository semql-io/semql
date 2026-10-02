"""F-M1: the MergeSpec must carry everything a renderer needs to
regenerate the merge SQL without the original SemanticQuery, catalog, or
partition plans.

These assert the gaps closed in F-M1:

- the time-bucket grain rides on the time ``DimensionOutput``
  (``None`` in distributive mode, the grain in raw_rows),
- ``MeasureOutput.merge_agg`` faithfully records raw-rows aggregates
  (``avg`` / percentiles) and ``ratio`` carries its per-side aggs,
- ``cross_partition_clauses`` are resolved to fragment coordinates
  (``(negated, fragment_index, column_name, op, values)``), so no
  catalog / cube-index lookup is needed at render time.
"""

from __future__ import annotations

from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
)
from semql.federate import DimensionOutput, MeasureOutput, compile_federated_query
from semql.spec import BoolExpr, Filter, TimeWindow


def _orders(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
            Measure(name="amount_median", sql="{o}.amount", agg="median"),
            Measure(name="aov", sql="", agg="ratio", numerator="revenue", denominator="count"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        dialect=dialect,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )


def _catalog() -> dict[str, Cube]:
    return Catalog([_orders(), _customers()]).as_dict()


def _time_output(spec_dims: list[DimensionOutput], name: str) -> DimensionOutput:
    return next(d for d in spec_dims if d.output_name == name)


def _measure(spec_measures: list[MeasureOutput], name: str) -> MeasureOutput:
    return next(m for m in spec_measures if m.output_name == name)


# ---------------------------------------------------------------------------
# Time-bucket grain on the spec
# ---------------------------------------------------------------------------


def test_raw_rows_time_output_carries_grain() -> None:
    """Raw-rows buckets at the merge, so the grain must be on the spec."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2020-01-01", "2020-02-01"),
        ),
    )
    plan = compile_federated_query(q, _catalog(), mode="raw_rows")
    time_out = _time_output(plan.merge_spec.dimensions, "created_at_day")
    assert time_out.time_grain == "day"


def test_distributive_time_output_has_no_grain() -> None:
    """Distributive buckets in the fragment; the merge passes the column
    through, so its grain is ``None`` (no re-bucketing at merge)."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2020-01-01", "2020-02-01"),
        ),
    )
    plan = compile_federated_query(q, _catalog(), mode="distributive")
    time_out = _time_output(plan.merge_spec.dimensions, "created_at_day")
    assert time_out.time_grain is None


def test_distributive_granularity_column_meta_aligned() -> None:
    """Regression: distributive federation with a granularity used to
    crash — the time column was emitted as ``created_at_day`` but its
    ColumnMeta was misnamed ``created_at`` (the bare dimension), so the
    output column list and meta list disagreed. They must align."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2020-01-01", "2020-02-01"),
        ),
    )
    plan = compile_federated_query(q, _catalog(), mode="distributive")
    assert "created_at_month" in plan.columns
    assert [m.name for m in plan.column_meta] == plan.columns


# ---------------------------------------------------------------------------
# merge_agg faithfully records every raw-rows aggregate
# ---------------------------------------------------------------------------


def test_raw_rows_percentile_merge_agg() -> None:
    q = SemanticQuery(measures=["orders.amount_median"], dimensions=["customers.region"])
    plan = compile_federated_query(q, _catalog(), mode="raw_rows")
    assert _measure(plan.merge_spec.measures, "amount_median").merge_agg == "median"


def test_raw_rows_avg_merge_agg() -> None:
    q = SemanticQuery(measures=["orders.avg_amount"], dimensions=["customers.region"])
    plan = compile_federated_query(q, _catalog(), mode="raw_rows")
    assert _measure(plan.merge_spec.measures, "avg_amount").merge_agg == "avg"


def test_raw_rows_ratio_carries_per_side_aggs() -> None:
    """A ratio recomposes ``num_agg(num) / NULLIF(den_agg(den), 0)`` — the
    per-side aggs must be on the spec, not only in the rendered SQL."""
    q = SemanticQuery(measures=["orders.aov"], dimensions=["customers.region"])
    plan = compile_federated_query(q, _catalog(), mode="raw_rows")
    aov = _measure(plan.merge_spec.measures, "aov")
    assert aov.merge_agg == "ratio"
    assert aov.numerator_agg == "sum"
    assert aov.denominator_agg == "count"


# ---------------------------------------------------------------------------
# cross_partition_clauses resolved to fragment coordinates
# ---------------------------------------------------------------------------


def test_cross_partition_clause_resolved_to_fragment_coords() -> None:
    """An OR spanning two backends becomes a single cross-partition
    clause whose literals carry ``(negated, fragment_index, column,
    op, values)`` — fully resolved, no cube name left to look up."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["shipped"]),
                Filter(dimension="customers.tier", op="eq", values=["gold"]),
            ],
        ),
    )
    plan = compile_federated_query(q, _catalog(), mode="distributive")
    clauses = plan.merge_spec.cross_partition_clauses
    assert len(clauses) == 1
    literals = clauses[0]
    # Every literal is a fully-resolved 5-tuple with an int fragment idx.
    for negated, frag_idx, col, op, values in literals:
        assert isinstance(negated, bool)
        assert isinstance(frag_idx, int)
        assert isinstance(col, str)
        assert isinstance(op, str)
        assert isinstance(values, tuple)
    coords = {(frag_idx, col) for _neg, frag_idx, col, _op, _vals in literals}
    assert coords == {(0, "status"), (1, "tier")}


def test_many_to_one_merge_records_exact_materialized_one_side_key() -> None:
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        _catalog(),
        mode="distributive",
    )
    (requirement,) = plan.merge_spec.merge_key_requirements
    assert requirement.fragment_index == 1
    assert requirement.columns == ("id",)
    assert requirement.nulls_equal is False
