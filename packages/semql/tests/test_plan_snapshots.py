"""Snapshot tests for the LogicalPlan IR repr.

A regression in plan shape (e.g. a node type added, a field renamed)
will surface here *before* the emitted SQL drifts.  The plan is the
intermediate IR between the spec and the sqlglot AST — silent
changes at this layer are still load-bearing for the wire-up.

Tested fixtures mirror ``test_snapshots`` so a plan snapshot + an
SQL snapshot together pin both ends of the pipeline.
"""

from __future__ import annotations

from semql import (
    CompareWindow,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from semql.logical import to_logical_plan
from syrupy.assertion import SnapshotAssertion


def _orders_catalog() -> dict[str, Cube]:
    return {
        "orders": Cube(
            name="orders",
            backend=Dialect.POSTGRES,
            table="{schema}.orders",
            alias="o",
            base_predicate="{o}.deleted_at IS NULL",
            measures=[
                Measure(name="revenue", sql="{o}.amount", agg="sum"),
            ],
            dimensions=[
                Dimension(name="region", sql="{o}.region", type="string"),
            ],
        )
    }


def test_plan_snap_simple_aggregation(snapshot: SnapshotAssertion) -> None:
    """Single-measure, single-dimension aggregation — the simplest
    non-trivial plan shape."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(query, catalog)
    assert repr(plan) == snapshot


def test_plan_snap_filtered_aggregation(snapshot: SnapshotAssertion) -> None:
    """Flat filter list — pins Predicate carrying a Filter leaf."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
        ],
    )
    plan = to_logical_plan(query, catalog)
    assert repr(plan) == snapshot


def test_plan_snap_where_tree(snapshot: SnapshotAssertion) -> None:
    """BoolExpr tree — pins Predicate carrying a non-leaf tree."""
    from semql.spec import BoolExpr

    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.region", op="eq", values=["ca"]),
            ],
        ),
    )
    plan = to_logical_plan(query, catalog)
    assert repr(plan) == snapshot


def test_plan_snap_time_breakdown_with_granularity(snapshot: SnapshotAssertion) -> None:
    """TimeWindow with granularity populates Aggregate.time and Project."""
    catalog = {
        "orders": Cube(
            name="orders",
            backend=Dialect.POSTGRES,
            table="{schema}.orders",
            alias="o",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(
                    name="created_at",
                    sql="{o}.created_at",
                    granularities=("day",),
                ),
            ],
        )
    }
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2025-01-01T00:00:00", "2025-01-08T00:00:00"),
        ),
    )
    plan = to_logical_plan(query, catalog)
    assert repr(plan) == snapshot


def test_plan_snap_compare_mode(snapshot: SnapshotAssertion) -> None:
    """CompareWindow populates plan.compare with computed ranges."""
    catalog = {
        "orders": Cube(
            name="orders",
            backend=Dialect.POSTGRES,
            table="{schema}.orders",
            alias="o",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(
                    name="created_at",
                    sql="{o}.created_at",
                    granularities=("day",),
                ),
            ],
        )
    }
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2025-01-01T00:00:00", "2025-01-08T00:00:00"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    plan = to_logical_plan(query, catalog)
    assert repr(plan) == snapshot
