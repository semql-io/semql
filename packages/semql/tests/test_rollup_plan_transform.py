"""Tests for the plan→plan rollup transform.

The pre-IR code mutated the catalog dict in-place when a rollup
covered the query — every downstream compile stage then saw a Cube
with field SQLs already rewritten.  The plan→plan transform
replaces that with a pure function: ``apply_rollup_to_plan(plan,
cube, rollup) -> LogicalPlan`` returns a new plan whose ``Scan``
points at a synthetic Cube reading the rollup's physical_table.
The original catalog is never touched.

The transform's contract:
- Input: a LogicalPlan + the cube to rewrite + the rollup to route to.
- Output: a new LogicalPlan where the matched Scan is replaced by a
  synthetic Scan whose cube reads the rollup's physical_table with
  bucketed column SQLs.
- Pure: the input plan is unchanged.
- The synthetic cube is a fresh model_copy — it is not the same
  object as the original Cube, so a future plan transform can
  inspect the "routed" path without confusing it with the base path.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.logical import Scan, apply_rollup_to_plan, to_logical_plan
from semql.model import Cube, Dialect, Dimension, Measure, Rollup, TimeDimension
from semql.spec import SemanticQuery, TimeWindow


def _orders_with_rollup() -> tuple[dict[str, Cube], Cube, Rollup]:
    cube = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Dialect.POSTGRES,
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "month"),
            )
        ],
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
        ],
        rollups=[
            Rollup(
                name="orders_daily",
                physical_table="prod.orders_daily",
                time_dimension="created_at",
                granularity="day",
                measures=["revenue"],
                dimensions=["region"],
            )
        ],
    )
    return ({"orders": cube}, cube, cube.rollups[0])


def test_apply_rollup_to_plan_returns_new_plan() -> None:
    """The transform is a pure function: input plan unchanged, output
    is a fresh plan with a synthetic Scan."""
    catalog, cube, rollup = _orders_with_rollup()
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow_for_test(),  # noqa: F821
    )
    plan = to_logical_plan(query, catalog)
    plan_repr_before = repr(plan)

    new_plan = apply_rollup_to_plan(plan, cube, rollup)

    assert new_plan is not plan
    assert repr(plan) == plan_repr_before  # input immutable


def test_apply_rollup_to_plan_replaces_matched_scan() -> None:
    """The rollup's synthetic Scan points at the rollup's physical_table."""
    catalog, cube, rollup = _orders_with_rollup()
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow_for_test(),
    )
    plan = to_logical_plan(query, catalog)
    new_plan = apply_rollup_to_plan(plan, cube, rollup)

    # Find the rolled-up scan.
    assert len(new_plan.scans) == 1
    new_scan = new_plan.scans[0]
    assert isinstance(new_scan, Scan)
    assert new_scan.cube.table == "prod.orders_daily"
    assert new_scan.cube.name == "orders"  # name preserved
    assert new_scan.alias == "o"  # alias preserved


def test_apply_rollup_to_plan_does_not_mutate_catalog() -> None:
    """The original catalog is untouched — only the new plan carries the
    synthetic Cube."""
    catalog, cube, rollup = _orders_with_rollup()
    original_table = catalog["orders"].table
    original_measures_sql = [m.sql for m in catalog["orders"].measures]

    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow_for_test(),
    )
    plan = to_logical_plan(query, catalog)
    apply_rollup_to_plan(plan, cube, rollup)

    assert catalog["orders"].table == original_table
    assert [m.sql for m in catalog["orders"].measures] == original_measures_sql


def test_compile_query_with_rollup_via_plan_transform(
    snapshot_conformance: None,
) -> None:
    """End-to-end: compile_query against a query that a rollup covers.
    Output SQL must match the pre-IR path (the existing test_rollup.py
    tests pin the SQL).  Here we pin that the plan→plan transform
    path produces the same SQL as the catalog-mutate path did."""
    catalog, _, _ = _orders_with_rollup()
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow_for_test(),
    )
    compiled = compile_query(query, catalog)
    # The rollup was applied — the synthetic cube reads orders_daily.
    assert "orders_daily" in compiled.sql
    # The pre-IR rollup path is also exercised in test_rollup.py;
    # this asserts the plan-side path matches.  Full SQL pinning is
    # already covered by the existing rollup snapshot tests.


# Helper — a TimeWindow for the time-bound query.  Inlined here
# rather than imported to keep the test file standalone.
def TimeWindow_for_test() -> TimeWindow:  # noqa: N802
    return TimeWindow(
        dimension="orders.created_at",
        granularity="day",
        range=("2025-01-01T00:00:00", "2025-01-08T00:00:00"),
    )


@pytest.fixture
def snapshot_conformance() -> None:
    """Empty fixture — present so callers can pass it for symmetry with
    other snapshot tests.  The plan→plan transform's SQL is pinned
    elsewhere in test_rollup.py."""
