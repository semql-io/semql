"""Tests for the refined LogicalPlan IR node vocabulary.

Stage 1 of the LogicalPlan IR refactor (see
``docs/notes/logical-plan-ir-design.org``):

- ``Filter.expr`` is the spec tree (``BoolExpr | Filter``), not a SQL string.
- New types: ``ColumnRef``, ``TimeBreakdown``, ``OrderBy``, ``Limit``,
  ``CompareSplit``.
- ``Project.columns`` is ``list[ColumnRef]`` carrying kind metadata.
- ``LogicalPlan`` carries ``order``, ``limit``, ``time_window``,
  ``compare`` so the plan is self-contained for emission.
- Each node has a readable ``__repr__`` for the explain() use case.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from semql.compile import compile_query
from semql.logical import (
    ColumnRef,
    CompareSplit,
    Join,
    Limit,
    LogicalPlan,
    OrderBy,
    Predicate,
    Scan,
    TimeBreakdown,
    to_logical_plan,
)
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension
from semql.spec import (
    BoolExpr,
    CompareWindow,
    SemanticQuery,
    TimeWindow,
)
from semql.spec import (
    Filter as SpecFilter,
)


def _orders_catalog() -> dict[str, Cube]:
    return {
        "orders": Cube(
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
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }


def test_filter_carries_spec_tree_not_sql_string() -> None:
    """Filter.expr is a BoolExpr | Filter from semql.spec, not a string."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        where=BoolExpr(
            op="and",
            children=[
                SpecFilter(dimension="orders.region", op="eq", values=["us"]),
                SpecFilter(dimension="orders.region", op="neq", values=["ca"]),
            ],
        ),
    )

    plan = to_logical_plan(query, catalog)

    assert len(plan.filters) == 1
    assert isinstance(plan.filters[0], Predicate)
    assert isinstance(plan.filters[0].expr, BoolExpr)
    assert plan.filters[0].expr.op == "and"
    # Tree is preserved verbatim, not lowered to SQL.
    assert all(isinstance(c, SpecFilter) for c in plan.filters[0].expr.children)


def test_filter_node_carries_flat_filter_leaf() -> None:
    """A flat ``q.filters`` list yields one Filter node per leaf."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        filters=[
            SpecFilter(dimension="orders.region", op="eq", values=["us"]),
            SpecFilter(dimension="orders.region", op="neq", values=["ca"]),
        ],
    )

    plan = to_logical_plan(query, catalog)

    assert len(plan.filters) == 2
    assert all(isinstance(f.expr, SpecFilter) for f in plan.filters)


def test_project_columns_carry_kind_metadata() -> None:
    """Project.columns are ColumnRef with kind set per the field type."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )

    plan = to_logical_plan(query, catalog)

    assert len(plan.project.columns) == 2
    by_kind = {col.kind for col in plan.project.columns}
    assert by_kind == {"dimension", "measure"}
    for col in plan.project.columns:
        assert isinstance(col, ColumnRef)
        assert col.cube.name == "orders"
        assert col.alias  # non-empty


def test_aggregate_time_breakdown_from_time_dimension() -> None:
    """A query with granularity populates Aggregate.time."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2025-01-01T00:00:00", "2025-01-02T00:00:00"),
        ),
    )

    plan = to_logical_plan(query, catalog)

    assert plan.aggregate is not None
    assert plan.aggregate.time is not None
    assert isinstance(plan.aggregate.time, TimeBreakdown)
    assert plan.aggregate.time.field_name == "created_at"
    assert plan.aggregate.time.granularity == "day"
    assert plan.time_window is query.time_dimension


def test_order_limit_captured_on_plan() -> None:
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        order=[("orders.revenue", "desc")],
        limit=10,
        offset=5,
    )

    plan = to_logical_plan(query, catalog)

    assert isinstance(plan.order, OrderBy)
    assert plan.order.keys == [("orders.revenue", "desc")]
    assert isinstance(plan.limit, Limit)
    assert plan.limit.limit == 10
    assert plan.limit.offset == 5


def test_compare_sets_compare_split_node() -> None:
    """CompareWindow wraps the plan in a CompareSplit node."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2025-01-01T00:00:00", "2025-01-02T00:00:00"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )

    plan = to_logical_plan(query, catalog)

    assert plan.compare is not None
    assert isinstance(plan.compare, CompareSplit)
    # The CompareSplit is a *template* — it carries one inner plan
    # the emitter instantiates twice.  The two ranges are computed
    # at plan-build time (previous_period shifts the current_range
    # back by its own duration).
    assert plan.compare.plan is not None
    assert plan.compare.plan.compare is None
    assert query.time_dimension is not None
    cs = datetime.fromisoformat(query.time_dimension.range[0])
    ce = datetime.fromisoformat(query.time_dimension.range[1])
    duration = ce - cs
    assert plan.compare.current_range == query.time_dimension.range
    assert plan.compare.prior_range == (
        (cs - duration).isoformat(),
        cs.isoformat(),
    )


def test_plan_repr_is_one_line_per_node() -> None:
    """LogicalPlan.__repr__ is human-readable (one short line per node)."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(query, catalog)

    text = repr(plan)
    # One line per node category at minimum.
    assert "scans=" in text
    assert "joins=" in text
    assert "filters=" in text
    assert "aggregate=" in text
    assert "project=" in text


def test_to_logical_plan_simple() -> None:
    """Backwards-compat: pre-existing simple case still works."""
    orders = Cube(
        name="orders",
        alias="orders",
        table="raw_orders",
        backend=Dialect.POSTGRES,
        dimensions=[Dimension(name="id", sql="id", type="string")],
        measures=[Measure(name="revenue", sql="amount", agg="sum")],
    )
    catalog = {"orders": orders}
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.id"],
    )

    plan = to_logical_plan(query, catalog)
    assert isinstance(plan, LogicalPlan)
    assert len(plan.scans) == 1
    assert isinstance(plan.scans[0], Scan)
    assert plan.scans[0].cube.name == "orders"


@pytest.mark.xfail(
    reason=(
        "Pre-wire-up regression guard.  Before Stage 2 of the LogicalPlan "
        "IR refactor landed, compile_query did NOT call to_logical_plan. "
        "After Stage 2 the IR is wired in (see "
        "test_compile_query_lowers_via_logical_plan for the positive "
        "assertion).  This xfail preserves the breadcrumb."
    ),
    strict=True,
)
def test_compile_query_does_not_depend_on_partial_logical_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-wire-up regression guard.  Marked xfail to keep the breadcrumb.

    Before the LogicalPlan wire-up (Stage 2), compile_query did NOT
    call to_logical_plan — the IR was an inert intermediate scaffolding
    that lived in semql.logical without driving the compiler.  This
    test pinned that "no dependency" state.

    After Stage 2 the IR is wired in.  See
    ``test_compile_query_lowers_via_logical_plan`` for the positive
    assertion that replaces this guard.
    """
    orders = Cube(
        name="orders",
        alias="orders",
        table="raw_orders",
        backend=Dialect.POSTGRES,
        dimensions=[Dimension(name="id", sql="id", type="string")],
        measures=[Measure(name="revenue", sql="amount", agg="sum")],
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "compile_query must not call the partial LogicalPlan lowering; "
            "see test_compile_query_lowers_via_logical_plan for the post-wire-up check."
        )

    monkeypatch.setattr("semql.logical.to_logical_plan", fail_if_called)
    compiled = compile_query(SemanticQuery(measures=["orders.revenue"]), {"orders": orders})
    assert "SUM" in compiled.sql


def test_compile_query_lowers_via_logical_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive post-wire-up assertion.  compile_query must lower the
    SemanticQuery through to_logical_plan and reach the emitter; the
    output SQL is unchanged from the pre-wire-up path."""
    orders = Cube(
        name="orders",
        alias="orders",
        table="raw_orders",
        backend=Dialect.POSTGRES,
        dimensions=[Dimension(name="id", sql="id", type="string")],
        measures=[Measure(name="revenue", sql="amount", agg="sum")],
    )
    catalog = {"orders": orders}
    query = SemanticQuery(measures=["orders.revenue"])

    call_count = {"n": 0}

    real_to_logical_plan = to_logical_plan

    def counting(*args: object, **kwargs: object) -> object:
        call_count["n"] += 1
        return real_to_logical_plan(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("semql.logical.to_logical_plan", counting)
    compiled = compile_query(query, catalog)

    assert call_count["n"] >= 1, "compile_query must call to_logical_plan"
    assert "SUM" in compiled.sql
    # The plan is also exposed on the env (introspection surface).
    assert compiled  # smoke: build succeeded


def test_join_node_unchanged_shape() -> None:
    """The Join node shape stays the same — Scan/Join/Filter/Aggregate/Project/LogicalPlan
    scaffolding remains compatible."""
    catalog = _orders_catalog()
    query = SemanticQuery(measures=["orders.revenue"])
    plan = to_logical_plan(query, catalog)
    # No joins in single-cube query.
    assert plan.joins == []
    assert all(isinstance(j, Join) for j in plan.joins)


def test_project_columns_default_to_cube_local_name() -> None:
    """When no name collision, ColumnRef.alias matches the field local-name."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(query, catalog)
    aliases = {c.alias for c in plan.project.columns}
    assert "revenue" in aliases
    assert "region" in aliases
