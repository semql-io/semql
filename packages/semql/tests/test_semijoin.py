"""Compile-level tests for the cross-backend semi-join staged compiler.

A semi-join restricts an outer dimension to the value set of an inner query,
shipping the values as a literal ``IN`` list rather than joining across
backends. These tests cover the compile artifact (:class:`SemiJoinPlan`) and
the refusals; end-to-end execution lives in
``semql-engine/tests/test_semijoin_engine.py``.
"""

from __future__ import annotations

import pytest
from semql import (
    AuthContext,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    SemiJoin,
)
from semql.compile import compile_query
from semql.errors import CompileError, FederationError
from semql.federate import compile_federated_query
from semql.semijoin import SemiJoinPlan, compile_semi_join_query


def _activity(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="activity",
        dialect=dialect,
        table="activity",
        alias="a",
        primary_key="id",
        measures=[Measure(name="active_secs", sql="{a}.secs", agg="sum", unit="duration")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(
                name="employee_id", sql="{a}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{a}.employee_id = {e}.id")],
    )


def _employees(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="employees",
        dialect=dialect,
        table="employees",
        alias="e",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{e}.id", type="number"),
            Dimension(name="dept", sql="{e}.dept", type="string"),
            # A deliberately incompatible key: string where activity.employee_id is number.
            Dimension(name="badge", sql="{e}.badge", type="string"),
        ],
    )


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_activity(), _employees())}


def _sales_semi_join(op: str = "in") -> SemiJoin:
    return SemiJoin(
        dimension="activity.employee_id",
        op=op,  # type: ignore[arg-type]
        select="employees.id",
        source=SemanticQuery(
            dimensions=["employees.id"],
            filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
        ),
    )


def _query(op: str = "in") -> SemanticQuery:
    return SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[_sales_semi_join(op)],
    )


def _orders_customers_catalog(*, protected_customers: bool = False) -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            )
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        required_roles=["customer_reader"] if protected_customers else [],
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    selected = Cube(
        name="selected",
        dialect=Dialect.DUCKDB,
        table="selected",
        alias="s",
        dimensions=[Dimension(name="id", sql="{s}.id", type="number")],
    )
    return {cube.name: cube for cube in (orders, customers, selected)}


def _orders_semi_join_query() -> SemanticQuery:
    return SemanticQuery(
        measures=["orders.revenue"],
        semi_joins=[
            SemiJoin(
                dimension="customers.id",
                select="selected.id",
                source=SemanticQuery(dimensions=["selected.id"]),
            )
        ],
    )


def test_preflight_shape_retains_filter_only_binding_cube_without_changing_output() -> None:
    plan = compile_semi_join_query(_orders_semi_join_query(), _orders_customers_catalog())
    assert {fragment.dialect for fragment in plan.outer_plan.fragments} == {
        Dialect.POSTGRES,
        Dialect.BIGQUERY,
    }
    assert plan.columns == ["revenue"]
    assert plan.column_meta[0].storage_type == "number"
    assert plan.column_meta[0].additive is True


def test_unauthorized_filter_only_binding_cube_refuses_during_compile() -> None:
    with pytest.raises(CompileError):
        compile_semi_join_query(
            _orders_semi_join_query(),
            _orders_customers_catalog(protected_customers=True),
            viewer=AuthContext(viewer_id="viewer", roles=[]),
        )


def test_compiles_to_staged_plan() -> None:
    plan = compile_semi_join_query(_query(), _catalog())
    assert isinstance(plan, SemiJoinPlan)
    assert len(plan.inner_plans) == 1
    assert len(plan.bindings) == 1
    # Inner projects the select dimension under its bare field name.
    assert plan.inner_plans[0].columns == ["id"]
    b = plan.bindings[0]
    assert (b.bind_dimension, b.op, b.select_column) == ("activity.employee_id", "in", "id")
    # Output shape is the outer query's — the IN list never changes projection.
    assert plan.columns == ["employee_id", "active_secs"]


def test_bind_outer_splices_value_list_into_outer_query() -> None:
    plan = compile_semi_join_query(_query(), _catalog())
    outer = plan.bind_outer([Filter(dimension="activity.employee_id", op="in", values=[1, 4, 7])])
    sql = outer.fragments[0].sql
    assert "a.employee_id IN" in sql
    # The outer touches only one backend, so it's a single-fragment plan.
    assert len(outer.fragments) == 1
    assert outer.columns == ["employee_id", "active_secs"]


def test_not_in_is_carried_into_the_binding() -> None:
    plan = compile_semi_join_query(_query(op="not_in"), _catalog())
    assert plan.bindings[0].op == "not_in"


def test_inner_filter_lands_in_inner_plan_not_outer() -> None:
    plan = compile_semi_join_query(_query(), _catalog())
    # The dept='Sales' predicate belongs to the inner (employees) plan.
    assert "e.dept" in plan.inner_plans[0].fragments[0].sql
    # The outer shape carries no employees predicate.
    assert "dept" not in str(plan.columns)


def test_type_incompatible_key_refuses() -> None:
    # activity.employee_id is number; employees.badge is string — coercing the
    # value list would drop or invent matches, so refuse.
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[
            SemiJoin(
                dimension="activity.employee_id",
                select="employees.badge",
                source=SemanticQuery(dimensions=["employees.badge"]),
            )
        ],
    )
    with pytest.raises(FederationError) as ei:
        compile_semi_join_query(q, _catalog())
    assert ei.value.reason == "cross_cube_type_coercion"


def test_unknown_select_reference_refuses() -> None:
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[
            SemiJoin(
                dimension="activity.employee_id",
                select="employees.nope",
                source=SemanticQuery(dimensions=["employees.nope"]),
            )
        ],
    )
    with pytest.raises(FederationError) as ei:
        compile_semi_join_query(q, _catalog())
    assert ei.value.reason == "unqualified_or_unknown_reference"


def test_compile_semi_join_query_requires_semi_joins() -> None:
    q = SemanticQuery(measures=["activity.active_secs"], dimensions=["activity.employee_id"])
    with pytest.raises(FederationError) as ei:
        compile_semi_join_query(q, _catalog())
    assert ei.value.reason == "semi_join_absent"


def test_compile_query_refuses_to_silently_drop_semi_joins() -> None:
    with pytest.raises(FederationError) as ei:
        compile_query(_query(), _catalog())
    assert ei.value.reason == "semi_join_needs_staged_compile"


def test_compile_federated_query_refuses_to_silently_drop_semi_joins() -> None:
    with pytest.raises(FederationError) as ei:
        compile_federated_query(_query(), _catalog())
    assert ei.value.reason == "semi_join_needs_staged_compile"
