"""Tests for referenced-cube size-hint accounting and admission budgets."""

from __future__ import annotations

import pytest
from semql import (
    BoolExpr,
    Cube,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Measure,
    SemanticQuery,
    View,
    estimate_cost,
)
from semql.cost import CostEstimate, QueryBudget


def _orders(size_hint: int | None = 1_000_000) -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        size_hint=size_hint,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
    )


def test_estimate_cost_no_cube_size_hint_returns_unknown() -> None:
    """If the cube has no size_hint, the estimate is 'unknown' rather
    than a wrong number. (Better to be honest than to lie.)"""
    cube = _orders(size_hint=None)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    est = estimate_cost(q, cat)
    assert est.rows_scanned_unknown
    assert est.cubes_estimated == {}
    assert est.cubes_unknown == ("orders",)


def test_estimate_cost_uses_size_hint_for_rows_scanned() -> None:
    cube = _orders(size_hint=10_000)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"])
    est = estimate_cost(q, cat)
    assert not est.rows_scanned_unknown
    assert est.cubes_estimated["orders"] == 10_000
    assert est.total_rows_scanned == 10_000


def test_estimate_cost_aggregates_across_cubes() -> None:
    cube1 = _orders(size_hint=10_000)
    cube2 = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        size_hint=1_000,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    cat = {"orders": cube1, "customers": cube2}
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.id"],
    )
    est = estimate_cost(q, cat)
    assert est.total_rows_scanned == 11_000
    assert est.cubes_estimated == {"orders": 10_000, "customers": 1_000}


def test_estimate_cost_returns_zero_for_empty_query() -> None:
    """An empty SemanticQuery scans nothing."""
    cube = _orders(size_hint=1_000)
    cat = {"orders": cube}
    q = SemanticQuery()
    est = estimate_cost(q, cat)
    assert est.total_rows_scanned == 0


def test_query_budget_enforces_rows_scanned_ceiling() -> None:
    """A budget with a 5,000-row ceiling is exceeded by a 10,000-row query."""
    cube = _orders(size_hint=10_000)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"])
    budget = QueryBudget(max_rows_scanned=5_000)
    with pytest.raises(Exception) as exc_info:
        budget.check(estimate_cost(q, cat))
    assert "exceeds budget" in str(exc_info.value).lower()


def test_query_budget_passes_when_within_ceiling() -> None:
    cube = _orders(size_hint=100)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"])
    budget = QueryBudget(max_rows_scanned=5_000)
    budget.check(estimate_cost(q, cat))  # should not raise


def test_query_budget_unknown_cost_rejects_by_default() -> None:
    cube = _orders(size_hint=None)
    query = SemanticQuery(measures=["orders.revenue"])
    with pytest.raises(Exception, match="unknown-cost policy"):
        QueryBudget(max_rows_scanned=5_000).check(estimate_cost(query, {"orders": cube}))


def test_query_budget_can_explicitly_allow_unknown_cost() -> None:
    cube = _orders(size_hint=None)
    query = SemanticQuery(measures=["orders.revenue"])
    QueryBudget(max_rows_scanned=5_000, unknown_cost_policy="allow").check(
        estimate_cost(query, {"orders": cube})
    )


def test_query_budget_known_subtotal_exceeded_even_with_unknown_cube() -> None:
    known = _orders(size_hint=1_000)
    unknown = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        size_hint=None,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    query = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.id"])
    estimate = estimate_cost(query, {"orders": known, "customers": unknown})
    with pytest.raises(Exception, match="1000"):
        QueryBudget(max_rows_scanned=10, unknown_cost_policy="allow").check(estimate)


def test_query_budget_counts_each_unknown_cube() -> None:
    first = _orders(size_hint=None)
    second = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        size_hint=None,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    query = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.id"])
    estimate = estimate_cost(query, {"orders": first, "customers": second})
    assert estimate.cubes_unknown == ("customers", "orders")
    with pytest.raises(Exception, match="2 cube"):
        QueryBudget(max_cubes=1).check(estimate)


def test_estimate_cost_resolves_views_derived_operands_where_and_having() -> None:
    orders = _orders(size_hint=100)
    customers = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        size_hint=300,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
        measures=[Measure(name="spend", sql="{c}.spend", agg="sum", unit="currency")],
    )
    query = SemanticQuery(
        measures=["sales.revenue", "customers.spend"],
        derived_measures=[
            InlineDerived(name="combined", op="sum", operands=["customers.spend", "orders.revenue"])
        ],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="customers.id", op="gt", values=[1]),
                Filter(dimension="orders.id", op="gt", values=[1]),
            ],
        ),
        having=[Filter(dimension="orders.revenue", op="gt", values=[1])],
    )
    estimate = estimate_cost(
        query,
        {"orders": orders, "customers": customers},
        views={"sales": View(name="sales", fields={"revenue": "orders.revenue"})},
    )
    assert estimate.cubes_estimated == {"customers": 300, "orders": 100}


def test_query_budget_known_subtotal_checked_before_unknown_policy() -> None:
    estimate = CostEstimate(
        total_rows_scanned=1_000,
        cubes_estimated={"orders": 1_000},
        cubes_unknown=("customers",),
        rows_scanned_unknown=True,
    )
    with pytest.raises(Exception, match="1000"):
        QueryBudget(max_rows_scanned=10).check(estimate)


def test_query_budget_max_cubes_ceiling() -> None:
    """A budget can also cap the number of cubes touched."""
    cube1 = _orders(size_hint=100)
    cube2 = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        size_hint=100,
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    cat = {"orders": cube1, "customers": cube2}
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.id"])
    budget = QueryBudget(max_cubes=1)
    with pytest.raises(Exception) as exc_info:
        budget.check(estimate_cost(q, cat))
    assert "cube" in str(exc_info.value).lower()


def test_cost_estimate_is_pydantic_value_type() -> None:
    """CostEstimate is a frozen Pydantic model including unknown identities."""
    est = CostEstimate(
        total_rows_scanned=100,
        cubes_estimated={"orders": 100},
        cubes_unknown=("customers",),
        rows_scanned_unknown=True,
    )
    restored = CostEstimate.model_validate(est.model_dump())
    assert restored.total_rows_scanned == 100
    assert restored.cubes_estimated == {"orders": 100}
    assert restored.cubes_unknown == ("customers",)


def test_cost_estimate_rejects_unknown_summary_without_identity() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="requires unknown cube identities"):
        CostEstimate(rows_scanned_unknown=True)

    estimate = CostEstimate(cubes_unknown=("customers",))
    assert estimate.rows_scanned_unknown


def test_size_hint_must_be_non_negative() -> None:
    """A negative size_hint is a data error; refuse to build the cube."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _orders(size_hint=-1)


def test_estimate_cost_counts_each_referenced_cube_once() -> None:
    cube = _orders(size_hint=1_000)
    query = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    estimate = estimate_cost(query, {"orders": cube})
    assert estimate.cubes_estimated == {"orders": 1_000}
    assert estimate.total_rows_scanned == 1_000
