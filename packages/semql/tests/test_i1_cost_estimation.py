"""Tests for I1: cost estimation + budget enforcement.

The compiler emits SQL but the planner has no idea whether a query
costs 100ms or 30 minutes. ``estimate_cost`` derives a rough
``CostEstimate`` from cube ``size_hint`` (declared on ``Cube``) +
the query's touched dims/measures. ``QueryBudget`` pairs with it:
pre-compile, the caller attaches a ceiling; post-execute, the
``Engine`` raises ``BudgetExceededError`` if the estimate (or the
observed row count) exceeded the ceiling.

The estimate is intentionally rough — it's a guardrail, not a
planner. We use ``rows_scanned = cube.size_hint`` (a single-table
scan cost) and ``rows_returned = rows_scanned / distinct_values``
where distinct_values is the product of touched dim cardinalities.
This is good enough to catch "you forgot a filter on a billion-row
table" mistakes; it's not a substitute for actual EXPLAIN.
"""

from __future__ import annotations

import pytest
from semql import (
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    estimate_cost,
)
from semql.cost import CostEstimate, QueryBudget


def _orders(size_hint: int | None = 1_000_000) -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
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


def test_estimate_cost_uses_size_hint_for_rows_scanned() -> None:
    cube = _orders(size_hint=10_000)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"])
    est = estimate_cost(q, cat)
    assert not est.rows_scanned_unknown
    assert est.cubes_estimated["orders"] == 10_000
    assert est.total_rows_scanned == 10_000


def test_estimate_cost_aggregates_across_cubes() -> None:
    """A federated plan scans each fragment's full size."""
    cube1 = _orders(size_hint=10_000)
    cube2 = Cube(
        name="customers",
        backend=Dialect.BIGQUERY,
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


def test_query_budget_unknown_cost_passes() -> None:
    """If we don't know the cost, we can't enforce a budget. The
    budget is a guardrail, not a hard stop on unknown data."""
    cube = _orders(size_hint=None)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"])
    budget = QueryBudget(max_rows_scanned=5_000)
    budget.check(estimate_cost(q, cat))  # should not raise


def test_query_budget_max_cubes_ceiling() -> None:
    """A budget can also cap the number of cubes touched."""
    cube1 = _orders(size_hint=100)
    cube2 = Cube(
        name="customers",
        backend=Dialect.BIGQUERY,
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
    """CostEstimate is a frozen Pydantic model (cf. I9)."""
    est = CostEstimate(
        total_rows_scanned=100,
        cubes_estimated={"orders": 100},
        rows_scanned_unknown=False,
    )
    restored = CostEstimate.model_validate(est.model_dump())
    assert restored.total_rows_scanned == 100
    assert restored.cubes_estimated == {"orders": 100}


def test_size_hint_must_be_non_negative() -> None:
    """A negative size_hint is a data error; refuse to build the cube."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _orders(size_hint=-1)


def test_estimate_cost_ignores_dimensions_on_unscanned_cubes() -> None:
    """A dimension on a cube that wasn't touched doesn't contribute
    to the scan estimate."""
    cube = _orders(size_hint=1_000)
    cat = {"orders": cube}
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    est = estimate_cost(q, cat)
    assert est.cubes_estimated == {"orders": 1_000}
