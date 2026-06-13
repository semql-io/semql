"""Fan-out guard: refuse additive measures a join would inflate.

A ``one_to_many`` / ``many_to_one`` join duplicates the rows of its "one"
side; ``SUM`` / ``COUNT`` over a measure on that cube then double-counts
(the canonical semantic-layer wrong result). ``Join.relationship`` records
the cardinality; the compiler now reads it.
"""

from __future__ import annotations

import pytest
from semql.catalog import Catalog
from semql.errors import CompileError
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import SemanticQuery


def _customers(measures: list[Measure]) -> Cube:
    # customers is the "one" side: one customer, many orders.
    return Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
        measures=measures,
        joins=[Join(to="orders", relationship="one_to_many", on="{c}.id = {o}.customer_id")],
    )


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="customer_id", sql="{o}.customer_id", type="number"),
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        # Same cardinality declared from the orders side (many orders, one
        # customer) so the join graph is reachable whichever cube roots the
        # query. The relationship is intrinsic — both edges agree customers
        # is the "one" (fanned-out) side.
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


@pytest.mark.parametrize("agg", ["sum", "count"])
def test_fan_out_refuses_additive_measure_on_duplicated_side(agg: str) -> None:
    sql = "*" if agg == "count" else "{c}.lifetime_value"
    customers = _customers([Measure(name="m", sql=sql, agg=agg, unit="count")])  # type: ignore[arg-type]
    cat = Catalog([customers, _orders()])
    # Joining customers (one side) to orders (many) duplicates customer
    # rows; aggregating customers.m over them over-counts.
    q = SemanticQuery(measures=["customers.m"], dimensions=["orders.region"])
    with pytest.raises(CompileError, match="fans out"):
        cat.compile(q)


def test_fan_out_allows_measure_on_the_many_side() -> None:
    # SUM(orders.revenue) grouped by a customers dimension: orders is the
    # "many" side, each order row appears once — no inflation.
    customers = _customers([])
    cat = Catalog([customers, _orders()])
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    out = cat.compile(q)
    assert "SUM" in out.sql


@pytest.mark.parametrize("agg", ["min", "max", "count_distinct"])
def test_fan_out_allows_duplication_invariant_aggs(agg: str) -> None:
    # min / max / count_distinct are unchanged by row duplication, so they
    # stay legal even on the fanned-out side.
    customers = _customers([Measure(name="m", sql="{c}.score", agg=agg, unit="count")])  # type: ignore[arg-type]
    cat = Catalog([customers, _orders()])
    q = SemanticQuery(measures=["customers.m"], dimensions=["orders.region"])
    out = cat.compile(q)  # must not raise
    assert out.sql


def test_fan_out_allows_single_cube_aggregation() -> None:
    # Same measure, no join traversed — aggregating on its own grain is fine.
    customers = _customers([Measure(name="m", sql="*", agg="count", unit="count")])
    cat = Catalog([customers, _orders()])
    q = SemanticQuery(measures=["customers.m"], dimensions=["customers.region"])
    out = cat.compile(q)
    assert "COUNT" in out.sql.upper()
