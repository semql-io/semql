"""Minimal multi-backend catalog fixture for compiler tests.

Cubes are kept generic — no domain knowledge from any specific
product — so they test the compiler contract, not a particular schema.

Topology
--------
orders (PG) ──→ customers (PG, hidden)
          └───→ products  (PG, hidden)
sessions  (CH)                          # cross-backend isolation
restricted (PG, required_filters)       # required-filter enforcement
"""

from __future__ import annotations

import pytest
from semql.introspect import META_CUBES
from semql.model import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    TimeDimension,
)

CONTEXT = {"schema": "test_schema"}


@pytest.fixture(scope="session")
def catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
            Dimension(name="is_paid", sql="{o}.is_paid", type="bool"),
        ],
        time_dimensions=[
            # Deliberately excludes "hour" to test granularity rejection.
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
            Join(to="products", relationship="many_to_one", on="{o}.product_id = {p}.id"),
        ],
    )

    # Both customers and products expose a `name` dimension — used to test
    # the column-collision prefixing logic.
    customers = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        expose_in_prompt=False,
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="name", sql="{c}.name", type="string"),
            Dimension(name="email", sql="{c}.email", type="string"),
            Dimension(name="is_active", sql="{c}.is_active", type="bool"),
        ],
    )

    products = Cube(
        name="products",
        backend=Dialect.POSTGRES,
        table="{schema}.products",
        alias="p",
        expose_in_prompt=False,
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="name", sql="{p}.name", type="string"),
            Dimension(name="sku", sql="{p}.sku", type="string"),
        ],
    )

    sessions = Cube(
        name="sessions",
        backend=Dialect.CLICKHOUSE,
        table="{schema}.sessions",
        alias="s",
        base_predicate="{s}.event_type = 'active'",
        measures=[
            Measure(name="duration", sql="{s}.duration_sec", agg="sum", unit="duration"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="app_name", sql="{s}.app_name", type="string"),
        ],
        time_dimensions=[
            TimeDimension(name="started_at", sql="{s}.started_at"),
        ],
    )

    # required_filters forces callers to always filter on `flag_type`.
    restricted = Cube(
        name="restricted",
        backend=Dialect.POSTGRES,
        table="{schema}.restricted",
        alias="r",
        expose_in_prompt=False,
        required_filters=["flag_type"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="flag_type", sql="{r}.flag_type", type="string")],
    )

    cubes = [orders, customers, products, sessions, restricted, *META_CUBES]
    return {c.name: c for c in cubes}
