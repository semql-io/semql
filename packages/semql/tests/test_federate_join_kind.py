"""Gap C regression: a cross-backend bridge onto a *filter-bearing* foreign
partition must be an ``inner`` join at the merge (restrict the host to matching
rows), while a pure *lookup* foreign partition stays ``left`` (preserve host
rows, contribute attributes only).

Before the fix every bridge rendered ``LEFT``, so a filter on a foreign cube
failed to restrict the fact fragment and inflated the result. These assert the
``join_kind`` on the structured ``MergeSpec`` — the dialect-agnostic contract
the renderer turns into ``INNER`` / ``LEFT JOIN``.

Fixtures: orders (Postgres fact) bridged to customers (BigQuery dim) on
``orders.customer_id = customers.id``.
"""

from __future__ import annotations

from semql.federate import compile_federated_query
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import Filter, SemanticQuery


def _orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_orders(), _customers())}


def test_filter_only_foreign_cube_uses_inner_join() -> None:
    """`revenue where customers.region = EMEA` — customers is filter-only and
    on another backend; the merge must INNER-JOIN to drop orders whose customer
    is not in EMEA."""
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            filters=[Filter(dimension="customers.region", op="eq", values=["EMEA"])],
        ),
        _catalog(),
    )
    assert len(plan.merge_spec.bridges) == 1
    assert plan.merge_spec.bridges[0].join_kind == "inner"


def test_lookup_foreign_cube_stays_left_join() -> None:
    """`revenue per customers.region` — customers contributes an output
    dimension with no filter; the merge LEFT-JOINs so orders with a missing /
    unmatched customer are preserved."""
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"]),
        _catalog(),
    )
    assert len(plan.merge_spec.bridges) == 1
    assert plan.merge_spec.bridges[0].join_kind == "left"


def test_foreign_cube_grouped_and_filtered_uses_inner_join() -> None:
    """A foreign cube that both groups *and* filters is restricting -> inner."""
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            filters=[Filter(dimension="customers.region", op="eq", values=["EMEA"])],
        ),
        _catalog(),
    )
    assert plan.merge_spec.bridges[0].join_kind == "inner"
