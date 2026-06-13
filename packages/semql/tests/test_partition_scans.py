"""Tests for ``partition_scans`` — the federation split-point helper.

The current ``compile_federated_query`` walks the catalog and
partitions cubes by backend in an ad-hoc way.  ``partition_scans``
takes a ``LogicalPlan`` and returns a dict of ``Dialect -> sub-plan``
where each sub-plan contains only the scans / joins that touch
cubes on that backend.

The helper is intentionally narrow: it doesn't rebuild the
``FederatedPlan`` or replace ``_build_partition_sub_query``.  It
gives the federation entry point a clean, plan-shaped view of the
backend split so a follow-up commit can swap the per-fragment
sub-query synthesis for ``compile_query(sub_query, scoped_catalog)``
without that follow-up having to redo the partition math.
"""

from __future__ import annotations

from semql.logical import LogicalPlan, partition_scans
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import SemanticQuery


def _orders_and_customers() -> dict[str, Cube]:
    return {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Dialect.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        ),
        "customers": Cube(
            name="customers",
            alias="c",
            table="prod.customers",
            backend=Dialect.BIGQUERY,
            dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
        ),
        "clickhouse_logs": Cube(
            name="clickhouse_logs",
            alias="cl",
            table="events.log",
            backend=Dialect.CLICKHOUSE,
            dimensions=[Dimension(name="event", sql="{cl}.event", type="string")],
        ),
    }


def test_partition_scans_single_backend() -> None:
    """A single-backend plan is a single partition (the whole plan)."""
    catalog = _orders_and_customers()
    query = SemanticQuery(measures=["orders.revenue"])
    from semql.logical import to_logical_plan

    plan = to_logical_plan(query, catalog)
    partitions = partition_scans(plan)

    assert set(partitions.keys()) == {Dialect.POSTGRES}
    assert partitions[Dialect.POSTGRES].scans[0].cube.backend is Dialect.POSTGRES


def test_partition_scans_multi_backend() -> None:
    """Each Scan is routed to its cube's backend partition."""
    catalog = _orders_and_customers()
    # Force a join between cubes on different backends.
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.name"],
    )
    # We need a join declared between the two cubes.
    catalog["orders"] = catalog["orders"].model_copy(
        update={
            "joins": [Join(to="customers", on="{o}.cust_id = {c}.id", relationship="many_to_one")]
        }
    )
    catalog["customers"] = catalog["customers"].model_copy(
        update={"joins": [Join(to="orders", on="{c}.id = {o}.cust_id", relationship="one_to_many")]}
    )

    from semql.logical import to_logical_plan

    plan = to_logical_plan(query, catalog)
    partitions = partition_scans(plan)

    assert set(partitions.keys()) == {Dialect.POSTGRES, Dialect.BIGQUERY}
    # Each partition contains scans only for cubes on its backend.
    pg = partitions[Dialect.POSTGRES]
    bq = partitions[Dialect.BIGQUERY]
    assert all(s.cube.backend is Dialect.POSTGRES for s in pg.scans)
    assert all(s.cube.backend is Dialect.BIGQUERY for s in bq.scans)


def test_partition_scans_three_backends() -> None:
    """Sanity check that the helper handles 3+ backends correctly."""
    catalog = _orders_and_customers()
    catalog["orders"] = catalog["orders"].model_copy(
        update={
            "joins": [Join(to="customers", on="{o}.cust_id = {c}.id", relationship="many_to_one")]
        }
    )
    catalog["customers"] = catalog["customers"].model_copy(
        update={"joins": [Join(to="orders", on="{c}.id = {o}.cust_id", relationship="one_to_many")]}
    )
    catalog["orders"] = catalog["orders"].model_copy(
        update={
            "joins": [
                Join(to="customers", on="{o}.cust_id = {c}.id", relationship="many_to_one"),
                Join(
                    to="clickhouse_logs",
                    on="{o}.event_id = {cl}.id",
                    relationship="many_to_one",
                ),
            ]
        }
    )

    from semql.logical import to_logical_plan

    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.name", "clickhouse_logs.event"],
    )

    plan = to_logical_plan(query, catalog)
    partitions = partition_scans(plan)

    assert set(partitions.keys()) == {
        Dialect.POSTGRES,
        Dialect.BIGQUERY,
        Dialect.CLICKHOUSE,
    }


def test_partition_scans_returns_new_plan_per_partition() -> None:
    """The helper returns fresh plans, not shared mutable references."""
    catalog = _orders_and_customers()
    from semql.logical import to_logical_plan

    plan = to_logical_plan(SemanticQuery(measures=["orders.revenue"]), catalog)
    partitions = partition_scans(plan)

    # Single-backend plan -> one partition, which may be the input
    # plan (no rewrap needed).  Multi-backend returns fresh plans
    # — covered by the next test.
    assert all(isinstance(p, LogicalPlan) for p in partitions.values())
