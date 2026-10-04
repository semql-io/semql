"""Snapshot tests for emitted SQL (``syrupy``).

The assertion here is "the output didn't drift," not "the output
matches this exact pattern." Substring assertions in ``test_compile``
catch shape regressions; these catch silent formatting drift across
sqlglot upgrades, dialect tweaks, and strategy refactors.

To accept a deliberate change, run pytest with ``--snapshot-update``
and review the diff in ``tests/__snapshots__/``.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    compile_federated_query,
)
from syrupy.assertion import SnapshotAssertion


def _orders_catalog() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
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
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.cust_id = {c}.id")],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )
    return Catalog([orders, customers])


def _sessions_catalog() -> Catalog:
    sessions = Cube(
        name="sessions",
        dialect=Dialect.CLICKHOUSE,
        table="sessions",
        alias="s",
        base_predicate="{s}.event_type = 'active'",
        measures=[
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="duration", sql="{s}.duration_sec", agg="sum", unit="duration"),
            Measure(
                name="unique_users",
                sql="{s}.user_id",
                agg="count_distinct",
                unit="count",
                non_additive=True,
            ),
        ],
        dimensions=[Dimension(name="app_name", sql="{s}.app_name", type="string")],
        time_dimensions=[TimeDimension(name="started_at", sql="{s}.started_at")],
    )
    return Catalog([sessions])


@pytest.fixture
def context() -> dict[str, str]:
    return {"schema": "prod"}


# ---------------------------------------------------------------------------
# Representative shapes across the compiler's surface
# ---------------------------------------------------------------------------


def test_snap_simple_pg_aggregation(snapshot: SnapshotAssertion, context: dict[str, str]) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_filtered_aggregation(snapshot: SnapshotAssertion, context: dict[str, str]) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[
                Filter(dimension="orders.status", op="in", values=["paid", "pending"]),
            ],
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_time_breakdown_with_granularity(
    snapshot: SnapshotAssertion, context: dict[str, str]
) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(
            measures=["orders.count"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_join_with_having_and_order(
    snapshot: SnapshotAssertion, context: dict[str, str]
) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.name"],
            having=[Filter(dimension="revenue", op="gt", values=[1000])],
            order=[("revenue", "desc")],
            limit=10,
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_ungrouped_row_listing(snapshot: SnapshotAssertion, context: dict[str, str]) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(
            dimensions=["orders.region", "orders.status"],
            ungrouped=True,
            limit=50,
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_clickhouse_time_truncation(snapshot: SnapshotAssertion) -> None:
    out = _sessions_catalog().compile(
        SemanticQuery(
            measures=["sessions.count"],
            time_dimension=TimeWindow(
                dimension="sessions.started_at",
                granularity="hour",
                range=("2026-01-01", "2026-01-02"),
            ),
        ),
    )
    assert out.sql == snapshot


def test_snap_clickhouse_contains_filter(snapshot: SnapshotAssertion) -> None:
    out = _sessions_catalog().compile(
        SemanticQuery(
            measures=["sessions.count"],
            filters=[Filter(dimension="sessions.app_name", op="contains", values=["chrome"])],
        ),
    )
    assert out.sql == snapshot


def test_snap_count_distinct_non_additive_measure(snapshot: SnapshotAssertion) -> None:
    out = _sessions_catalog().compile(
        SemanticQuery(measures=["sessions.unique_users"], dimensions=["sessions.app_name"]),
    )
    assert out.sql == snapshot


def test_snap_tenancy_discriminator(snapshot: SnapshotAssertion) -> None:
    events = Cube(
        name="events",
        dialect=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_columns=["tenant_id"],
        security_sql="{e}.team_id = {ctx.team_id}",
        security_ctx_keys=["team_id"],
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )
    out = Catalog([events]).compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        context={"tenant": "acme", "ctx.team_id": "growth"},
    )
    assert out.sql == snapshot


# ---------------------------------------------------------------------------
# Multi-fact symmetric aggregation — the chasm-trap-safe emit path. Two/three
# fact cubes sharing a conformed bridge are pre-aggregated per fact and FULL
# OUTER JOINed on the key, never cross-multiplied.
# ---------------------------------------------------------------------------


def _chasm_catalog(dialect: Dialect = Dialect.POSTGRES) -> Catalog:
    users = Cube(
        name="users",
        dialect=dialect,
        table="{schema}.users",
        alias="u",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{u}.id", type="number"),
            Dimension(name="name", sql="{u}.name", type="string"),
        ],
    )
    orders = Cube(
        name="orders",
        dialect=dialect,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="identity_id", sql="{o}.identity_id", type="number"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[Join(to="users", relationship="many_to_one", on="{o}.identity_id = {u}.id")],
    )
    reviews = Cube(
        name="reviews",
        dialect=dialect,
        table="{schema}.reviews",
        alias="r",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="identity_id", sql="{r}.identity_id", type="number")],
        joins=[Join(to="users", relationship="many_to_one", on="{r}.identity_id = {u}.id")],
    )
    payments = Cube(
        name="payments",
        dialect=dialect,
        table="{schema}.payments",
        alias="p",
        measures=[Measure(name="total", sql="{p}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="identity_id", sql="{p}.identity_id", type="number")],
        joins=[Join(to="users", relationship="many_to_one", on="{p}.identity_id = {u}.id")],
    )
    return Catalog([users, orders, reviews, payments])


def test_snap_symmetric_two_fact(snapshot: SnapshotAssertion, context: dict[str, str]) -> None:
    out = _chasm_catalog().compile(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[Filter(dimension="users.name", op="eq", values=["Nikhil"])],
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_symmetric_with_bridge_dimension(
    snapshot: SnapshotAssertion, context: dict[str, str]
) -> None:
    out = _chasm_catalog().compile(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            dimensions=["users.name"],
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_symmetric_fact_local_filter(
    snapshot: SnapshotAssertion, context: dict[str, str]
) -> None:
    # The orders.status filter must render inside the orders subquery; the
    # users.name filter on the outer query.
    out = _chasm_catalog().compile(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[
                Filter(dimension="users.name", op="eq", values=["Nikhil"]),
                Filter(dimension="orders.status", op="eq", values=["shipped"]),
            ],
        ),
        context=context,
    )
    assert out.sql == snapshot


def test_snap_symmetric_three_fact_mixed_agg(
    snapshot: SnapshotAssertion, context: dict[str, str]
) -> None:
    # count + count + sum across three facts → two FULL OUTER JOINs and a
    # 3-arg COALESCE bridge key.
    out = _chasm_catalog().compile(
        SemanticQuery(measures=["orders.count", "reviews.count", "payments.total"]),
        context=context,
    )
    assert out.sql == snapshot


# ---------------------------------------------------------------------------
# Federation — a cross-backend query whose foreign cube enters only via a
# filter. Snapshot every fragment's SQL so a routing regression surfaces.
# ---------------------------------------------------------------------------


def test_snap_federation_filter_only_fragments(snapshot: SnapshotAssertion) -> None:
    orders = Cube(
        name="orders",
        dialect=Dialect.CLICKHOUSE,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="user_id", sql="{o}.user_id", type="number")],
        joins=[Join(to="users", relationship="many_to_one", on="{o}.user_id = {u}.id")],
    )
    users = Cube(
        name="users",
        dialect=Dialect.POSTGRES,
        table="{schema}.users",
        alias="u",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{u}.id", type="number"),
            Dimension(name="name", sql="{u}.name", type="string"),
        ],
    )
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="users.name", op="eq", values=["Nikhil"])],
        ),
        {c.name: c for c in (orders, users)},
        context={"schema": "prod"},
    )
    rendered = "\n---\n".join(f"[{f.dialect.value}] {f.sql}" for f in plan.fragments)
    assert rendered == snapshot
