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
    CompareWindow,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from syrupy.assertion import SnapshotAssertion


def _orders_catalog() -> Catalog:
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
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.cust_id = {c}.id")],
    )
    customers = Cube(
        name="customers",
        backend=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )
    return Catalog([orders, customers])


def _sessions_catalog() -> Catalog:
    sessions = Cube(
        name="sessions",
        backend=Dialect.CLICKHOUSE,
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


def test_snap_compare_previous_period(snapshot: SnapshotAssertion, context: dict[str, str]) -> None:
    out = _orders_catalog().compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                range=("2026-01-01", "2026-02-01"),
            ),
            compare=CompareWindow(),
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
        backend=Dialect.POSTGRES,
        table="events",
        alias="e",
        tenancy="discriminator",
        tenancy_column="tenant_id",
        security_sql="{e}.team_id = {ctx.team_id}",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{e}.region", type="string")],
    )
    out = Catalog([events]).compile(
        SemanticQuery(measures=["events.count"], dimensions=["events.region"]),
        context={"tenant": "acme", "ctx.team_id": "growth"},
    )
    assert out.sql == snapshot
