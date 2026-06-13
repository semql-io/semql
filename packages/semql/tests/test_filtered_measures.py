"""Tests for ``Measure.filter`` — conditional aggregation.

A filtered measure scopes its aggregation to rows matching an
inline predicate: ``SUM(amount) FILTER (WHERE status = 'paid')``.
Lets one query ask "paid revenue vs pending revenue" without three
round-trips.

The ``filter`` field uses raw SQL with the same ``{alias}``
placeholder convention as ``Segment.sql`` and ``base_predicate`` —
catalog authors write SQL, planners reference the result by name.

sqlglot renders ``exp.Filter`` natively on Postgres / DuckDB /
BigQuery / ClickHouse and transpiles to ``COUNT(IFF(...))`` on
Snowflake — no per-dialect work needed.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
)


def _orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(
                name="paid_count",
                sql="*",
                agg="count",
                filter="{o}.status = 'paid'",
                description="Orders with confirmed payment.",
            ),
            Measure(
                name="paid_revenue",
                sql="{o}.amount",
                agg="sum",
                filter="{o}.status = 'paid'",
            ),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


# ---------------------------------------------------------------------------
# Model — Measure.filter field shape.
# ---------------------------------------------------------------------------


def test_measure_filter_defaults_to_none() -> None:
    m = Measure(name="count", sql="*", agg="count")
    assert m.filter is None


def test_measure_accepts_filter_sql_fragment() -> None:
    m = Measure(name="paid_count", sql="*", agg="count", filter="{o}.status = 'paid'")
    assert m.filter == "{o}.status = 'paid'"


# ---------------------------------------------------------------------------
# Compiler — emit FILTER (WHERE ...) clause.
# ---------------------------------------------------------------------------


def test_filtered_count_emits_filter_where_clause() -> None:
    out = Catalog([_orders()]).compile(SemanticQuery(measures=["orders.paid_count"]))
    assert "COUNT(*)" in out.sql
    assert "FILTER" in out.sql.upper()
    assert "WHERE" in out.sql.upper()
    assert "o.status" in out.sql


def test_filtered_sum_emits_filter_where_clause() -> None:
    out = Catalog([_orders()]).compile(SemanticQuery(measures=["orders.paid_revenue"]))
    assert "SUM(o.amount)" in out.sql
    assert "FILTER" in out.sql.upper()


def test_unfiltered_measure_unchanged() -> None:
    """No filter set → no FILTER clause; existing measures unaffected."""
    out = Catalog([_orders()]).compile(SemanticQuery(measures=["orders.count"]))
    assert "FILTER" not in out.sql.upper()


def test_multiple_filtered_measures_in_one_query() -> None:
    out = Catalog([_orders()]).compile(
        SemanticQuery(measures=["orders.count", "orders.paid_count", "orders.paid_revenue"])
    )
    # Three aggregates, two of them filtered.
    assert out.sql.upper().count("FILTER") == 2


def test_filtered_measure_with_dimension_groupby() -> None:
    out = Catalog([_orders()]).compile(
        SemanticQuery(
            measures=["orders.paid_revenue"],
            dimensions=["orders.region"],
        )
    )
    assert "GROUP BY" in out.sql.upper()
    assert "FILTER" in out.sql.upper()
    assert "o.region" in out.sql


def test_filter_alias_placeholder_resolves() -> None:
    """The {o} in the filter SQL must resolve to the cube's alias —
    same machinery as base_predicate / Join.on / Segment.sql."""
    out = Catalog([_orders()]).compile(SemanticQuery(measures=["orders.paid_count"]))
    assert "{o}" not in out.sql
    assert "o.status" in out.sql


def test_filtered_measure_composes_with_outer_where() -> None:
    """The outer WHERE narrows the input set; the FILTER narrows
    *aggregation* within that set. Both apply, independently."""
    out = Catalog([_orders()]).compile(
        SemanticQuery(
            measures=["orders.paid_revenue"],
            filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
        )
    )
    assert "WHERE" in out.sql.upper()
    assert "FILTER" in out.sql.upper()
    assert "o.region" in out.sql  # outer WHERE
    assert "o.status" in out.sql  # FILTER clause
    assert "us" in out.params.values()


# ---------------------------------------------------------------------------
# Dialect coverage — Snowflake transpiles FILTER to COUNT(IFF(...)).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend", "expected_marker"),
    [
        (Dialect.POSTGRES, "FILTER"),
        (Dialect.CLICKHOUSE, "FILTER"),
        (Dialect.DUCKDB, "FILTER"),
        (Dialect.BIGQUERY, "FILTER"),
        # Snowflake has no native FILTER — sqlglot transpiles to IFF.
        (Dialect.SNOWFLAKE, "IFF"),
    ],
)
def test_filter_renders_per_dialect(backend: Dialect, expected_marker: str) -> None:
    cube = Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="paid_count",
                sql="*",
                agg="count",
                filter="{o}.status = 'paid'",
            ),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    out = Catalog([cube]).compile(SemanticQuery(measures=["orders.paid_count"]))
    assert expected_marker in out.sql.upper()


# ---------------------------------------------------------------------------
# Compare — the filter sits inside each CTE, not the outer JOIN.
# ---------------------------------------------------------------------------


def test_filtered_measure_in_compare_window() -> None:
    from semql import CompareWindow, TimeDimension, TimeWindow

    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="paid_revenue",
                sql="{o}.amount",
                agg="sum",
                filter="{o}.status = 'paid'",
            ),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    out = Catalog([cube]).compile(
        SemanticQuery(
            measures=["orders.paid_revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                range=("2026-01-01T00:00:00", "2026-02-01T00:00:00"),
            ),
            compare=CompareWindow(mode="previous_period"),
        )
    )
    # FILTER appears in both current and prior CTEs.
    assert out.sql.upper().count("FILTER") == 2
    assert "paid_revenue_current" in out.sql
    assert "paid_revenue_prior" in out.sql
    assert "paid_revenue_delta" in out.sql
