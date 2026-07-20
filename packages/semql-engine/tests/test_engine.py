"""End-to-end tests for semql-engine.

We spin up two in-memory DuckDB databases (acting as Postgres and
BigQuery for the sake of the test — both speak DuckDB, but the
catalog says they're different backends so the federated compiler
emits two fragments and a merge SQL). The engine wires both via
``DuckDBAdapter`` and runs the plan end-to-end.

Single-source DuckDB exercises the degenerate path; the federated
tests exercise the multi-fragment materialise-and-merge path.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import duckdb
import pytest
from semql import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.federate import FederatedPlan
from semql_engine import AdapterResult, DuckDBAdapter, Engine, EngineError
from semql_engine.merge import render_merge_sql

# ---------------------------------------------------------------------------
# Test-only adapter: stands in for a non-DuckDB backend.
#
# In real deployments, Postgres / BigQuery fragments run via dedicated
# adapters (psycopg, google-cloud-bigquery). The test environment uses
# DuckDB to stand in for both — but the compiler emits Postgres-style
# ``%(name)s`` and BigQuery-style ``@name`` placeholders, which DuckDB
# doesn't understand. This adapter rewrites them to DuckDB's ``$name``
# form before delegating to a wrapped ``DuckDBAdapter``. Production
# adapters don't need this — they speak their native dialect.
# ---------------------------------------------------------------------------


class _DialectTranslatingAdapter:
    """Rewrites Postgres ``%(name)s`` and BigQuery ``@name`` placeholders
    to DuckDB ``$name`` so a single in-memory DuckDB can stand in for
    multiple backends in tests."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._inner = DuckDBAdapter(connection)

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        sql = re.sub(r"%\((\w+)\)s", r"$\1", sql)  # PG → DuckDB
        sql = re.sub(r"@(\w+)", r"$\1", sql)  # BQ → DuckDB
        return self._inner.execute(sql, params)


# ---------------------------------------------------------------------------
# Catalog + fixtures: orders on "Postgres", customers on "BigQuery".
# Both run in DuckDB in the test.
# ---------------------------------------------------------------------------


def _orders_cube(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="order_count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )


def _customers_cube(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        dialect=dialect,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )


def _catalog(*cubes: Cube) -> dict[str, Cube]:
    return {c.name: c for c in cubes}


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB standing in for the Postgres "orders" source."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), "
        "(2, 10, 'paid', 200.0), "
        "(3, 11, 'paid', 50.0), "
        "(4, 11, 'pending', 25.0), "
        "(5, 12, 'paid', 300.0)"
    )
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB standing in for the BigQuery "customers" source."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE customers (id INTEGER, region TEXT, tier TEXT)")
    con.execute(
        "INSERT INTO customers VALUES (10, 'EU', 'gold'), (11, 'US', 'silver'), (12, 'EU', 'gold')"
    )
    return con


# ---------------------------------------------------------------------------
# Federated execution end-to-end
# ---------------------------------------------------------------------------


def test_engine_runs_two_fragment_plan_and_merges(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """The canonical federated case: fact (revenue) lives on one
    source, dim label (region) on another, joined via customer_id/id."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    assert isinstance(plan, FederatedPlan)

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)

    # Output columns match plan.columns.
    assert result.columns == ["region", "revenue"]
    # EU: 100+200+300=600; US: 50+25=75
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {"EU": 600.0, "US": 75.0}


def test_engine_handles_filter_pushdown_correctly(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """A filter on ``orders.status='paid'`` lands in the fact fragment;
    the dim fragment is untouched. Verify the engine still produces
    correct numbers."""
    from semql.spec import Filter

    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # EU paid: 100+200+300=600; US paid: 50 (25 is pending)
    assert rows == {"EU": 600.0, "US": 50.0}


def test_engine_handles_avg_decomposition(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Avg is decomposed into ``(sum, count)`` in the fact fragment and
    recomposed in the merge SQL. The end-to-end number should match a
    naive ``AVG`` on a non-federated query."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.avg_amount"],
            dimensions=["customers.region"],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # EU: (100+200+300)/3 = 200; US: (50+25)/2 = 37.5
    assert rows == {"EU": 200.0, "US": 37.5}


def test_iter_rows_yields_dicts(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    rows = list(engine.iter_rows(plan))
    assert all(set(r.keys()) == {"region", "revenue"} for r in rows)


# ---------------------------------------------------------------------------
# Single-fragment / degenerate path
# ---------------------------------------------------------------------------


def test_engine_handles_degenerate_single_fragment_plan(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """Single-backend FederatedPlan: one fragment, trivial merge.
    The engine runs the fragment and the pass-through merge correctly."""
    catalog = _catalog(_orders_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.status"],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, DuckDBAdapter(pg_con))

    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # paid: 100+200+50+300=650; pending: 25
    assert rows == {"paid": 650.0, "pending": 25.0}


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_engine_refuses_plan_with_unregistered_backend(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    """The plan references BigQuery; we only registered Postgres.
    EngineError is raised before we run anything that would corrupt
    state."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, DuckDBAdapter(pg_con))
    # BigQuery NOT registered.

    with pytest.raises(EngineError, match="No adapter registered"):
        engine.run(plan)


def test_engine_repeatable_runs_dont_leak_state(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Two ``run()`` calls in a row must produce identical results —
    frag_N tables from the first run must not influence the second."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    r1 = engine.run(plan)
    r2 = engine.run(plan)

    # The merge SQL groups but does not ORDER BY, so DuckDB is free to
    # return rows in any order — comparing order-sensitively made this
    # flaky. The invariant under test is state isolation (no frag_N leak
    # between runs); a leak would change the *multiset* of rows (stale
    # data joined against fresh), which an order-insensitive compare still
    # catches. Sort by a stringified key so mixed/None cells stay sortable.
    def _key(rows: list[tuple[object, ...]]) -> list[tuple[str, ...]]:
        return sorted(tuple(str(c) for c in row) for row in rows)

    assert r1.columns == r2.columns
    assert _key(r1.rows) == _key(r2.rows)


# ---------------------------------------------------------------------------
# MergeSpec + CompiledQuery sanity (would catch a regression in the plan IR
# that broke executor assumptions).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Raw-row follow-ups — filtered / ratio measures, time_dim, where-tree CNF
# ---------------------------------------------------------------------------


def test_engine_runs_raw_rows_with_filtered_measure(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Filtered measures project ``CASE WHEN <filter> THEN <sql> ELSE NULL END``
    at the fragment; the merge's SUM ignores NULLs, so 'paid revenue
    by region' lands exactly as the user would write by hand."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(
                name="paid_revenue",
                sql="{o}.amount",
                agg="sum",
                filter="{o}.status = 'paid'",
            ),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )
    catalog = _catalog(orders, _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.paid_revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
        mode="raw_rows",
    )
    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # EU paid: 100+200+300=600; US paid: 50 (the pending 25 is filtered out).
    assert rows == {"EU": 600.0, "US": 50.0}


def test_engine_runs_raw_rows_with_cross_partition_or(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """``status='paid' OR tier='gold'`` spans backends — raw-row CNF
    routes the disjunction to the merge SQL, which applies it after
    the cross-source JOIN."""
    from semql.spec import BoolExpr, Filter

    catalog = _catalog(_orders_cube(), _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.status", op="eq", values=["paid"]),
                    Filter(dimension="customers.tier", op="eq", values=["gold"]),
                ],
            ),
        ),
        catalog,
        mode="raw_rows",
    )
    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # EU customers (10, 12 are gold) match either branch. Orders 1+2
    # (cust 10, paid) + 5 (cust 12, paid). Both also satisfy tier='gold'.
    # → EU sum = 100+200+300 = 600.
    # US has cust 11 (silver). Orders 3 (paid, $50) matches via status;
    # order 4 (pending, $25) is silver and not paid → drops.
    # → US sum = 50.
    assert rows == {"EU": 600.0, "US": 50.0}


# ---------------------------------------------------------------------------
# Raw-row federation — non-distributive aggs end-to-end
# ---------------------------------------------------------------------------


def test_engine_runs_raw_rows_plan_with_count_distinct(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """In raw-row mode the primary fragment emits row-level
    customer_ids; the merge DuckDB step does the
    ``COUNT(DISTINCT ...)`` after the cross-source join. Distributive
    mode would refuse this query."""
    orders_with_distinct = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(
                name="distinct_customers",
                sql="{o}.customer_id",
                agg="count_distinct",
            ),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )
    catalog = _catalog(orders_with_distinct, _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.distinct_customers"],
            dimensions=["customers.region"],
        ),
        catalog,
        mode="raw_rows",
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # EU customer_ids in the dataset: {10, 12} (two distinct).
    # US customer_ids in the dataset: {11}        (one distinct).
    assert rows == {"EU": 2, "US": 1}


def test_engine_runs_raw_rows_plan_with_having(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """HAVING applied at the merge step against the recomposed
    aggregate filters out regions whose distinct customer count is
    below the threshold."""
    orders_with_distinct = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(
                name="distinct_customers",
                sql="{o}.customer_id",
                agg="count_distinct",
            ),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )
    catalog = _catalog(orders_with_distinct, _customers_cube())
    from semql.spec import Filter

    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.distinct_customers"],
            dimensions=["customers.region"],
            having=[Filter(dimension="orders.distinct_customers", op="gte", values=[2])],
        ),
        catalog,
        mode="raw_rows",
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)
    rows = {r[0]: r[1] for r in result.rows}
    # Only EU (2 distinct) survives the HAVING >= 2.
    assert rows == {"EU": 2}


def test_engine_runs_distributive_plan_with_having(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Distributive HAVING applies at the merge, after re-aggregation.
    Golden-compare: the HAVING plan's rows equal the no-HAVING plan's
    rows filtered in Python."""
    from semql.spec import Filter

    catalog = _catalog(_orders_cube(), _customers_cube())
    base = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    threshold = 100.0
    with_having = base.model_copy(
        update={"having": [Filter(dimension="orders.revenue", op="gte", values=[threshold])]}
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    baseline = engine.run(compile_federated_query(base, catalog))
    golden = {r[0]: r[1] for r in baseline.rows if r[1] >= threshold}
    # Sanity: the threshold actually splits the groups (EU=600, US=75).
    assert golden == {"EU": 600.0}

    result = engine.run(compile_federated_query(with_having, catalog))
    assert {r[0]: r[1] for r in result.rows} == golden


def test_engine_runs_distributive_having_per_time_bucket(
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """With a time dimension, HAVING filters per (dimension, bucket) row —
    a group that clears the threshold on one day can miss it on another."""
    from semql import TimeDimension
    from semql.spec import Filter, TimeWindow

    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )
    pg = duckdb.connect(":memory:")
    pg.execute(
        "CREATE TABLE orders (id INTEGER, customer_id INTEGER, created_at TIMESTAMP, amount DOUBLE)"
    )
    pg.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, '2024-01-01 08:00', 100.0), "  # EU day 1
        "(2, 10, '2024-01-01 09:00', 200.0), "  # EU day 1 → 300 total
        "(3, 10, '2024-01-02 08:00', 50.0), "  # EU day 2 → below threshold
        "(4, 11, '2024-01-01 08:00', 150.0)"  # US day 1 → exactly at threshold
    )
    catalog = _catalog(orders, _customers_cube())
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2024-01-01", "2024-02-01"),
            ),
            having=[Filter(dimension="orders.revenue", op="gte", values=[150])],
        ),
        catalog,
    )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    result = engine.run(plan)

    idx_region = result.columns.index("region")
    idx_time = result.columns.index("created_at_day")
    idx_rev = result.columns.index("revenue")
    rows = {(r[idx_region], str(r[idx_time])[:10]): r[idx_rev] for r in result.rows}
    # EU day 2 (50.0) is filtered out even though EU day 1 passes.
    assert rows == {
        ("EU", "2024-01-01"): 300.0,
        ("US", "2024-01-01"): 150.0,
    }


def test_merge_plan_has_required_attributes() -> None:
    catalog = _catalog(_orders_cube())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    # Single-source path through compile_federated_query also feeds the
    # executor when wrapped manually; not the usual path but verifies
    # the IR shapes are decoupled cleanly. The plan carries the
    # structured spec (no merge SQL — that's rendered in the engine).
    plan = compile_federated_query(q, catalog)
    assert render_merge_sql(plan.merge_spec)[0].startswith("SELECT")


def test_engine_custom_merge_engine_receives_merge_spec(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    from semql import MergeSpec

    catalog = _catalog(_orders_cube(), _customers_cube())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    plan = compile_federated_query(q, catalog)

    seen_specs: list[MergeSpec] = []

    class RecordingMergeEngine:
        def merge(self, fragment_results: list[AdapterResult], spec: MergeSpec) -> AdapterResult:
            seen_specs.append(spec)
            assert len(fragment_results) == 2
            return AdapterResult(columns=plan.columns, rows=[("custom", 123.0)])

    engine = Engine(merge_engine=RecordingMergeEngine())
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    result = engine.run(plan)
    assert seen_specs == [plan.merge_spec]
    assert result.columns == plan.columns
    assert result.rows == [("custom", 123.0)]
