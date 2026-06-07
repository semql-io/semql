"""End-to-end tests for semql-engine.

We spin up two in-memory DuckDB databases (acting as Postgres and
BigQuery for the sake of the test — both speak DuckDB, but the
catalogue says they're different backends so the federated compiler
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
    Backend,
    Cube,
    Dimension,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
    compile_query,
)
from semql.federate import FederatedPlan, MergePlan
from semql_engine import AdapterResult, DuckDBAdapter, Engine, EngineError

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
# Catalogue + fixtures: orders on "Postgres", customers on "BigQuery".
# Both run in DuckDB in the test.
# ---------------------------------------------------------------------------


def _orders_cube(backend: Backend = Backend.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
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


def _customers_cube(backend: Backend = Backend.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        backend=backend,
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
    engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))

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
    engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))

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
    engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))

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
    engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))

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
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con))

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
    engine.register(Backend.POSTGRES, DuckDBAdapter(pg_con))
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
    engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))

    r1 = engine.run(plan)
    r2 = engine.run(plan)
    assert r1.rows == r2.rows


# ---------------------------------------------------------------------------
# MergePlan + Compiled sanity (would catch a regression in the plan IR
# that broke executor assumptions).
# ---------------------------------------------------------------------------


def test_merge_plan_has_required_attributes() -> None:
    catalog = _catalog(_orders_cube())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    # Single-source path through plain compile_query also feeds the
    # executor when wrapped manually; not the usual path but verifies
    # the IR shapes are decoupled cleanly.
    compiled = compile_query(q, catalog)
    plan = FederatedPlan(
        fragments=[compiled],
        merge=MergePlan(sql="SELECT * FROM frag_0"),
        columns=compiled.columns,
        column_meta=compiled.column_meta,
    )
    assert plan.merge.sql.startswith("SELECT")
