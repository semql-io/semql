"""Cross-backend symmetric aggregation, end-to-end.

The user case: "compare active time vs hours worked per employee", where
``active_secs`` lives on one backend (BigQuery here) and ``hours`` on another
(Postgres), both conformed to a shared ``employees`` bridge. Each fact is
pre-aggregated to the employee grain on its own backend, then LEFT-joined to
the employee universe at the merge — so counts can't fan out, and an employee
present on only one side still appears (NULL on the missing measure).

Two in-memory DuckDBs stand in for the two backends, exactly as
``test_engine.py`` does; the catalog says they're different dialects so the
federated compiler emits one fragment per fact plus the bridge.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import duckdb
import pytest
from semql import Cube, Dialect, Dimension, Join, Measure, SemanticQuery, compile_federated_query
from semql.federate import FederatedPlan
from semql_engine import AdapterResult, DuckDBAdapter, Engine


class _DialectTranslatingAdapter:
    """Rewrites Postgres ``%(name)s`` / BigQuery ``@name`` placeholders to
    DuckDB ``$name`` so one in-memory DuckDB stands in for each backend."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._inner = DuckDBAdapter(connection)

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        sql = re.sub(r"%\((\w+)\)s", r"$\1", sql)
        sql = re.sub(r"@(\w+)", r"$\1", sql)
        return self._inner.execute(sql, params)


def _activity_cube(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="activity",
        dialect=dialect,
        table="activity",
        alias="a",
        primary_key="id",
        measures=[Measure(name="active_secs", sql="{a}.secs", agg="sum", unit="duration")],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(
                name="employee_id", sql="{a}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{a}.employee_id = {e}.id")],
    )


def _worklog_cube(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="worklog",
        dialect=dialect,
        table="worklog",
        alias="w",
        primary_key="id",
        measures=[Measure(name="hours", sql="{w}.hours", agg="sum", unit="duration")],
        dimensions=[
            Dimension(name="id", sql="{w}.id", type="number"),
            Dimension(
                name="employee_id", sql="{w}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{w}.employee_id = {e}.id")],
    )


def _employees_cube(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="employees",
        dialect=dialect,
        table="employees",
        alias="e",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{e}.id", type="number"),
            Dimension(name="name", sql="{e}.name", type="string"),
        ],
    )


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    """Postgres stand-in: holds the employees bridge + the worklog fact."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE employees (id INTEGER, name TEXT)")
    con.execute("INSERT INTO employees VALUES (1, 'Ana'), (2, 'Bob'), (3, 'Cay')")
    con.execute("CREATE TABLE worklog (id INTEGER, employee_id INTEGER, hours DOUBLE)")
    # Ana=8, Cay=5; Bob has no worklog row.
    con.execute("INSERT INTO worklog VALUES (1, 1, 8.0), (2, 3, 5.0)")
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    """BigQuery stand-in: holds the activity fact."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE activity (id INTEGER, employee_id INTEGER, secs DOUBLE)")
    # Ana=100+200=300, Bob=50; Cay has no activity row.
    con.execute("INSERT INTO activity VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 50.0)")
    return con


def _engine(pg: duckdb.DuckDBPyConnection, bq: duckdb.DuckDBPyConnection) -> Engine:
    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq))
    return engine


def test_cross_backend_two_measures_per_employee(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """active_secs (BQ) + hours (PG) per employee — no fan-out, every
    employee present, NULL where one side is missing."""
    catalog = {c.name: c for c in (_activity_cube(), _worklog_cube(), _employees_cube())}
    plan = compile_federated_query(
        SemanticQuery(
            measures=["activity.active_secs", "worklog.hours"],
            dimensions=["employees.name"],
        ),
        catalog,
    )
    assert isinstance(plan, FederatedPlan)
    # One fragment per fact + the bridge.
    assert len(plan.fragments) == 3

    result = _engine(pg_con, bq_con).run(plan)
    assert result.columns == ["name", "active_secs", "hours"]
    rows = {r[0]: (r[1], r[2]) for r in result.rows}
    assert rows == {
        "Ana": (300.0, 8.0),  # both sides
        "Bob": (50.0, None),  # activity only — still present
        "Cay": (None, 5.0),  # worklog only — still present
    }


def test_cross_backend_symmetric_does_not_inflate(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """Ana has 2 activity rows and 1 worklog row; a naive join would
    cross-multiply. Pre-aggregation per fact keeps active=300, hours=8."""
    catalog = {c.name: c for c in (_activity_cube(), _worklog_cube(), _employees_cube())}
    plan = compile_federated_query(
        SemanticQuery(
            measures=["activity.active_secs", "worklog.hours"],
            dimensions=["employees.name"],
        ),
        catalog,
    )
    result = _engine(pg_con, bq_con).run(plan)
    ana = next(r for r in result.rows if r[0] == "Ana")
    assert ana == ("Ana", 300.0, 8.0)  # not 300*1 / 8*2 cross-product
