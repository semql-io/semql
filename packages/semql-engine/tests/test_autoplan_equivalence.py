"""Auto-Planner equivalence: the safety net.

The Auto-Planner rewrites a flat cross-source query into an injected semi-join.
The rewrite is sound only if it returns the rows the question actually asks for.
These tests run the auto-planned semi-join path against two in-memory backends
and assert the *correct* answer.

They also pin a finding this safety net surfaced: the existing bridge-merge
federation path returns a **different, wrong** answer for a filter-only foreign
cube — it emits ``LEFT JOIN`` against the bridge fragment, so non-matching fact
rows survive (Ops employee counted under a "Sales only" filter). That is the
documented "filter-only queries bypass federation" gap, and it is precisely what
the semi-join rewrite routes around. ``test_bridge_merge_diverges_*`` is a
strict xfail: when federation is fixed to INNER-JOIN filter-only bridges it will
XPASS and force us to re-converge the two paths.

Backends mirror ``test_semijoin_engine.py``: ``activity`` (BigQuery stand-in)
carries the measure; ``employees`` (Postgres stand-in) is metadata, bridged on
``activity.employee_id = employees.id``.
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
    Filter,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
    compile_semi_join_query,
)
from semql.autoplan import autoplan
from semql_engine import AdapterResult, DuckDBAdapter, Engine, run_semi_join


class _DialectTranslatingAdapter:
    """Rewrites Postgres ``%(name)s`` / BigQuery ``@name`` placeholders to
    DuckDB ``$name`` so one in-memory DuckDB stands in for each backend."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._inner = DuckDBAdapter(connection)

    def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
        sql = re.sub(r"%\((\w+)\)s", r"$\1", sql)
        sql = re.sub(r"@(\w+)", r"$\1", sql)
        return self._inner.execute(sql, params)


def _activity_cube() -> Cube:
    return Cube(
        name="activity",
        dialect=Dialect.BIGQUERY,
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


def _employees_cube() -> Cube:
    return Cube(
        name="employees",
        dialect=Dialect.POSTGRES,
        table="employees",
        alias="e",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{e}.id", type="number"),
            Dimension(name="dept", sql="{e}.dept", type="string"),
        ],
    )


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_activity_cube(), _employees_cube())}


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE employees (id INTEGER, dept TEXT)")
    con.execute("INSERT INTO employees VALUES (1, 'Sales'), (2, 'Ops'), (3, 'Sales')")
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE activity (id INTEGER, employee_id INTEGER, secs DOUBLE)")
    con.execute(
        "INSERT INTO activity VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 50.0), (4, 3, 5.0)"
    )
    return con


def _engine(pg: duckdb.DuckDBPyConnection, bq: duckdb.DuckDBPyConnection) -> Engine:
    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq))
    return engine


_RowMap = dict[tuple[Any, ...], tuple[Any, ...]]


def _bridge_merge_rows(q: SemanticQuery, engine: Engine) -> tuple[_RowMap, list[str]]:
    """Run the flat query through the federation bridge-merge path."""
    plan = compile_federated_query(q, _catalog())
    result = engine.run(plan)
    return {tuple(r[:-1]): tuple(r) for r in result.rows}, list(result.columns)


def _semi_join_rows(q: SemanticQuery, engine: Engine) -> tuple[_RowMap, list[str]]:
    """Auto-plan the flat query into a semi-join and run the staged path."""
    planned = autoplan(q, _catalog())
    assert planned.query.semi_joins, "autoplan should have injected a semi-join"
    plan = compile_semi_join_query(planned.query, _catalog())
    result = run_semi_join(plan, engine)
    return {tuple(r[:-1]): tuple(r) for r in result.rows}, list(result.columns)


def test_grouped_query_semi_join_equals_bridge_merge(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """`active_secs per employee where dept = Sales`, grouped by an
    activity-side dimension. Both paths agree, and on the correct rows: Sales
    employees 1 (300) and 3 (5), Ops employee 2 excluded."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )

    bm_rows, bm_cols = _bridge_merge_rows(q, _engine(pg_con, bq_con))
    sj_rows, sj_cols = _semi_join_rows(q, _engine(pg_con, bq_con))

    assert sj_cols == bm_cols
    assert sj_rows == bm_rows
    assert {k[0]: v[-1] for k, v in sj_rows.items()} == {1: 300.0, 3: 5.0}


def test_aggregate_only_query_semi_join_equals_bridge_merge(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """No grouping dimension — a single total. Both paths agree on the correct
    Sales total: 300 + 5 = 305 (Ops employee 2's 50 excluded). Regression guard
    for Gap C: a filter-only foreign cube must INNER-JOIN at the merge, not
    LEFT (which would leak the Ops 50 and report 355)."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
    )

    bm_rows, bm_cols = _bridge_merge_rows(q, _engine(pg_con, bq_con))
    sj_rows, sj_cols = _semi_join_rows(q, _engine(pg_con, bq_con))

    assert sj_cols == bm_cols
    assert sj_rows == bm_rows
    assert [v[-1] for v in sj_rows.values()] == [305.0]
