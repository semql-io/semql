"""Cross-backend semi-join, end-to-end.

The user case: "active time per employee, but only for employees in the Sales
department." ``active_secs`` lives on one backend (BigQuery here), the
``employees`` directory on another (Postgres). Instead of joining across the
boundary, the Sales employee ids are read from Postgres and shipped to
BigQuery as a literal ``IN`` list.

Two in-memory DuckDBs stand in for the two backends, exactly as
``test_engine.py`` / ``test_cross_backend_symmetric.py`` do.
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
    SemiJoin,
    compile_semi_join_query,
)
from semql_engine import (
    AdapterResult,
    DuckDBAdapter,
    Engine,
    ExecutionContractError,
    ExecutionLimits,
    run_semi_join,
)


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


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    """Postgres stand-in: the employees directory."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE employees (id INTEGER, dept TEXT)")
    # Ana + Cay in Sales; Bob in Ops.
    con.execute("INSERT INTO employees VALUES (1, 'Sales'), (2, 'Ops'), (3, 'Sales')")
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    """BigQuery stand-in: the activity fact."""
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE activity (id INTEGER, employee_id INTEGER, secs DOUBLE)")
    # Ana(1)=100+200=300, Bob(2)=50, Cay(3)=5.
    con.execute(
        "INSERT INTO activity VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 50.0), (4, 3, 5.0)"
    )
    return con


def _engine(pg: duckdb.DuckDBPyConnection, bq: duckdb.DuckDBPyConnection) -> Engine:
    engine = Engine()
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq))
    return engine


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_activity_cube(), _employees_cube())}


def _semi_join_query(op: str = "in") -> SemanticQuery:
    return SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[
            SemiJoin(
                dimension="activity.employee_id",
                op=op,  # type: ignore[arg-type]
                select="employees.id",
                source=SemanticQuery(
                    dimensions=["employees.id"],
                    filters=[Filter(dimension="employees.dept", op="eq", values=["Sales"])],
                ),
            )
        ],
    )


def test_semi_join_restricts_outer_to_inner_value_list(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """Only Sales employees (1, 3) survive; Bob (2) is excluded."""
    plan = compile_semi_join_query(_semi_join_query(), _catalog())
    result = run_semi_join(plan, _engine(pg_con, bq_con))
    assert result.columns == ["employee_id", "active_secs"]
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {1: 300.0, 3: 5.0}  # no employee 2 (Ops)


def test_semi_join_not_in_excludes_inner_value_list(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """not_in Sales -> only the Ops employee (2) survives."""
    plan = compile_semi_join_query(_semi_join_query(op="not_in"), _catalog())
    result = run_semi_join(plan, _engine(pg_con, bq_con))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {2: 50.0}


def test_semi_join_empty_in_short_circuits_to_zero_rows(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """No employee matches the inner filter -> in (empty) -> zero rows,
    without compiling an invalid ``IN ()``."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[
            SemiJoin(
                dimension="activity.employee_id",
                select="employees.id",
                source=SemanticQuery(
                    dimensions=["employees.id"],
                    filters=[Filter(dimension="employees.dept", op="eq", values=["Nonexistent"])],
                ),
            )
        ],
    )
    plan = compile_semi_join_query(q, _catalog())
    result = run_semi_join(plan, _engine(pg_con, bq_con))
    assert result.columns == ["employee_id", "active_secs"]
    assert result.rows == []


def test_semi_join_empty_not_in_is_a_noop(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    """not_in (empty) restricts nothing -> every employee survives."""
    q = SemanticQuery(
        measures=["activity.active_secs"],
        dimensions=["activity.employee_id"],
        semi_joins=[
            SemiJoin(
                dimension="activity.employee_id",
                op="not_in",
                select="employees.id",
                source=SemanticQuery(
                    dimensions=["employees.id"],
                    filters=[Filter(dimension="employees.dept", op="eq", values=["Nonexistent"])],
                ),
            )
        ],
    )
    plan = compile_semi_join_query(q, _catalog())
    result = run_semi_join(plan, _engine(pg_con, bq_con))
    rows = {r[0]: r[1] for r in result.rows}
    assert rows == {1: 300.0, 2: 50.0, 3: 5.0}


def test_nested_semi_join_runs_reuse_single_admission_slot(
    pg_con: duckdb.DuckDBPyConnection, bq_con: duckdb.DuckDBPyConnection
) -> None:
    engine = Engine(limits=ExecutionLimits(max_in_flight=1, admission_timeout=0.1))
    engine.register(Dialect.POSTGRES, _DialectTranslatingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    plan = compile_semi_join_query(_semi_join_query(), _catalog())
    result = run_semi_join(plan, engine)
    assert {row[0] for row in result.rows} == {1, 3}


def test_semi_join_execution_timeout_is_shared_across_stages(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr("semql_engine.execution.time", SimpleNamespace(monotonic=lambda: clock[0]))
    calls = 0

    class AdvancingAdapter(_DialectTranslatingAdapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            nonlocal calls
            calls += 1
            result = super().execute(sql, params)
            clock[0] += 0.6
            return result

    engine = Engine(limits=ExecutionLimits(execution_timeout=1.0))
    engine.register(Dialect.POSTGRES, AdvancingAdapter(pg_con))
    engine.register(Dialect.BIGQUERY, AdvancingAdapter(bq_con))
    plan = compile_semi_join_query(_semi_join_query(), _catalog())
    with pytest.raises(ExecutionContractError) as error:
        run_semi_join(plan, engine)
    assert error.value.reason == "execution_deadline_exceeded"
    assert calls == 2


def test_later_missing_outer_adapter_fails_before_semi_join_selection(
    pg_con: duckdb.DuckDBPyConnection,
) -> None:
    calls: list[str] = []

    class CountingAdapter(_DialectTranslatingAdapter):
        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(sql)
            return super().execute(sql, params)

    engine = Engine(adapters={Dialect.POSTGRES: CountingAdapter(pg_con)})
    plan = compile_semi_join_query(_semi_join_query(), _catalog())
    with pytest.raises(ExecutionContractError) as error:
        run_semi_join(plan, engine)
    assert error.value.reason == "adapter_missing"
    assert calls == []


def _filter_only_foreign_binding_plan() -> tuple[Any, dict[Dialect, duckdb.DuckDBPyConnection]]:
    connections: dict[Dialect, duckdb.DuckDBPyConnection] = {}
    orders = duckdb.connect(":memory:")
    orders.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DOUBLE)")
    orders.execute("INSERT INTO orders VALUES (1, 10, 12.0), (2, 20, 8.0)")
    connections[Dialect.POSTGRES] = orders

    customers = duckdb.connect(":memory:")
    customers.execute("CREATE TABLE customers (id INTEGER)")
    customers.execute("INSERT INTO customers VALUES (10), (20)")
    connections[Dialect.BIGQUERY] = customers

    selected = duckdb.connect(":memory:")
    selected.execute("CREATE TABLE selected (id INTEGER)")
    selected.execute("INSERT INTO selected VALUES (10)")
    connections[Dialect.DUCKDB] = selected

    orders_cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            )
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )
    customers_cube = Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )
    selected_cube = Cube(
        name="selected",
        dialect=Dialect.DUCKDB,
        table="selected",
        alias="s",
        dimensions=[Dimension(name="id", sql="{s}.id", type="number")],
    )
    query = SemanticQuery(
        measures=["orders.revenue"],
        semi_joins=[
            SemiJoin(
                dimension="customers.id",
                select="selected.id",
                source=SemanticQuery(dimensions=["selected.id"]),
            )
        ],
    )
    return compile_semi_join_query(
        query,
        {cube.name: cube for cube in (orders_cube, customers_cube, selected_cube)},
    ), connections


def test_filter_only_foreign_binding_missing_adapter_fails_before_any_io() -> None:
    plan, connections = _filter_only_foreign_binding_plan()
    calls: list[Dialect] = []

    class CountingAdapter(_DialectTranslatingAdapter):
        def __init__(self, dialect: Dialect, connection: duckdb.DuckDBPyConnection) -> None:
            super().__init__(connection)
            self.dialect = dialect

        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            calls.append(self.dialect)
            return super().execute(sql, params)

    engine = Engine(
        adapters={
            Dialect.POSTGRES: CountingAdapter(Dialect.POSTGRES, connections[Dialect.POSTGRES]),
            Dialect.DUCKDB: CountingAdapter(Dialect.DUCKDB, connections[Dialect.DUCKDB]),
        }
    )
    with pytest.raises(ExecutionContractError) as error:
        run_semi_join(plan, engine)
    assert error.value.reason == "adapter_missing"
    assert calls == []


def test_filter_only_foreign_binding_runs_when_every_adapter_is_available() -> None:
    plan, connections = _filter_only_foreign_binding_plan()
    engine = Engine(
        adapters={
            dialect: _DialectTranslatingAdapter(connection)
            for dialect, connection in connections.items()
        }
    )
    result = run_semi_join(plan, engine)
    assert result.columns == ["revenue"]
    assert result.rows == [(12.0,)]
