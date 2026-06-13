"""Merge-SQL filter values bind as parameters, never inline literals.

The federation merge runs in-process DuckDB over fragment result sets.
Cross-partition ``where`` residuals and ``having`` terms used to be
hand-built with ``_lit()`` (single-quote doubling only) — the one place
in core that inlined values instead of binding them. ``Filter.values``
in a text-to-SQL chatbot are LLM/user-derived, so the invariant
"identity values bind as parameters, never as literals" must hold here
too. These tests pin that: the merge SQL carries ``$name`` placeholders
and ``MergePlan.params`` carries the values.
"""

from __future__ import annotations

from collections.abc import Mapping

import duckdb
from semql import (
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.spec import BoolExpr
from semql_engine import AdapterResult, Engine


def _orders(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
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


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_orders(), _customers())}


def test_cross_partition_filter_value_binds_as_param() -> None:
    """A cross-partition OR residual lands in the merge WHERE as a bound
    parameter; the raw value never appears as an inline literal."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.tier", op="eq", values=["gold"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog)
    merge_sql = plan.merge.sql
    # The cross-partition leaf 'gold' lives on the customers (non-primary)
    # partition, so it survives into the merge WHERE.
    assert "tier" in merge_sql
    # It must be a bound parameter, not an inline literal.
    assert "'gold'" not in merge_sql
    assert "gold" in plan.merge.params.values()


def test_cross_partition_in_filter_binds_each_value() -> None:
    """An ``IN`` list on a cross-partition dim binds every member."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.tier", op="in", values=["gold", "silver"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog)
    assert "'gold'" not in plan.merge.sql
    assert "'silver'" not in plan.merge.sql
    assert "gold" in plan.merge.params.values()
    assert "silver" in plan.merge.params.values()


def test_cross_partition_injection_value_is_inert() -> None:
    """A filter value carrying SQL metacharacters is treated as data:
    it matches no row and cannot widen the result set."""
    catalog = _catalog()
    malicious = "gold' OR '1'='1"
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["__never__"]),
                Filter(dimension="customers.tier", op="eq", values=[malicious]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog)
    assert malicious not in plan.merge.sql

    pg_con = duckdb.connect(":memory:")
    pg_con.execute(
        "CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)"
    )
    pg_con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), (2, 11, 'paid', 50.0), (3, 12, 'paid', 300.0)"
    )
    bq_con = duckdb.connect(":memory:")
    bq_con.execute("CREATE TABLE customers (id INTEGER, region TEXT, tier TEXT)")
    bq_con.execute(
        "INSERT INTO customers VALUES "
        "(10, 'EU', 'gold'), (11, 'US', 'silver'), (12, 'EU', 'silver')"
    )

    class _Adapter:
        def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
            self._con = con

        def execute(self, sql: str, params: Mapping[str, object]) -> AdapterResult:
            cur = self._con.execute(sql, params)
            return AdapterResult(columns=[d[0] for d in cur.description], rows=cur.fetchall())

    engine = Engine()
    engine.register(Dialect.POSTGRES, _Adapter(pg_con))
    engine.register(Dialect.BIGQUERY, _Adapter(bq_con))
    result = engine.run(plan)
    # The injection string matches no tier and status '__never__' matches
    # nothing either, so the OR yields zero rows — injection inert.
    assert result.rows == []


def test_having_term_value_binds_as_param() -> None:
    """raw_rows-mode HAVING binds its threshold as a parameter."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        having=[Filter(dimension="orders.revenue", op="gt", values=[1234])],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    assert "HAVING" in plan.merge.sql
    assert "1234" not in plan.merge.sql
    assert 1234 in plan.merge.params.values()
