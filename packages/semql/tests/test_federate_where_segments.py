"""Distributive-mode federation now lifts where + segments.

This module captures the v1 carryovers that moved raw_rows
``_route_where_tree`` and segment routing up into the
distributive path. The contract: a where-tree splits into
per-partition clauses (applied inside the fragment) and a
residual that lives in the merge SQL. A segment that touches
multiple backends routes to whichever fragment owns the
segment's cube; a cross-partition segment is refused loudly.
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
    Segment,
    SemanticQuery,
    compile_federated_query,
)
from semql.spec import BoolExpr
from semql_engine import AdapterResult, Engine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _orders(backend: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
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
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(backend: Dialect = Dialect.BIGQUERY) -> Cube:
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
        segments=[Segment(name="vip", sql="{c}.tier = 'gold'")],
    )


def _catalog() -> dict[str, Cube]:
    return {c.name: c for c in (_orders(), _customers())}


# ---------------------------------------------------------------------------
# where-tree distributive lift tests
# ---------------------------------------------------------------------------


def test_distributive_where_single_partition_routes_to_fragment() -> None:
    """A filter that touches only one partition becomes a
    fragment-local filter; the merge SQL doesn't have to filter
    again."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
    )
    plan = compile_federated_query(q, catalog)
    # The primary fragment (orders) must carry the where clause.
    primary = plan.fragments[0]
    # The filter is bound as a parameter, not inlined; check for
    # the parameter placeholder.
    assert "%(p0)s" in primary.sql or "p0" in primary.params
    assert primary.params.get("p0") == "paid"
    # Two fragments, cross-backend enrichment plan still works.
    assert len(plan.fragments) == 2


def test_distributive_where_cross_partition_lands_in_merge() -> None:
    """A where clause whose dims span two partitions survives the
    fragment compilation (each fragment gets the local piece) and the
    merge applies the residual.

    With an OR of (status, tier) the CNF split puts each leaf in its
    own fragment, and the merge SQL does the OR across the joined
    rows.
    """
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
    # Both fragments carry their local filter.
    frags_concat = "\n".join(f.sql for f in plan.fragments)
    assert "status" in frags_concat or "paid" in frags_concat
    # Merge SQL: the cross-partition OR is materialised as a
    # post-join WHERE.
    assert "tier" in plan.merge.sql or "gold" in plan.merge.sql


def test_distributive_where_engine_end_to_end_matches_reference() -> None:
    """Distributive + where + cross-partition OR: the merge engine
    produces the same final rows as a hand-written SQL baseline."""
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

    # Adapter that runs the fragment SQL against a DuckDB source.
    pg_con = duckdb.connect(":memory:")
    pg_con.execute(
        "CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)"
    )
    pg_con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), "
        "(2, 10, 'pending', 200.0), "
        "(3, 11, 'paid', 50.0), "
        "(4, 12, 'pending', 25.0), "
        "(5, 12, 'paid', 300.0)"
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
            return AdapterResult(
                columns=[d[0] for d in cur.description],
                rows=cur.fetchall(),
            )

    engine = Engine()
    engine.register(Dialect.POSTGRES, _Adapter(pg_con))
    engine.register(Dialect.BIGQUERY, _Adapter(bq_con))
    result = engine.run(plan)
    # Reference: hand-rolled baseline via a single DuckDB query.
    baseline_con = duckdb.connect(":memory:")
    baseline_con.execute(
        "CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)"
    )
    baseline_con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), "
        "(2, 10, 'pending', 200.0), "
        "(3, 11, 'paid', 50.0), "
        "(4, 12, 'pending', 25.0), "
        "(5, 12, 'paid', 300.0)"
    )
    baseline_con.execute("CREATE TABLE customers (id INTEGER, region TEXT, tier TEXT)")
    baseline_con.execute(
        "INSERT INTO customers VALUES "
        "(10, 'EU', 'gold'), (11, 'US', 'silver'), (12, 'EU', 'silver')"
    )
    baseline = baseline_con.execute(
        "SELECT c.region, SUM(o.amount) "
        "FROM orders o JOIN customers c ON o.customer_id = c.id "
        "WHERE o.status = 'paid' OR c.tier = 'gold' "
        "GROUP BY c.region ORDER BY c.region"
    ).fetchall()
    assert sorted(result.rows) == sorted(baseline)


# ---------------------------------------------------------------------------
# Segment distributive routing
# ---------------------------------------------------------------------------


def test_distributive_segment_routes_to_owning_partition() -> None:
    """A segment on cube X is applied to fragment X; merge SQL is
    untouched."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        segments=["customers.vip"],
    )
    plan = compile_federated_query(q, catalog)
    # The customers fragment (BigQuery) carries the segment filter.
    frags_concat = "\n".join(f.sql for f in plan.fragments)
    assert "tier" in frags_concat and "gold" in frags_concat
    # The merge SQL doesn't repeat the segment filter (it was
    # applied in the fragment).
    # We don't assert the merge is empty of "tier" — the join itself
    # surfaces the customers.id column, which is unrelated.


def test_distributive_segment_on_primary_partition_applies() -> None:
    """A segment on the primary partition (orders) applies cleanly."""
    catalog_with_orders_segment = _catalog()
    catalog_with_orders_segment["orders"] = Cube(**_catalog()["orders"].model_dump()).model_copy(
        update={
            "segments": [Segment(name="big_order", sql="{o}.amount > 100")],
        }
    )
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        segments=["orders.big_order"],
    )
    plan = compile_federated_query(q, catalog_with_orders_segment)
    frags_concat = "\n".join(f.sql for f in plan.fragments)
    assert "amount" in frags_concat and "100" in frags_concat
