"""Multi-fact symmetric aggregation.

Two or more fact cubes that each carry an additive measure and share a
conformed bridge cube (e.g. ``orders`` and ``reviews`` both joining
``users`` on an identity id) form a "chasm trap": a flat
``orders ⋈ users ⋈ reviews`` join cross-multiplies the rows, so every
COUNT/SUM is inflated by the other fact's matching cardinality.

The compiler answers this fan-safely: each fact is pre-aggregated to the
conformed-dimension grain in its own subquery, the subqueries are FULL
OUTER JOINed on that key, and the bridge cube is joined on
``bridge_key = COALESCE(fact keys)``. Fact-cube filters land inside the
owning subquery; bridge-cube filters land on the outer query.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query, explain_plan
from semql.errors import CompileError
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import Filter, SemanticQuery


def _catalog(dialect: Dialect = Dialect.DUCKDB) -> dict[str, Cube]:
    users = Cube(
        name="users",
        dialect=dialect,
        table="users",
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
        table="orders",
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
        table="reviews",
        alias="r",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="identity_id", sql="{r}.identity_id", type="number")],
        joins=[Join(to="users", relationship="many_to_one", on="{r}.identity_id = {u}.id")],
    )
    return {c.name: c for c in (users, orders, reviews)}


# ---------------------------------------------------------------------------
# Plan detection
# ---------------------------------------------------------------------------


def test_plan_flags_symmetric_for_chasm_query() -> None:
    cat = _catalog()
    plan = explain_plan(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[Filter(dimension="users.name", op="eq", values=["Nikhil"])],
        ),
        cat,
    )
    assert plan.symmetric is not None
    assert {f.cube.name for f in plan.symmetric.facts} == {"orders", "reviews"}
    assert plan.symmetric.bridge.name == "users"


def test_plan_leaves_single_fact_unflagged() -> None:
    cat = _catalog()
    plan = explain_plan(
        SemanticQuery(measures=["orders.count"], dimensions=["users.name"]),
        cat,
    )
    assert plan.symmetric is None


# ---------------------------------------------------------------------------
# Emitted SQL shape
# ---------------------------------------------------------------------------


def test_symmetric_sql_shape() -> None:
    cat = _catalog()
    out = compile_query(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[Filter(dimension="users.name", op="eq", values=["Nikhil"])],
        ),
        cat,
    )
    sql = out.sql
    # Distinct, collision-prefixed output columns — not two identical COUNT(*).
    assert out.columns == ["orders_count", "reviews_count"]
    # Two pre-aggregating subqueries, each GROUP BY the conformed key.
    assert sql.count("GROUP BY") == 2
    assert sql.count("COUNT(*)") == 2
    # Joined fan-safely, bridged on the coalesced key.
    assert "FULL OUTER JOIN" in sql
    assert "COALESCE" in sql
    # Outer references the subquery aggregate columns, not a top-level COUNT.
    assert "_f0.orders_count AS orders_count" in sql
    assert "_f1.reviews_count AS reviews_count" in sql
    # Bridge filter is on the outer query.
    assert "name" in sql and "Nikhil" in out.params.values()


def test_fact_filter_lands_inside_its_subquery() -> None:
    # orders.status filter must constrain rows *before* the COUNT, i.e. live
    # inside the orders subquery — never on the outer query (which would
    # filter an already-aggregated row).
    cat = _catalog()
    out = compile_query(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[
                Filter(dimension="users.name", op="eq", values=["Nikhil"]),
                Filter(dimension="orders.status", op="eq", values=["shipped"]),
            ],
        ),
        cat,
    )
    sql = out.sql
    # The status predicate appears before the FULL OUTER JOIN — i.e. within
    # the first (orders) subquery, not in the trailing outer WHERE.
    status_pos = sql.index("status")
    join_pos = sql.index("FULL OUTER JOIN")
    assert status_pos < join_pos
    assert "shipped" in out.params.values()


def test_three_facts_chain_full_outer_joins() -> None:
    cat = _catalog()
    payments = Cube(
        name="payments",
        dialect=Dialect.DUCKDB,
        table="payments",
        alias="p",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="identity_id", sql="{p}.identity_id", type="number")],
        joins=[Join(to="users", relationship="many_to_one", on="{p}.identity_id = {u}.id")],
    )
    cat["payments"] = payments
    out = compile_query(
        SemanticQuery(measures=["orders.count", "reviews.count", "payments.count"]),
        cat,
    )
    assert out.columns == ["orders_count", "reviews_count", "payments_count"]
    assert out.sql.count("FULL OUTER JOIN") == 2
    assert out.sql.count("GROUP BY") == 3


# ---------------------------------------------------------------------------
# End-to-end correctness — the load-bearing test. 3 orders + 5 reviews for
# one user must report (3, 5), NEVER the (15, 15) cross-product.
# ---------------------------------------------------------------------------


def test_symmetric_aggregation_counts_are_not_inflated() -> None:
    duckdb = pytest.importorskip("duckdb")
    cat = _catalog(Dialect.DUCKDB)
    out = compile_query(
        SemanticQuery(
            measures=["orders.count", "reviews.count"],
            filters=[Filter(dimension="users.name", op="eq", values=["Nikhil"])],
        ),
        cat,
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE users(id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO users VALUES (1, 'Nikhil'), (2, 'Other')")
    con.execute("CREATE TABLE orders(identity_id INTEGER, status VARCHAR)")
    con.execute("INSERT INTO orders VALUES (1,'shipped'),(1,'shipped'),(1,'pending'),(2,'shipped')")
    con.execute("CREATE TABLE reviews(identity_id INTEGER)")
    con.execute("INSERT INTO reviews VALUES (1),(1),(1),(1),(1),(2),(2)")
    rows = con.execute(out.sql, out.params).fetchall()
    # Nikhil (id=1): 3 orders, 5 reviews — not 3*5=15.
    assert rows == [(3, 5)]


def test_symmetric_aggregation_full_outer_keeps_one_sided_identities() -> None:
    # A user with reviews but no orders still appears, with a NULL/absent
    # orders count — the FULL OUTER JOIN preserves both sides.
    duckdb = pytest.importorskip("duckdb")
    cat = _catalog(Dialect.DUCKDB)
    out = compile_query(
        SemanticQuery(measures=["orders.count", "reviews.count"], dimensions=["users.name"]),
        cat,
    )
    con = duckdb.connect()
    con.execute("CREATE TABLE users(id INTEGER, name VARCHAR)")
    con.execute("INSERT INTO users VALUES (1, 'Nikhil'), (2, 'ReviewerOnly')")
    con.execute("CREATE TABLE orders(identity_id INTEGER, status VARCHAR)")
    con.execute("INSERT INTO orders VALUES (1,'shipped'),(1,'shipped')")
    con.execute("CREATE TABLE reviews(identity_id INTEGER)")
    con.execute("INSERT INTO reviews VALUES (1),(2),(2),(2)")
    result = {row[0]: (row[1], row[2]) for row in con.execute(out.sql, out.params).fetchall()}
    assert result["Nikhil"] == (2, 1)
    # ReviewerOnly: 0/None orders, 3 reviews — present despite no orders.
    assert result["ReviewerOnly"][1] == 3


# ---------------------------------------------------------------------------
# Refusals — shapes the symmetric path doesn't handle stay refused, never
# silently emit a cross-product.
# ---------------------------------------------------------------------------


def test_non_additive_measure_in_chasm_is_not_symmetric() -> None:
    # An avg measure isn't handled by symmetric aggregation; the plan must
    # not flag it (it falls back to the non-symmetric path).
    cat = _catalog()
    cat["orders"] = cat["orders"].model_copy(
        update={
            "measures": [
                Measure(name="count", sql="*", agg="count", unit="count"),
                Measure(name="avg_id", sql="{o}.identity_id", agg="avg", unit="count"),
            ]
        }
    )
    plan = explain_plan(
        SemanticQuery(measures=["orders.avg_id", "reviews.count"]),
        cat,
    )
    assert plan.symmetric is None


def test_fact_dimension_in_chasm_refuses() -> None:
    # A dimension on a fact cube (not the bridge) is out of scope for v1
    # symmetric aggregation, so the chasm guard refuses rather than emit.
    cat = _catalog()
    with pytest.raises(CompileError):
        compile_query(
            SemanticQuery(
                measures=["orders.count", "reviews.count"],
                dimensions=["orders.status"],
            ),
            cat,
        )
