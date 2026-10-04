"""SQL-fixture compiler tests: semantic SQL → SemanticQuery → physical SQL.

Author a terse *semantic* SQL string (identifiers are catalog names);
the harness parses it into a ``SemanticQuery`` and compiles it. The
emitted *physical* SQL is captured as a ``syrupy`` snapshot — and the
snapshot **is the hand-reviewed oracle**: on a deliberate change, run
``uv run pytest --snapshot-update`` and eyeball the ``.ambr`` diff.

This is NOT a round-trip identity check. Input (catalog-name SQL) and
output (physical, dialect-rendered SQL) differ by design — the point is
to exercise the *compiler* over many shapes, cheaply, with an oracle
that's independent of both the parser and the compiler.

The goal is breadth: a fixture for every supported SQL path — every
WHERE operator (including the negated ``NOT IN`` / ``IS NOT NULL``
forms), every aggregate family, every ORDER BY / HAVING / GROUP BY /
LIMIT shape, and the Malloy-style multi-cube JOIN. Adding a case:
append one string to ``CASES``, run with ``--snapshot-update``, review
the new ``.ambr`` block.

Several of these fixtures are regression locks for compiler/parser bugs
found by exactly this stress test — e.g. ordering by an *unprojected*
measure once emitted the raw column (``ORDER BY o.amount``) instead of
its aggregate, and ``NOT IN`` / ``IS NOT NULL`` / ``MEDIAN(...)`` were
silently dropped. The snapshot is what catches a regression of those.

Comparison uses seeded DuckDB execution to assert independent numeric,
NULL-group, and alias expectations instead of repinning rendered SQL wording.
"""

from __future__ import annotations

import duckdb
import pytest
from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, TimeDimension
from semql.parse import parse_sql_statement
from syrupy.assertion import SnapshotAssertion


def _catalog() -> Catalog:
    """The catalog the fixtures are written against.

    ``orders`` (the many side) joins to ``customers`` and ``products``
    (the one side) so Malloy-style ``FROM orders o JOIN customers c``
    queries can be authored — the parser uses the JOIN only to learn
    which cubes participate; the compiler derives the actual join from
    this catalog definition (the SQL ON clause is not read).

    The measures span the aggregate families (sum / count / avg / min /
    max / count_distinct / median) so a fixture can stress how each is
    rendered in SELECT, ORDER BY, and HAVING. The SQL function wrapping a
    measure (``SUM(...)``, ``MEDIAN(...)``) is only a *marker* that the
    column is a measure; the measure's catalog ``agg`` decides the
    rendered aggregate.
    """
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
            Measure(name="max_amount", sql="{o}.amount", agg="max", unit="currency"),
            Measure(name="min_amount", sql="{o}.amount", agg="min", unit="currency"),
            Measure(
                name="unique_customers",
                sql="{o}.customer_id",
                agg="count_distinct",
                non_additive=True,
            ),
            Measure(
                name="median_amount",
                sql="{o}.amount",
                agg="median",
                unit="currency",
                non_additive=True,
            ),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
            Dimension(name="is_paid", sql="{o}.is_paid", type="bool"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
        joins=[
            # Multiple joins of multiple relationship types fan out from
            # ``orders`` (the central fact): two many_to_one (customers,
            # products), a one_to_many (order_items, the fan-out source),
            # and a one_to_one (shipments).
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
            Join(to="products", relationship="many_to_one", on="{o}.product_id = {p}.id"),
            Join(to="order_items", relationship="one_to_many", on="{o}.id = {i}.order_id"),
            Join(to="shipments", relationship="one_to_one", on="{o}.id = {sh}.order_id"),
        ],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="{schema}.customers",
        alias="c",
        dimensions=[
            Dimension(name="name", sql="{c}.name", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
        # Chain: orders → customers → regions (a 3-cube transitive path).
        joins=[Join(to="regions", relationship="many_to_one", on="{c}.region_id = {rg}.id")],
    )
    regions = Cube(
        name="regions",
        dialect=Dialect.POSTGRES,
        table="{schema}.regions",
        alias="rg",
        dimensions=[Dimension(name="name", sql="{rg}.name", type="string")],
    )
    products = Cube(
        name="products",
        dialect=Dialect.POSTGRES,
        table="{schema}.products",
        alias="p",
        dimensions=[
            Dimension(name="category", sql="{p}.category", type="string"),
            Dimension(name="sku", sql="{p}.sku", type="string"),
        ],
    )
    order_items = Cube(
        name="order_items",
        dialect=Dialect.POSTGRES,
        table="{schema}.order_items",
        alias="i",
        measures=[Measure(name="qty", sql="{i}.quantity", agg="sum", unit="count")],
        dimensions=[Dimension(name="sku", sql="{i}.sku", type="string")],
    )
    shipments = Cube(
        name="shipments",
        dialect=Dialect.POSTGRES,
        table="{schema}.shipments",
        alias="sh",
        dimensions=[Dimension(name="carrier", sql="{sh}.carrier", type="string")],
    )
    return Catalog([orders, customers, regions, products, order_items, shipments])


CATALOG = _catalog()
CONTEXT = {"schema": "prod"}


# Each entry is a semantic-SQL fixture. Append freely — one line per case.
CASES: list[str] = [
    # --- projection + grouping ---
    "SELECT region, SUM(revenue) FROM orders GROUP BY region",
    "SELECT region, status, SUM(revenue) FROM orders GROUP BY region, status",
    # dimensions only (no aggregate) — distinct rows, no GROUP BY aggregate
    "SELECT region, status FROM orders",
    # --- SELECT aliases relabel the output column (measure + dimension) ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region",
    "SELECT region AS r, SUM(revenue) AS rev FROM orders GROUP BY region",
    # --- aggregate families: each measure's catalog agg drives rendering ---
    "SELECT region, AVG(avg_amount) AS avg_amt FROM orders GROUP BY region",
    "SELECT region, MIN(min_amount), MAX(max_amount) FROM orders GROUP BY region",
    "SELECT region, COUNT(unique_customers) AS uniq FROM orders GROUP BY region",
    "SELECT region, MEDIAN(median_amount) AS med FROM orders GROUP BY region",
    # several measures at once
    "SELECT region, SUM(revenue) AS rev, COUNT(*) AS n, AVG(avg_amount) AS a"
    " FROM orders GROUP BY region",
    # --- COUNT(*) ---
    "SELECT COUNT(*) FROM orders",
    "SELECT region, COUNT(*) AS n FROM orders GROUP BY region ORDER BY n DESC",
    # --- ORDER BY: by alias, by qualified measure, by dimension, multi-key ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region ORDER BY rev DESC",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region ORDER BY orders.revenue DESC",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region ORDER BY region ASC",
    "SELECT region, status, SUM(revenue) AS rev FROM orders GROUP BY region, status"
    " ORDER BY rev DESC, region ASC",
    # ORDER BY an *unprojected* measure — must emit the aggregate, not the raw
    # column (regression lock: this once emitted ``ORDER BY o.amount``).
    "SELECT region FROM orders GROUP BY region ORDER BY SUM(revenue) DESC",
    "SELECT region FROM orders GROUP BY region ORDER BY MEDIAN(median_amount) DESC",
    # ORDER BY an unprojected COUNT(*) — resolves to the row-count measure
    # (regression lock: COUNT(*)'s inner is ``*``, once dropped the key).
    "SELECT region FROM orders GROUP BY region ORDER BY COUNT(*) DESC",
    # two measures in ORDER BY, opposite directions
    "SELECT region, SUM(revenue) AS rev, MIN(min_amount) AS lo FROM orders"
    " GROUP BY region ORDER BY rev DESC, lo ASC",
    # --- WHERE: every comparison operator ---
    "SELECT region, SUM(revenue) FROM orders WHERE status = 'paid' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE status != 'paid' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE amount > 100 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE amount >= 100 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE amount < 100 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE amount <= 100 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE region IN ('EMEA', 'APAC') GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE region NOT IN ('EMEA', 'APAC') GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE region LIKE 'EM%' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE status IS NULL GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE status IS NOT NULL GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE is_paid = true GROUP BY region",
    # NOT-wrapped comparisons — negate the inner op rather than drop it.
    "SELECT region, SUM(revenue) FROM orders WHERE NOT (status = 'paid') GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE NOT amount > 100 GROUP BY region",
    # --- WHERE structure: implicit AND, OR-tree, nested AND/OR ---
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE status = 'paid' AND amount > 100 GROUP BY region",
    # AND of three predicates; and a two-sided numeric range (two filters)
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE status = 'paid' AND region = 'EMEA' AND amount > 5 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE amount > 10 AND amount < 100 GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE status = 'paid' OR region = 'EMEA' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE status = 'paid' AND (region = 'EMEA' OR region = 'APAC') GROUP BY region",
    # --- BETWEEN → time window (alone, and with another filter) ---
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' AND status = 'paid'"
    " GROUP BY region",
    # --- HAVING: single, AND of two, on COUNT, on a different measure ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region HAVING SUM(revenue) > 1000",
    "SELECT region, SUM(revenue) AS rev, COUNT(*) AS n FROM orders GROUP BY region"
    " HAVING SUM(revenue) > 1000 AND COUNT(*) > 5",
    "SELECT region, COUNT(*) AS n FROM orders GROUP BY region HAVING COUNT(*) >= 10",
    # --- LIMIT / OFFSET ---
    "SELECT region, SUM(revenue) FROM orders GROUP BY region LIMIT 10",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region LIMIT 10 OFFSET 20",
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region ORDER BY rev DESC LIMIT 5",
    # --- Malloy-style JOIN: aggregate the many side (orders) by a dimension
    #     on the one side (customers / products). The ON clause is ignored;
    #     the compiler derives the join from the catalog. ---
    "SELECT c.name, SUM(o.revenue) AS rev FROM orders o"
    " JOIN customers c ON o.customer_id = c.id GROUP BY c.name ORDER BY rev DESC",
    "SELECT c.tier, COUNT(*) AS n FROM orders o"
    " JOIN customers c ON o.customer_id = c.id WHERE o.status = 'paid' GROUP BY c.tier",
    "SELECT p.category, AVG(o.avg_amount) AS a FROM orders o"
    " JOIN products p ON o.product_id = p.id GROUP BY p.category",
    # JOIN with filter on a joined-cube dimension + ORDER BY a measure
    "SELECT c.tier, SUM(o.revenue) AS rev FROM orders o"
    " JOIN customers c ON o.customer_id = c.id WHERE c.tier = 'gold'"
    " GROUP BY c.tier ORDER BY rev DESC",
    # three cubes (star): orders aggregated by dimensions from two one-sides
    "SELECT c.tier, p.category, SUM(o.revenue) AS rev FROM orders o"
    " JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id"
    " GROUP BY c.tier, p.category ORDER BY rev DESC",
    # --- multiple join TYPES + chains ---
    # three-cube CHAIN (orders → customers → regions): the middle cube
    # carries the transit even though no customers column is selected; both
    # ON clauses come from the catalog joins, not the SQL.
    "SELECT rg.name, SUM(o.revenue) AS rev FROM orders o"
    " JOIN customers c ON o.customer_id = c.id JOIN regions rg ON c.region_id = rg.id"
    " GROUP BY rg.name ORDER BY rev DESC",
    # chain with a filter + COUNT(*) on the far cube's dimension
    "SELECT rg.name, COUNT(*) AS n FROM orders o"
    " JOIN customers c ON o.customer_id = c.id JOIN regions rg ON c.region_id = rg.id"
    " WHERE o.status = 'paid' GROUP BY rg.name",
    # one_to_one join (orders → shipments): no fan-out, plain INNER JOIN
    "SELECT sh.carrier, SUM(o.revenue) AS rev FROM orders o"
    " JOIN shipments sh ON o.id = sh.order_id GROUP BY sh.carrier ORDER BY rev DESC",
    # four cubes at once: two many_to_one + one one_to_one off the fact
    "SELECT c.tier, p.category, sh.carrier, SUM(o.revenue) AS rev FROM orders o"
    " JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id"
    " JOIN shipments sh ON o.id = sh.order_id GROUP BY c.tier, p.category, sh.carrier",
]


@pytest.mark.parametrize("sql", CASES, ids=lambda s: s)
def test_sql_fixture_compiles_to_snapshot(sql: str, snapshot: SnapshotAssertion) -> None:
    decision = parse_sql_statement(sql, CATALOG.as_dict(), strict=True)
    assert decision.parse_errors == (), decision.parse_errors
    out = CATALOG.compile(decision.query, context=CONTEXT)
    assert out.sql == snapshot


def test_comparison_sql_alias_preserves_null_group_identity_and_numeric_outputs() -> None:
    catalog = Catalog(
        [
            Cube(
                name="orders",
                table="orders",
                alias="o",
                dialect=Dialect.DUCKDB,
                measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
                dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
                time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
            )
        ]
    )
    decision = parse_sql_statement(
        "SELECT /*+ COMPARE prior_period */ region, SUM(revenue) AS rev FROM orders "
        "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' GROUP BY region",
        catalog.as_dict(),
        strict=True,
    )
    compiled = catalog.compile(decision.query)
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE orders(region VARCHAR, amount INTEGER, created_at TIMESTAMP)"
        )
        connection.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(None, 5, "2026-01-10"), (None, 2, "2025-12-15")],
        )
        cursor = connection.execute(compiled.sql, compiled.params)
        assert [column[0] for column in cursor.description] == [
            "region",
            "rev_current",
            "rev_prior",
            "rev_delta",
            "rev_pct_change",
        ]
        assert cursor.fetchall() == [(None, 5, 2, 3, 150.0)]
