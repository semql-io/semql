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

Adding a case: append one string to ``CASES``, run with
``--snapshot-update``, review the new ``.ambr`` block. That's the whole
loop — write hundreds this way.

(A row-level execution oracle — run on seeded DuckDB, assert rows — is
deliberately deferred here as too slow; the SQL snapshot is the contract.)
"""

from __future__ import annotations

import pytest
from semql import Catalog, Cube, Dialect, Dimension, Measure, TimeDimension
from semql.parse import parse_sql_statement
from syrupy.assertion import SnapshotAssertion


def _catalog() -> Catalog:
    """A small single-cube catalog the fixtures are written against.

    Single cube keeps the semantic SQL unambiguous (the parser is
    single-cube today); grow this — or add cubes + joins — as the
    parser's surface grows.
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
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
    )
    return Catalog([orders])


CATALOG = _catalog()
CONTEXT = {"schema": "prod"}


# Each entry is a semantic-SQL fixture. Append freely — one line per case.
CASES: list[str] = [
    # --- projection + grouping ---
    "SELECT region, SUM(revenue) FROM orders GROUP BY region",
    "SELECT region, status, SUM(revenue) FROM orders GROUP BY region, status",
    # --- SELECT aliases relabel the output column ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region",
    # --- ORDER BY: by alias, by qualified measure, by dimension ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region ORDER BY rev DESC",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region ORDER BY orders.revenue DESC",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region ORDER BY region ASC",
    # --- COUNT(*) ---
    "SELECT COUNT(*) FROM orders",
    "SELECT region, COUNT(*) AS n FROM orders GROUP BY region ORDER BY n DESC",
    # --- WHERE: comparison, IN, OR-tree, IS NULL ---
    "SELECT region, SUM(revenue) FROM orders WHERE status = 'paid' GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders WHERE region IN ('EMEA', 'APAC') GROUP BY region",
    "SELECT region, SUM(revenue) FROM orders"
    " WHERE status = 'paid' OR region = 'EMEA' GROUP BY region",
    # --- HAVING / LIMIT / OFFSET ---
    "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region HAVING SUM(revenue) > 1000",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region LIMIT 10",
    "SELECT region, SUM(revenue) FROM orders GROUP BY region LIMIT 10 OFFSET 20",
]


@pytest.mark.parametrize("sql", CASES, ids=lambda s: s)
def test_sql_fixture_compiles_to_snapshot(sql: str, snapshot: SnapshotAssertion) -> None:
    decision = parse_sql_statement(sql, CATALOG.as_dict(), strict=True)
    assert decision.parse_errors == (), decision.parse_errors
    out = CATALOG.compile(decision.query, context=CONTEXT)
    assert out.sql == snapshot
