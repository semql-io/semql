"""C12 (ktx-ports M1) — aggregates inside a subquery must not be collected
as top-level measures.

The SQL→SemanticQuery parser walks projection expressions to find
aggregate calls. ``Expression.walk()`` descends into nested SELECTs, so a
scalar-subquery projection like ``(SELECT avg(amount) FROM t)`` had its
inner ``avg`` mis-classified as a top-level measure of the outer query —
silently, producing a wrong SemanticQuery. Aggregate collection must stop
at the subquery boundary.
"""

from __future__ import annotations

import pytest
from semql.model import Cube, Dialect, Dimension, Measure
from semql.parse import parse_sql_statement


@pytest.fixture
def cat() -> dict[str, Cube]:
    return {
        "t": Cube(
            name="t",
            backend=Dialect.POSTGRES,
            table="s.t",
            alias="t",
            measures=[
                Measure(name="avg_amount", sql="{t}.amount", agg="avg", unit="number"),
                Measure(name="n", sql="*", agg="count", unit="count"),
            ],
            dimensions=[
                Dimension(name="r", sql="{t}.r", type="string"),
                Dimension(name="amount", sql="{t}.amount", type="number"),
            ],
        )
    }


def test_aggregate_in_projection_subquery_not_collected(cat: dict[str, Cube]) -> None:
    # The only aggregate is inside a subquery → it is NOT a top-level measure.
    d = parse_sql_statement("SELECT r, (SELECT avg(amount) FROM t) FROM t", cat, strict=False)
    assert d.query.measures == [], d.query.measures
    # And the unsupported subquery is surfaced, not silently dropped.
    assert d.parse_warnings or d.parse_errors


def test_top_level_aggregate_still_collected(cat: dict[str, Cube]) -> None:
    # Control: a genuine top-level aggregate is still detected.
    d = parse_sql_statement("SELECT r, avg(amount) FROM t GROUP BY r", cat, strict=False)
    assert d.query.measures != []
