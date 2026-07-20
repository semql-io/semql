"""Round-trip tests for the ``SemanticQuery`` → semantic-SQL serializer.

The contract (see :mod:`semql.unparse`) is::

    parse_sql_statement(query_to_sql(q, catalog), catalog).query == q

for every parser-canonical query shape. Equality is exact model ``==`` on the
re-parsed query — the serializer mirrors the parser's own normalisation, so a
query the parser emits round-trips to itself.

The broad-coverage arm reuses the semantic-SQL fixtures in
``test_sql_roundtrip`` (one string per supported shape). Each fixture is
parsed to a canonical ``SemanticQuery`` — so it is canonical *by construction*
— then serialized and re-parsed; the two queries must be equal. This keeps the
serializer's coverage locked to the parser's supported set: a new parser
fixture automatically becomes a new round-trip case.

A second arm covers shapes ``test_sql_roundtrip`` doesn't (time granularity via
``DATE_TRUNC``, the catalog-free marker path) and the flagged-unsupported
features that must raise rather than silently drop.
"""

from __future__ import annotations

import pytest
from semql.parse import parse_sql_statement
from semql.spec import (
    CompareWindow,
    Filter,
    InlineDerived,
    SemanticQuery,
    TimeWindow,
)
from semql.unparse import UnparseError, query_to_sql

from .test_sql_roundtrip import CASES, CATALOG


def _parse(sql: str) -> SemanticQuery:
    decision = parse_sql_statement(sql, CATALOG.as_dict(), strict=True)
    assert decision.parse_errors == (), decision.parse_errors
    return decision.query


@pytest.mark.parametrize("sql", CASES, ids=lambda s: s)
def test_roundtrip_with_catalog(sql: str) -> None:
    """Every supported semantic-SQL fixture: parse → serialize → parse is a
    fixed point on the canonical ``SemanticQuery``."""
    q = _parse(sql)
    regenerated = query_to_sql(q, CATALOG.as_dict())
    q2 = _parse(regenerated)
    assert q2 == q, f"regenerated SQL: {regenerated!r}"


@pytest.mark.parametrize("sql", CASES, ids=lambda s: s)
def test_roundtrip_without_catalog(sql: str) -> None:
    """The catalog-free serializer emits generic ``SUM(<ref>)`` markers that
    still round-trip: the parser re-derives the real aggregate from the
    catalog it is given at *parse* time, so equality holds against the
    catalog-parsed original."""
    q = _parse(sql)
    regenerated = query_to_sql(q, catalog=None)
    q2 = _parse(regenerated)
    assert q2 == q, f"regenerated SQL: {regenerated!r}"


# ---------------------------------------------------------------------------
# Shapes the semantic-SQL fixtures don't exercise
# ---------------------------------------------------------------------------


def test_roundtrip_time_granularity() -> None:
    """A ``time_dimension.granularity`` serializes to ``DATE_TRUNC`` in SELECT
    (plus the ``BETWEEN`` window) and round-trips."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-04-01"),
        ),
    )
    regenerated = query_to_sql(q, CATALOG.as_dict())
    assert "DATE_TRUNC('month', orders.created_at)" in regenerated
    assert _parse(regenerated) == q


def test_roundtrip_nested_or_and_tree() -> None:
    """An ``(a OR b) AND (c OR d)`` where-tree — a top ``and`` over two ``or``
    children — round-trips with its structure intact."""
    sql = (
        "SELECT region, SUM(revenue) FROM orders "
        "WHERE (region = 'EMEA' OR region = 'APAC') "
        "AND (status = 'paid' OR status = 'shipped') GROUP BY region"
    )
    q = _parse(sql)
    assert q.where is not None and q.where.op == "and"
    assert _parse(query_to_sql(q, CATALOG.as_dict())) == q


def test_roundtrip_offset_and_multi_alias() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        aliases={"rev": "orders.revenue", "r": "orders.region"},
        order=[("rev", "desc")],
        limit=10,
        offset=20,
    )
    assert _parse(query_to_sql(q, CATALOG.as_dict())) == q


def test_generated_sql_uses_true_aggregate_with_catalog() -> None:
    """With a catalog, a measure is wrapped in its declared aggregate and a
    row-count measure emits ``COUNT(*)``."""
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count", "orders.avg_amount"],
        dimensions=["orders.region"],
    )
    sql = query_to_sql(q, CATALOG.as_dict())
    assert "SUM(orders.revenue)" in sql
    assert "AVG(orders.avg_amount)" in sql
    assert "COUNT(*)" in sql


def test_two_row_count_measures_on_one_cube_round_trips_without_bare_count_star() -> None:
    """A cube declaring two row-count measures (``agg='count', sql='*'``)
    makes bare ``COUNT(*)`` ambiguous — referencing either one must emit an
    explicit, unambiguous form that round-trips to the *same* measure, not
    whichever one the parser would otherwise guess first."""
    from semql.model import Cube, Dialect, Measure

    catalog = {
        "orders": Cube(
            name="orders",
            dialect=Dialect.POSTGRES,
            table="{schema}.orders",
            alias="o",
            measures=[
                Measure(name="count", sql="*", agg="count", unit="count"),
                Measure(name="num_rows", sql="*", agg="count", unit="count"),
            ],
        )
    }
    q = SemanticQuery(measures=["orders.num_rows"])
    sql = query_to_sql(q, catalog)
    assert sql.strip() != "SELECT COUNT(*) FROM orders"
    decision = parse_sql_statement(sql, catalog, strict=True)
    assert decision.parse_errors == (), decision.parse_errors
    assert decision.query == q


# ---------------------------------------------------------------------------
# Unsupported features are flagged, never silently dropped
# ---------------------------------------------------------------------------


_UNSUPPORTED: list[tuple[str, SemanticQuery]] = [
    (
        "ungrouped",
        SemanticQuery(dimensions=["orders.region"], ungrouped=True, limit=5),
    ),
    (
        "left_joins",
        SemanticQuery(measures=["orders.revenue"], left_joins=["customers"]),
    ),
    (
        "segments",
        SemanticQuery(measures=["orders.revenue"], segments=["orders.big"]),
    ),
    (
        "derived_measures",
        SemanticQuery(
            derived_measures=[
                InlineDerived(name="x", op="sum", operands=["orders.revenue", "orders.avg_amount"])
            ]
        ),
    ),
    (
        "compare(mode='explicit')",
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at", range=("2026-01-01", "2026-04-01")
            ),
            compare=CompareWindow(mode="explicit", range=("2025-10-01", "2026-01-01")),
        ),
    ),
    (
        "fill_nulls_with",
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="month",
                range=("2026-01-01", "2026-04-01"),
                fill_nulls_with=0,
            ),
        ),
    ),
]


@pytest.mark.parametrize("feature,query", _UNSUPPORTED, ids=[f for f, _ in _UNSUPPORTED])
def test_unsupported_feature_is_flagged(feature: str, query: SemanticQuery) -> None:
    with pytest.raises(UnparseError):
        query_to_sql(query, CATALOG.as_dict())


def test_unqualified_reference_is_flagged() -> None:
    """A bare (unqualified) ref can't name a FROM cube, so it is flagged."""
    q = SemanticQuery(
        measures=["revenue"], filters=[Filter(dimension="region", op="eq", values=["EMEA"])]
    )
    with pytest.raises(UnparseError, match=r"(?i)unqualified"):
        query_to_sql(q)
