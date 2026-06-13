# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""SQL → SemanticQuery parser (Parser role in the prompt pipeline).

Adds a fifth role to the four-role pipeline (Router / Generator /
Presenter / Drilldown) so LLM agents can write queries in familiar
SQL syntax while still benefiting from SemQL's semantic layer
(catalog resolution, auth, row-level scope).

Spec at ``docs/specs/sql-parser.md``. Pure function:
``parse_sql_statement(statement, catalog=None, *, strict=True) ->
ParserDecision``. The catalog is optional — without it, references
are collected as raw strings; with it, they're validated against
the catalog and unknown references produce diagnostics.

Operator mapping:
  = → eq, != / <> → neq, IN / NOT IN, >, >=, <, <=, LIKE → contains,
  IS NULL / IS NOT NULL.

Constructs: SELECT / FROM (cube inferred from refs) / WHERE
(implicit AND; parenthetical OR → ``where: BoolExpr``) / GROUP BY /
HAVING / ORDER BY / LIMIT / OFFSET / BETWEEN → ``time_dimension`` /
``COMPARE TO prior_period`` → ``compare``.

Strict mode (default) raises on unknown refs; lenient collects
warnings and continues.
"""

from __future__ import annotations

import pytest
from semql.parse import ParserDecision, parse_sql_statement
from semql.spec import SemanticQuery

# ---------------------------------------------------------------------------
# Basic SELECT
# ---------------------------------------------------------------------------


def test_simple_select_with_measure_and_dimension() -> None:
    """A basic SELECT builds measures and dimensions lists."""
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders GROUP BY region",
        catalog=None,
    )
    assert isinstance(out, ParserDecision)
    assert isinstance(out.query, SemanticQuery)
    # SUM(amount) — we don't yet know if "amount" is a measure; the
    # parser doesn't enforce without a catalog.
    assert "region" in out.query.dimensions or "region" in out.resolved_references


def test_select_columns_go_to_dimensions_when_no_agg() -> None:
    """Bare column names (no aggregate function) become dimensions."""
    out = parse_sql_statement(
        "SELECT region, status FROM orders",
        catalog=None,
    )
    q = out.query
    # Bare columns (no aggregate) become dimensions, emitted as
    # qualified ``cube.field`` refs (the compiler's required form).
    assert "orders.region" in q.dimensions
    assert "orders.status" in q.dimensions


def test_aggregated_columns_become_measures() -> None:
    """Aggregated columns (SUM / COUNT / AVG / MIN / MAX) become measures."""
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders GROUP BY region",
        catalog=None,
    )
    q = out.query
    # SUM(amount) is a measure; region is a dimension. Both qualified.
    assert "orders.amount" in q.measures
    assert "orders.region" in q.dimensions


# ---------------------------------------------------------------------------
# SELECT aliases (``AS x``) + COUNT(*)
# ---------------------------------------------------------------------------


def test_select_measure_alias_captured(catalog: dict) -> None:
    """``SUM(revenue) AS rev`` records ``rev -> orders.revenue`` in
    ``aliases`` so the compiler relabels the output column."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region",
        catalog=catalog,
    )
    q = out.query
    assert "orders.revenue" in q.measures
    assert q.aliases == {"rev": "orders.revenue"}
    assert out.parse_errors == ()


def test_order_by_select_alias_uses_alias_key(catalog: dict) -> None:
    """Ordering by a SELECT alias emits the bare alias key — not a
    bogus qualified field (``orders.rev``). The compiler resolves the
    alias key against the output columns."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) AS rev FROM orders GROUP BY region ORDER BY rev DESC",
        catalog=catalog,
    )
    q = out.query
    assert q.aliases == {"rev": "orders.revenue"}
    assert ("rev", "desc") in q.order
    assert ("orders.rev", "desc") not in q.order


def test_count_star_maps_to_count_measure(catalog: dict) -> None:
    """``COUNT(*)`` maps to the cube's count measure (``agg=count``,
    ``sql=*``) instead of being silently dropped."""
    out = parse_sql_statement("SELECT COUNT(*) FROM orders", catalog=catalog)
    q = out.query
    assert "orders.count" in q.measures
    assert out.parse_errors == ()


def test_count_star_with_alias_captured(catalog: dict) -> None:
    """``COUNT(*) AS n`` maps to the count measure and records the alias."""
    out = parse_sql_statement(
        "SELECT region, COUNT(*) AS n FROM orders GROUP BY region",
        catalog=catalog,
    )
    q = out.query
    assert "orders.count" in q.measures
    assert q.aliases == {"n": "orders.count"}


def test_having_over_aggregate_becomes_measure_filter(catalog: dict) -> None:
    """``HAVING SUM(revenue) > 1000`` unwraps the aggregate to a filter
    on the measure — it must not be silently dropped."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders GROUP BY region HAVING SUM(revenue) > 1000",
        catalog=catalog,
    )
    q = out.query
    assert any(
        f.dimension == "orders.revenue" and f.op == "gt" and f.values == [1000]
        for f in q.having
    ), q.having


# ---------------------------------------------------------------------------
# WHERE
# ---------------------------------------------------------------------------


def test_where_eq_becomes_filter() -> None:
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders WHERE status = 'paid' GROUP BY region",
        catalog=None,
    )
    q = out.query
    # With a cube name in the FROM clause, dimensions get prefixed
    # to ``cube.field``.
    assert any(f.dimension == "orders.status" and f.op == "eq" for f in q.filters)


def test_where_in_becomes_in_filter() -> None:
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders WHERE region IN ('emea', 'west') GROUP BY region",
        catalog=None,
    )
    q = out.query
    assert any(f.dimension == "orders.region" and f.op == "in" for f in q.filters)


def test_where_gt_becomes_gt_filter() -> None:
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders WHERE amount > 100 GROUP BY region",
        catalog=None,
    )
    q = out.query
    assert any(f.dimension == "orders.amount" and f.op == "gt" for f in q.filters)


def test_where_is_null_becomes_is_null_filter() -> None:
    out = parse_sql_statement(
        "SELECT region FROM orders WHERE amount IS NULL",
        catalog=None,
    )
    q = out.query
    assert any(f.op == "is_null" for f in q.filters)


def test_where_or_becomes_bool_expr() -> None:
    """A parenthetical OR goes to ``where: BoolExpr``."""
    out = parse_sql_statement(
        "SELECT region FROM orders WHERE (status = 'paid' OR region = 'emea')",
        catalog=None,
    )
    q = out.query
    assert q.where is not None
    # Top-level is OR.
    assert q.where.op == "or"


# ---------------------------------------------------------------------------
# ORDER BY, LIMIT, OFFSET
# ---------------------------------------------------------------------------


def test_order_by_parses_direction() -> None:
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders GROUP BY region ORDER BY SUM(amount) DESC",
        catalog=None,
    )
    q = out.query
    assert any(ref == "orders.amount" and direction == "desc" for ref, direction in q.order)


def test_limit_parses() -> None:
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders GROUP BY region LIMIT 25",
        catalog=None,
    )
    q = out.query
    assert q.limit == 25


def test_offset_parses() -> None:
    out = parse_sql_statement(
        "SELECT region FROM orders LIMIT 10 OFFSET 5",
        catalog=None,
    )
    q = out.query
    assert q.limit == 10
    assert q.offset == 5


# ---------------------------------------------------------------------------
# BETWEEN → time_dimension
# ---------------------------------------------------------------------------


def test_between_becomes_time_window() -> None:
    """``WHERE dim BETWEEN 'a' AND 'b'`` → ``time_dimension: TimeWindow``."""
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders "
        "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' "
        "GROUP BY region",
        catalog=None,
    )
    q = out.query
    assert q.time_dimension is not None
    assert q.time_dimension.dimension == "orders.created_at"
    assert q.time_dimension.range == ("2026-01-01", "2026-03-31")


# ---------------------------------------------------------------------------
# Strict / lenient modes
# ---------------------------------------------------------------------------


def test_unknown_cube_in_strict_mode_raises(catalog: dict) -> None:
    """Strict mode (default) fails on unknown cubes when a catalog is provided."""
    from semql.parse import ParseError

    with pytest.raises(ParseError):
        parse_sql_statement(
            "SELECT region FROM nonexistent",
            catalog=catalog,
            strict=True,
        )


def test_unknown_cube_in_lenient_mode_collects_warning(catalog: dict) -> None:
    """Lenient mode collects the unknown-cube error in ``parse_errors``
    rather than raising — the caller decides how to surface it."""
    out = parse_sql_statement(
        "SELECT region FROM nonexistent",
        catalog=catalog,
        strict=False,
    )
    assert any("nonexistent" in e for e in out.parse_errors)


def test_catalog_aware_resolution_validates_measures(catalog: dict) -> None:
    """With a catalog, known field references are validated."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders GROUP BY region",
        catalog=catalog,
    )
    q = out.query
    # ``revenue`` is a measure on orders; the parser identified it
    # as a measure. ``region`` is a dimension. Both qualified.
    assert "orders.revenue" in q.measures
    assert "orders.region" in q.dimensions
    # No unknown-cube errors.
    assert out.parse_errors == ()


# ---------------------------------------------------------------------------
# ParserDecision shape
# ---------------------------------------------------------------------------


def test_parser_decision_carries_original_statement() -> None:
    """The original SQL string is preserved on the decision for reference."""
    sql = "SELECT region FROM orders"
    out = parse_sql_statement(sql, catalog=None)
    assert out.original_statement == sql


def test_parser_decision_resolved_references() -> None:
    """``resolved_references`` maps the parsed ref to its canonical form."""
    out = parse_sql_statement(
        "SELECT region, SUM(amount) FROM orders GROUP BY region",
        catalog=None,
    )
    # Resolved references maps the bare ref to ``cube.field``.
    refs = out.resolved_references
    assert refs.get("region") == "orders.region"


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


def test_compare_to_prior_period_via_hint() -> None:
    """A ``/*+ COMPARE prior_period */`` hint populates the ``compare`` field."""
    out = parse_sql_statement(
        "SELECT /*+ COMPARE prior_period */ region, SUM(amount) "
        "FROM orders "
        "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' "
        "GROUP BY region",
        catalog=None,
    )
    q = out.query
    assert q.compare is not None
    assert q.compare.mode == "previous_period"
