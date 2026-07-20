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
from semql.compile import compile_query
from semql.errors import CompileError
from semql.parse import ParseError, ParserDecision, parse_sql_statement
from semql.spec import SemanticQuery

CONTEXT = {"schema": "test_schema"}


# ---------------------------------------------------------------------------
# Multi-cube JOIN (Malloy-style: the JOIN declares participation; the
# catalog is the source of truth for how the cubes relate).
# ---------------------------------------------------------------------------


def test_join_resolves_qualified_columns_to_their_cubes(catalog: dict) -> None:
    """``FROM orders o JOIN customers c`` resolves ``o.revenue`` and
    ``c.name`` to the right cube via the table aliases — producing a
    multi-cube SemanticQuery. The ON clause is not emitted; the catalog
    join is the source of truth."""
    out = parse_sql_statement(
        "SELECT c.name, SUM(o.revenue) FROM orders o "
        "JOIN customers c ON o.customer_id = c.id GROUP BY c.name",
        catalog=catalog,
    )
    q = out.query
    assert "orders.revenue" in q.measures
    assert "customers.name" in q.dimensions
    assert out.parse_errors == ()


def test_join_query_compiles_to_join_sql(catalog: dict) -> None:
    """The parsed multi-cube query compiles — the compiler infers the
    join from the catalog (the parser emitted no join spec)."""
    out = parse_sql_statement(
        "SELECT c.name, SUM(o.revenue) FROM orders o "
        "JOIN customers c ON o.customer_id = c.id GROUP BY c.name",
        catalog=catalog,
    )
    sql = compile_query(out.query, catalog, context=CONTEXT).sql
    assert "JOIN" in sql.upper()
    assert "SUM(o.amount)" in sql


def test_join_unqualified_column_resolves_by_unique_owner(catalog: dict) -> None:
    """An unqualified column in a multi-cube query resolves to the one
    cube that declares it (``name`` only exists on customers here)."""
    out = parse_sql_statement(
        "SELECT name, SUM(o.revenue) FROM orders o "
        "JOIN customers c ON o.customer_id = c.id GROUP BY name",
        catalog=catalog,
    )
    q = out.query
    assert "customers.name" in q.dimensions


def test_join_between_non_joinable_cubes_is_refused(catalog: dict) -> None:
    """Joining cubes with no catalog relationship is refused — the
    parser won't invent a join the catalog doesn't declare."""
    with pytest.raises(Exception, match=r"(?i)join|relate|not.*catalog"):
        parse_sql_statement(
            "SELECT s.app_name, SUM(o.revenue) FROM orders o "
            "JOIN sessions s ON o.x = s.y GROUP BY s.app_name",
            catalog=catalog,
            strict=True,
        )


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
        f.dimension == "orders.revenue" and f.op == "gt" and f.values == [1000] for f in q.having
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
# DATE_TRUNC in SELECT → time_dimension.granularity
# ---------------------------------------------------------------------------


def test_date_trunc_in_select_sets_granularity() -> None:
    """``DATE_TRUNC('<grain>', dim)`` in SELECT + a matching ``BETWEEN`` on
    the same dimension sets ``time_dimension.granularity`` (and does not add
    the time dimension to ``dimensions``)."""
    out = parse_sql_statement(
        "SELECT DATE_TRUNC('month', created_at), SUM(amount) FROM orders "
        "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' "
        "GROUP BY DATE_TRUNC('month', created_at)",
        catalog=None,
    )
    q = out.query
    assert q.time_dimension is not None
    assert q.time_dimension.dimension == "orders.created_at"
    assert q.time_dimension.granularity == "month"
    assert q.time_dimension.range == ("2026-01-01", "2026-03-31")
    # The bucketed time dimension is not a plain grouping dimension.
    assert "orders.created_at" not in q.dimensions


def test_date_trunc_without_matching_between_is_error() -> None:
    """A ``DATE_TRUNC`` bucket with no ``BETWEEN`` window on the same
    dimension is flagged rather than silently dropped."""
    from semql.parse import ParseError

    with pytest.raises(ParseError, match=r"(?i)date_trunc|between|window"):
        parse_sql_statement(
            "SELECT DATE_TRUNC('month', created_at), SUM(amount) FROM orders",
            catalog=None,
        )


def test_two_date_trunc_buckets_same_dimension_conflicting_grains_is_error() -> None:
    """Two ``DATE_TRUNC`` projections on the same dimension with conflicting
    grains must not silently keep the last one — both are discarded in favor
    of a diagnostic."""
    from semql.parse import ParseError

    with pytest.raises(ParseError, match=r"(?i)multiple.*date_trunc"):
        parse_sql_statement(
            "SELECT DATE_TRUNC('month', created_at), DATE_TRUNC('week', created_at), "
            "SUM(amount) FROM orders "
            "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31'",
            catalog=None,
        )


def test_two_date_trunc_buckets_one_matching_one_not_is_error() -> None:
    """A valid bucket followed by a mismatched one must not be discarded in
    favor of a misleading error about the wrong bucket — both are rejected
    with a diagnostic naming the real problem (multiple buckets)."""
    from semql.parse import ParseError

    with pytest.raises(ParseError, match=r"(?i)multiple.*date_trunc"):
        parse_sql_statement(
            "SELECT DATE_TRUNC('month', created_at), DATE_TRUNC('day', started_at), "
            "SUM(amount) FROM orders "
            "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31'",
            catalog=None,
        )


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


# ---------------------------------------------------------------------------
# Invalid queries must FAIL — strict mode raises, lenient mode records a
# ``parse_errors`` entry. These pin the parser's refusal contract: an
# unsupported or unresolvable query never compiles to a wrong-but-silent SQL.
# ---------------------------------------------------------------------------


# (label, sql) — each must surface at least one parse_errors entry (lenient)
# and raise ParseError (strict). Parser-level failures only; compile-level
# refusals (e.g. required_filters) are exercised separately below.
INVALID_PARSE_CASES: list[tuple[str, str]] = [
    ("unknown_cube", "SELECT region FROM nonexistent"),
    ("unknown_dimension", "SELECT bogus_dim FROM orders"),
    ("unknown_measure", "SELECT SUM(bogus) FROM orders"),
    ("unknown_field_in_where", "SELECT region FROM orders WHERE bogus = 1 GROUP BY region"),
    (
        "non_joinable_cubes",
        "SELECT s.app_name, SUM(o.revenue) FROM orders o "
        "JOIN sessions s ON o.x = s.y GROUP BY s.app_name",
    ),
    (
        "ambiguous_unqualified_column",
        "SELECT name, SUM(o.revenue) FROM orders o "
        "JOIN customers c ON o.a = c.b JOIN products p ON o.c = p.d GROUP BY name",
    ),
    ("delete_statement", "DELETE FROM orders"),
    ("update_statement", "UPDATE orders SET region = 'x'"),
    ("not_a_select", "WITH x AS (SELECT 1) INSERT INTO orders VALUES (1)"),
    ("unparseable_garbage", "this is not sql at all !!!"),
    ("subquery_projection", "SELECT (SELECT 1) FROM orders"),
    (
        "having_with_or",
        "SELECT region, COUNT(*) AS n FROM orders GROUP BY region "
        "HAVING COUNT(*) > 5 OR COUNT(*) < 1",
    ),
]


_INVALID_SQLS = [c[1] for c in INVALID_PARSE_CASES]
_INVALID_IDS = [c[0] for c in INVALID_PARSE_CASES]


@pytest.mark.parametrize("sql", _INVALID_SQLS, ids=_INVALID_IDS)
def test_invalid_query_records_parse_error_in_lenient_mode(sql: str, catalog: dict) -> None:
    """Lenient mode collects ≥1 ``parse_errors`` for an invalid query
    instead of silently returning an empty/wrong SemanticQuery."""
    out = parse_sql_statement(sql, catalog=catalog, strict=False)
    assert out.parse_errors, f"expected a parse error for: {sql}"


@pytest.mark.parametrize("sql", _INVALID_SQLS, ids=_INVALID_IDS)
def test_invalid_query_raises_in_strict_mode(sql: str, catalog: dict) -> None:
    """Strict mode (the default) raises ``ParseError`` on an invalid query."""
    with pytest.raises(ParseError):
        parse_sql_statement(sql, catalog=catalog, strict=True)


def test_fan_out_measure_refused_at_compile() -> None:
    """A ``one_to_many`` JOIN that keeps the many side referenced fans out
    the parent's additive measure — the compiler refuses rather than emit
    an over-counting SUM. (Parses fine; the refusal is a compile contract.)"""
    from semql import Catalog, Cube, Dialect, Dimension, Join, Measure

    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[Join(to="order_items", relationship="one_to_many", on="{o}.id = {i}.order_id")],
    )
    order_items = Cube(
        name="order_items",
        dialect=Dialect.POSTGRES,
        table="{schema}.order_items",
        alias="i",
        dimensions=[Dimension(name="sku", sql="{i}.sku", type="string")],
    )
    cat = Catalog([orders, order_items]).as_dict()
    out = parse_sql_statement(
        "SELECT i.sku, SUM(o.revenue) AS rev FROM orders o "
        "JOIN order_items i ON o.id = i.order_id GROUP BY i.sku",
        catalog=cat,
    )
    assert out.parse_errors == ()
    with pytest.raises(CompileError, match=r"(?i)fan.?out|over-count|duplicat"):
        compile_query(out.query, cat, context=CONTEXT)


def test_required_filter_cube_refused_at_compile(catalog: dict) -> None:
    """A cube with ``required_filters`` parses cleanly but the compiler
    refuses when the mandatory filter is absent — the refusal is a
    compile-time contract, not a parser one."""
    out = parse_sql_statement("SELECT COUNT(*) FROM restricted", catalog=catalog, strict=True)
    assert out.parse_errors == ()
    with pytest.raises(CompileError, match=r"(?i)require|filter"):
        compile_query(out.query, catalog, context=CONTEXT)


# ---------------------------------------------------------------------------
# Regression tests — each pins a bug found by the SQL stress test. Without
# the fix, the predicate / measure / order key was silently dropped or
# emitted as wrong SQL (the same class as the ORDER-BY-aggregate bug).
# ---------------------------------------------------------------------------


def test_not_in_is_not_dropped(catalog: dict) -> None:
    """``NOT IN`` parses as ``exp.Not`` wrapping ``In``; it must become a
    ``not_in`` filter, not vanish."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders WHERE region NOT IN ('a', 'b') GROUP BY region",
        catalog=catalog,
    )
    assert any(
        f.dimension == "orders.region" and f.op == "not_in" and f.values == ["a", "b"]
        for f in out.query.filters
    ), out.query.filters


def test_is_not_null_is_not_dropped(catalog: dict) -> None:
    """``IS NOT NULL`` parses as ``exp.Not`` wrapping ``Is``; it must
    become a ``not_null`` filter, not vanish."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders WHERE status IS NOT NULL GROUP BY region",
        catalog=catalog,
    )
    assert any(f.dimension == "orders.status" and f.op == "not_null" for f in out.query.filters), (
        out.query.filters
    )


def test_not_wrapped_comparison_negates_operator(catalog: dict) -> None:
    """``NOT (x = y)`` and ``NOT x > y`` flip the operator rather than drop."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders "
        "WHERE NOT (status = 'paid') AND NOT amount > 100 GROUP BY region",
        catalog=catalog,
    )
    ops = {(f.dimension, f.op) for f in out.query.filters}
    assert ("orders.status", "neq") in ops, ops
    assert ("orders.amount", "lte") in ops, ops


def test_boolean_literal_becomes_bool_not_string(catalog: dict) -> None:
    """``WHERE is_paid = true`` yields a Python ``True``, not the string
    ``'TRUE'`` (which a bool dimension filter would reject)."""
    out = parse_sql_statement(
        "SELECT region, SUM(revenue) FROM orders WHERE is_paid = true GROUP BY region",
        catalog=catalog,
    )
    paid = [f for f in out.query.filters if f.dimension == "orders.is_paid"]
    assert paid and paid[0].values == [True], paid
    # And it compiles (a bool dimension accepts the bool value).
    assert "is_paid" in compile_query(out.query, catalog, context=CONTEXT).sql


def test_median_measure_is_not_dropped(catalog: dict) -> None:
    """A ``MEDIAN(...)`` wrapper marks a measure even though sqlglot models
    it as a dedicated ``exp.Median`` node (not ``Anonymous``)."""
    # ``orders`` in the shared fixture has no median measure, so use a
    # local catalog with one — the point is the parser/aggregate plumbing.
    from semql import Catalog, Cube, Dialect, Dimension, Measure

    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="med", sql="{o}.amount", agg="median", non_additive=True)],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([cube]).as_dict()
    out = parse_sql_statement("SELECT region, MEDIAN(med) FROM orders GROUP BY region", catalog=cat)
    assert "orders.med" in out.query.measures
    assert "PERCENTILE_CONT" in compile_query(out.query, cat, context=CONTEXT).sql


def test_having_count_star_is_not_dropped(catalog: dict) -> None:
    """``HAVING COUNT(*) > N`` — the inner ``*`` resolves to the cube's
    row-count measure rather than dropping the predicate."""
    out = parse_sql_statement(
        "SELECT region, COUNT(*) AS n FROM orders GROUP BY region HAVING COUNT(*) >= 10",
        catalog=catalog,
    )
    assert any(f.dimension == "orders.count" and f.op == "gte" for f in out.query.having), (
        out.query.having
    )
    assert "HAVING COUNT(*)" in compile_query(out.query, catalog, context=CONTEXT).sql


def test_order_by_unprojected_measure_emits_aggregate(catalog: dict) -> None:
    """Ordering by a measure NOT in the SELECT must emit its aggregate
    (``SUM(...)``), never the raw column — a GROUP BY query rejects the
    bare column as NOT_AN_AGGREGATE."""
    out = parse_sql_statement(
        "SELECT region FROM orders GROUP BY region ORDER BY SUM(revenue) DESC",
        catalog=catalog,
    )
    sql = compile_query(out.query, catalog, context=CONTEXT).sql
    assert "ORDER BY SUM(" in sql, sql


def test_compare_with_measure_alias_compiles_consistently(catalog: dict) -> None:
    """COMPARE-mode CTEs project canonical column names; a SELECT alias
    must not leave the outer ``current.<measure>`` refs dangling."""
    out = parse_sql_statement(
        "SELECT /*+ COMPARE prior_period */ region, SUM(revenue) AS rev FROM orders "
        "WHERE created_at BETWEEN '2026-01-01' AND '2026-03-31' GROUP BY region",
        catalog=catalog,
    )
    sql = compile_query(out.query, catalog, context=CONTEXT).sql
    # Outer references the canonical measure column; the CTE projects it.
    assert "current.revenue" in sql
    assert "SUM(o.amount) AS revenue" in sql
    # The user alias is NOT applied to the CTE column (would dangle).
    assert "AS rev " not in sql and not sql.endswith("AS rev")
