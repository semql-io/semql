"""SQL-injection coverage.

The compiler's security promise: every *user value* (filter values, IN
lists, time-window endpoints) is a bound parameter — never spliced into
SQL text. Catalog-author SQL (``Cube.sql``, ``base_predicate``) is trusted
input (threat model, test-plan §8) and is *not* under test here; this file
hammers the untrusted-value boundary with classic injection payloads and
asserts, for every dialect:

1. the payload travels in ``out.params``, not ``out.sql``;
2. the emitted SQL is a single statement (no stacked-query escape); and
3. ``is_read_only_statement`` still holds (one read-only SELECT).
"""

from __future__ import annotations

import pytest
import sqlglot
from semql import (
    BoolExpr,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    FilterOp,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    is_read_only_statement,
)
from semql.compile import CompiledQuery
from sqlglot import exp

# Canonical SQL-injection payloads — quote breakouts, stacked queries,
# comment truncation, UNION exfiltration, boolean tautologies, and the
# driver/SemQL placeholder syntaxes (which must come back as data).
PAYLOADS = [
    "'; DROP TABLE orders; --",
    "' OR '1'='1",
    "' OR 1=1 --",
    "1; DELETE FROM orders",
    "' UNION SELECT username, password FROM users --",
    "admin'--",
    "\\'; DROP TABLE orders; --",
    "'/*",
    "*/--",
    "%(p0)s",  # Postgres pyformat placeholder
    "{o}.amount",  # SemQL substitution token
    "$1",
    "?; SELECT 1",
    "0x27",
    "char(39)",
    "') OR ('a'='a",
]

_DIALECTS = [
    Dialect.POSTGRES,
    Dialect.CLICKHOUSE,
    Dialect.DUCKDB,
    Dialect.BIGQUERY,
    Dialect.SNOWFLAKE,
]


def _catalog(dialect: Dialect) -> Catalog:
    alias = "t"
    return Catalog(
        [
            Cube(
                name="orders",
                dialect=dialect,
                table="orders",
                alias=alias,
                measures=[Measure(name="count", sql="*", agg="count")],
                dimensions=[
                    Dimension(name="region", sql=f"{{{alias}}}.region", type="string"),
                    Dimension(name="status", sql=f"{{{alias}}}.status", type="string"),
                ],
                time_dimensions=[TimeDimension(name="created_at", sql=f"{{{alias}}}.created_at")],
            )
        ]
    )


def _string_literals(sql: str, dialect: str) -> list[str]:
    """Every string-literal value emitted into the SQL. A parameterised
    value appears here as *nothing* (it's a placeholder); a spliced value
    appears as a literal. Robust to placeholder syntax that happens to look
    like a payload (e.g. ``%(p0)s``)."""
    statements = sqlglot.parse(sql, dialect=dialect)
    assert len(statements) == 1, f"expected one statement, got {len(statements)}:\n{sql}"
    root = statements[0]
    assert root is not None
    return [node.this for node in root.find_all(exp.Literal) if node.is_string]


def _assert_neutralised(out: CompiledQuery, payload: str, dialect: str) -> None:
    sql = out.sql
    params = out.params
    # Bound as a parameter — substring, because ``contains`` wraps the value
    # as ``%payload%`` before binding.
    assert any(payload in str(v) for v in params.values()), (
        f"payload not bound as a parameter: {payload!r} (params={params})"
    )
    # Never emitted as (part of) a SQL string literal — the actual injection
    # vector. This is what a substring check on ``sql`` only approximates.
    literals = _string_literals(sql, dialect)
    assert not any(payload in lit for lit in literals), (
        f"payload spliced into a SQL literal: {payload!r}\n{sql}"
    )
    assert is_read_only_statement(sql)


# ---------------------------------------------------------------------------
# Single-value operators across every dialect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", _DIALECTS, ids=lambda b: b.value)
@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("op", ["eq", "neq", "contains"])
def test_injection_via_single_value_filter(dialect: Dialect, payload: str, op: FilterOp) -> None:
    out = _catalog(dialect).compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op=op, values=[payload])],
        )
    )
    _assert_neutralised(out, payload, dialect.value)


# ---------------------------------------------------------------------------
# IN-list: every element is hostile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dialect", _DIALECTS, ids=lambda b: b.value)
def test_injection_via_in_list(dialect: Dialect) -> None:
    out = _catalog(dialect).compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op="in", values=list(PAYLOADS))],
        )
    )
    bound = set(out.params.values())
    literals = _string_literals(out.sql, dialect.value)
    for payload in PAYLOADS:
        assert payload in bound, f"IN element not bound: {payload!r}"
        assert not any(payload in lit for lit in literals), f"IN element spliced: {payload!r}"
    assert is_read_only_statement(out.sql)


# ---------------------------------------------------------------------------
# Time-window endpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_injection_via_time_window_range(payload: str) -> None:
    out = _catalog(Dialect.POSTGRES).compile(
        SemanticQuery(
            measures=["orders.count"],
            time_dimension=TimeWindow(dimension="orders.created_at", range=(payload, "2026-01-01")),
        )
    )
    _assert_neutralised(out, payload, "postgres")


# ---------------------------------------------------------------------------
# Nested BoolExpr — payloads at multiple leaves and depths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", PAYLOADS)
def test_injection_via_nested_boolexpr(payload: str) -> None:
    where = BoolExpr(
        op="or",
        children=[
            Filter(dimension="orders.region", op="eq", values=[payload]),
            BoolExpr(
                op="and",
                children=[
                    Filter(dimension="orders.status", op="neq", values=[payload]),
                    BoolExpr(
                        op="not",
                        children=[
                            Filter(dimension="orders.region", op="contains", values=[payload])
                        ],
                    ),
                ],
            ),
        ],
    )
    out = _catalog(Dialect.POSTGRES).compile(SemanticQuery(measures=["orders.count"], where=where))
    _assert_neutralised(out, payload, "postgres")


# ---------------------------------------------------------------------------
# A tautology payload must not widen the result set: it stays one bound
# literal compared for equality, never a live ``OR 1=1`` predicate.
# ---------------------------------------------------------------------------


def test_tautology_payload_is_compared_not_evaluated() -> None:
    out = _catalog(Dialect.POSTGRES).compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op="eq", values=["' OR 1=1 --"])],
        )
    )
    # The equality predicate binds the whole string; there is no bare 1=1.
    assert "1=1" not in out.sql.replace(" ", "")
    assert "' OR 1=1 --" in out.params.values()


# ---------------------------------------------------------------------------
# Writable-CTE bypass (SEMQL-READONLY-WRITABLE-CTE): SELECT root is not
# sufficient — DML inside a CTE body must also be rejected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM users RETURNING id) SELECT * FROM d",
        "WITH d AS (INSERT INTO log VALUES (1) RETURNING id) SELECT * FROM d",
        "WITH d AS (UPDATE users SET x=1 RETURNING id) SELECT * FROM d",
    ],
    ids=["writable-cte-delete", "writable-cte-insert", "writable-cte-update"],
)
def test_writable_cte_rejected_by_read_only_guard(sql: str) -> None:
    """is_read_only_statement must reject SELECT roots whose CTEs contain DML."""
    assert not is_read_only_statement(sql, dialect="postgres")
