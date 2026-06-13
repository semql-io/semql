"""C7 (ktx-ports M1) — reserved-identifier quoting.

A catalog SQL fragment may reference a column whose name is a SQL
reserved word (``order``, ``value``, ``group``). sqlglot's own
reserved-word quoting is inconsistent across dialects — empty for
Postgres/ClickHouse, partial for DuckDB — so such a fragment can emit
unquoted, invalid SQL (Postgres: ``t.order`` is a syntax error). The
compiler must quote reserved identifiers itself, and must NOT over-quote
ordinary identifiers (no churn on ``t.region``).
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.model import Cube, Dialect, Dimension, Measure
from semql.spec import SemanticQuery


def _cube(backend: Dialect) -> Cube:
    return Cube(
        name="t",
        backend=backend,
        table="s.t",
        alias="t",
        measures=[Measure(name="n", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="ord", sql="{t}.order", type="string"),
            Dimension(name="val", sql="{t}.value", type="string"),
            Dimension(name="grp", sql="{t}.group", type="string"),
            Dimension(name="plain", sql="{t}.region", type="string"),
        ],
    )


@pytest.mark.parametrize("backend", [Dialect.POSTGRES, Dialect.DUCKDB])
def test_reserved_identifiers_are_quoted(backend: Dialect) -> None:
    cat = {"t": _cube(backend)}
    q = SemanticQuery(
        measures=["t.n"],
        dimensions=["t.ord", "t.val", "t.grp", "t.plain"],
    )
    sql = compile_query(q, cat, context={}).sql
    assert 't."order"' in sql, sql
    assert 't."value"' in sql, sql
    assert 't."group"' in sql, sql


@pytest.mark.parametrize("backend", [Dialect.POSTGRES, Dialect.DUCKDB])
def test_ordinary_identifiers_are_not_quoted(backend: Dialect) -> None:
    cat = {"t": _cube(backend)}
    q = SemanticQuery(measures=["t.n"], dimensions=["t.plain"])
    sql = compile_query(q, cat, context={}).sql
    # No over-quoting of a non-reserved identifier.
    assert "t.region" in sql, sql
    assert 't."region"' not in sql, sql
