# pyright: reportPrivateImportUsage=false
"""R1 — additional dialect strategies.

Adds six backends. The split is deliberate:

- **First-class** (Redshift, Trino, Databricks): analytics engines whose
  ``date_trunc`` / percentile / ``ILIKE`` map onto clean native idioms
  via sqlglot transpilation. Registered in the default registry.
- **Experimental / opt-in** (SQL Server, MySQL, Oracle): OLTP engines
  where sqlglot transpiles ``date_trunc`` / percentile to best-effort
  forms we can't exercise in CI (no live instances). NOT in the default
  registry — callers opt in via ``experimental_dialects()`` passed
  through the ``dialects=`` override.

All six lean on transpilable nodes (``exp.TimestampTrunc``,
``exp.PercentileCont``) rather than ``exp.Anonymous``, so sqlglot's
per-dialect renderer does the heavy lifting. ``emit_time_spine``
(the gap-filling row generator) is deferred for all six — it raises a
clear ``NotImplementedError`` until each backend's row-generation idiom
is implemented.

These tests pin the rendered SQL so a future sqlglot upgrade can't
silently change the wire format under us.
"""

from __future__ import annotations

import pytest
from semql.backend import (
    DatabricksDialect,
    DialectStrategy,
    MySqlDialect,
    OracleDialect,
    RedshiftDialect,
    SqlServerDialect,
    TrinoDialect,
    dialect_for,
    experimental_dialects,
    render,
)
from semql.compile import compile_query
from semql.dialect import dialect_for as sqlglot_dialect_for
from semql.model import Cube, Dialect, Dimension, Measure, TimeDimension
from semql.spec import Filter, SemanticQuery, TimeWindow
from sqlglot import exp

# ---------------------------------------------------------------------------
# Enum + sqlglot mapping
# ---------------------------------------------------------------------------

_NEW = [
    (Dialect.REDSHIFT, "redshift"),
    (Dialect.TRINO, "trino"),
    (Dialect.DATABRICKS, "databricks"),
    (Dialect.SQLSERVER, "tsql"),
    (Dialect.MYSQL, "mysql"),
    (Dialect.ORACLE, "oracle"),
]


@pytest.mark.parametrize(("dialect", "expected"), _NEW)
def test_sqlglot_dialect_string(dialect: Dialect, expected: str) -> None:
    assert sqlglot_dialect_for(dialect) == expected


# ---------------------------------------------------------------------------
# Registry split — first-class vs experimental
# ---------------------------------------------------------------------------

_FIRST_CLASS = [Dialect.REDSHIFT, Dialect.TRINO, Dialect.DATABRICKS]
_EXPERIMENTAL = [Dialect.SQLSERVER, Dialect.MYSQL, Dialect.ORACLE]


@pytest.mark.parametrize("dialect", _FIRST_CLASS)
def test_first_class_registered_by_default(dialect: Dialect) -> None:
    strat = dialect_for(dialect)
    assert isinstance(strat, DialectStrategy)
    assert strat.dialect is dialect


@pytest.mark.parametrize("dialect", _EXPERIMENTAL)
def test_experimental_not_registered_by_default(dialect: Dialect) -> None:
    with pytest.raises(KeyError) as exc:
        dialect_for(dialect)
    # The error must point the caller at the opt-in path.
    assert "experimental_dialects" in str(exc.value)


@pytest.mark.parametrize("dialect", _EXPERIMENTAL)
def test_experimental_available_via_opt_in(dialect: Dialect) -> None:
    overrides = experimental_dialects()
    strat = dialect_for(dialect, overrides)
    assert isinstance(strat, DialectStrategy)
    assert strat.dialect is dialect


def test_experimental_dialects_returns_fresh_copy() -> None:
    a = experimental_dialects()
    a.clear()
    assert set(experimental_dialects()) == set(_EXPERIMENTAL)


# ---------------------------------------------------------------------------
# Placeholder paramstyle (sqlglot native per dialect)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (RedshiftDialect(), "%(p0)s"),
        (TrinoDialect(), ":p0"),
        (DatabricksDialect(), ":p0"),
        (SqlServerDialect(), ":p0"),
        (MySqlDialect(), ":p0"),
        (OracleDialect(), ":p0"),
    ],
)
def test_placeholder_paramstyle(strategy: DialectStrategy, expected: str) -> None:
    node = strategy.placeholder("p0", "string")
    assert render(node, strategy.dialect) == expected


# ---------------------------------------------------------------------------
# date_trunc transpilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "needle"),
    [
        (RedshiftDialect(), "DATE_TRUNC('DAY'"),
        (TrinoDialect(), "DATE_TRUNC('DAY'"),
        (DatabricksDialect(), "DATE_TRUNC('DAY'"),
        (SqlServerDialect(), "DATETRUNC(DAY"),
        (OracleDialect(), "TIMESTAMP_TRUNC("),
    ],
)
def test_trunc_transpiles(strategy: DialectStrategy, needle: str) -> None:
    col = exp.column("created_at", table="o")
    node = strategy.trunc("day", col)
    assert needle in render(node, strategy.dialect)


def test_mysql_trunc_renders_date_add_expansion() -> None:
    # MySQL has no DATE_TRUNC; sqlglot expands it to a DATE_ADD/TIMESTAMPDIFF
    # trick that is functionally correct on MySQL.
    col = exp.column("created_at", table="o")
    out = render(MySqlDialect().trunc("day", col), Dialect.MYSQL)
    assert "TIMESTAMPDIFF(DAY" in out and "DATE_ADD" in out


# ---------------------------------------------------------------------------
# Percentile transpilation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "needle"),
    [
        (RedshiftDialect(), "PERCENTILE_CONT(0.5) WITHIN GROUP"),
        (TrinoDialect(), "APPROX_PERCENTILE("),
        (DatabricksDialect(), "PERCENTILE_APPROX("),
        (OracleDialect(), "PERCENTILE_CONT(0.5) WITHIN GROUP"),
    ],
)
def test_percentile_transpiles(strategy: DialectStrategy, needle: str) -> None:
    col = exp.column("amount", table="o")
    node = strategy.emit_percentile(0.5, col)
    assert needle in render(node, strategy.dialect)


# ---------------------------------------------------------------------------
# time_spine is deferred for all six
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        RedshiftDialect(),
        TrinoDialect(),
        DatabricksDialect(),
        SqlServerDialect(),
        MySqlDialect(),
        OracleDialect(),
    ],
)
def test_time_spine_deferred(strategy: DialectStrategy) -> None:
    start = exp.Literal.string("2026-01-01")
    end = exp.Literal.string("2026-02-01")
    with pytest.raises(NotImplementedError):
        strategy.emit_time_spine("day", start, end, "bucket")


# ---------------------------------------------------------------------------
# End-to-end compile through the public path
# ---------------------------------------------------------------------------


def _cube(dialect: Dialect) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
        table="public.orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="amount_median", sql="{o}.amount", agg="median"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )


@pytest.mark.parametrize("dialect", _FIRST_CLASS)
def test_first_class_end_to_end_time_bucket(dialect: Dialect) -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = compile_query(q, {"orders": _cube(dialect)})
    assert "DATE_TRUNC" in out.sql.upper()
    assert "SUM(" in out.sql.upper()


def test_first_class_end_to_end_filter_param() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.region", op="eq", values=["EU"])],
    )
    out = compile_query(q, {"orders": _cube(Dialect.REDSHIFT)})
    # Redshift renders the psycopg2 paramstyle and binds the value.
    assert "%(" in out.sql
    assert "EU" in out.params.values() or "EU" in out.params.get("p0", "")


def test_trino_percentile_end_to_end() -> None:
    q = SemanticQuery(measures=["orders.amount_median"], dimensions=["orders.region"])
    out = compile_query(q, {"orders": _cube(Dialect.TRINO)})
    assert "APPROX_PERCENTILE" in out.sql.upper()


# ---------------------------------------------------------------------------
# Experimental opt-in end-to-end
# ---------------------------------------------------------------------------


def test_experimental_compile_refused_without_opt_in() -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    with pytest.raises(KeyError) as exc:
        compile_query(q, {"orders": _cube(Dialect.ORACLE)})
    assert "experimental_dialects" in str(exc.value)


def test_experimental_compile_works_with_opt_in() -> None:
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    out = compile_query(q, {"orders": _cube(Dialect.ORACLE)}, dialects=experimental_dialects())
    assert "SUM(" in out.sql.upper()
