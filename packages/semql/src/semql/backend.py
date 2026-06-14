# pyright: reportPrivateImportUsage=false
# sqlglot's ``Expression`` and friends live in ``sqlglot.expressions``
# but aren't re-exported via ``__all__``. They're public by convention
# and by sqlglot's own type stubs.
"""Per-backend dialect objects — the dialect-specific seam.

The compiler stays dialect-agnostic for the parts it can: identifier
resolution, join graph BFS, GROUP BY composition, parameter-name
allocation, ordering. The dialect owns the rest, and emits sqlglot
AST nodes (not strings):

- ``placeholder(name, dim_type)`` — bound-param node
  (``exp.Placeholder`` typed for the dialect).
- ``trunc(granularity, expr)`` — date truncation node
  (``date_trunc`` for PG, ``toStartOf<Hour|Day|Week|Month>`` for CH).
- ``emit_contains(field, value, bind)`` — substring search,
  including the *value transform* tied to that shape (Postgres bakes
  ``%v%`` into the bound value; ClickHouse passes the raw substring).
- ``emit_source(cube, catalog, resolve_sql)`` — how a cube becomes a
  ``FROM`` source. Vanilla cubes return a ``Table`` node with alias;
  META cubes return a ``Subquery`` over a ``VALUES`` literal.

The Protocol is structural (``typing.Protocol``) so downstream
Snowflake / BigQuery adapters can satisfy it without importing from
this module.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import sqlglot
from sqlglot import TokenType as _TT
from sqlglot import exp

from semql.dialect import dialect_for as sqlglot_dialect_for
from semql.dialect import placeholder_for
from semql.introspect import build_meta_values
from semql.model import Cube, DerivedTable, Dialect, PhysicalTable, WeekStartLiteral

ParamBinder = Callable[[Any, str], exp.Placeholder]
"""Bind ``(value, dim_type)`` and return an ``exp.Placeholder`` node for it."""

SqlResolver = Callable[[str], str]
"""Resolve ``{alias}`` / ``{schema}`` placeholders in a SQL fragment."""


@runtime_checkable
class DialectStrategy(Protocol):
    """Structural contract every backend dialect honours.

    Strategies are stateless and pure: each method returns a sqlglot
    AST node (and, in ``emit_contains``, may invoke ``bind`` to
    register a bound parameter). The Protocol form lets third-party
    adapters satisfy it without inheriting from us — useful for
    out-of-tree Snowflake / BigQuery strategies.
    """

    dialect: Dialect

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder: ...
    def trunc(
        self,
        granularity: str,
        expr: exp.Expression,
        timezone: str | None = None,
        week_start: WeekStartLiteral = "monday",
    ) -> exp.Expression: ...
    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        """Continuous percentile of ``expr`` at quantile ``q`` (0..1).

        Dialect-specific because the ANSI ``PERCENTILE_CONT(q) WITHIN
        GROUP (ORDER BY expr)`` shape isn't universal — ClickHouse uses
        ``quantile(q)(expr)`` and BigQuery uses
        ``APPROX_QUANTILES(expr, 100)[OFFSET(p)]``."""
        ...

    def emit_contains(
        self, field: exp.Expression, value: str, bind: ParamBinder
    ) -> exp.Expression: ...
    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,
    ) -> exp.Expression: ...
    def emit_time_spine(
        self,
        granularity: str,
        start: exp.Expression,
        end: exp.Expression,
        bucket_alias: str,
    ) -> exp.Expression: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


_CH_TRUNC: dict[str, str] = {
    "second": "toStartOfSecond",
    "minute": "toStartOfMinute",
    "hour": "toStartOfHour",
    "day": "toStartOfDay",
    "week": "toStartOfWeek",
    "month": "toStartOfMonth",
    "quarter": "toStartOfQuarter",
    "year": "toStartOfYear",
}


def _sunday_from_monday_week(
    monday_trunc: Callable[[exp.Expression], exp.Expression], expr: exp.Expression
) -> exp.Expression:
    """Shift a Monday-native week truncation to a Sunday-start week.

    ``date_trunc('week', …)`` (and ``exp.TimestampTrunc`` over ``WEEK``)
    is Monday-native on every SQL backend we target. To start the week on
    Sunday, shift the input forward one day, truncate, then shift the
    result back one day: a Sunday lands on the following Monday's week
    (→ itself once shifted back), and a Saturday on the same Monday-week
    (→ the prior Sunday). ``monday_trunc`` builds the dialect's native
    Monday week-trunc node for a given expression."""
    one_day = exp.Interval(this=exp.Literal.number(1), unit=exp.Var(this="DAY"))
    shifted = exp.Paren(this=exp.Add(this=expr, expression=one_day.copy()))
    return exp.Sub(this=monday_trunc(shifted), expression=one_day.copy())


@functools.lru_cache(maxsize=512)
def _ident_needs_quote(name: str, dialect: str) -> bool:
    """Whether ``name`` must be quoted to survive as a bare identifier
    under ``dialect`` — true when the dialect tokenizer reads it as a
    keyword rather than a plain VAR/IDENTIFIER (e.g. ``USING``, ``SELECT``).

    The tokenize + dialect resolution this performs is the expensive part
    of :func:`_ident` and is a pure function of ``(name, dialect)`` over a
    tiny key space (table / schema / alias names), so it is memoised.
    The boolean is cached — never the ``Identifier`` node, which callers
    reparent into distinct trees and must therefore receive fresh."""
    toks = sqlglot.tokenize(name, dialect=dialect)
    return not toks or toks[0].token_type not in (_TT.VAR, _TT.IDENTIFIER)


def _ident(name: str, dialect: str = "postgres") -> exp.Identifier:
    """Return an Identifier for ``name``, quoted if the dialect tokenizer
    would treat it as a keyword (e.g. USING, SELECT, TABLE).

    ``dialect`` must be the *emitting* dialect: keyword sets are
    dialect-specific (``sample`` is a ClickHouse keyword but an ordinary
    word in Postgres), so a table name has to be tested against the same
    dialect it will be emitted under or it slips through unquoted."""
    return exp.to_identifier(name, quoted=_ident_needs_quote(name, dialect))


def _aliased_source(cube: Cube, resolve_sql: SqlResolver) -> exp.Expression:
    """Build the aliased FROM source for ``cube``.

    Dispatches on ``cube.resolved_source``: plain-table cubes return a
    ``Table`` node (``schema.tbl AS alias``); derived-table cubes return a
    ``Subquery`` wrapping the resolved SQL (``(<sql>) AS alias``). Both
    flow through the same ``resolve_sql`` placeholder substitution.

    Cubes with ``physical_sources`` are handled by
    :func:`semql.partition.emit_physical_sources` via the dialect
    ``emit_source`` method — they have no single ``CubeSource`` to
    resolve here."""
    src = cube.resolved_source
    if isinstance(src, DerivedTable):
        return _aliased_derived(cube, src, resolve_sql)
    return _aliased_table(cube, src, resolve_sql)


def _aliased_table(cube: Cube, src: PhysicalTable, resolve_sql: SqlResolver) -> exp.Table:
    """Build a ``Table`` AST node for ``cube`` with its alias attached."""
    resolved = resolve_sql(src.table)
    # Build the Table node manually rather than via exp.to_table() so that
    # reserved-word table names (e.g. ``using``, ClickHouse ``sample``) are
    # properly quoted instead of being emitted as bare keywords. Test the
    # name against the cube's *own* dialect — keyword sets differ per
    # dialect, so a Postgres-only check would miss ClickHouse/BigQuery words.
    dialect = sqlglot_dialect_for(cube.dialect)
    parts = resolved.split(".", 1)
    if len(parts) == 2:
        tbl = exp.Table(this=_ident(parts[1], dialect), db=_ident(parts[0], dialect))
    else:
        tbl = exp.Table(this=_ident(parts[0], dialect))
    tbl.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return tbl


def _aliased_derived(cube: Cube, src: DerivedTable, resolve_sql: SqlResolver) -> exp.Subquery:
    """Build a ``Subquery`` AST node over ``src.sql`` aliased to ``cube.alias``.

    The derived SQL goes through the same placeholder substitution as
    plain-table sources (``{tenant_schema}`` / ``{key}``). Parsing uses
    the cube's own dialect so backend-specific shapes (ClickHouse ARRAY
    JOIN, BigQuery UNNEST, etc.) survive the round-trip."""
    resolved = resolve_sql(src.sql)
    parsed = sqlglot.parse_one(resolved, dialect=sqlglot_dialect_for(cube.dialect))
    if not isinstance(parsed, exp.Subquery):
        parsed = exp.Subquery(this=parsed)
    parsed.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return parsed


def _meta_subquery(cube: Cube, catalog: dict[str, Cube]) -> exp.Subquery:
    """Build a ``Subquery`` AST node over the catalog snapshot for ``cube``."""
    body = build_meta_values(cube.name, catalog)
    sub = sqlglot.parse_one(body, dialect="postgres")
    if not isinstance(sub, exp.Subquery):
        sub = exp.Subquery(this=sub)
    sub.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return sub


class _StdSqlDialect:
    """Shared base for backends whose dialect is handled correctly by
    sqlglot's stock renderer — Postgres, DuckDB, BigQuery, Snowflake.

    ``date_trunc`` goes through ``exp.Anonymous`` to keep the lowercase
    name verbatim. ``ILIKE`` AST gets transpiled per dialect on emit
    (BigQuery rewrites to ``LOWER(...) LIKE LOWER(...)``; PG / SF / DuckDB
    keep ``ILIKE`` natively). ``%v%`` always bakes into the bound value —
    the ``LIKE`` semantics survive the BigQuery transpile."""

    dialect: Dialect  # set by subclass

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, self.dialect)

    def trunc(
        self,
        granularity: str,
        expr: exp.Expression,
        timezone: str | None = None,
        week_start: WeekStartLiteral = "monday",
    ) -> exp.Expression:
        # ``AT TIME ZONE`` is the sqlglot-canonical shift; the renderer
        # transpiles it per dialect — ``AT TIME ZONE`` on Postgres/DuckDB,
        # ``TIMESTAMP(DATETIME(...))`` on BigQuery, ``CONVERT_TIMEZONE`` on
        # Snowflake — so this one branch covers every ``_StdSqlDialect``.
        target = (
            expr
            if timezone is None
            else exp.AtTimeZone(this=expr, zone=exp.Literal.string(timezone))
        )

        def _dt(e: exp.Expression) -> exp.Expression:
            return exp.Anonymous(this="date_trunc", expressions=[exp.Literal.string("week"), e])

        if granularity == "week" and week_start == "sunday":
            return _sunday_from_monday_week(_dt, target)
        return exp.Anonymous(
            this="date_trunc",
            expressions=[exp.Literal.string(granularity), target],
        )

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        """``PERCENTILE_CONT(q) WITHIN GROUP (ORDER BY expr)`` — ANSI
        form, works as-is on Postgres / DuckDB / Snowflake. BigQuery
        and ClickHouse override this with their own quantile shapes."""
        return exp.WithinGroup(
            this=exp.Anonymous(this="PERCENTILE_CONT", expressions=[exp.Literal.number(q)]),
            expression=exp.Order(expressions=[exp.Ordered(this=expr)]),
        )

    def emit_contains(self, field: exp.Expression, value: str, bind: ParamBinder) -> exp.Expression:
        ph = bind(f"%{value}%", "string")
        return exp.ILike(this=field, expression=ph)

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> exp.Expression:
        return _aliased_source(cube, resolve_sql)

    def emit_time_spine(
        self,
        granularity: str,
        start: exp.Expression,
        end: exp.Expression,
        bucket_alias: str,
    ) -> exp.Expression:
        # Default spine for Postgres + DuckDB:
        # generate_series(date_trunc(g, start), date_trunc(g, end - 1 step), 1 step)
        # BigQuery and Snowflake override this with their own table-function shapes.
        step = exp.Interval(
            this=exp.Literal.number(1),
            unit=exp.Var(this=granularity.upper()),
        )
        trunc_start = exp.Anonymous(
            this="date_trunc", expressions=[exp.Literal.string(granularity), start]
        )
        end_minus_step = exp.Paren(this=exp.Sub(this=end, expression=step.copy()))
        trunc_end = exp.Anonymous(
            this="date_trunc", expressions=[exp.Literal.string(granularity), end_minus_step]
        )
        series = exp.Anonymous(
            this="generate_series",
            expressions=[trunc_start, trunc_end, step.copy()],
        )
        inner = exp.Select().select(exp.alias_(series, bucket_alias, copy=False), copy=False)
        return inner


class PostgresDialect(_StdSqlDialect):
    """Postgres convention. Placeholders render as ``%(name)s``."""

    dialect = Dialect.POSTGRES


class DuckDBDialect(_StdSqlDialect):
    """DuckDB convention. Placeholders render as ``$name`` (the canonical
    DuckDB named-parameter syntax). Otherwise identical to Postgres —
    DuckDB shares ``ILIKE``, ``date_trunc``, and the aliased-table FROM
    shape."""

    dialect = Dialect.DUCKDB


class BigQueryDialect(_StdSqlDialect):
    """BigQuery convention. Placeholders render as ``@name``. sqlglot
    transpiles the ``ILIKE`` AST to ``LOWER(...) LIKE LOWER(...)`` on
    emit, so case-insensitive contains still works against a column."""

    dialect = Dialect.BIGQUERY

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        # BigQuery has no ``PERCENTILE_CONT``. ``APPROX_QUANTILES(expr, 100)``
        # returns an ordered array of 101 quantile boundaries; ``[OFFSET(p)]``
        # picks the q-th. Approximate but cheap; documented as such in
        # ``AggLiteral``'s docstring so callers know the BigQuery branch is
        # an approximation.
        offset = int(round(q * 100))
        quantiles = exp.Anonymous(
            this="APPROX_QUANTILES",
            expressions=[expr, exp.Literal.number(100)],
        )
        return exp.Bracket(
            this=quantiles,
            expressions=[exp.Anonymous(this="OFFSET", expressions=[exp.Literal.number(offset)])],
        )

    def emit_time_spine(
        self,
        granularity: str,
        start: exp.Expression,
        end: exp.Expression,
        bucket_alias: str,
    ) -> exp.Expression:
        # BigQuery has no ``generate_series`` — use
        # ``UNNEST(GENERATE_DATE_ARRAY(...))`` to turn an array of dates
        # into rows, then ``DATE_TRUNC`` each one to the requested grain.
        # DISTINCT collapses duplicate week / month buckets.
        one_day = exp.Interval(this=exp.Literal.number(1), unit=exp.Var(this="DAY"))
        start_date = exp.Anonymous(this="DATE", expressions=[start])
        end_minus_one = exp.Paren(this=exp.Sub(this=end, expression=one_day.copy()))
        end_date = exp.Anonymous(this="DATE", expressions=[end_minus_one])
        series = exp.Anonymous(
            this="GENERATE_DATE_ARRAY",
            expressions=[start_date, end_date, one_day.copy()],
        )
        bucket_col = exp.Column(this=exp.to_identifier("d"))
        trunc_node = exp.Anonymous(
            this="DATE_TRUNC",
            expressions=[bucket_col, exp.Var(this=granularity.upper())],
        )
        inner = exp.Select(distinct=exp.Distinct())
        inner = inner.select(exp.alias_(trunc_node, bucket_alias, copy=False), copy=False)
        unnest = exp.Unnest(
            expressions=[series],
            alias=exp.TableAlias(this=exp.to_identifier("d")),
        )
        inner = inner.from_(unnest)
        return inner


class SnowflakeDialect(_StdSqlDialect):
    """Snowflake convention. Placeholders render as ``:name``. ``ILIKE``
    and ``date_trunc`` are native to Snowflake — no transpilation needed."""

    dialect = Dialect.SNOWFLAKE

    def emit_time_spine(
        self,
        granularity: str,
        start: exp.Expression,
        end: exp.Expression,
        bucket_alias: str,
    ) -> exp.Expression:
        # Snowflake uses ``TABLE(GENERATOR(ROWCOUNT => N))`` to materialise
        # N rows and ``SEQ4()`` for the per-row 0..N-1 counter. The
        # ``ROWCOUNT => ...`` named-arg syntax goes through ``exp.Kwarg``.
        to_date_start = exp.Anonymous(this="TO_DATE", expressions=[start])
        add_days = exp.Anonymous(
            this="DATEADD",
            expressions=[
                exp.Literal.string("day"),
                exp.Anonymous(this="SEQ4", expressions=[]),
                to_date_start,
            ],
        )
        bucket = exp.Anonymous(
            this="DATE_TRUNC",
            expressions=[exp.Literal.string(granularity), add_days],
        )
        day_diff = exp.Anonymous(
            this="DATEDIFF",
            expressions=[
                exp.Literal.string("day"),
                exp.Anonymous(this="TO_DATE", expressions=[start.copy()]),
                exp.Anonymous(this="TO_DATE", expressions=[end]),
            ],
        )
        generator = exp.Anonymous(
            this="GENERATOR",
            expressions=[
                exp.Kwarg(
                    this=exp.Column(this=exp.to_identifier("ROWCOUNT")),
                    expression=day_diff,
                ),
            ],
        )
        table_func = exp.Anonymous(this="TABLE", expressions=[generator])
        inner = exp.Select(distinct=exp.Distinct()).select(
            exp.alias_(bucket, bucket_alias, copy=False), copy=False
        )
        inner = inner.from_(table_func, copy=False)
        return inner


class _TranspilingSqlDialect(_StdSqlDialect):
    """Base for the R1 backends — engines sqlglot can transpile to from
    its canonical nodes (Redshift, Trino, Databricks, SQL Server, MySQL,
    Oracle).

    Where ``_StdSqlDialect`` pins ``date_trunc`` / ``PERCENTILE_CONT`` to
    ``exp.Anonymous`` (verbatim, no transpilation — correct for the five
    backends that share Postgres' spelling), this base emits the
    *canonical* sqlglot nodes (``exp.TimestampTrunc``, ``exp.PercentileCont``)
    so sqlglot's per-dialect renderer rewrites them to each engine's
    native idiom: ``DATE_TRUNC`` on Redshift/Trino/Databricks,
    ``DATETRUNC`` on SQL Server, the ``DATE_ADD`` trick on MySQL,
    ``TIMESTAMP_TRUNC`` on Oracle; percentile becomes ``APPROX_PERCENTILE``
    (Trino), ``PERCENTILE_APPROX`` (Databricks), or native
    ``PERCENTILE_CONT ... WITHIN GROUP`` (Redshift/Oracle/SQL Server/MySQL).

    ``placeholder`` / ``emit_contains`` / ``emit_source`` are inherited:
    ``placeholder_for`` already yields the right paramstyle per dialect
    (``%(name)s`` on Redshift, ``:name`` elsewhere), and sqlglot transpiles
    the ``ILIKE`` AST to ``LOWER(...) LIKE LOWER(...)`` where a backend
    lacks ``ILIKE``.

    ``emit_time_spine`` is deferred: the gap-fill row generator differs
    sharply per backend (sequence/UNNEST vs recursive CTE vs CONNECT BY)
    and is a separable follow-up — it raises ``NotImplementedError``."""

    def trunc(
        self,
        granularity: str,
        expr: exp.Expression,
        timezone: str | None = None,
        week_start: WeekStartLiteral = "monday",
    ) -> exp.Expression:
        target = (
            expr
            if timezone is None
            else exp.AtTimeZone(this=expr, zone=exp.Literal.string(timezone))
        )

        def _tt(e: exp.Expression) -> exp.Expression:
            # exp.TimestampTrunc is sqlglot's canonical truncation node; the
            # renderer rewrites it to each dialect's native spelling on emit.
            return exp.TimestampTrunc(this=e, unit=exp.Var(this=granularity.upper()))

        if granularity == "week" and week_start == "sunday":
            return _sunday_from_monday_week(
                lambda e: exp.TimestampTrunc(this=e, unit=exp.Var(this="WEEK")), target
            )
        return _tt(target)

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        # The quantile is the *only* argument to PercentileCont; the column
        # lives in the WITHIN GROUP ordering. This is the shape sqlglot
        # transpiles correctly (Trino → APPROX_PERCENTILE(expr, q),
        # Databricks → PERCENTILE_APPROX, ANSI engines keep WITHIN GROUP).
        return exp.WithinGroup(
            this=exp.PercentileCont(this=exp.Literal.number(q)),
            expression=exp.Order(expressions=[exp.Ordered(this=expr)]),
        )

    def emit_time_spine(
        self,
        granularity: str,  # noqa: ARG002
        start: exp.Expression,  # noqa: ARG002
        end: exp.Expression,  # noqa: ARG002
        bucket_alias: str,  # noqa: ARG002
    ) -> exp.Expression:
        raise NotImplementedError(
            f"Time-spine (fill_nulls) emission is not yet implemented for "
            f"{self.dialect.value!r}. Each backend's row-generation idiom "
            "(sequence/UNNEST, recursive CTE, CONNECT BY) is a separate "
            "follow-up; gap-filling time series is unsupported on this "
            "dialect for now."
        )


class RedshiftDialect(_TranspilingSqlDialect):
    """Redshift convention. Placeholders render as ``%(name)s`` (psycopg2
    paramstyle); ``DATE_TRUNC`` and native ``PERCENTILE_CONT ... WITHIN
    GROUP`` are first-class."""

    dialect = Dialect.REDSHIFT


class TrinoDialect(_TranspilingSqlDialect):
    """Trino convention. Placeholders render as ``:name`` (the SQLAlchemy
    Trino paramstyle). Percentile transpiles to ``APPROX_PERCENTILE`` —
    an approximation, documented as such on ``AggLiteral``."""

    dialect = Dialect.TRINO


class DatabricksDialect(_TranspilingSqlDialect):
    """Databricks (Spark SQL) convention. Placeholders render as ``:name``.
    Percentile transpiles to ``PERCENTILE_APPROX`` — an approximation."""

    dialect = Dialect.DATABRICKS


class SqlServerDialect(_TranspilingSqlDialect):
    """SQL Server (T-SQL) — **experimental** (opt-in via
    ``experimental_dialects()``). ``date_trunc`` transpiles to
    ``DATETRUNC`` (SQL Server 2022+); percentile transpiles to
    ``PERCENTILE_CONT ... WITHIN GROUP``, which on real SQL Server is a
    *window* function (requires ``OVER()``) — verify on a live instance
    before relying on it."""

    dialect = Dialect.SQLSERVER


class MySqlDialect(_TranspilingSqlDialect):
    """MySQL — **experimental** (opt-in via ``experimental_dialects()``).
    MySQL has no ``DATE_TRUNC``; sqlglot expands it to a
    ``DATE_ADD``/``TIMESTAMPDIFF`` trick. ``PERCENTILE_CONT`` is not native
    to MySQL — verify on a live instance before relying on it."""

    dialect = Dialect.MYSQL


class OracleDialect(_TranspilingSqlDialect):
    """Oracle — **experimental** (opt-in via ``experimental_dialects()``).
    sqlglot emits ``TIMESTAMP_TRUNC`` (Oracle's native truncation is
    ``TRUNC(date, fmt)``) and native ``PERCENTILE_CONT ... WITHIN GROUP``
    — verify the truncation spelling on a live instance before relying
    on it."""

    dialect = Dialect.ORACLE


class ClickHouseDialect:
    """ClickHouse convention. Placeholders are typed (``{name:Type}``);
    truncation uses the ``toStartOf<Hour|Day|Week|Month>`` family;
    ``contains`` passes the raw substring and emits
    ``positionCaseInsensitive``."""

    dialect = Dialect.CLICKHOUSE

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, Dialect.CLICKHOUSE)

    def trunc(
        self,
        granularity: str,
        expr: exp.Expression,
        timezone: str | None = None,
        week_start: WeekStartLiteral = "monday",
    ) -> exp.Expression:
        # ``exp.Anonymous`` keeps sqlglot from transpiling
        # ``toStartOfHour(...)`` into the canonical ``dateTrunc('HOUR', ...)``
        # form. The emitted SQL preserves the ClickHouse-idiomatic name.
        # ClickHouse's ``toStartOf*`` family takes an optional timezone as
        # its second argument.
        args: list[exp.Expression] = [expr]
        if granularity == "week":
            # ``toStartOfWeek(t, mode[, tz])`` — mode 1 = Monday-first,
            # 0 = Sunday-first. Pass it explicitly so the boundary matches
            # the other dialects (a bare ``toStartOfWeek`` defaults to
            # Sunday, out of step with the Monday-native ``date_trunc``).
            args.append(exp.Literal.number(1 if week_start == "monday" else 0))
        if timezone is not None:
            args.append(exp.Literal.string(timezone))
        return exp.Anonymous(this=_CH_TRUNC[granularity], expressions=args)

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        # ClickHouse's ``quantile(q)(expr)`` is the curried form — the
        # quantile parameter applies to the function itself, and the
        # column expression is the only positional argument. sqlglot has
        # no native node for parametric functions; emit via Anonymous
        # so the ClickHouse-idiomatic shape survives unmolested.
        return exp.Anonymous(
            this=f"quantile({q})",
            expressions=[expr],
        )

    def emit_contains(self, field: exp.Expression, value: str, bind: ParamBinder) -> exp.Expression:
        ph = bind(value, "string")
        pos = exp.Anonymous(this="positionCaseInsensitive", expressions=[field, ph])
        return exp.GT(this=pos, expression=exp.Literal.number(0))

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> exp.Expression:
        return _aliased_source(cube, resolve_sql)

    def emit_time_spine(
        self,
        granularity: str,
        start: exp.Expression,
        end: exp.Expression,
        bucket_alias: str,
    ) -> exp.Expression:
        # ClickHouse has no ``generate_series``. We expand a daily index
        # via ``numbers(dateDiff('day', start, end))`` and truncate each
        # day to ``toStartOf<Gran>(...)`` — DISTINCT collapses the
        # duplicates that show up for week / month grains.
        trunc_name = _CH_TRUNC[granularity]
        add_days = exp.Anonymous(
            this="addDays",
            expressions=[
                exp.Anonymous(this="toDate", expressions=[start.copy()]),
                exp.Column(this=exp.to_identifier("number")),
            ],
        )
        bucket = exp.Anonymous(this=trunc_name, expressions=[add_days])
        day_diff = exp.Anonymous(
            this="dateDiff",
            expressions=[
                exp.Literal.string("day"),
                exp.Anonymous(this="toDate", expressions=[start.copy()]),
                exp.Anonymous(this="toDate", expressions=[end.copy()]),
            ],
        )
        inner = exp.Select(distinct=exp.Distinct())
        inner = inner.select(exp.alias_(bucket, bucket_alias, copy=False), copy=False)
        inner = inner.from_(exp.Anonymous(this="numbers", expressions=[day_diff]), copy=False)
        return inner


class MetaDialect:
    """Reflection cubes — materialised as a ``VALUES`` subquery at compile
    time. Inherits Postgres parameter and truncation conventions for the
    (rare) cases the compiler emits one against a META cube."""

    dialect = Dialect.META

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, Dialect.META)

    def trunc(
        self,
        granularity: str,
        expr: exp.Expression,
        timezone: str | None = None,
        week_start: WeekStartLiteral = "monday",
    ) -> exp.Expression:
        # META cubes are caller-constructed VALUES tables; a timezone is
        # rarely meaningful here, but honour it via the same ``AT TIME
        # ZONE`` shift the Postgres convention uses so the Protocol stays
        # uniform.
        target = (
            expr
            if timezone is None
            else exp.AtTimeZone(this=expr, zone=exp.Literal.string(timezone))
        )
        if granularity == "week" and week_start == "sunday":
            return _sunday_from_monday_week(
                lambda e: exp.Anonymous(
                    this="date_trunc", expressions=[exp.Literal.string("week"), e]
                ),
                target,
            )
        return exp.Anonymous(
            this="date_trunc",
            expressions=[exp.Literal.string(granularity), target],
        )

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        # META cubes are caller-constructed VALUES tables and don't
        # carry numeric measures in practice — but satisfy the protocol
        # with the ANSI shape so a percentile measure on a META cube
        # doesn't crash the dialect lookup.
        return exp.WithinGroup(
            this=exp.Anonymous(this="PERCENTILE_CONT", expressions=[exp.Literal.number(q)]),
            expression=exp.Order(expressions=[exp.Ordered(this=expr)]),
        )

    def emit_contains(self, field: exp.Expression, value: str, bind: ParamBinder) -> exp.Expression:
        ph = bind(f"%{value}%", "string")
        return exp.ILike(this=field, expression=ph)

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,  # noqa: ARG002
    ) -> exp.Expression:
        return _meta_subquery(cube, catalog)

    def emit_time_spine(
        self,
        granularity: str,  # noqa: ARG002
        start: exp.Expression,  # noqa: ARG002
        end: exp.Expression,  # noqa: ARG002
        bucket_alias: str,  # noqa: ARG002
    ) -> exp.Expression:
        raise NotImplementedError("Time spine emission is not applicable to META reflection cubes.")


# ---------------------------------------------------------------------------
# Registry + DI hybrid
# ---------------------------------------------------------------------------


_DEFAULTS: dict[Dialect, DialectStrategy] = {
    Dialect.POSTGRES: PostgresDialect(),
    Dialect.CLICKHOUSE: ClickHouseDialect(),
    Dialect.META: MetaDialect(),
    Dialect.DUCKDB: DuckDBDialect(),
    Dialect.BIGQUERY: BigQueryDialect(),
    Dialect.SNOWFLAKE: SnowflakeDialect(),
    # R1 first-class analytics engines.
    Dialect.REDSHIFT: RedshiftDialect(),
    Dialect.TRINO: TrinoDialect(),
    Dialect.DATABRICKS: DatabricksDialect(),
}


# R1 experimental OLTP engines. Deliberately NOT in ``_DEFAULTS`` — sqlglot
# transpiles their date_trunc / percentile to best-effort forms we can't
# exercise in CI (no live instances). Callers opt in via
# ``experimental_dialects()`` passed through the ``dialects=`` override.
_EXPERIMENTAL: dict[Dialect, DialectStrategy] = {
    Dialect.SQLSERVER: SqlServerDialect(),
    Dialect.MYSQL: MySqlDialect(),
    Dialect.ORACLE: OracleDialect(),
}


def experimental_dialects() -> dict[Dialect, DialectStrategy]:
    """Opt-in strategies for the SQL Server / MySQL / Oracle backends.

    These OLTP engines lack the native ``date_trunc`` / percentile idioms
    of the analytics backends, so sqlglot transpiles them to best-effort
    forms that aren't exercised in CI (no live instances to run against).
    They are kept out of the default registry; enable them explicitly by
    passing the returned dict through the compiler's ``dialects=``
    override::

        from semql.backend import experimental_dialects

        out = compile_query(q, catalog, dialects=experimental_dialects())

    Verify the emitted SQL against a live instance before relying on it.
    Returns a fresh dict each call, so callers can mutate / merge freely.
    """
    return dict(_EXPERIMENTAL)


def dialect_for(
    dialect: Dialect,
    overrides: dict[Dialect, DialectStrategy] | None = None,
) -> DialectStrategy:
    """Look up the dialect for ``backend``, honouring ``overrides``.

    Callers (tests, downstream packages) can pass ``overrides`` to
    swap in a different dialect without touching the global registry —
    e.g. a ``RecordingDialect`` for delegation tests, or a custom
    ``SnowflakeDialect`` shipped from outside this repo.
    """
    if overrides is not None and dialect in overrides:
        return overrides[dialect]
    if dialect in _DEFAULTS:
        return _DEFAULTS[dialect]
    if dialect in _EXPERIMENTAL:
        raise KeyError(
            f"{dialect!r} is an experimental dialect and is not registered "
            "by default. Enable it by passing `experimental_dialects()` "
            "through the compiler's `dialects=` override."
        )
    raise KeyError(
        f"No registered DialectStrategy for {dialect!r}. "
        "Pass one via the `dialects` kwarg on compile_query."
    )


def render(node: exp.Expression, dialect: Dialect) -> str:
    """Render a sqlglot node as the dialect-canonical SQL for ``backend``.

    ``normalize_functions=False`` keeps function names verbatim (so an
    ``exp.Anonymous(this="date_trunc")`` emits ``date_trunc(...)`` rather
    than ``DATE_TRUNC(...)``) — the existing assertions and the
    production code that consumes our SQL both expect the lowercase
    form."""
    return node.sql(
        dialect=sqlglot_dialect_for(dialect),
        pretty=False,
        normalize_functions=False,
    )


__all__ = [
    "DialectStrategy",
    "BigQueryDialect",
    "ClickHouseDialect",
    "DatabricksDialect",
    "DuckDBDialect",
    "MetaDialect",
    "MySqlDialect",
    "OracleDialect",
    "ParamBinder",
    "PostgresDialect",
    "RedshiftDialect",
    "SnowflakeDialect",
    "SqlResolver",
    "SqlServerDialect",
    "TrinoDialect",
    "render",
    "dialect_for",
    "experimental_dialects",
]
