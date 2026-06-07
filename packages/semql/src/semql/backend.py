# pyright: reportPrivateImportUsage=false
# sqlglot's ``Expression`` and friends live in ``sqlglot.expressions``
# but aren't re-exported via ``__all__``. They're public by convention
# and by sqlglot's own type stubs.
"""Per-backend strategy objects — the dialect-specific seam.

The compiler stays dialect-agnostic for the parts it can: identifier
resolution, join graph BFS, GROUP BY composition, parameter-name
allocation, ordering. The strategy owns the rest, and emits sqlglot
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

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import sqlglot
from sqlglot import TokenType as _TT
from sqlglot import exp

from semql.dialect import dialect_for, placeholder_for
from semql.introspect import build_meta_values
from semql.model import Backend, Cube, DerivedTable, TableRef

ParamBinder = Callable[[Any, str], exp.Placeholder]
"""Bind ``(value, dim_type)`` and return an ``exp.Placeholder`` node for it."""

SqlResolver = Callable[[str], str]
"""Resolve ``{alias}`` / ``{schema}`` placeholders in a SQL fragment."""


@runtime_checkable
class BackendStrategy(Protocol):
    """Structural contract every backend strategy honours.

    Strategies are stateless and pure: each method returns a sqlglot
    AST node (and, in ``emit_contains``, may invoke ``bind`` to
    register a bound parameter). The Protocol form lets third-party
    adapters satisfy it without inheriting from us — useful for
    out-of-tree Snowflake / BigQuery strategies.
    """

    backend: Backend

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder: ...
    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression: ...
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
    "hour": "toStartOfHour",
    "day": "toStartOfDay",
    "week": "toStartOfWeek",
    "month": "toStartOfMonth",
}


def _ident(name: str) -> exp.Identifier:
    """Return an Identifier for ``name``, quoted if the dialect tokenizer
    would treat it as a keyword (e.g. USING, SELECT, TABLE)."""
    toks = sqlglot.tokenize(name, dialect="postgres")
    needs_quote = not toks or toks[0].token_type not in (_TT.VAR, _TT.IDENTIFIER)
    return exp.to_identifier(name, quoted=needs_quote)


def _aliased_source(cube: Cube, resolve_sql: SqlResolver) -> exp.Expression:
    """Build the aliased FROM source for ``cube``.

    Dispatches on ``cube.resolved_source``: plain-table cubes return a
    ``Table`` node (``schema.tbl AS alias``); derived-table cubes return a
    ``Subquery`` wrapping the resolved SQL (``(<sql>) AS alias``). Both
    flow through the same ``resolve_sql`` placeholder substitution."""
    src = cube.resolved_source
    if isinstance(src, DerivedTable):
        return _aliased_derived(cube, src, resolve_sql)
    return _aliased_table(cube, src, resolve_sql)


def _aliased_table(cube: Cube, src: TableRef, resolve_sql: SqlResolver) -> exp.Table:
    """Build a ``Table`` AST node for ``cube`` with its alias attached."""
    resolved = resolve_sql(src.table)
    # Build the Table node manually rather than via exp.to_table() so that
    # reserved-word table names (e.g. ``using``) are properly quoted instead
    # of being emitted as bare keywords.
    parts = resolved.split(".", 1)
    if len(parts) == 2:
        tbl = exp.Table(this=_ident(parts[1]), db=_ident(parts[0]))
    else:
        tbl = exp.Table(this=_ident(parts[0]))
    tbl.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return tbl


def _aliased_derived(cube: Cube, src: DerivedTable, resolve_sql: SqlResolver) -> exp.Subquery:
    """Build a ``Subquery`` AST node over ``src.sql`` aliased to ``cube.alias``.

    The derived SQL goes through the same placeholder substitution as
    plain-table sources (``{tenant_schema}`` / ``{key}``). Parsing uses
    the cube's own dialect so backend-specific shapes (ClickHouse ARRAY
    JOIN, BigQuery UNNEST, etc.) survive the round-trip."""
    resolved = resolve_sql(src.sql)
    parsed = sqlglot.parse_one(resolved, dialect=dialect_for(cube.backend))
    if not isinstance(parsed, exp.Subquery):
        parsed = exp.Subquery(this=parsed)
    parsed.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return parsed


def _meta_subquery(cube: Cube, catalog: dict[str, Cube]) -> exp.Subquery:
    """Build a ``Subquery`` AST node over the catalogue snapshot for ``cube``."""
    body = build_meta_values(cube.name, catalog)
    sub = sqlglot.parse_one(body, dialect="postgres")
    if not isinstance(sub, exp.Subquery):
        sub = exp.Subquery(this=sub)
    sub.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return sub


class _StdSqlStrategy:
    """Shared base for backends whose dialect is handled correctly by
    sqlglot's stock renderer — Postgres, DuckDB, BigQuery, Snowflake.

    ``date_trunc`` goes through ``exp.Anonymous`` to keep the lowercase
    name verbatim. ``ILIKE`` AST gets transpiled per dialect on emit
    (BigQuery rewrites to ``LOWER(...) LIKE LOWER(...)``; PG / SF / DuckDB
    keep ``ILIKE`` natively). ``%v%`` always bakes into the bound value —
    the ``LIKE`` semantics survive the BigQuery transpile."""

    backend: Backend  # set by subclass

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, self.backend)

    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression:
        return exp.Anonymous(
            this="date_trunc",
            expressions=[exp.Literal.string(granularity), expr],
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
        inner = exp.Select().select(exp.alias_(series, bucket_alias))
        return inner


class PostgresStrategy(_StdSqlStrategy):
    """Postgres convention. Placeholders render as ``%(name)s``."""

    backend = Backend.POSTGRES


class DuckDBStrategy(_StdSqlStrategy):
    """DuckDB convention. Placeholders render as ``$name`` (the canonical
    DuckDB named-parameter syntax). Otherwise identical to Postgres —
    DuckDB shares ``ILIKE``, ``date_trunc``, and the aliased-table FROM
    shape."""

    backend = Backend.DUCKDB


class BigQueryStrategy(_StdSqlStrategy):
    """BigQuery convention. Placeholders render as ``@name``. sqlglot
    transpiles the ``ILIKE`` AST to ``LOWER(...) LIKE LOWER(...)`` on
    emit, so case-insensitive contains still works against a column."""

    backend = Backend.BIGQUERY

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
        inner = inner.select(exp.alias_(trunc_node, bucket_alias))
        unnest = exp.Unnest(
            expressions=[series],
            alias=exp.TableAlias(this=exp.to_identifier("d")),
        )
        inner = inner.from_(unnest)
        return inner


class SnowflakeStrategy(_StdSqlStrategy):
    """Snowflake convention. Placeholders render as ``:name``. ``ILIKE``
    and ``date_trunc`` are native to Snowflake — no transpilation needed."""

    backend = Backend.SNOWFLAKE

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
        inner = exp.Select(distinct=exp.Distinct()).select(exp.alias_(bucket, bucket_alias))
        inner = inner.from_(table_func)
        return inner


class ClickHouseStrategy:
    """ClickHouse convention. Placeholders are typed (``{name:Type}``);
    truncation uses the ``toStartOf<Hour|Day|Week|Month>`` family;
    ``contains`` passes the raw substring and emits
    ``positionCaseInsensitive``."""

    backend = Backend.CLICKHOUSE

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, Backend.CLICKHOUSE)

    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression:
        # ``exp.Anonymous`` keeps sqlglot from transpiling
        # ``toStartOfHour(...)`` into the canonical ``dateTrunc('HOUR', ...)``
        # form. The emitted SQL preserves the ClickHouse-idiomatic name.
        return exp.Anonymous(this=_CH_TRUNC[granularity], expressions=[expr])

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
        inner = inner.select(exp.alias_(bucket, bucket_alias))
        inner = inner.from_(exp.Anonymous(this="numbers", expressions=[day_diff]))
        return inner


class MetaStrategy:
    """Reflection cubes — materialised as a ``VALUES`` subquery at compile
    time. Inherits Postgres parameter and truncation conventions for the
    (rare) cases the compiler emits one against a META cube."""

    backend = Backend.META

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, Backend.META)

    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression:
        return exp.Anonymous(
            this="date_trunc",
            expressions=[exp.Literal.string(granularity), expr],
        )

    def emit_percentile(self, q: float, expr: exp.Expression) -> exp.Expression:
        # META cubes are caller-constructed VALUES tables and don't
        # carry numeric measures in practice — but satisfy the protocol
        # with the ANSI shape so a percentile measure on a META cube
        # doesn't crash the strategy lookup.
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


_DEFAULTS: dict[Backend, BackendStrategy] = {
    Backend.POSTGRES: PostgresStrategy(),
    Backend.CLICKHOUSE: ClickHouseStrategy(),
    Backend.META: MetaStrategy(),
    Backend.DUCKDB: DuckDBStrategy(),
    Backend.BIGQUERY: BigQueryStrategy(),
    Backend.SNOWFLAKE: SnowflakeStrategy(),
}


def strategy_for(
    backend: Backend,
    overrides: dict[Backend, BackendStrategy] | None = None,
) -> BackendStrategy:
    """Look up the strategy for ``backend``, honouring ``overrides``.

    Callers (tests, downstream packages) can pass ``overrides`` to
    swap in a different strategy without touching the global registry —
    e.g. a ``RecordingStrategy`` for delegation tests, or a custom
    ``SnowflakeStrategy`` shipped from outside this repo.
    """
    if overrides is not None and backend in overrides:
        return overrides[backend]
    if backend in _DEFAULTS:
        return _DEFAULTS[backend]
    raise KeyError(
        f"No registered BackendStrategy for {backend!r}. "
        "Pass one via the `strategies` kwarg on compile_query."
    )


def render(node: exp.Expression, backend: Backend) -> str:
    """Render a sqlglot node as the dialect-canonical SQL for ``backend``.

    ``normalize_functions=False`` keeps function names verbatim (so an
    ``exp.Anonymous(this="date_trunc")`` emits ``date_trunc(...)`` rather
    than ``DATE_TRUNC(...)``) — the existing assertions and the
    production code that consumes our SQL both expect the lowercase
    form."""
    return node.sql(
        dialect=dialect_for(backend),
        pretty=False,
        normalize_functions=False,
    )


__all__ = [
    "BackendStrategy",
    "BigQueryStrategy",
    "ClickHouseStrategy",
    "DuckDBStrategy",
    "MetaStrategy",
    "ParamBinder",
    "PostgresStrategy",
    "SnowflakeStrategy",
    "SqlResolver",
    "render",
    "strategy_for",
]
