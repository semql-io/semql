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
from sqlglot import exp

from semql.dialect import dialect_for, placeholder_for
from semql.introspect import build_meta_values
from semql.model import Backend, Cube

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
    def emit_contains(
        self, field: exp.Expression, value: str, bind: ParamBinder
    ) -> exp.Expression: ...
    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,
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


def _aliased_table(cube: Cube, resolve_sql: SqlResolver) -> exp.Table:
    """Build a ``Table`` AST node for ``cube`` with its alias attached."""
    resolved = resolve_sql(cube.table)
    tbl = exp.to_table(resolved, dialect="postgres")
    tbl.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return tbl


def _meta_subquery(cube: Cube, catalog: dict[str, Cube]) -> exp.Subquery:
    """Build a ``Subquery`` AST node over the catalogue snapshot for ``cube``."""
    body = build_meta_values(cube.name, catalog)
    sub = sqlglot.parse_one(body, dialect="postgres")
    if not isinstance(sub, exp.Subquery):
        sub = exp.Subquery(this=sub)
    sub.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
    return sub


class PostgresStrategy:
    """Postgres / DuckDB convention. Placeholders use ``%(name)s``;
    truncation emits ``date_trunc('<gran>', expr)``; ``contains`` bakes
    ``%v%`` into the bound value and emits ``ILIKE``."""

    backend = Backend.POSTGRES

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        return placeholder_for(name, dim_type, Backend.POSTGRES)

    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression:
        # ``exp.Anonymous`` bypasses sqlglot's DATE_TRUNC normalisation so
        # the emitted SQL stays as lowercase ``date_trunc`` — matches the
        # existing assertion contract and what production code consumes.
        return exp.Anonymous(
            this="date_trunc",
            expressions=[exp.Literal.string(granularity), expr],
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
        return _aliased_table(cube, resolve_sql)


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
        return _aliased_table(cube, resolve_sql)


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


# ---------------------------------------------------------------------------
# Registry + DI hybrid
# ---------------------------------------------------------------------------


_DEFAULTS: dict[Backend, BackendStrategy] = {
    Backend.POSTGRES: PostgresStrategy(),
    Backend.CLICKHOUSE: ClickHouseStrategy(),
    Backend.META: MetaStrategy(),
    # DuckDB shares the Postgres placeholder convention; revisit if/when
    # we add a dedicated DuckDB strategy (current behaviour preserved).
    Backend.DUCKDB: PostgresStrategy(),
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
    "ClickHouseStrategy",
    "MetaStrategy",
    "ParamBinder",
    "PostgresStrategy",
    "SqlResolver",
    "render",
    "strategy_for",
]
