"""Per-backend strategy objects — the dialect-specific seam.

The compiler stays dialect-agnostic for the parts it can: identifier
resolution, join graph BFS, GROUP BY composition, parameter-name
allocation, ordering, IR. The strategy owns the rest:

- ``placeholder(name, dim_type)`` — the bound-param syntax
  (``%(name)s`` vs ``{name:Type}`` vs ``@name``).
- ``trunc(granularity, sql_expr)`` — date truncation (``date_trunc``
  vs ``toStartOfHour`` family).
- ``emit_contains(field_sql, value, bind)`` — substring search,
  including the *value transform* tied to that shape (Postgres bakes
  ``%v%`` into the bound value; ClickHouse passes the raw substring).
- ``emit_source(cube, catalog, resolve_sql)`` — how a cube becomes a
  ``FROM`` source. Vanilla cubes emit ``table AS alias``; META cubes
  materialise as a ``VALUES`` subquery.

The Protocol is structural (``typing.Protocol``) so sqlglot's dialect
classes — or a downstream Snowflake / BigQuery adapter — can satisfy
it without importing from this module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from semql.introspect import build_meta_values
from semql.model import Backend, Cube

ParamBinder = Callable[[Any, str], str]
"""Bind ``(value, dim_type)`` and return the placeholder SQL for it."""

SqlResolver = Callable[[str], str]
"""Resolve ``{alias}`` / ``{schema}`` placeholders in a SQL fragment."""


@runtime_checkable
class BackendStrategy(Protocol):
    """Structural contract every backend strategy honours.

    Strategies are stateless and pure: each method returns a SQL
    fragment (and, in ``emit_contains``, may invoke ``bind`` to register
    a bound parameter). The Protocol form lets third-party adapters
    satisfy it without inheriting from us — useful for sqlglot dialect
    classes or out-of-tree Snowflake / BigQuery strategies.
    """

    backend: Backend

    def placeholder(self, name: str, dim_type: str) -> str: ...
    def trunc(self, granularity: str, sql_expr: str) -> str: ...
    def emit_contains(self, field_sql: str, value: str, bind: ParamBinder) -> str: ...
    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


_CH_DIM_TYPE_TO_CH_TYPE: dict[str, str] = {
    "string": "String",
    "number": "Float64",
    "time": "DateTime",
    "bool": "UInt8",
    # CH UUIDs are quoted strings on the wire; bind as String.
    "uuid": "String",
}

_CH_TRUNC: dict[str, str] = {
    "hour": "toStartOfHour",
    "day": "toStartOfDay",
    "week": "toStartOfWeek",
    "month": "toStartOfMonth",
}


class PostgresStrategy:
    """Postgres / DuckDB convention. Placeholders use ``%(name)s``;
    truncation uses ``date_trunc``; ``contains`` bakes ``%v%`` into
    the bound value and emits ``ILIKE``."""

    backend = Backend.POSTGRES

    def placeholder(self, name: str, dim_type: str) -> str:  # noqa: ARG002
        return f"%({name})s"

    def trunc(self, granularity: str, sql_expr: str) -> str:
        return f"date_trunc('{granularity}', {sql_expr})"

    def emit_contains(self, field_sql: str, value: str, bind: ParamBinder) -> str:
        return f"{field_sql} ILIKE {bind(f'%{value}%', 'string')}"

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> str:
        return f"{resolve_sql(cube.table)} AS {cube.alias}"


class ClickHouseStrategy:
    """ClickHouse convention. Placeholders are typed (``{name:Type}``);
    truncation uses the ``toStartOf<Hour|Day|Week|Month>`` family;
    ``contains`` passes the raw substring and emits
    ``positionCaseInsensitive``."""

    backend = Backend.CLICKHOUSE

    def placeholder(self, name: str, dim_type: str) -> str:
        ch_type = _CH_DIM_TYPE_TO_CH_TYPE.get(dim_type, "String")
        return f"{{{name}:{ch_type}}}"

    def trunc(self, granularity: str, sql_expr: str) -> str:
        return f"{_CH_TRUNC[granularity]}({sql_expr})"

    def emit_contains(self, field_sql: str, value: str, bind: ParamBinder) -> str:
        return f"positionCaseInsensitive({field_sql}, {bind(value, 'string')}) > 0"

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> str:
        return f"{resolve_sql(cube.table)} AS {cube.alias}"


class MetaStrategy:
    """Reflection cubes — materialised as a ``VALUES`` subquery at compile
    time. Inherits the Postgres parameter and truncation conventions for
    the (rare) cases the compiler emits a parameter against a META cube."""

    backend = Backend.META

    def placeholder(self, name: str, dim_type: str) -> str:  # noqa: ARG002
        return f"%({name})s"

    def trunc(self, granularity: str, sql_expr: str) -> str:
        return f"date_trunc('{granularity}', {sql_expr})"

    def emit_contains(self, field_sql: str, value: str, bind: ParamBinder) -> str:
        return f"{field_sql} ILIKE {bind(f'%{value}%', 'string')}"

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,  # noqa: ARG002
    ) -> str:
        return f"{build_meta_values(cube.name, catalog)} AS {cube.alias}"


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


__all__ = [
    "BackendStrategy",
    "ClickHouseStrategy",
    "MetaStrategy",
    "ParamBinder",
    "PostgresStrategy",
    "SqlResolver",
    "strategy_for",
]
