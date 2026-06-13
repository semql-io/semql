"""sqlglot adapters — the seam Commit 2 of the sqlglot migration drops into.

This module is *additive*: it ships a SemQL ``Dialect`` → sqlglot
dialect name mapping and a typed-placeholder factory, but does not yet
participate in ``compile_query``. The next migration commit will
swap the compiler's string-building body onto sqlglot AST primitives,
and this module's helpers are what make that swap mechanical.

The two pieces here:
- ``dialect_for(backend)`` — the dialect string sqlglot's parser /
  renderer expects (``"postgres"``, ``"clickhouse"``, ...).
- ``placeholder_for(name, dim_type, backend)`` — an ``exp.Placeholder``
  AST node whose ``.sql(dialect=...)`` output matches the
  ``DialectStrategy.placeholder`` strings used today
  (``%(p0)s`` for Postgres, ``{p0:String}`` for ClickHouse).

Stock sqlglot renders ``exp.Placeholder(this="p0", kind=String)`` as
``{p0: Nullable(String)}`` for ClickHouse. Our existing emitted SQL —
and the production code reading it — uses the tighter ``{p0:String}``
form, so this module registers an override against the ``clickhouse``
dialect name so the canonical lookup picks it up.
"""

from __future__ import annotations

from sqlglot import exp
from sqlglot.dialects.clickhouse import ClickHouse
from sqlglot.dialects.dialect import Dialect as SqlglotDialect

from semql.model import Dialect

# Mirrors the mapping in ``backend.py``. Duplicated here on purpose —
# the dialect lives in string-land; this module lives in sqlglot-land.
# Once Commit 2 lands and the compiler uses sqlglot for emission, the
# dialect's typed-placeholder path will delegate here.
_CH_DIM_TYPE_TO_CH_TYPE: dict[str, str] = {
    "string": "String",
    "number": "Float64",
    "time": "DateTime",
    "bool": "UInt8",
    "uuid": "String",
}


_BACKEND_TO_DIALECT: dict[Dialect, str] = {
    Dialect.POSTGRES: "postgres",
    Dialect.CLICKHOUSE: "clickhouse",
    Dialect.DUCKDB: "duckdb",
    Dialect.BIGQUERY: "bigquery",
    Dialect.SNOWFLAKE: "snowflake",
    # META cubes are materialised as portable VALUES literals; the
    # Postgres dialect is neutral enough to parse them without rewrites.
    Dialect.META: "postgres",
}


def dialect_for(backend: Dialect) -> str:
    """Return the sqlglot dialect string for ``backend``.

    Unknown backends raise ``KeyError`` so a missing mapping is loud,
    not silent."""
    return _BACKEND_TO_DIALECT[backend]


# ---------------------------------------------------------------------------
# ClickHouse dialect override — tighter ``{name:Type}`` rendering
# ---------------------------------------------------------------------------


class _ChDialect(ClickHouse):
    """ClickHouse dialect with our tightened placeholder convention.

    The placeholder's ``kind`` arg is read as a plain string (the CH
    type name); the override emits ``{name:Type}`` rather than the
    stock ``{name: Nullable(Type)}``."""

    class Generator(ClickHouse.Generator):  # type: ignore[misc,valid-type]
        def placeholder_sql(self, expression: exp.Placeholder) -> str:
            kind = expression.args.get("kind")
            if kind is None:
                return f"{{{expression.name}}}"
            if isinstance(kind, str):
                return f"{{{expression.name}:{kind}}}"
            return f"{{{expression.name}:{self.sql(kind)}}}"


# Replace the registered ``clickhouse`` dialect class with our override.
# sqlglot's ``Dialect.get_or_raise("clickhouse")`` consults this dict, so
# any ``.sql(dialect="clickhouse")`` call anywhere in the process picks
# up the tighter placeholder rendering. (Importing this module is the
# only thing that activates the override — it's not a global state hack
# from import-time-of-anything-else.)
SqlglotDialect.classes["clickhouse"] = _ChDialect


def placeholder_for(name: str, dim_type: str, backend: Dialect) -> exp.Placeholder:
    """Build an ``exp.Placeholder`` AST for the given backend.

    The returned node's ``.sql(dialect=dialect_for(backend))`` matches
    the strings the existing ``DialectStrategy.placeholder`` emits. Use
    this when constructing predicates / projections via sqlglot AST in
    a future Commit-2 path."""
    p = exp.Placeholder(this=name)
    if backend is Dialect.CLICKHOUSE:
        p.set("kind", _CH_DIM_TYPE_TO_CH_TYPE.get(dim_type, "String"))
    return p


__all__ = ["dialect_for", "placeholder_for"]
