# pyright: reportPrivateImportUsage=false
"""Dialect DI tests: prove that compile.py actually delegates, and
that callers can swap in a custom dialect.

Without these the dialect extraction can regress silently —
compile.py could grow an inline dialect branch and the existing
test_compile assertions wouldn't catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from semql import Cube, Dimension, Filter, Measure, SemanticQuery, TimeWindow
from semql.backend import (
    DialectStrategy,
    ParamBinder,
    PostgresDialect,
    SqlResolver,
)
from semql.compile import compile_query
from semql.model import Dialect
from sqlglot import exp


def _as_dialect_map(
    s: object, backend: Dialect = Dialect.POSTGRES
) -> dict[Dialect, DialectStrategy]:
    """Helper: cast a structurally-conformant dialect into the
    Protocol-typed dict the compiler accepts. mypy can't infer Protocol
    conformance inside a dict literal, so we name it here once."""
    return {backend: cast(DialectStrategy, s)}


@dataclass
class RecordingDialect:
    """Wraps a real dialect and records every method call. The wrapped
    dialect still does the actual work, so the compiler output is
    unchanged."""

    inner: DialectStrategy
    placeholder_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    trunc_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    contains_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    source_calls: list[str] = field(default_factory=list[str])

    @property
    def backend(self) -> Dialect:
        return self.inner.backend

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:
        self.placeholder_calls.append((name, dim_type))
        return self.inner.placeholder(name, dim_type)

    def trunc(self, granularity: str, e: exp.Expression) -> exp.Expression:
        # Stringify the expression for the recorded call so assertions stay
        # readable. The inner dialect still drives the actual node shape.
        self.trunc_calls.append((granularity, e.sql(dialect="postgres")))
        return self.inner.trunc(granularity, e)

    def emit_contains(
        self,
        f: exp.Expression,
        value: str,
        bind: ParamBinder,
    ) -> exp.Expression:
        self.contains_calls.append((f.sql(dialect="postgres"), value))
        return self.inner.emit_contains(f, value, bind)

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,
    ) -> exp.Expression:
        self.source_calls.append(cube.name)
        return self.inner.emit_source(cube, catalog, resolve_sql)


def _orders_catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    return {"orders": orders}


def test_compiler_delegates_placeholder_for_filter_value() -> None:
    rec = RecordingDialect(PostgresDialect())
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    compile_query(q, _orders_catalog(), dialects=_as_dialect_map(rec))
    assert rec.placeholder_calls == [("p0", "string")]


def test_compiler_delegates_trunc_only_when_granularity_set() -> None:
    rec = RecordingDialect(PostgresDialect())
    catalog = _orders_catalog()
    orders = catalog["orders"]
    # Add a time dimension to exercise trunc().
    catalog["orders"] = orders.model_copy(
        update={
            "time_dimensions": [
                __import__("semql").TimeDimension(name="created_at", sql="{o}.created_at"),
            ],
        }
    )

    # Without granularity → no trunc call.
    compile_query(
        SemanticQuery(
            measures=["orders.count"],
            time_dimension=TimeWindow(
                dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
            ),
        ),
        catalog,
        dialects=_as_dialect_map(rec),
    )
    assert rec.trunc_calls == []

    # With granularity → exactly one trunc call (SELECT projection;
    # GROUP BY reuses the alias by default).
    rec2 = RecordingDialect(PostgresDialect())
    compile_query(
        SemanticQuery(
            measures=["orders.count"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        catalog,
        dialects=_as_dialect_map(rec2),
    )
    assert len(rec2.trunc_calls) == 1
    assert rec2.trunc_calls[0][0] == "day"


def test_compiler_delegates_emit_contains_only_when_op_is_contains() -> None:
    rec = RecordingDialect(PostgresDialect())
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    compile_query(q, _orders_catalog(), dialects=_as_dialect_map(rec))
    assert rec.contains_calls == []

    rec2 = RecordingDialect(PostgresDialect())
    q2 = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="contains", values=["us"])],
    )
    compile_query(q2, _orders_catalog(), dialects=_as_dialect_map(rec2))
    assert len(rec2.contains_calls) == 1
    # field is the resolved AST stringified; alias substitution already happened.
    assert rec2.contains_calls[0] == ("o.region", "us")


def test_compiler_delegates_emit_source_for_every_from_cube() -> None:
    rec = RecordingDialect(PostgresDialect())
    q = SemanticQuery(measures=["orders.count"])
    compile_query(q, _orders_catalog(), dialects=_as_dialect_map(rec))
    assert rec.source_calls == ["orders"]


# ---------------------------------------------------------------------------
# Dialect-override test — third-party backend pattern. Documents the
# DI surface for a downstream Snowflake / BigQuery adapter.
# ---------------------------------------------------------------------------


class _FakeSnowflakeDialect:
    backend = Dialect.SNOWFLAKE

    def placeholder(self, name: str, dim_type: str) -> exp.Placeholder:  # noqa: ARG002
        # Snowflake convention is ``:name`` — stash that as the raw form
        # by leaving the placeholder un-kinded (sqlglot's Snowflake
        # generator already emits ``:name`` for un-kinded Placeholders).
        return exp.Placeholder(this=name)

    def trunc(self, granularity: str, expr: exp.Expression) -> exp.Expression:
        return exp.Anonymous(
            this="DATE_TRUNC",
            expressions=[exp.Literal.string(granularity.upper()), expr],
        )

    def emit_contains(
        self,
        field: exp.Expression,
        value: str,
        bind: ParamBinder,
    ) -> exp.Expression:
        ph = bind(value, "string")
        return exp.Anonymous(this="CONTAINS", expressions=[field, ph])

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> exp.Expression:
        resolved = resolve_sql(cube.table)
        tbl = exp.to_table(resolved)
        tbl.set("alias", exp.TableAlias(this=exp.to_identifier(cube.alias)))
        return tbl


def test_third_party_strategy_through_override_kwarg() -> None:
    """A downstream Snowflake adapter only needs to define a dialect
    and pass it via ``dialects={...}`` — no fork of semql required."""
    orders = Cube(
        name="orders",
        backend=Dialect.SNOWFLAKE,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    catalog = {"orders": orders}
    out = compile_query(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
        ),
        catalog,
        dialects=_as_dialect_map(_FakeSnowflakeDialect(), backend=Dialect.SNOWFLAKE),
    )
    # The Snowflake placeholder convention (`:name`) shows up in the SQL.
    assert ":p0" in out.sql
    assert out.params == {"p0": "us"}


def test_unknown_value_aliased_for_lint() -> None:
    """Anchor for the unused Any import — kept so the noqa survives reformat."""
    x: Any = None  # noqa: ANN401
    assert x is None
