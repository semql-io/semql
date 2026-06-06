"""Strategy DI tests: prove that compile.py actually delegates, and
that callers can swap in a custom strategy.

Without these the strategy extraction can regress silently —
compile.py could grow an inline dialect branch and the existing 51
test_compile assertions wouldn't catch it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from semql import Cube, Dimension, Filter, Measure, SemanticQuery, TimeWindow
from semql.backend import (
    BackendStrategy,
    PostgresStrategy,
    SqlResolver,
)
from semql.compile import compile_query
from semql.model import Backend


def _as_strategy_map(
    s: object, backend: Backend = Backend.POSTGRES
) -> dict[Backend, BackendStrategy]:
    """Helper: cast a structurally-conformant strategy into the
    Protocol-typed dict the compiler accepts. mypy can't infer Protocol
    conformance inside a dict literal, so we name it here once."""
    return {backend: cast(BackendStrategy, s)}


@dataclass
class RecordingStrategy:
    """Wraps a real strategy and records every method call. The wrapped
    strategy still does the actual work, so the compiler output is
    unchanged."""

    inner: BackendStrategy
    placeholder_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    trunc_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    contains_calls: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    source_calls: list[str] = field(default_factory=list[str])

    @property
    def backend(self) -> Backend:
        return self.inner.backend

    def placeholder(self, name: str, dim_type: str) -> str:
        self.placeholder_calls.append((name, dim_type))
        return self.inner.placeholder(name, dim_type)

    def trunc(self, granularity: str, sql_expr: str) -> str:
        self.trunc_calls.append((granularity, sql_expr))
        return self.inner.trunc(granularity, sql_expr)

    def emit_contains(
        self,
        field_sql: str,
        value: str,
        bind: Any,  # noqa: ANN401
    ) -> str:
        self.contains_calls.append((field_sql, value))
        return self.inner.emit_contains(field_sql, value, bind)

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],
        resolve_sql: SqlResolver,
    ) -> str:
        self.source_calls.append(cube.name)
        return self.inner.emit_source(cube, catalog, resolve_sql)


def _orders_catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        backend=Backend.POSTGRES,
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
    rec = RecordingStrategy(PostgresStrategy())
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    compile_query(q, _orders_catalog(), strategies=_as_strategy_map(rec))
    assert rec.placeholder_calls == [("p0", "string")]


def test_compiler_delegates_trunc_only_when_granularity_set() -> None:
    rec = RecordingStrategy(PostgresStrategy())
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
        strategies=_as_strategy_map(rec),
    )
    assert rec.trunc_calls == []

    # With granularity → exactly one trunc call (SELECT projection;
    # GROUP BY reuses the alias by default).
    rec2 = RecordingStrategy(PostgresStrategy())
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
        strategies=_as_strategy_map(rec2),
    )
    assert len(rec2.trunc_calls) == 1
    assert rec2.trunc_calls[0][0] == "day"


def test_compiler_delegates_emit_contains_only_when_op_is_contains() -> None:
    rec = RecordingStrategy(PostgresStrategy())
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    compile_query(q, _orders_catalog(), strategies=_as_strategy_map(rec))
    assert rec.contains_calls == []

    rec2 = RecordingStrategy(PostgresStrategy())
    q2 = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.region", op="contains", values=["us"])],
    )
    compile_query(q2, _orders_catalog(), strategies=_as_strategy_map(rec2))
    assert len(rec2.contains_calls) == 1
    # field_sql is post-resolution — the {o} alias is already substituted.
    assert rec2.contains_calls[0] == ("o.region", "us")


def test_compiler_delegates_emit_source_for_every_from_cube() -> None:
    rec = RecordingStrategy(PostgresStrategy())
    q = SemanticQuery(measures=["orders.count"])
    compile_query(q, _orders_catalog(), strategies=_as_strategy_map(rec))
    assert rec.source_calls == ["orders"]


# ---------------------------------------------------------------------------
# Strategy-override test — third-party backend pattern. Documents the
# DI surface for a downstream Snowflake / BigQuery adapter.
# ---------------------------------------------------------------------------


class _FakeSnowflakeStrategy:
    backend = Backend.SNOWFLAKE

    def placeholder(self, name: str, dim_type: str) -> str:  # noqa: ARG002
        return f":{name}"

    def trunc(self, granularity: str, sql_expr: str) -> str:
        return f"DATE_TRUNC('{granularity.upper()}', {sql_expr})"

    def emit_contains(
        self,
        field_sql: str,
        value: str,
        bind: Any,  # noqa: ANN401
    ) -> str:
        return f"CONTAINS({field_sql}, {bind(value, 'string')})"

    def emit_source(
        self,
        cube: Cube,
        catalog: dict[str, Cube],  # noqa: ARG002
        resolve_sql: SqlResolver,
    ) -> str:
        return f"{resolve_sql(cube.table)} AS {cube.alias}"


def test_third_party_strategy_through_override_kwarg() -> None:
    """A downstream Snowflake adapter only needs to define a strategy
    and pass it via ``strategies={...}`` — no fork of semql required."""
    orders = Cube(
        name="orders",
        backend=Backend.SNOWFLAKE,
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
        strategies=_as_strategy_map(_FakeSnowflakeStrategy(), backend=Backend.SNOWFLAKE),
    )
    # The Snowflake placeholder convention (`:name`) shows up in the SQL.
    assert ":p0" in out.sql
    assert out.params == {"p0": "us"}
