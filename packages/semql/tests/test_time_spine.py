"""Time spine + ``fill_nulls_with`` — daily/weekly/monthly questions return
rows for every bucket in range, even when the underlying data has gaps.

Phase A scope:
- Per-query switch via ``TimeWindow.fill_nulls_with``.
- Only when a query has a time_dimension with granularity and no
  non-time dimensions (cartesian fill with dims is Phase B).
- Dialect coverage: Postgres + DuckDB (the std-sql ``generate_series``
  shape). ClickHouse / BigQuery / Snowflake raise a clear "not yet
  supported" error.

The compiler wraps the inner aggregation in a CTE, builds a parallel
spine CTE via ``DialectStrategy.emit_time_spine``, and the outer
SELECT does ``spine LEFT JOIN agg`` with ``COALESCE(measure, fill)``
per measure.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.errors import CompileError
from semql.model import Cube, Dialect, Measure, TimeDimension
from semql.spec import SemanticQuery, TimeWindow

CONTEXT = {"schema": "test"}


def _pg_orders() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week", "month"),
            ),
        ],
    )


def _duckdb_orders() -> Cube:
    cube = _pg_orders()
    return cube.model_copy(update={"backend": Dialect.DUCKDB})


def _ch_events() -> Cube:
    return Cube(
        name="events",
        backend=Dialect.CLICKHOUSE,
        table="{schema}.events",
        alias="e",
        measures=[Measure(name="count", sql="*", agg="count")],
        time_dimensions=[
            TimeDimension(
                name="ts",
                sql="{e}.ts",
                granularities=("day", "week", "month"),
            ),
        ],
    )


def _q(
    cube_field_prefix: str,
    time_dim: str,
    *,
    fill_nulls_with: int | None = 0,
    granularity: str | None = "day",
    measures: list[str] | None = None,
    dimensions: list[str] | None = None,
) -> SemanticQuery:
    return SemanticQuery(
        measures=measures or [f"{cube_field_prefix}.revenue"],
        dimensions=dimensions or [],
        time_dimension=TimeWindow(
            dimension=f"{cube_field_prefix}.{time_dim}",
            granularity=granularity,  # type: ignore[arg-type]
            range=("2024-01-01", "2024-02-01"),
            fill_nulls_with=fill_nulls_with,
        ),
    )


# ---------------------------------------------------------------------------
# Happy paths — PG + DuckDB spine emission
# ---------------------------------------------------------------------------


def test_fill_nulls_emits_spine_cte_postgres() -> None:
    cat = {"orders": _pg_orders()}
    compiled = compile_query(_q("orders", "created_at"), cat, context=CONTEXT)
    sql = compiled.sql
    # CTEs for the inner aggregation and the spine itself.
    assert "WITH" in sql.upper()
    assert "spine" in sql.lower()
    assert "generate_series" in sql.lower()
    # Outer SELECT joins the spine LEFT to the aggregation and
    # COALESCEs the measure to the fill value.
    assert "LEFT JOIN" in sql.upper()
    assert "COALESCE" in sql.upper()
    assert ", 0" in sql or ",0" in sql  # the fill value


def test_fill_nulls_emits_spine_cte_duckdb() -> None:
    cat = {"orders": _duckdb_orders()}
    compiled = compile_query(_q("orders", "created_at"), cat, context=CONTEXT)
    sql = compiled.sql
    assert "spine" in sql.lower()
    assert "generate_series" in sql.lower()
    assert "COALESCE" in sql.upper()


def test_fill_nulls_columns_match_unfilled_query() -> None:
    """Adding fill_nulls_with must not change the output schema —
    consumers can flip the switch without column-rename churn."""
    cat = {"orders": _pg_orders()}
    q_filled = _q("orders", "created_at", fill_nulls_with=0)
    q_bare = _q("orders", "created_at", fill_nulls_with=None)
    assert (
        compile_query(q_filled, cat, context=CONTEXT).columns
        == compile_query(q_bare, cat, context=CONTEXT).columns
    )


def test_fill_nulls_with_count_measure_also_coalesced() -> None:
    cat = {"orders": _pg_orders()}
    q = _q(
        "orders",
        "created_at",
        fill_nulls_with=0,
        measures=["orders.revenue", "orders.count"],
    )
    sql = compile_query(q, cat, context=CONTEXT).sql
    # Both measures get a COALESCE — the fill value applies uniformly.
    assert sql.upper().count("COALESCE") >= 2


# ---------------------------------------------------------------------------
# Phase A restrictions — clear errors for unsupported shapes
# ---------------------------------------------------------------------------


def test_fill_nulls_requires_granularity() -> None:
    cat = {"orders": _pg_orders()}
    q = _q("orders", "created_at", granularity=None)
    with pytest.raises(CompileError, match="granularity"):
        compile_query(q, cat, context=CONTEXT)


def test_fill_nulls_rejects_non_time_dimensions() -> None:
    """Phase B will support spine × dim cartesian fill; for now a
    clear error keeps callers from getting silently wrong results."""
    from semql.model import Dimension

    cube = _pg_orders()
    cube_with_dim = cube.model_copy(
        update={"dimensions": [Dimension(name="region", sql="{o}.region", type="string")]}
    )
    cat = {"orders": cube_with_dim}
    q = _q("orders", "created_at", dimensions=["orders.region"])
    with pytest.raises(CompileError, match="non-time dimensions"):
        compile_query(q, cat, context=CONTEXT)


def test_fill_nulls_emits_spine_clickhouse() -> None:
    cat = {"events": _ch_events()}
    q = _q("events", "ts", measures=["events.count"])
    sql = compile_query(q, cat, context=CONTEXT).sql
    # CH spine uses ``numbers()`` + ``toStartOf<Gran>(addDays(...))``
    # because it has no ``generate_series``.
    assert "spine" in sql.lower()
    assert "numbers(" in sql.lower()
    assert "toStartOfDay" in sql or "toStartOfWeek" in sql or "toStartOfMonth" in sql
    assert "COALESCE" in sql.upper()


def test_fill_nulls_emits_spine_bigquery() -> None:
    cube = Cube(
        name="orders",
        backend=Dialect.BIGQUERY,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        time_dimensions=[
            TimeDimension(
                name="created_at", sql="{o}.created_at", granularities=("day", "week", "month")
            ),
        ],
    )
    cat = {"orders": cube}
    q = _q("orders", "created_at", measures=["orders.count"])
    sql = compile_query(q, cat, context=CONTEXT).sql
    # BQ spine: UNNEST(GENERATE_DATE_ARRAY(...))
    assert "spine" in sql.lower()
    assert "GENERATE_DATE_ARRAY" in sql.upper()
    assert "UNNEST" in sql.upper()
    assert "COALESCE" in sql.upper()


def test_fill_nulls_emits_spine_snowflake() -> None:
    cube = Cube(
        name="orders",
        backend=Dialect.SNOWFLAKE,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count")],
        time_dimensions=[
            TimeDimension(
                name="created_at", sql="{o}.created_at", granularities=("day", "week", "month")
            ),
        ],
    )
    cat = {"orders": cube}
    q = _q("orders", "created_at", measures=["orders.count"])
    sql = compile_query(q, cat, context=CONTEXT).sql
    # SF spine: TABLE(GENERATOR(ROWCOUNT => ...)) + SEQ4()
    assert "spine" in sql.lower()
    assert "GENERATOR" in sql.upper()
    assert "SEQ4" in sql.upper()
    assert "COALESCE" in sql.upper()


# ---------------------------------------------------------------------------
# Off path — fill_nulls_with=None behaves exactly as before
# ---------------------------------------------------------------------------


def test_fill_nulls_none_emits_no_spine() -> None:
    cat = {"orders": _pg_orders()}
    sql = compile_query(_q("orders", "created_at", fill_nulls_with=None), cat, context=CONTEXT).sql
    assert "spine" not in sql.lower()
    assert "generate_series" not in sql.lower()
    assert "COALESCE" not in sql.upper()
