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

import duckdb
import pytest
from semql.compile import compile_query
from semql.errors import CompileError
from semql.model import Cube, Dialect, Measure, TimeDimension
from semql.spec import InlineDerived, SemanticQuery, TimeWindow

CONTEXT = {"schema": "test"}


def _pg_orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
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
    return cube.model_copy(update={"dialect": Dialect.DUCKDB})


def _ch_events() -> Cube:
    return Cube(
        name="events",
        dialect=Dialect.CLICKHOUSE,
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


def test_duckdb_spine_casts_bound_window_values_to_temporal_types() -> None:
    compiled = compile_query(
        _q("orders", "created_at"), {"orders": _duckdb_orders()}, context=CONTEXT
    )
    assert compiled.sql.count("CAST(") >= 2
    assert "AS TIMESTAMP" in compiled.sql


def test_duckdb_dense_fill_executes_and_returns_missing_daily_buckets() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE SCHEMA test")
        con.execute("CREATE TABLE test.orders (created_at TIMESTAMP, amount DOUBLE)")
        con.execute(
            "INSERT INTO test.orders VALUES "
            "('2024-01-01 12:00:00', 5.0), ('2024-01-03 08:00:00', 7.0)"
        )
        query = _q("orders", "created_at").model_copy(
            update={"order": [("orders.created_at", "asc")]}
        )
        compiled = compile_query(query, {"orders": _duckdb_orders()}, context=CONTEXT)
        rows = con.execute(compiled.sql, compiled.params).fetchall()
        assert len(rows) == 31
        assert rows[0][0].date().isoformat() == "2024-01-01"
        assert rows[0][1] == 5.0
        assert rows[1][0].date().isoformat() == "2024-01-02"
        assert rows[1][1] == 0
        assert rows[2][0].date().isoformat() == "2024-01-03"
        assert rows[2][1] == 7.0
        assert rows[-1][0].date().isoformat() == "2024-01-31"
        assert rows[-1][1] == 0
    finally:
        con.close()


@pytest.mark.parametrize(
    ("granularity", "window", "facts", "expected"),
    [
        pytest.param(
            "day",
            ("2024-01-01", "2024-01-03"),
            [("2024-01-01 08:00:00", 5.0), ("2024-01-02 10:00:00", 7.0)],
            [("2024-01-01", 5.0), ("2024-01-02", 7.0)],
            id="aligned-exclusive-end",
        ),
        pytest.param(
            "day",
            ("2024-01-01 12:00:00", "2024-01-02 12:00:00"),
            [("2024-01-02 10:00:00", 7.0)],
            [("2024-01-01", 0.0), ("2024-01-02", 7.0)],
            id="unaligned-final-intersection",
        ),
        pytest.param(
            "day",
            ("2024-01-01 12:00:00", "2024-01-01 13:00:00"),
            [("2024-01-01 12:30:00", 9.0)],
            [("2024-01-01", 9.0)],
            id="substep-window",
        ),
        pytest.param(
            "month",
            ("2024-01-31 12:00:00", "2024-03-01 00:00:00"),
            [("2024-01-31 18:00:00", 11.0), ("2024-02-29 20:00:00", 13.0)],
            [("2024-01-01", 11.0), ("2024-02-01", 13.0)],
            id="calendar-month",
        ),
    ],
)
def test_duckdb_dense_spine_preserves_every_intersecting_bucket(
    granularity: str,
    window: tuple[str, str],
    facts: list[tuple[str, float]],
    expected: list[tuple[str, float]],
) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE SCHEMA test")
        con.execute("CREATE TABLE test.orders (created_at TIMESTAMP, amount DOUBLE)")
        con.executemany("INSERT INTO test.orders VALUES (?, ?)", facts)
        query = SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity=granularity,  # type: ignore[arg-type]
                range=window,
                fill_nulls_with=0,
            ),
            order=[("orders.created_at", "asc")],
        )
        compiled = compile_query(query, {"orders": _duckdb_orders()}, context=CONTEXT)
        rows = con.execute(compiled.sql, compiled.params).fetchall()
        observed = [(bucket.date().isoformat(), amount) for bucket, amount in rows]
        assert observed == expected
    finally:
        con.close()


@pytest.mark.parametrize(
    ("operation", "expected_values"),
    [
        pytest.param("ratio", [7.5, None, 60.0], id="ratio"),
        pytest.param("sum", [17.0, 0.0, 61.0], id="sum"),
        pytest.param("diff", [13.0, 0.0, 59.0], id="diff"),
    ],
)
def test_duckdb_dense_fill_preserves_complete_derived_projection_contract(
    operation: str, expected_values: list[float | None]
) -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE SCHEMA test")
        con.execute("CREATE TABLE test.orders (created_at TIMESTAMP, amount DOUBLE)")
        con.execute(
            "INSERT INTO test.orders VALUES "
            "('2024-01-01 08:00:00', 10.0), "
            "('2024-01-01 09:00:00', 5.0), "
            "('2024-01-03 10:00:00', 60.0)"
        )
        query = SemanticQuery(
            measures=["orders.revenue"],
            aliases={"gross": "orders.revenue"},
            derived_measures=[
                InlineDerived(
                    name="derived_value",
                    op=operation,  # type: ignore[arg-type]
                    operands=["orders.revenue", "orders.count"],
                )
            ],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2024-01-01", "2024-01-04"),
                fill_nulls_with=0,
            ),
            order=[("orders.created_at", "asc")],
        )
        compiled = compile_query(query, {"orders": _duckdb_orders()}, context=CONTEXT)
        cursor = con.execute(compiled.sql, compiled.params)
        result_columns = [item[0] for item in cursor.description]
        rows = cursor.fetchall()

        expected_columns = ["created_at_day", "gross", "derived_value"]
        assert compiled.columns == expected_columns
        assert result_columns == expected_columns
        assert [meta.name for meta in compiled.column_meta] == expected_columns
        assert [output.sql_alias for output in compiled.analysis.outputs] == expected_columns
        assert all(len(row) == len(expected_columns) for row in rows)
        assert [(row[0].date().isoformat(), row[1], row[2]) for row in rows] == [
            ("2024-01-01", 15.0, expected_values[0]),
            ("2024-01-02", 0.0, expected_values[1]),
            ("2024-01-03", 60.0, expected_values[2]),
        ]
        assert "count" not in result_columns
    finally:
        con.close()


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
    with pytest.raises(CompileError):
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
        dialect=Dialect.BIGQUERY,
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
        dialect=Dialect.SNOWFLAKE,
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
