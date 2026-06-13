"""Drift findings under a live DuckDB connection.

Each test sets up an in-memory DuckDB schema that either matches the
catalog (clean run) or diverges in a specific way (per-finding test).
The goal is one specific drift per test so a regression in any
single probe path surfaces cleanly.
"""

from __future__ import annotations

from collections.abc import Generator

import duckdb
import pytest
from semql import Catalog, Cube, Dialect, Dimension, Join, Measure, TimeDimension
from semql_validate_db import DbValidationError, validate_against_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    c = duckdb.connect(":memory:")
    try:
        yield c
    finally:
        c.close()


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at"),
        ],
    )


def _customers_cube() -> Cube:
    return Cube(
        name="customers",
        backend=Dialect.DUCKDB,
        table="customers",
        alias="c",
        primary_key="id",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="email", sql="{c}.email", type="string"),
        ],
    )


def _create_clean_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE orders (amount DOUBLE, region TEXT, "
        "created_at TIMESTAMP, customer_id INTEGER)"
    )
    conn.execute("CREATE TABLE customers (id INTEGER, email TEXT)")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_clean_catalog_returns_empty_findings(conn: duckdb.DuckDBPyConnection) -> None:
    _create_clean_schema(conn)
    catalog = Catalog([_orders_cube(), _customers_cube()])
    assert validate_against_db(catalog, connection=conn) == []


# ---------------------------------------------------------------------------
# Drift cases — one finding per scenario
# ---------------------------------------------------------------------------


def test_missing_table_surfaces_per_cube(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE TABLE customers (id INTEGER, email TEXT)")
    # `orders` is NOT created — every other probe on orders should
    # short-circuit and we get exactly one finding.
    catalog = Catalog([_orders_cube(), _customers_cube()])
    findings = validate_against_db(catalog, connection=conn)
    codes = [f.code for f in findings]
    assert "missing_table" in codes
    missing = [f for f in findings if f.code == "missing_table"]
    assert len(missing) == 1
    assert missing[0].cube == "orders"
    assert missing[0].field is None


def test_missing_column_on_dimension(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE orders (amount DOUBLE, created_at TIMESTAMP)"  # no `region`
    )
    cube = _orders_cube()
    # Drop the FK + customers join so the join probe doesn't add noise.
    cube_no_joins = cube.model_copy(update={"joins": []})
    catalog = Catalog([cube_no_joins])
    findings = validate_against_db(catalog, connection=conn)
    missing_cols = [f for f in findings if f.code == "missing_column"]
    assert any(f.cube == "orders" and f.field == "region" for f in missing_cols), findings


def test_missing_column_on_measure(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        "CREATE TABLE orders (region TEXT, created_at TIMESTAMP)"  # no `amount`
    )
    cube = _orders_cube().model_copy(update={"joins": []})
    catalog = Catalog([cube])
    findings = validate_against_db(catalog, connection=conn)
    revenue_findings = [f for f in findings if f.code == "missing_column" and f.field == "revenue"]
    assert revenue_findings, findings


def test_count_star_skipped_when_table_present(conn: duckdb.DuckDBPyConnection) -> None:
    """The ``count(*)`` measure has nothing column-specific to probe
    — the table probe already covered it, so we don't waste a query."""
    _create_clean_schema(conn)
    cube = _orders_cube().model_copy(update={"joins": []})
    findings = validate_against_db(Catalog([cube]), connection=conn)
    assert findings == []


def test_base_predicate_invalid(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE TABLE orders (amount DOUBLE, region TEXT, created_at TIMESTAMP)")
    cube = _orders_cube().model_copy(
        update={"base_predicate": "{o}.never_existed IS NULL", "joins": []}
    )
    findings = validate_against_db(Catalog([cube]), connection=conn)
    assert any(f.code == "base_predicate_invalid" for f in findings), findings


def test_join_predicate_invalid(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE TABLE orders (amount DOUBLE, region TEXT, created_at TIMESTAMP)")
    conn.execute("CREATE TABLE customers (id INTEGER, email TEXT)")
    orders_with_bad_join = _orders_cube().model_copy(
        update={
            "joins": [
                Join(
                    to="customers",
                    relationship="many_to_one",
                    on="{o}.no_such_column = {c}.id",
                )
            ]
        }
    )
    catalog = Catalog([orders_with_bad_join, _customers_cube()])
    findings = validate_against_db(catalog, connection=conn)
    assert any(
        f.code == "join_predicate_invalid" and f.cube == "orders" and f.field == "customers"
        for f in findings
    ), findings


def test_required_filter_dimension_missing(conn: duckdb.DuckDBPyConnection) -> None:
    """A required_filters entry that doesn't match any dimension is
    a static catalog defect — surface it at pre-deploy time even
    when the table itself is fine."""
    conn.execute("CREATE TABLE orders (amount DOUBLE, region TEXT, created_at TIMESTAMP)")
    cube = _orders_cube().model_copy(update={"required_filters": ["tenant_id"], "joins": []})
    catalog = Catalog([cube])
    findings = validate_against_db(catalog, connection=conn)
    rf = [f for f in findings if f.code == "required_filter_dimension_missing"]
    assert len(rf) == 1
    assert rf[0].cube == "orders"
    assert rf[0].field == "tenant_id"


# ---------------------------------------------------------------------------
# Context substitution + META skip
# ---------------------------------------------------------------------------


def test_context_substitutes_schema_placeholders(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SCHEMA analytics")
    conn.execute("CREATE TABLE analytics.orders (amount DOUBLE, region TEXT, created_at TIMESTAMP)")
    cube = _orders_cube().model_copy(update={"table": "{schema}.orders", "joins": []})
    catalog = Catalog([cube])
    findings = validate_against_db(catalog, connection=conn, context={"schema": "analytics"})
    assert findings == []


def test_meta_cubes_are_skipped(conn: duckdb.DuckDBPyConnection) -> None:
    """META reflection cubes don't live in the physical DB. Catalog
    auto-appends them, so a naïve loop over the catalog would probe
    them — we don't, so an empty DuckDB still validates a META-only
    catalog cleanly."""
    catalog = Catalog([])  # no real cubes — only the auto-appended META set
    findings = validate_against_db(catalog, connection=conn)
    assert findings == []


# ---------------------------------------------------------------------------
# Findings carry the database's own error message
# ---------------------------------------------------------------------------


def test_finding_carries_db_error_detail(conn: duckdb.DuckDBPyConnection) -> None:
    """The driver's error string lands in ``detail`` so a CI log can
    show what the database actually said without us re-parsing."""
    catalog = Catalog([_orders_cube().model_copy(update={"joins": []})])
    findings: list[DbValidationError] = validate_against_db(catalog, connection=conn)
    assert findings, "expected at least one finding when the table is missing"
    detail = findings[0].detail or ""
    assert detail, "expected detail to carry the driver's error message"
