"""End-to-end introspection via DuckDB's information_schema.

DuckDB ships an ANSI ``information_schema`` so it doubles as the
reference target for :class:`InformationSchemaProbe`. Each test sets
up a small in-memory schema with a specific shape (FK, numeric
measure-name column, date column, etc.) and verifies the introspector
emits the expected cubes / annotations.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import cast

import duckdb
import pytest
from semql.model import Cube, Dialect
from semql_introspect import (
    InformationSchemaProbe,
    introspect,
    introspect_catalog,
    introspect_to_python,
    introspect_to_result,
)


@pytest.fixture
def conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    c = duckdb.connect(":memory:")
    try:
        yield c
    finally:
        c.close()


def _setup_orders_customers(c: duckdb.DuckDBPyConnection) -> None:
    """Two-table schema with FK + numeric measure + date column."""
    c.execute(
        "CREATE TABLE customers (  id INTEGER PRIMARY KEY,  region VARCHAR,  signup_date DATE)"
    )
    c.execute(
        "CREATE TABLE orders ("
        "  id INTEGER PRIMARY KEY,"
        "  customer_id INTEGER REFERENCES customers(id),"
        "  amount NUMERIC(10,2),"
        "  status VARCHAR,"
        "  created_at TIMESTAMP"
        ")"
    )


# ---------------------------------------------------------------------------
# Cube discovery + classification
# ---------------------------------------------------------------------------


def test_introspect_lists_each_table_as_a_cube(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")
    names = {c.name for c in cubes}
    assert names == {"orders", "customers"}


def test_cube_emits_table_name_alias_and_backend(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    assert cubes["orders"].table == "orders"
    assert cubes["orders"].backend is Dialect.DUCKDB
    # Alias for ``orders`` falls to the single-token shorthand.
    assert cubes["orders"].alias == "o"
    # Multi-token alias path: change the table name and re-run.
    conn.execute("CREATE TABLE user_events (id INTEGER PRIMARY KEY, kind VARCHAR)")
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    assert cubes["user_events"].alias == "ue"


def test_date_column_becomes_time_dimension(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    td_names = {td.name for td in cubes["orders"].time_dimensions}
    assert "created_at" in td_names
    # Same for customers.signup_date.
    cust_td = {td.name for td in cubes["customers"].time_dimensions}
    assert "signup_date" in cust_td


def test_amount_column_becomes_sum_measure(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    measure_names = {m.name for m in cubes["orders"].measures}
    assert "amount" in measure_names
    amount = next(m for m in cubes["orders"].measures if m.name == "amount")
    assert amount.agg == "sum"
    assert amount.sql == "{o}.amount"


def test_id_columns_become_count_distinct_measures(conn: duckdb.DuckDBPyConnection) -> None:
    """A non-PK / non-FK column ending in ``_id`` should surface as a
    ``count_distinct`` measure. The PK and FK ``id`` / ``customer_id``
    columns get filtered out by the FK / PK rules first."""
    conn.execute(
        "CREATE TABLE events ("
        "  event_id BIGINT PRIMARY KEY,"
        "  device_id BIGINT,"  # no FK declared, ends in _id → count_distinct
        "  kind VARCHAR"
        ")"
    )
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    m_names = {m.name for m in cubes["events"].measures}
    assert "distinct_device_id" in m_names
    distinct = next(m for m in cubes["events"].measures if m.name == "distinct_device_id")
    assert distinct.agg == "count_distinct"


def test_pk_column_becomes_dimension_not_measure(conn: duckdb.DuckDBPyConnection) -> None:
    """Even though ``id`` ends in nothing distinctive, the PK rule
    fires before the ``_id`` heuristic — PKs are dimensions and the
    cube tracks identity via ``primary_key``."""
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    orders = cubes["orders"]
    dim_names = {d.name for d in orders.dimensions}
    measure_names = {m.name for m in orders.measures}
    assert "id" in dim_names
    assert "id" not in measure_names
    assert "distinct_id" not in measure_names
    assert orders.primary_key == "id"


# ---------------------------------------------------------------------------
# Foreign keys → Join + Dimension.foreign_key
# ---------------------------------------------------------------------------


def test_fk_emits_join_and_foreign_key_dimension(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    orders = cubes["orders"]
    # FK source column is a Dimension with foreign_key= set.
    customer_id_dim = next((d for d in orders.dimensions if d.name == "customer_id"), None)
    assert customer_id_dim is not None
    assert customer_id_dim.foreign_key == "customers"
    # Auto-derived Join.
    join_targets = [j.to for j in orders.joins]
    assert "customers" in join_targets
    join = next(j for j in orders.joins if j.to == "customers")
    assert join.relationship == "many_to_one"
    assert join.on == "{o}.customer_id = {c}.id"


def test_fk_column_does_not_become_count_distinct_measure(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """``orders.customer_id`` ends in ``_id`` but it's also an FK —
    the FK rule must win so the column isn't double-modelled."""
    _setup_orders_customers(conn)
    cubes = {c.name: c for c in introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")}
    measure_names = {m.name for m in cubes["orders"].measures}
    assert "distinct_customer_id" not in measure_names


# ---------------------------------------------------------------------------
# Filtering — include / exclude
# ---------------------------------------------------------------------------


def test_include_filter_drops_other_tables(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = introspect_catalog(
        conn,
        backend=Dialect.DUCKDB,
        schema="main",
        include_tables=["orders"],
    )
    assert {c.name for c in cubes} == {"orders"}
    # FK target ``customers`` is dropped — the orders.customer_id
    # dimension still carries ``foreign_key="customers"`` but the
    # Join is dropped because there's no target cube to point at.
    orders = cubes[0]
    join_targets = {j.to for j in orders.joins}
    assert "customers" not in join_targets


def test_exclude_filter_drops_named_tables(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    cubes = introspect_catalog(
        conn,
        backend=Dialect.DUCKDB,
        schema="main",
        exclude_tables=["customers"],
    )
    assert {c.name for c in cubes} == {"orders"}


# ---------------------------------------------------------------------------
# Heuristic annotations (# TODO: review hints)
# ---------------------------------------------------------------------------


def test_amount_measure_carries_heuristic_annotation(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _setup_orders_customers(conn)
    result = introspect_to_result(conn, backend=Dialect.DUCKDB, schema="main")
    reasons = {(a.cube, a.field): a.reason for a in result.annotations}
    assert ("orders", "amount") in reasons
    assert "measure-name token" in reasons[("orders", "amount")]


def test_id_count_distinct_carries_heuristic_annotation(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute(
        "CREATE TABLE events (  event_id BIGINT PRIMARY KEY,  device_id BIGINT,  kind VARCHAR)"
    )
    result = introspect_to_result(conn, backend=Dialect.DUCKDB, schema="main")
    reasons = {(a.cube, a.field): a.reason for a in result.annotations}
    assert ("events", "distinct_device_id") in reasons
    assert "_id" in reasons[("events", "distinct_device_id")]


# ---------------------------------------------------------------------------
# Python emission round-trips
# ---------------------------------------------------------------------------


def test_emitted_python_imports_and_compiles(conn: duckdb.DuckDBPyConnection) -> None:
    """The emitted module should be valid Python — import it via exec
    and verify the CUBES list contains the expected cubes."""
    _setup_orders_customers(conn)
    src = introspect_to_python(conn, backend=Dialect.DUCKDB, schema="main")
    ns: dict[str, object] = {}
    exec(compile(src, "<introspected>", "exec"), ns)
    cubes = ns["CUBES"]
    assert isinstance(cubes, list)
    cubes_typed = cast(list[Cube], cubes)
    assert {c.name for c in cubes_typed} == {"orders", "customers"}


def test_emitted_python_carries_todo_comments(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    src = introspect_to_python(conn, backend=Dialect.DUCKDB, schema="main")
    assert "# TODO: review" in src
    assert "measure-name token" in src


def test_emitted_python_wires_foreign_key_and_join(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _setup_orders_customers(conn)
    src = introspect_to_python(conn, backend=Dialect.DUCKDB, schema="main")
    assert "foreign_key='customers'" in src
    assert "Join(to='customers'" in src


def test_emitted_python_accepts_custom_header(conn: duckdb.DuckDBPyConnection) -> None:
    _setup_orders_customers(conn)
    src = introspect_to_python(
        conn,
        backend=Dialect.DUCKDB,
        schema="main",
        header="Generated from staging.duckdb",
    )
    assert '"""Generated from staging.duckdb"""' in src
    # Default header text shouldn't appear.
    assert "Auto-generated by semql-introspect" not in src


# ---------------------------------------------------------------------------
# Alias deduplication for tables sharing initials
# ---------------------------------------------------------------------------


def test_aliases_dedupe_when_initials_collide(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute("CREATE TABLE user_events (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE user_eligibility (id INTEGER PRIMARY KEY)")
    cubes = introspect_catalog(conn, backend=Dialect.DUCKDB, schema="main")
    aliases = {c.alias for c in cubes}
    # Both tables would naturally alias to "ue"; the dedupe step adds a suffix.
    assert len(aliases) == 2


# ---------------------------------------------------------------------------
# Probe-level direct use (covers the custom-dialect extension path)
# ---------------------------------------------------------------------------


def test_probe_can_be_passed_directly_to_introspect(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    _setup_orders_customers(conn)
    probe = InformationSchemaProbe(conn, schema="main")
    result = introspect(probe, backend=Dialect.DUCKDB)
    assert {c.name for c in result.cubes} == {"orders", "customers"}
