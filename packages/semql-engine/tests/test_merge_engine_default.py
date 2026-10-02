"""F-M3: ``DuckDBMergeEngine`` is a real MergeEngine that renders the
spec and executes it, and the engine's built-in inline merge now warns
(deprecated) while producing identical results.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import duckdb
import pytest
from semql import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    SemanticQuery,
    compile_federated_query,
)
from semql.federate import FederatedPlan
from semql_engine import (
    AdapterResult,
    DBAPIAdapter,
    DuckDBAdapter,
    DuckDBMergeEngine,
    Engine,
    ExecutionContractError,
    MergeEngine,
)


def _orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id", sql="{o}.customer_id", type="number", foreign_key="customers"
            ),
        ],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers() -> Cube:
    return Cube(
        name="customers",
        dialect=Dialect.BIGQUERY,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
        ],
    )


class _Adapter:
    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def execute(self, sql: str, params: Mapping[str, object]) -> AdapterResult:
        cur = self._con.execute(sql, dict(params))
        return AdapterResult(columns=[d[0] for d in cur.description], rows=cur.fetchall())


def _adapters() -> dict[Dialect, _Adapter]:
    pg = duckdb.connect(":memory:")
    pg.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DOUBLE)")
    pg.execute("INSERT INTO orders VALUES (1, 10, 100.0), (2, 11, 50.0), (3, 10, 25.0)")
    bq = duckdb.connect(":memory:")
    bq.execute("CREATE TABLE customers (id INTEGER, region TEXT)")
    bq.execute("INSERT INTO customers VALUES (10, 'EU'), (11, 'US')")
    return {Dialect.POSTGRES: _Adapter(pg), Dialect.BIGQUERY: _Adapter(bq)}


def _plan() -> FederatedPlan:
    catalog = {c.name: c for c in (_orders(), _customers())}
    return compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"]),
        catalog,
    )


def _engine(merge_engine: MergeEngine | None) -> Engine:
    eng = Engine(merge_engine=merge_engine)
    for dialect, adapter in _adapters().items():
        eng.register(dialect, adapter)
    return eng


def _typed_adapters(
    amount_type: str,
    rows: list[tuple[int, int, Any]],
) -> dict[Dialect, DuckDBAdapter]:
    orders = duckdb.connect(":memory:")
    orders.execute(f"CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount {amount_type})")
    if rows:
        orders.executemany("INSERT INTO orders VALUES (?, ?, ?)", rows)
    customers = duckdb.connect(":memory:")
    customers.execute("CREATE TABLE customers (id INTEGER, region TEXT)")
    customers.execute("INSERT INTO customers VALUES (10, 'EU'), (11, 'US')")
    return {
        Dialect.POSTGRES: DuckDBAdapter(orders),
        Dialect.BIGQUERY: DuckDBAdapter(customers),
    }


def _typed_engine(
    merge_engine: MergeEngine | None,
    amount_type: str,
    rows: list[tuple[int, int, Any]],
) -> Engine:
    engine = Engine(merge_engine=merge_engine)
    for dialect, adapter in _typed_adapters(amount_type, rows).items():
        engine.register(dialect, adapter)
    return engine


def test_duckdb_merge_engine_renders_and_executes() -> None:
    """A DuckDBMergeEngine merge produces the joined + re-aggregated rows
    and emits no deprecation warning (it's the supported path)."""
    plan = _plan()
    engine = _engine(DuckDBMergeEngine())
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = engine.run(plan)
    assert result.columns == ["region", "revenue"]
    assert {r[0]: r[1] for r in result.rows} == {"EU": 125.0, "US": 50.0}


def test_inline_merge_matches_duckdb_merge_engine() -> None:
    """The deprecated inline path yields identical results."""
    inline = _engine(None).run(_plan())
    structured = _engine(DuckDBMergeEngine()).run(_plan())
    assert {r[0]: r[1] for r in inline.rows} == {r[0]: r[1] for r in structured.rows}


def test_inline_merge_warns_deprecated() -> None:
    engine = _engine(None)
    with pytest.warns(DeprecationWarning, match="inline DuckDB merge is deprecated"):
        engine.run(_plan())


def test_inline_merge_warns_only_once_per_engine() -> None:
    engine = _engine(None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        engine.run(_plan())
        engine.run(_plan())
    inline_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(inline_warnings) == 1


@pytest.mark.parametrize("merge_engine", [None, DuckDBMergeEngine()], ids=["inline", "explicit"])
@pytest.mark.parametrize(
    ("amount_type", "source_rows", "expected"),
    [
        ("DECIMAL(12, 2)", [(1, 10, Decimal("12.34"))], {"EU": Decimal("12.34")}),
        ("DECIMAL(12, 2)", [(1, 10, None)], {"EU": None}),
        ("DECIMAL(12, 2)", [], {}),
        ("INTEGER", [(1, 10, 12)], {"EU": 12}),
        ("DOUBLE", [(1, 10, 12.5)], {"EU": 12.5}),
    ],
    ids=["decimal", "all-null-decimal", "empty-decimal", "integer", "float"],
)
def test_fragment_physical_schema_survives_materialized_merge(
    merge_engine: MergeEngine | None,
    amount_type: str,
    source_rows: list[tuple[int, int, Any]],
    expected: dict[str, Any],
) -> None:
    result = _typed_engine(merge_engine, amount_type, source_rows).run(_plan())

    actual = {region: revenue for region, revenue in result.rows}
    assert actual == expected
    if amount_type.startswith("DECIMAL") and source_rows and source_rows[0][2] is not None:
        assert isinstance(next(iter(actual.values())), Decimal)


def test_duckdb_adapter_exposes_decimal_precision_and_scale() -> None:
    con = duckdb.connect(":memory:")
    result = DuckDBAdapter(con).execute(
        "SELECT CAST(12.34 AS DECIMAL(12, 2)) AS amount, CAST(7 AS INTEGER) AS quantity",
        {},
    )

    assert result.column_types == ["DECIMAL(12,2)", "INTEGER"]


@pytest.mark.parametrize(
    "source_rows",
    [[(1, 10, None)], []],
    ids=["all-null", "empty"],
)
def test_legacy_adapter_without_schema_rejects_ambiguous_merge_columns(
    source_rows: list[tuple[int, int, Any]],
) -> None:
    class LegacyAdapter:
        def __init__(self, inner: DuckDBAdapter) -> None:
            self._inner = inner

        def execute(self, sql: str, params: Mapping[str, Any]) -> AdapterResult:
            result = self._inner.execute(sql, params)
            return AdapterResult(columns=result.columns, rows=result.rows)

    engine = Engine()
    for dialect, adapter in _typed_adapters("DECIMAL(12, 2)", source_rows).items():
        engine.register(dialect, LegacyAdapter(adapter))

    with pytest.raises(ExecutionContractError) as raised:
        engine.run(_plan())

    assert raised.value.reason == "physical_schema_missing"


@pytest.mark.parametrize(
    ("amount_type", "source_value", "type_code", "expected"),
    [
        ("DECIMAL(30, 10)", Decimal("0.1234567891"), "NUMERIC", Decimal("0.1234567891")),
        ("DOUBLE", 0.123456789101112, float, 0.123456789101112),
    ],
    ids=["unbounded-numeric", "python-float-double"],
)
def test_dbapi_physical_metadata_preserves_lossless_numeric_merge_values(
    amount_type: str,
    source_value: Any,
    type_code: Any,
    expected: Any,
) -> None:
    class NumericType:
        name = "NUMERIC"

    class MetadataCursor:
        def __init__(
            self,
            connection: duckdb.DuckDBPyConnection,
            revenue_type: Any,
        ) -> None:
            self._cursor = connection.cursor()
            self._revenue_type = NumericType if revenue_type == "NUMERIC" else revenue_type

        def execute(self, sql: str, params: Mapping[str, object] | None = None) -> None:
            if params:
                self._cursor.execute(sql, dict(params))
            else:
                self._cursor.execute(sql)

        @property
        def description(self) -> list[tuple[Any, ...]]:
            description = self._cursor.description or []
            return [
                (
                    row[0],
                    self._revenue_type if row[0] == "revenue" else row[1],
                    *row[2:4],
                    None,
                    None,
                    *row[6:],
                )
                for row in description
            ]

        def fetchall(self) -> list[tuple[Any, ...]]:
            return self._cursor.fetchall()

        def close(self) -> None:
            self._cursor.close()

    class MetadataConnection:
        def __init__(
            self,
            connection: duckdb.DuckDBPyConnection,
            revenue_type: Any,
        ) -> None:
            self._connection = connection
            self._revenue_type = revenue_type

        def cursor(self) -> MetadataCursor:
            return MetadataCursor(self._connection, self._revenue_type)

    orders = duckdb.connect(":memory:")
    orders.execute(f"CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount {amount_type})")
    orders.execute("INSERT INTO orders VALUES (1, 10, ?)", [source_value])
    customers = duckdb.connect(":memory:")
    customers.execute("CREATE TABLE customers (id INTEGER, region TEXT)")
    customers.execute("INSERT INTO customers VALUES (10, 'EU')")

    engine = Engine(merge_engine=DuckDBMergeEngine())
    engine.register(
        Dialect.POSTGRES,
        DBAPIAdapter(MetadataConnection(orders, type_code)),
    )
    engine.register(Dialect.BIGQUERY, DuckDBAdapter(customers))

    result = engine.run(_plan())

    assert result.rows == [("EU", expected)]


def _legacy_numeric_engine(
    merge_engine: MergeEngine | None,
    revenue_by_customer: Mapping[int, Any],
) -> Engine:
    plan = _plan()
    adapters = _typed_adapters(
        "INTEGER",
        [(1, 10, 1), (2, 11, 1)],
    )
    typed_orders = adapters[Dialect.POSTGRES]
    assert plan.fragments[0].dialect is Dialect.POSTGRES
    revenue_index = plan.fragments[0].columns.index("revenue")
    customer_index = plan.fragments[0].columns.index("customer_id")

    class LegacyOrdersAdapter:
        def execute(self, sql: str, params: Mapping[str, object]) -> AdapterResult:
            result = typed_orders.execute(sql, params)
            rows: list[tuple[Any, ...]] = []
            for source_row in result.rows:
                row = list(source_row)
                row[revenue_index] = revenue_by_customer[row[customer_index]]
                rows.append(tuple(row))
            return AdapterResult(columns=result.columns, rows=rows)

    engine = Engine(merge_engine=merge_engine)
    engine.register(Dialect.POSTGRES, LegacyOrdersAdapter())
    engine.register(Dialect.BIGQUERY, adapters[Dialect.BIGQUERY])
    return engine


@pytest.mark.parametrize("merge_engine", [None, DuckDBMergeEngine()], ids=["inline", "explicit"])
@pytest.mark.parametrize(
    ("revenue_by_customer", "expected"),
    [
        ({10: 1, 11: 0.5}, {"EU": 1.0, "US": 0.5}),
        (
            {10: Decimal("1.2"), 11: Decimal("1234.567")},
            {"EU": Decimal("1.2"), "US": Decimal("1234.567")},
        ),
    ],
    ids=["integer-then-fractional-float", "decimal-precision-and-scale"],
)
def test_legacy_numeric_schema_uses_all_observed_values_in_real_merge(
    merge_engine: MergeEngine | None,
    revenue_by_customer: Mapping[int, Any],
    expected: Mapping[str, Any],
) -> None:
    result = _legacy_numeric_engine(merge_engine, revenue_by_customer).run(_plan())

    assert {region: revenue for region, revenue in result.rows} == expected


@pytest.mark.parametrize(
    "revenue_by_customer",
    [
        {
            10: Decimal("12345678901234567890123456789012345678.9"),
            11: Decimal("2.0"),
        },
        {
            10: Decimal("123456789012345678901234567890123456789"),
            11: Decimal("2"),
        },
    ],
    ids=["fractional-overflow", "integer-only-overflow"],
)
def test_legacy_decimal_beyond_duckdb_precision_rejects_lossy_materialization(
    revenue_by_customer: Mapping[int, Any],
) -> None:
    engine = _legacy_numeric_engine(DuckDBMergeEngine(), revenue_by_customer)

    with pytest.raises(ExecutionContractError) as raised:
        engine.run(_plan())

    assert raised.value.reason == "physical_schema_unsupported"


@pytest.mark.parametrize(
    "outside",
    [2**127, -(2**127) - 1],
    ids=["positive-overflow", "negative-overflow"],
)
def test_legacy_integer_outside_duckdb_hugeint_rejects_lossy_materialization(
    outside: int,
) -> None:
    engine = _legacy_numeric_engine(
        DuckDBMergeEngine(),
        {10: outside, 11: 1},
    )

    with pytest.raises(ExecutionContractError) as raised:
        engine.run(_plan())

    assert raised.value.reason == "physical_schema_unsupported"
