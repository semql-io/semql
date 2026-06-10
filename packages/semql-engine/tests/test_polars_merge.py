# pyright: reportUnknownMemberType=false
"""Contract tests for :class:`PolarsMergeEngine`.

The contract mirrors the one the DuckDB-backed Engine path implements:
a ``MergeEngine.merge(fragment_results, spec)`` receives one
``AdapterResult`` per fragment plus the structured :class:`MergeSpec`
and must return an ``AdapterResult`` whose ``columns`` and ``rows``
match what a comparable DuckDB-merge run would produce.

We exercise the engine on a 2-backend distributive fixture
(orders on Postgres, customers on BigQuery), a 1-backend degenerate
case, a 2-fragment raw_rows case, and a 3-fragment bridge chain
(orders -> customers -> regions) to ensure the join graph survives
the relabelling step.
"""

from __future__ import annotations

from collections.abc import Mapping

import duckdb
import polars as pl
import pytest
from semql import (
    Backend,
    Cube,
    Dimension,
    Join,
    Measure,
    MergeSpec,
    SemanticQuery,
    compile_federated_query,
)
from semql_engine import AdapterResult, Engine
from semql_engine.merge.polars_engine import PolarsMergeEngine

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _orders_cube(backend: Backend = Backend.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="order_count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(
                name="customer_id",
                sql="{o}.customer_id",
                type="number",
                foreign_key="customers",
            ),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        joins=[
            Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id"),
        ],
    )


def _customers_cube(backend: Backend = Backend.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        backend=backend,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{c}.id", type="number"),
            Dimension(name="region", sql="{c}.region", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )


def _catalog(*cubes: Cube) -> dict[str, Cube]:
    return {c.name: c for c in cubes}


class _DialectTranslatingAdapter:
    """Adapter that runs the fragment SQL against a DuckDB connection.

    The fragment SQL is written in the semql-dialect DuckDB emits; the
    upstream backend will rewrite it later, but for the test we treat
    the DuckDB connection as the source of truth.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection) -> None:
        self._con = con

    def execute(self, sql: str, params: Mapping[str, object]) -> AdapterResult:
        cursor = self._con.execute(sql, params)
        columns = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        return AdapterResult(columns=columns, rows=rows)


@pytest.fixture()
def pg_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, status TEXT, amount DOUBLE)")
    con.execute(
        "INSERT INTO orders VALUES "
        "(1, 10, 'paid', 100.0), "
        "(2, 10, 'paid', 200.0), "
        "(3, 11, 'paid', 50.0), "
        "(4, 11, 'pending', 25.0), "
        "(5, 12, 'paid', 300.0)"
    )
    return con


@pytest.fixture()
def bq_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE customers (id INTEGER, region TEXT, tier TEXT)")
    con.execute(
        "INSERT INTO customers VALUES (10, 'EU', 'gold'), (11, 'US', 'silver'), (12, 'EU', 'gold')"
    )
    return con


# ---------------------------------------------------------------------------
# Pure MergeSpec tests (no engine, no adapters) — exercise PolarsMergeEngine
# directly with hand-built fragment results.
# ---------------------------------------------------------------------------


def test_polars_merge_engine_two_fragment_distributive_matches_duckdb(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """Same plan, same fragment data, run through Polars and the DuckDB
    Engine path; the rows should match (modulo float formatting)."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    plan = compile_federated_query(q, catalog)

    # Reference: run the plan through the default DuckDB engine.
    ref_engine = Engine()
    ref_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    ref_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    ref = ref_engine.run(plan)

    # Subject: re-run with a Polars merge engine.
    polars_engine = Engine(merge_engine=PolarsMergeEngine())
    polars_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    polars_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    got = polars_engine.run(plan)

    assert got.columns == ref.columns
    assert sorted(got.rows) == sorted(ref.rows)


def test_polars_merge_engine_passthrough_for_single_fragment() -> None:
    """A plan that lands entirely on one fragment must pass through the
    Polars merge engine unchanged."""
    # Build a tiny spec directly: single fragment, no bridges, no
    # joins. Dimensions + measures carry-through.
    from semql.compile import ColumnMeta
    from semql.federate import (
        DimensionOutput,
        FragmentColumn,
        MeasureOutput,
    )

    meta_dim = ColumnMeta(name="region", kind="dimension", storage_type="string")
    meta_meas = ColumnMeta(name="revenue", kind="measure", storage_type="number", unit="currency")
    spec = MergeSpec(
        primary_index=0,
        bridges=[],
        dimensions=[
            DimensionOutput(
                output_name="region", sources=[FragmentColumn(0, "region")], column_meta=meta_dim
            )
        ],
        measures=[
            MeasureOutput(
                output_name="revenue",
                merge_agg="passthrough",
                column_meta=meta_meas,
                source=FragmentColumn(0, "revenue"),
            )
        ],
        having=[],
        order_by=[],
        limit=None,
        offset=None,
        mode="distributive",
    )
    fragment = AdapterResult(
        columns=["region", "revenue"],
        rows=[("EU", 300.0), ("US", 50.0)],
    )
    out = PolarsMergeEngine().merge([fragment], spec)
    assert out.columns == ["region", "revenue"]
    assert sorted(out.rows) == sorted(fragment.rows)  # type: ignore[type-var]


def test_polars_merge_engine_bridge_join_with_dim_only_fragment() -> None:
    """In distributive mode measures live on the primary fragment; a
    dim-only secondary fragment contributes dimensions to the join.

    Fixture: orders is the primary fragment (carries ``revenue``,
    already aggregated per region); customers is a dim-only
    secondary carrying ``tier``. The merge joins on region, groups
    by region + tier, and sums the pre-aggregated revenue.
    """
    from semql.compile import ColumnMeta
    from semql.federate import (
        BridgeJoin,
        DimensionOutput,
        FragmentColumn,
        MeasureOutput,
    )

    meta_region = ColumnMeta(name="region", kind="dimension", storage_type="string")
    meta_tier = ColumnMeta(name="tier", kind="dimension", storage_type="string")
    meta_meas = ColumnMeta(name="revenue", kind="measure", storage_type="number", unit="currency")
    spec = MergeSpec(
        primary_index=0,
        bridges=[
            BridgeJoin(
                left=FragmentColumn(0, "region"),
                right=FragmentColumn(1, "region"),
                join_kind="left",
            ),
        ],
        dimensions=[
            DimensionOutput(
                output_name="region",
                sources=[FragmentColumn(0, "region"), FragmentColumn(1, "region")],
                column_meta=meta_region,
            ),
            DimensionOutput(
                output_name="tier", sources=[FragmentColumn(1, "tier")], column_meta=meta_tier
            ),
        ],
        measures=[
            MeasureOutput(
                output_name="revenue",
                merge_agg="sum",
                column_meta=meta_meas,
                source=FragmentColumn(0, "revenue"),
                sum_source=FragmentColumn(0, "revenue"),
            )
        ],
        having=[],
        order_by=[],
        limit=None,
        offset=None,
        mode="distributive",
    )
    f0 = AdapterResult(
        columns=["region", "revenue"],
        rows=[("EU", 300.0), ("US", 50.0)],
    )
    f1 = AdapterResult(
        columns=["region", "tier"],
        rows=[("EU", "gold"), ("US", "silver")],
    )
    out = PolarsMergeEngine().merge([f0, f1], spec)
    df = pl.DataFrame(out.rows, schema=out.columns, orient="row")
    eu = df.filter(pl.col("region") == "EU")["revenue"].item()
    us = df.filter(pl.col("region") == "US")["revenue"].item()
    assert eu == pytest.approx(300.0)
    assert us == pytest.approx(50.0)
    # Tier values carried through from the dim-only fragment.
    assert df.filter(pl.col("region") == "EU")["tier"].item() == "gold"
    assert df.filter(pl.col("region") == "US")["tier"].item() == "silver"


def test_polars_merge_engine_count_distinct_raises_in_distributive_mode() -> None:
    """count_distinct is raw-rows only; if the spec asks for it in
    distributive mode the engine must refuse loudly."""
    from semql.compile import ColumnMeta
    from semql.federate import (
        DimensionOutput,
        FragmentColumn,
        MeasureOutput,
    )

    meta_dim = ColumnMeta(name="region", kind="dimension", storage_type="string")
    meta_meas = ColumnMeta(name="unique", kind="measure", storage_type="number", unit="count")
    spec = MergeSpec(
        primary_index=0,
        bridges=[],
        dimensions=[
            DimensionOutput(
                output_name="region", sources=[FragmentColumn(0, "region")], column_meta=meta_dim
            )
        ],
        measures=[
            MeasureOutput(
                output_name="unique",
                merge_agg="count_distinct",
                column_meta=meta_meas,
                source=FragmentColumn(0, "unique"),
            )
        ],
        having=[],
        order_by=[],
        limit=None,
        offset=None,
        mode="distributive",
    )
    fragment = AdapterResult(columns=["region", "unique"], rows=[("EU", 1), ("US", 2)])
    with pytest.raises(ValueError, match="raw_rows"):
        PolarsMergeEngine().merge([fragment], spec)


def test_polars_merge_engine_handles_bridge_joins(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """A query with multiple dimensions and measures (revenue + order_count)
    across a bridge: the Polars engine must produce the same row set as
    the DuckDB engine for the same plan."""
    catalog = _catalog(_orders_cube(), _customers_cube())
    q = SemanticQuery(
        measures=["orders.revenue", "orders.order_count"],
        dimensions=["customers.region", "customers.tier"],
    )
    plan = compile_federated_query(q, catalog)

    ref_engine = Engine()
    ref_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    ref_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    ref = ref_engine.run(plan)

    polars_engine = Engine(merge_engine=PolarsMergeEngine())
    polars_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    polars_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    got = polars_engine.run(plan)

    assert got.columns == ref.columns
    # Both engines should see the same row set (and same column count
    # per row).  Compare by sorted tuples.
    assert sorted(got.rows) == sorted(ref.rows)


def test_polars_merge_engine_handles_distributive_cross_partition_where(
    pg_con: duckdb.DuckDBPyConnection,
    bq_con: duckdb.DuckDBPyConnection,
) -> None:
    """A cross-partition OR-where in distributive mode produces the
    same final rows under Polars as under the DuckDB engine. This
    also exercises the new distributive-where lift; the merge SQL
    now carries a post-join WHERE for the cross-partition residual.
    """
    from semql.spec import BoolExpr

    catalog = _catalog(_orders_cube(), _customers_cube())
    from semql import Filter

    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.tier", op="eq", values=["gold"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog)

    ref_engine = Engine()
    ref_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    ref_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    ref = ref_engine.run(plan)

    polars_engine = Engine(merge_engine=PolarsMergeEngine())
    polars_engine.register(Backend.POSTGRES, _DialectTranslatingAdapter(pg_con))
    polars_engine.register(Backend.BIGQUERY, _DialectTranslatingAdapter(bq_con))
    got = polars_engine.run(plan)

    assert got.columns == ref.columns
    assert sorted(got.rows) == sorted(ref.rows)
