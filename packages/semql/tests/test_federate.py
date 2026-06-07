"""Tests for ``semql.federate.compile_federated_query``.

We verify both refusals (every v1 restriction has a structured
``FederationError``) and the happy path: per-backend fragment SQL plus
a DuckDB merge SQL that the in-process executor (or any sans-io
caller) can run against materialised fragment results.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.errors import FederationError
from semql.federate import (
    FederatedPlan,
    MergePlan,
    compile_federated_query,
)
from semql.model import (
    Backend,
    Cube,
    Dimension,
    Join,
    Measure,
    TimeDimension,
)
from semql.spec import Filter, SemanticQuery, TimeWindow

# ---------------------------------------------------------------------------
# Fixtures — a two-backend catalog: orders (Postgres fact) + customers
# (BigQuery dim). The "id" / "customer_id" columns are declared as
# Dimensions so federation can project them as join keys.
# ---------------------------------------------------------------------------


def _orders(backend: Backend = Backend.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        backend=backend,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
            Measure(name="avg_amount", sql="{o}.amount", agg="avg", unit="currency"),
            Measure(name="distinct_customers", sql="{o}.customer_id", agg="count_distinct"),
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
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")],
    )


def _customers(backend: Backend = Backend.BIGQUERY) -> Cube:
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


def _federated_catalog() -> dict[str, Cube]:
    return _catalog(_orders(), _customers())


# ---------------------------------------------------------------------------
# Single-backend degenerate path — wraps Compiled in a one-fragment plan.
# ---------------------------------------------------------------------------


def test_single_backend_query_returns_one_fragment_plan() -> None:
    """When all touched cubes share a backend, no real federation is
    needed; we still return a ``FederatedPlan`` for API uniformity, but
    with a single fragment and a trivial pass-through merge."""
    catalog = _catalog(_orders())
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
        catalog,
    )
    assert len(plan.fragments) == 1
    assert plan.fragments[0].backend is Backend.POSTGRES
    assert plan.merge.sql == "SELECT * FROM frag_0"
    # Columns + meta match the underlying Compiled.
    assert plan.columns == plan.fragments[0].columns
    assert [m.name for m in plan.column_meta] == plan.columns


# ---------------------------------------------------------------------------
# Two-backend enrichment — fact (PG) + dim (BQ).
# ---------------------------------------------------------------------------


def test_cross_backend_enrichment_emits_two_fragments() -> None:
    """The classic federated dashboard shape: fact in Postgres, dim
    label in BigQuery. We get one fragment per backend, plus a DuckDB
    merge that joins them on the bridge key."""
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    assert isinstance(plan, FederatedPlan)
    assert len(plan.fragments) == 2

    # Primary partition (Postgres) comes first.
    assert plan.fragments[0].backend is Backend.POSTGRES
    assert plan.fragments[1].backend is Backend.BIGQUERY

    # Primary fragment exposes the bridge key (customer_id) + revenue.
    assert "customer_id" in plan.fragments[0].columns
    assert "revenue" in plan.fragments[0].columns
    # Dim fragment exposes the bridge key (id) + region.
    assert "id" in plan.fragments[1].columns
    assert "region" in plan.fragments[1].columns


def test_merge_sql_joins_fragments_on_bridge_keys() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    merge_sql = plan.merge.sql
    assert "FROM frag_0" in merge_sql
    assert "LEFT JOIN frag_1" in merge_sql
    # Bridge key equality is in the merge.
    assert '"customer_id"' in merge_sql
    assert '"id"' in merge_sql
    # Sum re-aggregation at merge.
    assert 'SUM(f0."revenue")' in merge_sql
    # Group by the dim column.
    assert "GROUP BY 1" in merge_sql


def test_final_columns_match_user_query_shape() -> None:
    """``FederatedPlan.columns`` is the user-facing output column order
    — same convention as ``Compiled.columns`` (dims, then time, then
    measures)."""
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue", "orders.count"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    assert plan.columns == ["region", "revenue", "count"]


# ---------------------------------------------------------------------------
# Avg decomposition — sum / count pair in the fragment, recomposed at merge.
# ---------------------------------------------------------------------------


def test_avg_measure_decomposed_in_primary_fragment() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.avg_amount"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    # Primary fragment exposes both decomposed columns.
    primary_cols = plan.fragments[0].columns
    assert any(c.endswith("__avg_sum") for c in primary_cols)
    assert any(c.endswith("__avg_count") for c in primary_cols)
    # Merge recomposes avg as SUM(sum) / NULLIF(SUM(count), 0).
    merge_sql = plan.merge.sql
    assert "SUM(f0" in merge_sql
    assert "NULLIF" in merge_sql
    # Final output column is the original measure name.
    assert "avg_amount" in plan.columns


# ---------------------------------------------------------------------------
# Filters route to the correct fragment.
# ---------------------------------------------------------------------------


def test_filter_on_fact_routes_to_fact_fragment() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
        catalog,
    )
    # The filter value lands in the fact fragment's params.
    assert "paid" in plan.fragments[0].params.values()
    assert "paid" not in plan.fragments[1].params.values()


def test_filter_on_dim_routes_to_dim_fragment() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
            filters=[Filter(dimension="customers.tier", op="eq", values=["gold"])],
        ),
        catalog,
    )
    assert "gold" in plan.fragments[1].params.values()
    assert "gold" not in plan.fragments[0].params.values()


# ---------------------------------------------------------------------------
# Refusals — every v1 restriction surfaces as FederationError with a
# structured ``reason``.
# ---------------------------------------------------------------------------


def test_refuses_non_distributive_aggregation() -> None:
    catalog = _federated_catalog()
    with pytest.raises(FederationError) as exc:
        compile_federated_query(
            SemanticQuery(
                measures=["orders.distinct_customers"],
                dimensions=["customers.region"],
            ),
            catalog,
        )
    assert exc.value.reason.startswith("non_distributive_aggregation")


def test_refuses_where_tree() -> None:
    """The boolean predicate tree can OR/NOT across backends in ways
    we can't safely partition. Refuse — callers use flat filters."""
    catalog = _federated_catalog()
    from semql.spec import BoolExpr

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
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "where_tree_in_federated"


def test_refuses_compare_mode() -> None:
    from semql.spec import CompareWindow

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "compare_in_federated"


def test_refuses_multi_column_join_key() -> None:
    """Cross-backend Join.on must be a single equality. Compound keys
    are refused in v1."""
    orders = _orders().model_copy(
        update={
            "joins": [
                Join(
                    to="customers",
                    relationship="many_to_one",
                    on="{o}.customer_id = {c}.id AND {o}.tenant = {c}.tenant",
                ),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "bridge_join_not_simple_equality"


def test_refuses_join_key_not_declared_as_dimension() -> None:
    """If the FK column isn't declared as a Dimension, federation can't
    project it. Catalogue author must opt in by declaring the dim."""
    orders = _orders().model_copy(
        update={
            "dimensions": [
                # No customer_id dimension declared.
                Dimension(name="status", sql="{o}.status", type="string"),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "join_key_not_a_dimension"


def test_refuses_measures_spanning_multiple_backends() -> None:
    """v1 requires all measures on one backend. Customers cube doesn't
    have a measure; let's bolt one on for the test."""
    customers_with_measure = _customers().model_copy(
        update={
            "measures": [
                Measure(name="customer_count", sql="*", agg="count", unit="count"),
            ]
        }
    )
    catalog = _catalog(_orders(), customers_with_measure)
    q = SemanticQuery(
        measures=["orders.revenue", "customers.customer_count"],
        dimensions=["customers.region"],
    )
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "measures_span_backends"


def test_refuses_when_no_cross_backend_join_declared() -> None:
    """Two cubes on different backends with no Join between them can't
    be federated."""
    orders_no_join = _orders().model_copy(update={"joins": []})
    catalog = _catalog(orders_no_join, _customers())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"])
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "no_cross_backend_join"


# ---------------------------------------------------------------------------
# Single-backend uses the existing compile path unchanged.
# ---------------------------------------------------------------------------


def test_single_backend_path_matches_compile_query_output() -> None:
    """The degenerate single-backend ``FederatedPlan`` should hold the
    same SQL as a direct ``compile_query`` call would produce — same
    output columns, same params."""
    catalog = _catalog(_orders())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    direct = compile_query(q, catalog)
    plan = compile_federated_query(q, catalog)
    assert plan.fragments[0].sql == direct.sql
    assert plan.fragments[0].params == direct.params


# ---------------------------------------------------------------------------
# Plan IR shape — basic sanity checks.
# ---------------------------------------------------------------------------


def test_merge_plan_is_a_dataclass_with_sql_and_params() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"]),
        catalog,
    )
    assert isinstance(plan.merge, MergePlan)
    assert isinstance(plan.merge.sql, str)
    assert isinstance(plan.merge.params, dict)
