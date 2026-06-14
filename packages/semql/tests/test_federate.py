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
    MergeSpec,
    compile_federated_query,
)
from semql.model import (
    Cube,
    Dialect,
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


def _orders(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="orders",
        dialect=dialect,
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


def _customers(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="customers",
        dialect=dialect,
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
# Single-backend degenerate path — wraps CompiledQuery in a one-fragment plan.
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
    assert plan.fragments[0].dialect is Dialect.POSTGRES
    # Degenerate single-backend merge: every measure passes through, no
    # re-aggregation (the engine renders this as a plain projection).
    assert all(m.merge_agg == "passthrough" for m in plan.merge_spec.measures)
    assert plan.merge_spec.cross_partition_clauses == ()
    # Columns + meta match the underlying CompiledQuery.
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
    assert plan.fragments[0].dialect is Dialect.POSTGRES
    assert plan.fragments[1].dialect is Dialect.BIGQUERY

    # Primary fragment exposes the bridge key (customer_id) + revenue.
    assert "customer_id" in plan.fragments[0].columns
    assert "revenue" in plan.fragments[0].columns
    # Dim fragment exposes the bridge key (id) + region.
    assert "id" in plan.fragments[1].columns
    assert "region" in plan.fragments[1].columns


def test_merge_spec_joins_fragments_on_bridge_keys() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        catalog,
    )
    spec = plan.merge_spec
    assert spec.primary_index == 0
    # One bridge join, primary (frag 0) customer_id == satellite (frag 1) id.
    (bridge,) = spec.bridges
    assert (bridge.left.fragment_index, bridge.left.column_name) == (0, "customer_id")
    assert (bridge.right.fragment_index, bridge.right.column_name) == (1, "id")
    # Revenue re-aggregates with SUM, sourced from the primary fragment.
    (measure,) = spec.measures
    assert measure.output_name == "revenue"
    assert measure.merge_agg == "sum"
    assert measure.source is not None
    assert (measure.source.fragment_index, measure.source.column_name) == (0, "revenue")


def test_final_columns_match_user_query_shape() -> None:
    """``FederatedPlan.columns`` is the user-facing output column order
    — same convention as ``CompiledQuery.columns`` (dims, then time, then
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
    # Spec recomposes avg from the decomposed sum/count sources (the
    # engine renders SUM(sum) / NULLIF(SUM(count), 0)).
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "avg_recomposed"
    assert measure.sum_source is not None and measure.count_source is not None
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


def test_filter_only_foreign_cube_engages_federation() -> None:
    """A cross-backend query whose foreign cube is referenced *only* by a
    filter (no grouping dimension) must still engage federation.

    "How many orders did this customer-tier place?" with the measure in
    Postgres (orders) and the filtered dimension in BigQuery (customers),
    and NO ``customers`` dimension in the output. The foreign cube enters
    the touched set via the filter; without that, the query collapses to
    a single (Postgres) backend, falls to single-backend compile, and the
    BigQuery ``customers`` bridge gets pulled onto the join path —
    invalid/cross-dialect SQL. Federation must split it into two
    fragments and route the filter to the customers fragment."""
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(
            measures=["orders.revenue"],
            filters=[Filter(dimension="customers.tier", op="eq", values=["gold"])],
        ),
        catalog,
    )
    assert len(plan.fragments) == 2
    # The filter value lands in the customers (BigQuery) fragment, not the
    # orders (Postgres) measure fragment.
    bq_idx = next(i for i, f in enumerate(plan.fragments) if f.dialect is Dialect.BIGQUERY)
    pg_idx = next(i for i, f in enumerate(plan.fragments) if f.dialect is Dialect.POSTGRES)
    assert "gold" in plan.fragments[bq_idx].params.values()
    assert "gold" not in plan.fragments[pg_idx].params.values()
    # And the merge still bridges the two fragments on the join key.
    assert plan.merge_spec.bridges


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


def test_distributive_mode_lifts_where_tree() -> None:
    """Distributive federation now lifts the where-tree: per-partition
    clauses land in the fragment, the cross-partition residual lives
    in the merge SQL. The raw_rows-only refusal is gone.
    """
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
    plan = compile_federated_query(q, catalog)
    # Both fragments now exist; the where-tree is split.
    assert len(plan.fragments) == 2
    frags_sql = "\n".join(f.sql for f in plan.fragments)
    # orders fragment carries its local piece
    assert "status" in frags_sql or "paid" in frags_sql
    # customers fragment carries its local piece
    assert "tier" in frags_sql or "gold" in frags_sql


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
    project it. Catalog author must opt in by declaring the dim."""
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


def test_plan_carries_structured_merge_spec() -> None:
    catalog = _federated_catalog()
    plan = compile_federated_query(
        SemanticQuery(measures=["orders.revenue"], dimensions=["customers.region"]),
        catalog,
    )
    assert isinstance(plan.merge_spec, MergeSpec)
    assert plan.merge_spec.measures  # at least the revenue measure
    assert not hasattr(plan, "merge")  # no rendered merge SQL on the plan


# ---------------------------------------------------------------------------
# Raw-row mode — lifts non-distributive aggs and having
# ---------------------------------------------------------------------------


def test_raw_rows_lifts_count_distinct_refusal() -> None:
    """``count_distinct`` was refused in distributive mode (sum-of-counts
    isn't a count of distinct values). Raw-row mode emits raw value
    columns from the primary fragment and defers the COUNT(DISTINCT ...)
    to the merge."""
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # Primary fragment is ungrouped — selects raw customer_id values.
    primary = plan.fragments[0]
    assert "customer_id" in primary.sql.lower()
    # Merge re-aggregates with COUNT(DISTINCT ...).
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "count_distinct"
    assert plan.columns == ["region", "distinct_customers"]


def test_raw_rows_supports_min_and_max() -> None:
    orders = _orders().model_copy(
        update={
            "measures": [
                Measure(name="min_amount", sql="{o}.amount", agg="min"),
                Measure(name="max_amount", sql="{o}.amount", agg="max"),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(
        measures=["orders.min_amount", "orders.max_amount"],
        dimensions=["customers.region"],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    aggs = {m.output_name: m.merge_agg for m in plan.merge_spec.measures}
    assert aggs == {"min_amount": "min", "max_amount": "max"}


def test_raw_rows_handles_count_star() -> None:
    """COUNT(*) measures have no raw column — the merge emits
    COUNT(*) directly over the joined rows."""
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.count"],
        dimensions=["customers.region"],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # COUNT(*) has no raw source column — the merge emits COUNT(*).
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "count"
    assert measure.source is not None and measure.source.column_name == ""


def test_raw_rows_lifts_having_refusal() -> None:
    """Distributive mode refuses HAVING; raw-row mode applies it at
    merge against the recomposed measure aliases."""
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        having=[
            Filter(dimension="orders.distinct_customers", op="gte", values=[2]),
        ],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    (having,) = plan.merge_spec.having
    assert having.dimension == "orders.distinct_customers"


def test_raw_rows_having_rejects_unknown_measure() -> None:
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        having=[Filter(dimension="orders.revenue", op="gt", values=[100])],
    )
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog, mode="raw_rows")
    assert exc.value.reason == "having_unknown_measure"


def test_distributive_mode_still_refuses_having() -> None:
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["customers.region"],
        having=[Filter(dimension="orders.revenue", op="gt", values=[100])],
    )
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog)
    assert exc.value.reason == "having_in_distributive_federated"


def test_raw_rows_supports_ratio_measure() -> None:
    """Ratio measures expand recursively in raw-row mode: numerator
    and denominator each get a raw-column projection, and the merge
    composes ``<num_agg>(num_col) / NULLIF(<den_agg>(den_col), 0)``."""
    orders = _orders().model_copy(
        update={
            "measures": [
                Measure(name="revenue", sql="{o}.amount", agg="sum"),
                Measure(name="count", sql="*", agg="count"),
                Measure(
                    name="aov",
                    sql="",
                    agg="ratio",
                    numerator="revenue",
                    denominator="count",
                ),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.aov"], dimensions=["customers.region"])
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # The spec recomposes the ratio from its per-side aggregates (the
    # engine renders SUM(revenue_raw) / NULLIF(COUNT(*), 0)).
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "ratio"
    assert measure.numerator_agg == "sum"
    assert measure.denominator_agg == "count"
    assert "aov" in plan.columns


def test_raw_rows_rejects_nested_ratio_measures() -> None:
    """A ratio whose numerator or denominator is itself a ratio is
    refused — the raw-row expander only walks one level deep."""
    orders = _orders().model_copy(
        update={
            "measures": [
                Measure(name="revenue", sql="{o}.amount", agg="sum"),
                Measure(name="count", sql="*", agg="count"),
                Measure(
                    name="aov",
                    sql="",
                    agg="ratio",
                    numerator="revenue",
                    denominator="count",
                ),
                Measure(
                    name="aov_of_aov",
                    sql="",
                    agg="ratio",
                    numerator="aov",
                    denominator="count",
                ),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.aov_of_aov"], dimensions=["customers.region"])
    with pytest.raises(FederationError) as exc:
        compile_federated_query(q, catalog, mode="raw_rows")
    assert exc.value.reason == "nested_ratio_in_raw_rows"


def test_raw_rows_supports_filtered_measure() -> None:
    """Filtered measures project ``CASE WHEN <filter> THEN <sql> ELSE NULL END``
    as the raw column; the merge agg ignores NULLs, so filter
    semantics compose for SUM / COUNT / AVG / MIN / MAX /
    COUNT(DISTINCT)."""
    orders = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="paid_revenue",
                    sql="{o}.amount",
                    agg="sum",
                    filter="{o}.status = 'paid'",
                )
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.paid_revenue"], dimensions=["customers.region"])
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # Primary fragment carries a CASE-WHEN-wrapped raw column.
    primary = plan.fragments[0]
    assert "CASE WHEN" in primary.sql.upper()
    # Merge SUMs the projected (filtered-by-NULL) raw col.
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "sum"


def test_raw_rows_supports_filtered_count_star() -> None:
    """COUNT(*) FILTER(...) becomes ``CASE WHEN <filter> THEN 1 ELSE NULL END``
    projected at the fragment and ``COUNT(<col>)`` at merge — counting
    non-NULL gives the filtered count."""
    orders = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="paid_count",
                    sql="*",
                    agg="count",
                    filter="{o}.status = 'paid'",
                )
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(measures=["orders.paid_count"], dimensions=["customers.region"])
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    primary = plan.fragments[0]
    assert "CASE WHEN" in primary.sql.upper()
    # Filtered COUNT(*) → COUNT(col) at merge (not COUNT(*)): the measure
    # carries a real raw source column, so it's not the count-star sentinel.
    (measure,) = plan.merge_spec.measures
    assert measure.merge_agg == "count"
    assert measure.source is not None and measure.source.column_name != ""


def test_raw_rows_supports_time_dimension() -> None:
    """Raw-row mode projects the raw timestamp at the fragment and
    buckets via ``date_trunc(grain, col)`` at merge."""
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # The merge buckets the raw timestamp — the spec carries the grain on
    # the time dimension output (the engine renders date_trunc('day', ...)).
    time_out = next(d for d in plan.merge_spec.dimensions if d.output_name == "created_at_day")
    assert time_out.time_grain == "day"
    # Output column name reflects the bucketed grain.
    assert "created_at_day" in plan.columns
    # Fragment received the range as fragment-side filters on the raw
    # timestamp column (so the fragment doesn't pull every row).
    primary = plan.fragments[0]
    assert "2026-01-01" in str(primary.params.values()) or "2026-01-01" in primary.sql


def test_raw_rows_single_partition_where_routes_to_fragment() -> None:
    """A where-tree whose leaves all live on one cube routes into
    that cube's fragment — even when the tree is an OR (which can't
    be expressed via flat ``filters``)."""
    from semql.spec import BoolExpr

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="orders.status", op="eq", values=["pending"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    primary = plan.fragments[0]
    # Both branch values made it into the orders fragment.
    assert "paid" in str(primary.params.values())
    assert "pending" in str(primary.params.values())
    # No cross-partition residual — the clause stayed inside the fragment.
    assert plan.merge_spec.cross_partition_clauses == ()


def test_raw_rows_cross_partition_or_lifted_to_merge() -> None:
    """An OR clause that spans backends can't push down to either
    fragment; the merge SQL emits the disjunction after the JOIN."""
    from semql.spec import BoolExpr

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.tier", op="eq", values=["gold"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # Both partitions had to project the referenced dim — even though
    # neither was in the query's dimension list.
    primary = plan.fragments[0]
    satellite = plan.fragments[1]
    assert "status" in primary.columns
    assert "tier" in satellite.columns
    # The OR rides into the spec as one cross-partition clause with two
    # literals (the engine renders it as a post-join WHERE ... OR ...).
    (clause,) = plan.merge_spec.cross_partition_clauses
    assert len(clause) == 2
    cols = {col for _neg, _idx, col, _op, _vals in clause}
    assert cols == {"status", "tier"}


def test_raw_rows_cross_partition_and_splits_by_cnf() -> None:
    """``A AND B`` where A is on one partition and B on another should
    route A to its partition and B to its partition — neither lives
    in the merge SQL."""
    from semql.spec import BoolExpr

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="and",
            children=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.tier", op="eq", values=["gold"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    primary = plan.fragments[0]
    satellite = plan.fragments[1]
    # Each conjunct lives in its own fragment.
    assert "paid" in str(primary.params.values())
    assert "gold" in str(satellite.params.values())
    # Merge stays free of cross-partition predicates.
    assert plan.merge_spec.cross_partition_clauses == ()


def test_raw_rows_where_tree_with_not_routes_correctly() -> None:
    """``NOT(filter)`` on a leaf is preserved through CNF — the
    fragment-side WHERE keeps the negation."""
    from semql.spec import BoolExpr

    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        where=BoolExpr(
            op="not",
            children=[
                Filter(dimension="orders.status", op="eq", values=["pending"]),
            ],
        ),
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    primary = plan.fragments[0]
    # 'pending' is the filtered-out value; it should still appear as
    # a bound param on the fragment side.
    assert "pending" in str(primary.params.values())


def test_raw_rows_routes_single_partition_segment() -> None:
    """A Segment whose predicate references one cube routes to that
    cube's partition and AND-composes into the fragment's WHERE."""
    from semql.model import Segment

    orders = _orders().model_copy(
        update={
            "segments": [
                Segment(name="paid", sql="{o}.status = 'paid'"),
            ]
        }
    )
    catalog = _catalog(orders, _customers())
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
        segments=["orders.paid"],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    primary = plan.fragments[0]
    assert "status" in primary.sql.lower()
    assert "paid" in str(primary.params.values()) or "'paid'" in primary.sql


def test_raw_rows_single_backend_path_is_unchanged() -> None:
    """Single-backend queries still delegate to compile_query
    regardless of ``mode`` — the mode only kicks in when fragments
    span backends."""
    catalog = _catalog(_orders())
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    distributive = compile_federated_query(q, catalog, mode="distributive")
    raw_rows = compile_federated_query(q, catalog, mode="raw_rows")
    assert distributive.fragments[0].sql == raw_rows.fragments[0].sql


def test_raw_rows_fragment_is_ungrouped_with_large_limit_ok() -> None:
    """Sanity: the raw-row fragment compiles without the 1000-row
    ``ungrouped`` cap tripping. (The flag is internal; verifying via
    a compiled query that would otherwise have raised.)"""
    catalog = _federated_catalog()
    q = SemanticQuery(
        measures=["orders.distinct_customers"],
        dimensions=["customers.region"],
    )
    plan = compile_federated_query(q, catalog, mode="raw_rows")
    # Fragment sql has no LIMIT clause — raw-row mode releases the cap.
    assert "LIMIT" not in plan.fragments[0].sql.upper()


# ---------------------------------------------------------------------------
# Cross-backend symmetric aggregation: two additive-measure facts on
# different backends, conformed to one shared bridge cube.
# ---------------------------------------------------------------------------


def _activity_fact(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="activity",
        dialect=dialect,
        table="activity",
        alias="a",
        primary_key="id",
        measures=[
            Measure(name="active_secs", sql="{a}.secs", agg="sum", unit="duration"),
            Measure(name="avg_secs", sql="{a}.secs", agg="avg", unit="duration"),
        ],
        dimensions=[
            Dimension(name="id", sql="{a}.id", type="number"),
            Dimension(
                name="employee_id", sql="{a}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{a}.employee_id = {e}.id")],
    )


def _worklog_fact(dialect: Dialect = Dialect.BIGQUERY) -> Cube:
    return Cube(
        name="worklog",
        dialect=dialect,
        table="worklog",
        alias="w",
        primary_key="id",
        measures=[Measure(name="hours", sql="{w}.hours", agg="sum", unit="duration")],
        dimensions=[
            Dimension(name="id", sql="{w}.id", type="number"),
            Dimension(
                name="employee_id", sql="{w}.employee_id", type="number", foreign_key="employees"
            ),
        ],
        joins=[Join(to="employees", relationship="many_to_one", on="{w}.employee_id = {e}.id")],
    )


def _employees_bridge(dialect: Dialect = Dialect.POSTGRES) -> Cube:
    return Cube(
        name="employees",
        dialect=dialect,
        table="employees",
        alias="e",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{e}.id", type="number"),
            Dimension(name="name", sql="{e}.name", type="string"),
        ],
    )


def _symmetric_catalog() -> dict[str, Cube]:
    return _catalog(_activity_fact(), _worklog_fact(), _employees_bridge())


def test_cross_backend_symmetric_compiles_to_bridge_plus_facts() -> None:
    """Two additive measures on two backends, conformed to one bridge,
    grouped by a bridge dimension — emits a bridge-primary plan with each
    fact LEFT-joined on the conformed key (no measures_span refusal)."""
    plan = compile_federated_query(
        SemanticQuery(
            measures=["activity.active_secs", "worklog.hours"],
            dimensions=["employees.name"],
        ),
        _symmetric_catalog(),
    )
    assert isinstance(plan, FederatedPlan)
    assert len(plan.fragments) == 3  # bridge + 2 facts
    spec = plan.merge_spec
    assert spec.primary_index == 0  # bridge is the FROM target
    # One LEFT join per fact, both anchored on the bridge fragment.
    assert len(spec.bridges) == 2
    assert all(b.left.fragment_index == 0 and b.join_kind == "left" for b in spec.bridges)
    assert {b.right.fragment_index for b in spec.bridges} == {1, 2}
    # Both measures re-aggregate with SUM (sum-of-sums / sum-of-counts).
    assert [m.merge_agg for m in spec.measures] == ["sum", "sum"]
    assert [m.output_name for m in spec.measures] == ["active_secs", "hours"]
    # Dimension comes from the bridge fragment only.
    assert [d.output_name for d in spec.dimensions] == ["name"]
    assert spec.dimensions[0].sources == [plan.merge_spec.dimensions[0].sources[0]]
    assert plan.columns == ["name", "active_secs", "hours"]


def test_cross_backend_symmetric_requires_a_bridge_dimension() -> None:
    """No grouping dimension: v1 refuses (the bridge isn't referenced, and
    pure cross-backend totals are a separate, simpler plan)."""
    with pytest.raises(FederationError) as exc:
        compile_federated_query(
            SemanticQuery(measures=["activity.active_secs", "worklog.hours"]),
            _symmetric_catalog(),
        )
    assert exc.value.reason == "measures_span_backends"


def test_cross_backend_symmetric_avg_falls_back_to_refusal() -> None:
    """Non-additive measure (avg) is out of the conservative shape — the
    generic measures_span refusal still fires."""
    with pytest.raises(FederationError) as exc:
        compile_federated_query(
            SemanticQuery(
                measures=["activity.avg_secs", "worklog.hours"],
                dimensions=["employees.name"],
            ),
            _symmetric_catalog(),
        )
    assert exc.value.reason == "measures_span_backends"


def test_cross_backend_symmetric_non_bridge_dimension_refuses() -> None:
    """A dimension on a fact (not the shared bridge) changes the grain the
    emit path doesn't model — refuse rather than risk a wrong shape."""
    with pytest.raises(FederationError) as exc:
        compile_federated_query(
            SemanticQuery(
                measures=["activity.active_secs", "worklog.hours"],
                dimensions=["activity.employee_id"],
            ),
            _symmetric_catalog(),
        )
    assert exc.value.reason == "measures_span_backends"
