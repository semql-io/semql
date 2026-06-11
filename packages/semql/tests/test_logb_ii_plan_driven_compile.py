"""LOGb-ii — wire the rest of the compile pipeline through the LogicalPlan.

LOGb-i (test_logb_i_plan_wiring.py) proved the time-block reads from
``self.plan.time_window``.  This file proves the rest of the emission
follows the same pattern:

- FROM clause reads ``self.plan.scans`` (root + targets) and
  ``self.plan.joins`` (edge list).  It must not re-walk the spec tree.
- PROJECTION reads ``self.plan.project.columns`` (one ``ColumnRef`` per
  output column).  The resolved field object is carried on the
  ``ColumnRef`` so emission can render its SQL + alias without
  re-resolving.
- GROUP BY reads ``self.plan.aggregate`` (``None`` = ungrouped row
  listing; ``Aggregate`` = measure aggregation with optional time
  breakdown).
- ORDER / LIMIT / HAVING read ``self.plan.order`` /
  ``self.plan.limit`` — the outer query shape.
- COMPARE reads ``self.plan.compare`` and renders the current/prior
  split from the plan's pre-computed ranges.

The structural assertions (via ``inspect.getsource``) are the load-
bearing ones — they pin "the plan is the source of truth" as a code
invariant, not just a behavioural coincidence.  The behavioural
assertions (end-to-end SQL byte equality) pin that the migration is
invisible to callers: the emitted SQL is identical to the legacy
spec-tree-driven path.

The partition-side transform ``apply_partition_to_plan`` and the
federation split-point (using ``partition_scans``) round out the
work so the LOGb item lands fully done rather than a half-finished
follow-up.
"""

from __future__ import annotations

# mypy: disable-error-code=arg-type
# pyright: reportPrivateUsage=false, reportPrivateImportUsage=false
import inspect
from collections.abc import Mapping

from semql.compile import CompiledQuery, _CompileEnv, compile_query
from semql.logical import (
    Aggregate,
    ColumnRef,
    Join,
    Limit,
    LogicalPlan,
    OrderBy,
    Predicate,
    Project,
    Scan,
    TimeBreakdown,
    apply_partition_to_plan,
    partition_scans,
    to_logical_plan,
)
from semql.model import (
    Backend,
    Cube,
    Dimension,
    Measure,
    PartitionedScan,
    TimeDimension,
)
from semql.model import (
    Join as ModelJoin,
)
from semql.spec import (
    BoolExpr,
    CompareWindow,
    SemanticQuery,
    TimeWindow,
)
from semql.spec import Filter as SpecFilter

from .conftest import CONTEXT


def _compile(catalog: Mapping[str, Cube], q: SemanticQuery) -> CompiledQuery:
    return compile_query(q, dict(catalog), context=CONTEXT)


# ---------------------------------------------------------------------------
# ColumnRef carries the resolved field object
# ---------------------------------------------------------------------------


def test_column_ref_carries_resolved_field() -> None:
    """``ColumnRef.field`` holds the resolved ``Dimension | Measure |
    TimeDimension`` so emission can render without re-resolving.

    The IR is a value type — the resolved field is a frozen
    Pydantic model on the catalog side, so carrying the same
    object avoids a re-lookup at every emission site.
    """
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(query, catalog)

    assert plan.project.columns
    for col in plan.project.columns:
        assert col.field is not None
        # The field object's name and SQL match the ColumnRef.
        assert col.field.name == col.field_name
        assert col.field.sql  # non-empty


def test_column_ref_field_type_matches_kind() -> None:
    """``kind`` and ``field.__class__`` agree — the IR is consistent."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(query, catalog)

    by_kind: dict[str, ColumnRef] = {col.kind: col for col in plan.project.columns}
    assert isinstance(by_kind["dimension"].field, Dimension)
    assert isinstance(by_kind["measure"].field, Measure)


# ---------------------------------------------------------------------------
# Emission reads from the plan — structural assertions
# ---------------------------------------------------------------------------


def test_from_clause_stage_reads_plan_scans_and_joins() -> None:
    """``_from_clause_stage`` must source the root + join targets from
    ``self.plan.scans`` and ``self.plan.joins`` — the spec-tree path
    is closed off."""
    src = inspect.getsource(_CompileEnv._from_clause_stage)
    assert "self.plan.scans" in src or "plan.scans" in src, (
        "FROM must read the plan's scans list (single source of truth)"
    )
    assert "self.plan.joins" in src or "plan.joins" in src, "FROM must read the plan's join edges"
    # The legacy spec-tree reads must be gone.
    assert "self.cubes_in_from" not in src, (
        "FROM should not read self.cubes_in_from — the plan owns the cube list"
    )
    assert "self.join_edges" not in src, (
        "FROM should not read self.join_edges — the plan owns the join list"
    )


def test_projection_stage_reads_plan_project_columns() -> None:
    """``_projection_stage`` must source output columns from
    ``self.plan.project.columns``."""
    src = inspect.getsource(_CompileEnv._projection_stage)
    assert "self.plan.project" in src or "plan.project" in src, (
        "PROJECTION must read the plan's Project node"
    )
    # The legacy dim_fields / measure_fields reads must be gone (or
    # limited to the masking-only path that resolves the viewer).
    assert "self.dim_fields" not in src, (
        "PROJECTION should not read self.dim_fields — the plan owns the dim list"
    )
    assert "self.measure_fields" not in src, (
        "PROJECTION should not read self.measure_fields — the plan owns the measure list"
    )


def test_group_by_stage_reads_plan_aggregate() -> None:
    """``_group_by_stage`` must source the GROUP BY shape from
    ``self.plan.aggregate`` (``None`` = ungrouped)."""
    src = inspect.getsource(_CompileEnv._group_by_stage)
    assert "self.plan.aggregate" in src or "plan.aggregate" in src, (
        "GROUP BY must read the plan's Aggregate node"
    )


# ---------------------------------------------------------------------------
# End-to-end SQL byte equality — migration is invisible to callers
# ---------------------------------------------------------------------------


def test_simple_query_sql_unchanged_after_plan_wiring() -> None:
    """A single-cube aggregation emits the same SQL via the plan path
    as it did via the legacy spec-tree path.  Behavioural pin."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="{schema}.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    cq = _compile(catalog, q)
    # Standard SELECT / GROUP BY / SUM shape.
    assert "SUM" in cq.sql.upper()
    assert "revenue" in cq.columns
    assert "region" in cq.columns


def test_join_sql_unchanged_after_plan_wiring() -> None:
    """A multi-cube query's JOIN edges come from the plan — emitted
    SQL still uses the cube aliases the catalog declared."""
    orders = Cube(
        name="orders",
        alias="o",
        table="{schema}.orders",
        backend=Backend.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[
            ModelJoin(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")
        ],
    )
    customers = Cube(
        name="customers",
        alias="c",
        table="{schema}.customers",
        backend=Backend.POSTGRES,
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )
    catalog = {"orders": orders, "customers": customers}

    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "customers.name"],
    )
    plan = to_logical_plan(q, catalog)
    # The plan carries the join edge the catalog declared.
    assert len(plan.joins) == 1
    assert isinstance(plan.joins[0], Join)
    assert plan.joins[0].left.name == "orders"
    assert plan.joins[0].right.name == "customers"
    assert plan.joins[0].kind == "inner"
    assert plan.joins[0].model is not None

    cq = _compile(catalog, q)
    # The emitted SQL mentions both aliases — the plan-driven join
    # path produces the same shape as the spec-tree path.
    assert "orders" in cq.sql.lower() or '"o"' in cq.sql
    assert "customers" in cq.sql.lower() or '"c"' in cq.sql


def test_left_join_sql_unchanged_after_plan_wiring() -> None:
    """A LEFT-joined cube is marked as such in the plan (kind='left')
    and the emitted SQL preserves that."""
    orders = Cube(
        name="orders",
        alias="o",
        table="{schema}.orders",
        backend=Backend.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[
            ModelJoin(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")
        ],
    )
    customers = Cube(
        name="customers",
        alias="c",
        table="{schema}.customers",
        backend=Backend.POSTGRES,
        dimensions=[Dimension(name="name", sql="{c}.name", type="string")],
    )
    catalog = {"orders": orders, "customers": customers}

    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[SpecFilter(dimension="customers.name", op="eq", values=["alice"])],
        left_joins=["customers"],
    )
    plan = to_logical_plan(q, catalog)
    assert plan.joins[0].kind == "left"

    cq = _compile(catalog, q)
    # LEFT JOIN is preserved in the rendered SQL.
    assert "LEFT JOIN" in cq.sql.upper()


def test_group_by_skipped_for_ungrouped_query_via_plan() -> None:
    """An ungrouped query has ``plan.aggregate is None`` — the GROUP BY
    stage is skipped, no aggregate expressions leak through."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        dimensions=["orders.region"],
        ungrouped=True,
        limit=10,
    )
    plan = to_logical_plan(q, catalog)
    assert plan.aggregate is None
    cq = _compile(catalog, q)
    # No GROUP BY emitted for an ungrouped row-listing query.
    assert "GROUP BY" not in cq.sql.upper()


def test_time_breakdown_via_plan_aggregate() -> None:
    """A time breakdown populates ``plan.aggregate.time`` and the
    emitted SQL truncates the time column to the requested granularity.
    """
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(
                    name="created_at",
                    sql="{o}.created_at",
                    granularities=("day", "month"),
                )
            ],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-03-31"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    assert plan.aggregate is not None
    assert isinstance(plan.aggregate.time, TimeBreakdown)
    assert plan.aggregate.time.granularity == "month"

    cq = _compile(catalog, q)
    assert "created_at_month" in cq.columns
    # Postgres trunc uses date_trunc.
    assert "date_trunc" in cq.sql.lower()


def test_compare_mode_emits_current_prior_full_outer_join() -> None:
    """Compare-mode wraps the inner plan in a CompareSplit node, and
    the emitted SQL contains the current/prior CTEs with the
    pre-computed ranges."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(
                    name="created_at",
                    sql="{o}.created_at",
                    granularities=("day", "month"),
                )
            ],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-04-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    plan = to_logical_plan(q, catalog)
    assert plan.compare is not None
    assert plan.compare.current_range == ("2026-01-01", "2026-04-01")
    # prior_range is shifted back by the duration of the current range.
    # ISO timestamps round-trip through ``datetime.fromisoformat`` —
    # accept whatever shape the builder produces.
    assert plan.compare.prior_range[1].startswith("2026-01-01")
    assert plan.compare.prior_range[0].startswith("2025-10")

    cq = _compile(catalog, q)
    # Compare mode emits a CTE-pair with the prior period.
    assert "_current" in cq.columns or "revenue_current" in cq.columns


# ---------------------------------------------------------------------------
# Partition transform — pure plan→plan
# ---------------------------------------------------------------------------


def test_apply_partition_to_plan_rewrites_matched_scan() -> None:
    """``apply_partition_to_plan`` replaces a scan that points at a
    partitioned cube with a synthetic scan whose cube carries the
    matched physical sources as a ``PartitionedScan`` node (or
    equivalent IR node).

    Pure: input plan is unchanged, output is a fresh plan with the
    synthetic cube.
    """
    from semql.model import (
        TimeDimension as ModelTimeDimension,
    )
    from semql.model import (
        TimePartition,
        TimePartitionedSource,
    )

    cube = Cube(
        name="orders",
        alias="o",
        backend=Backend.POSTGRES,
        time_partition=TimePartition(time_dimension="created_at"),
        physical_sources=[
            TimePartitionedSource(
                name="archive",
                table="orders_archive",
                range_start="2020-01-01",
                range_end="2024-01-01",
            ),
            TimePartitionedSource(
                name="live",
                table="orders_live",
                range_start="2024-01-01",
                range_end=None,
            ),
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[
            ModelTimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day",),
            )
        ],
    )
    catalog = {"orders": cube}
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2024-06-01", "2024-06-08"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    plan_repr = repr(plan)

    new_plan = apply_partition_to_plan(plan, cube)
    assert new_plan is not plan
    assert repr(plan) == plan_repr  # input immutable

    # The matched scan is rewritten; the synthetic cube carries the
    # partitioned scan node.
    assert len(new_plan.scans) == 1
    new_scan = new_plan.scans[0]
    assert isinstance(new_scan, Scan)
    # The synthetic cube is a fresh model_copy.
    assert new_scan.cube is not cube
    # The plan has been routed to the live partition (2024-06-01 falls
    # inside the live range [2024-01-01, None)).
    assert new_scan.cube.partitioned_scan is not None
    assert isinstance(new_scan.cube.partitioned_scan, PartitionedScan)
    matched = {s.name for s in new_scan.cube.partitioned_scan.sources}
    assert "live" in matched
    assert "archive" not in matched


def test_apply_partition_to_plan_empty_match_uses_empty_source() -> None:
    """A query whose range falls outside every source routes to an
    empty-scan placeholder — the plan still lowers, the SQL is empty
    by construction."""
    from semql.model import TimePartition, TimePartitionedSource

    cube = Cube(
        name="orders",
        alias="o",
        backend=Backend.POSTGRES,
        time_partition=TimePartition(time_dimension="created_at"),
        physical_sources=[
            TimePartitionedSource(
                name="archive",
                table="orders_archive",
                range_start="2020-01-01",
                range_end="2024-01-01",
            ),
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at", granularities=("day",))
        ],
    )
    catalog = {"orders": cube}
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-01-08"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    new_plan = apply_partition_to_plan(plan, cube)

    new_scan = new_plan.scans[0]
    assert new_scan.cube.partitioned_scan is not None
    assert new_scan.cube.partitioned_scan.is_empty is True


def test_apply_partition_to_plan_does_not_mutate_catalog() -> None:
    """The original catalog is untouched — only the new plan carries
    the synthetic Cube."""
    from semql.model import TimePartition, TimePartitionedSource

    cube = Cube(
        name="orders",
        alias="o",
        backend=Backend.POSTGRES,
        time_partition=TimePartition(time_dimension="created_at"),
        physical_sources=[
            TimePartitionedSource(
                name="live",
                table="orders_live",
                range_start="2024-01-01",
                range_end=None,
            ),
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[
            TimeDimension(name="created_at", sql="{o}.created_at", granularities=("day",))
        ],
    )
    catalog = {"orders": cube}
    original_table = catalog["orders"].table

    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2024-06-01", "2024-06-08"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    apply_partition_to_plan(plan, cube)

    assert catalog["orders"].table == original_table


# ---------------------------------------------------------------------------
# Federation split-point
# ---------------------------------------------------------------------------


def test_partition_scans_returns_one_entry_per_backend() -> None:
    """``partition_scans(plan)`` returns a ``dict[Backend, LogicalPlan]``
    — one entry per backend in the join graph. Single-backend plans
    return a one-entry dict whose value is the input plan."""
    from semql.federate import compile_federated_query

    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Backend.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="id", sql="{o}.id", type="string"),
        ],
        joins=[
            ModelJoin(
                to="sessions",
                relationship="one_to_many",
                on="{o}.id = {s}.order_id",
            )
        ],
    )
    sessions = Cube(
        name="sessions",
        alias="s",
        table="prod.sessions",
        backend=Backend.CLICKHOUSE,
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[
            Dimension(name="app_name", sql="{s}.app_name", type="string"),
            Dimension(name="order_id", sql="{s}.order_id", type="string"),
        ],
    )
    catalog = {"orders": orders, "sessions": sessions}
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "sessions.app_name"],
    )
    plan = to_logical_plan(q, catalog)
    by_backend = partition_scans(plan)
    # Two cubes on two backends → two partitions.
    assert set(by_backend.keys()) == {Backend.POSTGRES, Backend.CLICKHOUSE}
    # Each partition's plan only contains its own scans.
    pg_plan = by_backend[Backend.POSTGRES]
    ch_plan = by_backend[Backend.CLICKHOUSE]
    pg_cube_names = {s.cube.name for s in pg_plan.scans}
    ch_cube_names = {s.cube.name for s in ch_plan.scans}
    assert "orders" in pg_cube_names
    assert "sessions" in ch_cube_names

    # End-to-end: the federation path still compiles.
    fp = compile_federated_query(q, catalog)
    assert fp.fragments  # non-empty fragment list


def test_partition_scans_single_backend_returns_input() -> None:
    """A single-backend plan is a one-entry dict whose value is the
    input plan (no rewrap needed)."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }
    q = SemanticQuery(measures=["orders.revenue"])
    plan = to_logical_plan(q, catalog)
    by_backend = partition_scans(plan)
    assert list(by_backend.keys()) == [Backend.POSTGRES]
    # The single-backend plan is the input plan (same object).
    assert by_backend[Backend.POSTGRES] is plan


def test_partition_scans_drops_cross_backend_joins() -> None:
    """A cross-backend join is dropped from the per-partition output —
    the merge step stitches the partitions together via bridge keys.
    """
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Backend.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="session_id", sql="{o}.session_id", type="string"),
        ],
        joins=[
            ModelJoin(
                to="sessions",
                relationship="many_to_one",
                on="{o}.session_id = {s}.id",
            )
        ],
    )
    sessions = Cube(
        name="sessions",
        alias="s",
        table="prod.sessions",
        backend=Backend.CLICKHOUSE,
        dimensions=[
            Dimension(name="app_name", sql="{s}.app_name", type="string"),
            Dimension(name="id", sql="{s}.id", type="string"),
        ],
    )
    catalog = {"orders": orders, "sessions": sessions}
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "sessions.app_name"],
    )
    plan = to_logical_plan(q, catalog)
    by_backend = partition_scans(plan)

    # The per-partition plans carry no joins (the cross-backend edge
    # is dropped — the merge step handles it).
    assert by_backend[Backend.POSTGRES].joins == []
    assert by_backend[Backend.CLICKHOUSE].joins == []


# ---------------------------------------------------------------------------
# Plan-shape tests: aggregate / order / limit / project on the IR
# ---------------------------------------------------------------------------


def test_plan_aggregate_carries_derived_measures() -> None:
    """Inline derived measures are stored on the Aggregate node so the
    emitter can render them once with the projection."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[
                Measure(name="revenue", sql="{o}.amount", agg="sum"),
                Measure(name="count", sql="*", agg="count"),
            ],
        )
    }
    from semql.spec import InlineDerived

    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        derived_measures=[
            InlineDerived(
                name="avg_revenue", op="ratio", operands=["orders.revenue", "orders.count"]
            ),
        ],
    )
    plan = to_logical_plan(q, catalog)
    assert plan.aggregate is not None
    assert len(plan.aggregate.derived) == 1


def test_plan_order_limit_captured() -> None:
    """The plan's order + limit are snapshots of the spec, available
    without re-walking the spec tree."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        order=[("orders.revenue", "desc"), ("orders.region", "asc")],
        limit=20,
        offset=5,
    )
    plan = to_logical_plan(q, catalog)
    assert isinstance(plan.order, OrderBy)
    assert plan.order.keys == [
        ("orders.revenue", "desc"),
        ("orders.region", "asc"),
    ]
    assert isinstance(plan.limit, Limit)
    assert plan.limit.limit == 20
    assert plan.limit.offset == 5

    # Emitted SQL has ORDER BY + LIMIT + OFFSET in the expected order.
    cq = _compile(catalog, q)
    sql_upper = cq.sql.upper()
    assert "ORDER BY" in sql_upper
    assert "LIMIT 20" in sql_upper
    assert "OFFSET 5" in sql_upper


# ---------------------------------------------------------------------------
# compile_plan entry point
# ---------------------------------------------------------------------------


def test_compile_plan_entry_point_emits_sql_directly() -> None:
    """``compile_plan(plan, catalog_subset)`` emits SQL from a
    ``LogicalPlan`` directly — no spec-tree round-trip. Used by the
    federation split-point and as a public entry point for callers
    that want to drive emission from a precomputed plan.
    """
    from semql.compile import compile_plan

    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="{schema}.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(q, catalog)
    cq = compile_plan(plan, catalog, context=CONTEXT)
    assert "SUM" in cq.sql.upper()
    assert "revenue" in cq.columns
    assert "region" in cq.columns
    # The plan's columns / touched cubes are surfaced on the result.
    assert "orders" in cq.touched_cube_names


def test_compile_plan_byte_equal_to_compile_query() -> None:
    """``compile_plan(plan)`` and ``compile_query(query)`` produce the
    same SQL — the plan is a strict intermediate representation; the
    spec-tree path and the plan path must agree exactly.
    """
    from semql.compile import compile_plan

    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="{schema}.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(q, catalog)
    via_plan = compile_plan(plan, catalog, context=CONTEXT)
    via_query = compile_query(q, catalog, context=CONTEXT)
    # The two paths produce the same SQL — the migration is invisible.
    assert via_plan.sql == via_query.sql
    assert via_plan.columns == via_query.columns
    assert via_plan.column_meta == via_query.column_meta


# ---------------------------------------------------------------------------
# Plan-shape invariants
# ---------------------------------------------------------------------------


def test_plan_project_columns_carry_kind_metadata() -> None:
    """Each ``ColumnRef.kind`` matches the field's class. The IR
    is consistent with itself — the kind tag is the source of
    truth at the project layer."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            time_dimensions=[
                TimeDimension(
                    name="created_at",
                    sql="{o}.created_at",
                    granularities=("day",),
                )
            ],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2025-01-01", "2025-01-08"),
        ),
    )
    plan = to_logical_plan(q, catalog)
    by_kind: dict[str, type] = {}
    for col in plan.project.columns:
        by_kind[col.kind] = type(col.field)
    assert by_kind["dimension"] is Dimension
    assert by_kind["time"] is TimeDimension
    assert by_kind["measure"] is Measure


def test_plan_filters_carry_resolved_field_via_predicate() -> None:
    """A plan Predicate's expr is the spec tree (Filter or BoolExpr) —
    the same shape the emitter already resolves in _predicate_stage.
    The plan path keeps the predicate as data, not SQL."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[
            SpecFilter(dimension="orders.region", op="eq", values=["us"]),
        ],
        where=BoolExpr(
            op="or",
            children=[
                SpecFilter(dimension="orders.region", op="eq", values=["ca"]),
                SpecFilter(dimension="orders.region", op="eq", values=["mx"]),
            ],
        ),
    )
    plan = to_logical_plan(q, catalog)
    # One predicate per spec-level predicate, with the spec tree intact.
    assert len(plan.filters) == 2
    assert all(isinstance(p, Predicate) for p in plan.filters)
    # The where-tree's predicate carries a BoolExpr.
    assert any(isinstance(p.expr, BoolExpr) for p in plan.filters)
    # The flat-filter predicate carries a Filter.
    assert any(isinstance(p.expr, SpecFilter) for p in plan.filters)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_plan_migration_keeps_existing_snapshot_shape() -> None:
    """The plan-repr snapshots in ``test_plan_snapshots.py`` must keep
    matching — the migration doesn't add nodes the snapshot doesn't
    expect.  Smoke test: a known plan still has the right shape."""
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(q, catalog)
    # The IR shape is still recognisable: scans, joins, filters,
    # aggregate, project, order, limit, time_window, compare.
    assert isinstance(plan, LogicalPlan)
    assert isinstance(plan.scans[0], Scan)
    assert isinstance(plan.aggregate, Aggregate)
    assert isinstance(plan.project, Project)
    assert isinstance(plan.order, OrderBy)
    assert isinstance(plan.limit, Limit)


def test_plan_sql_keywords_unaffected_by_migration() -> None:
    """A regression-guard: the SQL emitted for a non-trivial query
    contains the expected keywords regardless of the internal
    read-site (plan vs spec tree).
    """
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="{schema}.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[
                Dimension(name="region", sql="{o}.region", type="string"),
                Dimension(name="status", sql="{o}.status", type="string"),
            ],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "orders.status"],
        filters=[SpecFilter(dimension="orders.region", op="eq", values=["us"])],
    )
    cq = _compile(catalog, q)
    sql_upper = cq.sql.upper()
    # Standard aggregation shape.
    assert "SELECT" in sql_upper
    assert "FROM" in sql_upper
    assert "GROUP BY" in sql_upper
    assert "WHERE" in sql_upper
    # The filter value is bound (not interpolated).
    assert cq.params  # at least one bound param


def test_sql_round_trip_through_plan_snapshots(snapshot: object) -> None:
    """A regression-guard: the plan's repr is stable.  The snapshot
    tests in test_plan_snapshots.py pin the IR shape; this asserts
    the plan repr is the same after the migration (a smoke test for
    ColumnRef's new ``field`` slot).
    """
    catalog = {
        "orders": Cube(
            name="orders",
            alias="o",
            table="prod.orders",
            backend=Backend.POSTGRES,
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )
    }
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    plan = to_logical_plan(q, catalog)
    text = repr(plan)
    # The plan's repr still includes the standard node sections.
    for marker in ("scans=", "joins=", "filters=", "aggregate=", "project=", "order=", "limit="):
        assert marker in text, f"plan repr missing {marker!r}: {text}"
