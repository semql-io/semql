"""Tests for pre-aggregation / rollup routing (S3 + DEFERRED §5).

A ``Rollup`` on a cube declares a materialised table holding rows
pre-grouped at a given grain (dims + optional time bucket). The
compiler matches a query against each rollup; on a fit, the SQL is
rewritten to read the rollup's ``physical_table`` instead of the
cube's base ``table`` — without the planner / caller knowing.

Phase 1 matching is conservative: exact-grain (granularity matches),
all referenced dims / measures stored, every filter touches only
stored columns, no joins / segments / compare windows. When multiple
rollups fit, the one with the fewest stored dimensions wins (smallest
table = fastest read). The applied rollup name is surfaced on
``CompiledQuery.applied_rollup`` for observability.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Backend,
    Catalog,
    Cube,
    Dimension,
    Filter,
    Measure,
    Rollup,
    Segment,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from semql.spec import CompareWindow


def _orders_cube(*, rollups: list[Rollup] | None = None) -> Cube:
    return Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
            Measure(name="min_amount", sql="{o}.amount", agg="min"),
            Measure(name="max_amount", sql="{o}.amount", agg="max"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        time_dimensions=[
            TimeDimension(name="placed_at", sql="{o}.placed_at"),
        ],
        rollups=rollups or [],
    )


# ---------------------------------------------------------------------------
# Catalog validation — refusal at registration
# ---------------------------------------------------------------------------


def test_rollup_unknown_dimension_rejected() -> None:
    with pytest.raises(ValidationError, match=r"(?i)dimension 'nope' is not declared"):
        _orders_cube(
            rollups=[
                Rollup(
                    name="by_nope",
                    physical_table="orders_by_nope",
                    dimensions=["nope"],
                    measures=["revenue"],
                )
            ]
        )


def test_rollup_unknown_measure_rejected() -> None:
    with pytest.raises(ValidationError, match=r"(?i)measure 'phantom' is not declared"):
        _orders_cube(
            rollups=[
                Rollup(
                    name="by_phantom",
                    physical_table="orders_phantom",
                    dimensions=["region"],
                    measures=["phantom"],
                )
            ]
        )


def test_rollup_non_distributive_agg_rejected() -> None:
    """``count_distinct`` / ``avg`` / ``median`` / ``ratio`` can't be
    re-aggregated trivially over a rollup grain. Refuse at registration."""
    with pytest.raises(ValidationError, match=r"(?i)agg='count_distinct'"):
        Cube(
            name="orders",
            backend=Backend.POSTGRES,
            table="orders",
            alias="o",
            measures=[
                Measure(name="users", sql="{o}.user_id", agg="count_distinct"),
            ],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            rollups=[
                Rollup(
                    name="dau",
                    physical_table="orders_dau",
                    dimensions=["region"],
                    measures=["users"],
                )
            ],
        )


def test_rollup_unpaired_time_dim_rejected() -> None:
    """``time_dimension`` set without ``granularity`` is meaningless;
    ``granularity`` without a time dim has nothing to address."""
    with pytest.raises(ValidationError, match=r"(?i)must both be set or both be unset"):
        Rollup(
            name="bad",
            physical_table="x",
            time_dimension="placed_at",
        )


def test_rollup_time_dim_not_on_cube_rejected() -> None:
    with pytest.raises(ValidationError, match=r"(?i)time_dimension 'not_a_time'"):
        _orders_cube(
            rollups=[
                Rollup(
                    name="bad",
                    physical_table="orders_bad",
                    time_dimension="not_a_time",
                    granularity="day",
                    measures=["revenue"],
                )
            ]
        )


def test_rollup_granularity_not_allowed_on_time_dim_rejected() -> None:
    with pytest.raises(ValidationError, match=r"(?i)granularity 'hour' not permitted"):
        Cube(
            name="orders",
            backend=Backend.POSTGRES,
            table="orders",
            alias="o",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            time_dimensions=[
                # Only day / week permitted on this TD; the rollup asks for hour.
                TimeDimension(name="placed_at", sql="{o}.placed_at", granularities=("day", "week")),
            ],
            rollups=[
                Rollup(
                    name="hourly",
                    physical_table="orders_hourly",
                    time_dimension="placed_at",
                    granularity="hour",
                    measures=["revenue"],
                )
            ],
        )


def test_rollup_duplicate_names_rejected() -> None:
    with pytest.raises(ValidationError, match=r"(?i)duplicate rollup name"):
        _orders_cube(
            rollups=[
                Rollup(name="r", physical_table="t1", measures=["revenue"]),
                Rollup(name="r", physical_table="t2", measures=["count"]),
            ]
        )


# ---------------------------------------------------------------------------
# Routing — match path
# ---------------------------------------------------------------------------


def _cat_with_daily_rollup() -> Catalog:
    cube = _orders_cube(
        rollups=[
            Rollup(
                name="daily_region",
                physical_table="orders_daily_region",
                alias="o",
                dimensions=["region"],
                time_dimension="placed_at",
                granularity="day",
                measures=["revenue", "count"],
            )
        ]
    )
    return Catalog([cube])


def test_exact_grain_match_routes_to_rollup_table() -> None:
    """Dims + measures + time grain all match → rollup table picked."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup == "daily_region"
    # The rollup's physical_table is the FROM source.
    assert "orders_daily_region" in out.sql
    # Base table is NOT referenced.
    assert " orders " not in out.sql.lower().replace("orders_daily_region", "")
    # The measure SQL points at the stored column ``revenue``, not amount.
    assert "revenue" in out.sql
    assert "amount" not in out.sql


def test_match_aggregates_over_rollup_columns() -> None:
    """SUM(stored_revenue) over the rollup gives the same total as
    SUM(amount) over the base table — that's the whole point. The
    emitted SQL must address the stored column, not the base column."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    upper = out.sql.upper()
    assert "SUM(" in upper
    assert "COUNT(" in upper
    # Stored count column is named ``count`` — the rollup's measures
    # share names with the base catalog measures.
    assert "count" in out.sql.lower()


def test_filter_on_stored_dimension_routes_to_rollup() -> None:
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup == "daily_region"
    assert "WHERE" in out.sql.upper()


# ---------------------------------------------------------------------------
# Routing — miss / fall-back paths
# ---------------------------------------------------------------------------


def test_dimension_not_stored_falls_back_to_base_table() -> None:
    """Query asks for a dim (``status``) not in the rollup grain →
    rollup can't answer → compile uses the base table."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.status"],  # status not in rollup
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None
    assert "orders_daily_region" not in out.sql


def test_measure_not_stored_falls_back_to_base_table() -> None:
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.min_amount"],  # not in rollup
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None


def test_granularity_mismatch_falls_back_to_base_table() -> None:
    """Rollup is daily; query asks for hourly → no match (Phase 1
    doesn't down-aggregate a daily-bucketed column)."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="hour",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None


def test_filter_on_non_stored_dimension_falls_back() -> None:
    """``status`` isn't in the rollup grain — filtering by it on the
    rollup would silently include rows the planner expected to filter
    out. Phase 1 refuses; fall back."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None


def test_segments_in_query_disable_rollup_routing() -> None:
    """Segments are SQL fragments templated against the cube's base
    alias — re-keying them at a rollup grain isn't safe. Refuse."""
    cube = _orders_cube(
        rollups=[
            Rollup(
                name="daily_region",
                physical_table="orders_daily_region",
                dimensions=["region"],
                time_dimension="placed_at",
                granularity="day",
                measures=["revenue"],
            )
        ],
    )
    cube_with_seg = cube.model_copy(
        update={
            "segments": [Segment(name="paid", sql="{o}.status = 'paid'")],
        }
    )
    cat = Catalog([cube_with_seg])
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        segments=["orders.paid"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None


def test_compare_window_disables_rollup_routing() -> None:
    """Compare CTEs need row-level time access to derive the prior
    window. Daily-bucketed time has no sub-day resolution; refuse."""
    cat = _cat_with_daily_rollup()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.placed_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    out = cat.compile(q)
    assert out.applied_rollup is None


def test_no_rollup_means_no_routing() -> None:
    """Cube without rollups: applied_rollup stays None."""
    cat = Catalog([_orders_cube()])
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    out = cat.compile(q)
    assert out.applied_rollup is None
    assert "orders" in out.sql


# ---------------------------------------------------------------------------
# Routing — multi-rollup tiebreak
# ---------------------------------------------------------------------------


def test_smallest_matching_rollup_wins() -> None:
    """Two rollups can both answer the query — pick the one with
    fewer stored dimensions (smaller table)."""
    cube = _orders_cube(
        rollups=[
            # Wider rollup: region + status. Could answer ``region``-only too.
            Rollup(
                name="region_status",
                physical_table="orders_region_status",
                dimensions=["region", "status"],
                measures=["revenue"],
            ),
            # Narrower rollup: region-only.
            Rollup(
                name="region_only",
                physical_table="orders_region_only",
                dimensions=["region"],
                measures=["revenue"],
            ),
        ]
    )
    cat = Catalog([cube])
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    out = cat.compile(q)
    # The smaller rollup (region only) is chosen.
    assert out.applied_rollup == "region_only"
    assert "orders_region_only" in out.sql
    assert "orders_region_status" not in out.sql


def test_time_dim_only_rollup_serves_query_without_time() -> None:
    """A rollup that stores a time bucket can still answer a query that
    omits ``time_dimension`` — SUM-over-all-days is correctly the
    SUM-over-rows-of-stored-daily-sums."""
    cube = _orders_cube(
        rollups=[
            Rollup(
                name="daily_region",
                physical_table="orders_daily_region",
                dimensions=["region"],
                time_dimension="placed_at",
                granularity="day",
                measures=["revenue"],
            )
        ]
    )
    cat = Catalog([cube])
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    out = cat.compile(q)
    assert out.applied_rollup == "daily_region"


# ---------------------------------------------------------------------------
# Routing — multi-cube queries disabled
# ---------------------------------------------------------------------------


def test_multi_cube_query_skips_rollup_routing() -> None:
    """Phase 1 routes only single-cube queries."""
    orders = _orders_cube(
        rollups=[
            Rollup(
                name="r",
                physical_table="orders_rollup",
                dimensions=["region"],
                measures=["revenue"],
            )
        ]
    )
    # Add a second cube + FK so the join graph resolves.
    customers = Cube(
        name="customers",
        backend=Backend.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
        measures=[Measure(name="signups", sql="*", agg="count")],
    )
    orders_with_fk = orders.model_copy(
        update={
            "dimensions": [
                *orders.dimensions,
                Dimension(
                    name="customer_id",
                    sql="{o}.customer_id",
                    type="number",
                    foreign_key="customers",
                ),
            ]
        }
    )
    cat = Catalog([orders_with_fk, customers])
    # Multi-cube via a customers *dimension* (forces the join) rather than
    # customers.signups: COUNT(customers) across the many_to_one join would
    # fan out (count order-rows, not customers) and is now refused. The
    # rollup-routing skip is what this test cares about, not the measure.
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "customers.id"],
    )
    out = cat.compile(q)
    assert out.applied_rollup is None
