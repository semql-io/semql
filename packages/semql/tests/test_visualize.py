# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportPrivateUsage=false
# (Test helpers pass bare ``frozenset`` literals of chart-type strings to
# ``supported_charts``; annotating each as ``frozenset[VizChartType]`` adds
# no test value and the literal-narrowing gymnastics aren't worth it here.)
"""Unit tests for ``semql.visualize``.

The decision table inside ``_pick_chart_type`` is the only place
SemQL says "use a pie chart" / "use a line chart" / "use a data table".
The branches are short but the *boundaries* (``PIE_MAX_SLICES``,
``BAR_MAX_BARS``) and the conflict-resolution behaviour
(multiple cubes with different ``default_chart_type``) are the things
that drift silently when someone re-orders the if-chain.
"""

from __future__ import annotations

import pytest
from semql.model import Cube, Dialect, Dimension, Join, Measure, TimeDimension
from semql.spec import CompareWindow, SemanticQuery, TimeWindow
from semql.visualize import (
    BAR_MAX_BARS,
    LOG_SCALE_RATIO,
    NULL_RATE_CAVEAT_THRESHOLD,
    PIE_MAX_SLICES,
    DecisionReason,
    RenderHints,
    ScoredChart,
    ShapeStats,
    VizColumn,
    VizDecision,
    VizFeatures,
    decide_visualization,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _orders(default_chart_type: object | None = None) -> Cube:
    kwargs: dict[str, object] = {
        "name": "orders",
        "dialect": Dialect.POSTGRES,
        "table": "orders",
        "alias": "o",
        "measures": [
            Measure(
                name="revenue",
                sql="{o}.amount",
                agg="sum",
                unit="currency",
            ),
            Measure(name="orders", sql="*", agg="count", unit="count"),
            Measure(
                name="conversion_rate",
                sql="{o}.x",
                agg="avg",
                unit="pct",
            ),
        ],
        "dimensions": [
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        "time_dimensions": [
            TimeDimension(name="created_at", sql="{o}.created_at"),
        ],
    }
    if default_chart_type is not None:
        kwargs["default_chart_type"] = default_chart_type
    return Cube(**kwargs)  # type: ignore[arg-type]


def _customers(default_chart_type: object | None = None) -> Cube:
    kwargs: dict[str, object] = {
        "name": "customers",
        "dialect": Dialect.POSTGRES,
        "table": "customers",
        "alias": "c",
        "measures": [Measure(name="count", sql="*", agg="count", unit="count")],
        "dimensions": [Dimension(name="region", sql="{c}.region", type="string")],
    }
    if default_chart_type is not None:
        kwargs["default_chart_type"] = default_chart_type
    return Cube(**kwargs)  # type: ignore[arg-type]


def _catalog(*cubes: Cube) -> dict[str, Cube]:
    return {c.name: c for c in cubes}


def _decide(
    query: SemanticQuery,
    n_rows: int,
    *,
    catalog: dict[str, Cube],
    supported_charts: frozenset | None = None,
    shape_stats: ShapeStats | None = None,
) -> VizDecision:
    """Compile the query against the catalog and feed the resulting
    ``CompiledQuery`` bundle to ``decide_visualization``. Tests pass through
    this wrapper so they don't have to construct a ``CompiledQuery`` by hand
    every time — the round-trip via the real compiler is also a useful
    integration check that ``CompiledQuery.column_meta`` carries the right
    information for the visualiser."""
    from semql.compile import compile_query

    out = compile_query(query, catalog)
    return decide_visualization(
        query=query,
        compiled=out,
        n_rows=n_rows,
        catalog=catalog,
        supported_charts=supported_charts,
        shape_stats=shape_stats,
    )


# ---------------------------------------------------------------------------
# Branch: cube default_chart_type override
# ---------------------------------------------------------------------------


def test_single_override_wins_regardless_of_shape() -> None:
    cube = _orders(default_chart_type="data_table")
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=2,  # would otherwise be a pie chart
        catalog=_catalog(cube),
    )
    assert decision.chart_type == "data_table"
    assert "default_chart_type" in decision.reason.note


def test_conflicting_overrides_fall_through_to_normal_logic() -> None:
    """Two cubes touched with *different* default_chart_type values
    cancel each other out so the normal decision logic runs."""
    orders = _orders(default_chart_type="bar_chart")
    customers = _customers(default_chart_type="pie_chart")
    # touch both cubes by including a region dimension and a count
    # measure from each (via a join in the compile path) — for the
    # viz decision we can just pass both into the catalog and reference
    # one cube's fields; the resolver only walks the query, so we
    # need a query that touches both cubes.
    orders_join = orders.model_copy(
        update={
            "joins": [
                Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id"),
            ]
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        n_rows=2,
        catalog=_catalog(orders_join, customers),
    )
    # Two distinct overrides → fall through to the n_rows/n_dims logic;
    # 1 dim + 1 measure + n_rows<=PIE_MAX_SLICES → pie chart.
    assert decision.chart_type == "pie_chart"


def test_matching_overrides_count_as_one() -> None:
    """Both cubes share the *same* override → still wins."""
    orders = _orders(default_chart_type="data_table")
    customers = _customers(default_chart_type="data_table")
    orders_join = orders.model_copy(
        update={
            "joins": [
                Join(to="customers", relationship="many_to_one", on="{o}.cid = {c}.id"),
            ]
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["customers.region"],
        ),
        n_rows=2,
        catalog=_catalog(orders_join, customers),
    )
    assert decision.chart_type == "data_table"


# ---------------------------------------------------------------------------
# Branch: ungrouped → data_table
# ---------------------------------------------------------------------------


def test_ungrouped_always_data_table() -> None:
    decision = _decide(
        SemanticQuery(dimensions=["orders.region"], ungrouped=True, limit=10),
        n_rows=5,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "data_table"
    assert "ungrouped" in decision.reason.note


# ---------------------------------------------------------------------------
# Branch: text_only — single measure, no dimensions
# ---------------------------------------------------------------------------


def test_single_measure_no_dim_returns_text_only() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"]),
        n_rows=1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "text_only"


# ---------------------------------------------------------------------------
# Branch: line_chart — time series with granularity
# ---------------------------------------------------------------------------


def test_time_breakdown_returns_line_chart() -> None:
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"


def test_time_without_granularity_is_not_line_chart() -> None:
    """Time dimension WITHOUT granularity is just a filter — falls
    through to bar / pie / data-table logic."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=2,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type != "line_chart"


# ---------------------------------------------------------------------------
# Branch: pie_chart — 1 dim, 1 measure, n_rows <= PIE_MAX_SLICES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_rows", [1, PIE_MAX_SLICES])
def test_pie_chart_at_and_below_boundary(n_rows: int) -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=n_rows,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "pie_chart"


def test_pie_chart_off_by_one_falls_to_bar() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=PIE_MAX_SLICES + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "bar_chart"


def test_two_measures_one_dim_is_scatter() -> None:
    """2 measures + 1 dim → scatter: the two measures are the X/Y axes and
    the dimension labels the points — never pie, never bar."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.orders"],
            dimensions=["orders.region"],
        ),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "scatter_chart"
    assert decision.x_axis == "Revenue"
    assert decision.y_axes == ["Orders"]


# ---------------------------------------------------------------------------
# Branch: bar_chart — 1 dim, n_rows <= BAR_MAX_BARS
# ---------------------------------------------------------------------------


def test_bar_chart_at_boundary() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=BAR_MAX_BARS,
        catalog=_catalog(_orders()),
    )
    # n_rows == BAR_MAX_BARS and > PIE_MAX_SLICES → bar chart.
    assert decision.chart_type == "bar_chart"


def test_bar_chart_off_by_one_falls_to_data_table() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=BAR_MAX_BARS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "data_table"


# ---------------------------------------------------------------------------
# Branch: stacked_bar_chart — 2 categorical dims + 1 measure, manageable n
# ---------------------------------------------------------------------------


def test_two_dims_one_measure_is_stacked_bar() -> None:
    """2 categorical dims + 1 measure (manageable n) → stacked bar: the
    first dim is the primary axis, the second becomes the stack series."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region", "orders.status"],
        ),
        n_rows=4,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "stacked_bar_chart"
    assert decision.x_axis == "Region"
    assert decision.series == "Status"
    assert decision.y_axes == ["Revenue"]


# ---------------------------------------------------------------------------
# Branch: data_table — multi-dim too large for a chart
# ---------------------------------------------------------------------------


def test_large_multi_dim_returns_data_table() -> None:
    """Two dims past the heatmap cell cap is too dense to chart → data_table."""
    from semql.visualize import HEATMAP_MAX_CELLS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region", "orders.status"],
        ),
        n_rows=HEATMAP_MAX_CELLS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "data_table"


# ---------------------------------------------------------------------------
# Output shape — VizColumn / axis labels / title / format inference
# ---------------------------------------------------------------------------


def test_columns_are_populated_with_per_column_metadata() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert [c.name for c in decision.columns] == ["region", "revenue"]
    revenue_col = next(c for c in decision.columns if c.name == "revenue")
    assert revenue_col.is_measure is True
    # unit="currency" is not in the inference table (only pct/count/duration);
    # default falls through to "raw". Callers wanting "currency" set format=
    # explicitly on the Measure.
    assert revenue_col.format == "raw"
    region_col = next(c for c in decision.columns if c.name == "region")
    assert region_col.is_measure is False
    assert region_col.is_time is False


def test_time_dimension_column_marked_is_time() -> None:
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=10,
        catalog=_catalog(_orders()),
    )
    ts_col = next(c for c in decision.columns if c.name == "created_at_day")
    assert ts_col.is_time is True


# ---------------------------------------------------------------------------
# Format inference per unit
# ---------------------------------------------------------------------------


def test_unit_count_becomes_integer_format() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.orders"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    orders_col = next(c for c in decision.columns if c.name == "orders")
    assert orders_col.format == "integer"


def test_unit_pct_becomes_percent_format() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.conversion_rate"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    col = next(c for c in decision.columns if c.name == "conversion_rate")
    assert col.format == "percent"


def test_explicit_format_overrides_unit_inference() -> None:
    """``Measure.format`` if explicitly set wins over unit-based guess."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="revenue",
                    sql="{o}.x",
                    agg="sum",
                    unit="count",  # would infer "integer"
                    format="percent",  # but explicit wins
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    col = next(c for c in decision.columns if c.name == "revenue")
    assert col.format == "percent"


# ---------------------------------------------------------------------------
# Title and axis labels
# ---------------------------------------------------------------------------


def test_title_combines_measure_and_dimension_labels() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert "Revenue" in decision.title
    assert "Region" in decision.title


def test_bar_chart_axis_labels_filled() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "bar_chart"
    assert decision.x_axis == "Region"
    assert decision.y_axes == ["Revenue"]


def test_pie_chart_axes_labels_single_value() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "pie_chart"
    assert decision.x_axis == "Region"
    assert decision.y_axes == ["Revenue"]


def test_data_table_has_no_axes() -> None:
    from semql.visualize import HEATMAP_MAX_CELLS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region", "orders.status"],
        ),
        n_rows=HEATMAP_MAX_CELLS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "data_table"
    assert decision.x_axis is None
    assert decision.y_axes == []


# ---------------------------------------------------------------------------
# Display name overrides _humanize
# ---------------------------------------------------------------------------


def test_explicit_display_name_overrides_humanize() -> None:
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="revenue",
                    sql="{o}.x",
                    agg="sum",
                    unit="currency",
                    display_name="Net Revenue (USD)",
                ),
            ],
            "dimensions": [
                Dimension(
                    name="region",
                    sql="{o}.region",
                    type="string",
                    display_name="Sales Region",
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog={"orders": cube},
    )
    revenue_col = next(c for c in decision.columns if c.name == "revenue")
    region_col = next(c for c in decision.columns if c.name == "region")
    assert revenue_col.display_name == "Net Revenue (USD)"
    assert region_col.display_name == "Sales Region"


# ---------------------------------------------------------------------------
# VizColumn dataclass shape
# ---------------------------------------------------------------------------


def test_viz_column_can_be_constructed_directly() -> None:
    col = VizColumn(name="x", display_name="X", format="raw", is_measure=False, is_time=False)
    assert col.name == "x"
    assert col.format == "raw"


# ---------------------------------------------------------------------------
# Unit / display_unit propagation
# ---------------------------------------------------------------------------


def test_measure_unit_and_display_unit_surface_on_viz_column() -> None:
    """Both unit fields ride out on VizColumn so downstream renderers
    can call units.convert(unit, display_unit) and apply the factor
    to row data."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="watch_time",
                    sql="{o}.duration",
                    agg="sum",
                    unit="seconds",
                    display_unit="hours",
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.watch_time"], dimensions=["orders.region"]),
        n_rows=3,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    wt = next(c for c in decision.columns if c.name == "watch_time")
    assert wt.unit == "seconds"
    assert wt.display_unit == "hours"


def test_display_unit_drives_format_inference_for_time() -> None:
    """``unit="seconds", display_unit="hours"`` should infer
    ``format="duration"`` (hours is a time unit). Without
    display_unit, the same measure would still be a duration because
    the storage unit is seconds — this test specifically checks the
    display_unit path takes precedence."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="session_ms",
                    sql="{o}.x",
                    agg="avg",
                    unit="bytes",  # would NOT infer duration
                    display_unit="hours",  # but this WOULD
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.session_ms"], dimensions=["orders.region"]),
        n_rows=3,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    col = next(c for c in decision.columns if c.name == "session_ms")
    assert col.format == "duration"


def test_dimension_unit_fields_surface_on_viz_column() -> None:
    """Dimensions can also carry unit / display_unit (e.g. a
    duration_seconds dimension); they should propagate the same way."""
    cube = _orders().model_copy(
        update={
            "dimensions": [
                Dimension(
                    name="duration",
                    sql="{o}.dur",
                    type="number",
                    unit="seconds",
                    display_unit="minutes",
                    format="duration",
                ),
                Dimension(name="region", sql="{o}.region", type="string"),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.duration"]),
        n_rows=3,
        catalog={"orders": cube},
    )
    dur = next(c for c in decision.columns if c.name == "duration")
    assert dur.unit == "seconds"
    assert dur.display_unit == "minutes"
    assert dur.format == "duration"


# ---------------------------------------------------------------------------
# Expanded chart vocabulary: area, histogram
# ---------------------------------------------------------------------------


def test_area_chart_for_multi_measure_same_unit_time_series() -> None:
    """Several *additive same-unit* measures over time compose → stacked
    area (a single-measure time series stays a line)."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
                Measure(name="refunds", sql="{o}.refund", agg="sum", unit="currency"),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.refunds"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    assert decision.chart_type == "area_chart"
    assert decision.y_axes == ["Revenue", "Refunds"]


def test_diverging_unit_multi_measure_time_series_overlays_lines() -> None:
    """Revenue (currency) and order count (count) can't be summed onto one
    stacked axis — overlay lines instead of an area, and record the reject."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.orders"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind == "time_series_overlaid_line"
    assert "area_chart" in decision.reason.alternatives
    assert decision.y_axes == ["Revenue", "Orders"]


def test_percent_measure_time_series_is_not_stacked() -> None:
    """A percentage/ratio measure is an average, not a sum — a multi-measure
    time series that includes one must not stack into an area."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.conversion_rate"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind == "time_series_overlaid_line"


def test_same_unit_different_display_unit_still_stacks() -> None:
    """Two measures with the same true ``unit`` but different
    ``display_unit`` conversions (e.g. one shown in seconds, one in hours)
    are the same dimensional quantity — they must still stack, not be
    wrongly refused for a difference that's presentation-only."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="revenue",
                    sql="{o}.amount",
                    agg="sum",
                    unit="currency",
                    display_unit="usd",
                ),
                Measure(
                    name="refunds",
                    sql="{o}.refund",
                    agg="sum",
                    unit="currency",
                    display_unit="cents",
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.refunds"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    assert decision.chart_type == "area_chart"


def test_diverging_unit_same_display_unit_does_not_stack() -> None:
    """Two measures with genuinely different true units that happen to
    share a ``display_unit`` string must not be stacked — the true unit,
    not the display label, decides whether summing them means anything."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(
                    name="revenue",
                    sql="{o}.amount",
                    agg="sum",
                    unit="currency",
                    display_unit="total",
                ),
                Measure(
                    name="session_count",
                    sql="{o}.session_id",
                    agg="count",
                    unit="count",
                    display_unit="total",
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.session_count"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind == "time_series_overlaid_line"


def test_time_series_with_category_is_multi_series_line() -> None:
    """A single measure over time *and* a categorical dimension is one line
    per category: the time column is the x axis and the category is the
    series. Previously the category was silently dropped from the decision."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=60,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind == "time_series_multi_series"
    assert decision.x_axis is not None and "Created At" in decision.x_axis
    assert decision.series == "Region"
    assert decision.y_axes == ["Revenue"]


def test_long_daily_series_with_category_prefers_multi_series_over_calendar() -> None:
    """A calendar heatmap can't show categories — a long daily series broken
    down by a dimension stays a multi-series line, not a calendar heatmap."""
    from semql.visualize import CALENDAR_MIN_DAYS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-12-31"),
            ),
        ),
        n_rows=CALENDAR_MIN_DAYS + 100,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"
    assert decision.series == "Region"


def test_histogram_for_numeric_dimension() -> None:
    """One measure over a single *numeric* dimension is a distribution →
    histogram (a categorical dimension would be a bar/pie)."""
    cube = _orders().model_copy(
        update={"dimensions": [Dimension(name="bucket", sql="{o}.bucket", type="number")]}
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.bucket"]),
        n_rows=12,
        catalog={"orders": cube},
    )
    assert decision.chart_type == "histogram"
    assert decision.x_axis == "Bucket"
    assert decision.y_axes == ["Revenue"]


def test_histogram_for_large_numeric_dimension() -> None:
    """A distribution over *many* numeric buckets is the strongest histogram
    case, not a data table — the numeric-dimension branch has no upper row
    cap (a categorical dimension of the same size would still fall to a
    table)."""
    cube = _orders().model_copy(
        update={"dimensions": [Dimension(name="bucket", sql="{o}.bucket", type="number")]}
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.bucket"]),
        n_rows=BAR_MAX_BARS + 200,
        catalog={"orders": cube},
    )
    assert decision.chart_type == "histogram"


# ---------------------------------------------------------------------------
# Client-declared chart capabilities (supported_charts)
# ---------------------------------------------------------------------------


def test_supported_charts_falls_back_when_pick_unsupported() -> None:
    """1 dim + 1 measure (small n) naturally picks a pie; a client that
    only draws bar/table falls back to data_table and says so."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=frozenset({"bar_chart", "data_table"}),
    )
    assert decision.chart_type == "data_table"
    assert "unsupported" in decision.reason.note


def test_supported_charts_respected_when_available() -> None:
    """When the natural pick is in the supported set, it is used unchanged."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=frozenset({"pie_chart", "data_table"}),
    )
    assert decision.chart_type == "pie_chart"


def test_supported_charts_none_imposes_no_constraint() -> None:
    """The default (no declaration) leaves the natural pick untouched."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=None,
    )
    assert decision.chart_type == "pie_chart"


# ---------------------------------------------------------------------------
# Heatmaps: calendar (timeline) + xy (correlation matrix)
# ---------------------------------------------------------------------------


def test_calendar_heatmap_for_long_daily_series() -> None:
    """A long per-day single-measure series → GitHub-style calendar heatmap;
    the time column is the x axis and the measure colours the day cells."""
    from semql.visualize import CALENDAR_MIN_DAYS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-12-31"),
            ),
        ),
        n_rows=CALENDAR_MIN_DAYS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "calendar_heatmap"
    assert decision.y_axes == ["Revenue"]


def test_short_daily_series_stays_line() -> None:
    """A short daily series is a line, not a calendar heatmap."""
    from semql.visualize import CALENDAR_MIN_DAYS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=CALENDAR_MIN_DAYS,  # not strictly greater → line
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "line_chart"


def test_xy_heatmap_for_two_dim_grid() -> None:
    """Two categorical dims + one measure over a grid too large to stack →
    xy heatmap: dim1 is the x axis, dim2 the row series, measure the colour."""
    from semql.visualize import STACKED_BAR_MAX_CELLS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region", "orders.status"],
        ),
        n_rows=STACKED_BAR_MAX_CELLS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "xy_heatmap"
    assert decision.x_axis == "Region"
    assert decision.series == "Status"
    assert decision.y_axes == ["Revenue"]


# ---------------------------------------------------------------------------
# Compare queries — the time-series branch would mis-classify as area_chart
# ---------------------------------------------------------------------------


def test_compare_with_time_picks_compare_line_chart() -> None:
    """A compare query with a time breakdown emits current/prior/delta
    columns. The cardinality-only branch would call that "more measures"
    and pick area_chart; compare_line_chart is the explicit shape."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="month",
                range=("2026-01-01", "2026-04-01"),
            ),
            compare=CompareWindow(mode="previous_period"),
        ),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "compare_line_chart"
    assert decision.reason.kind == "compare_current_prior"
    assert decision.x_axis is not None and "Created At" in decision.x_axis
    # The y-axes are the synthetic per-measure compare columns the
    # compiler emitted (current / prior / delta / pct_change).
    y_names = " | ".join(decision.y_axes)
    for needle in ("current", "prior", "delta", "% change"):
        assert needle in y_names, f"missing {needle!r} in y_axes={decision.y_axes}"


# Note: the "compare without time" case is unreachable through the real
# compiler — the compile-time check in ``_validate_compare_shape`` raises
# before the visualiser runs. The visualiser's defensive branch stays
# in the code as a future-proofing default in case the validation
# order ever changes.


# ---------------------------------------------------------------------------
# DecisionReason — typed alternative to the old reason: str
# ---------------------------------------------------------------------------


def test_reason_is_typed_decision_reason_with_kind() -> None:
    """The decision's reason is a structured value, not a debug string."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert isinstance(decision.reason, DecisionReason)
    assert decision.reason.kind == "pie_small"
    assert "1 dim, 1 measure" in decision.reason.note
    # Pie is the natural pick — no rejection, alternatives stays empty.
    assert decision.reason.alternatives == []


def test_reason_alternatives_record_rejected_pie() -> None:
    """When ShapeStats says "negatives" we reject the natural pie and
    record the rejection in ``alternatives`` for audit surfaces."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(has_negatives=True),
    )
    assert decision.chart_type == "bar_chart"
    assert decision.reason.kind == "shape_stats_fallback"
    assert "pie_chart" in decision.reason.alternatives
    assert "negatives" in decision.reason.note


def test_reason_client_capability_fallback_records_rejected() -> None:
    """When the renderer can't draw the natural pick, the rejected
    type is recorded alongside the fallback."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=frozenset({"bar_chart", "data_table"}),
    )
    assert decision.chart_type == "data_table"
    assert decision.reason.kind == "client_capability_fallback"
    assert "pie_chart" in decision.reason.alternatives


def test_reason_str_round_trip_for_debug_print() -> None:
    """``str(reason)`` is the human-readable note, so the old
    ``assert "x" in decision.reason`` pattern still works through the
    note field for ad-hoc debugging without parsing ``kind``."""
    decision = _decide(
        SemanticQuery(dimensions=["orders.region"], ungrouped=True, limit=5),
        n_rows=5,
        catalog=_catalog(_orders()),
    )
    assert "ungrouped" in str(decision.reason)


# ---------------------------------------------------------------------------
# ShapeStats — post-execute override hook
# ---------------------------------------------------------------------------


def test_shape_stats_has_negatives_downgrades_pie_to_bar() -> None:
    """A pie of negative values is meaningless — caller-known negatives
    downgrade the natural pie to a bar."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(has_negatives=True),
    )
    assert decision.chart_type == "bar_chart"
    assert decision.reason.kind == "shape_stats_fallback"


def test_shape_stats_single_category_falls_to_text_only() -> None:
    """One distinct category is degenerate for a pie/bar — fall to
    text_only so the caller surfaces the single value, not a chart."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(n_distinct_categories=1),
    )
    assert decision.chart_type == "text_only"
    assert "n_distinct_categories=1" in decision.reason.note


def test_shape_stats_sparse_skips_calendar_heatmap() -> None:
    """A daily series with mostly-empty days shouldn't be a calendar
    heatmap — caller-known sparsity keeps the line branch."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-12-31"),
            ),
        ),
        n_rows=200,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(is_sparse=True),
    )
    assert decision.chart_type == "line_chart"


def test_shape_stats_none_keeps_cardinality_decision() -> None:
    """``shape_stats=None`` (the default) is no-op — the cardinality
    decision stands."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=None,
    )
    assert decision.chart_type == "pie_chart"


def test_shape_stats_is_flat_helper() -> None:
    """``ShapeStats.is_flat`` is True when min == max, None when either
    bound is missing."""
    assert ShapeStats(measure_min=5.0, measure_max=5.0).is_flat is True
    assert ShapeStats(measure_min=5.0, measure_max=7.0).is_flat is False
    assert ShapeStats(measure_min=5.0).is_flat is None
    assert ShapeStats().is_flat is None


# ---------------------------------------------------------------------------
# Compare-line chart axis handling
# ---------------------------------------------------------------------------


def test_compare_line_chart_x_axis_is_time_axis() -> None:
    """Compare-line follows the same x-axis rule as line/area: the
    first non-measure column (the time bucket) is the x axis; all
    measures (now including _current / _prior / _delta / _pct_change
    variants) ride on y_axes."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="month",
                range=("2026-01-01", "2026-04-01"),
            ),
            compare=CompareWindow(mode="previous_period"),
        ),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "compare_line_chart"
    assert decision.x_axis is not None and "Created At" in decision.x_axis
    # All four per-measure compare columns show up as y series.
    y_names = " | ".join(decision.y_axes)
    for needle in ("current", "prior", "delta", "% change"):
        assert needle in y_names, f"missing {needle!r} in y_axes={decision.y_axes}"


# ---------------------------------------------------------------------------
# §1 — Confidence: coarse enum mapped from the winning reason kind.
# ---------------------------------------------------------------------------


def test_confidence_high_for_unambiguous_shape() -> None:
    """A time series with granularity is unambiguous — escalation
    wouldn't help, so confidence is high."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.reason.kind == "time_series_line"
    assert decision.confidence == "high"


def test_confidence_medium_for_sound_default() -> None:
    """A small pie is a sound default the question could override —
    medium confidence."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.reason.kind == "pie_small"
    assert decision.confidence == "medium"


def test_confidence_low_for_dense_marginal_pick() -> None:
    """An xy heatmap is the dense/marginal grid case — low confidence."""
    from semql.visualize import STACKED_BAR_MAX_CELLS

    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region", "orders.status"],
        ),
        n_rows=STACKED_BAR_MAX_CELLS + 1,
        catalog=_catalog(_orders()),
    )
    assert decision.reason.kind == "xy_heatmap"
    assert decision.confidence == "low"


def test_confidence_high_for_client_capability_fallback() -> None:
    """A hard client-capability constraint is high confidence — the
    renderer can't draw anything else anyway."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=frozenset({"bar_chart", "data_table"}),
    )
    assert decision.reason.kind == "client_capability_fallback"
    assert decision.confidence == "high"


# ---------------------------------------------------------------------------
# §2 — Candidates: a constrained, deduped, capability-filtered menu.
# ---------------------------------------------------------------------------


def test_candidates_chosen_first() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert decision.candidates[0].chart_type == decision.chart_type
    assert decision.candidates[0].confidence == decision.confidence
    assert decision.candidates[0].reason == decision.reason


def test_candidates_are_deduped() -> None:
    """``alternatives`` recorded a rejected 'pie_chart'; the runner-up
    path for the chosen ``bar_chart`` is empty, so the menu has no
    repeats even though both sources could otherwise collide."""
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(has_negatives=True),
    )
    chart_types = [c.chart_type for c in decision.candidates]
    assert len(chart_types) == len(set(chart_types))
    assert "pie_chart" in chart_types


def test_candidates_include_universal_fallbacks() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    chart_types = {c.chart_type for c in decision.candidates}
    assert "data_table" in chart_types
    assert "text_only" in chart_types


def test_candidates_filtered_by_supported_charts() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        supported_charts=frozenset({"bar_chart", "data_table"}),
    )
    chart_types = {c.chart_type for c in decision.candidates}
    assert chart_types <= {"bar_chart", "data_table"}


def test_candidates_runner_up_is_low_confidence() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    runner_ups = [c for c in decision.candidates if c.chart_type != decision.chart_type]
    assert runner_ups  # pie has runner-ups (bar, data_table, text_only)
    assert all(c.confidence == "low" for c in runner_ups)


# ---------------------------------------------------------------------------
# §3 — Feature bundle: the structural "why".
# ---------------------------------------------------------------------------


def test_features_values_for_simple_pie() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    f = decision.features
    assert isinstance(f, VizFeatures)
    assert f.n_rows == 3
    assert f.n_measures == 1
    assert f.n_dimensions == 1
    assert f.n_categorical_dims == 1
    assert f.has_time_breakdown is False
    assert f.measures_additive is True  # revenue: agg=sum, not non_additive
    assert f.measures_share_unit is None  # only 1 measure — the question doesn't apply
    assert f.is_flat is None
    assert f.null_rate is None
    assert f.caveats == []


def test_measures_additive_false_beats_unknown() -> None:
    """A known non-additive measure alongside one of unknown additivity
    must report ``False`` (per the docstring: "False iff at least one
    measure is known non-additive"), not ``None`` — an unknown measure
    can't rescue a stacking decision a known non-additive one rules out."""
    from semql.visualize import _measures_additive

    known_non_additive = VizColumn(
        name="conversion_rate",
        display_name="Conversion Rate",
        format="percent",
        is_measure=True,
        is_time=False,
        additive=False,
    )
    unknown = VizColumn(
        name="derived",
        display_name="Derived",
        format="raw",
        is_measure=True,
        is_time=False,
        additive=None,
    )
    assert _measures_additive([known_non_additive, unknown]) is False


def test_features_has_time_breakdown_true_for_time_series() -> None:
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.features.has_time_breakdown is True
    assert decision.features.n_dimensions == 1


# ---------------------------------------------------------------------------
# §4 — is_flat → text_only; null_rate → caveat only (no chart change).
# ---------------------------------------------------------------------------


def test_is_flat_forces_text_only_with_caveat() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(measure_min=5.0, measure_max=5.0),
    )
    assert decision.chart_type == "text_only"
    assert decision.reason.kind == "flat_series"
    assert decision.confidence == "high"
    assert decision.features.is_flat is True
    assert decision.features.caveats  # non-empty
    assert any("variation" in c for c in decision.features.caveats)


def test_is_flat_time_series_stays_line() -> None:
    """A flat *time* series is NOT collapsed: a constant line over time
    legitimately shows "this held steady", so it stays a line. The flatness
    still rides out as an advisory caveat on the feature bundle."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(measure_min=10.0, measure_max=10.0),
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind != "flat_series"
    # Flatness is surfaced, not acted on, for time series.
    assert decision.features.is_flat is True
    assert any("variation" in c for c in decision.features.caveats)


def test_null_rate_above_threshold_adds_caveat_without_changing_chart() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(null_rate=NULL_RATE_CAVEAT_THRESHOLD),
    )
    assert decision.chart_type == "pie_chart"  # unchanged
    assert decision.features.null_rate == NULL_RATE_CAVEAT_THRESHOLD
    assert decision.features.caveats
    assert any("null_rate" in c for c in decision.features.caveats)


def test_null_rate_below_threshold_no_caveat() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(null_rate=NULL_RATE_CAVEAT_THRESHOLD - 0.05),
    )
    assert decision.chart_type == "pie_chart"
    assert decision.features.caveats == []


# ---------------------------------------------------------------------------
# §5 — Additivity drives stacking; ordinal dimension drives sort; bubble.
# ---------------------------------------------------------------------------


def test_non_additive_measure_flag_forces_overlaid_lines() -> None:
    """Two sum measures sharing a unit would normally stack — but one is
    explicitly flagged ``non_additive`` (e.g. a running/point-in-time
    balance that shouldn't be summed) → overlay lines instead."""
    cube = _orders().model_copy(
        update={
            "measures": [
                Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
                Measure(
                    name="balance",
                    sql="{o}.balance",
                    agg="sum",
                    unit="currency",
                    non_additive=True,
                ),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.balance"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog={"orders": cube.model_copy(update={"dimensions": _orders().dimensions})},
    )
    assert decision.chart_type == "line_chart"
    assert decision.reason.kind == "time_series_overlaid_line"
    revenue_col = next(c for c in decision.columns if c.name == "revenue")
    balance_col = next(c for c in decision.columns if c.name == "balance")
    assert revenue_col.additive is True
    assert balance_col.additive is False


def test_ordinal_x_axis_sorts_natural() -> None:
    """An ordinal dimension (weekday) sorts by its natural order, not by
    measure value — the default bar/pie sort."""
    cube = _orders().model_copy(
        update={
            "dimensions": [
                Dimension(name="weekday", sql="{o}.weekday", type="string", ordinal=True),
            ],
        }
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.weekday"]),
        n_rows=15,  # > PIE_MAX_SLICES so it's the bar_chart branch
        catalog={"orders": cube},
    )
    assert decision.chart_type == "bar_chart"
    assert decision.hints.sort == "natural"
    weekday_col = next(c for c in decision.columns if c.name == "weekday")
    assert weekday_col.ordinal is True


def test_non_ordinal_x_axis_sorts_value_desc() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "bar_chart"
    assert decision.hints.sort == "value_desc"


def test_histogram_sort_hint_is_natural() -> None:
    cube = _orders().model_copy(
        update={"dimensions": [Dimension(name="bucket", sql="{o}.bucket", type="number")]}
    )
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.bucket"]),
        n_rows=12,
        catalog={"orders": cube},
    )
    assert decision.chart_type == "histogram"
    assert decision.hints.sort == "natural"


def test_bubble_chart_for_three_measures_one_dim() -> None:
    """3 measures + 1 dim → bubble chart: x=m0, y=m1, size=m2; the
    dimension labels each point."""
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue", "orders.orders", "orders.conversion_rate"],
            dimensions=["orders.region"],
        ),
        n_rows=5,
        catalog=_catalog(_orders()),
    )
    assert decision.chart_type == "bubble_chart"
    assert decision.reason.kind == "bubble_xyz"
    assert decision.x_axis == "Revenue"
    assert decision.y_axes == ["Orders"]
    assert decision.size_axis == "Conversion Rate"
    assert decision.confidence == "high"


# ---------------------------------------------------------------------------
# §6 — Render hints: y_scale (log), top_n.
# ---------------------------------------------------------------------------


def test_log_scale_hint_for_strong_positive_skew() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,  # bar_chart branch
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(measure_min=1.0, measure_max=1.0 * LOG_SCALE_RATIO),
    )
    assert decision.chart_type == "bar_chart"
    assert decision.hints.y_scale == "log"


def test_linear_scale_without_stats() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,
        catalog=_catalog(_orders()),
    )
    assert decision.hints.y_scale == "linear"


def test_linear_scale_for_mild_ratio() -> None:
    decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,
        catalog=_catalog(_orders()),
        shape_stats=ShapeStats(measure_min=1.0, measure_max=10.0),
    )
    assert decision.hints.y_scale == "linear"


def test_top_n_hint_for_pie_and_bar() -> None:
    pie_decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=3,
        catalog=_catalog(_orders()),
    )
    assert pie_decision.hints.top_n == PIE_MAX_SLICES

    bar_decision = _decide(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
        n_rows=15,
        catalog=_catalog(_orders()),
    )
    assert bar_decision.hints.top_n == BAR_MAX_BARS


def test_top_n_hint_is_none_for_other_charts() -> None:
    decision = _decide(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        ),
        n_rows=31,
        catalog=_catalog(_orders()),
    )
    assert decision.hints.top_n is None


def test_render_hints_can_be_constructed_directly() -> None:
    hints = RenderHints()
    assert hints.y_scale == "linear"
    assert hints.sort is None
    assert hints.top_n is None


def test_scored_chart_can_be_constructed_directly() -> None:
    sc = ScoredChart(
        chart_type="bar_chart",
        confidence="low",
        reason=DecisionReason(kind="bar_medium", note="x"),
    )
    assert sc.chart_type == "bar_chart"
