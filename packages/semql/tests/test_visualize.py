# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
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
from semql.spec import SemanticQuery, TimeWindow
from semql.visualize import (
    BAR_MAX_BARS,
    PIE_MAX_SLICES,
    VizColumn,
    VizDecision,
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
    assert "default_chart_type" in decision.reason


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
    assert "ungrouped" in decision.reason


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


def test_area_chart_for_multi_measure_time_series() -> None:
    """A time series with several measures composes over time → stacked
    area (a single-measure time series stays a line)."""
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
    assert decision.chart_type == "area_chart"
    assert decision.y_axes == ["Revenue", "Orders"]


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
    assert "unsupported" in decision.reason


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
