"""Deterministic visualization decision from a `CompiledQuery` bundle.

Given a compiled query, we know:
  - Per-output-column kind, display name, unit, display_unit, format
    (from ``CompiledQuery.column_meta``).
  - Which cubes the query touched (from ``CompiledQuery.touched_cube_names``)
    so we can apply any ``Cube.default_chart_type`` override.
  - How many rows the query produced (passed in as ``n_rows``).
  - The originating ``SemanticQuery`` for shape facts the compiler
    doesn't surface (``ungrouped`` flag, granularity).

That's enough to pick chart type, axes, formats, and labels without
re-resolving the query against the catalog. The function returns a
``VizDecision``; callers can serialise it as a hint for a presenter
LLM or apply it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from semql.compile import ColumnMeta, CompiledQuery
from semql.model import ChartTypeLiteral, Cube, FormatLiteral, StorageType
from semql.spec import SemanticQuery

PIE_MAX_SLICES = 10
BAR_MAX_BARS = 30

# Chart types the picker can emit, plus the ``text_only`` viz-only fallback.
VizChartType = ChartTypeLiteral | Literal["text_only"]

# Numeric storage types — a single one of these on a dimension axis marks a
# distribution (histogram) rather than a categorical breakdown (bar).
_NUMERIC_STORAGE: frozenset[StorageType] = frozenset({"integer", "float", "number"})


@dataclass
class VizColumn:
    """Per-output-column presentation metadata. Order matches `CompiledQuery.columns`.

    ``unit`` is the storage unit (e.g. ``"seconds"``) and ``display_unit``
    is the unit the value should be rendered in (e.g. ``"hours"``). The
    visualizer doesn't convert values — it only surfaces the pair so a
    downstream renderer can call ``catalog.unit_registry.factor(unit,
    display_unit)`` and apply the multiplier to row data.
    """

    name: str
    display_name: str
    format: FormatLiteral
    is_measure: bool
    is_time: bool
    unit: str | None = None
    display_unit: str | None = None
    storage_type: StorageType | None = None


@dataclass
class VizDecision:
    chart_type: VizChartType
    title: str
    x_axis: str | None
    y_axes: list[str]
    columns: list[VizColumn]
    reason: str = ""
    # The breakdown/series dimension for a ``stacked_bar_chart`` (the second
    # dimension whose values become the stacks). ``None`` for every other
    # chart type.
    series: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _humanize(name: str) -> str:
    return name.replace("_", " ").title()


_TIME_UNITS = frozenset(
    {
        "seconds",
        "s",
        "sec",
        "secs",
        "minutes",
        "min",
        "mins",
        "hours",
        "hr",
        "hrs",
        "days",
        "day",
        "weeks",
        "wk",
        "wks",
        "milliseconds",
        "ms",
        "microseconds",
        "us",
        "µs",
        "duration",
    }
)


def _infer_format(meta: ColumnMeta) -> FormatLiteral:
    if meta.format is not None:
        return meta.format
    # display_unit (if set) tells us how the value will be SHOWN —
    # prefer it for format inference so e.g. seconds-stored-as-hours
    # still renders as a duration.
    hint = (meta.display_unit or meta.unit or "").lower()
    if hint in ("pct", "percent"):
        return "percent"
    if hint == "count":
        return "integer"
    if hint in _TIME_UNITS:
        return "duration"
    return "raw"


# ---------------------------------------------------------------------------
# Chart-type decision — pure on the spec
# ---------------------------------------------------------------------------


def _pick_chart_type(
    query: SemanticQuery,
    touched_cubes: list[Cube],
    n_rows: int,
    columns: list[VizColumn],
) -> tuple[VizChartType, str]:
    overrides: set[ChartTypeLiteral] = {
        c.default_chart_type for c in touched_cubes if c.default_chart_type is not None
    }
    if len(overrides) == 1:
        chart = next(iter(overrides))
        return chart, f"cube default_chart_type={chart}"

    if query.ungrouped:
        return "data_table", "ungrouped row listing"

    n_measures = len(query.measures)
    has_time_breakdown = (
        query.time_dimension is not None and query.time_dimension.granularity is not None
    )
    n_dims = len(query.dimensions) + (1 if has_time_breakdown else 0)
    # Categorical (non-measure, non-time) axis columns — the dimensions a
    # chart breaks down by. Used to tell a numeric distribution (histogram)
    # from a categorical breakdown (bar / stacked bar).
    cat_dims = [c for c in columns if not c.is_measure and not c.is_time]

    if n_measures >= 1 and n_dims == 0:
        return "text_only", "single-value answer"

    # Time series. Several measures composing over time → stacked area; a
    # single series → line.
    if has_time_breakdown:
        if n_measures >= 2:
            return "area_chart", f"time series, {n_measures} measures (stacked composition)"
        return "line_chart", "time series with granularity"

    # Two measures with one labelling dimension → XY scatter.
    if n_measures == 2 and n_dims == 1:
        return "scatter_chart", "2 measures plotted against each other"

    # One measure over a single *numeric* dimension → frequency distribution.
    if (
        n_dims == 1
        and n_measures == 1
        and len(cat_dims) == 1
        and cat_dims[0].storage_type in _NUMERIC_STORAGE
        and n_rows <= BAR_MAX_BARS
    ):
        return "histogram", "1 measure over a numeric dimension (distribution)"

    if n_dims == 1 and n_measures == 1 and n_rows <= PIE_MAX_SLICES:
        return "pie_chart", f"1 dim, 1 measure, n_rows={n_rows} <= {PIE_MAX_SLICES}"

    if n_dims == 1 and n_rows <= BAR_MAX_BARS:
        return "bar_chart", f"1 dim, n_rows={n_rows} <= {BAR_MAX_BARS}"

    # Two categorical dimensions + one measure → a primary axis broken down
    # by the second dimension (stacked bars).
    if n_dims == 2 and n_measures == 1 and len(cat_dims) == 2 and n_rows <= BAR_MAX_BARS:
        return "stacked_bar_chart", f"2 dims, 1 measure, n_rows={n_rows} (primary axis + breakdown)"

    return "data_table", f"multi-dim or n_rows={n_rows} too large for a chart"


def _apply_supported(
    chart_type: VizChartType,
    reason: str,
    supported: frozenset[VizChartType] | None,
) -> tuple[VizChartType, str]:
    """Constrain the natural choice to the client's declared capabilities.

    ``supported`` is the set of chart types the caller's renderer can draw.
    ``None`` (or empty) means no constraint. When the picked chart isn't
    supported, fall back to the most universal supported option
    (``data_table`` then ``text_only``), else the first supported type."""
    if not supported or chart_type in supported:
        return chart_type, reason
    for fallback in ("data_table", "text_only"):
        if fallback in supported:
            return fallback, f"{reason}; {chart_type} unsupported by client → {fallback}"
    chosen = sorted(supported)[0]
    return chosen, f"{reason}; {chart_type} unsupported by client → {chosen}"


def _viz_column(meta: ColumnMeta) -> VizColumn:
    return VizColumn(
        name=meta.name,
        display_name=meta.display_name or _humanize(meta.name),
        format=_infer_format(meta),
        is_measure=meta.kind in ("measure", "computed"),
        is_time=meta.kind == "time",
        unit=meta.unit,
        display_unit=meta.display_unit,
        storage_type=meta.storage_type,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def decide_visualization(
    query: SemanticQuery,
    compiled: CompiledQuery,
    n_rows: int,
    *,
    catalog: dict[str, Cube],
    supported_charts: frozenset[VizChartType] | None = None,
) -> VizDecision:
    """Return chart_type + axis labels + per-column formats for a query.

    Reads from ``compiled.column_meta`` and ``compiled.touched_cube_names``
    — no catalog re-resolution. ``catalog`` is consulted only to look
    up ``Cube.default_chart_type`` for the touched cubes.

    ``query`` carries shape facts the compiler doesn't surface on
    ``CompiledQuery`` (``ungrouped`` flag, time-dimension granularity).
    ``n_rows`` is the actual row count; pass ``0`` for dry-run / explain
    paths.

    ``supported_charts`` lets the *calling renderer* declare which chart
    types it can draw. Chart support is a property of the rendering client,
    not the data model, so it's a call-time argument rather than a catalog
    field. When the naturally-best chart isn't in the set, the decision
    falls back to a supported type (``data_table`` / ``text_only`` first).
    ``None`` imposes no constraint.
    """
    touched_cubes = [catalog[name] for name in compiled.touched_cube_names if name in catalog]
    ordered: list[VizColumn] = [_viz_column(m) for m in compiled.column_meta]
    chart_type, reason = _pick_chart_type(query, touched_cubes, n_rows, ordered)
    chart_type, reason = _apply_supported(chart_type, reason, supported_charts)

    x_axis: str | None = None
    y_axes: list[str] = []
    series: str | None = None
    non_measures = [c for c in ordered if not c.is_measure]
    measures_out = [c for c in ordered if c.is_measure]
    if chart_type in ("bar_chart", "line_chart", "area_chart", "histogram"):
        if non_measures:
            x_axis = non_measures[0].display_name
        y_axes = [c.display_name for c in measures_out]
    elif chart_type == "stacked_bar_chart":
        # First dimension is the primary axis; the second becomes the stack.
        if non_measures:
            x_axis = non_measures[0].display_name
        if len(non_measures) >= 2:
            series = non_measures[1].display_name
        y_axes = [c.display_name for c in measures_out]
    elif chart_type == "scatter_chart":
        # Both axes are measures; the dimension labels the points.
        if len(measures_out) >= 2:
            x_axis = measures_out[0].display_name
            y_axes = [measures_out[1].display_name]
    elif chart_type == "pie_chart":
        x_axis = non_measures[0].display_name if non_measures else None
        y_axes = [measures_out[0].display_name] if measures_out else []

    title_bits: list[str] = []
    first_measure = next((c for c in ordered if c.is_measure), None)
    first_non_measure = next((c for c in ordered if not c.is_measure), None)
    first_time = next((c for c in ordered if c.is_time), None)
    if first_measure is not None:
        title_bits.append(first_measure.display_name)
    if first_non_measure is not None and not first_non_measure.is_time:
        title_bits.append("by " + first_non_measure.display_name)
    elif first_time is not None:
        title_bits.append("over " + first_time.display_name)
    title = " ".join(title_bits) if title_bits else "Result"

    return VizDecision(
        chart_type=chart_type,
        title=title,
        x_axis=x_axis,
        y_axes=y_axes,
        columns=ordered,
        reason=reason,
        series=series,
    )


__all__ = [
    "BAR_MAX_BARS",
    "PIE_MAX_SLICES",
    "VizColumn",
    "VizDecision",
    "decide_visualization",
]
