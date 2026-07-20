"""Deterministic visualization decision from a `CompiledQuery` bundle.

Given a compiled query, we know:
  - Per-output-column kind, display name, unit, display_unit, format
    (from ``CompiledQuery.column_meta``).
  - Which cubes the query touched (from ``CompiledQuery.touched_cube_names``)
    so we can apply any ``Cube.default_chart_type`` override.
  - How many rows the query produced (passed in as ``n_rows``).
  - The originating ``SemanticQuery`` for shape facts the compiler
    doesn't surface (``ungrouped`` flag, granularity, ``compare`` flag).
  - Optionally, a caller-computed :class:`ShapeStats` describing the
    actual result distribution (negatives, distinctness, nulls, etc.).

That's enough to pick chart type, axes, formats, and labels without
re-resolving the query against the catalog. The function returns a
``VizDecision``; callers can serialise it as a hint for a presenter
LLM or apply it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from semql.compile import ColumnMeta, CompiledQuery
from semql.model import ChartTypeLiteral, Cube, FormatLiteral, StorageType
from semql.spec import SemanticQuery

PIE_MAX_SLICES = 10
BAR_MAX_BARS = 30
# Two categorical dims + one measure: a small grid stacks cleanly as bars;
# a larger one reads better as an xy heatmap; past the cell cap it's a table.
STACKED_BAR_MAX_CELLS = 12
HEATMAP_MAX_CELLS = 400
# A per-day time series longer than this many points is the GitHub-style
# calendar-heatmap case; shorter daily series stay a line.
CALENDAR_MIN_DAYS = 60
# ``ShapeStats.null_rate`` at or above this threshold earns a
# ``VizFeatures.caveats`` entry. Does NOT change the chart pick — it's
# a "trust but verify" flag for the consumer, not a shape decision.
NULL_RATE_CAVEAT_THRESHOLD = 0.2
# ``ShapeStats.measure_max / measure_min`` at or above this ratio (with
# both positive) is a strong positive skew — the log-scale render hint.
LOG_SCALE_RATIO = 1000

# Chart types the picker can emit, plus the viz-only ``"text_only"``
# fallback. ``"compare_line_chart"`` is the explicit shape for a
# ``SemanticQuery.compare`` result — current/prior/delta/pct_change
# columns side-by-side, not the stacked-area that the time-series
# branch would otherwise pick for multi-measure-with-time.
# ``"bubble_chart"`` is likewise viz-only (NOT part of the model's
# ``ChartTypeLiteral``): 3 measures + 1 dimension → x/y/size axes.
_CompareChart = Literal["compare_line_chart"]
_BubbleChart = Literal["bubble_chart"]
VizChartType = ChartTypeLiteral | _CompareChart | _BubbleChart | Literal["text_only"]

Confidence = Literal["high", "medium", "low"]

# Numeric storage types — a single one of these on a dimension axis marks a
# distribution (histogram) rather than a categorical breakdown (bar).
_NUMERIC_STORAGE: frozenset[StorageType] = frozenset({"integer", "float", "number"})


# ---------------------------------------------------------------------------
# Caller-computed result statistics — sans-I/O override on shape decisions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeStats:
    """Caller-computed summary statistics for the executed result rows.

    The visualizer decides on **shape cardinality** (counts of measures,
    dimensions, time buckets) but cannot see row values. A caller that
    has executed the SQL and computed basic statistics can pass them
    here to override cardinality-only decisions:

    - ``has_negatives`` — a pie of negative values is meaningless; force a bar.
    - ``measure_min`` / ``measure_max`` — detect a "flat" series (min == max)
      and refuse charts that imply variation.
    - ``n_distinct_categories`` — a pie / bar of a single distinct category
      is degenerate; fall back to a table.
    - ``null_rate`` — at or above ``NULL_RATE_CAVEAT_THRESHOLD`` this
      warrants a caveat surfaced in ``VizDecision.features.caveats``. Does
      NOT change the chart pick.
    - ``is_sparse`` — for the calendar-heatmap case: a daily series that
      is mostly empty should not be a calendar heatmap.

    All fields are optional; the caller can pass only what they have. The
    visualizer is conservative: when the caller *does* pass a stat, it
    takes precedence over the cardinality-only decision for the matching
    branch. When the caller passes ``None``, the cardinality decision
    stands. The visualizer never samples rows itself — that would couple
    it to the executor and break the sans-I/O invariant.
    """

    has_negatives: bool | None = None
    measure_min: float | None = None
    measure_max: float | None = None
    n_distinct_categories: int | None = None
    null_rate: float | None = None
    is_sparse: bool | None = None

    @property
    def is_flat(self) -> bool | None:
        """``True`` when the caller knows the series has zero variation.

        ``None`` when either bound is missing. Used to refuse charts that
        imply variation (line, bar) in favour of a table or text."""
        if self.measure_min is None or self.measure_max is None:
            return None
        return self.measure_min == self.measure_max


# ---------------------------------------------------------------------------
# Per-column presentation metadata
# ---------------------------------------------------------------------------


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
    # Measures only: additive iff ``agg in {"sum", "count"} and not
    # non_additive``, computed by the compiler (``ColumnMeta.additive``).
    # ``None`` for non-measure columns, or when built without going
    # through the compiler.
    additive: bool | None = None
    # Propagated from ``Dimension.ordinal`` / ``ColumnMeta.ordinal``.
    ordinal: bool = False


# ---------------------------------------------------------------------------
# Decision reason — typed per ``*Decision`` contract
# ---------------------------------------------------------------------------

DecisionReasonKind = Literal[
    # Per-cube override won outright.
    "cube_override",
    # Cardinality-only branches.
    "ungrouped_row",
    "single_value",
    "compare_current_prior",
    "time_series_line",
    # A time series broken down by a categorical dimension → one line per
    # category (the breakdown rides on ``VizDecision.series``). Distinct
    # from ``time_series_line`` (single series) so an audit surface can see
    # the breakdown was recognised rather than dropped.
    "time_series_multi_series",
    "time_series_area",
    # Several measures over time whose units diverge or which are
    # non-additive (a percentage/ratio): stacking would sum incomparable
    # quantities, so we overlay lines instead of stacking an area.
    "time_series_overlaid_line",
    "time_series_calendar_heatmap",
    "scatter_xy",
    # 3 measures + 1 dimension: x=m0, y=m1, size=m2, dim labels the points.
    "bubble_xyz",
    "histogram_distribution",
    "pie_small",
    "bar_medium",
    "stacked_bar",
    "xy_heatmap",
    # A series with zero variation (``ShapeStats.measure_min ==
    # measure_max``) shouldn't imply variation — refuse the chart and
    # fall back to a plain text answer.
    "flat_series",
    # Fallbacks — the cardinality branch was overridden by either a
    # caller-supplied ``ShapeStats`` or by client capability.
    "data_table_fallback",
    "text_only_fallback",
    "client_capability_fallback",
    "shape_stats_fallback",
    # Sentinel when no branch matched and the data_table default
    # wins by process of elimination.
    "no_chart_match",
]
"""Closed set of reasons a :class:`VizDecision` was made.

``kind`` is the parse-resistant tag for ``"why this chart?"`` consumers
(UI badges, audit logs, A/B experiment analysis). ``note`` is the
human-readable free-form explanation. ``alternatives`` lists the chart
types that were considered and rejected, so a "why not a pie?" tooltip
has something structured to show."""


@dataclass(frozen=True)
class DecisionReason:
    """Typed alternative to ``VizDecision.reason: str``.

    Per ``docs/specs/naming-convention.md:97`` the ``*Decision`` family
    carries the choice *and* the reasons/alternatives. ``note`` keeps
    the existing free-form string for human readers; ``kind`` is for
    parsers; ``alternatives`` records rejected chart types so an audit
    surface can answer "why not X?" without re-running the decision.
    """

    kind: DecisionReasonKind
    note: str
    alternatives: list[VizChartType] = field(default_factory=lambda: list[VizChartType]())

    def __str__(self) -> str:
        return self.note


# ---------------------------------------------------------------------------
# Confidence — coarse enum: how likely a human/LLM with the question
# would pick something different (escalation value), NOT "are we right".
# ---------------------------------------------------------------------------

_HIGH_CONFIDENCE_KINDS: frozenset[DecisionReasonKind] = frozenset(
    {
        "cube_override",
        "ungrouped_row",
        "single_value",
        "compare_current_prior",
        "time_series_line",
        "time_series_multi_series",
        "scatter_xy",
        "histogram_distribution",
        "bubble_xyz",
        "flat_series",
        "shape_stats_fallback",
        "client_capability_fallback",
    }
)
_MEDIUM_CONFIDENCE_KINDS: frozenset[DecisionReasonKind] = frozenset(
    {
        "pie_small",
        "bar_medium",
        "stacked_bar",
        "time_series_area",
        "time_series_overlaid_line",
        "time_series_calendar_heatmap",
        "text_only_fallback",
    }
)
# The rest — ``xy_heatmap``, ``data_table_fallback``, ``no_chart_match`` —
# are "low": we fell back or the pick is dense/marginal.


def _confidence_for(kind: DecisionReasonKind) -> Confidence:
    if kind in _HIGH_CONFIDENCE_KINDS:
        return "high"
    if kind in _MEDIUM_CONFIDENCE_KINDS:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Candidates — a constrained menu a consumer's LLM can pick from without
# being able to select an off-list or client-unsupported chart.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredChart:
    chart_type: VizChartType
    confidence: Confidence
    reason: DecisionReason


# Each chart type's natural degrade path, tried in order after the chosen
# chart and its recorded ``alternatives``. ``data_table`` / ``text_only``
# are appended unconditionally as the universal last resorts, so they
# don't need to be repeated here.
_RUNNER_UPS: dict[VizChartType, list[VizChartType]] = {
    "pie_chart": ["bar_chart"],
    "bar_chart": [],
    "line_chart": ["area_chart"],
    "area_chart": ["line_chart"],
    "stacked_bar_chart": ["bar_chart"],
    "scatter_chart": [],
    "bubble_chart": ["scatter_chart"],
    "histogram": ["bar_chart"],
    "calendar_heatmap": ["line_chart"],
    "xy_heatmap": ["stacked_bar_chart"],
    "compare_line_chart": ["line_chart"],
    "data_table": [],
    "text_only": [],
}


def _build_candidates(
    chart_type: VizChartType,
    reason: DecisionReason,
    confidence: Confidence,
    supported: frozenset[VizChartType] | None,
) -> list[ScoredChart]:
    """Build the ranked, deduped, capability-filtered candidate menu.

    Order: ``[chosen] + reason.alternatives + _RUNNER_UPS[chosen] +
    ["data_table", "text_only"]``, de-duped preserving first occurrence,
    then filtered by ``supported`` (``None``/empty imposes no filter).
    The chosen entry keeps its real confidence + reason; every runner-up
    is ``low`` confidence with a generic reason.
    """
    raw: list[VizChartType] = [
        chart_type,
        *reason.alternatives,
        *_RUNNER_UPS.get(chart_type, []),
        "data_table",
        "text_only",
    ]
    seen: set[VizChartType] = set()
    ordered: list[VizChartType] = []
    for c in raw:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if supported:
        ordered = [c for c in ordered if c in supported]

    out: list[ScoredChart] = []
    for c in ordered:
        if c == chart_type:
            out.append(ScoredChart(chart_type=c, confidence=confidence, reason=reason))
        else:
            out.append(
                ScoredChart(
                    chart_type=c,
                    confidence="low",
                    reason=DecisionReason(
                        kind="no_chart_match",
                        note=f"runner-up candidate; {chart_type} was the chosen pick",
                    ),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Feature bundle — the structural "why", for a consumer building its own
# LLM prompt without re-deriving shape from the compiled query.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VizFeatures:
    n_rows: int
    n_measures: int
    n_dimensions: int
    n_categorical_dims: int
    has_time_breakdown: bool
    measures_additive: bool | None
    measures_share_unit: bool | None
    is_flat: bool | None
    null_rate: float | None
    caveats: list[str]


# ---------------------------------------------------------------------------
# Render hints — axis sort order, scale, and a rendering guardrail
# ("top_n") for the chosen chart. Hints only; the library never
# transforms rows itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderHints:
    y_scale: Literal["linear", "log"] = "linear"
    sort: Literal["value_desc", "value_asc", "natural"] | None = None
    top_n: int | None = None


# ---------------------------------------------------------------------------
# Output value
# ---------------------------------------------------------------------------


@dataclass
class VizDecision:
    chart_type: VizChartType
    title: str
    x_axis: str | None
    y_axes: list[str]
    columns: list[VizColumn]
    reason: DecisionReason = field(
        default_factory=lambda: DecisionReason(kind="no_chart_match", note="")
    )
    # The breakdown/series dimension for a ``stacked_bar_chart`` (the second
    # dimension whose values become the stacks). ``None`` for every other
    # chart type.
    series: str | None = None
    # How likely a human/LLM with the question would pick something
    # different from ``chart_type`` — escalation value, not correctness.
    confidence: Confidence = "medium"
    # Ranked, deduped, ``supported_charts``-filtered menu a consumer's own
    # LLM can pick from. ``chosen`` first; see ``_build_candidates``.
    candidates: list[ScoredChart] = field(default_factory=lambda: list[ScoredChart]())
    # Structural "why" — everything needed to build an LLM prompt without
    # re-deriving shape from the compiled query.
    features: VizFeatures = field(
        default_factory=lambda: VizFeatures(
            n_rows=0,
            n_measures=0,
            n_dimensions=0,
            n_categorical_dims=0,
            has_time_breakdown=False,
            measures_additive=None,
            measures_share_unit=None,
            is_flat=None,
            null_rate=None,
            caveats=[],
        )
    )
    # Axis / render hints for the chosen chart (sort order, y-scale,
    # a rendering guardrail). Hints only — the library never transforms
    # rows itself.
    hints: RenderHints = field(default_factory=RenderHints)
    # ``bubble_chart`` only: the measure driving bubble size (m2). ``None``
    # for every other chart type.
    size_axis: str | None = None


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
# Chart-type decision — pure on the spec (and optionally shape_stats)
# ---------------------------------------------------------------------------


def _reason(
    kind: DecisionReasonKind, note: str, alternatives: list[VizChartType] | None = None
) -> DecisionReason:
    return DecisionReason(kind=kind, note=note, alternatives=list(alternatives or []))


def _stackable(measures: list[VizColumn]) -> bool:
    """Whether several measures can share a *stacked* area/bar.

    Stacking asserts the stack height is a meaningful sum of its parts.
    That only holds when the measures are **additive** and share a
    **common unit**:

    - A non-additive measure (a percentage / ratio / average) is not a
      sum — stacking two conversion rates yields a number that means
      nothing. ``VizColumn.additive`` (propagated from
      ``ColumnMeta.additive``, authoritative from ``agg in {"sum",
      "count"} and not non_additive``) is the ground truth when the
      compiler has computed it. When it's ``None`` (a ``VizColumn``
      built without going through the compiler), fall back to the
      ``format == "percent"`` heuristic.
    - Measures with different units (currency vs. count) can't be added
      onto the same axis at all.

    When either test fails, the caller overlays lines (optionally
    dual-axis) instead of stacking.
    """
    if len(measures) < 2:
        return True
    for m in measures:
        if m.additive is not None:
            if not m.additive:
                return False
        elif m.format == "percent":
            return False
    units = {(m.unit or m.display_unit) for m in measures}
    return len(units) <= 1


def _measures_additive(measures: list[VizColumn]) -> bool | None:
    """``VizFeatures.measures_additive``: ``True`` iff every measure is
    additive, ``False`` iff at least one is known non-additive, ``None``
    when there are no measures or additivity is unknown for any of them."""
    if not measures:
        return None
    flags = [m.additive for m in measures]
    if any(f is False for f in flags):
        return False
    if any(f is None for f in flags):
        return None
    return True


def _measures_share_unit(measures: list[VizColumn]) -> bool | None:
    """``VizFeatures.measures_share_unit``: ``None`` when there's fewer
    than 2 measures (the question doesn't apply)."""
    if len(measures) < 2:
        return None
    units = {(m.unit or m.display_unit) for m in measures}
    return len(units) <= 1


def _build_hints(
    chart_type: VizChartType,
    x_axis_col: VizColumn | None,
    shape_stats: ShapeStats | None,
) -> RenderHints:
    """Axis sort order, y-scale, and a rendering-guardrail top_n hint.

    Hints only — the library never transforms rows itself; a renderer
    reads these to decide draw order / scale / "bucket the rest into
    Other", but the row data it receives is untouched.
    """
    sort: Literal["value_desc", "value_asc", "natural"] | None = None
    if chart_type in ("bar_chart", "pie_chart", "stacked_bar_chart"):
        sort = "natural" if (x_axis_col is not None and x_axis_col.ordinal) else "value_desc"
    elif chart_type == "histogram":
        sort = "natural"

    y_scale: Literal["linear", "log"] = "linear"
    if (
        shape_stats is not None
        and shape_stats.measure_min is not None
        and shape_stats.measure_max is not None
        and shape_stats.measure_min > 0
        and shape_stats.measure_max / shape_stats.measure_min >= LOG_SCALE_RATIO
    ):
        y_scale = "log"

    top_n: int | None = None
    if chart_type == "pie_chart":
        top_n = PIE_MAX_SLICES
    elif chart_type == "bar_chart":
        top_n = BAR_MAX_BARS

    return RenderHints(y_scale=y_scale, sort=sort, top_n=top_n)


def _pick_chart_type(
    query: SemanticQuery,
    touched_cubes: list[Cube],
    n_rows: int,
    columns: list[VizColumn],
    shape_stats: ShapeStats | None = None,
) -> tuple[VizChartType, DecisionReason]:
    overrides: set[ChartTypeLiteral] = {
        c.default_chart_type for c in touched_cubes if c.default_chart_type is not None
    }
    if len(overrides) == 1:
        chart = next(iter(overrides))
        return chart, _reason("cube_override", f"cube default_chart_type={chart}")

    if query.ungrouped:
        return "data_table", _reason("ungrouped_row", "ungrouped row listing")

    n_measures = len(query.measures)
    has_time_breakdown = (
        query.time_dimension is not None and query.time_dimension.granularity is not None
    )
    n_dims = len(query.dimensions) + (1 if has_time_breakdown else 0)
    # Categorical (non-measure, non-time) axis columns — the dimensions a
    # chart breaks down by. Used to tell a numeric distribution (histogram)
    # from a categorical breakdown (bar / stacked bar).
    cat_dims = [c for c in columns if not c.is_measure and not c.is_time]

    # A *categorical* breakdown with zero variation (every category has the
    # same value) has nothing to compare — collapse a would-be bar/pie to a
    # plain text answer. A flat *time* series is deliberately excluded: a
    # constant line over time legitimately communicates "this held steady",
    # so it stays a line rather than collapsing.
    if shape_stats is not None and shape_stats.is_flat and not has_time_breakdown:
        return "text_only", _reason(
            "flat_series", "categorical breakdown has zero variation (measure_min == measure_max)"
        )

    # Compare query. The compiler emits per-measure ``<m>_current`` /
    # ``<m>_prior`` / ``<m>_delta`` / ``<m>_pct_change`` columns;
    # reading the cardinality-only branch the visualiser would call
    # those "more measures" and pick a stacked area (the multi-measure
    # time series case), which is almost never right — a compare is a
    # side-by-side / delta view, not a composition-over-time. Branch
    # here, before the generic time-series logic.
    if query.compare is not None:
        if not has_time_breakdown:
            return "data_table", _reason(
                "compare_current_prior",
                "compare query without a time dimension has no natural chart",
            )
        return (
            "compare_line_chart",
            _reason(
                "compare_current_prior",
                "compare query: current / prior / delta / pct_change side-by-side",
            ),
        )

    if n_measures >= 1 and n_dims == 0:
        return "text_only", _reason("single_value", "single-value answer")

    # Time series. A long *daily* single-measure series is the GitHub-style
    # calendar heatmap; several measures compose as a stacked area; anything
    # else is a line.
    if has_time_breakdown:
        granularity = query.time_dimension.granularity if query.time_dimension else None
        # A categorical breakdown *alongside* time is one line per category
        # (multi-series line). This takes precedence over the calendar and
        # area special cases below, which assume the only non-time axis is
        # the measure — without this branch the breakdown dimension is
        # silently dropped from the decision.
        if n_measures == 1 and len(cat_dims) >= 1:
            return (
                "line_chart",
                _reason(
                    "time_series_multi_series",
                    f"time series broken down by '{cat_dims[0].name}' → one line per category",
                ),
            )
        # Shape-stats override: a sparse daily series shouldn't be a
        # calendar heatmap. (A flat *time* series stays a line — see the
        # `is_flat` handling below, which only collapses a flat
        # *categorical* breakdown to `text_only`.)
        if (
            n_measures == 1
            and granularity == "day"
            and n_rows > CALENDAR_MIN_DAYS
            and not (shape_stats is not None and shape_stats.is_sparse)
        ):
            return (
                "calendar_heatmap",
                _reason(
                    "time_series_calendar_heatmap",
                    f"daily series, n_rows={n_rows} > {CALENDAR_MIN_DAYS}",
                ),
            )
        if n_measures >= 2:
            # Stack (area) only when the measures are additive and share a
            # unit; otherwise summing them is meaningless, so overlay lines.
            measure_cols = [c for c in columns if c.is_measure]
            if _stackable(measure_cols):
                return (
                    "area_chart",
                    _reason(
                        "time_series_area",
                        f"time series, {n_measures} additive same-unit measures "
                        "(stacked composition)",
                    ),
                )
            return (
                "line_chart",
                _reason(
                    "time_series_overlaid_line",
                    f"time series, {n_measures} measures with diverging units or a "
                    "non-additive ratio → overlaid lines, not a stacked area",
                    alternatives=["area_chart"],
                ),
            )
        return "line_chart", _reason("time_series_line", "time series with granularity")

    # Three measures with one labelling dimension → bubble chart: x=m0,
    # y=m1, size=m2, the dimension labels each point. Checked before the
    # 2-measure scatter branch below (which is otherwise unchanged).
    if n_measures == 3 and n_dims == 1:
        return (
            "bubble_chart",
            _reason("bubble_xyz", "3 measures plotted as x/y/size, dimension labels points"),
        )

    # Two measures with one labelling dimension → XY scatter.
    if n_measures == 2 and n_dims == 1:
        return (
            "scatter_chart",
            _reason("scatter_xy", "2 measures plotted against each other"),
        )

    # One measure over a single *numeric* dimension → frequency distribution.
    # No upper row cap: a distribution over *many* numeric buckets is exactly
    # when a histogram beats every alternative — capping it would demote the
    # densest, most histogram-worthy data to a bar and then a table.
    if (
        n_dims == 1
        and n_measures == 1
        and len(cat_dims) == 1
        and cat_dims[0].storage_type in _NUMERIC_STORAGE
    ):
        return (
            "histogram",
            _reason("histogram_distribution", "1 measure over a numeric dimension (distribution)"),
        )

    # Shape-stats overrides for the small-breakdown branches. Each
    # override records the rejected natural pick in ``alternatives`` so
    # an audit surface can answer "why not a pie?".
    if n_dims == 1 and n_measures == 1 and n_rows <= PIE_MAX_SLICES:
        if shape_stats is not None and shape_stats.has_negatives:
            return (
                "bar_chart",
                _reason(
                    "shape_stats_fallback",
                    f"1 dim, 1 measure, n_rows={n_rows} <= {PIE_MAX_SLICES}; "
                    "negatives present → pie would be misleading",
                    alternatives=["pie_chart"],
                ),
            )
        if shape_stats is not None and shape_stats.n_distinct_categories == 1:
            return (
                "text_only",
                _reason(
                    "shape_stats_fallback",
                    "1 dim, 1 measure; n_distinct_categories=1 → no breakdown to show",
                    alternatives=["pie_chart", "bar_chart", "data_table"],
                ),
            )
        return (
            "pie_chart",
            _reason(
                "pie_small",
                f"1 dim, 1 measure, n_rows={n_rows} <= {PIE_MAX_SLICES}",
            ),
        )

    if n_dims == 1 and n_rows <= BAR_MAX_BARS:
        return (
            "bar_chart",
            _reason("bar_medium", f"1 dim, n_rows={n_rows} <= {BAR_MAX_BARS}"),
        )

    # Two categorical dimensions + one measure. A small grid stacks cleanly
    # (primary axis + breakdown); a larger one is an xy heatmap (the measure
    # is the cell colour); past the cell cap it's a table.
    if n_dims == 2 and n_measures == 1 and len(cat_dims) == 2:
        if n_rows <= STACKED_BAR_MAX_CELLS:
            return (
                "stacked_bar_chart",
                _reason(
                    "stacked_bar",
                    f"2 dims, 1 measure, n_rows={n_rows} (axis + breakdown)",
                ),
            )
        if n_rows <= HEATMAP_MAX_CELLS:
            return (
                "xy_heatmap",
                _reason(
                    "xy_heatmap",
                    f"2 dims, 1 measure, n_rows={n_rows} (matrix, measure=colour)",
                ),
            )

    return (
        "data_table",
        _reason(
            "data_table_fallback",
            f"multi-dim or n_rows={n_rows} too large for a chart",
        ),
    )


def _apply_supported(
    chart_type: VizChartType,
    reason: DecisionReason,
    supported: frozenset[VizChartType] | None,
) -> tuple[VizChartType, DecisionReason]:
    """Constrain the natural choice to the client's declared capabilities.

    ``supported`` is the set of chart types the caller's renderer can draw.
    ``None`` (or empty) means no constraint. When the picked chart isn't
    supported, fall back to the most universal supported option
    (``data_table`` then ``text_only``), else the first supported type."""
    if not supported or chart_type in supported:
        return chart_type, reason
    for fallback in ("data_table", "text_only"):
        if fallback in supported:
            new_kind: DecisionReasonKind = "client_capability_fallback"
            return (
                fallback,
                DecisionReason(
                    kind=new_kind,
                    note=f"{reason.note}; {chart_type} unsupported by client → {fallback}",
                    alternatives=reason.alternatives + [chart_type],
                ),
            )
    chosen = sorted(supported)[0]
    return (
        chosen,
        DecisionReason(
            kind="client_capability_fallback",
            note=f"{reason.note}; {chart_type} unsupported by client → {chosen}",
            alternatives=reason.alternatives + [chart_type],
        ),
    )


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
        additive=meta.additive,
        ordinal=meta.ordinal,
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
    shape_stats: ShapeStats | None = None,
) -> VizDecision:
    """Return chart_type + axis labels + per-column formats for a query.

    Reads from ``compiled.column_meta`` and ``compiled.touched_cube_names``
    — no catalog re-resolution. ``catalog`` is consulted only to look
    up ``Cube.default_chart_type`` for the touched cubes.

    ``query`` carries shape facts the compiler doesn't surface on
    ``CompiledQuery`` (``ungrouped`` flag, time-dimension granularity,
    ``compare`` flag). ``n_rows`` is the actual row count; pass ``0``
    for dry-run / explain paths.

    ``supported_charts`` lets the *calling renderer* declare which chart
    types it can draw. Chart support is a property of the rendering client,
    not the data model, so it's a call-time argument rather than a catalog
    field. When the naturally-best chart isn't in the set, the decision
    falls back to a supported type (``data_table`` / ``text_only`` first).
    ``None`` imposes no constraint.

    ``shape_stats`` is the *post-execute* override hook: a caller that
    has run the SQL can pass basic distribution facts
    (``has_negatives``, ``n_distinct_categories``, ``is_sparse``, etc.)
    to override the cardinality-only decisions — negatives block a pie
    chart, a single distinct category falls to ``text_only``, a sparse
    daily series skips the calendar heatmap. ``None`` (the default)
    keeps the cardinality-only decision intact. The visualizer never
    samples rows itself; the contract is "caller computes, visualizer
    respects."
    """
    touched_cubes = [catalog[name] for name in compiled.touched_cube_names if name in catalog]
    ordered: list[VizColumn] = [_viz_column(m) for m in compiled.column_meta]
    chart_type, reason = _pick_chart_type(
        query, touched_cubes, n_rows, ordered, shape_stats=shape_stats
    )
    chart_type, reason = _apply_supported(chart_type, reason, supported_charts)

    x_axis: str | None = None
    y_axes: list[str] = []
    series: str | None = None
    size_axis: str | None = None
    non_measures = [c for c in ordered if not c.is_measure]
    measures_out = [c for c in ordered if c.is_measure]
    x_axis_col: VizColumn | None = non_measures[0] if non_measures else None
    if chart_type in (
        "bar_chart",
        "line_chart",
        "area_chart",
        "histogram",
        "calendar_heatmap",
        "compare_line_chart",
    ):
        # calendar_heatmap is a time series too: x = the time/day column,
        # y = the measure that colours each day cell. compare_line_chart
        # is the side-by-side current/prior/delta/pct_change view; x is
        # the time bucket, y is the measure(s) — same shape as line/area.
        # Prefer the time column for the x axis (a time series' natural
        # abscissa); fall back to the first non-measure otherwise.
        time_cols = [c for c in non_measures if c.is_time]
        cat_cols = [c for c in non_measures if not c.is_time]
        if time_cols:
            x_axis = time_cols[0].display_name
        elif non_measures:
            x_axis = non_measures[0].display_name
        y_axes = [c.display_name for c in measures_out]
        # A line over time with a leftover categorical dimension is a
        # multi-series line: the category becomes the series (one line each).
        if chart_type == "line_chart" and time_cols and cat_cols:
            series = cat_cols[0].display_name
    elif chart_type in ("stacked_bar_chart", "xy_heatmap"):
        # First dimension is the primary (x) axis; the second is the stack
        # series (stacked_bar) or the row axis (xy_heatmap). The measure is
        # the stacked/coloured value.
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
    elif chart_type == "bubble_chart":
        # x=m0, y=m1, size=m2; the dimension labels each point (same
        # non-axis role as the 2-measure scatter case above).
        if len(measures_out) >= 3:
            x_axis = measures_out[0].display_name
            y_axes = [measures_out[1].display_name]
            size_axis = measures_out[2].display_name
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

    confidence = _confidence_for(reason.kind)
    candidates = _build_candidates(chart_type, reason, confidence, supported_charts)
    hints = _build_hints(chart_type, x_axis_col, shape_stats)

    n_dims_total = len(non_measures)
    n_cat_dims = len([c for c in non_measures if not c.is_time])
    has_time_breakdown = any(c.is_time for c in non_measures)
    is_flat = shape_stats.is_flat if shape_stats is not None else None
    null_rate = shape_stats.null_rate if shape_stats is not None else None
    caveats: list[str] = []
    if is_flat:
        caveats.append("series has zero variation (measure_min == measure_max)")
    if null_rate is not None and null_rate >= NULL_RATE_CAVEAT_THRESHOLD:
        caveats.append(f"null_rate={null_rate:.2f} is at or above the caveat threshold")
    features = VizFeatures(
        n_rows=n_rows,
        n_measures=len(measures_out),
        n_dimensions=n_dims_total,
        n_categorical_dims=n_cat_dims,
        has_time_breakdown=has_time_breakdown,
        measures_additive=_measures_additive(measures_out),
        measures_share_unit=_measures_share_unit(measures_out),
        is_flat=is_flat,
        null_rate=null_rate,
        caveats=caveats,
    )

    return VizDecision(
        chart_type=chart_type,
        title=title,
        x_axis=x_axis,
        y_axes=y_axes,
        columns=ordered,
        reason=reason,
        series=series,
        confidence=confidence,
        candidates=candidates,
        features=features,
        hints=hints,
        size_axis=size_axis,
    )


__all__ = [
    "BAR_MAX_BARS",
    "CALENDAR_MIN_DAYS",
    "Confidence",
    "DecisionReason",
    "DecisionReasonKind",
    "HEATMAP_MAX_CELLS",
    "LOG_SCALE_RATIO",
    "NULL_RATE_CAVEAT_THRESHOLD",
    "PIE_MAX_SLICES",
    "RenderHints",
    "STACKED_BAR_MAX_CELLS",
    "ScoredChart",
    "ShapeStats",
    "VizChartType",
    "VizColumn",
    "VizDecision",
    "VizFeatures",
    "decide_visualization",
]
