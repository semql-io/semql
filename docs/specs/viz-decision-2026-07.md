# Visualization decision — escalation-ready output (2026-07)

## Context

`semql.visualize.decide_visualization` is a **pure, sans-I/O** function: it
maps a `CompiledQuery` + shape facts to a `VizDecision`. Tessera never calls
an LLM and never touches rows itself; a flat series or high null-rate override
comes from a caller-supplied `ShapeStats`. See memory `tessera-no-llm-calls`.

The "let an LLM decide" story is therefore an **output-contract** problem, not
a control-flow one. We enrich the decision so a library consumer can (a) tell
when the deterministic pick is shaky, and (b) hand *their own* LLM a safe,
typed, constrained menu — with the natural-language question, which the
library never sees.

This spec covers the batch after the four correctness fixes
(multi-series line, non-additive/diverging-unit stacking, histogram cap).

## 1. Confidence — coarse enum

```python
Confidence = Literal["high", "medium", "low"]
```

`VizDecision.confidence: Confidence`. Semantics: **how likely a human/LLM with
the question would pick something different** (escalation value), NOT "are we
right". Mapped from the winning `DecisionReason.kind`:

- **high** — shape is unambiguous, escalation won't help:
  `cube_override`, `ungrouped_row`, `single_value`, `compare_current_prior`,
  `time_series_line`, `time_series_multi_series`, `scatter_xy`,
  `histogram_distribution`, `bubble_xyz`, `flat_series`,
  `shape_stats_fallback` (data-driven), `client_capability_fallback`
  (a hard constraint — the renderer can't draw anything else anyway).
- **medium** — a sound default the question could override:
  `pie_small`, `bar_medium`, `stacked_bar`, `time_series_area`,
  `time_series_overlaid_line`, `time_series_calendar_heatmap`,
  `text_only_fallback`.
- **low** — we fell back or the pick is dense/marginal:
  `xy_heatmap`, `data_table_fallback`, `no_chart_match`.

## 2. Candidates — a constrained menu (added alongside `alternatives`)

`reason.alternatives` keeps its current meaning (rejected/considered, for
"why not X" audit). We ADD a separate ranked menu:

```python
@dataclass(frozen=True)
class ScoredChart:
    chart_type: VizChartType
    confidence: Confidence
    reason: DecisionReason

VizDecision.candidates: list[ScoredChart]  # chosen first
```

Built from: `[chosen] + reason.alternatives + _RUNNER_UPS[chosen] +
["data_table", "text_only"]`, de-duped preserving order, filtered by
`supported_charts`. The chosen candidate keeps its confidence + reason; the
rest are `low` with a generic "runner-up" reason. `_RUNNER_UPS` maps each
chart type to its natural degrade path (pie→bar→table, line→area→table, …).
This is the enum a consumer's LLM picks from — it cannot select an off-list
or unsupported chart.

## 3. Feature bundle — the structural "why"

```python
@dataclass(frozen=True)
class VizFeatures:
    n_rows: int
    n_measures: int
    n_dimensions: int          # includes the time breakdown
    n_categorical_dims: int
    has_time_breakdown: bool
    measures_additive: bool | None
    measures_share_unit: bool | None
    is_flat: bool | None
    null_rate: float | None
    caveats: list[str]

VizDecision.features: VizFeatures
```

Gives the consumer everything needed to build an LLM prompt without
re-deriving shape. `caveats` are human-readable (null-rate, flatness).

## 4. Wire up `is_flat` and `null_rate`

- **`is_flat`** (`ShapeStats.measure_min == measure_max`): a *categorical*
  breakdown with zero variation has nothing to compare — when `is_flat is
  True` and there's no time breakdown, return `text_only` (kind
  `flat_series`, confidence high) and add a caveat. A flat *time* series is
  deliberately excluded: a constant line over time legitimately shows "this
  held steady", so it stays a line; the flatness still rides out as an
  advisory caveat on the feature bundle.
- **`null_rate`**: does NOT change the chart. When `null_rate >=
  NULL_RATE_CAVEAT_THRESHOLD` (0.2), append a caveat to `features.caveats`.
  Removes the stale `MaskedFallback` reference in the docstring (never a valid
  kind).

## 5. Inputs we could derive but don't

- **Additivity (#10)** — authoritative from the model: a measure is additive
  iff `agg in {"sum","count"}` and `not non_additive`. Surface it:
  `ColumnMeta.additive: bool | None` (measures only; computed in
  `_build_column_meta`), propagated to `VizColumn.additive`. `_stackable`
  uses `additive` when present, falling back to the `format=="percent"`
  heuristic. This makes the stacking fix authoritative rather than a guess.
- **Ordinal dimensions (#11)** — new `Dimension.ordinal: bool = False`
  (weekday, month-name, rating bucket: categorical but ordered). Propagated to
  `ColumnMeta.ordinal` → `VizColumn.ordinal`. Drives the sort hint (§6): an
  ordinal x-axis sorts by natural order, not by measure value.
- **Bubble chart (#12)** — viz-only chart type (`bubble_chart`, like
  `compare_line_chart`; NOT added to the model `ChartTypeLiteral`). 3 measures
  + 1 dim → x=m0, y=m1, size=m2, dim labels points. Adds
  `VizDecision.size_axis: str | None`. Reason kind `bubble_xyz`. The existing
  2-measure scatter branch is unchanged; 3 measures is the new branch above it.

## 6. Axis / render hints

```python
@dataclass(frozen=True)
class RenderHints:
    y_scale: Literal["linear", "log"] = "linear"
    sort: Literal["value_desc", "value_asc", "natural"] | None = None
    top_n: int | None = None

VizDecision.hints: RenderHints
```

- **sort**: `bar_chart` / `pie_chart` / `stacked_bar_chart` → `value_desc`,
  UNLESS the x-axis dimension is `ordinal` → `natural`. `histogram` →
  `natural` (numeric buckets). Others → `None`.
- **y_scale**: `log` when `ShapeStats` shows strong positive skew —
  `measure_min > 0` and `measure_max / measure_min >= LOG_SCALE_RATIO` (1000);
  else `linear`. Requires stats; without them, `linear`.
- **top_n**: `pie_chart` → `PIE_MAX_SLICES`, `bar_chart` → `BAR_MAX_BARS`
  (a rendering guardrail: bucket the remainder into "Other"). The library
  does NOT transform rows — this is a hint only.

## Back-compat & surface

- All new `VizDecision` fields have defaults → existing constructors keep
  working. Beta surface (viz is explicitly BETA), so additive changes are fine.
- New exports: `Confidence`, `ScoredChart`, `RenderHints`, `VizFeatures` from
  `semql` and `semql.visualize` `__all__`; update `test_api_surface` /
  `test_package_surface`.
- `semql_mcp.viz._viz_to_payload` gains the new fields (VizColumn extras ride
  through `asdict`; the manually-built payload dict needs the new keys added).

## Testing (Red/Green, per workstream)

Each of §1–§6 lands as its own atomic commit with tests first: confidence
mapping per kind; candidates menu (dedupe, supported filter, chosen-first);
feature bundle values; flat→text_only + null caveat; additive from
agg/non_additive + stacking; ordinal sort; bubble; log-scale + top_n hints.
