---
name: semql-cube
description: >
  Author SemQL cubes, views, and the catalog wiring (auth, scope,
  prompt fragments). The technical counterpart to
  `semql-requirement-discovery`: where that skill captures intent,
  this one writes the Python. Use when the user says "model this
  table," "add a measure," "wire up team-scoped auth," or hands you
  a requirements doc to implement.
---

# Authoring SemQL cubes

This skill writes Python. If the user hasn't decided *what* they need
(entities, questions, auth concerns), point them at
`semql-requirement-discovery` first — that skill captures intent at
the domain level. This skill picks up at the technical translation:
chosen entity → `Cube`, identified scoping rule → `ScopeFn`,
question shape → `SemanticQuery`.

You may interview the user when a requirements doc doesn't cover a
technical detail (alias choice, SQL fragment shape, granularities,
ScopePredicate SQL). Ask in concrete technical terms: "What's the
join predicate's column on each side?" rather than "How do these
entities relate?"

## A cube is a logical table

It declares where rows live (`backend`, `table`), the always-on
membership predicate (`base_predicate`), the addressable fields
(*measures*, *dimensions*, *time dimensions*, *segments*), and the
*joins* to other cubes.

A **catalog** is a validated `list[Cube]` plus optional `views`,
`policy`, and `scope_fns`. It exposes `compile`, `prompt`,
`as_dict`, and acts as the iteration root for the introspection
primitives.

## Step 1: pick the backend

```python
from semql import Backend

Backend.POSTGRES     # %(name)s placeholder, native ILIKE, generate_series spine
Backend.CLICKHOUSE   # {name:Type} placeholder, toStartOf<Gran>, numbers() spine
Backend.DUCKDB       # $name placeholder, native ILIKE, generate_series spine
Backend.BIGQUERY     # @name placeholder, LOWER LIKE LOWER, UNNEST(GENERATE_DATE_ARRAY) spine
Backend.SNOWFLAKE    # :name placeholder, native ILIKE, TABLE(GENERATOR) + SEQ4 spine
Backend.META         # in-memory VALUES; reflection only — see META_CUBES
```

Cross-backend queries are rejected at compile time. If one query
needs columns from two backends, split it.

## Step 2: declare the cube

```python
from semql import Cube, Dimension, Measure, TimeDimension, Backend

orders = Cube(
    name="orders",
    backend=Backend.POSTGRES,
    table="public.orders",            # may contain {schema} / {tenant_schema}
    alias="o",                        # what the FROM clause uses
    base_predicate="{o}.deleted_at IS NULL",
    description="One row per checkout, paid or not.",
    primary_key="id",                 # the row-identifying dim
    measures=[
        Measure(name="count", sql="*", agg="count", unit="count"),
        Measure(
            name="revenue", sql="{o}.amount", agg="sum",
            unit="currency", format="currency",
            description="Total order revenue.",
        ),
    ],
    dimensions=[
        Dimension(name="id", sql="{o}.id", type="number"),
        Dimension(name="region", sql="{o}.region", type="string"),
        # foreign_key auto-derives a many_to_one Join to the named cube
        # (which must declare primary_key).
        Dimension(name="customer_id", sql="{o}.customer_id",
                  type="number", foreign_key="customers"),
    ],
    time_dimensions=[
        TimeDimension(
            name="created_at", sql="{o}.created_at",
            granularities=("hour", "day", "week", "month"),
        ),
    ],
)
```

### The `{alias}` placeholder

Every `sql` fragment uses `{<alias>}` (or `{<cube_name>}` — both
resolve to the alias). Other known placeholders:

- `{schema}` / `{tenant_schema}` — caller-supplied via `context=` on
  `Catalog.compile`.
- `{ctx.X}` (inside `security_sql` and `ScopePredicate.sql`) — binds
  as a parameter, never inlined as a literal.

## Step 3: measure aggregations

| `Measure.agg`     | Use when                                        |
|-------------------|-------------------------------------------------|
| `sum`             | adds (revenue, count of events)                 |
| `count`           | row count; `sql="*"` is fine                    |
| `count_distinct`  | unique cardinality — pair with `non_additive=True` |
| `avg` / `min` / `max` | obvious                                     |
| `ratio`           | composed from two other measures on the same cube |

### Non-additive flag

```python
Measure(name="unique_users", sql="{o}.user_id", agg="count_distinct",
        non_additive=True)
```

Flags measures that can't be summed across a coarser grain (counting
distinct users across days ≠ summing daily distinct counts). Surfaced
in the prompt; the rollup work refuses to roll them up.

### Filtered measures

```python
Measure(
    name="paid_revenue", sql="{o}.amount", agg="sum",
    filter="{o}.status = 'paid'",        # SUM(amount) FILTER (WHERE ...)
)
```

Renders as `FILTER (WHERE ...)` on PG / CH / DuckDB / BigQuery, and
transpiles to `SUM(IFF(..., amount, NULL))` on Snowflake.

### Ratio measures

```python
Measure(name="paid_rate", agg="ratio",
        numerator="paid_revenue", denominator="revenue",
        sql="")  # ignored; ratio composes from numerator/denominator
```

Compiler emits `paid_revenue / NULLIF(revenue, 0)`. Composes with
filtered measures — a filtered ratio is just a ratio of two filtered
sums.

## Step 4: dimensions

| `Dimension.type` | Behaviour                                   |
|------------------|---------------------------------------------|
| `string`         | default; filter values must be strings      |
| `number`         | numeric comparisons (`gt`/`lt`/…)           |
| `time`           | filter values must parse as ISO-8601        |
| `bool`           | filter values must be Python `bool`         |
| `uuid`           | filter values must parse as UUIDs           |

Presentation hints (`unit`, `format`) mirror Measure's — visualisers
read them; the compiler ignores them.

```python
Dimension(name="duration_seconds", sql="{o}.duration_sec",
          type="number", unit="seconds", format="duration")
```

## Step 5: presentation hints (closed enums)

Visualisation hints live on three slots:

- `Cube.default_chart_type` — the cube's preferred default.
- `Measure.format` and `Dimension.format` — per-field rendering hint.
- `Measure.unit` and `Dimension.unit` — free-form label that pairs
  with `format` (`unit="USD"` with `format="currency"`).

### Supported chart types — use these names exactly

| `ChartTypeLiteral` | When                                       |
|--------------------|--------------------------------------------|
| `line_chart`       | time series with `granularity`             |
| `bar_chart`        | categorical comparison                     |
| `pie_chart`        | share of whole — small N only              |
| `data_table`       | fallback, row listings, multi-measure detail |

`semql.visualize.decide_visualization` picks the chart type
automatically from the query shape + row count. `default_chart_type`
is the override the cube author baked in. Leave it `None` unless
the cube's natural shape genuinely differs from the auto-decision
(META cubes pin `data_table`, for example).

### Supported formats — use these names exactly

| `FormatLiteral` | Use for                                    |
|-----------------|--------------------------------------------|
| `currency`      | money (`unit="USD"` etc.)                  |
| `percent`       | already a rate (`0.12` = 12%)              |
| `integer`       | counts, IDs                                |
| `duration`      | seconds / minutes / hours                  |
| `raw`           | fallback — equivalent to leaving it `None` |

### Don't invent viz

If the requirements doc says "forecast," "pivot," "sparkline," or
"heatmap" — flag it. SemQL doesn't render those. Pick a supported
substitute (`line_chart` for forecast-as-trendline,
`data_table` for pivot, etc.) and note the trade-off in the cube's
`description`. **Never add a chart-type name outside `ChartTypeLiteral`.**
The Pydantic validator rejects it; users hit a 500 if you ship that.

### Only override when defaults are wrong

`format=None` is correct for most fields. Set it when:
- A numeric measure's name doesn't reveal its unit
  (`avg_processing_time_seconds` → `format=duration`).
- A rate-style measure stored as a fraction needs the percent
  rendering (`conversion_rate` → `format=percent`).
- Currency / multi-currency catalogs need the explicit hint.

Setting it for every field bloats the prompt and gives the LLM
duplicate context. Skip when the name + dim type already say it.

## Step 6: joins

A `Join` is a directed edge. The BFS finds a path; multiple edges
compose.

```python
from semql import Join

orders = Cube(
    ...,
    joins=[
        Join(to="customers", relationship="many_to_one",
             on="{o}.customer_id = {customers}.id"),
    ],
)
```

Prefer `Dimension.foreign_key="<other_cube>"` on the FK column — the
Catalog auto-derives the `many_to_one` Join from the FK to the
target's `primary_key`. Explicit `Join` with the same target wins.

## Step 7: reusable predicates and required filters

```python
from semql import Segment

orders = Cube(
    ...,
    segments=[
        Segment(name="paid", sql="{o}.status = 'paid'",
                description="Confirmed payment received."),
    ],
    required_filters=["region"],  # query MUST filter on this dim
)
```

A query then references `segments=["orders.paid"]` to AND-compose the
predicate without re-deriving it.

## Step 8: drill paths + extends

```python
orders = Cube(
    ...,
    drill_paths=[["region", "city"]],   # UI drill affordance hint
    extends="base_audited",              # inherit measures/dims by name
)
```

`drill_paths` is metadata for downstream UIs (the compiler ignores
it; the drilldown prompt fragment reads it). `extends` flattens the
named parent's measures / dimensions / time_dimensions / segments
into this cube — child redeclarations win, new items append. Cycles
raise at Catalog construction.

## Step 9: views — curated facades

```python
from semql import View

revenue_view = View(
    name="revenue_overview",
    description="Curated revenue facade for line-of-business questions.",
    fields={
        "revenue": "orders.revenue",
        "region": "orders.region",
        "created_at": "orders.created_at",
    },
)
```

Views expose a renamed subset of cube fields. Used for prompt
trimming (a 30-cube catalog can expose half a dozen 5-field views
for common question shapes) and join disambiguation. Reference
fields as `view.local_name`; the compiler rewrites to the underlying
cube field and preserves the view-local alias in the output column.

## Step 10: tenancy + RLS + scope

Three predicates can wrap a cube's FROM source, all AND-composed
*inside* the alias subquery (so an outer `OR` can't bypass any of
them):

### Tenancy

```python
Cube(
    ...,
    tenancy="discriminator",     # schema | discriminator | none
    tenancy_column="tenant_id",  # required for discriminator
)
```

`schema` substitutes `{tenant_schema}` in `table`. `discriminator`
wraps the source in `WHERE tenant_col = bind(tenant)` — pass
`context={"tenant": ...}` at compile time. `none` skips the check
(META cubes, public lookups).

### `security_sql` — caller-attached RLS

```python
Cube(
    ...,
    security_sql="{o}.owner_id = {ctx.viewer_id}",
)
```

`{ctx.X}` placeholders bind as parameters — never inlined as
literals. `viewer.viewer_id` auto-flattens to `ctx.viewer_id` when a
viewer is passed (see Step 12).

### `Cube.scope` — registered ScopeFn

For row-level rules that apply across multiple cubes (e.g. "every
table scoped to your team"), name a scope on each cube:

```python
Cube(name="orders", ..., scope="my_team")
Cube(name="tickets", ..., scope="my_team")
```

Then register the function on the Catalog (Step 11). The compiler
calls it with `(cube, viewer)` and AND-injects the returned
`ScopePredicate` inside the alias subquery — same protection as
tenancy / security_sql.

## Step 11: cube-level authorisation

```python
Cube(
    name="finance_ledger",
    ...,
    required_roles=["finance"],   # ANY-match; empty list = open
)
```

When the caller passes a `viewer`, cubes whose `required_roles` don't
intersect the viewer's roles disappear from `iter_cubes` /
prompt fragments / MCP tool registration, and the compiler refuses
queries that touch them (loud `CompileError`, not silent filtering).

## Step 12: assemble the Catalog

```python
from semql import AuthContext, Catalog, ScopePredicate

def my_team(cube: "Cube", viewer: AuthContext) -> "ScopePredicate | None":
    if "admin" in viewer.roles:
        return None              # admins see all rows
    return ScopePredicate(
        sql="{o}.rep_id IN (SELECT id FROM reps WHERE team = {ctx.viewer_team})",
        ctx_keys=["ctx.viewer_team"],
    )


def deny_pii(cube: "Cube", viewer: AuthContext) -> bool:
    # Policy override: cube-level visibility on top of required_roles.
    if "pii" in cube.metadata and "pii_certified" not in viewer.roles:
        return False
    return True


catalog = Catalog(
    cubes=[orders, customers, finance_ledger],
    views=[revenue_view],
    scope_fns={"my_team": my_team},
    policy=deny_pii,
)
```

`Catalog([...])` validates on construction: duplicate cube names,
unknown `Join.to` targets, missing `primary_key` on
`foreign_key` targets, unregistered `Cube.scope` names — all raise
loudly here so the compiler can trust the wiring later.

## Step 13: query (with viewer)

```python
from semql import AuthContext, Filter, SemanticQuery, TimeWindow

viewer = AuthContext(viewer_id="alice@example.com", roles=["sales"])

compiled = catalog.compile(
    SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-04-01"),
            fill_nulls_with=0,        # one row per month, COALESCE missing → 0
        ),
        filters=[Filter(dimension="orders.region", op="in", values=["us", "ca"])],
        segments=["orders.paid"],
        order=[("revenue", "desc")],
        limit=100,
    ),
    viewer=viewer,
    context={"ctx.viewer_team": "EMEA-East", "tenant": "acme"},
)
print(compiled.sql, compiled.params)
```

- `viewer=` filters unauthorised cubes (loud refusal) and auto-binds
  `ctx.viewer_id`.
- `context=` supplies `{schema}` / `{tenant}` / any `{ctx.X}` your
  `security_sql` or `ScopePredicate` declares.
- `fill_nulls_with` requires `granularity` and rejects queries with
  non-time dimensions (Phase A: time-series only).
- For OR / NOT, use `where=BoolExpr(op="or", children=[...])`.
- Compare windows: `compare=CompareWindow(mode="previous_period")`.

## Step 14: surface to LLMs — the four-role pipeline

Four fragment builders, each paired with a typed Pydantic output the
LLM emits. Splice the fragment into the role's system prompt; parse
the response with the matching model.

| Role | Fragment | Output |
|---|---|---|
| Router | `build_router_prompt_fragment(catalog, viewer=...)` | `RouterDecision { route_to, cubes, views }` |
| Generator | `build_query_generator_prompt_fragment(catalog, scope_to=decision.cubes+decision.views, viewer=...)` | `QueryPlan { steps: list[QueryStep] }` |
| Presenter | `build_presenter_prompt_fragment(query_labels=..., result_summary=...)` | `Presentation { summary, highlights, caveats }` |
| Drilldown | `build_drilldown_prompt_fragment(cube, focused_row=...)` | `DrilldownSuggestions { suggestions }` |

```python
from semql import build_router_prompt_fragment

router_prompt = build_router_prompt_fragment(catalog.as_dict(), viewer=viewer)
# ... send to LLM, parse RouterDecision, feed cubes+views into Generator ...
```

See `demos/pipeline_demo.py` for the full data flow without an LLM.

The single-stage `catalog.prompt(viewer=...)` is still available for
callers that don't want the four-stage breakdown.

## Common pitfalls

- **Forgetting `{alias}`** — `sql="amount"` won't compile; use
  `sql="{o}.amount"`.
- **Aggregating in `Measure.sql`** — let `agg=` do it.
  `sql="{o}.amount"` + `agg="sum"` emits `SUM(o.amount)`.
- **Required filter inside an OR branch** —
  `where=BoolExpr(op="or", ...)` doesn't satisfy `required_filters`.
- **Bare measure in `having`** — `having=[Filter(dimension="revenue",
  ...)]` and `having=[Filter(dimension="orders.revenue", ...)]` both
  resolve; the bare form looks up by alias.
- **Granularity not declared** — `granularity="hour"` against a dim
  with `granularities=("day", "month")` fails.
- **`scope` without registration** — `Cube(scope="X")` requires
  `Catalog(scope_fns={"X": fn})`; missing is a construction-time
  error, not a compile-time one.
- **`viewer` only on `compile`** — also pass it to `catalog.prompt`
  and to the four pipeline fragment builders, or the LLM sees cubes
  the viewer can't query.
- **Auto-binding `viewer_id` vs explicit context** — `ctx.viewer_id`
  set in `context=` wins over `viewer.viewer_id`; useful for
  impersonation flows, surprising otherwise.

## Step 2b — time-partitioned physical sources

When a fact table is spread across N physical tables partitioned by
time — "orders before 2024 are in `orders_archive` (rolled up
monthly), 2024 onwards in `orders_live`" — declare the partition
set on the cube. The compiler routes each query to the matching
sources based on the query's `TimeWindow.range` and `UNION ALL`s
the results. The auth / tenancy / scope wrappers wrap the union
as a whole, the same way they wrap a single source.

```python
from semql import TimePartition, TimePartitionedSource

orders = Cube(
    name="orders",
    backend=Backend.POSTGRES,
    alias="o",
    time_partition=TimePartition(time_dimension="placed_at"),
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
            range_end=None,                # open-ended
        ),
    ],
    measures=[...], dimensions=[...], time_dimensions=[...],
)
```

Each source's `range_start` / `range_end` are ISO-8601 lower / upper
bounds; `None` on a side means open-ended. Half-open intervals:
a source with `range_end="2024-01-01"` covers rows strictly before
that date. Touching ranges (e.g. `[2020, 2024)` and `[2024, 2030)`)
don't overlap, so no double-counting at the boundary.

The cube's `time_partition.time_dimension` must name a real
`TimeDimension` on the cube — the router uses it as the
routing key. `time_partition` is required whenever
`physical_sources` is non-empty; setting `time_partition` alone
is a configuration error.

**Schema drift.** When the older table uses different physical
column names, declare per-source renames. The cube's logical
field names are the source of truth:

```python
TimePartitionedSource(
    name="archive",
    table="orders_archive",
    range_start="2020-01-01",
    range_end="2024-01-01",
    column_renames={"placed_at": "ts", "total_amt": "amount"},
)
```

The compiler emits `SELECT ts AS placed_at, total_amt AS amount,
... FROM orders_archive` — the outer query and the planner see
the canonical cube names. Missing renames fall back to the
logical name verbatim. Every source must expose every cube
field, either by name or via `column_renames`; the catalog
validates that every rename target is a real field on the cube.

**What doesn't apply.**

- `tenancy="schema"` substitution (`{tenant_schema}`) on
  partitioned sources: each source's `table` accepts the
  same substitution.
- `source=DerivedTable(...)` and `time_partition` are
  mutually exclusive — a cube has one source declaration,
  full stop.
- `Rollup` is orthogonal: a cube can declare both
  `physical_sources` and `rollups`. The rollup match runs
  first; if a rollup fits, it bypasses the partition set
  entirely. The `physical_sources_hit` field on
  `CompiledQuery` lists the actual sources scanned (or the
  rollup's `physical_table`) for observability.

See `docs/specs/time-partitioned-sources.md` for the full
design record and the test surface.

## See also

- `skills/semql-requirement-discovery.md` — upstream skill that
  captures intent at the domain level. Hand off via a markdown
  requirements doc.
- `demos/pipeline_demo.py` — runnable end-to-end demo.
- `docs/api/semql.md` — auto-generated API reference for every
  public symbol. Run `uv run scripts/gen_api_docs.py` to refresh.
- `PHILOSOPHY.md` — design invariants. Identity is the caller's;
  authorisation is the compiler's; bypass-proof beats convenient.
- Existing test fixtures under `packages/semql/tests/` — the best
  living examples of well-shaped cubes (`test_auth.py`,
  `test_scope.py`, `test_time_spine.py` are particularly worth
  browsing).
