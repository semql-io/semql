# Spec: Time-partitioned physical sources

A fact-table pattern that real warehouses ship and SemQL doesn't
currently model: a single logical cube whose rows live in N physical
tables, partitioned by time. "Orders before 2024 are in
`orders_archive` (rolled up monthly); 2024 onwards in `orders_live`."
Today, authors either UNION ALL by hand in a `DerivedTable` (no
compile-time guarantees, no plan/explain visibility) or model it as
two cubes (loses the "one logical entity" guarantee the prompt + auth
surface rely on).

The design record is `TODOS.org:191` (#48). This spec is the API
surface and the compile-time contract.

## Goals

- A cube author can declare a list of physical sources, each tied to
  a half-open time range on a named time dimension.
- The compiler picks the right source(s) for a query based on the
  query's `TimeWindow.range`, unions the matches, and emits the
  same dialect-correct SQL the user would have hand-rolled.
- Unbounded queries (no time predicate) read every source.
- The auth / tenancy / scope wrappers wrap the entire union, so
  outer `OR` predicates can't bypass them — same guarantee as the
  existing `Cube` model.
- The choice is observable: `CompiledQuery.physical_sources_hit`
  lists the sources actually scanned.

## Non-goals

- Per-source `security_sql`. One cube = one auth surface.
- Dynamic partition discovery (a registry / catalog-of-catalogs).
- Source-level query pushdown (e.g. predicate pushdown past the
  union). That's a follow-up optimisation, not part of the first cut.
- A `rollup` per source. Rollups remain orthogonal — they pre-aggregate
  a cube at a coarser grain, not partition it.

## Model

New variant on the existing `CubeSource` discriminated union
(`model.py:474`):

```python
class TimePartitionedSource(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str                                    # unique within the cube
    table: str                                   # physical [schema.]table
    alias: str = "s"                             # per-source FROM alias
    # Half-open range on ``time_dimension``. ``None`` on either side
    # means open-ended in that direction.
    range_start: str | None = None               # ISO-8601 lower bound, inclusive
    range_end: str | None = None                 # ISO-8601 upper bound, exclusive
    # Per-source column renames. ``{"event_ts": "occurred_at"}`` says
    # this source's physical column ``occurred_at`` materialises the
    # cube's logical field ``event_ts``. Empty = use the logical name
    # verbatim. Field-level, not table-level: every entry maps a
    # logical field name to this source's physical column.
    column_renames: dict[str, str] = Field(default_factory=dict)
```

`range_start` and `range_end` are stringly-typed ISO-8601 bounds; the
compiler parses and compares them at compile time. Stringly-typed
keeps the model Pydantic-clean and lets the bound format be
backend-aware (a CH `DateTime64` literal differs from a Postgres
`date`). For the first cut, ISO-8601 covers every backend; revisit
when a backend genuinely needs something else.

`Cube.source` becomes `CubeSource | list[TimePartitionedSource] | None`.
Validation:

- A cube has exactly one source declaration: `table` (shorthand),
  `source` (explicit `PhysicalTable` / `DerivedTable`), or
  `physical_sources: list[TimePartitionedSource]`. Mixing them
  raises at construction.
- `physical_sources` must be non-empty.
- Source names are unique within a cube.
- `time_dimension` (the routing dim) must name a `TimeDimension` on
  the cube. Declared once on the partition set:

  ```python
  Cube(
      ...,
      time_partition=TimePartition(time_dimension="created_at"),
      physical_sources=[...],
  )
  ```

- Every `column_renames` key must name a measure, dimension, time
  dimension, or segment on the cube. The value is a free-form SQL
  fragment; the compiler emits it verbatim (the source's column
  names are caller-owned, like `Cube.table`).
- Overlap is allowed (the compiler unions the overlap). Two
  sources with `None` on the same side AND a real bound on the
  other is also allowed — e.g. a "live" source with `range_end=None`
  and a "historical" source with `range_start="2024-01-01"`. Refusing
  overlap would forbid the common "everything before X is rolled up;
  everything after X is live" pattern.
- A range with `range_start >= range_end` is a construction error.

## Routing

Compiler contract (in `compile.py`, called from
`_emit_simple_query`'s source-resolution step). **The routing uses
only the plan's `TimeWindow.range`** — bare time predicates in
`filters` (e.g. `Filter(dimension="orders.created_at", op="between",
...)`) do *not* narrow the source set. A query that filters time via
bare predicates scans every source; the outer WHERE filters the
result. Slightly inefficient, but the contract is unambiguous and
the test surface is small. A follow-up can attempt to intersect
with bare time predicates (handling `gte`/`lte`/`between` and
combinations) once the first cut is in production.

```
matched = []
for src in cube.physical_sources:
    if _ranges_intersect(query_time_range, src.range):
        matched.append(src)

if not matched:
    # Query's time range falls in a gap. Two options:
    # - Loud CompileError listing the gap
    # - Empty result (caller's predicate filters everything)
    # Decision: empty result. The query's WHERE on time_dim is
    # what filters; the partition set is the address space.
    # A loud error would force authors to alias their partitions
    # to "cover" gaps that queries would never hit anyway.
```

`_ranges_intersect` treats `None` as `±infinity`. Half-open on both
sides: `range_start` inclusive, `range_end` exclusive. Touching
ranges (e.g. `["2024-01-01", "2025-01-01")` and `["2025-01-01",
"2026-01-01")`) don't overlap.

When a rollup matches the query, the rollup's `physical_table` is
appended to `matched` with `range=(None, None)` — the rollup is
"all-time" for routing purposes. This means a query that hits a
monthly rollup ignores the source partitioning entirely and reads
just the rollup.

### SQL emission

Each matched source becomes a CTE:

```sql
WITH
  s_orders_archive AS (
    SELECT <cols_with_renames> FROM archive.orders
    WHERE <cube-level predicates other than time_dim>
  ),
  s_orders_live AS (
    SELECT <cols_with_renames> FROM live.orders
    WHERE <cube-level predicates other than time_dim>
  )
SELECT
  date_trunc('month', COALESCE(s_orders_archive.created_at,
                               s_orders_live.created_at)) AS created_at_month,
  SUM(...) AS revenue
FROM (
  SELECT * FROM s_orders_archive
  UNION ALL
  SELECT * FROM s_orders_live
) AS u
WHERE u.<time_dim> >= :p_0 AND u.<time_dim> < :p_1
GROUP BY 1
```

The cube-level wrappers (`security_sql`, `Cube.scope`, discriminator
`tenancy`) wrap the *outer* union subquery, not each per-source CTE,
because they apply to the cube as a whole. Predicate pushdown past
the union is left to the database; we don't try to be clever.

The query's `TimeWindow.range` is preserved as a filter on the outer
query (the existing path). Source matching is just the *selection*
of which sub-CTEs to build; the *predicate* is the query's own time
filter. This means a query whose time range happens to lie inside
one source still emits the source as a single-table CTE (no UNION
ALL) — verified in snapshot tests.

## Column renames

`column_renames` is a per-source `{logical_name: physical_column}`
map. The compiler emits a SELECT-list in each source CTE that
project each cube field to its physical column in that source:

```sql
s_orders_archive AS (
  SELECT
    id        AS id,             -- no rename
    total_amt AS revenue,        -- revenue -> total_amt in this source
    placed_at AS created_at      -- created_at -> placed_at in this source
  FROM archive.orders
)
```

The outer query refers to the cube's logical names (`revenue`,
`created_at`), so the renames are invisible to the planner and the
prompt. The compiler raises at construction if a `column_renames`
key doesn't name a real field on the cube. The compiler raises at
compile time if a query requests a field that *no* source renames
to (i.e. the field is supposed to exist everywhere, and one source
forgot to expose it) — or, more pragmatically, raises at
construction: every physical source must expose every cube field,
either under its logical name or via a `column_renames` entry.

## Observability

`CompiledQuery` gains:

```python
physical_sources_hit: tuple[str, ...] = ()
```

Populated whenever the cube has `physical_sources` set, with the
names of the sources actually scanned (after range intersection).
Empty tuple for cubes without the feature — backwards compatible.

`Catalog.explain()` (already landed as part of #LOG) renders the
matched sources, the range intersection, and the per-source
column renames. New unit test pins the explain output.

## API breakage

`CubeSource = PhysicalTable | DerivedTable` widens to
`CubeSource = PhysicalTable | DerivedTable | list[TimePartitionedSource]`
*plus* a new top-level `physical_sources: list[TimePartitionedSource]`
on `Cube`. This is a structural change. Run
`uv run scripts/check_api_break.py --base main` and document the
intentional break in the commit body — the discriminator widening
on `CubeSource` is the only public-surface change; everything else
is additive.

## Test plan

`tests/test_time_partition.py` (new file, mirrors the per-feature
test pattern in `tests/test_rollups.py`):

1. **Single source, query range inside it.** Source covers
   `[2020-01-01, 2030-01-01)`, query asks for `[2024-01-01,
   2024-04-01)`. Emit a single-table CTE, no UNION ALL.
2. **Two sources, query spans both.** Sources
   `[2020-01-01, 2024-01-01)` + `[2024-01-01, 2030-01-01)`,
   query `[2023-09-01, 2024-06-01)`. Emit two CTEs, UNION ALL,
   outer time filter.
3. **Query outside all sources.** Query `[2010-01-01,
   2011-01-01)`, sources cover `[2020-01-01, ∞)`. Emit zero
   CTEs from this cube; the query's other predicates drive
   the result (typically empty).
4. **Unbounded query.** No time predicate. Emit CTEs for
   every source.
5. **Rollup match wins.** Cube has a rollup covering the
   query's grain; the partition set is bypassed, only the
   rollup is scanned.
6. **Column renames.** Source A renames `created_at` to
   `placed_at`; the emitted CTE projects `placed_at AS
   created_at`.
7. **Range edge case — touching.** Sources
   `[2020-01-01, 2024-01-01)` + `[2024-01-01, 2026-01-01)`,
   query `[2024-01-01, 2024-04-01)`. Hits exactly the second
   source. No double-counting.
8. **Per-source security wrapper applies to the union, not
   each source.** Cube has `security_sql`, query routes to
   two sources; the wrapper appears around the outer query,
   not the per-source CTEs.
9. **Construction errors.** Empty list, duplicate source
   names, invalid `range_start >= range_end`, `column_renames`
   key not on the cube, `time_dimension` not a
   `TimeDimension` on the cube, mixing `physical_sources`
   with `table` / `source`.
10. **Snapshot.** `tests/__snapshots__/test_time_partition.ambr`
    pins the four canonical SQL shapes.

## Implementation order

1. Model: `TimePartitionedSource`, `TimePartition`, and the
   `Cube.physical_sources` / `Cube.time_partition` slots
   (`model.py`).
2. Construction-time validation: uniqueness, range ordering,
   column-rename key membership, no mixing with `table` /
   `source`.
3. Tests 1, 2, 3, 4, 7, 8, 9, 10 (red).
4. Compile-time routing: source selection, CTE emission,
   UNION ALL assembly, range filter placement (green).
5. `physical_sources_hit` on `CompiledQuery`, `Catalog.explain`
   rendering, test 5, test 6.
6. `just check` (fmt, lint, typecheck, test).
7. `uv run scripts/check_api_break.py --base main` and
   document the break in the commit body.
8. Update `skills/semql-cube.md` with the authoring shape and
   link this spec.

## Open questions

Resolved during the design interview, recorded for posterity:

- **Routing key.** Time range on a named time dim, not a static
  expression match. Time windows are how every real warehouse
  declares the pattern.
- **Rollup flavour.** "Same shape, possibly rolled up at the
  source" — the cube's column contract holds. Different shapes
  need different cubes.
- **Unbounded query.** Scan every source; the alternative
  (refuse / use a designated default) is more surprising than
  helpful.
- **Schema drift.** Per-source renames allowed; per-source
  column subsets are not (drift detection at construction time
  is the audit trail).
- **Model placement.** New variant on `CubeSource`; don't
  deprecate `table` / `source`.
- **Auth.** Wrapper applies to the unioned source.
- **Granularity.** Declared once on the routing dim; sources
  may store at different native resolutions, truncation
  happens in the outer SELECT.
- **Observability.** Expose on `CompiledQuery` and `explain`,
  not behind a flag.
- **Time filter source.** `TimeWindow.range` only; bare time
  predicates don't narrow the source set. Documented above
  in the compiler contract.

## See also

- `TODOS.org:191` — design record (#48).
- `packages/semql/src/semql/model.py:474` — current
  `CubeSource` union; this spec widens it.
- `packages/semql/src/semql/model.py:301` — existing `Rollup`
  model; orthogonal feature, but the two interact in the
  routing step.
- `skills/semql-cube.md` — authoring skill, to be updated
  with the new pattern.
- `docs/decisions.md` — D-series record (no D-entry needed;
  the decision context is this spec).
