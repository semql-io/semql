# Cross-source resolution & the Auto-Planner

**Status:** design / proposed
**Date:** 2026-06-16
**Companion:** [`lookup-enrichment-critique-2026-06.md`](./lookup-enrichment-critique-2026-06.md)

## Motivating question

> "How many books did Nikhil buy last month?"

Customer **metadata** (name) lives in DB A; order **facts** live in DB B.
Answering it requires filtering a measure on backend B by dimension values
that live on backend A. The phrase hides two distinct problems:

1. **Entity / value resolution** — `"Nikhil"` (human string) → `customer_id`;
   `"books"` (category phrase) → a product/category key. NL phrase → identifier.
2. **Cross-source predicate pushdown** — the measure (`count(orders)`) is on
   backend B but must be constrained by predicates whose *values* live on
   backend A.

This spec commits to three capabilities and one orchestrator that chooses
among them:

- **B — Lookup-as-filter** (inbound resolution of declared vocabularies)
- **C — Caller-authored resolution** (status quo `SemiJoin`, documented + preserved as override)
- **G — DuckDB ATTACH / pushdown** (real cross-DB join in the engine)
- **F — Auto-Planner** (the brain: picks a strategy per cross-source filter)

```
flat SemanticQuery (cross-source filters, no hand-authored SemiJoin)
   │
   ▼  Auto-Planner (new rewrite pass, pre-compile)
   ├─ filter on a Lookup dim?       → resolve values exactly (Option B feeds selectivity)
   ├─ caller already wrote SemiJoin? → respect it, don't re-plan (Option C)
   └─ choose strategy per cross-source filter:
        ├─ semi_join     (inject SemiJoin, splice IN-list)   ← cheap / selective
        ├─ bridge_merge  (existing federation ship-and-join)  ← needs foreign attrs in output
        └─ attach        (DuckDB ATTACH, push the join down)  ← Option G, if available
```

## What already exists (grounding)

| Concern | Where | Notes |
|---|---|---|
| Vocabulary + resolution | `semql/model.py:1508` (`Lookup`), `semql/lookups.py` (`resolve`, `materialize`, `enrich_result`) | `resolve()` does exact→substring→fuzzy, returns canonical keys. Consumed only by the prompt layer today; compiler never touches it. |
| Value-list filter | `semql/spec.py:292` (`SemiJoin`), `semql/semijoin.py` (`compile_semi_join_query`) | Inner query → de-duped value list → spliced as `IN (...)` into outer. `_check_key_types` guards type compatibility. |
| Federation / bridge-merge | `semql/federate.py` (`compile_federated_query`, `_touched:388`, `_find_bridges:423`, `_parse_bridge`) | `_touched` already pulls *filter-only* foreign cubes into scope (`_filter_where_cube_names`), so a flat cross-source filter federates today — via bridge-merge. |
| Execution | `semql-engine/engine.py:439` (`_execute_uncached`), `MergeEngine` protocol | Per-dialect `Adapter` pulls rows → materialize into DuckDB `frag_i` temp tables → merge SQL. DuckDB is already the federation substrate. Inline merge being deprecated in favour of `MergeEngine`. |
| Cost | `semql/cost.py` (`estimate_cost`, `Cube.size_hint`) | Deliberately ignores selectivity ("a guardrail, not a planner"). |

**Key insight:** a flat query with a cross-source *filter* already works — but it
federates via **bridge-merge**, which ships *all* of last month's orders into
DuckDB and only then inner-joins against the one matching customer. For a
selective filter that is hugely wasteful. The semi-join path is far cheaper
(resolve Nikhil → 1 id → `orders.customer_id IN (4471)` → selective fact scan),
but today the *caller* must know to hand-author it. The Auto-Planner makes the
right choice automatically.

## Canonical terms

- **filter-only foreign cube** — a cube referenced *only* by `filters`/`where`,
  never by `measures` or output `dimensions`. Detectable via the existing
  `_touched` / `_filter_where_cube_names` split.
- **resolved key set** — the bridge-key values produced by applying the foreign
  filter; the prospective IN-list.
- **pushdown strategy** ∈ `{semi_join, bridge_merge, attach}`.

---

## Option B — Lookup-as-filter (inbound resolution)

Today `Lookup`'s vocabulary half (`values`/`loader`, `max_inline`, `resolve()`)
is consumed only by `semql_prompt.planner_prompt` and any `resolve_<dim>` tool.
Make a `Filter` whose `dimension` has a registered `Lookup` resolve at plan time.

Two sub-modes:
- **Inline** (cardinality ≤ `max_inline`): `resolve(catalog, "products.category", "books")`
  → `["books"]` → rewrite the filter to canonical value(s). Exact and free for
  enums (category, status, region).
- **Key resolution** (lookup over a reference table via `loader`/`enricher`):
  `"books"` → `category_id=12`, producing a value list that feeds a semi-join.

Resolution stays out of the **compiler** (preserving the sans-I/O invariant): it
fires in the engine/prompt pipeline — the same boundary `materialize`/`resolve`
already respect — and resolved values are passed into compilation as concrete
`Filter.values`.

**Pros:** reuses `resolve()` + `Lookup` wholesale; deterministic for enums; gives
the Auto-Planner an *exact* key count (best possible selectivity signal); solves
"books" cleanly.
**Cons:** only helps dimensions with a declared `Lookup`; useless for open
high-cardinality text ("Nikhil") → those fall to semi-join; fuzzy tier can return
multiple candidates → needs a disambiguation contract (deferred).
**User story:** *As an analyst, I say "books" and "EMEA"; the planner resolves both
to canonical keys via their lookups before compiling — I never type an id, and the
plan filters precisely.*

---

## Option C — Caller-authored resolution (status quo, documented)

A caller/LLM hand-authors a `SemiJoin` whose inner query filters
`customers.name = 'Nikhil'`, projects `customers.customer_id`, and splices it as
an IN-list into the orders query. The join *is* the resolution.

**Change:** mostly documentation, plus one guarantee — the Auto-Planner **must
not override caller-supplied `semi_joins`**. If `q.semi_joins` already covers a
foreign cube, treat it as planned and skip auto-planning for that cube. This makes
C the explicit escape hatch.

**Pros:** zero new compile logic; deterministic; caller owns selectivity and match
semantics; essential as the override path.
**Cons:** no name→id disambiguation (the join is the match — "Nikhil" vs
"Nikhil P." silently differ); verbose for an LLM; error-prone (wrong bridge key —
though `_check_key_types` catches type mismatches).
**User story:** *As an integrator who knows my data, I hand-write the semi-join for
an unusual join and trust the planner to leave it alone.*

**Doc deliverable:** a "Cross-source queries" guide showing the three authoring
levels — (1) flat query, let the planner decide; (2) declare a `Lookup` for clean
resolution; (3) hand-author a `SemiJoin` to override.

---

## Option G — DuckDB ATTACH / pushdown

The engine already materializes fragments into DuckDB and merges there. Add an
**attach execution mode**: instead of adapters pulling rows and DuckDB joining
materialized temp tables, DuckDB `ATTACH`es the source databases
(postgres/mysql/sqlite scanners) and runs the cross-source join as one DuckDB
query with pushdown. Implemented as a new `MergeEngine` / execution strategy,
gated by an engine capability flag `attachable: dict[Dialect, AttachSpec]`
(connection string, read-only). When both fact and dim backends are attachable,
the planner may emit the `attach` strategy.

**Pros:** real joins — no IN-list ceiling, no "measures on primary partition
only," **no non-distributive-agg refusal** (count_distinct/min/max work because
it's one engine); leans on DuckDB pushdown; simplest model for arbitrary shapes.
**Cons:** engine needs live credentials to *both* DBs — a real governance/security
shift (the two-DB split may have been deliberate isolation); pushdown quality
varies by scanner; a bad plan can full-scan a huge fact table; read-only
enforcement (`_assert_*_read_only`) must extend to attached scans.
**User story:** *As an operator who already centralizes DB access in the engine, I
enable ATTACH so the hardest cross-source questions (distinct counts, foreign
group-bys) just work without per-query strategy.*

---

## Option F — The Auto-Planner (detailed)

A new rewrite pass in front of `compile_federated_query`. It chooses a pushdown
strategy per cross-source filter and rewrites the query accordingly. Reuses
`_find_bridges`/`_parse_bridge` for bridge keys and `estimate_cost`/`size_hint`
for selectivity.

### Placement (resolved)

The planner is a **new module — `semql/autoplan.py`** — not an extension of any
existing one. Module roles were confirmed by reading the code:

- `semql/plan.py` — prompt-role *output types* (`RouterDecision`, `QueryPlan`,
  `Presentation`, `DrilldownSuggestions`). Despite the name, **not** a query
  planner. Naming the new module `autoplan.py` avoids the collision.
- `semql/rewrite.py` — a **closed** conversational-edit vocabulary (`AddFilter`,
  `Drilldown`, …) whose docstring states new ops are catalog-shape changes, not a
  runtime extension point. Not the home for federation rewrites.
- `semql/partition.py` — time-partitioned physical-source routing. Unrelated.
- `semql/logical.py` — the real `LogicalPlan` IR and where cross-backend
  partitioning already lives (`partition_scans`, `build_join_graph`,
  `find_join_path`, `_detect_symmetric_agg`). The planner **reuses**
  `build_join_graph`/`find_join_path` for reachability + nested-resolution
  detection, and `federate._find_bridges`/`_parse_bridge` for bridge keys.

`autoplan.py` is a sans-io `SemanticQuery → SemanticQuery` pass (plus a strategy
verdict) sitting in front of `compile_federated_query` / `compile_semi_join_query`.
It operates on **already-resolved** `Filter.values` — see the resolution split
below.

### Algorithm (per cross-source filter group — foreign cube `F` filtering primary `P`)

```
0. If q.semi_joins already covers F  → skip (Option C override).

1. Find the bridge join P↔F via _find_bridges.
   None → REFUSE ("no Join declared between P and F; add one or hand-author a SemiJoin").

2. Does F contribute an OUTPUT dimension (grouped/projected, not just filter)?
   YES → semi_join impossible (it only filters):
        attach available?       → attach
        elif agg distributive?  → bridge_merge        (existing path)
        else                    → REFUSE (non-distributive + foreign group-by;
                                          suggest enabling ATTACH)
   NO (filter-only) → go to 3.

3. Estimate resolved key set size:
        filter dim has a Lookup? → exact count from resolve()          (Option B)
        else                     → size_hint × selectivity heuristic
                                   (eq ≈ unique, in ≈ |values|)
   est ≤ SEMI_JOIN_MAX (config, e.g. 10k) → semi_join  (inject SemiJoin)
   est >  SEMI_JOIN_MAX:
        attach available?       → attach
        elif agg distributive?  → bridge_merge
        else                    → semi_join + log("large IN-list: N keys")  ← no silent cap
```

### SemiJoin injection (the common branch)

```python
SemiJoin(
    dimension=f"{P}.{bridge.left_key}", op="in",
    select=f"{F}.{bridge.right_key}",
    source=SemanticQuery(
        dimensions=[f"{F}.{bridge.right_key}"],
        filters=[<the foreign filter(s)>],
    ),
)
```

Then compile via the existing semi-join path. `_check_key_types` guards type
compatibility.

### Multi-dimension case ("books" *and* "Nikhil", both foreign)

Inject **sibling** semi-joins (one per foreign cube), never nested (nested is
refused). Two independent cubes (customers, products) each bridging to orders →
two IN-lists splice into the orders fragment. If "books" were reachable only
*through* customers, that's nested → REFUSE with a clear message.

### Selectivity & cost

`cost.py` deliberately ignores selectivity. The planner adds a thin selectivity
layer: Lookup-resolved filters give exact counts; equality on a dimension flagged
unique/high-distinct ≈ few keys; otherwise a configurable default. This is a
*strategy heuristic*, not a runtime predictor — wrong guesses cost performance,
never correctness (semi-join and bridge-merge return identical rows; only ATTACH
changes agg capability).

### Resolution split & disambiguation contract (resolved)

Resolution (human phrase → canonical key) is **I/O** (a `loader` or reference-table
query), so it stays at the edge — never in the sans-io compiler. There is already a
live surface to mirror: the MCP server (`semql-mcp/server.py:347`) exposes
`resolve_lookup(dimension, query, max_candidates=5) → {values: [...]}`, backed by
`lookups.resolve()` (exact → substring → fuzzy). The LLM calls it, receives a
*candidate list*, and picks. Disambiguation already lives at the tool/agent layer.

Two clean stages:

1. **Resolution stage** (edge: engine / prompt pipeline). Turns cross-source
   filter phrases into canonical key sets, reusing `lookups.resolve()`. Emits a
   typed `ResolutionOutcome` per filter:
   - `Resolved(values)` — a single canonical value (tier-1 exact, or exactly one
     hit). Splice into `Filter.values`; proceed.
   - `Ambiguous(candidates, labels)` — >1 hit. **Structured refusal** carrying the
     candidates (same shape as `resolve_lookup`'s multi-element `values`), so the
     caller/LLM re-issues with the chosen canonical value. Not an exception — no
     data lost.
   - `Unresolved()` — 0 hits. Structured refusal: "no match for 'X' on dim D; ask
     the user to clarify" (mirrors `resolve()` returning `[]`).
2. **Auto-Planner stage** (`autoplan.py`, sans-io). Receives a query whose
   cross-source filters already hold canonical values; decides strategy + injects
   SemiJoins. Never does I/O — preserving the compiler invariant.

**Scope for P1:** the planner auto-resolves only **Lookup-backed** dimensions to
keys via this contract. Open-text dims with no `Lookup` (e.g. `customers.name`)
are pushed into the semi-join *inner query* as a verbatim predicate — the join
*is* the resolution (Option C semantics). Fuzzy name→id disambiguation (two
"Nikhil"s) is therefore a data condition, surfaced only if the caller opts into a
disambiguation projection; full open-text resolution stays deferred (names out of
scope for now).

### Refusals must be loud

Per the existing federation-refusal style and the "no silent caps" rule, every
REFUSE names the cause and the remedy (declare a Join / enable ATTACH /
hand-author a SemiJoin). Every large-IN fallback `log`s the key count.

## Phasing

1. **P1 — SHIPPED** (`semql/autoplan.py`). Filter-only → semi-join injection;
   always semi-join. Solves the motivating question end-to-end. Surfaced + fixed
   the federation `LEFT JOIN` "Gap C" so bridge-merge agrees with semi-join.
2. **P2 — SHIPPED.** Cost-based `semi_join` ↔ `bridge_merge` via the operator +
   `size_hint` heuristic: non-distributive measure → semi_join; selective filter
   (eq/in or Lookup-backed) → semi_join; broad filter with `size_hint >
   semi_join_max` → bridge_merge; else semi_join. `semi_join_max` defaults to
   10 000, overridable per call. Records a `CrossSourceDecision` per foreign cube.
3. **P3** — `attach` strategy + engine capability flag (Option G).

## Tests (Red/Green)

- **Strategy-selection golden tests:** assert the chosen strategy per query
  (filter-only selective → `semi_join`; foreign group-by → `bridge_merge`;
  count_distinct + foreign filter + attach → `attach`).
- **Equivalence test (safety net):** on a fixture dataset, assert `semi_join` and
  `bridge_merge` plans return *identical* rows for the same query — guards the
  planner's freedom to choose.
- Boundary test at `SEMI_JOIN_MAX`.
- Multi-foreign-cube sibling injection.
- Nested-resolution refusal; missing-bridge refusal.

## Open questions / deferred

- **Open-text name→id resolution** (when "Nikhil" matches several customers, with
  no `Lookup` to constrain). The resolution-stage contract above handles
  Lookup-backed dims; full fuzzy resolution of unbounded text fields, and how a
  disambiguation projection surfaces the fan-out, stays deferred (names out of
  scope for now).
- ATTACH read-only enforcement + credential/governance model for the engine.

### Resolved during design

- **Planner placement** — new module `semql/autoplan.py`; `plan.py`/`rewrite.py`/
  `partition.py` confirmed unsuitable (see Placement above).
- **Disambiguation contract** — typed `ResolutionOutcome`
  (`Resolved`/`Ambiguous`/`Unresolved`) at the I/O edge, mirroring the existing
  `resolve_lookup` MCP tool (see Resolution split above).

## Assumptions

1. "books" is a categorical dimension suited to a `Lookup`; "Nikhil" is open text
   resolved via semi-join.
2. The planner respects any caller-supplied `SemiJoin` rather than re-planning it.
