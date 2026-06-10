# Spec: GraphQL-Borrowed Features for SemQL

## Overview

Evaluation of features borrowable from GraphQL for SemQL, screened
against `PHILOSOPHY.md` and the existing `TODOS.org` backlog. Six
candidate features, three recommended for action.

## Scope of this doc

This is a **decision record**, not a feature backlog. For each of six
GraphQL features:

1. What it would look like in SemQL
2. Compatibility with PHILOSOPHY.md
3. Effort estimate
4. Recommendation (implement / defer / reject / document)

## Candidate 1: Field Aliases — `SemanticQuery.aliases`

**GraphQL source:**
```graphql
revenue: orders { total: amount, net: amount - discount }
```

The same field can appear under different names in different parts of
a result.

**SemQL gap:**

`View` (`model.py:861`) re-labels fields at catalog time. There is no
query-time mechanism. If a dashboard needs `orders.revenue` rendered
as "Net Revenue" *and* "Gross Revenue" in the same result, today it
needs two compiled queries.

**Proposed shape:**

```python
class SemanticQuery(BaseModel):
    ...existing fields...
    aliases: dict[str, str] = Field(default_factory=dict)
    # Maps output column name -> qualified field ref
    # {"net": "orders.revenue"} emits column "net" pointing at orders.revenue
```

**PHILOSOPHY compatibility:** ✅ Fits "Compile errors are better than
runtime errors" (alias keys must resolve to declared fields). The
compiler rejects alias keys that collide with existing output column
names. Output column is the alias key, not the qualified ref.

**Effort:** ~200 LOC + tests. Touches:
- `spec.py` — add `aliases` field to `SemanticQuery`
- `compile.py` `_projection_stage` — emit alias as the column name
- `validate.py` — alias keys must be unique and resolve to declared fields
- `tests/test_compile.py` — failing test first, then implementation

**Recommendation:** **Implement.** High leverage, low risk, fills a
real gap. The deferred-alias cleanup work in `TODOS.org:1341-1362` is
a separate concern (deprecation of old names) and can ride alongside.

---

## Candidate 2: `__typename` Discriminator on Resolve

**GraphQL source:** `__typename` lets a client ask "what kind of
object is this?" without re-introspecting the schema.

**SemQL gap:** `resolve_field` returns `Measure | Dimension |
TimeDimension | Segment`. Callers must `isinstance`-check to
disambiguate. `ColumnMeta.kind` already exists on `CompiledQuery`
(`compile.py:130`) — `Literal["measure", "dimension", "time",
"computed"]` — but consumers cannot ask "what kind is this *unresolved*
ref?".

The MCP per-cube tool factory (`semql-mcp/server.py:100-101`) needs
this for tool-name construction; today it does isinstance checks.

**Proposed change:** Add a `kind: Literal["measure", "dimension",
"time", "segment"]` accessor to the resolver's return shape. Either
as a separate return value, a tagged union, or a property on the
returned field.

**PHILOSOPHY compatibility:** ✅ Fits "Errors serve machines and
humans. Structure carries the meaning; `str()` carries the message."
MCP clients branch on `kind` without importing `BaseField`.

**Effort:** Trivial (~50 LOC). `introspect.resolve_field` already
returns the resolved field; just add a `kind` accessor. Plus tests.

**Recommendation:** **Implement.** Smallest isolated change. Clean win
for MCP and any consumer that needs to disambiguate fields.

---

## Candidate 3: Single-Endpoint-Per-Cube Codification

**GraphQL source:** Apollo / Cube.js pattern: one endpoint per type,
schema-generated from the type registry.

**SemQL status:** The MCP server's per-cube tool factory
(`semql-mcp/server.py:100-101`, `tests/test_per_cube_tools.py`)
**already implements this pattern** — `query_<cube>(measures=...,
dimensions=...)` is the per-cube-endpoint shape. The pattern is real,
working, tested.

**Recommendation:** **Document.** Don't add a parallel GraphQL
surface. Document the per-cube MCP pattern as "the GraphQL-style
single-endpoint shape" in the README. If a real GraphQL consumer
ever shows up (Apollo Federation, Hasura-style integration), extract
`as_graphql_schema` from the MCP factory at that point.

**PHILOSOPHY compatibility:** ✅ "Not a framework. It composes with
your stack — it does not own it." A standalone GraphQL surface is
framework-shaped. The MCP pattern is enough.

**Effort:** Doc-only. Re-evaluate if a concrete consumer materializes.

---

## Candidate 4: `@skip` / `@include` — `query.conditionals`

**GraphQL source:** `@skip(if: $someBool)` lets one query template
serve multiple variants without rewriting.

**SemQL gap:** None currently — the consumer typically branches
*outside* the query. MCP clients already get this for free (server
decides what to call).

**PHILOSOPHY compatibility:** Borderline. The compiler is "Pure: no
I/O, no globals, no time-of-day." A `conditionals` field requires
the caller to provide a value at compile time, which leaks runtime
state into the spec. The existing `rewrite(q, Drilldown(...))`
pattern is the cleaner shape.

**Recommendation:** **Defer.** The `RewriteOp` machinery
(`TODOS.org:68-69`) already covers the use case. Adding
`conditionals` is YAGNI.

---

## Candidate 5: Streaming / `@defer` / `@stream`

**GraphQL source:** `@defer` for partial results, `@stream` for list
fields.

**SemQL status:** `CompiledQuery` is a single-shot SQL string. The
closest analogue in TODOS is `AsyncEngine.iter_run`
(`TODOS.org:2841-2867`) — streaming merge at the executor layer.
Adapter-level streaming is being explored
(`TODOS.org:2705-2713`).

**PHILOSOPHY compatibility:** The compiler is pure. Streaming is a
property of the executor, not the spec. A `@defer` on `SemanticQuery`
would force the compiler to know about partial results — wrong layer.

**Recommendation:** **Reject.** The existing executor-side streaming
work is the right place. Compiler-side streaming violates the
"compiler has no I/O" invariant.

---

## Candidate 6: Fragments — Query-Time Composition

**GraphQL source:** `fragment OrderFields on Order { id, total }` —
name a sub-selection once, reuse across queries.

**SemQL status:** `View` (`model.py:861`) is the catalog-time
equivalent. The user story for query-time fragments is weak — if a
fragment is reusable, it belongs in the catalog. If it's one-off, the
consumer is already a `View` author or doesn't need it.

**PHILOSOPHY compatibility:** "Catalogs must be serialisable and
versioned. A catalog that cannot cross a process boundary cannot serve
a real system." Query-time fragments cross the boundary by escaping
to the consumer — anti-pattern.

**Recommendation:** **Reject.** `View` is the answer; a new
mechanism is duplication.

---

## Summary

| # | Feature | Action | Effort |
|---|---------|--------|--------|
| 1 | Field aliases | **Implement** | ~200 LOC + tests |
| 2 | `__typename` kind tag | **Implement** | ~50 LOC |
| 3 | Per-cube codification | **Document** | Doc only |
| 4 | `@skip` / `@include` | **Defer** | — |
| 5 | Streaming | **Reject** | — |
| 6 | Fragments | **Reject** | — |

## Implementation Order

1. **`__typename` kind tag** (smallest, isolated) — touches
   `introspect.resolve_field`, MCP factory, tests. Land first.
2. **Field aliases** — touches `spec.py`, `compile.py`'s projection
   stage, validation. Test-first: write a failing test for
   `query(measures=["orders.revenue"], aliases={"net":
   "orders.revenue"})` → expect
   `CompiledQuery.columns == ["net", ...]`. Land second.

## Open Question: Where should the alias map live?

Two valid choices:

- **`SemanticQuery.aliases: dict[str, str]`** (catalog-agnostic,
  fires at compile time). Matches GraphQL ergonomics. Compiler stays
  pure. Validates against the catalog at resolve time.
- **`CompiledQuery.output_columns: dict[str, str]`** (post-compile
  view of the same data). Closer to "view over the result" — but
  `ColumnMeta` already exists on `CompiledQuery` for type info, not
  naming overrides.

**Recommendation:** Go with `SemanticQuery.aliases`. Matches GraphQL
ergonomics, keeps the compiler pure, validates against the catalog at
resolve time. The alternative is a leaky abstraction.

## A Note on META Cubes

META cubes (`catalog_cubes`, `catalog_measures`, `catalog_dimensions`)
already implement reflection over the catalog. A
`resolve_<cube>_aliases` META cube would be a natural fit for
exposing catalog-wide alias conventions, but it is not needed for the
per-query alias feature above. Park.

## PHILOSOPHY Invariants Preserved

- "Compile errors are better than runtime errors" — alias keys
  resolve at compile time; collisions raise.
- "The compiler has no I/O" — aliases are pure spec data; no
  runtime/streaming concerns introduced.
- "Errors serve machines and humans" — the kind tag is structural,
  not just a string.
- "Catalogs must be serialisable and versioned" — no change to
  catalog serialisation; aliases live on the query, not the catalog.
- "Not a framework" — no GraphQL surface; the per-cube MCP pattern is
  the integration point, not a new framework.
