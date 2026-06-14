# Entities: row-mode reads and opt-in mutations

Status: design agreed 2026-06-12 (interview), not implemented.
Supersedes: the staged `Entity`/`MutableEntity` builder code in `model.py`
(partially reusable, see §9) and refines the S8 sketch in TODOS.org.

## 1. Motivation

The read layer answers analytic questions (aggregates over cubes). Two
escape hatches are missing for text-to-SQL pipelines and the MCP server:

1. **Row-mode reads** — "show me order 42", "list user 7's open orders".
   Point lookups and short lists over OLTP tables or warehouse tables,
   not aggregations. In-library scope: SQL backends only. The design
   must be *extendable* to table-shaped non-SQL sources (REST, KV)
   without library changes — users bring an adapter.
2. **Mutations** — LLM-constructed, compiled DML. Opt-in at every layer,
   never LLM-generated SQL.

## 2. Decisions (pinned)

| # | Decision | Choice |
|---|----------|--------|
| D1 | Read execution | **Plan + adapters**: compile to a restricted `RowPlan`; SQL backends render SQL; custom backends implement a row-capable adapter that interprets the plan. Compiler stays I/O-free. |
| D2 | Mutation authoring | LLM may construct `SemanticMutation` specs directly. Compiled, validated, never raw SQL. |
| D3 | Mutation targeting | PK targeting by default; predicate targeting **opt-in per entity**; every compiled mutation carries a preflight preview. |
| D4 | Class shape | `Entity` (read-only base) and `MutableEntity(Entity)` (adds write schema). Separate classes; read-only entities are first-class. |
| D5 | Read surface v1 | Fetch-by-key + list with **allowlisted filters** (`list_filters`), limit/cursor. Anything richer routes to the analytic layer. |
| D6 | MCP surface | Per-entity tools by default; a config flag collapses to generic tools (pairs with I12 prompt budget). |
| D7 | Multi-cube writes | Design now (§8), build later. v1 mutations target exactly one cube. |
| D8 | RowPlan ↔ LogicalPlan | **Derive**: entity reads lower to an internal `LogicalPlan` (ungrouped mode — already exists); a derivation function projects it into the adapter-facing `RowPlan`. The IR stays internal; the contract stays tiny. |
| D9 | Pagination | **Keyset cursors, executor-owned, opaque to clients.** SQL backends: keyset over `default_order` + `entity.key` tiebreaker, decoded into predicates at compile time. Custom backends: token passes through `RowPlan.cursor` verbatim; the adapter owns it. |
| D10 | List predicate vocabulary | **eq / in on dimensions + start/end range on time dimensions only**, all gated by `list_filters`. Adapter authors who can't range-filter simply don't allowlist the time dim — the allowlist is the single control. |

Canonical terms: **fetch** (read one row by key), **list** (read rows via
allowlisted filters), **RowPlan** (serializable restricted plan for
row-mode reads), **row-capable adapter** (adapter implementing
`execute_rows`), **pinned value** (ctx-derived column the LLM cannot
supply).

## 3. Model layer

```python
class Entity(BaseModel, frozen=True):
    name: str                         # business noun: "Order"
    cubes: list[str]                  # cubes[0] is the primary cube
    key: str | None = None            # "orders.id"; None => vocabulary-only
    fields: dict[str, str] = {}       # local -> qualified rename map
    list_filters: list[str] = []      # qualified dims allowed in EntityList.where
    default_order: str | None = None  # "orders.created_at desc"
    # prompt vocabulary (unchanged from staged code)
    description: str = ""
    display_name: str | None = None
    questions: list[str] = []
    keywords: list[str] = []
    metadata: dict[str, str] = {}

class MutableEntity(Entity, frozen=True):
    target_cube: str                              # must be in cubes; v1 single cube
    operations: frozenset[Op]                     # insert/update/delete/upsert
    mutable_fields: dict[str, MutableField]
    pinned_values: dict[str, CtxRef] = {}         # col -> AuthContext attr
    predicate_targeting: bool = False             # opt-in beyond PK (D3)

class MutableField(BaseModel, frozen=True):
    type: FieldType
    required: bool = False     # for insert
    nullable: bool = True
    immutable: bool = False    # insert-only; update refuses
```

Rules:
- `key is None` ⇒ entity is vocabulary-only: valid in the catalog, used
  by prompt renderers, but `compile_fetch`/`compile_list` refuse it.
- `MutableEntity` requires `key` (PK targeting needs it).
- The staged `MutableEntity` *builder* is dropped; the name is taken by
  the subclass. (No other model class has a builder; `model_copy` covers
  the use case.)
- Catalog gains `entities: list[Entity]` — one list; `isinstance` gates
  writes. Construction-time validation: cubes resolve; `key`,
  `fields`, `list_filters`, `default_order` resolve to real dimensions;
  `target_cube ∈ cubes`; mutable field names exist on `target_cube`;
  pinned columns are not in `mutable_fields`.

## 4. Read path

### Specs

```python
class EntityFetch(BaseModel, frozen=True):
    entity: str
    key: str | int
    fields: list[str] | None = None   # subset of entity fields

class EntityList(BaseModel, frozen=True):
    entity: str
    where: dict[str, Value | list[Value]] = {}  # allowlist-checked, eq/in
    time_range: tuple[str, str, str] | None = None  # (time_dim, start, end); dim must be allowlisted
    order: str | None = None                    # defaults to entity.default_order
    limit: int = 50                             # capped by catalog policy
    cursor: str | None = None                   # opaque; see Pagination below
```

### Compile pipeline and RowPlan (D8)

Entity reads lower to the existing internal IR, then derive the public
adapter contract from it:

```
compile_fetch / compile_list
  ├─ lower  → LogicalPlan (ungrouped mode, internal — logical.py)
  │     ├─ SQL backends: existing emitter → sql
  │     └─ reuses: join flattening, rollup/partition transforms, refusals
  └─ derive → RowPlan (frozen Pydantic, serializable, versioned)
        └─ custom adapters: execute_rows(RowPlan)
```

`LogicalPlan` stays internal (it carries live `Cube`/field objects and
arbitrary `BoolExpr` trees; it is not a wire contract and must stay free
to evolve). `RowPlan` is the projection adapters see — the *type*
enforces the restricted predicate vocabulary:

```python
class RowPred(BaseModel, frozen=True):
    column: str
    op: Literal["eq", "in", "time_range"]   # D10: ranges on time dims only
    param: str                              # bind-parameter name

class RowPlan(BaseModel, frozen=True):
    version: int                       # contract version, starts at 1
    source: SourceRef                  # cube name / table / backend
    columns: list[str]                 # post-rename output column names
    predicates: list[RowPred]          # user filters + key + decoded cursor
    scope_predicates: list[RowPred]    # structured scope (SQL fragments refuse, §auth)
    order: list[tuple[str, Literal["asc", "desc"]]]
    limit: int
    cursor: str | None                 # custom backends only: passes through verbatim

compile_fetch(f: EntityFetch, catalog, ctx) -> CompiledEntityQuery
compile_list(l: EntityList, catalog, ctx)  -> CompiledEntityQuery
# CompiledEntityQuery: plan: RowPlan, sql: str | None, params, columns
```

Golden tests assert the SQL path and the plan path return identical
records for the same spec (M3's toy adapter exists for exactly this).

### Pagination (D9)

Cursors are opaque to clients and owned by the executing layer:

- **SQL backends**: keyset. `compile_list` appends `entity.key` to
  `default_order` as a tiebreaker (total order guaranteed), encodes
  `(order values…, key value)` into the token, and decodes an incoming
  token back into keyset predicates (`WHERE (o, k) > (:o, :k)`) at
  compile time. Decoding is parsing, not I/O — compile stays pure.
- **Custom backends**: the token rides `RowPlan.cursor` verbatim; the
  adapter interprets and mints it (REST APIs have native cursors).
  Compile never inspects it.

### Auth portability rule

Scope predicates are injected into the plan at compile time (bypass-proof,
as on the analytic path). For **SQL backends**, raw `security_sql`
fragments are fine — they render into the WHERE as today. For **custom
backends**, every scope predicate must be structured; a raw SQL fragment
on a custom-backend entity is a **compile-time refusal** (same posture as
federation refusals). Add a lint rule flagging entities whose cube scope
is non-portable.

### Multi-cube fetch shape

`cubes[0]` is the primary; remaining cubes must be reachable via
many-to-one joins from it (existing join graph). Fetch flattens them into
one record (the `LogicalEntity` read path from the SaaS notes).
Constraint: multi-cube entities are **SQL-backend-only**; a custom-backend
entity must be single-cube (the adapter contract stays "one table-shaped
source").

### Engine

```python
class RowCapableAdapter(Protocol):        # semql-engine
    def execute_rows(self, plan: RowPlan) -> AdapterResult: ...
```

SQL adapters get a default implementation (execute `plan.sql`). A REST/KV
adapter maps structured predicates to query params / key lookups. This is
the entire non-SQL extension story: implement one method, register the
adapter under the cube's backend name.

## 5. Write path

### Spec

```python
class SemanticMutation(BaseModel, frozen=True):
    entity: str                       # resolved to MutableEntity.target_cube
    operation: Op
    values: dict[str, Value] = {}     # validated against mutable_fields
    pk: dict[str, Value] | None = None
    where: dict[str, Value | list[Value]] | None = None  # only if predicate_targeting
```

Exactly one of `pk`/`where` for update/delete; neither for insert;
`where` refused unless the entity opted in (D3).

### Compile

```python
compile_mutation(m, catalog, ctx) -> CompiledMutation
# sql, params          — single parameterised DML statement
# preview_sql, params  — SELECT of affected rows, same WHERE incl. scope
# affects: list[str]   — tables touched
```

Guarantees:
- Scope predicates are injected into every UPDATE/DELETE WHERE.
- INSERT/UPSERT: `pinned_values` are filled from `ctx`; if the LLM
  supplies a pinned column in `values`, compile **fails** (not silently
  overwritten — surface the attempt).
- `values` validated: keys ⊆ `mutable_fields`, required fields present on
  insert, `immutable` fields refused on update, types coerced/checked.
- `operation ∈ entity.operations`.
- DML is template-generated from the validated spec; no LLM text reaches
  SQL.

### Gates (all must pass)

1. `Catalog.allow_mutations: bool = False` — global hard gate.
2. Entity is a `MutableEntity` and operation is in its `operations`.
3. Viewer role policy (same policy mechanism as cube visibility).
4. Predicate targeting only if `predicate_targeting=True`.

### Confirm loop

Library stance unchanged from S8: SemQL compiles and provides
`preview_sql` + `affects`; the confirm/execute loop belongs to the
caller. `semql-mcp` should implement two-step on top: `mutate_<entity>`
with `confirm=false` (default) executes the preview and returns rows +
count; `confirm=true` executes the DML. No server-side token state in v1.

### Custom backends

v1 mutations are SQL-backend-only. A `MutableEntity` whose target cube is
on a custom backend is a catalog construction error. (A future
`execute_mutation(plan)` adapter method is the obvious extension; out of
scope now.)

## 6. MCP surface (D6)

Per-entity tools (default), generated like `query_<cube>`:

- `get_<entity>(key)` — entities with `key`
- `list_<entity>(<list_filters as typed params>, limit, cursor)`
- `mutate_<entity>(operation, values, pk|where, confirm)` — `MutableEntity`
  only, hidden unless `allow_mutations` and viewer passes the role gate

Config flag collapses to three generic tools
(`get_entity(name, key)`, `list_entity(...)`, `mutate(...)`) for very
large catalogs; future: auto-collapse when I12 prompt budget is exceeded.

## 7. Interactions with existing subsystems

- **Cost (I1)**: fetch/list get trivial estimates (key lookup / limit-bounded
  scan); mutations are exempt from QueryBudget but counted by observability.
- **Cache (P7)**: row-mode reads are **not cached** in v1 (freshness
  expectations differ from analytic reads); revisit with TTL-per-entity.
- **Observability**: `on_execute` fires for entity reads and mutations;
  mutation events include `affects` and row counts.
- **Federation**: entities never federate — single backend by
  construction. Cross-backend entity composition is refused.
- **Diff (I3)**: `CatalogDiff` covers entities; widening `operations` or
  adding `predicate_targeting` is a **breaking-attention** change class.

## 8. Multi-cube writes (designed, not built)

Reserved shape, so v1 names don't paint us in:

```python
class EntityMutation(BaseModel, frozen=True):   # future
    entity: str
    operations: list[SemanticMutation]          # ordered, per-cube

compile_entity_mutation(...) -> CompiledEntityMutation  # list of DML + previews
```

Engine executes the list in **one transaction per backend connection**;
entities whose write cubes span backends are refused (no distributed
transactions, ever). `MutableEntity` would gain
`write_cubes: dict[str, frozenset[Op]]` replacing `target_cube`; v1's
`target_cube` is the degenerate single-entry case, so the migration is
additive.

## 9. Disposition of staged code

- `Entity` value type: **keep**, extend with `list_filters`,
  `default_order`, `key`-implies-fetchable rule.
- `MutableEntity` builder: **drop** (name reassigned to the subclass);
  delete `test_mutable_entity.py`, keep `test_entity.py` mostly intact.
- `reproduce_entity.py`: delete (debug artifact).

## 10. Milestones & success criteria

Each milestone is red/green: tests written first, all listed criteria
verified before moving on.

- **M1 — model + catalog wiring.** `Entity`/`MutableEntity`/`MutableField`
  as in §3; `Catalog.entities` with construction-time validation.
  ✓ invalid refs fail construction with actionable errors; vocabulary-only
  entities accepted; builder removed.
- **M2 — read compile.** `EntityFetch`/`EntityList`, `RowPlan`,
  `compile_fetch`/`compile_list`, SQL rendering, scope injection,
  allowlist enforcement, portability refusals + lint rule.
  ✓ golden SQL per dialect; non-allowlisted filter refused; custom-backend
  entity with `security_sql` scope refused at compile.
- **M3 — engine execution.** `RowCapableAdapter`, default SQL impl,
  a toy in-memory table-shaped adapter as the reference custom backend.
  ✓ same EntityFetch returns identical records via Postgres adapter and
  toy adapter; scope honored on both.
- **M4 — mutations.** `SemanticMutation`, `compile_mutation`, gates,
  pinning, preview.
  ✓ all four gates individually block; pinned-column injection attempt
  fails loudly; preview WHERE ≡ DML WHERE (asserted structurally);
  update/delete without pk/where refused.
- **M5 — MCP tools.** Per-entity + generic modes, two-step confirm.
  ✓ tool schemas reflect field types; mutation tools absent when
  `allow_mutations=False` or role gate fails; confirm=false never
  executes DML.

## 11. Out of scope (v1)

Write pool / audit table; server-side confirmation tokens; multi-cube
writes (designed only, §8); mutations on custom backends; cursor-stable
snapshot isolation for lists; caching of row-mode reads; range
predicates on non-time dimensions ("orders over $100" routes to the
analytic layer, which already supports `ungrouped=True` row listing).

## 12. Resolved questions (2026-06-12)

1. **Cursors** → keyset, executor-owned, opaque (D9, §4 Pagination).
2. **Time ranges in lists** → yes, time dimensions only, allowlist-gated
   (D10).
3. **RowPlan ↔ LogicalPlan** → derive, don't subclass or duplicate (D8,
   §4 Compile pipeline). Key facts that decided it: `ungrouped=True`
   row-listing already exists in the IR (`logical.py` `to_logical_plan`
   step 4), and `LogicalPlan` carries live model objects — it is
   explicitly an internal middle representation, not a wire contract.

## Appendix A — Implementation status & review notes (2026-06-13)

Review of the codebase against this spec. Records what exists, the
gaps in the existing stub, and concerns to resolve in the spec before
code lands.

### A.1 Status: implementation landed

As of mid-June 2026, milestones M1–M5 have all shipped. The current
shape — for the code-as-written today, not as this spec proposed — is:

- `Entity`, `MutableEntity`, `MutableField` in `semql.model`
  (`packages/semql/src/semql/model.py:1859, 1844`).
- `EntityFetch`, `EntityList`, `RowPlan`, `RowPred`, `SourceRef`,
  `CompiledEntityQuery` in `semql.rows` (`packages/semql/src/semql/rows.py`).
- `SemanticMutation`, `CompiledMutation`, `compile_mutation` in
  `semql.mutate` (`packages/semql/src/semql/mutate.py:58, 69, 176`).
- `compile_fetch` / `compile_list` (`semql.rows:292, 345`).
- `Catalog.entities`, `Catalog.allow_mutations`, `Catalog.fetch`,
  `Catalog.list_rows`, `Catalog.mutate`, `Catalog.compile_collect_all`
  (`packages/semql/src/semql/catalog.py`).
- Per-entity MCP tool factories `get_<entity>`, `list_<entity>`,
  `mutate_<entity>` in `semql-mcp/server.py:863, 902, 979`.
- New types `Op`, `CtxRef`, `FieldType` defined and exported
  (`semql.__init__:226-294`).
- All re-exported from `semql.__init__`.

The audit table below is kept for the historical record — it shows what
the codebase looked like when this spec was drafted, not what it looks
like now.

| Spec component | In code? (as drafted) | In code? (as of 2026-06-15) |
|---|---|---|
| `Entity` (read-only vocabulary model) | partial — `model.py` stub, see A.2 | landed; catalog-validated |
| `MutableEntity`, `MutableField` | not built | landed (`model.py:1844, 1859`) |
| `EntityFetch` / `EntityList` / `RowPlan` / `RowPred` | not built | landed (`rows.py:59, 71, 102, 116`) |
| `compile_fetch` / `compile_list` | not built | landed (`rows.py:292, 345`) |
| `SemanticMutation` / `compile_mutation` | not built | landed (`mutate.py:58, 176`) |
| `Catalog.entities`, `Catalog.allow_mutations`, gates | not built | landed (`catalog.py`) |
| `RowCapableAdapter.execute_rows`, MCP `get_/list_/mutate_` tools | not built | landed (`server.py:863, 902, 979`) |
| assumed new types `Op`, `CtxRef`, `FieldType`, `max_list_limit` | not built | landed (`model.py:89, 152, 1823`) |

The "still open" A.4 concerns below (six of the seven) remain
substantive design questions even though the implementation exists.

### A.2 Critique of the existing `Entity` stub

> Historical note (A.2 was written 2026-06-13, before M1 landed).
> Items 1 and 2 are now closed by the catalog wiring milestone; item 3
> was a forward-looking note that is moot now that the public surface
> has stabilised.

1. **The docstring overstates validation (foot-gun).** `Entity`'s
   docstring claims "The Catalog validates the entity's references at
   construction time so callers can trust the surface." It does not —
   `Catalog` has no `entities` field and never sees an `Entity`. So
   `Entity(name="X", cubes=["does_not_exist"], key="bogus.col")`
   constructs cleanly and silently. The only validation is
   self-contained format checks (non-empty `cubes`, `cube.dim` shape).
   `test_entity.py` admits "Catalog wiring lives in…" — but it lives
   nowhere. M1 fixes this; until then the docstring should not promise
   it.
   *Closed 2026-06-15:* `Catalog._validate_entities`
   (`packages/semql/src/semql/catalog.py:435`) wires the validation the
   docstring promised.
2. **Dead aspirational comment.** `Entity._check_grounding` says "The
   Catalog wraps the entity in a copy-and-replace path when dedupe
   matters for prompt hashing" — no such path exists, so keyword dedup
   silently does not happen for entities.
3. **Already in the public `__all__`.** `Entity` is a v1 surface
   commitment before it does anything; when M1 lands it gains
   catalog-validation semantics — a behaviour change on an exported
   type. Acceptable pre-v1, but note it in the API-break log.

### A.3 Feasibility: the substrate already exists

The design is buildable as written because it leans on primitives that
already work:

- **Row reads** lower to `ungrouped=True` `LogicalPlan` — implemented
  in `logical.py` (confirms D8).
- **Scope injection** (the bypass-proof part) is exactly
  `_CompileEnv.wrap_for_tenancy` in `compile.py` — reusable verbatim
  for entity reads and mutation WHEREs.
- **Per-entity MCP tools** clone the existing `_make_query_cube_tool`
  pattern in `semql-mcp`.
- The bind-parameter discipline (`_CompileEnv.bind`) for never-inline
  values already exists.

### A.4 Design concerns to resolve before building

The design is security-first and largely excellent (four independent
opt-in gates, PK-default + predicate opt-in, pinned ctx values, custom-
backend structured-scope refusal, preview/confirm in the recipe, no
distributed transactions). Open concerns:

1. **Wire the post-compile statement-shape guard — and invoke it.**
   `safe.py:is_read_only_statement` ships today but is *never called*
   on the read path. The mutation compiler needs the analogous guard
   (assert the DML is exactly one INSERT/UPDATE/DELETE, WHERE present
   for update/delete) and it must actually be invoked, not just
   defined. The "exactly one of pk/where" rule (§5) already prevents
   WHERE-less DML structurally — keep it; belt-and-braces it.
2. **Bind, never inline — as a hard, tested invariant.** This codebase
   already inlines filter values as SQL literals in `federate.py`
   (`_lit`, quote-doubling only). The `RowPlan` / mutation path must
   route every value through bound parameters. §5 says "parameterised
   DML"; make it an asserted, tested invariant given the contrary
   precedent.
3. **INSERT scope cannot use a WHERE.** A scoped/tenancy cube must
   *force-pin* its discriminator column on INSERT, or an INSERT can
   write a row the viewer cannot read back. `pinned_values` is the
   mechanism; add the rule "scoped/tenancy target cube ⇒ discriminator
   must be pinned on insert" as a construction/compile **refusal**.
4. **Preview/confirm TOCTOU.** With no server-side token (v1, §11),
   `confirm=true` recompiles and may affect a different row set than
   the previewed count approved. Return the *actual* affected count
   (driver rowcount / `RETURNING`) and surface a mismatch; document the
   window loudly.
5. **Cap LLM-driven predicate mutations.** Predicate-targeted
   UPDATE/DELETE is the single most dangerous op in the library. Add a
   max-affected-rows cap (refuse when preview exceeds a threshold unless
   explicitly raised), on top of the existing gates.
6. **Mutations must invalidate the analytic read cache.** The
   `semql-engine` cache keys on SQL+params with no table-level
   invalidation; after a write, stale aggregates persist. §7 (Cache)
   should cover post-write invalidation of overlapping read-cache
   entries.
7. **Loose ends.** `EntityList.limit` is "capped by catalog policy" but
   no such field exists (define `max_list_limit` or similar);
   `Op` / `CtxRef` / `FieldType` are new types to add; multi-cube fetch
   flattening must reject one-to-many spans (reuse `compile.py`'s
   `_check_fan_out`) or a "fetch one" returns duplicate rows.

### A.5 "No second library" goal — met

Read/write *compile* sits in `semql` (core), execution in
`semql-engine`, tools in `semql-mcp`. A user wanting MCP writes installs
exactly the packages they already install for reads — no new
dependency. The only philosophical strain is that mutations expand
core's scope past "the SQL is the product / not an ORM"; `compile_mutation`
as a pure `spec → DML + params` function is a legitimate primitive and
fits, but call it out when touching PHILOSOPHY.md.

### A.6 Recommended first step

Build **M1 (model + Catalog wiring)** first: add `Catalog.entities`,
validate refs at construction, add `MutableEntity` / `MutableField`,
delete the dead grounding comment, and correct the `Entity` docstring.
Small, self-contained, unblocks the rest, and fixes the stub's false
validation claim (A.2).
