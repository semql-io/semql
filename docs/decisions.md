# Design decisions

Pinned answers to recurring "should we add X?" questions. Each entry
records the *reason* for the call so a future revisit can weigh
whether the constraints have changed.

The format is loosely an ADR — context, decision, consequences —
condensed to a paragraph each.

---

## D1. PyYAML — no, not in core

**Context.** Some BI tooling (Cube.js, dbt, LookML) loads cube
definitions from YAML files. The question recurs: should `semql`
ship a YAML loader, or expose one through an optional extra
(`pip install semql[yaml]`)?

**Decision.** Neither. Python is the native catalog language. A
YAML loader sits outside core and outside extras — if and when
demand is real, it ships as a separate package (`semql-yaml`).

**Why.** PHILOSOPHY.md is explicit: "Python is the native language
for cube definitions. Type safety, refactoring, and testing come
free." A YAML loader inside core invites the long tail of dynamic
loading concerns (schema validation against a remote registry,
templating, hot reload) and dilutes the type-checker payoff. The
out-of-tree shape preserves "dependencies you don't need should
cost nothing to avoid."

**Revisit.** If three independent users ship YAML loaders to bridge
non-Python services, and the loaders converge on a roughly common
schema, accept it as `semql-yaml`.

---

## D2. mypy and pyright — keep both, for now

**Context.** Both run under `just typecheck`. They overlap heavily;
neither subsumes the other. The dev loop runs ~1s slower than it
would with just one.

**Decision.** Keep both pre-v1. mypy stays the structural backbone
(strict mode catches the classic generics / variance bugs); pyright
catches more in narrow inference corners (Pydantic field
resolution, `runtime_checkable` Protocol drift). Together they have
caught real bugs that one alone missed during this codebase's
build-out — the cost is worth the redundancy.

**Why.** Pre-v1 is when type discipline pays the most: the surface
is moving, and a single bad inference can ripple into the public
API. Once the surface freezes at v1, the marginal value of the
second checker drops and we pick one.

**Revisit.** At v1 cut. Drop pyright if mypy's gaps haven't bitten
in the prior six months.

---

## D3. Fluent Interface on top of Pydantic — no

**Context.** Could we layer a builder-style `Cube.named("orders")
.with_table("orders").with_measure(...)` API on top of the
Pydantic constructor? Some users prefer chained-call ergonomics.

**Decision.** No. Pydantic kwargs are the only catalog-authoring
API. There is no fluent layer, no shorthand factory, no DSL.

**Why.** Two reasons. First, kwarg construction with type hints +
default values is already short — every "extra" character is a
catch by the type checker. Second, a fluent layer is a parallel
public surface that has to be kept in sync with the data model on
every change; doubling the maintenance cost of every new field is
not worth the cosmetic gain. PHILOSOPHY.md: "Composes with your
stack — it does not own it."

**Revisit.** If a downstream tool needs a callable / chainable
shape (e.g. a no-code UI builder), they layer it themselves on top
of `Cube(...)`. Don't pull it into core.

---

## D4. Catalog value types are frozen Pydantic models

**Context.** Should `Measure`, `Dimension`, `Cube`, `SemanticQuery`,
etc. be frozen?

**Decision.** Frozen. `model_config = ConfigDict(frozen=True)`
everywhere except `Cube` itself (which has too many fields and
internal cross-validation to lock down right now).

**Why.** Catalog and spec objects are value types in the
Evans/Fowler sense — equality is structural, identity is
irrelevant, and mutation invites consistency bugs the compiler
can't catch (rebinding `query.measures` after compile started would
produce a result that disagrees with the query the caller built).
Frozen is the default for value objects.

**Revisit.** Only if a real performance scenario emerges where
copy-on-modify dominates. The Pydantic v2 `model_copy(update=...)`
shape covers the common "tweak one field" need without re-opening
mutation.

---

## D5. `having` stays; `Measure.filter` is a different feature

**Context.** The ktx port plan locked "No HAVING — aggregates only
via `Measure.filter`", but `SemanticQuery.having` already exists on
main: exported, federated (`MergeSpec.having`), mapped by the SQL
parser, property-tested. The two looked like substitutes.

**Decision.** Keep both, with distinct jobs. `having` is the
*query-time* post-aggregate predicate — the LLM invents the
threshold per question ("groups where `sum(revenue) > 1000`").
`Measure.filter` (lands with the chasm-trap milestone) is
*catalog-time* conditional aggregation (`SUM(CASE WHEN approved …)`),
authored once by the catalog owner. The locality emission path is
scoped accordingly: no WHERE/HAVING auto-classifier; `Measure.filter`
applies inside per-group CTEs, `having` at the outer SELECT.

**Why.** An LLM cannot express an ad-hoc threshold through a catalog
field — removing `having` would delete real query-time
expressiveness, not consolidate it. The original "No HAVING" decision
is read as a constraint on the locality path's internal
classification, not as a directive to remove the public field.
(Maintainer-confirmed 2026-06-12; see
`docs/specs/roadmap-reconciliation-2026-06.md` §R1.)

**Revisit.** If `Measure.filter` plus `InlineDerived` turn out to
cover every observed `having` use in practice, re-open the
consolidation question before the v1 freeze — not after.

---

## D6. Drop `PolarsMergeEngine` — one merge implementation, not two

**Context.** Review defect A3: `PolarsMergeEngine` hand-reimplements
the merge in Polars and had drifted from the canonical DuckDB merge
SQL — it silently dropped cross-partition filter literals whose op it
didn't recognise (`in`/`not_in`/`is_null`/`not_null`/`contains`) and
read only `vals[0]`, so the *same* `FederatedPlan` returned different
rows depending on the merge engine. The fix was small, but the
question A3 actually poses is whether a parallel merge implementation
should exist at all.

**Decision.** Remove `PolarsMergeEngine` (engine, `merge`
subpackage, and contract tests — ~775 LoC). The DuckDB merge SQL
(`FederatedPlan.merge.sql`) is the single source of truth. The
generic `MergeEngine` / `AsyncMergeEngine` plug-in protocol stays —
a caller can still register a custom merge engine — but the project
ships exactly one implementation.

**Why.** Its stated reason to exist — "merge without a DuckDB
dependency at execute time" — was already false: `semql-engine`
hard-depends on `duckdb` (`engine.py:35`) and `Engine.__init__`
opens a `duckdb.connect(":memory:")` regardless of the merge engine.
`polars` was never declared in any `pyproject` (a phantom dep) and
the advertised `semql-engine[polars]` extra didn't exist. So the
engine bought no dependency reduction while adding a second
hand-written copy of the merge semantics — the exact divergence
hazard A3 is, and structurally the same debt as B1's parallel
federate compiler. Deleting it makes the A3 *class* of bug
impossible rather than merely fixing this instance.

**Revisit.** If a DataFrame-native merge runtime becomes a real
requirement (e.g. an Arrow/Polars-only deployment that genuinely
drops DuckDB), reintroduce it behind the existing `MergeEngine`
protocol — but gate it on a differential harness that runs every
`FederatedPlan` through both engines and asserts identical rows, so
it can never silently diverge again. (Maintainer-confirmed
2026-06-12.)

---

## D7. Time-window ranges are half-open and compared by instant

**Context.** Review defect A2. `TimeWindow.range`'s docstring said
"Inclusive (start, end)", but the compiler emits `dim >= start AND
dim < end` — a half-open `[start, end)` window — and the
time-partition router (`_ranges_intersect`) compared endpoint
*strings* lexically. The two questions A2 forces: is the window
inclusive or half-open, and how are endpoints ordered?

**Decision.** Half-open `[start, end)` is canonical — the emitted SQL
is the source of truth and already half-open; the docstring was the
bug and was corrected. Range endpoints are compared by *instant*, not
text: a shared `spec.parse_instant` parses each ISO-8601 endpoint to
an aware `datetime`, and the router and the `TimePartitionedSource`
range-ordering validator both compare the parsed values. Naive
(offset-less) timestamps are read as **UTC** so a naive endpoint stays
comparable with an offset-bearing one.

**Why.** Lexical comparison only *coincidentally* matches chronological
order — for zero-padded, same-offset ISO-8601. The instant two
endpoints differ in UTC offset (or precision), byte order diverges
from instant order: a query window in `-05:00` whose rows all fall in
the post-boundary physical source was routed to the *pre*-boundary
table and silently returned empty. Comparing instants is the only
comparison that matches what the `>= / <` filter actually selects.
Half-open is also what the rest of the model already assumes
(`TimePartitionedSource` docstring) and what review item B9 recommends
standardising on everywhere.

**Revisit.** The naive-is-UTC reading is a pragmatic default, not a
timezone model. When per-cube/per-dimension timezone semantics land
(B9), `parse_instant`'s default should defer to the declared zone, and
endpoints should be parsed at construction (so a malformed or
ambiguous timestamp is refused when the cube/query is built, not at
route time). (Maintainer-confirmed 2026-06-12.)

---

## D8. Cross-cube type coercion is refused, with `Dimension.coerce_to` opt-in

**Context.** Review item I10 (promoted to the W1 correctness tier by R6).
A federated bridge join equates two cubes' keys with a bare `a.k = b.k`;
when the keys' declared `Dimension.type` differed (a `uuid` order key vs
a `string` customer id), the merge engine coerced one side silently,
which can drop or invent matches. That's a refusal-over-omission
violation. The question: refuse, coerce, or warn — and where.

**Decision.** Refuse at compile time with `FederationError(reason=
"cross_cube_type_coercion")`. The escape hatch is `Dimension.coerce_to:
DimTypeLiteral | None` — a dimension declares the *additional* type it
is willing to be compared as. A join is allowed when the two keys share
at least one acceptable type, where a key's acceptable set is
`{type} ∪ {coerce_to}`. `coerce_to == type` is itself a construction
error (it coerces nothing). The opt-in is rendered in the planner
prompt next to the dimension's `type`.

**Scope.** The refusal covers the **federated bridge path only**
(`federate._parse_bridge`), where SemQL holds the join keys as
structured, typed dimensions. Same-backend joins specify their key in a
raw-SQL `on` clause whose column types SemQL cannot see — that's the
raw-SQL escape hatch (B2), and it stays uncovered until an expression
IR exists. The check sits in `_parse_bridge`, the single funnel both the
distributive and raw_rows merge paths route through, so neither can
emit a coercing join.

**Why.** "Wrong results are the only unacceptable outcome." A silent
type coercion in a join key is precisely a wrong-rows generator, and
unlike a missing label it's invisible in the output. Making the catalog
author write `coerce_to` turns an accident into a decision. Type
mismatch isn't representable for every case yet — there's no date-vs-
timestamp distinction in `DimTypeLiteral` (B9) — so I10 catches the
mismatches the type system can currently express (uuid/string,
number/string, …) and grows as the type vocabulary does.

**Revisit.** When same-backend joins gain a structured key
representation (B2/B3 — QualifiedRef + expression IR), extend the same
check to them. When the temporal model splits date from timestamp (B9),
those become catchable mismatches too. (Maintainer-confirmed
2026-06-12.)

---

## D9. `Join.kind` is honoured at emission (RESOLVED)

**Context.** W2 (review B1) listed "`Join.kind` honoured" — the emitter
hardcoded `join_type="left"` (`compile.py`) while the plan carries a
`kind` (`logical.py` sets `left` for cubes in `query.left_joins`, else
`inner`, matching the spec doc "Cubes to LEFT JOIN instead of INNER").
The naive fix — passing `plan_join.kind` to the emitter — was originally
deferred because `build_join_graph` rooted the FROM clause at
`touched[0]`. For the `left_joins` *spine* case that could land the
left-joined cube on the FROM root (`facts → spine`), so honouring `kind`
would emit a wrong-rooted INNER join and drop the very rows the spine
feature exists to keep (e.g. employees with zero punches).

**Decision.** Unblock the root-selection problem directly rather than
waiting for the full W3 Dijkstra rebuild. `build_join_graph` now roots at
the first touched cube that is **not** in `left_joins`
(`next((c for c in touched if c.name not in left_set), touched[0])`), so
every left-joined cube lands on the `right` side of an edge and the plan
stamps it `kind="left"`. The emitter then reads `join_type=plan_join.kind`.
For the no-`left_joins` default, the root is unchanged (`touched[0]`) and
every edge is `kind="inner"` — so a plain multi-cube join now emits
**INNER JOIN**, the documented intended default.

**Why.** The deferral hinged on one defect — wrong FROM root for the
spine case — not on the whole join graph being untrustworthy. A targeted
re-root fixes exactly that defect without pre-empting W3's broader
Dijkstra/`JoinPath` work (the edges and direction are otherwise
unchanged). Honouring `kind` is now a correctness improvement, not a
trade of one wrong result for another: spine queries keep their
zero-match rows (LEFT) and ordinary joins stop silently widening to LEFT.

**Behaviour change.** Default multi-cube joins flip LEFT→INNER. Blast
radius was two assertions (`test_compile.py` default-join test + one
`test_snapshots.ambr` line); explicit-`left_joins`, time-spine, and
federation paths are unaffected (verified: full suite green, 1714
passed). (Resolved 2026-06-13.)

**Update — Dijkstra rebuild landed (ktx M2 / C6, 2026-06-14).**
`find_join_path` is now a weighted, bidirectional Dijkstra (`logical.py`):

- *Bidirectional reachability.* The join graph is undirected for path
  finding — a join declared on one cube resolves from either side. This
  fixes a real defect: a child-cube measure aggregated by a parent-cube
  dimension (e.g. `SUM(order_items.qty) BY orders.region`, where only
  `orders` declares the `one_to_many` edge) previously raised
  `JoinPathError`; it now resolves via the reverse edge.
- *Fan-out weighting.* `one_to_many` traversals cost 10×, `many_to_one` /
  `one_to_one` cost 1× (`_edge_weight`, the single fan-out model shared
  with `_check_fan_out`). Given multiple paths, the planner prefers the
  one that doesn't multiply rows even if it has more hops.
- *Transit prohibition.* `forbid_transit` (the `left_joins` set) lets a
  path *end* at a LEFT-joined cube but never pass *through* one, so the
  chasm-trap refusal (a non-left cube reachable only behind a left-joined
  cube) is preserved. The root is exempt (it is a start, not a transit).

**Behaviour change.** Spine→facts edges now resolve *without* `left_joins`
as an INNER JOIN (the anti-join's LEFT JOIN still requires `left_joins`);
`test_left_joins.py::test_spine_to_facts_without_left_joins_*` was updated
to assert the INNER resolution. The `JoinPath` value type + `is_ambiguous`
warning surface (the rest of C6) remains a follow-on. (Verified: full
semql + engine suites green.)

---

## D10. The federation parallel-compiler deletion is deferred (post-W2)

**Context.** W2 (review B1) listed "replace federate.py's ~700-LoC
parallel compiler with a `LogicalPlan` split-point feeding the shared
emitter (killing `_lit` literal inlining)". The split-point primitive
exists (`logical.partition_scans`) and `compile_plan` now trusts a
prebuilt plan, so the load-bearing prerequisite is in place. But the
rewrite itself was deferred.

**Decision.** Treat W2 as functionally complete with the parallel
compiler still in place. The behaviour-affecting W2 goals all landed and
are green: the emitter trusts the plan (`compile_plan` no longer
re-plans — a rewritten scan / pushed-down predicate survives to
emission), one alias convention (`output_alias`), `CompareSplit` is
load-bearing, B6 keys predicate resolution by dimension, the
distributive path lifts the where-tree + segments (the R3 carryovers /
5 parked A4 tests), and `FederatedPlan` is frozen + version-stamped.
`Join.kind` is the only IR-adoption item parked, and that's D9 (W3).

**Why.** Deleting the parallel compiler changes *no behaviour* — every
federation test already passes through it — so it is regression risk
with no user-visible upside, and it is large: `partition_scans` today
gives each backend only its scans/joins; the rewrite must move filter /
segment routing, bridge-key projection, avg-decomposition, measure
routing, and a merge-spec derivation into the plan layer. That is a
multi-step effort of its own, best built incrementally and verified
against the existing federation suite as an oracle, not folded into
W2's tail. Shipping the correctness wins now and doing the refactor
deliberately later is the lower-risk sequencing.

**Revisit.** As its own workstream: extend `partition_scans` to a full
predicate router + bridge-projection injector, derive the merge from
partitioned plans, replace `_lit` inlining with bound params, then
delete `_build_partition_sub_query` / `_emit_merge_sql` and the
raw_rows twins. The existing federation tests are the behaviour oracle —
they must stay byte-stable through the swap. (Maintainer-confirmed
2026-06-13.)

---

## D11. SemQL grows a write path — opt-in, compiled, never LLM-authored SQL

**Context.** PHILOSOPHY.md frames SemQL as a read/compile layer:
`SemanticQuery + Catalog → CompiledQuery`, "running the SQL is the
caller's job." The entities work (`docs/specs/entities.md`, design
agreed 2026-06-12) adds two escape hatches the analytic layer can't
serve: **row-mode reads** (fetch one row by key, list a few by
allowlisted filters — "show me order 42") and **mutations** (insert /
update / delete / upsert). Mutations are a genuine scope expansion —
the layer has only ever emitted `SELECT`. The question is whether SemQL
should grow a write path at all, and if so, under which invariants — so
that adding DML doesn't quietly erode the properties that make the read
path trustworthy.

**Decision.** Yes, but writes are opt-in at every layer and *compiled*,
never LLM-generated. The LLM may construct a `SemanticMutation` spec
(which entity, which operation, which values / target); it never writes
SQL. `compile_mutation` template-generates a single parameterised DML
statement plus a preflight preview `SELECT`, binding every value as a
parameter. Four independent gates must all pass: (1) `Catalog.
allow_mutations` — a global hard gate, **off by default**; (2) the
target is a `MutableEntity` and the operation is in its `operations`
set; (3) the viewer passes the target cube's role policy — the same
door as cube visibility; (4) predicate (`where`) targeting only when the
entity opts in via `predicate_targeting=True`, PK targeting otherwise.
Row-mode reads lower to the existing ungrouped `LogicalPlan`, then derive
a serializable `RowPlan` that SQL backends render and custom (non-SQL)
backends interpret via a row-capable adapter.

**Why this does not violate PHILOSOPHY.** The load-bearing invariants
hold. *Compiler purity*: `compile_mutation` returns a statement + bound
params + preview; it performs no I/O — executing the DML stays the
caller's / engine's job, exactly as for reads. *Bind-never-inline*:
every mutation value binds as a parameter, never a SQL literal.
*Refusal over wrong results*: scope on a write is **fail-closed** — v1
injects only *structured* discriminator tenancy into every UPDATE/DELETE
WHERE (bound, bypass-proof), and a target cube carrying raw
`security_sql` or a `scope` ScopeFn is **refused**, because silently not
enforcing raw scope on a write is worse than refusing it. Refusing these
escape hatches outright wouldn't keep SemQL "read-only" — it would push
text-to-SQL and MCP users to hand-built SQL strings, which is precisely
the unaudited, unparameterised path the project exists to replace. So
the compiler-purity line in PHILOSOPHY.md still reads true and needs no
edit; this entry records *why* the write path is consistent with it
rather than an exception to it.

**Scope.** v1 mutations target exactly one cube (multi-cube writes are
designed in entities §8, built later). PK targeting is the default;
predicate targeting is per-entity opt-in and every compiled mutation
carries a preview. Row-mode reads are SQL-backend in-library; non-SQL
sources (REST, KV) bring a row-capable adapter — no library change.
`Entity` (read-only) is a first-class base; `MutableEntity(Entity)` adds
the write schema and requires a `key`. Reads over a vocabulary-only
entity (`key is None`) are refused at compile time.

**Revisit.** Multi-cube / transactional writes when entities §8 is
built. Raw-scope (`security_sql` / ScopeFn) write targets stay refused
until scope is expressible as a structured, bound predicate the write
path can enforce without trusting opaque SQL. If `allow_mutations`
adoption shows the global gate is redundant with the per-entity
`operations` allowlist, reconsider collapsing the gates — but not before
v1. (Design agreed 2026-06-12; entry drafted 2026-06-13.)

---

## D12. `SemanticQuery` is recursive — tool schemas go through `tool_json_schema()`

**Context.** The cross-backend semi-join primitive (`docs/specs`, landed
2026-06-15) added `SemanticQuery.semi_joins: list[SemiJoin]`, and a
`SemiJoin` carries an inner `source: SemanticQuery`. That self-reference
makes `SemanticQuery` a **recursive** model, so Pydantic's
`model_json_schema()` no longer emits an object-rooted schema — it emits a
bare root `$ref` (`{"$ref": "#/$defs/SemanticQuery", "$defs": {…}}`).
OpenAI, Anthropic, and Bedrock tool-calling all require the tool
`inputSchema` to be object-rooted (`type: "object"` at the top); Bedrock
Converse rejects a root `$ref` on every model family. So the moment the IR
gained a nested-query field, every tool-schema projection that called
`SemanticQuery.model_json_schema()` directly started shipping a schema the
APIs reject — not just the Bedrock adapter, the plain OpenAI/LangChain
projections too.

**Decision.** Core owns both the quirk and the fix.

- `semql.flatten_root_ref(schema)` (in `semql/_schema.py`, exported from
  the package root) is the single implementation: it splices a root `$ref`
  back up to the top level while keeping `$defs` and every *internal*
  (recursive) `$ref` intact, so the self-reference still resolves. It is a
  no-op on an already object-rooted schema and raises on a schema that
  cannot be made object-rooted (a `RootModel` over a union/scalar).
- `SemanticQuery.tool_json_schema()` is the **canonical front door**:
  `flatten_root_ref(cls.model_json_schema())`. Anything that ships a
  `SemanticQuery` schema as a tool spec must call this, never the raw
  `model_json_schema()`. All current projections route through it
  (`to_openai_function`, `to_openai_tools`) or re-flatten on top of it
  (`to_bedrock_converse_tools`).

**Why core, not the prompt layer.** The recursion is a property of a core
type, so the response to it belongs with the type — and putting the
mechanism in core means the projection layers don't each re-implement
flattening (or each forget to). `semql-prompt` and the Bedrock adapter
import `flatten_root_ref` from core; it is not re-exported from
`semql-prompt` (a cross-package re-export the API-doc generator could not
resolve, and an unnecessary second public home).

**Revisit.** If a future IR change removes the recursion (no nested
`SemanticQuery` field), `tool_json_schema()` degrades to a no-op wrapper
over `model_json_schema()` and could be retired — but it is the safe
default to keep regardless, since it costs nothing on a non-recursive
schema and guards against the next nested-query field reintroducing the
trap silently. (Landed with the semi-join primitive, 2026-06-15.)

**Update, 2026-07.** `tool_json_schema()` still contains `prefixItems`
(`TimeWindow.range` / `CompareWindow.range` / `order`'s tuples) and internal
recursive `$ref`s (`BoolExpr` self-reference; `SemanticQuery` <-> `SemiJoin`
through `semi_joins[].source`) — both left untouched by design, per the
decision above. Bedrock Nova's tool-use decoder has since been observed to
424 on both, not just a root `$ref`. Rather than change `tool_json_schema()`
(other decoders tolerate this shape fine, and `where`/`semi_joins` are real
query features some callers need in the schema), `SemanticQuery` gained a
second, stricter projection: `SemanticQuery.llm_json_schema()` (backed by
`semql._schema.to_llm_safe_schema`) rewrites every `prefixItems` node into a
plain `items`-typed array and removes every `$defs` cycle — which, for this
model, means dropping the `where` and `semi_joins` properties, the only
paths into the recursive defs. Its companion,
`SemanticQuery.from_llm_payload(payload)`, builds a canonical
`SemanticQuery` back from a payload shaped to that schema (plain-list
`range`/`order`; Pydantic coerces the lists back to tuples). Use
`tool_json_schema()` by default; reach for `llm_json_schema()` only against
a decoder known to reject `prefixItems` or recursive refs, and only for
queries that don't need `where`/`semi_joins`.
