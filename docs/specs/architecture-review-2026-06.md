# Architecture review — June 2026

Three-lens review (domain model / compiler / engine+ecosystem), judged
against PHILOSOPHY.md's own invariants. Findings are agent-reported with
file:line evidence; items marked ⚠ were empirically reproduced during
review, the rest should be confirmed with a failing test before fixing
(red/green).

> **Status legend.** Each "Resolved …" annotation below is anchored to a
> test file that fails without the fix (or, where the fix is a deletion
> that removes a whole code path, to a test that asserts the new
> behaviour). Items still marked "Still open" or "blocks X" have not
> been verified closed against a test. TODOS.org is gitignored, so this
> spec is self-contained only because the test pointers below exist —
> a new contributor can `git grep` the test names to confirm status
> without an external backlog.

## A. Correctness defects on main (fix before any new feature)

1. ⚠ **`compile_plan` silently drops filters** (`compile.py:2320-2407`).
   It reverse-engineers a SemanticQuery from the plan but copies only
   measures/dimensions/time/compare/order/limit — `filters`, `having`,
   `segments`, `ungrouped`, `left_joins`, `derived_measures`, `aliases`
   are discarded. Reproduced: `via_plan` SQL lacks the WHERE clause
   `via_query` has. The byte-equality test covers only the trivial
   shape. **Blocks**: federation split-point (next ribbon item) and the
   entities spec's D8 (RowPlan derives from LogicalPlan) — both route
   through this path.
2. **Partition routing can silently drop rows.** `TimeWindow.range` is
   documented inclusive (`spec.py:42-44`) but `_ranges_intersect` treats
   it half-open (`partition.py:72-85`), and comparison is *lexical
   string* comparison of unvalidated, unnormalised timestamps
   (`"2024-01-01"` vs `"2024-01-01T00:00:00"`). A query ending exactly
   at a source boundary skips that source's table.
   **Resolved 2026-06-12 (decisions.md D7):** the emitted WHERE is in
   fact half-open (`dim >= start AND dim < end`, `compile.py:1694-1706`)
   and the routing already matched it — so the boundary-skip framing was
   off; the real defect was the *lexical* comparison, which inverts the
   instant order the moment two endpoints carry different UTC offsets
   (a window in `-05:00` whose rows all live in the post-boundary source
   was routed to the *wrong* table and returned empty). Fixed by
   comparing parsed instants (`spec.parse_instant`, naive read as UTC)
   in both `_ranges_intersect`    and the `TimePartitionedSource`
   range-ordering validator, and by correcting the `TimeWindow.range`
   docstring to half-open. Full parse-at-construction + per-cube
   timezone semantics remain B9.
   *Verifying test:* `packages/semql/tests/test_partition_boundary.py`.
3. **PolarsMergeEngine drops filter literals it doesn't understand**
   (`polars_engine.py:260-261`): ops outside eq/neq/gt/gte/lt/lte hit
   `else: continue` inside an OR-disjunction → narrower disjunction →
   missing rows. Also reads only `vals[0]` (`:247`). The DuckDB merge
   path supports in/not_in/is_null/contains, so the same FederatedPlan
   returns different rows per merge engine. Same silent-skip pattern in
   the having loop (`:348`, currently unreachable).
   **Resolved 2026-06-12 (decisions.md D6):** `PolarsMergeEngine` was
   removed rather than fixed — a parallel hand-written merge was a
   standing divergence hazard and its no-DuckDB rationale was already
   false (`semql-engine` hard-depends on duckdb). The DuckDB merge SQL
   is now the single source of truth, so this    bug class is structurally
   gone.
   *Verifying test:* `packages/semql-engine/tests/test_merge_engine_default.py`
   (asserts the DuckDB merge is the only registered merge engine).
4. **5 federate tests fail on main** (acknowledged in TODOS.org as
   pre-existing). The gate is off exactly where the code is most
   fragile. Fix or quarantine.
   *Verifying tests (all five now green):* `test_federate.py`,
   `test_federate_cnf_reuse.py`, `test_federate_merge_spec.py`,
   `test_federate_where_segments.py`, `test_federated_plan_frozen.py`
   in `packages/semql/tests/`. Run `uv run pytest
   packages/semql/tests/test_federate*.py` to confirm.
5. **Engine cache hazards** (`engine.py:184-222`): cached
   ExecutionResult returned by reference (caller mutation poisons the
   cache); list-valued params (BQ ArrayQueryParameter) are unhashable →
   TypeError at lookup; no TTL.
   **Resolved 2026-06-12:** all three fixed. (a) `run` now hands out an
   isolated copy on every read *and* store (`_isolate`); the underlying
   `_execute_uncached` also stopped aliasing `plan.columns` /
   `plan.column_meta`, so a returned result no longer shares state with
   the plan or the cache. (b) `_cache_key` runs param values through
   `_freeze_param` (lists/tuples → tuples, dicts → key-sorted tuples,
   sets → frozensets), so container-valued params hash. (c) optional
   `cache_ttl` expires entries against an injectable monotonic clock.
   Covered by `test_cache_hazards.py` (renamed from
   `test_a5_cache_hazards.py`). AsyncEngine still lacks the
   cache entirely (B7) — unchanged here.
6. **MCP auth is per-server, not per-request** (`server.py:122,199`):
   `compile()` is never passed `viewer=`; lookup tools accept
   client-asserted `viewer_id`/`roles`. Tolerable on stdio; a
   multi-tenant hole on the advertised http/sse transports.
   Contradicts "AuthContext is request-scoped, never global".

## B. Structural debts (the pre-v1 window — impossible to fix after)

1. **IR adoption is half-done and duplicated.** The plan is built but
   the emitter half-trusts it: output-alias collision logic exists twice
   (`logical._col_alias` vs `_CompileEnv._col_name`) and both are
   load-bearing in the same query; CompareSplit prior-range math is
   computed twice (plan node is write-only); `Join.kind` is dead at
   emission (`compile.py:1582` hardcodes left); `apply_partition_to_plan`
   and `apply_rollup_to_plan` are test-only — production uses parallel
   inline mechanisms. **federate.py never adopted the IR at all**:
   ~700 LoC parallel compiler incl. a second private CNF implementation,
   duplicated sub-query builders and merge emitters, merge SQL built by
   f-string with values inlined as literals (`_lit`, against the
   bind-params invariant), no output-column collision handling, `views`
   dropped on the multi-backend path.
   **Mostly resolved 2026-06-13 (W2 stages 2–7):** the alias convention
   is now one helper (`output_alias`/`output_column_collisions`, stage 2,
   which also fixed a latent I7-alias dup-column divergence); CompareSplit
   is load-bearing — the emitter reads `current_range`/`prior_range` off
   the plan, not recomputed (stage 3b); `compile_plan` trusts a prebuilt
   plan and no longer re-plans, so a rewritten scan / pushed-down
   predicate reaches emission (stage 4, the A1 finish); the distributive
   federation path lifts the where-tree + segments into fragments with a
   cross-partition residual in the merge (stage 5, the R3 carryovers — 5
   parked A4 tests now green); `FederatedPlan` is frozen + version-stamped
   (stage 7). `Join.kind` honouring is now resolved (decisions.md D9 — the
   emitter reads `plan_join.kind` and `build_join_graph` re-roots off the
   left-joined cubes; default joins are INNER). Still open:
   the parallel-compiler deletion incl. `_lit` removal (decisions.md D10 →
   its own post-W2 workstream; the split-point primitive `partition_scans`
   and a plan-trusting `compile_plan` are now in place for it).
2. **Raw SQL strings are the default, not the escape hatch.** Nine entry
   points (`BaseField.sql`, `Measure.filter`, `base_predicate`,
   `Join.on`, `security_sql`, `ScopePredicate.sql`, `DerivedTable`,
   `NamedCTE`, `mask_value`). Blocks dialect portability, lineage,
   non-SQL backends (entities D1), structured scope (entities §4 auth
   portability). PHILOSOPHY says "when raw SQL is used, SemQL says so" —
   today silence implies raw SQL.
3. **Stringly-typed refs with divergent conventions.** `"cube.field"`
   split ad hoc at ~40 sites; shape check copy-pasted 4×; some fields
   qualified, some local, some "either" — the author must memorise
   which; typos in `order`/`Rollup` degrade silently.
4. **Join cardinality is modelled but unused.** `Join.relationship`
   exists; nothing detects fan-out — a one_to_many join under `sum`
   silently inflates the measure. The canonical semantic-layer wrong
   result, and the flag to prevent it is already in the model.
5. **Catalog serialisability is aspirational.** Catalog is a plain
   class holding callables (policy, scope_fns, hooks, loaders); no
   model_dump/from_dict/version. No defined CatalogSpec (data) vs
   CatalogRuntime (behaviour) boundary. Also a ~300-line first-error
   constructor while PHILOSOPHY promises a collect-all validate path;
   `to_openai_tools`/`to_langchain_tools` live on the core type.
6. **Predicate resolutions are identity-keyed** (`compile.py:1734-1747`,
   `id(leaf)` side tables). Works only because CNF reuses leaf objects.
   Any transform that copies a Filter (pushdown, pruning, federation
   routing — everything logical.py promises) breaks at emission.
   **Resolved 2026-06-12 (W2 stage 1):** resolution is now keyed by the
   leaf's qualified `dimension` (the field a leaf resolves to depends
   only on which `cube.field` it names), in both `where_leaf_resolutions`
   (`_resolve.py`) and `_CompileEnv._lookup_filter_field`. A copied leaf
   resolves identically to its original — the prerequisite for the
   federation split-point. Pinned by `test_b6_structural_predicate_keys.py`;
   no SQL change (snapshots byte-identical).
7. **Sync/async engine duplication with divergence.** AsyncEngine lacks
   the P7 cache and on_execute hook; shares code by calling unbound
   `Engine` methods with `# type: ignore`. Adapter protocol has no
   capability/extension story; transactions are nonexistent (DBAPI
   adapter never commits — DML would silently roll back).
8. **Error taxonomy fragments.** 47 bare `CompileError` f-strings vs 4
   typed leaves; three uncoordinated code vocabularies
   (exception attrs / ValidationError.code / FederationError.reason);
   `EngineError` is outside SemQLError; MCP flattens structured attrs to
   a string. "Errors serve machines" holds in prose only.
9. **Temporal model thin.** No quarter/year granularity, no timezone
   semantics, no date-vs-timestamp distinction, range endpoints are
   unparsed strings (see A2). Decide half-open `[start, end)` everywhere
   and parse at construction.
10. **Smaller**: no cube-alias uniqueness validation (duplicate SQL
    aliases possible); `model.py` at 1,543 lines and growing; frozen
    models with list fields aren't actually hashable (fixed only on
    Lookup); MutableEntity builder contradicted pinned decision D3
    (**removed 2026-06-13**: the staged builder class + its test are
    deleted, so the `MutableEntity` name is now free for the entities
    spec's read/write subclass; no production code referenced it);
    Entity docstring promises
    catalog validation that doesn't exist yet (entities spec M1 covers
    it); JWKS/httpx network I/O inside the "no I/O" core package;
    X509Mapper collapses `alice@a.com`/`alice@b.com` to viewer `alice`.

## C. Sequencing recommendation

The dependency chain for the entities feature runs straight through the
worst findings:

```
A1 fix (compile_plan honest, hours-days)
  → B1 federation split-point (delete parallel compiler, ~1-2 wk)
  → entities M2 (RowPlan derivation becomes trustworthy)
B7 capability sub-protocols (SupportsRowMode/SupportsMutation, ~1 day)
  → entities M3/M4 (adapter contract lands clean the first time)
A6 per-request MCP auth (~2-3 days)
  → entities M5 (per-entity tools + mutation confirm need it)
```

Suggested order:
1. **Now**: A1, A3 (+ differential merge-engine test, ~1 day, catches
   A3-class bugs mechanically), A2, A4. Each starts with a failing test.
2. **Pre-entities**: B1 (finish IR adoption + federation split-point),
   B6 (structural predicate resolution — also what entities D8 needs),
   B7, A6.
3. **Pre-v1, after entities**: B2 (expression mini-IR with explicit
   `RawSQL` escape hatch), B3 (QualifiedRef type), B4 (fan-out guard),
   B5 (CatalogSpec/Runtime split), B8 (one error contract:
   `code` + payload), B9.

## D. What's genuinely good (don't churn)

Error-message prose discipline; the decisions doc + revisit conditions;
auth architecture (scope-in-subquery, ctx_keys, mask_roles ⊆
required_roles, JWKS none-alg refusal); semantic-aggregation rigour
(REAGG_OK, non_additive, NULLIF ratios); `_resolve.py`'s
diagnostics-not-exceptions walker feeding both fail-fast compile and
collect-all validate; cnf.py; BackendDialect protocol (new backends are
genuinely cheap); CompiledQuery observability surface; dependency
direction across packages (semql never imports engine/mcp; plans are
pure data); per-cube MCP tool generation with Literal enums; prompt.py's
two-segment cache split with the "retrieval can only narrow" invariant;
test architecture shape (plan + SQL snapshots pinning both pipeline
ends) — it just needs the equality matrix widened and the red federate
tests resolved.
