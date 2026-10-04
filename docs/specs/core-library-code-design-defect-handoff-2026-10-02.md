# Core-library code-design review: defect handoff

## Discovery baseline and evidence

- **Discovery commit:** `fc57a9b3714b3fad34fa90201ad5737ddff4f0a5` (`main`, merge of PR #11).
- **Discovery date:** 2026-10-02, during the code-design review preceding this document.
- **Handoff recording timestamp:** `2026-10-02T21:34:26+05:30` / `2026-10-02T16:04:26Z`, captured from the workstation clock. Individual defect-discovery timestamps were not retained; this is the recording time, not a fabricated per-finding discovery time.
- **Review lens:** `/Users/n/npalladium/skills/skills/code-design/SKILL.md`, version 2.7.0, and its `checklists.md`.
- **Scope:** core `semql`, compiler/federation contracts, `semql-engine`, `semql-prompt`, semantic discovery, and `semql-auth`.
- **Discovery status:** all 15 findings were unfixed at discovery. All D01–D15 repairs are now implemented and verified; see [repair completion](#repair-completion). Baseline observations below remain unchanged, not rewritten as current behavior.

The source line ranges below refer to the discovery commit. Relative links resolve from this document to repository files; `#L…` fragments communicate the baseline line range, even if a local Markdown viewer does not support line navigation. After fixes move the lines, use the symbol names and discovery commit to recover the original context.

The review was read-only. In-memory probes used the existing environment through `PYTHONDONTWRITEBYTECODE=1 uv run --frozen --no-sync python -B`. Compiler scenarios executed against DuckDB; auth probes generated synthetic keys and intercepted the HTTP boundary; engine lifecycle probes used in-process adapters. No live databases, identity providers, credentials, dependency installations, source changes, commits, or pushes were involved. Four pre-existing untracked local documents were left untouched.

**Evidence limitation:** the full test suite, linters, and type checkers were not run during this audit. Test files listed below are repair targets, not passing verification claims. Compiler and engine findings include runtime evidence from independent review slices; catalog/model probes and the policy/budget prompt probes were also exercised directly by the coordinating reviewer. The temporary probe scripts were not persisted, so sections give reconstructible scenarios and observed output, rather than pretend to provide byte-identical archived scripts.

## Triage and suggested repair groups

Priority denotes relative repair urgency, not a vulnerability score or a claim about every deployment.

| ID | Priority | Defect | Suggested group |
|---|---|---|---|
| D01 | High | Parsed JWKS keys bypass document TTL | Auth cache lifetime |
| D02 | High | Dynamic policy bypass in prompt/tool discovery | Discovery authorization |
| D03 | High | Viewer-specific fields in invariant descriptions | Discovery authorization |
| D04 | High | Budget trimming deletes protected cube fields | Prompt structure |
| D05 | High | Dense spine drops final intersecting bucket | Compiler time spine |
| D06 | High | Dense spine drops declared derived outputs | Compiler time spine |
| D07 | High | `from_spec(runtime=…)` drops runtime policy | Catalog construction |
| D08 | Medium | Catalog wire round-trip loses operational limits | Catalog construction |
| D09 | Medium | Mutable nested fields change model hashes | Model value semantics |
| D10 | Medium | Comparison splits NULL groups | Compiler comparison |
| D11 | Medium | Comparison result `HAVING` fails at execution | Compiler comparison |
| D12 | High | Merge materialization loses numeric schema | Engine adapter contract |
| D13 | Medium | Async fragment failure orphans sibling work | Engine task ownership |
| D14 | Medium | Stream early-close does not close adapter iterator | Engine resource ownership |
| D15 | Medium | Cached results share mutable cell values | Engine cache isolation |

D02/D03 share the static-versus-viewer-specific boundary. D05/D06 share dense-fill wrapping but have distinct causes. D07/D08 expose catalog state-authority problems. D12 is a cross-adapter contract change; repair every built-in producer and all materialization consumers, not only one merge path. D13/D14 require explicit task and resource ownership, not merely exception suppression.

## D01 — Parsed JWKS keys bypass the document TTL

**Location:** [`auth.py:342–356`, `JWKSVerifier._public_key_for`](../../packages/semql-auth/src/semql_auth/auth.py#L342-L356). Related document-cache refresh and invalidation: [`auth.py:325–340`](../../packages/semql-auth/src/semql_auth/auth.py#L325-L340). TTL documentation: lines 286–287.

**Defect and impact:** `_public_key_for()` returns `_key_cache[kid]` before calling `_fetch_jwks()`, so the document's TTL is never checked for a known key. Requests using that key can continue verifying after it has been removed from the provider's current JWKS. `ttl=0`, documented as disabling caching, does not disable this parsed-key shortcut. Token expiry still applies independently; the defect is stale signing-key trust, not bypass of all JWT validation.

**Reproduction:**

1. Generate a synthetic RSA key, publish its public JWK under a fixed `kid`, and sign an RS256 token for a synthetic subject.
2. Intercept `httpx.get` to return an in-memory JWKS document; replace monotonic time with a controllable clock. Do not patch `_fetch_jwks()` itself, because that would bypass the TTL logic under review.
3. Construct `JWKSVerifier(..., ttl=1)` and call public `verify()` at monotonic time 100.
4. Remove the key from the returned JWKS, advance time to 10000, and verify the same token again.
5. Repeat with `ttl=0`.

**Observed:** both verifications returned subject `test-user`; HTTP-fetch count remained 1. The `ttl=0` variant also accepted the removed key without another fetch.

**Expected:** an expired or caching-disabled verifier must consult current key material rather than return a stale parsed key. A removed key must no longer verify once refresh is required.

**Repair direction:** check document freshness before returning a parsed key; tie parsed-key cache lifetime to the document generation. Invalidate both caches together. Preserve deliberate refresh-on-miss behavior and public `AuthError` semantics.

**Acceptance:** public `verify()` covers unexpired reuse, expired removal, same-`kid` replacement, `ttl=0`, and refresh failure. Run the entire [`test_token_verifier.py`](../../packages/semql-auth/tests/test_token_verifier.py) module. Directly testing `_fetch_jwks()` is insufficient to catch the bypass.

**Design lens:** one authority for duplicated mutable state; explicit cache lifetime; fail closed when authorization correctness cannot be maintained.

## D02 — Dynamic policy bypass in prompt and tool discovery

**Locations:** [`prompt.py:507–519`, `render_catalog_segments`](../../packages/semql-prompt/src/semql_prompt/prompt.py#L507-L519); [`prompt.py:755–777`, `project_tool_descriptions`](../../packages/semql-prompt/src/semql_prompt/prompt.py#L755-L777).

**Defect and impact:** both implementations enumerate with `viewer=None, policy=None`, then classify cubes with empty `required_roles` as public. Empty static roles do not imply that a dynamic `Catalog.policy` permits the current viewer. Policy-hidden names/fields consequently enter planner text and provider tools. This is a schema-disclosure/discovery defect; no SQL execution bypass was demonstrated.

**Reproduction:** construct a public `orders` cube with measure `revenue`; construct `Catalog([orders], policy=lambda cube, viewer: False)`; use `AuthContext(viewer_id="guest")`. Compare direct viewer/policy-filtered rendering with `planner_prompt(catalog, viewer=guest)` and `to_openai_tools(catalog, viewer=guest)`.

**Observed:** direct filtered rendering omitted the cube, while the planner contained `orders.revenue` and the OpenAI tools included `query_orders`. The coordinating probe printed `policy hidden in planner: False` and `policy hidden in tools: False`.

**Expected:** all these public discovery surfaces honor the same catalog policy for the same viewer.

**Repair direction:** keep dynamic-policy-dependent content out of cross-viewer static/invariant sets unless viewer-independence can actually be established. Render only authorized cubes in the per-viewer path. Apply the same decision to tool projection, not just planner rendering.

**Acceptance:** deny-all and viewer-dependent policies on cubes with empty roles are honored across direct/segmented planner APIs and provider exporters. Preserve intended behavior for catalogs without policy and the documented no-viewer mode. Run the entire [`test_prompt_security_regressions.py`](../../packages/semql-prompt/tests/test_prompt_security_regressions.py), [`test_prompt_cacheable.py`](../../packages/semql-prompt/tests/test_prompt_cacheable.py), and applicable provider-export modules.

**Design lens:** convenience APIs must preserve canonical behavior; authorization decisions must not be duplicated with weaker predicates.

## D03 — Viewer-specific protected fields enter invariant tool descriptions

**Location:** [`prompt.py:767–777`, `project_tool_descriptions`](../../packages/semql-prompt/src/semql_prompt/prompt.py#L767-L777). Its documented cross-viewer cache contract is at lines 740–751.

**Defect and impact:** the function renders the description with the current viewer, then places that result in `invariant` whenever the cube has no cube-level required roles. Such a cube may still have role-protected fields. A consumer following the documented invariant-cache contract can reuse an authorized viewer's description for an unauthorized viewer.

**Reproduction:** use a public `orders` cube containing public `revenue` and admin-only `secret`; call `project_tool_descriptions()` with admin and guest viewers; compare the invariant maps.

**Observed:** invariant maps differed. The admin map contained `secret`; the guest map did not. The security impact of later cross-viewer reuse follows from the advertised caching contract; the probe did not simulate a deployed client's cache.

**Expected:** the invariant map is viewer-independent and contains no viewer-protected fields. The merged authorized projection still includes fields the authorized viewer may see.

**Repair direction:** render invariant descriptions against an empty-role viewer and put authorized field additions/descriptions in the viewer-specific projection. Inspect `ToolDescriptionProjection.all()` precedence so an invariant entry cannot erase an authorized overlay for the same cube.

**Acceptance:** invariant maps are identical across anonymous/admin/guest viewers; merged admin descriptions retain protected fields; guest descriptions never contain them. Run [`test_tool_description_projection.py`](../../packages/semql-prompt/tests/test_tool_description_projection.py) and [`test_prompt_security_regressions.py`](../../packages/semql-prompt/tests/test_prompt_security_regressions.py) in full.

**Design lens:** encode the meaning of “invariant” in construction, not only a name/comment; cache only facts stable over the cache scope.

## D04 — Budget trimming deletes fields from a protected cube

**Location:** [`prompt_budget.py:56–79`, `_drop_domain_subsection`](../../packages/semql-prompt/src/semql_prompt/prompt_budget.py#L56-L79). Protected-cube preservation is promised at lines 3–8.

**Defect and impact:** without `## DOMAIN CONTEXT`, the function searches the entire text for `**Relations:**`. That marker also appears inside ordinary cube content. Removal extends to the next level-two heading or the text end, potentially deleting fields and subsequent protected cube blocks. A budget can thus achieve an apparent size reduction by breaking its preservation contract.

**Reproduction:** render an `orders` cube with `relations` prose and `revenue`; call `PromptBudget.apply(text, protected_cubes=frozenset({"orders"}))` with a ceiling low enough to trigger trimming.

**Observed:** the review-slice probe with a catalog-block input and `max_tokens=50` returned `fits=True`, `token_count=47`, `dropped=('relations',)` while removing `orders.revenue`. A separate full-planner probe with `max_tokens=100` returned `fits=False`, the same dropped reason, and also removed that field. These are different input renderings; do not rely on the numeric threshold as the bug's invariant.

**Expected:** protected field structure survives even when the result cannot fit. Optional prose may be trimmed, but a same-named marker must not authorize deleting the rest of a cube.

**Repair direction:** restrict removal to verified domain-section bounds or explicitly recognized standalone domain sections. Avoid whole-document substring deletion as a structural operation.

**Acceptance:** protected cube fields survive ordinary per-cube relations, missing domain context, multiple protected cubes, and cannot-fit outcomes. Run [`test_prompt_budget.py`](../../packages/semql-prompt/tests/test_prompt_budget.py) in full and smoke-render the final planner text.

**Design lens:** structural data should carry structure; handle unmatched shapes deliberately; preserve required content rather than silently degrade correctness.

## D05 — Dense spine drops the final intersecting bucket

**Location:** [`backend.py:305–318`, `_StdSqlDialect.emit_time_spine`](../../packages/semql/src/semql/backend.py#L305-L318).

**Defect and impact:** the last generated bucket is `date_trunc(granularity, end - one whole step)`. If the exclusive end is inside a bucket, subtracting a whole step omits the final intersecting bucket. The spine's left join can then discard observed aggregate data, not merely omit an empty bucket.

**Reproduction:**

- DuckDB table `events(ts TIMESTAMP, amount INTEGER)`.
- One row: `('2026-01-02 10:00:00', 7)`.
- Cube measure `sum(amount)` and time dimension `ts`.
- Daily `TimeWindow(range=("2026-01-01T12:00:00", "2026-01-02T12:00:00"))`.
- Execute once without fill and once with `fill_nulls_with=0`.

**Observed:** without fill, the Jan 2 bucket contained 7; with fill, only the Jan 1 bucket with 0 remained. The observed Jan 2 value was lost.

**Expected:** starts for both intersecting daily buckets exist; the Jan 2 bucket retains 7. The exclusive end itself must not introduce a bucket when it lies exactly at a boundary.

**Repair direction:** generate through the truncated end and exclude bucket starts at or beyond the exclusive end, rather than subtracting a full step before truncation. Inspect equivalent backend overrides before choosing a shared repair.

**Acceptance:** aligned and unaligned bounds, sub-step windows, and relevant calendar granularities preserve observed values and exclude boundary-only buckets. Run [`test_time_spine.py`](../../packages/semql/tests/test_time_spine.py) and [`test_backend.py`](../../packages/semql/tests/test_backend.py) in full; execute emitted SQL, not just inspect it. DuckDB reproduction is not evidence of execution on every supported backend.

**Design lens:** boundary/off-by-one reasoning; errors are preferable to silent wrong results.

## D06 — Dense fill drops declared inline-derived outputs

**Locations:** inner projection at [`compile.py:3026–3034`](../../packages/semql/src/semql/compile.py#L3026-L3034); outer spine projection at [`compile.py:3109–3128`](../../packages/semql/src/semql/compile.py#L3109-L3128).

**Defect and impact:** inline-derived outputs are added to the inner SQL and artifact schema, but the dense-fill outer SELECT iterates only `measure_col_names`. It drops the derived columns while `CompiledQuery.columns` and analysis continue declaring them. Consumers receive rows incompatible with their declared schema.

**Reproduction:** DuckDB orders with timestamp `created_at`, amount sum, and count measure. Request `measures=["orders.amount"]` plus `InlineDerived(name="per_order", op="ratio", operands=["orders.amount", "orders.count"])`; daily range `("2026-01-01", "2026-01-03")`, fill 0. Compile, execute, and compare artifact outputs with cursor column names.

**Observed:** artifact columns and analysis outputs declared `['created_at_day', 'amount', 'per_order']`; SQL returned only `['created_at_day', 'amount']`, with rows for Jan 1 amount 5 and Jan 2 amount 0. Static `validate()` returned no issues for this input.

**Expected:** executable output schema equals the declared artifact/analysis schema; the requested derived metric is not silently lost.

**Repair direction:** have wrappers consume the complete resolved projection contract, with explicit derived-measure fill semantics. If the shape cannot be supported coherently, reject it before emitting an inconsistent artifact rather than faking a derived value.

**Acceptance:** derive independent numeric expectations for observed and missing buckets; assert result-column order, row arity, metadata, and analysis agree. Cover operand projections and applicable aliases. Run [`test_time_spine.py`](../../packages/semql/tests/test_time_spine.py), [`test_inline_derived.py`](../../packages/semql/tests/test_inline_derived.py), and [`test_semantic_analysis.py`](../../packages/semql/tests/test_semantic_analysis.py) in full.

**Design lens:** one canonical projection contract; wrappers must preserve meaning rather than reconstruct a partial representation.

## D07 — `Catalog.from_spec(runtime=…)` ignores the runtime bundle

**Location:** [`catalog.py:805–852`, `Catalog.from_spec`](../../packages/semql/src/semql/catalog.py#L805-L852).

**Defect and impact:** the runtime parameter is documented as overriding per-call callables, but its fields are never used. The constructor forwards only the separate keyword arguments. This silently discards policy, hooks, registry, and other supplied behavior. A policy omission can turn a denied query into an accepted query when no other restriction blocks it.

**Reproduction:** create a public DuckDB orders cube and `CatalogSpec(cubes=(orders,))`. Supply `CatalogRuntime(policy=deny_all, scope_fns={}, unit_registry=None, error_transform=None, compile_hooks=[], sql_rewrite_hooks=[])` to `Catalog.from_spec`. Compile its measure with a non-null viewer. Compare with direct `Catalog([orders], policy=deny_all)`.

**Observed:** `restored.policy is deny_all` was false. The restored catalog accepted the query. Direct construction with the deny policy rejected it with `UnknownIdentifierError` (the public refusal deliberately does not disclose a hidden field).

**Expected:** supplied runtime behavior is retained according to the documented precedence; policy cannot silently vanish.

**Repair direction:** resolve runtime/keyword precedence once and forward the resolved values, or deliberately remove the misleading runtime parameter. Do not fix only `policy`: all fields need a consistent contract. An empty registry/list may be an explicit choice; avoid conflating it with absence through truthiness.

**Acceptance:** policy denial and scope/hook behavior survive `from_spec`; cover explicitly absent runtime fields and documented fallback precedence. Run [`test_catalog_spec_runtime.py`](../../packages/semql/tests/test_catalog_spec_runtime.py), [`test_scope.py`](../../packages/semql/tests/test_scope.py), and affected hook modules in full.

**Design lens:** construct fully formed objects; explicit/narrow dependencies; convenience paths funnel into canonical behavior.

## D08 — Catalog spec omits operational settings

**Locations:** runtime assignment at [`catalog.py:875–880`](../../packages/semql/src/semql/catalog.py#L875-L880); spec construction at [`catalog.py:963–971`](../../packages/semql/src/semql/catalog.py#L963-L971); row-cap enforcement at [`rows.py:406–417`](../../packages/semql/src/semql/rows.py#L406-L417).

**Defect and impact:** `Catalog.__init__()` does not pass `allow_mutations`, `max_list_limit`, or `max_mutation_rows` into its spec. Serialization therefore records defaults rather than configured behavior. Reconstructing the catalog changes admission/limit behavior across a process boundary.

**Reproduction:** create `Catalog(..., allow_mutations=True, max_list_limit=7, max_mutation_rows=3)`, serialize `catalog.spec`, validate it, and reconstruct with `Catalog.from_spec`. For an entity keyed by `orders.id`, compile `EntityList(entity="order", limit=50)` before and after the round-trip.

**Observed:** original settings were `(True, 7, 3)`; spec and restored settings were `(False, 1000, 1000)`. Actual `list_rows()` plan limits were 7 before and 50 after. The mutation-row enforcement consequence was not separately executed; the configuration loss itself was observed.

**Expected:** serialization preserves all advertised spec-backed operational settings.

**Repair direction:** include these settings when constructing the spec and identify one state authority for construction and serialization. Avoid adding yet another synchronized copy.

**Acceptance:** round-trip each setting through JSON, assert an actual row-list cap is unchanged, and exercise mutation admission/row ceilings through their authoritative action boundary. Run [`test_catalog_spec_runtime.py`](../../packages/semql/tests/test_catalog_spec_runtime.py), [`test_rows.py`](../../packages/semql/tests/test_rows.py), and affected mutation execution tests in full.

**Design lens:** compatibility across process boundaries; one authority for duplicated state; bounds must survive serialization.

## D09 — Nested mutation changes model hashes

**Locations:** [`model.py:235–284`, `_freeze` and `_HashableModel.__hash__`](../../packages/semql/src/semql/model.py#L235-L284); mutable field-role collection at [`model.py:303–314`](../../packages/semql/src/semql/model.py#L303-L314).

**Defect and impact:** Pydantic's `frozen=True` prevents attribute replacement, not mutation of nested lists/dicts. The custom hash recomputes from current nested values. A model can therefore change hash after insertion into a dict/set. Catalog role collections can also drift after validation despite the model's frozen contract. This is not proof that an external attacker can mutate host-owned catalog data.

**Minimal reproduction:**

```python
from semql import Dimension

field = Dimension(
    name="region", sql="{o}.region", type="string", required_roles=["admin"]
)
cache = {field: "cached dimension"}
old_hash = hash(field)
field.required_roles.append("reader")
print(old_hash != hash(field))
print(cache.get(field, "MISSING"))
```

**Observed:** `True`, then `MISSING` for lookup using the same object.

**Expected:** a hashable key's equality/hash-relevant state cannot mutate. A frozen catalog value must not silently acquire new structural/security state after validation.

**Repair direction:** make equality/hash-relevant nested state genuinely immutable, or remove hashability from models that intentionally expose mutable containers. Do not simply cache the initial hash while permitting value equality to change; that introduces another contract violation. Preserve opaque caller-owned metadata semantics rather than interpreting its contents.

**Acceptance:** mutation is prevented or types are intentionally unhashable; equal values hash equally where hashing remains supported; dictionary/set lookup stays stable. Run [`test_frozen_model_hashable.py`](../../packages/semql/tests/test_frozen_model_hashable.py), [`test_frozen_models.py`](../../packages/semql/tests/test_frozen_models.py), and affected model/metadata tests in full.

**Design lens:** effective immutability, not shallow flags; make illegal states unrepresentable.

## D10 — Comparison splits matching NULL groups

**Location:** [`compile.py:2724–2734`, comparison FULL OUTER JOIN`](../../packages/semql/src/semql/compile.py#L2724-L2734).

**Defect and impact:** current and prior aggregates group NULL dimensions correctly, but their join uses ordinary `=`. NULL does not equal NULL under SQL equality, so the same group splits into two rows with wrong comparison values and duplicate logical grain.

**Reproduction:** DuckDB `orders(region VARCHAR, amount DOUBLE, created_at TIMESTAMP)` with `(NULL, 5, '2026-01-01')` and `(NULL, 2, '2025-12-01')`. Query dimension region, sum amount, current range `('2026-01-01', '2026-02-01')`, and `CompareWindow()`.

**Observed:** in region/current/prior/delta/pct-change order, rows were `(NULL, 5, NULL, 5, NULL)` and `(NULL, NULL, 2, -2, NULL)`.

**Expected:** one row `(NULL, 5, 2, 3, 150)`.

**Repair direction:** dialect-supported null-safe equality on every comparison grouping key. Preserve distinction between group identity and ordinary filter comparison semantics.

**Acceptance:** execute one and multiple nullable dimension keys, non-null keys, and groups present only in one period; verify unique final grain and independent comparison values. Run [`test_compare.py`](../../packages/semql/tests/test_compare.py), [`test_compare_loadbearing.py`](../../packages/semql/tests/test_compare_loadbearing.py), and affected semantic-analysis tests in full.

**Design lens:** model absence explicitly; preserve semantic group identity across physical operation boundaries.

## D11 — Comparison result `HAVING` is non-executable

**Location:** [`compile.py:2750–2761`, comparison output filter`](../../packages/semql/src/semql/compile.py#L2750-L2761). Existing SQL-text-only coverage: [`test_compare.py:321–329`](../../packages/semql/tests/test_compare.py#L321-L329).

**Defect and impact:** semantic post-aggregate filters are attached as SQL `HAVING` to the non-aggregate outer comparison SELECT. The outer query has no `GROUP BY` or aggregate expression justifying that clause. DuckDB refuses the generated SQL.

**Reproduction:** compare January amount sum with `CompareWindow()` and `having=[Filter(dimension="compare.amount.delta", op="gt", values=[0])]`. Compile and execute.

**Observed:** compilation succeeded; execution raised `BinderException` saying `column amount must appear in the GROUP BY clause or be used in an aggregate function`.

**Expected:** filter the completed comparison rows by positive delta and return executable results.

**Repair direction:** wrap the completed comparison projection and apply `WHERE` against its output columns before final order/limit. Preserve parameter binding and aliases; do not suppress the backend error or remove the filter.

**Acceptance:** execute all supported synthetic facets and applicable aliases, with order/limit and no-dimension versus grouped shapes; assert independent selected-row expectations. Run [`test_compare.py`](../../packages/semql/tests/test_compare.py) and [`test_compare_loadbearing.py`](../../packages/semql/tests/test_compare_loadbearing.py) in full. Replace incidental SQL-substring coverage with behavior rather than pinning the new SQL shape.

**Design lens:** lower decisions according to the physical stage; generated SQL text is not execution proof.

## D12 — Merge materialization loses numeric schema

**Location:** [`engine.py:854–896`, `_infer_column_types` and `_duckdb_type_for`](../../packages/semql-engine/src/semql_engine/engine.py#L854-L896). Inspect [`adapter.py`](../../packages/semql-engine/src/semql_engine/adapter.py) and both merge paths as part of repair.

**Defect and impact:** `AdapterResult` does not retain physical column types. Materialization guesses from the first non-NULL Python value and falls back to VARCHAR for empty/all-NULL columns. `Decimal` also becomes VARCHAR. Valid numeric fragments then fail during merge aggregation. Source-side casts cannot repair a schema that is discarded before loading the merge table.

**Reproduction:** use a compiler-produced two-fragment orders/customers plan with a sum merged by customer region. Execute fragments against in-memory DuckDB-backed adapters. Exercise separately: decimal amount `12.34`, an all-NULL numeric amount, and an empty fact fragment. Execute independent source SQL for the expected result.

**Observed:** independent SQL returned `[('EU', Decimal('12.34'))]`, `[('EU', None)]`, and `[]`, respectively. Engine merge raised `BinderException: No function matches the given name and argument types 'sum(VARCHAR)'` in all three cases. Inline merge and `DuckDBMergeEngine` share loading logic, so changing merge-engine selection does not avoid the cause.

**Expected:** typed numeric results remain numeric through adapter/materialization/merge boundaries, including absence of sample values.

**Repair direction:** carry physical output types or an equivalently authoritative schema through adapter results and materialized results. Populate built-in adapters and preserve decimal precision/scale. Define a deliberate compatibility path for third-party adapters without declarations; do not silently convert numeric data to text or infer empty-column types from nonexistent values.

**Acceptance:** decimal, all-NULL numeric, empty fragments, and ordinary integer/float results execute under inline, explicit DuckDB merge, and async engine paths. Check values and output types against independent expectations. Run [`test_merge_engine_default.py`](../../packages/semql-engine/tests/test_merge_engine_default.py), [`test_merge_contract.py`](../../packages/semql-engine/tests/test_merge_contract.py), and affected adapter/async modules in full.

**Design lens:** declared type is the contract; narrow boundaries must carry the information required downstream.

## D13 — Fragment failure leaves sibling async work running

**Locations:** [`engine.py:1029–1040`, `AsyncEngine.run`](../../packages/semql-engine/src/semql_engine/engine.py#L1029-L1040); duplicate gather in [`engine.py:1135–1141`, non-fast `iter_run`](../../packages/semql-engine/src/semql_engine/engine.py#L1135-L1141).

**Defect and impact:** `asyncio.gather()` propagates a child's exception without cancelling its other children. The request can report failure while sibling backend work continues, retaining resources or overlapping a caller's retry/cleanup. No explicit longer-lived owner takes responsibility for these children.

**Reproduction:** two native asynchronous adapters coordinated by events. The failing adapter waits until the sibling starts, then raises `RuntimeError("fragment failed")`. The sibling waits on an explicit release event and logs completion/finalization. Catch the public run error, inspect sibling completion before release, then release it and drain it to avoid orphaning the probe itself.

**Observed:** `run_error fragment failed`; after error, completion log was empty with one pending sibling; after explicit release, log contained `sibling completed after error` and `sibling finalized`.

**Expected:** failure does not return until request-owned siblings have been cancelled/drained or transferred to an explicit owner under a documented contract.

**Repair direction:** explicit task ownership plus cancellation/draining, or structured concurrency preserving the intended public exception type. Apply the repair to both gathers. Cancelling a coroutine awaiting `to_thread` does not stop its underlying thread; define that adapter policy separately instead of claiming universal cancellation.

**Acceptance:** event-driven failure, caller cancellation, sibling finalization, success ordering, and no pending request-owned tasks for both run and non-fast streaming. Run [`test_async_concurrency.py`](../../packages/semql-engine/tests/test_async_concurrency.py) and [`test_async_engine.py`](../../packages/semql-engine/tests/test_async_engine.py) in full, plus affected adapter cancellation modules.

**Design lens:** never orphan concurrent work; guarantee cleanup; explicit parent ownership.

## D14 — Stream early-close does not close the adapter iterator

**Locations:** [`engine.py:1109–1133`, fast-path iterator acquisition/consumption`](../../packages/semql-engine/src/semql_engine/engine.py#L1109-L1133); outer close wrapper at [`engine.py:391–414`](../../packages/semql-engine/src/semql_engine/engine.py#L391-L414).

**Defect and impact:** the fast-path generator takes an adapter row iterator but has no `finally` forwarding closure to it. `AsyncExecutionIterator.aclose()` closes the outer generator, not the nested cursor-like iterator. A lazy adapter may retain its cursor/connection lease after explicit early termination.

**Reproduction:** compile a one-fragment dimension query qualifying for fast streaming. Return an adapter iterator implementing `close()` and a `closed` flag. Consume one chunk, await `stream.aclose()`, inspect that flag.

**Observed:** first chunk `[('1',)]`, fast path true, then `cursor_closed_after_aclose False`.

**Expected:** explicit close releases the engine-owned acquired iterator. Completion and exception paths also clean up according to the ownership contract.

**Repair direction:** define ownership of closeable `AdapterResult.rows`; close the acquired iterator in `finally`. Ensure cleanup also covers validation failure after acquisition. Do not close externally owned connections indiscriminately.

**Acceptance:** normal exhaustion, early `aclose`, adapter/iteration errors, and post-acquisition validation errors finalize acquired row resources exactly as contracted. Run [`test_async_engine.py`](../../packages/semql-engine/tests/test_async_engine.py) and affected iterator/adapter modules in full.

**Design lens:** resource acquisition implies an owner and a guaranteed release path.

## D15 — Cached rows share mutable cells with callers

**Locations:** insertion at [`engine.py:719–727`](../../packages/semql-engine/src/semql_engine/engine.py#L719-L727); return isolation at [`engine.py:426–442`, `_isolate`](../../packages/semql-engine/src/semql_engine/engine.py#L426-L442).

**Defect and impact:** both paths copy only the outer row list and assume tuple rows imply immutable contents. Adapter/custom merge results permit arbitrary cell values; a tuple can contain a mutable list/dict. One consumer can corrupt later cache hits, including by mutating the initial miss result.

**Reproduction:** compiler-produced single-fragment plan; adapter returns a list-valued cell `['original']`; passthrough custom merge engine; `Engine(cache_size=2)`. Run once, mutate the returned cell with `mutated_miss`, fetch a cache hit, mutate it with `mutated_hit`, fetch again.

**Observed:** first hit contained `['original', 'mutated_miss']` with the same cell identity; second hit contained `['original', 'mutated_miss', 'mutated_hit']`; cache-hit count was 2.

**Expected:** documented result isolation prevents caller mutation from changing stored results or another caller's result.

**Repair direction:** recursively isolate supported mutable values at insertion and retrieval, or explicitly decline caching unsupported mutable cells. Fixing retrieval alone leaves the miss result aliased to the cache. Do not introduce avoidable deep copies on unrelated non-cache execution paths without need.

**Acceptance:** nested list/dict mutation on miss and hit cannot affect later results; immutable scalar rows preserve ordinary behavior. Run [`test_cache_hazards.py`](../../packages/semql-engine/tests/test_cache_hazards.py) and [`test_cache_observability.py`](../../packages/semql-engine/tests/test_cache_observability.py) in full.

**Design lens:** define actual ownership/isolation of state, not just outer-container copying.

## Cross-cutting design liabilities

These explain the defects and guide repair; they are not extra independently scored findings.

1. **Multiple representations without one authority.** Catalog runtime fields, `CatalogSpec`, and `CatalogRuntime` coexist, but convenience construction can omit one representation's settings. Prefer one canonical resolved construction path and derive convenience/serialized views from it.
2. **Partial reconstruction of semantic contracts.** Dense-fill wrapping and merge materialization rebuild projections/types from partial lists or sample rows. Carry resolved output/schema contracts instead of guessing downstream.
3. **Prose promises exceed type guarantees.** “Frozen,” “invariant,” “isolated,” and “TTL” do not hold merely because a configuration flag, container type, name, or comment says so. Enforce their scope at construction and action boundaries.
4. **Textual SQL assertions miss consumer-visible failures.** In particular, the comparison-HAVING test demonstrates emitted syntax rather than successful filtering. Permanent regressions should exercise independent numeric, membership, schema, lifetime, or cache-isolation invariants.

## Repair verification and completion checklist

For every repair group:

- Reproduce the affected behavior on the discovery baseline before changing it. Do not treat a historical runtime reproduction as proof of current behavior if the fixing checkout has moved.
- Inspect symbol references and duplicate implementations. D02 has two discovery branches; D12 affects all adapter producers/materialization consumers; D13 has two gathers; D15 has insertion and retrieval. A one-site patch is insufficient.
- Prefer the smallest coherent change preserving unrelated behavior. Public signature/serialized-shape changes require caller migration and compatibility documentation; do not silently add shims or reinterpret trusted-host inputs as untrusted wire formats.
- Keep failing-before/passing-after regression coverage for consumer-visible invariants. Run the **whole affected test modules**, not only a single new test.
- Smoke the real public path after changes: emitted SQL execution, rendered prompt/tool surface, public token verification, actual stream lifecycle, or repeated cached results as appropriate. Mock forwarding alone is not proof.
- Review resource ownership on both successful and failed paths. Use event-driven concurrency tests rather than sleeps; ensure the reproducer itself drains all spawned work.
- Run repository-native lint/type/boundary checks and `just check` for integrated source changes. Record actual results, skipped integrations, and intentional API breakages rather than claiming unexercised backends.
- Update affected documentation/changelog after behavioral proof. Record fixing commit(s), verification timestamp, reproduction results, and any explicit scope decisions below or in the fixing PR.

The discovery review itself made no source fixes. The subsequent implementation and its bounded verification evidence are recorded below.

## Repair completion

- **Verification timestamp:** `2026-10-02T18:06:54Z`.
- **Fixing commit:** the repository commit containing this repair-completion record. Verification preceded the commit; no package publication or live identity-provider operation was performed.
- **Status:** all D01–D15 repaired, using four subagent slices and coordinated red/green verification. [Implementation plan and evidence](../plans/core-library-defect-repairs-2026-10-02.md); [pre-v1 migration requirements](../migrations/core-library-defect-repairs.md).
- **Integrated verification:** `env -u SEMQL_CONFORMANCE_POSTGRES -u SEMQL_CONFORMANCE_CLICKHOUSE SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check` passed formatting, lint, both strict type checkers, dependency/core/frozen-model boundaries, and the full suite: **3,267 passed, 4 skipped, 1 xfailed; 81 snapshots passed**.
- **Executed backend coverage:** DuckDB, PostgreSQL **16.14**, and ClickHouse **26.8.2.7**, with pinned disposable local Testcontainers for the latter two. This is not a claim of all-backend or live BigQuery conformance.
- **Skips:** two intentional DuckDB-only cross-backend cells and two optional LangChain cells. The separate dependency-enabled provider/security run passed **33 tests**; default-environment skip reasons were confirmed in a full-module run (**119 passed, 4 skipped**).

| Findings | Repaired contract and regression coverage |
| --- | --- |
| D01 | Public JWKS verification expires parsed keys with documents, honors zero-TTL refresh, and fails closed on removal/replacement or failed refresh. Synthetic signing keys and an intercepted HTTP boundary avoid live-provider access. |
| D02–D03 | Dynamic-policy discovery is authoritative for explicit viewers; reusable static/invariant content stays viewer-independent even when initialized without a viewer. Public no-viewer discovery is preserved in noncacheable overlays; admin/guest cache reuse and provider exports are covered. |
| D04 | Protected cube field structure survives optional prose/relation trimming, including unfamiliar or absent domain-section boundaries. |
| D05–D06 | Dense fill retains intersecting end buckets and the resolved output/alias/derived schema. Actual SQL execution checks half-open boundaries, filled operands, and missing or zero denominators. |
| D07–D08 | Catalog runtime precedence and serialized operational settings survive reconstruction. Real compile/policy/scope/hook/unit behavior and MCP list/mutation execution verify admission, read-only subtype preservation, and row caps after JSON round-trip. |
| D09 | Catalog value collections are detached and immutable; copy updates validate replacement state. Supported hashable graphs remain stable keys, while unchecked or unsupported mutable graphs cannot silently become hash keys. Validated physical-source transforms and unchecked catalog input are covered. |
| D10–D11 | NULL grouping keys join as the same comparison group without colliding with typed sentinels; comparison-output filters execute after projection and before order/limit. All comparison facets, grouped and scalar shapes, and typed nullable keys have executed-result oracles. |
| D12 | Physical numeric schema travels from adapters through materialization and every merge path, including empty/all-NULL rows. Decimal precision, integer/float producer metadata, schema-less inference across all observed values, and precision overflow are covered; unsupported lossless materialization fails explicitly instead of rounding or coercing to text. |
| D13–D14 | Failure/caller cancellation cancels and drains request-owned siblings; acquired adapter iterators close on early termination, exhaustion, and error. Event-driven lifecycle tests cover both async run and non-fast streaming, plus success ordering and finalization. |
| D15 | Cache insertion and retrieval isolate supported nested mutable cells; unsupported opaque cells are not cached. Miss/hit mutation cannot corrupt later results, and uncached execution avoids cache-only copying. |

Scope decisions and intentional pre-v1 contract changes are documented in the migration guide. Driver threads already running under `asyncio.to_thread` cannot be forcibly stopped; driver-level timeouts/cancellation remain the integrator's responsibility.
