# Core-library defect repairs

## Scope and success criteria

Implement D01–D15 in [the defect handoff](../specs/core-library-code-design-defect-handoff-2026-10-02.md). Preserve unrelated local documents; no commits, pushes, publication, or live identity-provider access.

Success means public-path behavior satisfies every finding's acceptance criteria, regression tests cover the broader defect classes, and affected full modules plus repository-native checks pass. Backend evidence must distinguish executed SQL from dialect rendering. Synthetic identity-provider boundaries and in-process lifecycle adapters are intentional isolated test fixtures.

## Repair slices

1. Auth/catalog/model value semantics: D01, D07, D08, D09.
2. Prompt discovery and structural budget trimming: D02, D03, D04.
3. Compiler time spine and comparison output contracts: D05, D06, D10, D11.
4. Engine schema, concurrency, resource ownership, and cache isolation: D12–D15.

Each slice has one subagent owning source and tests. The coordinator owns this plan, changelog, handoff status, integrated checks, and smoke verification. Exported-symbol changes require LSP reference inspection and complete caller migration. Physical adapter schema must preserve precision/scale and empty/all-NULL columns. Model hash fixes must not cache a hash while equality-relevant state remains mutable.

## Red/green sequence

Subagents first add isolated behavior regressions without implementation changes and report exact test selectors. The coordinator executes the red phase. Only after observed consumer-visible failures does the coordinator release each slice for implementation. Subagents do not run build/lint/test/format commands mid-flight. After all edits land, the coordinator runs green affected modules, native format/lint/type/boundary checks and the full suite, then exercises real public paths separately.

Tests target boundaries and invariants rather than emitted SQL wording or mocked forwarding: authorization across viewers, cache generations, interval intersection, projection/schema identity, serialized admission limits, stable model keys or intentional unhashability, NULL group identity, completed-output filtering, numeric schema without sample values, task finalization, iterator closure, and nested mutable-cell isolation.

## Verification evidence

### Red phase

- Auth/catalog/value regressions: 10 failures before fixes, covering stale key trust/rotation, dropped runtime precedence and limits, mutation admission, and nested value aliasing. A mixed hash test that initially reached an already-unhashable Cube was separated into mutation protection and supported runtime hashing contracts.
- Prompt regressions: 5 failed, 2 passed, 1 skipped before fixes. Failures cover policy-dependent static discovery, viewer-dependent invariant descriptions, overlay precedence, and protected cube structure. Mandatory OpenAI/Bedrock coverage was separated from optional LangChain coverage.
- Compiler regressions: 16 failed, 2 passed before fixes, including alias-bound dense projections, multiple nullable keys, all comparison facets, grouped/no-dimension shapes, and order/limit.
- Added real-backend conformance cases: 20 failed, 7 passed before fixes on DuckDB, PostgreSQL 16.14, and ClickHouse 26.8.2.7. Interval intersection, dense-derived schema, and NULL comparison grain failed on all three; comparison HAVING failed on DuckDB/PostgreSQL while ClickHouse accepted those cases.
- Engine numeric schema, iterator ownership, and cache regressions failed before fixes. The bounded async failure reproduction failed at sibling-finalization assertions for both `run()` and non-fast streaming; caller cancellation already passed on the baseline. Initial invalid async fixtures were corrected before releasing the source fix.
- A later cross-viewer reproduction failed for both no-viewer planner static content and tool-description invariants; both now keep dynamic-policy content exclusively in noncacheable overlays.
- Integration probes reproduced two additional numeric losses through the public DBAPI adapter and real DuckDB merge: unbounded decimal scale 10 was narrowed to 9, and Python `float` metadata was materialized as 32-bit float. These observations widened the producer regressions rather than weakening precision guarantees.

### Public-path verification

- Rendered policy-filtered prompts/provider projections, protected budget structure, runtime policy denial, catalog JSON admission settings, stable model keys, and real DuckDB dense/comparison execution all passed their separate smoke scenarios.
- Public JWKS verification passed unexpired reuse and expired key revocation with synthetic signing keys and an intercepted HTTP boundary; no live identity provider was accessed.
- Real decimal fragment execution passed inline, explicit, and async DuckDB merge paths. Cache miss/hit isolation, async sibling finalization before error return, and exact-once early iterator closure passed their public-path smoke scenarios.
- Cross-viewer cache reuse passed for no viewer, admin, and guest: identical static/invariant segments, unfiltered no-viewer noncacheable content, authorized admin overlays, and no guest cube disclosure.
- MCP entity execution passed original and JSON-restored mutation/list caps against real disposable DuckDB tables: over-cap writes were refused without modifying rows, admitted writes matched preview counts, and list caps constrained actual returned IDs.
- Optional LangChain verification used a transient `uv run --frozen --with langchain-core` environment: 33 passed across provider export and prompt-security modules. Its missing-dependency test now isolates module availability rather than relying on the workstation's installed packages.

### Integrated checks

- Verified at **2026-10-02T18:06:54Z**, before committing the repairs. All D01–D15 repairs are implemented and verified; no package publication or live identity-provider operation was performed.
- Command: `env -u SEMQL_CONFORMANCE_POSTGRES -u SEMQL_CONFORMANCE_CLICKHOUSE SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check`.
- Ruff formatting/lint, strict mypy (**267 source files**), Pyright (**0 errors**), Tach, core-boundary checks, and frozen-model checks all passed.
- Full suite: **3,267 passed, 4 skipped, 1 xfailed; 81 snapshots passed**. Real backend conformance ran against DuckDB, PostgreSQL **16.14**, and ClickHouse **26.8.2.7**, with external endpoints unset and pinned local Testcontainers opted in.
- Skip-reason/full-module verification: **119 passed, 4 skipped** across semantic backend conformance, OpenAI/LangChain exports, and prompt-security modules. Two skips are intentional DuckDB-only cross-backend cells; two require optional `langchain-core`, whose separately enabled run passed **33 tests** as recorded above.
- Final public smoke rerun passed rendered prompts and policy projection, catalog wire/runtime behavior, stable model keys, executed dense/comparison SQL, numeric inline/explicit/async merge, cache isolation, JWKS expiry/revocation, sibling cleanup, and exact-once early iterator closure. Unbounded DB-API `NUMERIC` retained `Decimal("0.1234567891")`; a Python `float` type code retained `0.123456789101112` in its independent merge probe.
- Integrated caller migration also preserved partitioned-scan validation and actual partition execution, and catalog rejection of explicitly unchecked `model_construct()` input. Obsolete mutable-default and rendered-comparison-SQL assertions were removed instead of repinned; comparison coverage now asserts executed rows, aliases, NULL groups, and numeric values.
- Griffe comparison of all eight public package modules against discovery commit `fc57a9b3714b3fad34fa90201ad5737ddff4f0a5` reported **20 changes in `semql`**, none in the other seven modules. Static API diff is not exhaustive: [migration requirements](../migrations/core-library-defect-repairs.md) also cover immutable collection annotations/ownership, catalog spec schema 2, backend strategy methods, physical adapter schema, resource ownership, and cache isolation.
- No real BigQuery warehouse, live JWKS provider, or universal backend support is claimed. BigQuery/ClickHouse SDK producer-shape regressions use isolated driver-boundary fixtures; PostgreSQL/ClickHouse execution evidence comes from the real local conformance services above.
