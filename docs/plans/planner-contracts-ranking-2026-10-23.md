# Planner contracts, continuation, and partitioned ranking

## Scope and authorization

Implement GitHub enhancements #15, #17, and #18; #16 is excluded. Open a PR against main. Do not merge, tag, release, or publish. Base: main 187162ce86ac455ce9a803306222d6224fcb78bd. Work in `.worktrees/feat/planner-contracts-ranking` on `feat/planner-contracts-ranking`.

## Agreed product decisions

- Four peer planner outcomes: Plan, NeedContext, NeedUserClarification, CannotPlan.
- Library-enforced continuation locks. The host owns conversation state, retrieval, model calls, authorization, budgets, and termination.
- Partitioned ranking supports single-backend grouped queries including derived measures. Comparison, dense-time fill, and federated ranking are explicitly unsupported.
- Exact-N ranking automatically appends remaining grouping keys in a disclosed deterministic order. Boundary-inclusive ties use only score keys, never the appended tie breakers.
- Canonical semantic query input models reject unknown fields. This is an intentional pre-v1 compatibility break, documented without performing a release/version bump.

## Terminology and invariants

A representation profile describes what a decoder can express, not authorization or executable support. A schema projection reports removed features and weakened validation. A generation contract keeps its output model/schema and prompt guidance aligned.

A planner outcome is an untrusted proposal. Plan carries the existing QueryPlan payload and requires nonempty steps; it is not compiled or authorized merely by parsing. CannotPlan carries bounded diagnostics, not authoritative model-selected access decisions. Context requests identify typed unresolved decisions and authorized semantic targets, never arbitrary executable tools, URLs, SQL, or identities.

Continuation state is host-owned. A typed decision identifies what may change; all other accepted query constraints remain locked. Resumption checks correlation/revision, permitted change, and unchanged constraints before compiling under current host identity. Mandatory continuation state cannot be silently trimmed. Missing context, authoritative no match, ambiguity, and unavailable sources are distinct.

Partitioned ranking selects grouped result rows after aggregation/derived projections/HAVING and before final order/limit/offset. NULLS LAST is explicit in the ranking contract. Include-ties means RANK <= N, not DENSE_RANK <= N. Exact-N means ROW_NUMBER <= N with deterministic group-key tie breakers. The internal rank ordinal is not an exposed output column. Semantic analysis records selection independently of aggregate population/grain.

## Implementation slices and ownership

### Ranking and canonical input strictness (#15)

Own core spec, logical plan, compiler ranking lowering, semantic analysis/comparator, query field walkers, parser/serializer preservation, federation rejection, and associated tests. Add immutable strict ranking value types and a `SemanticQuery.rank` clause; reject unknown fields throughout query input components without globally changing catalog model permissiveness. Preserve ranking through compile_query/compile_plan and physical rewrites. All referenced partition/score/group keys must be authorized; reject masked ranking/partition/tie keys rather than ordering hidden originals. Support selected derived outputs. Safely compose with ordinary aggregation and joins; do not silently bypass rank in symmetric/rollup paths. Unsupported shapes fail with typed diagnostics. Emit SQL through sqlglot AST, no SQL string rewriting or Python ranking.

### Profiles and guidance (#17)

Own schema projection implementation, new profile/report value types, and existing prompt/adaptive/repair/tool-export plumbing. Expose explicit rich/tool and restrictive schema profiles. Preserve required nested query payloads when projecting QueryPlan and new outcome envelopes; refuse unsupported projections rather than deleting required semantic payloads. Record removed fields and tuple-constraint weakening. Generate guidance from the same profile/features, including ranking availability. Retain canonical model validation after projected-schema validation. Keep compiler checks partial and separate from representation support. Profile-dependent static text must not collide in caches; authorized examples and catalog data remain request scoped. Thread explicit profiles consistently through query generation, adaptive prompting, query repair, and provider tools. No provider calls or implicit model-name selection.

### Outcomes, lookup certainty, and continuation (#18)

Own new semql-prompt outcome/continuation modules, core lookups.py improvements, and associated tests. Four discriminated immutable extra-forbid outcomes reuse QueryPlan and Diagnostic. Structured context requests/results and choices are bounded and correlated. Preserve ambiguity before truncation; duplicate exact labels cannot select the first key; single fuzzy candidates are suggestions, not authoritative resolution. Distinguish unavailable dynamic lookup from authoritative no match and report candidate completeness. Ensure the existing resolution helpers/callers retain coherent behavior after the cutover.

Provide pure continuation acceptance with typed allowed decisions, locked query constraints, and request/revision checks; apply only the allowed decision and compile the accepted plan with caller-owned current auth/context. No generic JSON patch protocol. Provide mandatory fenced continuation context usable by the adaptive builder; failures or budgets cannot erase locks. New schema/protocol integration uses the profile slice's output-model projection, with shared interfaces coordinated directly.

### Integration ownership

Parent owns public export consolidation, shared capability-check integration, README/changelog/migration notes, executable deterministic demos, full verification, commits/push/PR. Agents may add their tests but must not run builds, lint, type checks, formatters, or test suites mid-flight. Parent runs verification after integration. Shared files have one writer; communicate changes rather than overlapping edits. No commits by agents.

## Acceptance

- Each profile's emitted schema and guidance agree; nested Plan/query payloads remain required; projection losses are inspectable; output parsing is strict.
- New generation and repair paths produce the four-outcome contract consistently without contradictory empty-plan guidance.
- Lookup duplicate labels, truncated candidates, fuzzy suggestions, unavailable sources, and authoritative misses have distinct truthful outcomes.
- Deterministic context-resolution and user-clarification examples resume successfully; changed locked dates/population/metrics/scope, stale replies, and hidden targets reject. Current authorization is reapplied.
- Mandatory continuation state cannot be trimmed into a fitting but semantically incomplete prompt.
- Ranking fixtures establish correct membership for multiple partitions, aggregate-before-ranking, ties, NULLs, derived ratios, aliases, grouping keys, authorization/masks, plan round trips, rollups/symmetric paths, final global caps, and semantic comparison. Unsupported paths reject before execution.
- Unknown fields fail at canonical query boundaries, including nested query inputs.
- Run targeted regressions, actual public API/CLI demos, native `just check`, API-break reporting, and real supported backend conformance. DuckDB alone is not proof of PostgreSQL/ClickHouse support. No live LLM is needed or implied.
- Update user-facing documentation and migration notes after behavioral proof. Open a PR with issues, intentional compatibility breaks, and exact verification evidence.

## Implementation and verification

Implemented the three agreed enhancements on `feat/planner-contracts-ranking`.
Issue #16 remains excluded. Package versions remain unchanged at `0.12.0`;
the changes are recorded under Unreleased.

- `SchemaProfile`, `SchemaProjection`, and `GenerationContract` pair schemas with
  matching guidance, expose projection losses, and canonically validate nested
  queries. Adaptive prompts, repair prompts, and provider exports use the same
  profile rules. Returned schema copies cannot mutate the contract.
- `PlannerResponse` distinguishes nonempty plans, context requests, user
  clarification, and bounded failure diagnostics. Host-owned continuation state
  correlates replies, permits only confirmed typed choices, preserves all other
  query constraints, and reapplies current scope and authorization.
- Lookup outcomes distinguish authoritative resolution, ambiguity, unavailable
  context, and authoritative misses. Candidate counts and truncation remain
  explicit; a single fuzzy candidate is not authoritative.
- `PartitionRank` operates on grouped results after aggregation, derived
  projections, and HAVING, before final ordering and global limits. Exact-N
  ordering discloses deterministic grouping-key tie breakers; include-ties uses
  score keys only. Ranking participates in logical plans and semantic comparison.
  Masked scores and tie breakers reject, including bare output references on the
  symmetric aggregation path.

Verification completed locally:

- Native `just check` with disposable backend conformance enabled:
  **3,507 passed, 4 skipped, 1 xfailed; 78 snapshots passed**.
  Ruff, mypy (280 files), Pyright, Tach, core-boundary, and frozen-model checks
  passed. The suite exercised real DuckDB, PostgreSQL, and ClickHouse queries,
  including ranking after rollup and symmetric aggregation.
- After strengthening the schema-copy regression to mutate nested definitions,
  the generation-profile module passed all 7 tests and its Ruff checks.
- Optional LangChain-enabled provider/profile tests passed all 25 cases.
- Public API probes validated all four outcomes against both projected JSON
  schemas, rejected missing payloads and invalid tuple values canonically, and
  executed ranking through time filters, dimension synonyms, and catalog views.
  Qualified and output-name time partitions returned the same rows and compared
  equivalent with explicit bound-context equality.
- `check_api_break.py` against base
  `187162ce86ac455ce9a803306222d6224fcb78bd` reported no static breaks in the three
  modules it checks (`semql`, `semql_mcp`, `semql_erd`). This does not cover the
  intentional behavioral and prompt-contract changes; those are documented in
  [the migration guide](../migrations/planner-contracts-ranking.md).
- All eight workspace packages built as wheels and source distributions.
  The existing artifact smoke installed the eight wheels together outside the
  workspace, checked dependency compatibility and import origins, and passed
  compile/serialization/alias-equivalence checks with actual DuckDB execution.

- Executable ranking, continuation, adaptive-prompt, and repair demos passed.
  Ranking returned exact winners `A/a/10` and `B/d/8`, adding `A/b/10` only for
  boundary-inclusive ties while excluding a private higher scorer. Definition
  continuation executed `net=15`; category clarification executed `gross=20`;
  changing a locked date range rejected. Repair used `PlannerResponse` and
  rejected an out-of-selection augmentation.

No live model/provider or BigQuery/Snowflake warehouse was used. Local artifact
builds are verification only, not a release or publication.
