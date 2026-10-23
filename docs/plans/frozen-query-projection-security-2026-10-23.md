# Frozen query specs and trusted projection security

## Scope and baseline

Implement the two verified review findings on fetched `main` at `ff26b1a`
(version `0.11.1`), isolated in `.worktrees/fix/frozen-spec-plan-mask`.
Preserve the original dirty checkout and all its tracked/untracked files.
User authorized implementation by default-level subagents, a lockstep minor
version bump, and a normal push to `main`. No publication or tag push.

## Design

1. Reuse `semql.model.FrozenModel` and existing immutable `Sequence`/`Mapping`
   collection conventions for query/spec models. Preserve list/dict inputs and
   ordinary serialized payloads. Update consumers rather than adding mutable
   compatibility shims. Validated copies must uphold model invariants.
2. Before projection authorization/emission in `compile_plan`, restore field
   security attributes from the trusted catalog. Preserve privileged physical
   source/expression rewrites and aliases; keep emitted values, mask metadata,
   and semantic analysis consistent. Cover both dimensions and measures.
3. Use the native bump recipe to advance all eight packages to `0.12.0`,
   regenerate the lockfile, and document the immutable-spec migration.

## Acceptance and verification

- Nested query/filter/Boolean/derived/alias containers cannot change in place
  or through retained caller-owned values. Invalid copies reject.
- Forged projection role/mask attributes cannot cause raw sensitive values to
  be emitted. Actual DuckDB results agree with metadata and analysis.
- Existing physical optimizer rewrites and valid query transformations work.
- Run complete affected test modules, full repository checks, both public
  prompt demos, public API compatibility reporting, and direct behavior smokes.
- Inspect every affected caller/duplicate pattern and the integrated diff for
  minimal scope; finish with a normal fast-forward push, not force-push.

## Original-checkout inventory

The separate original checkout has 53 modified tracked and 15 untracked files.
Comparison with the fetched baseline found 52 exact matches, 12 older divergent
variants, and four local-only documents. None are inputs to this change.

## Completion evidence

- Two default-level implementation subagents implemented the independent
  spec-immutability and projection-policy fixes; integration retained the
  existing frozen-model convention and physical optimizer contract.
- A pristine `ff26b1a` source probe reproduced both failures: mutating a Boolean
  conjunction changed an actual DuckDB sum from `7` to `107`; a forged masked
  projection returned raw email addresses despite masked metadata/analysis.
- The final direct probe blocks nested mutation, rejects an invalid Boolean
  copy, retains sum `7`, and returns only `REDACTED` for the forged projection,
  with matching masked metadata and analysis.
- Validated copies exposed unchecked internal consumers. Both federation
  routing strategies now put singleton predicates in `filters`; the negative
  property generator keeps its spec structurally valid; saved-query keyword
  normalization runs before freezing. Immutable alias maps are accepted by the
  semantic-SQL serializer. These are required caller migrations, not new query
  semantics.
- Two incidental predicate-representation snapshots and their tests were
  removed rather than pinning immutable-container implementation details.
  Executed filter and Boolean behavior remains covered throughout the suite.
- Complete affected test modules passed, followed by
  `env -u SEMQL_CONFORMANCE_POSTGRES -u SEMQL_CONFORMANCE_CLICKHOUSE
  SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check`: **3,427 passed, 4 skipped,
  1 xfailed; 78 snapshots passed**. Ruff, strict mypy (271 files), Pyright,
  Tach, core-boundary and frozen-model guards passed. Backend conformance used
  the repository's disposable PostgreSQL/ClickHouse fixtures plus DuckDB.
- Both adaptive/repair demos passed. The native API reporter against
  `ff26b1a` reported no static breakages; the behavioral collection/copying
  migration is intentionally documented for the pre-v1 minor bump.
- All eight manifests and lock entries are `0.12.0`, with internal ranges
  `>=0.12.0,<0.13`. All 16 local wheel/sdist artifacts passed Twine; the
  installed-wheel smoke passed all package versions/import origins, dependency
  compatibility, compile/serialization/alias equivalence, and actual DuckDB
  results in a temporary environment.
- Original-checkout fingerprints for all 68 dirty files remained unchanged.
  No original-checkout cleanup, tag, package publication, or GitHub Release
  was performed.
