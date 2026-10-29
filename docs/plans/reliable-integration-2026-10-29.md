# Reliable integration contracts

## Decisions and scope

Implement the accepted source-neutral integration lessons without changing SemQL's trust model. Programmer-authored catalog declarations remain trusted; there are no new proof requirements, runtime certification gates, or waivers. Existing authorization and known-invalid execution checks remain intact.

1. Loss-aware introspection: retain complete source key structure, report unsupported/heuristic mapping explicitly, deterministic output, and preserve the supported scalar cube key model. Do not invent foreign import formats or extend compiler key semantics.
2. Provenance: distinguish source-format identity, host semantic revision, and deterministic catalog content integrity. Digests are descriptive—not authorization, callable-code identity, or execution gates. Preserve opaque user metadata and local analysis IDs.
3. Execution controls: opt-in limits for admission, elapsed execution, materialized/returned rows and value-byte bounds; preserve unlimited programmer-controlled defaults. Cover sync, async, cache, stream and staged/entity consumers that reuse the execution boundary. Document cooperative deadlines versus actual driver cancellation; permits remain owned by still-running blocking work. Reject overflow rather than silently truncate. Complete typed preflight failures before any source calls.
4. Conformance: reusable observable adapter assertions, fault-provoking harness tests, actual-backend bindings, explicit required-environment behavior, and a reproducible backend/query-shape evidence matrix distinguishing compiler-only, stand-in and actual backend cells. No universal backend claim.

## Ownership and integration

Introspection owns only semql-introspect. Provenance owns core catalog/analysis identity additions and their tests. Execution owns semql-engine production code and control/refusal tests. Conformance owns shared test support, adapter conformance tests, existing actual-backend conformance module and evidence matrix. Parent owns public documentation, changelog, this plan and final integration. Public export changes require LSP reference inspection. No source-project identifiers in published documents.

## Verification

Read relevant symbols and references before edits. Keep consumer-visible boundary/transition regressions, not source-text or wiring tests. Workers skip builds, tests, lint and formatting during edits. After all changes integrate, run complete affected test modules/package suites and native formatting/lint/types/boundaries; fix observed failures and rerun affected complete modules. Exercise public imports/catalog identity, actual DuckDB introspection/execution, timeouts with continuing work, overload, size overflow and real PostgreSQL/ClickHouse conformance. Record observed results only.

The user subsequently authorized opening a PR, with reviewer-agent review first
and no source-project references in repository content or PR metadata. Integration
uses `.worktrees/feat/reliable-integration`. No merge, version bump, or release is
authorized.

## Compatibility

- Custom schema probes return tuple-valued primary keys and grouped foreign-key
  constraints. Generated catalogs retain reports instead of scalar key loss.
- `SemiJoinPlan` version 2 requires keyword-only `outer_plan` for preflight of
  binding-only source dependencies. Existing positional fields keep their order.
  Recompile version-1 artifacts; do not fabricate a template or change its marker.
- The native API checker against baseline
  `e97a50ba97272fc73078ed726098c5389b9302bf` reports the required `outer_plan`
  parameter as one intentional breaking change. It is documented, not waived
  or suppressed. The current checker covers core/MCP/ERD and does not enumerate
  the custom-probe interface changes.

## Verification results

Final integrated native check, with PostgreSQL and ClickHouse required:

```sh
SEMQL_CONFORMANCE_REQUIRED=1 SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check
```

- Ruff formatting/lint passed; mypy checked 292 files; Pyright reported zero
  errors. Tach, core-package boundaries and frozen-model checks passed.
- Full suite: **3,693 passed, 8 skipped, 1 xfailed; 78 snapshots passed**.
  Actual PostgreSQL 16.14 and ClickHouse 26.8.2.7 fixtures executed; the
  [matrix](../specs/adapter-conformance-matrix.md) bounds those claims.
- CI-equivalent package coverage, without external services: engine **85.80%**
  against its unchanged 83% floor (**388 passed, 108 skipped**); introspection
  **87.25%** against its unchanged 83% floor (**25 passed**). External backend
  skips in that coverage-only run are not used as real-backend evidence.
- Public DuckDB smoke exercised composite-key reporting and generated Python,
  provenance/spec reconstruction, stable and changed digests independent of
  host revision, actual grouped sums, sync/async execution and streaming,
  exact-cap success and typed overflow.
- A public async driver-timeout probe first reproduced misclassification as an
  engine deadline and then confirmed preservation of the original `TimeoutError`.
  Regressions also execute cross-task stream closure, late-result disposal,
  blocked merge/fetch ownership, multiworker admission retention, empty-result
  deadlines and filter-only semi-join adapter preflight.
- Reviewer-agent final source review found no remaining actionable defects
  after fixes. It did not run checks; runtime results above are separate evidence.
- Changed-file source-neutrality and documentation local links were checked.
  No package versions, dependency ranges, release tags, or credential settings
  were changed.
