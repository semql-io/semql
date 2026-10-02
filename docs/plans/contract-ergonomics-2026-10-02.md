# Contract ergonomics and truthful guardrails

Date: 2026-10-02.

## Scope

Improve existing SemQL operations, not query-language breadth. Work in the `feat/contract-ergonomics` worktree; preserve the original checkout. Deliver a pull request with all eight packages bumped to 0.9.0 and sibling requirements updated in lockstep. Do not merge or publish this change.

## Implementation slices

1. Query budgets retain unknown cube identities, enforce known size-hint subtotals, and expose explicit unknown-cost admission policy. Prompt budgets explicitly report fit/cannot-fit and permit caller-supplied counting without a tokenizer dependency.
2. Render compiled semantic analysis into useful, safe explanations without another resolver. Carry diagnostic query locations where provenance is available, with bounded repair guidance and safe public payloads.
3. Add executable integration recipes and a 0.7-to-0.8 migration guide, with authenticated scope context, semantic comparison, enrichment, federation, and cache/stream responsibilities. Retain provider adapters.
4. Extend real-backend result cells for existing null, empty, duplicate-key, numeric/time boundary behavior and resource-cleanup coverage where missing. Add isolated distribution installation and execution smoke before release publication.

## Shared constraints

Reuse current architecture, diagnostics, test conventions, and analysis vocabulary. No new operators, dialects, optimizer, provider schema change, retry policy, or implicit result truncation. Public explanations must not reveal bound values or privileged catalog internals. Child agents do not run builds, linters, formatters, or tests mid-flight; integration owns verification.

## Verification and delivery

Run the repository canonical check with Testcontainers enabled, targeted real-backend fixtures, executable examples, installed-wheel smoke, distribution metadata checks, and workflow validation. Verify adverse budget scenarios fail truthfully. Update changelog and relevant user documentation after observed smoke results. Commit and push explicitly scoped changes, open a PR against main, and report verification and intentional compatibility changes.

## Observed verification

- `SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check`: format/lint, mypy (267 files),
  pyright, Tach, core-boundary and frozen-model checks passed. Full suite:
  **3,103 passed, 3 skipped, 1 expected failure; 83 snapshots passed**.
- `just test-conformance --no-cov -p no:tach`: actual DuckDB, PostgreSQL, and
  ClickHouse result cells, **43 passed, 2 intentional skips** (DuckDB-source
  variants of cross-backend cells, which require PostgreSQL/ClickHouse sources).
- All eight 0.9.0 wheels and eight source distributions built and passed
  `twine check`; isolated installation imported all packages, checked exact
  versions/sibling bounds, and executed serialization/alias-equivalence/SQL-result smoke.
- Integration recipes ran against installed 0.9.0 wheels from outside the
  workspace: authorized scope, semantic comparison with/without host attestation,
  alias-aware enrichment, explicit federation, cache namespaces, early-exit cleanup.
- Actual adverse-budget smoke rejected two unknown cube identities at `max_cubes=1`;
  oversized required prompt text remained intact with `fits=False`. Safe repair
  diagnostics exposed a filter location without the submitted private filter value.
- API diff against the 0.8.0 base passed after retaining positional validation
  constructor order and distinguishing documentation-only Pydantic descriptions
  from executable field changes. A separate filter smoke retained default,
  constraint, and constructor changes. This checker is not proof of behavior
  compatibility; the intentional admission/result changes are documented in
  `docs/migrations/0.8-to-0.9.md`.

No new query operators/dialects or provider-facing schema cutover is claimed.
Package publication, merge, and release tagging remain excluded.
