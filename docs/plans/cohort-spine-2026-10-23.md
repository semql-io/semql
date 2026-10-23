# Single-hop unique-member cohort spine

## Scope and authorization

Implement issue #16 following the executable cohort spike. Use branch `feat/cohort-spine`, based on main `066c3d408e0f2f3b8e08e87aa9ca7215ba7c875b`, in the matching isolated worktree. Initial authorization covered implementation with a default subagent, commit/push, and a pull request. The user subsequently authorized a version bump, merge, tag, and publication. Prepare all eight packages as 0.14.0; merge only after final-head checks, then require merged-main CI before tagging and publishing through the existing OIDC workflow.

## Contract

A cohort is a bounded, uniquely keyed relation selected by an existing SemanticQuery. Source dimensions identify member keys and explicitly retained selection-time attributes. A member key can contain multiple typed dimensions. NULL keys and duplicate member keys fail, including a member selected under conflicting attributes. Deduplication and truncation are never substitutes for integrity.

Named measurements map each source key to a selected measurement dimension and use ordinary SemanticQuery filters, windows, measures, and inline-derived arithmetic. Each measurement must return at most one row per member key; any retained measurement-time attributes are distinct from the selection-time attributes and must be single-valued at member grain. Missing measurement rows preserve the member, with NULL values and explicit presence/coverage evidence. Existing selection ordering determines result order; no new implicit result order.

Comparisons explicitly reference two named measurements and compatible numeric outputs. Missing/undefined operands yield NULL comparison values; no implicit zero fill. Do not add arbitrary expression evaluation or reaggregation of finalized results. Additive components and non-additive metrics remain ordinary compiler concerns; calculate cohort-wide metrics over restricted facts rather than summing member metrics.

The input is a dedicated immutable, extra-forbid wrapper around existing SemanticQuery, not a recursive SemanticQuery field or a generic DAG. It can use existing schema projection through an explicit output_model; do not alter the default PlannerResponse contract or advertise unsupported recursive cohort plans.

## Architecture and execution

Core SemQL owns typed specification, pure validation/compilation, stage bindings, declared result shape, and selection/measurement/comparison provenance. semql-engine owns adapter execution, bounded materialization, member-key validation, alignment, comparison arithmetic, and coverage evidence. No I/O or sibling-package imports in core.

Use staged execution for same- and cross-backend relations, preserving existing Engine adapter routing. Compile/authorize all static stage shapes before any adapter execution. Fetch source with a non-silent bound sentinel, validate it, then bind exact member-key predicates into measurement queries before compilation/execution. Composite key membership must preserve tuples, not separate IN lists. Apply source/measurement row bounds through SQL and reject overflow. Validate measurement key uniqueness and membership before returning results; a dependent failure aborts the operation with no partial successful result. An empty cohort performs no measurement I/O.

Authorization is independently compiled for every stage under host-supplied viewer, policy, scope functions, views, and context. Masked source/target keys cannot be used for identity; masked attributes cannot be used for retained attribution. Runtime evidence must not invent authorization distinctions for absent rows. The result does not promise a source snapshot, durable cohort persistence, or a transaction across backends. Cache use follows the existing Engine contract and must not lose identity/provenance.

Reuse typed ContractError / ExecutionContractError and repository immutable value types. Validate versioned plans, bound parameters, stage output shapes, field types, and all source/key/attribute/measurement references using current catalog and compiler mechanisms. Never turn child SemanticAnalysis into a false claim that the composed result is semantically equivalent; expose explicit cohort provenance and conservative analysis coverage.

## Ownership

The default subagent owns implementation and focused behavioral regressions in core/engine, plus required boundary declarations. Parent owns final export integration, executable demo, README/changelog/migration guidance, verification, commit/push/PR. No subagent builds, linting, formatting, or test runs mid-flight; parent verifies once integrated.

## Acceptance

- Changed winners retain the original cohort; missing measurements preserve members.
- Selection-time and measurement-time attributes are distinguishable; multi-valued measurement attribution rejects rather than multiplying rows.
- Composite keys bind exact tuples with no cross-product leakage.
- NULL/duplicate/oversized source keys reject before dependent I/O; empty cohort skips dependent I/O.
- Measurement overflow, duplicate keys, unexpected keys, schema/plan/version failures, and dependent failures fail loudly, without partial results.
- Absent observations, observed NULL metrics, and zero-denominator ratios remain distinguishable through presence and supplied evidence; comparisons preserve NULLs.
- Current authorization, tenancy, field roles, masks, and scope callbacks remain effective in source and dependent stages.
- Model copies/serialization remain immutable and strict; schema profiles preserve required query payloads and enforce omitted nested features.
- No arbitrary DAG, nested cohorts, automatic finalized-result rollup, or implicit snapshot guarantee.
- Deterministic numeric fixtures and independent randomized membership/value oracles; actual DuckDB plus real PostgreSQL/ClickHouse adapter conformance for advertised support, not stand-ins alone.
- Native just check, targeted tests, public execution demo, and API-break reporting before commit/push/PR.

## Evidence

Implemented the contract with staged execution in core/engine, opt-in prompt
guidance, public exports, a runnable demo, and the cohort contract guide.
All eight packages and their internal dependency ranges are prepared at 0.14.0.
Publication is authorized but remains gated on the final PR/main checks and
release artifact verification.

### Observed verification

- Native `just check` with external backend endpoints unset and
  `SEMQL_CONFORMANCE_TESTCONTAINERS=1`: **3,616 passed, 8 skipped, 1 xfailed;
  78 snapshots passed**. Ruff, mypy (287 files), Pyright, Tach, core boundaries,
  and frozen-model checks passed.
- Targeted cohort and semantic-SQL regression run with DuckDB's optional `pytz`
  timestamp dependency supplied through `uv run --with pytz`: **207 passed**.
  This exercised native DATE, naive/aware timestamps, UUID, exact DECIMAL
  identity, and refusal of lossy decimal semantic-SQL rendering.
- Actual backend cohort conformance: **5 passed, 1 intentionally skipped**:
  same-backend DuckDB/PostgreSQL/ClickHouse and PostgreSQL/ClickHouse selections
  feeding DuckDB measurements. No warehouse stand-in or placeholder translation
  was used for those external backend cells.
- Independent fixed-seed membership/value oracle: **24 fixtures**, including
  changed winners, changed attribution, absent measurements, NULL inputs,
  zero denominators, and unequal-denominator rates.
- `uv run python demos/cohort_spine_demo.py` executed actual DuckDB SQL:
  original winners A/C/E/J/K remained selected, A retained selection region West
  despite current observations in East, E remained absent, J retained observed
  zero-denominator evidence, and K retained observed-NULL evidence. New winners
  B/D were not substituted.
- Public API checker against base
  `066c3d408e0f2f3b8e08e87aa9ca7215ba7c875b` reported no breaking changes
  in its checked surfaces (`semql`, `semql_mcp`, `semql_erd`).

The first PR CI run passed its executed tests/static/build/API checks but failed
the unchanged engine branch-coverage floor (82.27% versus 83%). Added real-driver
type-drift failure cases, Boolean identity, floating comparison zero-denominator
boundaries, and actual View-based execution, replacing internal binding-only
assertions. The CI-equivalent engine-only run then passed **325 tests** with
**84.16% branch-inclusive coverage**, retaining the 83% floor.

PR: <https://github.com/semql-io/semql/pull/21>. Authorized for merge and release
after the versioned-head checks pass.

### Review-driven repairs

Review and executable regressions established/fixed exact composite membership
without exponential CNF expansion, preserved source limits, native Decimal key
binding and full-precision comparison subtraction, malformed-projection
preflight, canonical View output bindings, independent stage authorization and
masked identity refusal, and preservation of output units/format/mask metadata.
Ratios/percent changes are dimensionless, and unrequested comparison-unit
inference cannot reject an otherwise valid ordinary measurement.

Grouped/dense-fill child time dimensions, measurement CompareWindow, nested
staged semi-joins, and aliases are explicitly unsupported rather than silently
changing member grain or dropping semantics. Plain time ranges remain supported.
There is no live model/provider call, snapshot guarantee, durable persistence,
generic finalized-result reaggregation, or arbitrary stage DAG.

### Version 0.14.0 release verification

The native lockstep bump updates all eight package versions and internal sibling
ranges to `>=0.14.0,<0.15`; `uv lock` records the same release.
Versioned native `just check` passed **3,616 tests, 8 skips, 1 xfail, and
78 snapshots**, plus lint/types/boundary guards and disposable actual backend
conformance. The cohort demo and API-break checker also passed.

All eight wheels and eight source distributions built successfully and passed
Twine metadata validation. Isolated wheel installation passed dependency checks,
version/import-origin validation, compile/serialization/alias-equivalence smoke,
and actual DuckDB execution. The cohort demo also executed from those installed
wheels outside the checkout, preserving original winners and missingness.

The final versioned commit must pass PR CI before merge, and the merged commit
must pass main CI before creating `v0.14.0`. Publishing uses the existing enabled
OIDC workflow; no credentials or publishing settings are changed.
