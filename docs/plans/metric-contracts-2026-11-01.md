# Metric contracts implementation—2026-11-01

## Decision and scope

Implement the accepted spike's incremental production design, not a public expression algebra. Existing compilation, authorization, query defaults and opaque metadata remain unchanged. Contract checking is explicit and opt-in. Mandatory unknown obligations fail closed; trusted catalog declarations never become runtime data-quality proofs.

Baseline: `origin/main` at `5b57ed4`, with implementation isolated on `feat/metric-contracts`. The existing dirty main checkout and the spike worktree are preserved. No version bump, merge, tag or release is part of this change.

## Required behavior

1. Frozen, schema-versioned metric contracts describe composite grain/FDs, required/permitted time axes and buckets, named population segments, required dimensions, non-additive axes, snapshots and soft preferences. Existing measures, ratio operands, filters, units and join declarations stay authoritative.
2. `semql.validate.check_semantics(analysis, contracts, expectations=None, *, require_confirmed_intent=False)` consumes existing compiler analysis without I/O, recompilation or SQL evaluation. Structural validity, declared semantic validity, intent alignment and lowering support remain independent. A model cannot independently confirm its own interpretation.
3. `Catalog` accepts a contract mapping; `CatalogSpec` stores immutable contract tuples, serialized by canonical reference and validated against actual catalog field kinds. The empty default preserves legacy payloads and ordinary compilation. `Catalog.check_semantics` delegates and separately reports successful lowering for a trusted same-catalog compiled artifact.
4. `attach_semantic_check` explicitly retains the full report—obligations, assumptions, advice, partial alignment, contract revisions and host intent evidence reference—on existing analysis, linked to artifact-local metric nodes. Ordinary compile never attaches or acts on a check. Retained annotations are not authorization or a substitute for fresh checking after input changes.
5. A private sufficient-state descriptor is consumed by existing federation mean fragment and merge lowering. SUM/non-NULL COUNT preserves metric filters, unequal group weighting and NULL behavior. This makes an existing recipe explicit; it adds no expression evaluator, distinct-set materialization, wire-format break or new backend support.
6. Six executed counterexamples cover ratio/mean, distinct overlap, snapshot time, fanout, time-axis mismatch, and fixed/independently selected populations. Calendar execution and complete cohort comparison alignment remain mandatory unknowns when requested without sufficient evidence.

## Modules and dependency directions

- `semql.contracts` is a foundation module containing frozen declarations, proof/advice/result types, and finite conditional FD closure. It does not import analysis, compiler, catalog, adapters, databases or LLM clients.
- Existing `semql.validate` exposes the pure checker; its private implementation is `semql._semantic_check`. It uses existing `SemanticAnalysis` and safe `Diagnostic`, not another query or provenance graph.
- `semql.catalog` owns registry validation, persistence and convenience checking. Business-calendar identifiers are opaque references, not catalog time-field names.
- `semql.analysis` owns additive check annotations and artifact-local links. Reports retain conditional declarations separately from unverified physical uniqueness.
- Existing `semql.federate` owns its private mean-state descriptor and consumes it in the current lowering path. Existing aggregate definitions and merge IR remain the execution contract.
- The executable demonstration lives in `demos/metric_contracts_demo.py`; actual execution assertions live in `test_metric_contract_execution.py`. The engine regression independently checks weighted filtered means with NULL and all-NULL groups.

By the deletion test, the shared checker removes duplicated host-side constraints and partial-intent rules. A second expression DAG would add catalog lifting, identity, serialization and lowering responsibilities without establishing another required capability. It is therefore excluded.

## Compatibility and trust

The registry and checker are opt-in. Empty-contract compilation does not change SQL, parameters or output metadata. CatalogSpec round-trips preserve contracts; caller mappings are detached rather than mutable shared state. References must resolve to real measures/dimensions/time dimensions/segments; contracts cannot smuggle an alternate predicate language through metadata.

The host controls declarations, expectation origin and independent evidence references. Low-level checking trusts authentic analysis and its matching catalog snapshot. It does not authenticate supplied reports, compare private bound values, execute business calendars, prove arbitrary predicate equivalence, or verify physical keys/freshness/completeness. Confirmed alignment covers only explicit expectation clauses; mixed inferred/confirmed clauses require separate checks. Local population IDs never establish cross-artifact identity.

## Observed verification

Run in the feature worktree with Python 3.12 and frozen dependencies:

- `SEMQL_CONFORMANCE_TESTCONTAINERS=1 UV_PYTHON=3.12 just check`: **3641 passed, 8 skipped, 1 xfailed; 78 snapshots passed**. Ruff, mypy (291 files), Pyright, Tach, core-package and frozen-model boundaries passed. Disposable PostgreSQL/ClickHouse conformance was enabled. Core branch-inclusive coverage was 91% (rounded).
- Focused checker/catalog/analysis/engine integration run: **105 passed** before the final full-report annotation refinement, which is covered by the full run above.
- `uv run --frozen --python 3.12 scripts/check_api_break.py --base origin/main`: no breaking changes in the three checked public packages.
- `uv build --package semql`: wheel and sdist built at the unchanged 0.14.0 version; not published.
- The public guide example compiled and executed actual DuckDB SQL, excluded the half-open end boundary, and round-tripped its registry. Semantic check annotations round-trip in the behavioral regression, retaining an unproved calendar gate and confirmed intent evidence.
- The executable demo matched independent oracles: ratio 2.5 vs mean 2; distinct subtotals 4 vs union 3; snapshots 220 vs latest 120; fanout 100 vs 200; time-axis monthly values (10,20) vs (20,10); fixed current cohort 20 vs independently selected 119, with changes 3 vs 102. Fixed members remain `a,b`, missing `b` remains NULL, and full cohort comparison alignment stays unknown.

Existing deprecation/security-fixture warnings were not suppressed. There is no empirical LLM benchmark or claim of universal backend conformance.

## Incremental adoption and excluded exploration

Adopt declarations and pure checking first, expose only model-safe expectation inputs to planners, and let the host decide whether confirmed intent is mandatory. Consume sufficient states through existing execution paths rather than allowing finalized-value rollups. A general public algebra, approximate sketch protocols, optimizer, workflow engine, new SQL compiler, calendar executor and runtime evidence adapters are separately gated future capabilities, not incomplete portions of this implementation.
