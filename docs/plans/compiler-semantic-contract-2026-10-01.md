# Compiler semantic analysis and diagnostic contract

Date: 2026-10-01.

Status: implemented in the working tree; verification results are recorded below. The first release uses artifact-local IDs and structured descriptor comparison, trusts host catalog declarations for execution, and treats logical plans as privileged inputs. Package publication, version/dependency updates, and external consumer migration are not included in this implementation.

## Decision

Ship a bounded first release covering semantic output identity, lineage for existing supported operations, typed diagnostics, and executable parameter integrity. Add partitioned ranking, population-aware aggregation, and time-completeness capabilities independently after their execution gates pass.

SQL and analysis must be projections of the same effective validated plan. Analysis must not resolve the request again, parse emitted aliases, rerun scope functions, or claim that successful logical resolution establishes concrete lowering support.

The first release is a correctness contract, not merely additional presentation metadata. Every requested output must be represented in the final executable projection and analysis, or compilation must fail explicitly. Known-invalid transformations cannot be labelled merely unproven.

### Agreed contract decisions

- `LogicalPlan` is a trusted host/optimizer input; untrusted callers submit `SemanticQuery`.
- Existing symmetric aggregation uses observed-fact population, not implicit entity-universe expansion.
- Trusted catalog declarations are sufficient to execute. Unchecked declarations remain assumptions; implemented runtime checks still run and known violations reject. No default rejection or host-waiver mechanism is added merely because a declaration cannot be checked.
- The full documented first release remains in scope; new analytical operators ship independently.
- Catalog namespace and semantic revision are supplied by the host. Missing revision permits local analysis but prevents persisted/cross-process equivalence claims.
- IDs are local to an analysis artifact. Structured descriptors and their comparison behavior are the public semantic contract; optional persistent semantic IDs may be added later.
- Duplicate semantic projections are rejected. Presentation can display one output multiple times.
- Descriptor meaning and comparison behavior are stable within an analysis version. ID strings need not match across independent compilations.
- Every successful compilation attaches analysis with explicit coverage; consumers requiring complete analysis reject not-established coverage.
- Lossless internal error payloads remain available. Built-in external surfaces use a separate safe public diagnostic rendering.

## Release boundary and work packages

The first release has three independently verifiable work packages, followed by consumer integration. They are not one compiler rewrite:

| Package | Deliverable | Dependencies and completion boundary |
|---|---|---|
| WP1 — Correctness repairs | Requested-output accounting, correct count-rollup recipe, alias preservation, specified symmetric population, and executable time-only fill | Reproduce D1–D4 and D8 as independent regressions first. Repairs may ship before analysis; none may rely on analysis to hide a wrong result. |
| WP2 — Semantic contract | Artifact-local nodes/output IDs, structured comparison, result population and grain, existing expression recipes, diagnostic vocabulary, safe rendering, and serialized analysis | Enforce the privileged logical-plan boundary. Preserve logical provenance while WP1 changes physical recipes. D7/D10 checks share the compiler path. |
| WP3 — Execution contract | Final binding manifests, executable-version checks, bounded merge-input obligations, and current-artifact analysis on executed/cache-returned rows | Uses WP2 descriptors. D5/D6 and obligation failures must be typed; sync/async paths have the same contract. |
| WP4 — Consumer integration | Identity-based enrichment, supported-minimum update, and deliberate removal of proven-redundant wrappers/reconstruction | Requires the exact WP1–WP3 conformance cells used by that consumer. D9 is included; external consumer migration does not block publication of a conformant library artifact. |

Compiler and executor package changes remain subject to the existing lockstep release convention. WP1–WP3 plus library-side enrichment are the first-release gate. New ranking, general population-scoped/two-stage aggregation, entity × time fill, and calendar-alignment operators are separate releases. General unit inference, live-database validation services, and provider-schema replacement are not part of this release.

### Supported-shape matrix

This is the **target release matrix**, not a claim of current support. “Include” requires analysis, typed failures, and actual result fixtures for the cell. PostgreSQL, ClickHouse, and DuckDB are the initial conformance backends; a backend name alone never enables a cell. Existing other-backend paths remain available with accurately limited evidence, but are not advertised as execution-conformant by this plan.

| Query shape | Single-backend target | Federation target | First-release disposition |
|---|---|---|---|
| Catalog dimensions and sum/count/min/max/average at an explicit group grain | Include on each conformance backend | Include existing legal distributive recipes; average uses partial sums/non-null counts | Preserve logical metric descriptors across physical stages; reject unsafe joins. |
| Existing distinct-count/percentile aggregates | Preserve currently legal lowering; fixture each advertised backend cell | Preserve existing explicit raw-row recipes, not distributive approximation | No new aggregate algorithms or approximate substitutions. |
| Catalog-declared ratio | Include existing legal operand aggregates | Include explicit raw-row recipe; distributive mode rejects | Never automatically switch to raw-row streaming. |
| Grouped inline sum/diff/ratio | Include existing legal forms with hidden operands | Explicit typed rejection for multi-backend inline-derived forms in this release | D1 fix is fail-closed accounting, not new federation algebra. A single-fragment federated artifact may use the supported single-backend lowering. |
| Scalar inline-derived-only request | Typed rejection | Typed rejection | Retain the present shape limitation with a specific reason, not “empty query.” |
| Aliases and output reordering | Include for every included shape | Include for every included shape | No positional reconstruction; presentation mappings remain accurate. |
| Existing single-cube rollups | Include only valid declared recipes, including SUM of stored counts | No new combined rollup/federation capability | D3 is a numerical repair. Do not infer arbitrary combinations from the individual features. |
| Existing conformed-bridge symmetric additive aggregation | Include observed-fact population | Include the same observed-fact population | D4 compatibility decision below; entity-universe expansion is not implicit. |
| Existing time windows and comparisons | Include existing legal forms and current per-output policies | No new cross-backend compare lowering | Report actual existing restrictions, not generic backend support. |
| Time-only dense fill | Include existing time-only shape after D8 repair and per-backend execution proof | No new cross-backend fill lowering | Do not change null policy while repairing SQL typing. |
| Entity × time fill, partitioned ranking, general two-stage/population-scoped aggregation | Not added | Not added | Explicit capability rejection; independent operator releases. |
| Serialization, cache hits, and label enrichment | Include for supported outputs | Same contract | No compiler I/O; runtime-added labels carry separate provenance. |

Rows are not a Cartesian-product promise. Combined compare/fill/rollup/derived shapes require their own capability decision and fixture. Every successful compilation attaches analysis, including existing shapes outside this matrix. Coverage explicitly distinguishes an established contract from not-established analysis; unsupported executable shapes reject. Consumers requiring complete analysis reject not-established coverage. Unrepresentable lineage is never filled with guessed semantics. Known silent omission or invalid lowering is always repaired or rejected, regardless of matrix coverage.

The actual-backend test matrix must enumerate query shape, source dialect, federation mode/merge backend, and combination restrictions. A cell cannot be marked conformant using DuckDB placeholder-translating stand-ins.

### Population decision

For the existing conformed-bridge symmetric additive shape, select **observed facts** as the cross-strategy default: a bridge key participates if at least one contributing fact has an authorized, pre-aggregation-filtered row for that key, and the key satisfies the applicable bridge restrictions. An event whose measure is null still establishes presence. Post-aggregation predicates and selection apply afterward. Unmatched keys follow the existing declared join semantics; this decision does not broaden joins.

The reason is semantic, not preference for one emitter: aggregation over qualifying facts does not request expansion to every entity in a dimension table. D4 therefore changes the federated path to omit entity 4, with a documented membership compatibility change. Both paths must retain entities present in only one fact. Null measure values must not be used as a proxy for missing fact rows.

Analysis reports this inclusion rule separately from grain. It does not assert that all single-fact joins use a union population; those retain their selected join/predicate semantics.

An **entity universe** is a separately requested population rooted in a trusted, authorized entity relation and its predicates. General requests for this expansion, and separately scoped numerator/denominator populations, belong to the later population operator. The first release neither guesses them from a dimension projection nor changes denominators to simulate them. Callers currently relying on implicit federated universe expansion must retain their service-owned population path until that explicit capability is available.

### Defect disposition and acceptance mapping

The evidence is retained in Appendix A. Recreate each accepted defect as a minimal failing regression before editing its implementation; the unretained spike scripts are not reproducible regression assets.

| Finding | Package / disposition | Required acceptance |
|---|---|---|
| D1 — Omitted inline ratio | WP1: typed rejection for multi-backend inline-derived queries; no partial artifact. WP2: output accounting | The multi-backend request containing `share` fails with the inline-federation reason. Preserve grouped single-backend `0.2/0.9`; scalar derived-only has its own reason. |
| D2 — Lost aliases | WP1: fix | Reordered raw-row ratio fixture returns `region, rate, mean_amount, net` and `R, 0.375, 37.5, 150`; WP2 descriptor comparison establishes equivalent original outputs without comparing local ID strings. |
| D3 — Count rollup | WP1: fix physical recipe | Both base and selected rollup return A count 3 and B count 1; analysis retains logical COUNT and records physical SUM. |
| D4 — Population mismatch | WP1: compatibility change to observed-fact membership | Both paths return entities 1–3, not 4; include a present-all-null fact row and authorization exclusions. Do not bless legacy federation as the oracle. |
| D5 — Unbound fragment parameter | WP3: reject before any adapter | Missing and renamed binding cases report artifact/name, expose no values, and invoke zero source adapters in sync/async execution. |
| D6 — Ignored plan version | WP3: reject unsupported versions | Version 999 fails before cache use or adapters; supported-version execution and round-trip remain valid. |
| D7 — Validation/lowering disagreement | WP2: share applicable checks and expose coverage | Entity × day fill has the same unsupported-shape category in collecting validation and compilation. Static-only validation must not report executable support. |
| D8 — Unexecutable dense fill | WP1: fix temporal lowering/encoding | Execute the time-only fixture on each advertised conformance backend with untouched caller bindings; independently assert bucket dates, sums, and counts, including present-null and missing buckets. |
| D9 — Alias-sensitive enrichment | WP4: extend identity-based matching | Both `region` and `area` obtain their labels; A/B remain distinct despite equal labels; original local references, metric descriptors, row count, and grain remain unchanged within each enriched artifact. |
| D10 — Untyped fanout failure | WP2: structured check-site details | Safe distinct counts remain x=2/y=1 with an advisory; unsafe SUM fails with metric/join references and a bounded reason without message parsing. |

Reconstruction must capture catalog, query, fixture rows, strategy, expected result/error, and backend version. Existing full test modules are regression guards; their presence is not evidence that they cover these cases. A finding disproved by reconstruction must be corrected in the evidence record, not forced into a passing test.

## Scope and evidence boundary

The spikes used throwaway scripts launched with `uv run --no-sync python -B` and in-memory DuckDB connections. No source, test, dependency, or release files were changed by those experiments. The scripts were not retained as repository verification assets.

Federated execution used the existing test pattern: separate DuckDB connections stand in for PostgreSQL and BigQuery, with named-placeholder translation at the adapter boundary. These observations exercise compiler partitioning, fragment execution, and merging; they do not establish actual PostgreSQL, BigQuery, or ClickHouse conformance.

The full test suite, actual-backend suites, package build, and external consumer integration were not verified. Successful compilation and source snapshots are not execution evidence.

Relevant architectural constraints are in [PHILOSOPHY.md](../../PHILOSOPHY.md): compilation has no I/O, authorization belongs in the compiler, and executors and federation execution remain outside core. The existing [hardening plan](production-library-hardening-2026-09-19.md) and [plan-trust security note](../specs/compile-plan-field-masking-security-note-2026-09-19.md) motivate an explicit boundary: caller-provided logical plans are privileged host inputs, not an untrusted request format.

## Concrete use cases

These are concrete target query shapes and fixture cases, not evidence of externally verified demand. Before adding an operator, confirm multiple consumer needs for it.

| Use case | Required distinction |
|---|---|
| Compare activity time and worklog hours per employee across sources | Observed fact population versus the authorized employee universe; employees missing from one or both facts; fan-safe aggregation |
| Select the most-active entity over a reporting window | Entity grain versus entity-day grain; full candidate aggregation before selection |
| Select top contributors per day or top customers within each region | Independent partitions, deterministic exact-N selection, and explicitly chosen tie semantics |
| Calculate revenue per customer or average activity per employee | Event averages versus entity-total averages; zero-event entities; explicit denominator population |
| Compare one reporting month with the previous calendar month | Calendar alignment versus equal-duration shifting; partial periods and reporting timezone |
| Produce an entity × day series including inactive entities | Explicit bounded entity universe, dense bucket generation, and distinct missing-row/null/zero policies |
| Enrich keys with names after execution | Aliases do not break lookup matching; duplicate labels do not collapse distinct keys |
| Reuse cached rows for differently named catalog metrics | Physical value reuse does not return another request's semantic identity or validation evidence |
| Consume compiled artifacts across package/process boundaries | Analysis survives serialization; unsupported contract versions and missing bindings fail before execution |

No report-specific names or employee/task/punch operators belong in the operation algebra. Service-owned cohort definitions, authenticated identity, and admission policy remain host responsibilities.

## Proposed first-release contract

### One resolution and validation pipeline

Reuse `_resolve.walk_query_fields`, the logical plan, trusted-catalog canonicalization, and existing lowering decisions. Compilation fails at the first failure; collecting validation reports the failures it can determine. Neither path establishes support without applicable concrete lowering checks.

Attach analysis after effective transformations and output definitions are established. Preserve semantic origin explicitly through rollup/partition/federation transformations; do not infer original metric identity from rewritten field SQL.

Do not initially ship a separate analysis-only entry point unless a concrete caller needs it. If added, it shares resolution, authorization, semantic checks, output definitions, and capability decisions; it may omit rendering, not those guarantees.

#### Privileged logical-plan boundary

`compile_plan()` accepts plans only from the trusted host/optimizer. Physical rollup, partition, and federation rewrites must survive emission. External adapters accept untrusted `SemanticQuery` input and must not deserialize or expose caller-supplied `LogicalPlan` objects through request tools or endpoints.

This release does not implement a verifier for arbitrary untrusted plans or claim that a plan's familiar catalog names prove its expression/policy provenance. Trusted optimizer code is responsible for preserving semantic origins and policy-bearing metadata; compilation retains its existing authorization and consistency checks. Privileged status is not a reason to delete those checks or loosen the normal `SemanticQuery` boundary. Document this contract on the public entry point and verify adapter isolation before making authoritative lineage claims.

### Shared identity and expression vocabulary

Use existing `QualifiedRef` for catalog fields. Its grammar is exactly `cube.field`; generated references such as `compare.amount.delta` are not catalog fields.

Use a small typed expression graph for existing aggregates, arithmetic, comparisons, and their effective null/applicability policies. Output descriptors reference semantic nodes. Every operand ID resolves to a node, including hidden operands not projected to the result.

Distinguish:

- Output address: the projected occurrence a consumer refers to.
- Semantic definition: metric/expression, operand scopes, stages, units, and period roles.
- Bound query context: effective cohort, reporting interval, and authorization scope.

Use opaque artifact-local IDs. A `node_id` addresses a semantic node within its containing analysis, and an `output_id` addresses a projected output that references a node. Population and binding-dependency references have the same artifact-local scope. Allocate them deterministically during compilation, without randomness or external state, but do not promise identical strings across compiler versions or independently transformed inputs. No public hash format or persistent ID-generation algorithm is specified in this release.

IDs are not aliases, output positions, metric definitions, or credentials. They remain intact through the artifact's supported serialization round-trip and execution/enrichment propagation; transformations of that artifact preserve references to unchanged nodes and outputs. Independent compilations may assign different IDs even when their results have equivalent semantics. Consumers must not persist a bare local ID as a reusable metric reference or compare ID strings across artifacts. Persisting an entire artifact with its local references is supported.

Descriptors contain node kind, qualified catalog references, logical aggregation stages, operands, storage-unit declaration, period role, null/applicability policy, and population/binding dependencies. Catalog namespace and semantic revision are envelope context supplied by the trusted host. The host changes the revision when SQL definitions, metric/filter/unit semantics, or scope-function behavior changes. Rollup/federation rewriting retains the original logical catalog context rather than inventing a new revision for physical sources.

Without a host-supplied immutable semantic revision, analysis and within-artifact traversal remain available. Comparison may use an explicitly shared immutable trusted catalog snapshot in-process, but persisted/cross-process equivalence is not established. Matching field names or Python object identity alone is insufficient evidence of unchanged catalog semantics. A missing revision does not make ordinary execution invalid.

Reject duplicate projections of the same semantic node with a typed duplicate-output reason. Detect duplication from the resolved semantic definitions, not alias or ID-string coincidence. Displaying one result twice is a presentation concern. Output IDs and node IDs are separate reference domains; consumers must follow `output.node_id` rather than assume their strings are equal.

#### Structured comparison contract

Provide one public structured comparison API returning `equivalent`, `different`, or `not_established`, with structured reasons. It compares logical descriptors, not raw JSON, aliases, output positions, local ID strings, or physical execution plans. Consumers should use this API instead of implementing their own canonicalization.

Comparison follows operand, population, and binding-dependency references in each artifact and matches their definitions. Thus `ratio(node_2, node_3)` may be equivalent to `ratio(node_8, node_9)` when the referenced descriptors agree. Compare ordered operands in order; grouping sets are order-insensitive. This release does not prove general algebraic equivalence. The reference graph must be acyclic and complete; dangling references or invalid artifacts produce typed validation failures, not a semantic `different` result.

Output-expression comparison and whole-result comparison are distinct scopes. Whole-result comparison additionally checks grain, population inclusion, predicates, time policy, selection, and the requested outputs; aliases and SELECT-list order are presentation differences, while row-order/limit/tie semantics are not. Neither scope treats physical partial/merge stages as a change in logical aggregation.

Compare logical binding dependencies by predicate/time/scope definitions, not transient parameter names or local slot IDs. Values remain in private execution context. Whole-result equivalence requires the caller to provide compatible bound context through a private comparison input or a trusted equality attestation. Missing context, incompatible/unknown schema coverage, opaque unresolved semantics, or incomparable catalog revisions yield `not_established`; a known comparable semantic mismatch yields `different`. Equality remains conditional on declared catalog assumptions and does not prove source freshness or identical live data.

Descriptor meaning and comparison behavior are stable within an analysis version. Internal refactors may change local ID allocation, but not the meaning or documented comparison outcome of the public descriptors. Behavior-changing descriptor revisions require an explicit analysis-contract version change. Tests pin comparison outcomes and reference integrity, not incidental ID spellings.

#### Compatible extension to public semantic IDs

A later release may add an optional persistent `semantic_id` without changing `node_id`, `output_id`, or local-reference scope. Existing consumers continue traversing local references and comparing descriptors. New consumers may use the public ID when available and fall back to structured comparison when absent. The persistent scheme can require a catalog revision even though local-only artifacts do not.

Descriptor comparison remains authoritative. Do not replace local IDs with public hashes, infer public identity from matching local strings, or claim that matching public IDs alone proves equal bound populations, authorization context, or actual data.

Supported decoders must tolerate unknown optional annotation fields, so an advisory `semantic_id` can be additive. Unknown semantic node kinds, required semantics, or incompatible contract versions cannot be ignored: reject or report not-established comparison coverage as appropriate. Readers with a strict older schema require a supported older representation or explicit version negotiation; “additive” does not promise compatibility with arbitrary decoders. No public-ID algorithm or generation API is implemented in the first release.

#### Example projection and identity rules

The following YAML illustrates a version-1 artifact. `n1`, `n2`, `n3`, `o1`, and `p1` are local references, not stable public identities. The descriptors are shown on their nodes; they are compared by meaning rather than by literal reference strings:

```yaml
schema_version: 1
catalog_context:
  namespace: reporting
  semantic_revision: r17
nodes:
  - node_id: n1
    kind: aggregate
    catalog_ref: events.amount
    aggregate: sum
    logical_stage: event_to_group
    population_ref: p1
    period_role: current
    unit: {state: declared, value: currency}
    null_policy: ignore_null_inputs
  - node_id: n2
    kind: aggregate
    catalog_ref: events.views
    aggregate: sum
    logical_stage: event_to_group
    population_ref: p1
    period_role: current
    unit: {state: declared, value: count}
    null_policy: ignore_null_inputs
  - node_id: n3
    kind: arithmetic
    catalog_ref: events.share
    operation: ratio
    operand_ids: [n1, n2]
    zero_denominator: "null"
    null_policy: propagate
    unit: {state: unproven}
outputs:
  - output_id: o1
    node_id: n3
    sql_alias: rate
result:
  grain: {group_refs: [], kind: scalar}
  population_ref: p1
populations:
  - population_id: p1
    inclusion: observed_facts
    source_refs: [events]
    predicate_dependencies: [cohort_filter, authorization_scope]
binding_dependencies:
  - slot: cohort_filter
    binding_source: private_executor_context
  - slot: authorization_scope
    binding_source: private_executor_context
```

This describes a catalog-declared scalar ratio, not the unsupported scalar inline-derived-only request. The binding entries indicate private dependencies; they do not serialize binding values. Execution derivation separately relates these nodes to fragment columns/recipes. Native AVG and a physical partial SUM/COUNT plus merge can describe the same logical event-average node, even if their independent artifacts assign different local IDs.

| Change | Local-reference behavior | Structured comparison |
|---|---|---|
| Rename alias, reorder projected outputs, attach a label | Preserve existing references within an artifact; independent compilation has no string-equality promise | Original outputs remain equivalent; label provenance is additional |
| Use base table versus valid rollup, or native AVG versus valid recomposition | IDs are local in each artifact | Equivalent with matching population/context and valid recipes/assumptions |
| Change metric reference or logical event-average to entity-total-average | No cross-artifact ID inference | Different logical descriptors |
| Change current to prior role, null policy, or logical operand-population rule | No cross-artifact ID inference | Different semantics |
| Change a cohort/time/authorization value for an otherwise equivalent dependency | IDs do not encode private values | Whole-result context differs when known; without private context, not established |
| Change catalog semantic revision without changing field names | Local strings may coincide accidentally | Not established across revisions absent trusted compatibility knowledge |
| Duplicate an identical semantic projection | Rejected in version 1 | No alias-based escape or positional occurrence identity |
| Serialize and reconstruct the same artifact | Preserve local IDs and every graph reference | Same descriptors and context; no new source-validation evidence |
| Add an optional persistent `semantic_id` in a future version-compatible representation | Existing local references keep their meaning | Existing descriptor comparison still works without the annotation |

Reuse this vocabulary for `CompiledQuery` and final `FederatedPlan`/`MergeSpec` outputs. Physical fragment coordinates and merge recipes remain execution derivation, not a separate federation-specific semantic result model.

### Result semantics versus execution derivation

The shared projection describes:

- Outputs and hidden operands, including catalog context/revision and actual alias mapping.
- General grouping grain, including non-entity dimensions and time buckets.
- Population inclusion rule and operand populations, separately from grain.
- Aggregation stages and valid recipes, including non-null count semantics.
- Effective predicates and their origin/stage, including isolation-scope predicates.
- Selection order, partitions, limit, tie policy, and determinism conditions.
- Reporting ranges, alignment, bucket generation, null/missing behavior, and numeric scale.
- Selected sources/joins, declared cardinalities, assumptions, and obligations.

Equivalent result semantics may have different physical derivations. Native `AVG` and sum/count recomposition can agree on the semantic metric while exposing different execution stages. Do not require equality of entire analysis objects to establish equivalence.

Raw catalog expressions may remain opaque behind trusted catalog references. Do not reverse-engineer raw SQL into invented lineage. Do not interpret the catalog's user-owned `metadata` dictionary to derive this contract.

### Uncertainty and runtime obligations

Keep separate:

1. Properties enforced by lowering.
2. Declared catalog assumptions.
3. Unproven source/data facts.
4. Unsupported capabilities.
5. Known-invalid transformations.
6. Execution evidence for obligations actually checked.

Dense bucket generation does not prove source completeness or freshness. Unique final groups do not prove intermediate key uniqueness. The first release implements only these execution obligations: artifact version/binding integrity, exact projected-column correspondence, and uniqueness of a declared one-side merge key in the materialized fragment relation immediately before a merge that relies on it. Check composite keys together and use the merge's actual null-key matching semantics. Do not validate uniqueness after an aggregation has already hidden duplicates.

Version, binding, and output-correspondence failures always reject. Implemented merge-key checks run whenever the consuming merge relies on those keys, and observed violations reject; a catalog declaration does not override contradictory evidence.

Trusted catalog declarations of keys/cardinalities are sufficient for execution when the executor cannot inspect the underlying database-side relation. Record these as declared assumptions, not verified facts. Do not add a default rejection or a host-waiver API merely because a declaration is uncheckable. The host may apply stricter admission policy outside the compiler. Missing cardinality needed to select a safe recipe remains a capability failure; accepting a declaration is different from inventing one.

There is no database-probing service in this release. Evidence applies only to the exact materialized relation consumed by the merge, or to a backend snapshot the adapter can explicitly attest was shared with the consuming query. A separate preflight query is not proof against later data changes. Cached evidence may accompany the same cached rows and matching obligation definition, but cannot establish fresh-source validity or satisfy a different unchecked obligation. Where no applicable evidence exists, retain the declared-assumption/unproven status without claiming a new check occurred.

Do not rerun scope functions to build analysis. Use the effective predicates resolved for emission. Do not expose private identity/cohort bindings or raw policy SQL in routine analysis/diagnostic payloads.

### Typed diagnostics and concrete capabilities

Preserve `SemQLError` and existing catch relationships. Reuse shared resolver/validation/planner records rather than introducing an independent classifier.

Provide stable codes, explicit severity, bounded reason categories, implicated semantic references, and typed operation/stage details. Attach details at checking/raising sites. Human messages are renderings, not policy inputs.

Capability depends on the concrete query, dialects, and selected strategy. Unsupported operator, unsafe aggregation, absent cardinality needed for safe lowering, unsupported fill, and unsupported federation shape are explicit failures. An unverified but trusted cardinality declaration is not by itself an unsupported capability. Alternative strategies may be described without silently selecting them or weakening membership.

Keep lossless internal exception serialization separate from safe public diagnostic rendering. Internal payloads and repair arguments remain potentially sensitive and continue to support authorized repair/round-trip consumers. Built-in API/MCP adapters and routine logging surfaces must use the safe public rendering, not the internal serializer. Render messages from allowlisted fields; exclude bound values, raw SQL, arbitrary exception messages, and value-bearing repair arguments unless disclosure is explicitly authorized for that surface.

### Executable parameter requirements

Record logical requirements at parameter allocation; `_CompileEnv.bind(value, dim_type)` is an existing attachment point. Retain names/types through wrappers, reuse, fragments, and merge rendering. The merge binder uses the same requirement vocabulary.

Reconcile requirements against final executable placeholders after supported transformations/hooks. Validate all fragments before the first adapter invocation and merge bindings before merge execution. Missing/renamed bindings fail with typed artifact/name details and no value disclosure.

Logical type, Python representation, and driver encoding are distinct. Temporal metadata alone does not repair an unexecutable temporal expression.

### Serialization, execution results, and versioning

Update explicit `CompiledQuery.model_dump()` / `model_validate()` support so analysis, artifact-local references, diagnostics, and requirements survive supported round-trips. Every successful compilation attaches analysis with explicit coverage; older serialized artifacts without analysis are marked unavailable, not retroactively certified. The existing artifact dump includes private parameter values and is not a safe routine diagnostic representation.

Distinguish package version, catalog context/revision, analysis schema version, and executable-plan version. Consumers requiring analysis reject unsupported schemas or explicitly treat analysis as unavailable. Executors reject unsupported plan versions before cache use or adapter invocation.

Keep the analysis associated with executed rows through sync, async, cache, and enrichment paths. Preserve current-artifact identity on cache hits. Validation evidence is associated with the execution and obligations checked, not invented from a cache hit. Immutable analysis can be shared rather than unnecessarily deep-copied.

## Independently added operators

### Partitioned ranking

Contract: rank expression, partition keys, ranked entity grain, direction, N, null ordering, and tie policy. Predicate-stage precedence is explicit. Aggregate the complete authorized candidate population before selection.

Differentiate exact N with deterministic tie-breaks, competition rank, and dense rank. Explicit global limits remain separate selection operations. Budgets reject unacceptable work; they do not impose hidden candidate pre-limits or silently collapse grain.

Membership fixture:

```text
A: March 1 = 100
B: March 1 = 60; March 2 = 60

Global top entity-day: A / March 1 / 100
Global top whole-window entity: B / 120
Top entity within each day: March 1 → A / 100; March 2 → B / 60
```

The first two were executed through current compilation. The partitioned answer is the independent reference for the proposed operation, not an implemented capability.

### Population-aware ratios and two-stage aggregation

Contract: explicit operand populations, intermediate grain, stage sequence, null policy, and legal recomposition recipe. Service-owned cohorts supply actual population predicates.

Reference fixture: A has events `10, 20, 30`; B has `90`; C is in the authorized universe with no events. Each event has `100` views.

| Semantics | Reference value |
|---|---:|
| Average of events | 37.5 |
| Average of observed entity totals | 75 |
| Average of all entity totals with explicit zero-fill | 50 |
| Sum of event ratios | 1.5 |
| Ratio of summed operands | 0.375 |

Reject unsafe joins and invalid/non-additive re-aggregation. Do not reject a valid sum/non-null-count decomposition merely because the final average is not additive. Verify denominator membership, not only numeric coincidence.

### Time completeness and comparison

Extend existing `TimeWindow` and `CompareWindow`. Define timezone, observed/dense buckets, equal-duration/calendar alignment, partial buckets, missing-row and present-null policies, division-by-zero behavior, negative-base percentage behavior, and output scaling.

Entity × day fill requires an explicit authorized entity universe and bounded expansion. Unsupported dialect/shape returns a capability failure, not a partial series. Preserve existing defaults until an explicit compatibility change is accepted.

## Implementation sequence and targets

### 1. Implement the agreed trust boundary and semantic definitions

Targets: `semql/refs.py`, `semql/_resolve.py`, `semql/logical.py`, `semql/model.py`, and the documented `compile_plan()` trust boundary in `semql/compile.py`.

Implement the local-reference/structured-comparison, observed-population, and trusted-declaration decisions above. Enforce `LogicalPlan` as a privileged host/optimizer input at external adapters and document `compile_plan()` accordingly. Keep legitimate physical rewrites and existing authorization/consistency checks; do not implement arbitrary untrusted-plan validation in this release.

### 2. Implement projection for existing supported compilation

Targets: effective output/transform handling in `semql/compile.py`, `semql/logical.py`, and `semql/rollup.py`.

Retain logical origins through physical rewrites. Cover aliases, generated comparison columns, hidden operands, masks, units, and period policies. Correct count-rollup lowering to the valid stored-count SUM recipe. Every requested output is accounted for; multi-backend inline-derived shapes are rejected as specified by the matrix.

### 3. Extend diagnostics, capabilities, and bindings

Targets: `semql/errors.py`, `semql/validate.py`, `semql/_resolve.py`, `semql/compile.py`, `semql/backend.py`, and merge binding in `semql_engine/merge/duckdb.py`.

Use one diagnostic vocabulary and shared checks. Add safe rendering and final-artifact binding manifests. Bring concrete capability results into agreement with lowering; do not treat static resolution success as executable support.

### 4. Integrate federation and execution boundaries

Targets: `semql/federate.py`, `semql_engine/merge/duckdb.py`, and `semql_engine/engine.py`.

Use the same final semantic projection, retain valid partial/merge recipes, preserve aliases, and enforce the observed-fact symmetric population decision. Reject multi-backend inline-derived shapes before producing a partial plan. Check versions before cache use, bindings before adapters, and one-side uniqueness before consuming materialized merge inputs. Trust catalog declarations where inputs are not inspectable, without labelling them verified. Propagate current analysis, local-reference integrity, and honestly scoped evidence through sync/async/cache paths.

### 5. Integrate enrichment and perform consumer cutover

Targets: `semql/lookups.py`, applicable package public exports/serialization, and consumer call sites identified during implementation.

Match enrichment inputs through semantic output mappings, not plain field names. Preserve key/grain identity and add provenance for attached fields without compiler I/O. Remove superseded SQL wrappers, positional metadata reconstruction, and warning-message parsing only after replacement behavior passes shared fixtures.

Retain authenticated identity, authorization enforcement, admission, and genuinely service-owned cohort/reporting policies. Do not replace provider-facing `SafeSemanticQuery` adapters solely to reduce duplication. Provider schema changes are a separate compatibility effort with their own provider verification.

Use the consumer's package manager to update its supported minimum. Preserve the documented lockstep package-release convention in [RELEASING.md](../RELEASING.md). This plan does not authorize dependency changes, commits, tags, or publishing.

### 6. Add each operator separately

Targets: existing spec/plan/compiler/backend paths plus shared federation/execution handling where that operation is supported.

Before adding an operator, confirm multiple concrete needs, a defined output contract, and real-dialect execution fixtures. Do not make all consumers wait for a general analytical-algebra rewrite.

## Verification plan

The following are future implementation checks, not checks completed by this spike.

### Existing regression surfaces

Use the relevant repository tests as regression guards:

```sh
uv run pytest packages/semql/tests/test_output_aliases.py packages/semql/tests/test_inline_derived.py packages/semql/tests/test_rollup.py packages/semql/tests/test_rollup_plan_transform.py packages/semql/tests/test_compare_loadbearing.py
uv run pytest packages/semql/tests/test_federate.py packages/semql/tests/test_federate_merge_spec.py packages/semql/tests/test_compile_warnings.py packages/semql/tests/test_fan_out_guard.py packages/semql/tests/test_validate_and_resolve.py packages/semql/tests/test_error_envelope.py
uv run pytest packages/semql-engine/tests/test_merge_param_binding.py packages/semql-engine/tests/test_cross_backend_symmetric.py packages/semql-engine/tests/test_async_engine.py packages/semql-engine/tests/test_cache_hazards.py packages/semql/tests/test_lookups.py
```

After implementation integration, run `just check`. Public-surface changes also require `uv run scripts/check_api_break.py --base main`. Document intentional compatibility changes rather than retaining obsolete aliases or message-parsing paths. Do not re-pin numeric or membership expectations to match a defect; remove incidental SQL/wording assertions when they cease to test consumer behavior.

### First-release exit gate

- Alias/reorder changes preserve logical output descriptors, and structured comparison establishes equivalence without comparing local ID strings. Within an artifact, transformations and serialization preserve references to unchanged nodes/outputs.
- Every requested output is emitted with analysis or rejected with a stable typed reason.
- Hidden operands resolve to nodes with metric, stage, unit, scope, null policy, and period information.
- Wrong metric source, aggregation stage, population, and period are distinguishable without positional reconstruction or alias parsing.
- Valid rollup and federation recipes match independent numeric and membership references.
- The included symmetric single/federated shapes agree on observed-fact population as well as values; entities absent from all facts are excluded, and qualifying all-null fact rows still establish membership.
- Implemented merge-key checks reject violations; uncheckable trusted catalog declarations allow execution and remain assumptions. Missing required declarations and known-invalid recipes still reject.
- Missing/renamed bindings fail before adapters; unsupported executable versions fail before cache use or adapters.
- Diagnostics distinguish unsupported shape, unsafe aggregation, invalid query, and binding failure without message regexes.
- Internal error round-trips retain authorized repair data, while built-in external surfaces use safe public rendering that excludes private values and arbitrary error text.
- Every successful compilation attaches analysis with explicit coverage. Local references and descriptors survive serialization, execution, cache hits, and label enrichment; unknown/absent coverage is not certified.
- Analysis generation is deterministic and local, with no database/model call or repeated scope-function resolution.
- Structured comparison follows references, distinguishes expression from whole-result scope, and returns equivalent/different/not-established with reasons. Fixtures independently rename local IDs, omit bound context, and vary catalog revision to exercise each outcome.
- Duplicate semantic projections reject; a future optional public semantic-ID annotation can be ignored by supported older decoders without changing local traversal or comparison. Unknown required semantics are never silently ignored.

### Operator exit gates

For every supported lowering, run actual PostgreSQL, ClickHouse, and DuckDB fixtures as applicable, using the supported-shape matrix above as the release inventory. Unsupported cells must reject; combinations require their own cells. Existing other-backend availability is not an execution-conformance claim. Do not silently switch strategy or weaken membership.

Fixtures require independent membership and numeric answers and must cover ties, null operands, zero denominators, negative comparison bases, zero-event entities, violated critical keys, partial/DST periods, and bounded Cartesian fill. Test authorized populations and excluded entities. Equivalent single/federated forms preserve denominator, grain, period, and ranking membership.

DuckDB stand-ins, source snapshots, and compilation-only tests do not satisfy actual-backend gates. Real-backend fixtures are prerequisites for the library's advertised conformance cells. External consumer integration gates that consumer's cutover, not library publication; provider verification is required only for a separately authorized provider-schema change. None of those integrations was verified by this spike.

## Implementation verification — 2026-10-02

The executable matrix is `packages/semql-engine/tests/test_semantic_backend_conformance.py`.
It executes emitted SQL and untouched bindings against DuckDB 1.5.4,
PostgreSQL 16.14, and ClickHouse 26.8.2.7; no dialect-translating stand-ins are
used for these cells. The run passed **34 cells** with **2 skips**: the DuckDB
source cases that specifically require a different source backend for federation.

Verified result cases include grouped sum/count/min/max/average, hidden inline
operands, catalog ratios, reordered aliases, distinct count and percentiles,
stored-count rollups, dense time-only fill, and comparison missing/negative/zero
prior policies. PostgreSQL and ClickHouse sources with a DuckDB merge also
passed distributive AVG recomposition and raw-row ratio/AVG fixtures against
independent values and equivalent native-result descriptors. Symmetric
membership includes one-sided and present-null facts and excludes unauthorized
or dimension-only keys.

The separate DuckDB smoke executed compile → serialize → compare → federated
engine → alias-aware enrichment, preserving A=30 and B=7 even when their labels
were identical. Removing a required binding raised `binding_values_mismatch`
before execution.
An authorization smoke executed distinct A/B-scoped queries against actual rows:
their unattested comparison was `not_established`, and supplying distinct private
policy contexts produced `different`, without disclosing either value.

With both external endpoint environment variables set, the full
`uv run --no-sync --with 'psycopg[binary]' pytest` run passed **3,077 tests**,
with **3 skips**, **1 expected failure**, and **83 snapshots passed**.
Ruff formatting/lint, mypy (264 files), pyright, Tach, the core package boundary
check, and the frozen-model check also passed. Psycopg was supplied only to the
verification environment; no dependency/version change was needed.

The public API diff intentionally reports the required `analysis` argument on
`enrich_all`. Its return value is now `EnrichedResult`; callers consume `.rows`
and `.analysis`. All repository callers have been migrated. Do not add an
alias/position-based fallback to conceal this cutover. Format version 3,
observed-fact membership, duplicate-output rejection, and safe public diagnostic
rendering are additional behavioral compatibility changes documented in the
changelog; a signature diff alone cannot enumerate them.

The release matrix above remains a bounded shape matrix, not a Cartesian-product
promise. No new ranking, general population/two-stage operator, entity × time
fill, provider schema replacement, or ZenAI package update is claimed.

### Testcontainers integration

The conformance fixture also supports `SEMQL_CONFORMANCE_TESTCONTAINERS=1`.
Explicit backend endpoints take precedence; otherwise it provisions one
module-scoped container per external backend, using pinned images and random
host ports. DuckDB remains in-process. Container startup errors fail the opt-in
run; ordinary local runs without endpoints or opt-in retain their explicit skips.
The existing result assertions are shared by managed endpoints and containers.

`testcontainers` and `psycopg[binary]` are workspace development dependencies,
added through `uv`. `just test-conformance` is the reproducible local entry point;
the existing full-suite CI job opts in as well.

`SEMQL_CONFORMANCE_TESTCONTAINERS=1 just check` passed with **3,077 tests**,
**3 skips**, **1 expected failure**, and **83 snapshots passed**, including
real PostgreSQL and ClickHouse containers. Formatting, lint, mypy, pyright,
Tach, and both boundary checks passed. The container image versions are
`postgres:16.14-alpine` and `clickhouse/clickhouse-server:26.8.2.7`.

A separate lifecycle smoke invoked each external backend fixture, executed
compiled grouped sums against its container (A=60, B=90, Z=NULL), and verified
via the Docker API that its session-owned database container was removed on
fixture close. No native-server substitute was used for this verification.
Adding the driver dependency also removed an obsolete missing-import suppression
from the introspection CLI; driver loading remains lazy. Regenerating the lockfile
aligned its existing workspace package versions with the 0.7.0 manifests;
no package manifest version was bumped.

### Version 0.8.0 release preparation

The separately authorized release advances all eight package versions to 0.8.0,
pins every sibling requirement to `>=0.8.0,<0.9`, regenerates `uv.lock`, and
dates the changelog entry. The minor bump communicates the intentional pre-v1
API and behavioral changes above. Historical version references and executable
schema versions are not package versions and remain unchanged.

The release sequence is a verified fast-forward to `main`, followed by the
`v0.8.0` tag. The existing tag-triggered workflow builds artifacts and publishes
through PyPI OIDC Trusted Publishing; no local credential fallback is needed.
Publication is verified against all eight PyPI projects and a fresh installation.

Release-candidate verification passed: the container-enabled `just check` ran
3,077 passing tests, 3 skips, 1 expected failure, and 83 snapshots. All eight
wheels and eight source distributions built and passed `twine check`. An isolated
installation of the wheels verified every version and sibling range, imported
all packages, and executed compilation, alias-equivalence comparison, artifact
serialization, and DuckDB results (A=30, B=7) outside the workspace.

## Appendix A — Spike evidence

The following records retain the observed behavior before implementation. They explain the decisions above but do not override the target release matrix or claim that fixes have landed.

### Observed defects and limitations

#### D1. An inline-derived output disappears in federation

Fixture: four events with amounts `10, 20, 30, 90`, each with `100` views, grouped under one region. The request projected the amount sum, event average, and an inline ratio of summed amount to summed views.

| Output | Single backend | Default distributive federation |
|---|---:|---:|
| Amount sum | 150 | 150 |
| Event average | 37.5 | 37.5 |
| Inline ratio | 0.375 | Absent |

Compilation and execution did not reject the missing ratio. The requested-output equivalence assertion failed.

The grouped inline form is otherwise supported: projecting only entity ID and the inline ratio returned `(1, 0.2)` and `(2, 0.9)`, while neither numerator nor denominator was projected. The resolver retained both hidden operands.

A scalar request containing only the inline-derived output was separately rejected as an empty query. That is an input-shape limitation; the grouped result does not establish scalar derived-only support.

Targets: `semql/compile.py`, `semql/logical.py`, `semql/_resolve.py`, and `semql/federate.py`.

#### D2. Supported federation ignores requested measure aliases

The catalog-declared ratio in explicit `raw_rows` federation executed correctly. Reordering the measures and assigning aliases produced:

```text
Single backend: region, rate, mean_amount, net
Federated:     region, share, mean_amount, amount
Values:        R,      0.375, 37.5,        150
```

The values and measure order matched, but requested aliases did not. The column-contract assertion failed.

This is separate from semantic identity: aliases map artifact-local output references to actual columns, while structured descriptors establish cross-artifact semantic correspondence.

Targets: `semql/federate.py` and the shared projection/alias handling in `semql/compile.py` and `semql/logical.py`.

#### D3. Rollup selection changes a count's value

Base rows: region A has amounts `10, 20, 30`; region B has amount `90`. The independently populated rollup contains `(A, amount=60, event_count=3)` and `(B, amount=90, event_count=1)`.

| Execution | A sum | A count | B sum | B count |
|---|---:|---:|---:|---:|
| Base table | 60 | 3 | 90 | 1 |
| Selected `by_region` rollup | 60 | 1 | 90 | 1 |

The rollup result failed the independent reference assertion. Source inspection shows `apply_rollup()` changes the measure SQL to the stored column while retaining its original aggregation. Counting stored count values is not equivalent to summing precomputed counts.

A logical event count needs a valid physical re-aggregation recipe. Retaining the catalog name cannot certify an invalid recipe.

Targets: `semql/rollup.py`, `semql/logical.py`, and `semql/compile.py`.

#### D4. Single and federated two-fact plans differ in population membership

Fixture: employee 1 has activity `300` and worklog `8`; employee 2 has activity `50` only; employee 3 has worklog `5` only; employee 4 exists in the entity bridge but has no rows in either fact.

```text
Single backend: entities 1, 2, 3
Federated:     entities 1, 2, 3, 4
Entity 4:      NULL activity, NULL worklog
```

The equivalence assertion failed. Both observed-fact membership and entity-universe membership are legitimate contracts, but they are not equivalent. Metric definitions, common-row values, and grouping keys alone do not capture this difference.

Targets: symmetric aggregation in `semql/logical.py` and `semql/compile.py`; symmetric federation and merge population construction in `semql/federate.py`.

#### D5. Missing fragment bindings reach the driver

Removing one named binding from a compiled fragment caused both adapters to be invoked before DuckDB raised `InvalidInputException`. The failure was not a `SemQLError`.

Validate every fragment's executable binding requirements before invoking the first adapter. Validate merge requirements before merge execution. Report artifact identity and missing parameter name without exposing values.

Targets: binding allocation in `semql/compile.py`; merge binding in `semql_engine/merge/duckdb.py`; sync/async execution boundaries in `semql_engine/engine.py`.

#### D6. The federation version marker is not enforced by the executor

Replacing a valid plan's version marker with `999` did not prevent execution. Its otherwise unchanged shape executed and returned the expected values.

This does not demonstrate incompatibility with a real future format; it demonstrates that the marker alone provides no version enforcement.

Targets: `semql/federate.py` and `semql_engine/engine.py`.

#### D7. Logical resolution and collecting validation overstate compile readiness

An entity × day zero-fill request produced a logical plan and `validate()` returned `[]`, but concrete compilation rejected the unsupported non-time-dimension fill shape.

An empty static-diagnostic list cannot establish query/dialect/strategy support unless the applicable lowering-capability checks ran. An analysis-only API cannot simply expose `to_logical_plan()` as an executable-support decision.

Targets: `semql/validate.py`, `semql/_resolve.py`, `semql/logical.py`, and the corresponding checks in `semql/compile.py`.

#### D8. Time-only dense fill compiled but failed in DuckDB

A day-bucket query over ISO-string time bounds with `fill_nulls_with=0` compiled, then failed at execution because DuckDB could not resolve `date_trunc(STRING_LITERAL, STRING_LITERAL)` to a temporal overload.

The observed-bucket version executed: a present null-valued event returned a null sum with count 1; a genuine zero returned zero with count 1; the missing day was absent.

The dense-fill portion did not pass. Parameter casts or SQL rewrites were not introduced to make the fixture pass. Source inspection separately shows the existing fill wrapper coalesces aggregate values, not only missing-bucket values.

Targets: `semql/backend.py`, temporal binding and fill emission in `semql/compile.py`, and adapter encoding in `semql_engine/adapter.py`.

#### D9. Alias-based enrichment skips an aliased key

Two region keys, A and B, both mapped to `Shared label`. Without an alias, enrichment attached both labels and retained the separate rows and values `30` and `7`. Aliasing `region` to `area` caused `enrich_all()` to attach neither label.

This is documented current best-effort behavior, not an unexpected violation of the existing implementation contract. It is a limitation relative to the proposed alias-invariant contract. Enrichment currently matches the dimension's field name rather than a semantic output reference.

Target: `semql/lookups.py` and its callers.

#### D10. Unsafe fanout failures lack machine-readable operation details

A distinct-account count across a one-to-many orders join executed correctly: category x had 2 accounts and category y had 1. The plan carried typed `JoinDiagnostic` facts, but `CompiledQuery.warnings` contained strings.

An additive account-balance sum across the same fanout was correctly rejected. Its payload exposed only `code="CompileError"` and a message, without a reason category or implicated metric/join references.

Expose existing planner facts directly. Attach failure details at rejecting checks; do not classify messages after the fact. A safe operation over a fanout may produce an advisory, while an unsafe aggregation must remain a failure.

Targets: `semql/logical.py`, `semql/compile.py`, and `semql/errors.py`.

### Other findings that constrain the contract

These observations are not all implementation defects.

#### Declared cardinality is not runtime proof

Duplicating a declared-unique entity bridge key changed both single and federated results from sum `150`, average `37.5`, ratio `0.375` to sum `210`, average `30`, ratio `0.3`. Both still returned one final region row.

The data violated the declared assumption. Agreement between strategies and unique final grain did not detect the wrong values. Critical bridge-key checks must occur before the multiplicative merge where observable. An output-only check cannot discharge an upstream cardinality obligation.

#### Strategy-specific support already exists

A catalog-declared ratio was rejected in distributive mode with:

```text
code: FederationError
reason: non_distributive_aggregation:ratio
```

The reason survived serialization. The same ratio in explicit `raw_rows` mode returned `0.375`, matching the single-backend fixture. Preserve this taxonomy while separating the bounded category, aggregation, and strategy into structured fields.

#### Average recomposition preserves non-null count semantics

With one null amount among four rows, the single backend, distributive federation, and raw-row federation all returned sum `120` and average `40`. The independent denominator was 3 non-null operand values, not 4 rows; the latter would produce `30`.

Distributive lineage must identify `COUNT(operand)`, not just an unspecified count. Summing partial averages is unsafe; recomposition from matching sums and non-null counts is a valid existing recipe.

#### Existing time and comparison policies are specific

For March 2024, `previous_period` resolved to Jan 30–Mar 1: the immediately preceding equal-duration window. An explicit February calendar window produced a different prior value and delta: `55/45` versus `50/50`, with current value `100` in both cases. April 1 was correctly excluded by the half-open current interval.

Comparison edge cases showed different policies per output:

- Current/prior outputs preserve nulls.
- Delta coalesces each operand to zero before subtracting.
- Percentage change preserves null propagation and is calculated only for a positive prior value.
- A current value of `-5` and prior value of `-10` returned delta `5` and percentage change null; the ordinary nonzero-denominator formula would return `-50%`.
- Percentage change uses percent scaling: doubling returned `100`, not the fraction `1`.

An initial edge-case assertion assumed broader zero-filling and failed. That was an incorrect reference assumption, not evidence of a regression. Source inspection and the separate negative-base execution established the actual policy. New policies must not silently change these defaults.

#### Output identity is not bound-query equivalence

Two differently bound cohorts produced the same SQL, columns, and presentation metadata but different correct values: `(60, 20, 0.2)` and `(90, 90, 0.9)`.

An authorization fixture likewise had no request or logical-plan filters, yet catalog security predicates and private viewer bindings restricted totals to `60` and `90`. Analysis must include effective catalog/security/scope restrictions, not just explicit request filters.

Bindings remain private. Analysis can identify dependencies; the trusted caller compares the bound context. Do not publish ordinary hashes of sensitive, low-entropy values as a substitute for private context.

#### Error typing does not guarantee safe rendering

A synthetic `FilterTypeError` retained its input value in the human message, structured value, repair-tool arguments, and round-trip reconstruction. No external disclosure route was established by this probe.

Lossless internal payloads are potentially sensitive. Public diagnostics require allowlisted fields and messages generated from those fields, not arbitrary `str(exc)` copied into a response or log.

#### Declared units and display units are distinct

A measure storing `3600` seconds with an hours display preference, added to a measure storing `2` hours, produced raw numeric value `3602`. The derived output's unit remained unset. The compiler does not perform display conversion.

Preserve operand storage units and an explicit unknown/unproven derived unit where necessary. Do not infer units from aliases, formatting, or opaque catalog metadata. General dimensional analysis and automatic conversion are outside the first release.

#### Value-cache reuse need not imply shared metric identity

Two differently referenced catalog metrics with identical SQL and presentation metadata returned `30`, with one cache miss followed by one hit. This was valid numeric reuse. The execution result currently carries columns, metadata, and rows, not analysis.

New analysis must describe the current artifact on both hits and misses. Cached validation evidence belongs to the checks and execution actually performed; a cache hit is not a fresh data-completeness check.
