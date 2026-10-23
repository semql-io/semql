# Single-hop cohort spine

This is an additive API for fixed-membership analysis. Existing `SemanticQuery`,
`QueryPlan`, `PlannerResponse`, and semi-join contracts are unchanged. It is not
a general workflow runner or a persistent cohort service.

## Model and execute

`CohortQuery` wraps a `Cohort`, one or more named `CohortMeasurement` values,
and optional `CohortComparison` values. `Cohort.source` is an ordinary
`SemanticQuery`; it can select ranked winners using measures and derived rates.
Declare all source dimensions as either `member_keys` or retained `attributes`.
Only those declared keys/attributes enter the final selection projection, not
the source's selection metrics.

Each measurement uses an ordinary grouped query and maps every source member
key to a selected measurement dimension through `key_mapping`. Mapping order
is irrelevant; the declared `member_keys` order defines tuple identity. Extra
measurement dimensions must be declared as measurement `attributes` and must
be single-valued per member. Output aliases are not supported in cohort child
queries; use the compiler's unambiguous output column names when identifying
comparison metrics.
Plain time-window restrictions remain supported. Grouped/dense-fill time
dimensions are not accepted because their implicit time grain is not a declared
member attribute. Measurement `CompareWindow` mode is also rejected; declare
separate named measurement windows and `CohortComparison` values instead.

Use `compile_cohort_query(query, catalog.as_dict(), viewer=..., policy=...,
scope_fns=..., views=..., context=...)` to produce a pure staged plan. Pass the
catalog's corresponding registries/settings explicitly as with other low-level
compiler entry points. Compile with the current host identity; parsing a model
proposal is not authorization. Execute with `semql_engine.run_cohort(plan, engine)`.
The engine's registered adapters route each stage to its backend.

See [`demos/cohort_spine_demo.py`](../../demos/cohort_spine_demo.py) for an
executable ranked selection and fixed-member prior/current comparison.

## Grain and attribution

A cohort has **one row per member key tuple**. NULL keys, duplicate tuples,
masked identity, and masked retained attribution reject. The same store winning
in two source regions does not become two valid store members: declare a
meaningful composite identity or change the question. Never silently deduplicate
conflicting attribution.

`selection.<column>` identifies selection-time keys and attributes.
`<measurement-name>.<column>` identifies measurement-time attributes and metrics.
`comparison.<name>` identifies explicitly requested comparisons. Measurement key
columns do not repeat the selection identity in the final output.
`result.column_meta` preserves the child compiler's units, display hints,
format, and mask status under these prefixed names.

Changing measurement-time region does not change selection-time region. A
measurement returning multiple rows for one member rejects, even when those
rows belong to different current regions. This API does not implicitly regroup
or roll up a member into a new output grain.

## Missing evidence and comparisons

Every selected member remains in the result. An unmatched measurement supplies
NULL attributes/metrics. `result.presence[name]` aligns with `result.rows` and
indicates whether that measurement returned a member row. `result.coverage`
reports selected versus matched members per measurement; this is row-match
coverage, not proof that every metric is defined or every observation exists.

Select an observation count and relevant operand measures when distinguishing
missing observations, observed NULL inputs, and zero denominators. Existing
backend aggregate semantics remain applicable; for example, the treatment of
all-NULL SUM inputs is not made identical across engines by cohort alignment.

`CohortMetric(measurement=..., column=...)` addresses a compiled child metric
output. Comparison operations are `difference` (`left-right`), `ratio`
(`left/right`), and `percent_change` (`(left-right)/right*100`). NULL operands
propagate; zero denominators produce NULL. No implicit zero fill is performed.
Comparisons require matching unit signatures; distinct untyped measures are
not assumed interchangeable. Ratios and percentage changes are dimensionless.
Incompatible numeric precision classes reject. Decimal/float mixes are not
silently coerced; decimal arithmetic preserves subtraction precision and uses
at least 38 significant digits for division.

Do not sum finalized member rates, averages, or distinct counts. Compute
cohort-wide metrics over appropriately restricted facts with the existing
compiler; this API deliberately offers no generic result reaggregation.

## Bounds, failures, and authorization

`Cohort.max_members` defaults to 1,000 and cannot exceed 1,000. The source fetch
uses a bound-plus-one sentinel, preserving any smaller explicit source limit.
Overflow rejects rather than returning a truncated cohort. Measurements are
bound by exact member tuples, not independent per-key IN lists, and use their
own overflow sentinel. Measurement rank/limit/offset and nested staged semi-joins
reject rather than silently dropping members.
Composite predicates have bounded polynomial size rather than expanding a
disjunction of tuples exponentially during Boolean normalization. Numeric
member keys retain exact Decimal values through a JSON-safe `DecimalLiteral`
filter representation; UUID/date/timestamp driver values are bound using their
canonical representations. Source and target logical key types must match;
cross-type identity coercion is not inferred from `coerce_to`.
`DecimalLiteral` round-trips through the query's JSON representation and binds
as a native Decimal. Semantic-SQL rendering rejects it explicitly because the
parser's numeric-literal path cannot preserve arbitrary decimal precision.

All static child query shapes are compiled before source execution. The source
relation is checked before dependent measurement I/O; empty cohorts skip all
measurement I/O. Dependent compilation/execution/integrity failures abort the
operation without returning a partial success. Invalid known plan versions,
output shapes, or parameter contracts reject. Runtime integrity failures use
`ExecutionContractError`; compilation uses existing typed compiler errors.

Every child query is authorized independently, and bound measurements recompile
with the request's viewer, policy, scopes, views, and context. A cohort is not an
access grant. An unmatched authorized query does not prove that an inaccessible
record exists; do not disclose authorization distinctions from missing rows.

`result.provenance` retains the cohort request, stage query, child semantic
analysis, catalog context, row counts, and identity-validation evidence. It does
not assert that a child's semantic analysis describes the composed cohort result
or certifies equivalence of two complete cohort operations.

## Planning and execution boundaries

Generation is opt-in through `build_generation_contract(...,
output_model=CohortQuery)`. TOOL/STRICT profiles apply to every nested semantic
query and canonical validation is still required. The default `PlannerResponse`
remains unchanged. A restrictive profile need not omit the entire cohort:
the single-hop wrapper itself introduces no recursive cohort reference.

Execution is staged even on one backend. There is no cross-stage transaction,
snapshot, single-evaluation guarantee, automatic retry, async scheduler,
materialized-cohort persistence, or arbitrary dependency DAG. Each child query
remains subject to existing compiler/backend/federation capabilities. Compile
and execute plans within their host-owned request and authorization lifecycle.
