# Planner, schema-profile, continuation, and ranking migration

This change tightens semantic query inputs and changes the query generator's
outer response contract. Update integrations before relying on the new
planner, continuation, or ranking behavior. It does not imply a package-version
change.

## Strict canonical query inputs and plans

`SemanticQuery` and its nested query-input values now reject unknown fields
(`extra="forbid"`) and are immutable. `QueryPlan` is likewise immutable and
extra-forbid. It may still represent an empty host-side intermediate plan, but
an empty plan is no longer a planner failure signal: `PlanOutcome` requires at
least one step. Catalog models remain separate and are not made globally
extra-forbid by this change.

Validate model-produced plans against the new response protocol; update code
that previously parsed a bare `QueryPlan` or interpreted `steps=[]` as “cannot
answer.” A parsed outcome is an untrusted proposal, not authorization or proof
that compilation will succeed.

## Four planner outcomes and repair

The default generator output is `semql_prompt.PlannerResponse`, an object whose
discriminated `outcome.kind` is one of:

- `plan`: a nonempty `QueryPlan` in `outcome.plan`;
- `need_context`: bounded typed `requests`, optionally paired with correlated
  `results`;
- `need_user_clarification`: a `decision_id`, prompt, and bounded choices;
- `cannot_plan`: bounded typed diagnostics.

Optional top-level `request_id` and `revision` correlate a response to host
continuation state. Use the structured-output facility of your chosen client
with the supplied response schema, then validate the parsed payload through the
generation contract. SemQL builds no provider-specific agent, retry loop, or
model invocation. `build_query_repair_prompt` now requests the same
`PlannerResponse`; it builds a repair prompt and does not invoke a model, run a
query, or retry. `build_prompt_repair_prompt` remains a separate prompt
augmentation proposal flow. Historical prompt/failed-output disclosure and
acceptance remain the host's responsibility.

## Schema profiles: decoder representation, not compiler support

Use `build_generation_contract(catalog, profile=...)` to keep output model,
projected JSON schema, loss report, and prompt guidance together. Its default
output model is `PlannerResponse`. For example, consume
`contract.schema`, send `contract.text` to the model client, then pass the
already-parsed payload to `contract.validate_payload(payload)`. The paired
validator applies canonical Pydantic validation and profile feature
restrictions; validating only against JSON Schema is insufficient.

`SchemaProfile.TOOL` retains recursive Boolean filters and semi-joins.
`SchemaProfile.STRICT` removes `SemanticQuery.where` and `semi_joins`, and
represents fixed tuples as arrays for restrictive decoders. Inspect
`contract.report.removed_features` and `weakened_constraints`: tuple-position
validation has been weakened in the projected schema. Required nested query
and outcome payloads remain required. `SemanticFeature.WHERE`,
`SEMI_JOINS`, and `RANK` may also be explicitly disabled for a request;
projection and canonical validation both enforce those restrictions.

A profile describes what a decoder can represent. It neither grants access nor
certifies that a query shape is executable. Guidance explains ranking and
unsupported shapes, but authorization, catalog validation, shared capability
checks, and dialect lowering are still needed. Keep viewer-specific data
request-scoped; do not treat a projected schema as a security boundary.

## Prompt and mandatory-context budgets

Adaptive prompts and repair prompts return an explicit `fits` result. Optional
catalog material may be trimmed, but required instructions, protected catalog
content, schemas, and continuation locks may leave the prompt over budget.
Callers MUST check `fits` before submitting the request; a budget is not a
promise that the mandatory contract can fit. A host-supplied token counter may
be used; otherwise the built-in estimate is heuristic. The adaptive and
repair demos configure 40,000 tokens as an example, not as a universal
minimum or guaranteed fit.

## Lookup certainty and completeness

For lookup-backed query values use `semql.lookups.resolve_outcome` (or inspect
the typed `ContextResult` in a planner response), rather than treating a
nonempty list from `resolve` as evidence of a unique match. Outcomes distinguish:

- `resolved`: exactly one complete, authoritative non-fuzzy candidate;
- `ambiguous`: multiple candidates, truncation, or a fuzzy-only suggestion;
- `not_found`: an authoritative, complete empty result;
- `unavailable`: no usable dynamic source/context or a lookup loader failure.

The result reports candidate completeness and truncation. Fuzzy matches are
suggestions, never authoritative resolutions; candidate limits do not turn
multiple matches into a unique match. For `need_context`, each bounded request
uses a typed semantic target and correlated identifiers. A result must
correlate to that exact request. `not_found`, `ambiguous`, and `unavailable`
are different states: do not silently substitute an arbitrary candidate or
claim that an unavailable source found no value. Existing `resolve` remains a
candidate-list convenience API, not a certainty API.

## Host-confirmed continuation

Continuation is host-owned state, not a generic patch protocol. Build a frozen
`ContinuationState` from the accepted baseline and current authorized semantic
references. Decisions have typed targets (`FilterValueTarget`,
`MeasureChoiceTarget`, or `TimeRangeTarget`) and bounded typed choices. The
host, not the model, records the confirmed choice. Context targets and choices
must be authorized before disclosure.

`render_continuation` verifies current authorized cubes and visibility revision
before rendering the required fenced state. A response must correlate to the
same `request_id` and `revision`; `accept_continuation` checks those values,
current authorized scope and visibility revision, exact permitted mutations,
and unchanged baseline constraints. It then validates query scope and compiles
with the caller's current catalog, viewer, and context. Reject stale responses,
changed locked constraints, changed visibility, unconfirmed choices, or failed
compilation; never compile the model's changed plan outside this acceptance
path. The host retains identity, retrieval, user confirmation, model calls,
budget handling, and termination ownership. A rendered continuation lock is
mandatory prompt context and cannot be safely trimmed to manufacture a fit.


## Partitioned ranking

`PartitionRank` and `RankOrder` are public query types, attached as
`SemanticQuery.rank`. For example:

```python
from semql import PartitionRank, RankOrder, SemanticQuery

query = SemanticQuery(
    dimensions=["sales.region", "sales.product"],
    measures=["sales.revenue"],
    rank=PartitionRank(
        partition_by=["sales.region"],
        order_by=[RankOrder(field="sales.revenue", direction="desc")],
        count=3,
        ties="exact",
    ),
)
compiled = catalog.compile(query, viewer=viewer, context=context)
```

`partition_by` keys must be selected grouping keys, and score/tie references
must be selected grouped outputs; derived outputs may be selected and ranked.
Ranking is applied after aggregation, derived projections, and `HAVING`, and
before final `ORDER BY`, `LIMIT`, and `OFFSET`. The query's final order can
therefore order the retained rows, while a global cap still applies across
partitions. `RankOrder.nulls` accepts `first` or `last` and defaults to `last`.

`ties` is not a distinct-count operation: `ties="exact"` selects at most N
rows per partition (`ROW_NUMBER`) and appends remaining grouping keys in a
deterministic order to break ties. `ties="include"` uses score keys only and
includes all rows tied at the Nth boundary (`RANK <= N`, not `DENSE_RANK`).
The internal rank ordinal is not exposed as a result column. Ranking keys and
exact-mode tie breakers cannot use masked fields; authorization and catalog
resolution still apply.

Ranking is supported for grouped single-backend queries only. It is not
supported with comparisons, dense-time fill, or genuine multi-backend
federation; report unsupported shapes rather than falling back to raw SQL.
There is no semantic-SQL parser round-trip for ranking: the parser rejects
window functions including partition ranking, and semantic unparse rejects a
query containing `rank`. Use the typed `SemanticQuery` API. See
[`../../demos/partitioned_ranking_demo.py`](../../demos/partitioned_ranking_demo.py)
and [`../../demos/planner_continuation_demo.py`](../../demos/planner_continuation_demo.py)
for runnable examples.
