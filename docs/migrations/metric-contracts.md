# Opt-in metric contracts and semantic checking

Compilation establishes what SQL SemQL can lower; it does not establish that a query answers the user's question. Metric contracts add explicit author-declared constraints, and semantic expectations describe the independently supplied or model-inferred interpretation of a request. Neither changes the meaning of ordinary compilation.

## Declare constraints, not another metric

```python
from semql import (
    Catalog, Cube, Dialect, Dimension, Measure, MetricContract,
    SemanticExpectations, SemanticQuery, SourceGrain, TimeDimension, TimeWindow,
)

orders = Cube(
    name="orders", table="orders", alias="o", dialect=Dialect.DUCKDB,
    measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
    dimensions=[Dimension(name="id", sql="{o}.id", type="number")],
    time_dimensions=[TimeDimension(name="shipped", sql="{o}.shipped", type="date")],
)
catalog = Catalog([orders], metric_contracts={
    "orders.revenue": MetricContract(
        reference="orders.revenue", revision="revenue-v1",
        source_grain=SourceGrain(keys=("orders.id",)),
        required_time_axis=True,
        permitted_time_axes=("orders.shipped",),
        permitted_granularities=("month",),
        preferred_granularity="month",
    ),
})
compiled = catalog.compile(SemanticQuery(
    measures=["orders.revenue"],
    time_dimension=TimeWindow(
        dimension="orders.shipped", granularity="month",
        range=("2026-01-01", "2026-03-01"),
    ),
))
expectations = SemanticExpectations(
    metrics=("orders.revenue",), time_axis="orders.shipped", granularity="month",
    origin="independently_confirmed", evidence_ref="request:shipping-report",
)
report = catalog.check_semantics(
    compiled, expectations, require_confirmed_intent=True,
)
if not report.accepted:
    raise ValueError("Mandatory semantic or confirmed-intent obligations were not established")
```

The registry round-trips through `catalog.spec.model_dump`/`CatalogSpec.from_dict` and `Catalog.from_spec`. `Catalog` accepts a mapping; `CatalogSpec` stores a tuple of contracts keyed by their canonical `reference`, preserving immutable, JSON-serializable and deep-copyable data. Registry keys and field references are validated against their catalog field kinds; business-calendar references are opaque names, not catalog time fields. Contract revisions are author-owned semantic versions, not measurements of physical data quality. Existing `Measure` ratio operands, filters, units, dimensions, segments, time dimensions and joins remain authoritative; user `metadata` remains opaque.

`None` means an unspecified constraint; an empty tuple is an explicitly empty allowed set where supported. Required/forbidden populations refer to existing named segments. A forbidden segment forbids applying that named segment, not every logically equivalent SQL predicate. Defaults and preferred axes/granularities generate nonbinding advice: the checker never rewrites the query.

Composite source grain and functional dependencies are declared assumptions. `SourceGrain.determines(target)` performs finite closure under those declarations; `proved` is conditional on their truth. Failure to derive a dependency is `not_established`, not evidence that real rows violate it. Runtime uniqueness, freshness and completeness require independently scoped runtime evidence, not a declaration or an annotation supplied by a model.

## Keep four judgments separate

- **Structural validity:** integrity of the analysis's local references and supported shape.
- **Declared semantic validity:** conjunction of mandatory modeled contract obligations. A mandatory `not_established` obligation blocks acceptance.
- **Intent alignment:** agreement with the supplied, potentially partial expectations. Model-inferred expectations never establish independently confirmed intent; unspecified clauses remain outside the claim.
- **Execution support:** the low-level analysis checker does not prove lowering. `Catalog.check_semantics` accepts a trusted `CompiledQuery` from the matching catalog and reports its successful lowering separately. Neither API proves that a live backend is reachable, data is complete, or execution succeeds.

`accepted` checks structural/mandatory semantic obligations, plus independently confirmed alignment when `require_confirmed_intent=True`. It does not imply the execution-support judgment, full understanding of the user's request, or correctness of unspecified domain semantics. No applicable metric contracts do not prove universal semantic validity.

The low-level API is pure and does not compile or execute:

```python
from semql import check_semantics

report = check_semantics(compiled.analysis, catalog.metric_contracts, expectations)
for obligation in report.obligations:
    diagnostic = obligation.diagnostic()
    if diagnostic is not None:
        print(diagnostic.code, diagnostic.reason, diagnostic.references)
```

Contract checks include hidden metric operands, not just projected outputs. A ratio cannot evade a component's time/population requirements. Metric identity follows comparison projections without treating every hidden numerator as a requested output metric. Masked values and incomplete/unknown semantic evidence cannot supply numeric proofs.

Mask checks follow the contributing operand DAG even when an operand has no separate contract. Named segment obligations apply to every metric occurrence's own population; result-level intent checks use result-population evidence separately. Missing descriptors do not prove the absence of a forbidden population. Time-bucket and snapshot proofs require consistent policy/time-node evidence: missing policy does not turn a retained monthly bucket into an unbucketed result. Shared graph validation covers ordering, ranking and population/predicate edges as well as operands.

## Trust and evidence

The host—not an LLM—must control contract registration, expectation origin and independent evidence references. `independently_confirmed` is a host attestation, not cryptographic authentication. Do not expose that choice as an untrusted model-controlled confirmation mechanism. If one request mixes confirmed and inferred clauses, check separate expectations objects rather than upgrading all clauses to confirmed.

Only check authentic analysis with the registry for its catalog snapshot. The low-level checker does not authenticate artifact ownership, verify arbitrary SQL or compare hidden parameter values. Artifact-local IDs are not persistent semantic identities. Use existing `compare_analysis` with its catalog/bound-context evidence for cross-artifact comparisons.

Business-calendar execution and full fixed-cohort comparison alignment are deliberately not inferred from labels. When requested but not established by available analysis, they return mandatory unknown obligations. Existing `CohortQuery` remains the fixed-membership construction; checking only its selection-stage analysis does not certify the semantics of the full cohort result.

Check annotations can be attached explicitly and immutably through `semql.analysis.attach_semantic_check`. Each annotation retains the full `report`—including obligations, assumptions, advice, alignment and host intent evidence reference—and existing metric `node_ids`, not a second provenance graph. Ordinary compilation does not run checks or attach reports. Treat serialized annotations as retained evidence, not authorization or a waiver of fresh checking after changing query/catalog/intent.

## Sufficient states, without a new algebra

Existing federation mean lowering now carries a small private sufficient-state descriptor through fragment and merge construction. It retains the original input/filter and computes SUM plus non-NULL COUNT before finalizing the quotient. Unequal group sizes, filtered-out rows and all-NULL groups therefore retain SQL AVG semantics. This makes an existing lowering recipe explicit; it is not new general aggregation support or a public expression DAG.

Finalized averages, ratios and distinct counts are not generally additive. Snapshot balances can be additive across entities at a fixed instant, but summing snapshots across instants is not an as-of selection. A valid alternative construction remains valid: the library does not require one planner workflow or silently replace a query with a preferred construction.

## Run the execution examples

```sh
uv run --frozen --python 3.12 python demos/metric_contracts_demo.py
```

The demo executes six independent DuckDB counterexamples through public SemQL compilation, including the existing cohort runtime: ratio-of-sums vs mean-of-ratios, overlapping distinct counts, snapshots across time, declared/lying join fanout, created vs shipped time axes, and fixed vs independently selected populations. These are execution/oracle checks, not heterogeneous-backend conformance or a measured LLM-accuracy benchmark.
