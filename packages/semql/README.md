# semql

Pure-Python compiler from a semantic spec to backend SQL. Define
cubes (dimensions, measures, time-dimensions, joins) once; emit
correct, parameterised SQL for Postgres, ClickHouse, DuckDB,
Snowflake, BigQuery, and the analytics engines Redshift, Trino and
Databricks.

SQL Server, MySQL and Oracle ship as **experimental / opt-in**
dialects: sqlglot transpiles their `date_trunc` / percentile to
best-effort forms that aren't exercised in CI, so enable them
deliberately by passing `experimental_dialects()` through the
compiler's `dialects=` override and verify the SQL on a live
instance. (Gap-filling time-spines — `fill_nulls` — are not yet
implemented for any of the six new dialects.)

`semql` does **no I/O**: catalogs are Python data; the compiler
returns SQL + bound params; running the SQL is the caller's job.
Sibling packages add LLM-planner prompt fragments (`semql-prompt`),
MCP exposure (`semql-mcp`) and ER diagrams (`semql-erd`).

## Install

```sh
pip install semql
```

## Quick start

```python
from semql import (
    Dialect,
    Catalog,
    Cube,
    Dimension,
    Measure,
    SemanticQuery,
)

orders = Cube(
    name="orders",
    dialect=Dialect.POSTGRES,
    table="orders",
    alias="o",
    measures=[
        Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
    ],
    dimensions=[
        Dimension(name="region", sql="{o}.region", type="string"),
    ],
)

catalog = Catalog([orders])
compiled = catalog.compile(
    SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
)
# compiled.sql, compiled.params, compiled.columns, compiled.dialect
```

The `{o}` placeholder in a cube's `sql` is its alias; the compiler
resolves it (along with `{schema}`-style context placeholders and
`{ctx.X}` row-level-security placeholders) at compile time.

## Semantic and executable contracts

Every successful compilation attaches `compiled.analysis`, including explicit
coverage. Its frozen descriptors distinguish logical metrics and hidden operands,
result grain/population, predicates, selection, time/null policies, declared
catalog assumptions, and physical execution recipes. Complete **semantic
coverage** does not certify source freshness, completeness, or key uniqueness.

`node_id`, `output_id`, population IDs, and binding slots are artifact-local
references. Serialization preserves them; independent compilations need not use
the same strings. Compare definitions with `compare_analysis`, not IDs, SQL,
aliases, or output positions:

```python
from semql import CatalogContext, compare_analysis, compile_query

revision = CatalogContext(namespace="reporting", semantic_revision="r17")
first = compile_query(query, catalog.as_dict(), catalog_context=revision)
second = compile_query(aliased_query, catalog.as_dict(), catalog_context=revision)
comparison = compare_analysis(first.analysis, second.analysis, scope="expression")
# comparison.outcome: "equivalent", "different", or "not_established"
```

The host owns the immutable catalog revision, including changes to catalog SQL,
policies, and scope-function behavior. Without a revision, local analysis still
works, but persisted/cross-process equivalence is not established. An explicit
`shared_catalog_snapshot=True` attests a shared immutable snapshot in-process.

Expression comparison is not a promise of equal cohorts or result rows.
`scope="result"` also compares grain, population, joins, selection, and time
policy. Supply `left_context` and `right_context` as private mappings keyed by
each artifact's **semantic binding slots**, or provide `bound_context_equal`
after independently establishing that equality. Missing context yields
`not_established`; values never appear in comparison diagnostics. Neither scope
proves identical live data. Invalid/dangling/cyclic graphs raise `ContractError`;
unknown required semantics or versions are never certified.

Policy-context slots cover catalog predicates, tenancy, security context, and
evaluated scopes. Supply each slot's complete private request context, or attest
that context equality explicitly; the same catalog revision alone does not prove
equal authorized populations. Value-free predicates require no private binding.
Logical assumptions constrain result comparison; physical merge-key obligations
remain separate execution checks.

`CompiledQuery.model_dump()` / `model_validate()` preserve analysis, executable
version, diagnostics, and binding requirements. Older payloads without analysis
remain unavailable; recompile unsupported executable versions before execution.
Artifact dumps include private `params` and are **not** safe diagnostic payloads.
Use `SemQLError.to_public_payload()` for external errors; `to_payload()` remains
lossless and potentially sensitive for trusted repair consumers.

Duplicate semantic projections reject. To display a metric twice, reuse its
output in presentation. `compile_plan()` is a privileged host/optimizer entry
point, not a client-facing alternative to `SemanticQuery`.

### Query cost budgets

`estimate_cost(query, catalog.as_dict(), views=catalog.views)` sums the declared
`size_hint` values for resolvable referenced cubes. It is a known
subtotal/lower-bound guardrail, not a whole-plan scan estimate: it does not model
selectivity, join multiplication, or actual runtime scans. `CostEstimate.cubes_unknown` preserves
the names of referenced cubes without a size hint; `rows_scanned_unknown`
remains a boolean summary.

When `max_rows_scanned` is configured, the known subtotal always must fit and
unknown-size references reject by default. If your admission policy explicitly
allows unknown portions, opt in with
`QueryBudget(max_rows_scanned=..., unknown_cost_policy="allow")`. Cube ceilings
count each known and unknown referenced cube individually.

View mapping is optional when references are cube-qualified. Known lower bounds
and unknown identities remain separate so an explicit unknown-cost opt-in
cannot mask a known subtotal over the ceiling.

### Alias-aware enrichment

`enrich_all` now requires the result's analysis and returns `EnrichedResult`:

```python
from semql import ResolutionContext, enrich_all

enriched = enrich_all(rows, catalog, ResolutionContext(), analysis=compiled.analysis)
rows = enriched.rows
analysis = enriched.analysis
```

Lookup inputs are matched by qualified semantic identity, so aliases work and
equal labels never collapse distinct keys. Original rows are not mutated;
original semantic references and grain remain intact. Attached fields carry
lookup provenance in `analysis.enrichments`. Masked keys are not enriched, and
attachments cannot overwrite an existing output.

## What lives in the box

| Surface | Module |
|---|---|
| Cube / Measure / Dimension / TimeDimension / Join | `semql.model` |
| SemanticQuery / Filter / TimeWindow / CompareWindow | `semql.spec` |
| Catalog wrapper (validation, compile entry) | `semql.catalog` |
| Compiler — sqlglot AST → dialect SQL | `semql.compile` |
| Collect-all static validator | `semql.validate` |
| Reflection cubes (catalog_cubes, ...) | `semql.introspect` |
| Planner / router prompt fragments | `semql-prompt` (sibling package) |
| Dialect strategies + sqlglot dialect adapter | `semql.backend`, `semql.dialect` |
| Visualisation decision (chart type, axes, formats) | `semql.visualize` |
| `is_read_only_statement` post-hoc SQL guard | `semql.safe` |
| Structured error hierarchy | `semql.errors` |

## Features

- **Compare windows** — `CompareWindow(mode="previous_period")` wraps
  the inner query in `current` / `prior` CTEs joined via `FULL OUTER
  JOIN` and emits `{m}_current` / `{m}_prior` / `{m}_delta` /
  `{m}_pct_change` columns per measure.
- **Temporal model** — time dimensions group by `second` / `minute` /
  `hour` / `day` / `week` / `month` / `quarter` / `year`; a
  `type="date"` time dimension drops sub-day grain and timezone shifts;
  per-cube `timezone` makes
  `date_trunc` tenant-correct and transpiles per dialect (`AT TIME
  ZONE`, `CONVERT_TIMEZONE`, ClickHouse's native arg, …); per-cube
  `week_start` (`monday` default, or `sunday`) sets the `week` bucket
  boundary consistently across dialects.
- **Explicit raw SQL** — every hand-written fragment (`Measure.sql`,
  `Join.on`, `Cube.base_predicate`, …) is wrapped in a `RawSQL` marker
  at validation: when raw SQL is used, the model says so.
- **Tenancy** — per-cube `NONE` (default; honestly unscoped),
  `SCHEMA` (`{tenant_schema}` substituted from the identity's `tenant`,
  required when declared) or `DISCRIMINATOR` (compiler wraps the source
  in a subquery with one bound `WHERE` predicate per `tenancy_columns`
  entry, so composite tenant keys need no workaround). `tenant` is
  first-class on `AuthContext`; `Catalog(strict_tenancy=True)` rejects
  any cube left with no isolation, scope, or `required_roles`.
- **Row-level security** — `Cube.security_sql` AND-composes with
  tenancy inside the isolation subquery; `{ctx.X}` placeholders bind
  as parameters, never inline as literals.
- **MCP-ready** — `build_planner_prompt_fragment(catalog.as_dict())`
  (in `semql-prompt`) produces the planner system-prompt fragment;
  `semql-mcp` wraps it as a server.
- **Pluggable backends** — `DialectStrategy` Protocol lets out-of-tree
  Snowflake / BigQuery adapters slot in without forking the compiler.

## Philosophy

See `PHILOSOPHY.md` at the repo root. Highlights:
- The emitted SQL must be readable by the engineer debugging a
  production incident at 2am.
- Compile errors beat runtime errors; runtime errors beat wrong results.
- `compile()` fails at the first problem; `validate()` collects them all.
- Catalogs are data; the META cubes expose the catalog through the
  same compiler path a normal query takes.

## Status

Pre-v1. The shape is stable, but minor names / fields may move before
the v1 contract locks. Tests pin every public behaviour the README
documents.
