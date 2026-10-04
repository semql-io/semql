# semql-engine

In-process executor for [`semql`](https://github.com/semql-io/semql)
`FederatedPlan` results. Runs each per-backend fragment via a
caller-supplied `Adapter`, materialises the rows into in-memory DuckDB,
then runs the plan's merge SQL against the assembled tables.

`semql` core stays sans-io. `semql-engine` is the opt-in package that
turns a `FederatedPlan` into result rows when you want the cross-source
execution done for you.

## Install

```sh
pip install semql-engine
```

## Quickstart

```python
import duckdb
from semql import Catalog, Dialect, compile_federated_query
from semql_engine import DuckDBAdapter, Engine

catalog = Catalog([...])  # cubes spanning multiple backends
plan = compile_federated_query(query, catalog.as_dict())

engine = Engine()
engine.register(Dialect.POSTGRES, my_pg_adapter)
engine.register(Dialect.BIGQUERY, my_bq_adapter)
result = engine.run(plan)
result.columns  # ['region', 'revenue', ...]
result.rows     # [(...), ...]   — or stream with engine.iter_rows(plan)
```

## What it does

For every fragment in the plan, the engine calls the adapter registered
for that backend with `(sql, params)`. It loads the resulting rows into
a DuckDB table named `frag_<i>` (matching `FederatedPlan.fragments`
indices) and finally applies `plan.merge_spec` — rendered to DuckDB SQL
by `semql_engine.merge` — to produce the merged shape.

Single-fragment plans (single-backend queries that went through
`compile_federated_query` anyway) work transparently — the merge is a
pass-through.

## Execution integrity and analysis

Both synchronous and asynchronous execution validate executable versions and
every fragment's final binding manifest **before** any source adapter is called.
The synchronous cache path performs that preflight before lookup as well.
Merge bindings are validated before merge execution. Missing/renamed parameters,
unsupported versions, and mismatched output columns raise
`ExecutionContractError`, catchable as both `EngineError` and core `ContractError`.
Driver exceptions are not silently converted into successful results.

`result.analysis` belongs to the current artifact, including on cache hits.
`result.validation_evidence` describes materialized merge-key checks actually
performed; a cache hit does not invent fresh validation evidence. Iterator
results expose their own `analysis` and `validation_evidence`, not shared
per-engine state. The async engine does not provide a result cache.

Declared one-side merge keys are checked together on the exact materialized
fragment consumed by the merge, respecting its null-key semantics. Observed
duplicates and uncheckable explicit materialized-key obligations reject.
Underlying database keys/cardinalities that cannot be inspected remain trusted
catalog assumptions; the executor does not run a separate database-probing
service or require host waivers for those declarations.

Internal exception/artifact dumps may include private information. Use
`error.to_public_payload()` for routine external diagnostics. Recompile stored
plans with unsupported executable versions rather than changing their version
number by hand.

## Adapters

An `Adapter` is anything with `execute(sql, params) -> AdapterResult`.
`AdapterResult` carries `columns: list[str]`, positional `rows`, and optional
`column_types: list[str | None]`, aligned exactly with the columns. Physical
type declarations preserve numeric types, including decimal precision/scale,
even when a fragment is empty or its values are all NULL. Built-ins:

- `DuckDBAdapter(con)` — runs the SQL inside an existing DuckDB
  connection. Useful for local CSV / Parquet enrichment cubes.
- `DBAPIAdapter(con)` — wraps any PEP-249 connection (psycopg, mysql,
  sqlite, etc).

Bring your own for warehouses that need a vendor SDK.

Built-in adapters retain driver schema metadata where available. Custom adapters
should supply supported physical type names such as `INTEGER`, `DOUBLE`, or
`DECIMAL(12,2)`; common BigQuery/ClickHouse names are normalized for DuckDB.
Missing declarations can be inferred only from supported non-NULL values.
An empty/all-NULL undeclared column raises `physical_schema_missing`;
unsupported declared types raise `physical_schema_unsupported` rather than
silently materializing numeric data as text.

DB-API decimal precision/scale is retained when supplied. An unbounded
`NUMERIC` declaration without those parameters is unknown, not a fixed
BigQuery decimal; inference still requires observed values. Python `float`
type codes map to `DOUBLE` so their mantissa is not narrowed.

DuckDB materialization accepts decimal precision up to 38 digits. Higher declared
precision (including BigQuery's default `BIGNUMERIC(76,38)`) and over-precision
legacy `Decimal` values are rejected explicitly, not narrowed to an integer or
rounded to fit.

The engine owns acquired row iterators and closes closeable iterators after
materialization, exhaustion, errors, or explicit stream closure. It does not
close caller-owned adapter connections. Async fragment failure or caller
cancellation cancels and drains request-owned tasks before returning; an
already-running `asyncio.to_thread` worker cannot be forcibly stopped, so sync
drivers still need their own query timeouts/cancellation policy.

Cached rows isolate nested list/dict/tuple cells both on insertion and cache
hits. Mutating a miss result or a later hit cannot change the cached value.
Uncached execution does not perform these cache-isolation copies.

## Semi-joins

A cross-backend semi-join (restrict an outer dimension to the value set
of an inner query, shipped as a value list rather than a join) compiles
to a `SemiJoinPlan` via `semql.compile_semi_join_query`. Run it with
`run_semi_join(plan, engine)` — it executes each inner plan, projects the
key column to a value list, and runs the outer query with that list bound
as an `IN` / `NOT IN` filter:

```python
from semql import compile_semi_join_query
from semql_engine import Engine, run_semi_join

plan = compile_semi_join_query(query, catalog.as_dict())
result = run_semi_join(plan, engine)  # same ExecutionResult shape as run()
```

## Scope

The supported strategy is explicit:

- Distributive federation supports existing legal recipes, including SUM of
  partial counts and AVG recomposed from partial sums and non-null counts.
- Explicit `mode="raw_rows"` preserves existing raw-row aggregation recipes,
  including distinct counts, min/max, percentiles, and catalog-declared ratios.
  The compiler never silently switches to this more expensive strategy.
- Multi-backend inline-derived measures reject instead of disappearing.
- Aliases and requested output order survive the final merge.
- Symmetric multi-fact aggregation uses **observed qualifying facts**, not an
  implicit full dimension universe. Keys present in only one fact, including
  qualifying all-null fact rows, remain; keys absent from all facts do not.
- Cross-backend joins retain the existing supported equality-key restrictions.
  Cross-backend comparison/fill and independently planned new operators are not
  added by the semantic-contract release.

The engine itself is small; most of the federation logic lives in
`semql.federate`.

## Backend conformance tests

From the workspace root, with Docker running:

```sh
uv sync --python 3.12 --all-extras --frozen
just test-conformance
```

This enables `SEMQL_CONFORMANCE_TESTCONTAINERS=1`. The existing result fixtures
provision disposable PostgreSQL and ClickHouse containers on dynamically mapped
ports, execute emitted SQL without rewriting, and stop the containers afterward.
Startup or readiness failures fail the requested run rather than silently skipping.
DuckDB runs in-process and does not require Docker.

`SEMQL_CONFORMANCE_POSTGRES` (a psycopg DSN) and
`SEMQL_CONFORMANCE_CLICKHOUSE` (an HTTP URL) override provisioning per backend,
allowing the same fixtures to target externally managed disposable services.
Without the Testcontainers opt-in or an explicit endpoint, external-backend cells
skip in the ordinary local suite. The full-suite CI job enables Testcontainers.

## Status

Pre-v1. Compiler and executor packages release in lockstep. The first-release
semantic contract and verification inventory are recorded in
[`compiler-semantic-contract-2026-10-01.md`](../../docs/plans/compiler-semantic-contract-2026-10-01.md).
