# semql-engine

In-process executor for [`semql`](https://github.com/npalladium/semql)
`FederatedPlan` results. Runs each per-backend fragment via a
caller-supplied `Adapter`, materialises the rows into in-memory DuckDB,
then runs the plan's merge SQL against the assembled tables.

`semql` core stays sans-io. `semql-engine` is the opt-in package that
turns a `FederatedPlan` into result rows when you want the cross-source
execution done for you.

## Quickstart

```python
import duckdb
from semql import Catalog, compile_federated_query
from semql_engine import DuckDBAdapter, Engine

catalog = Catalog([...])  # cubes spanning multiple backends
plan = compile_federated_query(query, catalog.as_dict())

engine = Engine()
engine.register(Backend.POSTGRES, my_pg_adapter)
engine.register(Backend.BIGQUERY, my_bq_adapter)
rows = list(engine.run(plan))
```

## What it does

For every fragment in the plan, the engine calls the adapter registered
for that backend with `(sql, params)`. It loads the resulting rows into
a DuckDB table named `frag_<i>` (matching `FederatedPlan.fragments`
indices) and finally runs `plan.merge.sql` to produce the merged shape.

Single-fragment plans (single-backend queries that went through
`compile_federated_query` anyway) work transparently — the merge is a
pass-through.

## Adapters

An `Adapter` is anything with `execute(sql, params) -> AdapterResult`
where `AdapterResult` carries `columns: list[str]` and an iterable of
row dicts. Built-ins:

- `DuckDBAdapter(con)` — runs the SQL inside an existing DuckDB
  connection. Useful for local CSV / Parquet enrichment cubes.
- `DBAPIAdapter(con)` — wraps any PEP-249 connection (psycopg, mysql,
  sqlite, etc).

Bring your own for warehouses that need a vendor SDK.

## Scope

v1 mirrors `compile_federated_query` v1:

- Sum / count / avg supported (avg is decomposed at compile and
  recomposed in the merge SQL); other aggregations are refused by the
  compiler before the engine ever sees them.
- Equality bridge joins only.
- No `compare` mode, no boolean `where` tree across backends.

The engine itself is small; most of the federation logic lives in
`semql.federate`.
