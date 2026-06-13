# semql

Pure-Python compiler from a semantic spec to backend SQL. Define
cubes (dimensions, measures, time-dimensions, joins) once; emit
correct, parameterised SQL for Postgres, ClickHouse, DuckDB and
(via the strategy seam) Snowflake / BigQuery.

`semql` does **no I/O**: catalogs are Python data; the compiler
returns SQL + bound params; running the SQL is the caller's job.
Prompt-fragment rendering for LLM planners ships in the core
(`semql.prompt`). Sibling packages add MCP exposure (`semql-mcp`)
and ER diagrams (`semql-erd`).

## Install

```sh
pip install semql
```

## Quick start

```python
from semql import (
    Backend,
    Catalog,
    Cube,
    Dimension,
    Measure,
    SemanticQuery,
)

orders = Cube(
    name="orders",
    backend=Backend.POSTGRES,
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
# compiled.sql, compiled.params, compiled.columns, compiled.backend
```

The `{o}` placeholder in a cube's `sql` is its alias; the compiler
resolves it (along with `{schema}`-style context placeholders and
`{ctx.X}` row-level-security placeholders) at compile time.

## What lives in the box

| Surface | Module |
|---|---|
| Cube / Measure / Dimension / TimeDimension / Join | `semql.model` |
| SemanticQuery / Filter / TimeWindow / CompareWindow | `semql.spec` |
| Catalog wrapper (validation, prompt, compile entry) | `semql.catalog` |
| Compiler — sqlglot AST → dialect SQL | `semql.compile` |
| Collect-all static validator | `semql.validate` |
| Reflection cubes (catalog_cubes, ...) | `semql.introspect` |
| Planner / router prompt fragments | `semql.prompt` |
| Backend strategies + sqlglot dialect adapter | `semql.backend`, `semql.dialect` |
| Visualisation decision (chart type, axes, formats) | `semql.visualize` |
| `is_read_only_statement` post-hoc SQL guard | `semql.safe` |
| Structured error hierarchy | `semql.errors` |

## Features

- **Compare windows** — `CompareWindow(mode="previous_period")` wraps
  the inner query in `current` / `prior` CTEs joined via `FULL OUTER
  JOIN` and emits `{m}_current` / `{m}_prior` / `{m}_delta` /
  `{m}_pct_change` columns per measure.
- **Tenancy** — per-cube `SCHEMA` (default; `{tenant_schema}`
  substitution) or `DISCRIMINATOR` (compiler wraps the source in a
  subquery with `WHERE tenancy_column = $tenant`).
- **Row-level security** — `Cube.security_sql` AND-composes with
  tenancy inside the isolation subquery; `{ctx.X}` placeholders bind
  as parameters, never inline as literals.
- **MCP-ready** — `Catalog.prompt()` produces the planner system-prompt
  fragment; `semql-mcp` wraps it as a server.
- **Pluggable backends** — `BackendDialect` Protocol lets out-of-tree
  Snowflake / BigQuery adapters slot in without forking the compiler.

## Philosophy

See `PHILOSOPHY.md` at the repo root. Highlights:
- Correct SQL, not optimal. The query planner is the database's job.
- The emitted SQL must read like something a human could have written.
- `compile()` fails at the first problem; `validate()` collects them all.
- The catalog is data; reflection isn't an afterthought.

## Status

Pre-v1. The shape is stable, but minor names / fields may move before
the v1 contract locks. Tests pin every public behaviour the README
documents.
