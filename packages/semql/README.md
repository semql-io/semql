# semql

Pure-Python compiler from a semantic spec to backend SQL. Define
cubes (dimensions, measures, time-dimensions, joins) once; emit
correct, parameterised SQL for Postgres, ClickHouse, DuckDB and
(via the strategy seam) Snowflake / BigQuery.

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
- **Temporal model** — time dimensions group by `hour` / `day` /
  `week` / `month` / `quarter` / `year`; a `type="date"` time dimension
  drops sub-day grain and timezone shifts; per-cube `timezone` makes
  `date_trunc` tenant-correct and transpiles per dialect (`AT TIME
  ZONE`, `CONVERT_TIMEZONE`, ClickHouse's native arg, …).
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
- Correct SQL, not optimal. The query planner is the database's job.
- The emitted SQL must read like something a human could have written.
- `compile()` fails at the first problem; `validate()` collects them all.
- The catalog is data; reflection isn't an afterthought.

## Status

Pre-v1. The shape is stable, but minor names / fields may move before
the v1 contract locks. Tests pin every public behaviour the README
documents.
