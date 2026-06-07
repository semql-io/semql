# SemQL

Semantic data layer with SQL generation and MCP exposure.

## What it does

Define **semantic cubes** — dimensions, measures, filters, and joins over your tables.
Organize them in a **catalog**. SemQL handles the rest.

- **SQL generation** — turn a semantic spec into dialect-aware SQL
- **MCP server** — exposes `query_semantic(spec)` plus auto-generated
  `query_<cube>()` tools so any MCP client can query your catalog
- **Prompt fragment** — `catalog.prompt()` renders a system-prompt
  fragment that teaches an LLM your catalog schema so it can emit
  `SemanticQuery` specs your code compiles to SQL

## Packages

| Package | Description |
|---|---|
| `semql` | Core: cube definitions, catalog, SQL generation, prompt rendering |
| `semql-mcp` | MCP server wrapping a catalog |
| `semql-erd` | Graphviz ER-diagram generator for catalogues |
| `semql-validate-db` | Pre-deploy drift check against a live database |

## Install

```sh
pip install semql
pip install semql-mcp     # + MCP server
pip install semql-erd     # + ER diagrams
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
sql = compiled.sql
```

Cubes are Pydantic models — every field is keyword-only and typed. The
`{o}` placeholder in `sql` is the cube's `alias`; the compiler resolves
it (and `{schema}`-style context placeholders) at compile time so the
emitted SQL is always alias-qualified.

To surface the catalogue to a language model, render the prompt fragment:

```python
print(catalog.prompt())
```

That fragment describes the cube vocabulary and the `SemanticQuery`
shape so a planner LLM can emit valid specs your code compiles to SQL.

## Status

Early development. Contributions welcome.
