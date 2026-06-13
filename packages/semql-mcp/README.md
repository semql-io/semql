# semql-mcp

An MCP server that wraps a [`semql`](../semql) `Catalog` and exposes
its compiler / validator / prompt-renderer surfaces as tools any MCP
client can call. Built on [FastMCP](https://github.com/jlowin/fastmcp).

## Two modes

By default the server is **compile-only**. `semql` is a pure compiler —
no I/O — and this server keeps that contract. Tools return the emitted
SQL and bound parameters; the caller runs the SQL against whatever
backend they own.

Pass an `executor` at construction to opt into **exec mode**. A
`query_execute` tool registers in addition to the compile-only tools;
it runs the SQL against your executor and returns both the SQL/params
envelope and the resulting rows.

## Install

```sh
pip install semql-mcp
```

## Quick start — compile-only

```python
from semql import Dialect, Catalog, Cube, Dimension, Measure
from semql_mcp import MCPServer

catalog = Catalog([
    Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    ),
])

server = MCPServer(catalog)
server.run(transport="stdio")  # speak JSON-RPC over stdin/stdout
```

## Quick start — exec mode

Bring your own database driver and adapt its row shape to a list of
dicts:

```python
import psycopg
from psycopg.rows import dict_row

from semql_mcp import MCPServer


def executor(sql: str, params: dict) -> list[dict]:
    with psycopg.connect("postgresql://...", row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


server = MCPServer(catalog, executor=executor)
server.run(transport="stdio")
```

The MCP server never imports a database driver. Whatever you wire in
is what gets called; semql-mcp just hands it `(sql, params)` and
expects `list[dict]` back.

## Tools

Always registered:

| Tool | Description |
|---|---|
| `query_semantic(spec, context?)` | Compile a SemanticQuery; return `{dialect, sql, params, columns}`. |
| `validate(spec)` | Collect-all static validation; returns `list[ValidationError]`. Empty when the query would compile cleanly. |
| `explain(spec, context?)` | Compile and return just the SQL string. |
| `catalog_prompt(only_exposed=True, include_introspection=False)` | Render the planner prompt fragment for the catalog. |

Registered when `executor` is supplied:

| Tool | Description |
|---|---|
| `query_execute(spec, context?)` | Compile + run. Returns the `query_semantic` shape plus `rows: list[dict]`. Errors carry the SQL we tried to run so callers can replay / inspect it. |

### Auto-generated per-cube tools

For each `expose_in_prompt=True` (non-META) cube, the server also
registers a `query_<cube_name>` tool whose `measures`, `dimensions`,
`order` (and `time_window.dimension`, when applicable) parameters are
**`Literal`-typed enums** of the cube's actual fields. The planner
sees a JSON Schema with explicit allowed values rather than the bare
`list[str]` `query_semantic` accepts.

Field names are **bare** (no cube prefix); the tool auto-qualifies as
it builds the `SemanticQuery`:

```jsonc
// query_orders
{
  "measures": ["revenue"],
  "dimensions": ["region"],
  "filters": [{"dimension": "status", "op": "eq", "values": ["paid"]}],
  "time_window": {
    "dimension": "created_at",
    "granularity": "day",
    "range": ["2026-01-01", "2026-02-01"]
  },
  "limit": 100
}
```

Multi-cube queries (joins across cubes) still go through
`query_semantic` — the per-cube tools are scoped to a single cube by
construction. When `executor` is configured, the per-cube tools
return rows too.

## In-process testing

FastMCP's `Client` connects to a `FastMCP` instance without a transport
— useful for end-to-end testing of your catalog + planner together:

```python
import asyncio
from fastmcp import Client
from semql_mcp import MCPServer

server = MCPServer(catalog)

async def smoke() -> None:
    async with Client(server.mcp) as c:
        tools = await c.list_tools()
        print([t.name for t in tools])
        result = await c.call_tool("explain", {"spec": {"measures": ["orders.revenue"]}})
        print(result.data)

asyncio.run(smoke())
```

## Status

Early development. The tool surface is stable.
