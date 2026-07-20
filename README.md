# SemQL

Semantic Query Layer with SQL generation, authorisation, and a typed
LLM prompt pipeline.

## What it does

Define **semantic cubes** — dimensions, measures, filters, and joins
over your tables. Organize them in a **catalog**. SemQL turns a
declarative `SemanticQuery` into parameterised SQL across Postgres,
ClickHouse, DuckDB, BigQuery, and Snowflake, plus the analytics
engines Redshift, Trino, and Databricks.

- **SQL generation** — semantic spec → dialect-aware SQL with bound params
- **Authorisation** — `AuthContext` viewer + cube `required_roles` +
  registered `ScopeFn`s. The compiler refuses queries that touch a
  cube the viewer can't see, and injects row-level predicates inside
  the cube's alias subquery so outer ORs can't bypass them.
- **Time spine** — `fill_nulls_with` emits one row per bucket in
  range, COALESCEd, across the five stable backends. SQL Server,
  MySQL, and Oracle are experimental; gap-filling time spines are
  not yet implemented on any of them.
- **Pipeline prompts** — four typed roles (Router / Generator /
  Presenter / Drilldown) with structured Pydantic outputs. Plug into
  any LLM client; no vendor lock-in.
- **MCP server** — auto-generated `query_<cube>()` tools per cube.
- **Drift check** — `semql-validate-db` probes a live database
  against a catalog before deploy.

## Install

```sh
pip install semql
pip install semql-auth          # + credential→identity adapters
pip install semql-engine        # + in-process federated executor
pip install semql-erd           # + ER diagrams
pip install semql-introspect    # + catalog bootstrap from a live DB
pip install semql-mcp           # + MCP server
pip install semql-prompt        # + LLM prompt fragments
pip install semql-validate-db   # + drift checker
```

## Quick start

```python
from semql import (
    AuthContext, Dialect, Catalog, Cube, Dimension, Measure,
    ScopePredicate, SemanticQuery, TimeDimension, TimeWindow,
)

orders = Cube(
    name="orders",
    dialect=Dialect.POSTGRES,
    table="orders",
    alias="o",
    measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
    dimensions=[
        Dimension(name="region", sql="{o}.region", type="string"),
        Dimension(name="rep_id", sql="{o}.rep_id", type="string"),
    ],
    time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    scope="my_team",                    # row-level scoping via ScopeFn
)


def my_team(_cube, viewer):
    """Sales reps see only their team's orders. Admins see all."""
    if "admin" in viewer.roles:
        return None
    return ScopePredicate(
        sql="{o}.rep_id IN (SELECT id FROM reps WHERE team = {ctx.viewer_team})",
        ctx_keys=["ctx.viewer_team"],
    )


catalog = Catalog([orders], scope_fns={"my_team": my_team})

viewer = AuthContext(viewer_id="alice@example.com", roles=["sales"])
compiled = catalog.compile(
    SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2026-01-01", "2026-04-01"),
            fill_nulls_with=0,         # unbroken monthly axis
        ),
    ),
    viewer=viewer,
    context={"ctx.viewer_team": "EMEA-East"},
)

sql = compiled.sql       # parameterised SQL with spine + scope wrap
params = compiled.params  # {'p0': 'EMEA-East', 'p1': '2026-01-01', ...}
```

The emitted SQL wraps `orders` in a subquery that AND-composes the
scope predicate inside the alias, so no outer `OR` can reach a row
the viewer isn't authorised to see. Identity and context values
always bind as parameters — never SQL literals.

## Run it end-to-end

`semql` stops at SQL — running it is the caller's job. `semql-engine`
closes the loop: register one adapter per backend and it executes the
plan (federating across backends when a query spans them) and hands
back rows.

```python
import duckdb
from semql import (
    Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery,
    compile_federated_query,
)
from semql_engine import DuckDBAdapter, Engine

con = duckdb.connect()
con.execute("CREATE TABLE orders(region VARCHAR, amount INT)")
con.execute("INSERT INTO orders VALUES ('EMEA', 100), ('EMEA', 50), ('US', 30)")

orders = Cube(
    name="orders", dialect=Dialect.DUCKDB, table="orders", alias="o",
    measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
    dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
)
catalog = Catalog([orders])

plan = compile_federated_query(
    SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
    catalog.as_dict(),
)

engine = Engine()
engine.register(Dialect.DUCKDB, DuckDBAdapter(con))
result = engine.run(plan)

result.columns  # ['region', 'revenue']
result.rows     # [('EMEA', 150), ('US', 30)]
```

`compile_federated_query` returns a one-fragment plan for a
single-backend query and a multi-fragment plan (per-backend SQL + a
merge spec) when cubes span dialects — `engine.run` executes either.

## The four-role prompt pipeline

Bring your own LLM. Each role pairs a **prompt-fragment builder**
(splices into your system prompt) with a **typed Pydantic output**
(the structured response the LLM returns).

| Role | Fragment builder | Output model |
|---|---|---|
| Router | `build_router_prompt_fragment` | `RouterDecision` |
| Query Generator | `build_query_generator_prompt_fragment` | `QueryPlan` (list of `QueryStep` with intent) |
| Presenter | `build_presenter_prompt_fragment` | `Presentation` |
| Drilldown | `build_drilldown_prompt_fragment` | `DrilldownSuggestions` |

See `demos/pipeline_demo.py` for the full Router → Generator →
Compile → Presenter → Drilldown flow against a small realistic
catalog. Run with `uv run demos/pipeline_demo.py`.

## Surface to LLMs

```python
from semql_prompt import build_planner_prompt_fragment

# planner-facing fragment, viewer-filtered
print(build_planner_prompt_fragment(catalog.as_dict(), viewer=viewer))
```

Catalog cubes the viewer can't see vanish from the rendered prompt.

## Minimal example: MCP server

Wrap a catalog in a stdio MCP server so any MCP client (Claude Code,
Cursor, etc.) can call `query_semantic`, `validate`, `explain`, and the
auto-generated `query_<cube>` tools.

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
server.run(transport="stdio")
```

For exec mode, pass an `executor: (sql, params) -> list[dict]` and a
`query_execute` tool registers alongside the compile-only ones.

## Minimal example: Pydantic text-to-SQL chatbot agent

Pair prompt-fragment builders with typed Pydantic outputs. The
generator's `QueryPlan` contains `SemanticQuery` objects you feed
straight into `Catalog.compile`.

```python
from pydantic_ai import Agent
from semql import Catalog
from semql.plan import QueryPlan
from semql_prompt import build_query_generator_prompt_fragment

generator = Agent(
    model="openai:gpt-4o",
    system_prompt=build_query_generator_prompt_fragment(catalog.as_dict()),
    result_type=QueryPlan,
)

def answer(question: str, catalog: Catalog, run_sql) -> str:
    plan: QueryPlan = generator.run_sync(question).data
    if not plan.steps:
        return "I couldn't map that to the catalog."
    head = plan.steps[0]
    compiled = catalog.compile(head.query)
    rows = run_sql(compiled.sql, compiled.params)  # your DB driver
    return f"{head.label or 'Result'}: {rows}"
```

The four prompt-fragment builders (`build_router_prompt_fragment`,
`build_query_generator_prompt_fragment`, `build_presenter_prompt_fragment`,
`build_drilldown_prompt_fragment`) pair one-to-one with the four output
models in `semql.plan` (`RouterDecision`, `QueryPlan`, `Presentation`,
`DrilldownSuggestions`). `demos/pipeline_demo.py` wires the full
Router → Generator → Compile → Presenter → Drilldown chain.

## Skills

Two skills ship in the repo, both following the
[vercel-labs/skills](https://github.com/vercel-labs/skills) convention
(`skills/<name>/SKILL.md` with YAML frontmatter + tool-agnostic
markdown body):

- `skills/semql-requirement-discovery/SKILL.md` — interview a
  developer about analytics intent, emit a structured requirements
  doc.
- `skills/semql-cube/SKILL.md` — author Cube definitions from that
  doc (or directly).

The skills work across any agent that consumes the vercel-labs
skill format — Claude Code, Codex CLI, Cursor, Gemini CLI, GitHub
Copilot, and others. Use the [vercel-labs CLI](https://github.com/vercel-labs/skills)
to install them into your agent's expected directory:

```sh
# install both skills into Claude Code
npx skills add semql-io/semql -a claude-code

# install into multiple agents at once
npx skills add semql-io/semql -a claude-code -a codex -a cursor

# install a single skill, globally
npx skills add semql-io/semql -g -a claude-code --skill semql-cube
```

The skill bodies are deliberately tool-agnostic: they say *what* to
elicit ("ask the user which roles are restricted"), not *how* to
ask (the runtime's question-asking mechanism — Claude Code's
`AskUserQuestion`, a numbered CLI prompt, etc. — varies and is
called out where the skill needs it).

## Status

Early development. The public surface (everything re-exported from
`semql`) is intended to be stable; internals (anything prefixed with
`_`) are not.
