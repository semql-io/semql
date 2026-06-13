# SemQL

Semantic data layer with SQL generation, authorisation, and a typed
LLM prompt pipeline.

## What it does

Define **semantic cubes** — dimensions, measures, filters, and joins
over your tables. Organize them in a **catalog**. SemQL turns a
declarative `SemanticQuery` into parameterised SQL across Postgres /
ClickHouse / DuckDB / BigQuery / Snowflake.

- **SQL generation** — semantic spec → dialect-aware SQL with bound params
- **Authorisation** — `AuthContext` viewer + cube `required_roles` +
  registered `ScopeFn`s. The compiler refuses queries that touch a
  cube the viewer can't see, and injects row-level predicates inside
  the cube's alias subquery so outer ORs can't bypass them.
- **Time spine** — `fill_nulls_with` emits one row per bucket in
  range, COALESCEd, across all five backends.
- **Pipeline prompts** — four typed roles (Router / Generator /
  Presenter / Drilldown) with structured Pydantic outputs. Plug into
  any LLM client; no vendor lock-in.
- **MCP server** — auto-generated `query_<cube>()` tools per cube.
- **Drift check** — `semql-validate-db` probes a live database
  against a catalog before deploy.

## Packages

| Package | Description |
|---|---|
| `semql` | Core: catalog, compiler, auth, prompt fragments, introspection |
| `semql-mcp` | MCP server wrapping a catalog |
| `semql-erd` | Graphviz ER-diagram generator for catalogs |
| `semql-validate-db` | Pre-deploy drift check against a live database |

## Install

```sh
pip install semql
pip install semql-mcp           # + MCP server
pip install semql-erd           # + ER diagrams
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
    backend=Dialect.POSTGRES,
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
print(catalog.prompt(viewer=viewer))   # planner-facing fragment, viewer-filtered
```

Catalog cubes the viewer can't see vanish from the rendered prompt.

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
skill format — Claude Code, Codex CLI, Cursor, Gemini CLI, Copilot,
and others. Use the vercel-labs CLI to install them into your
agent's expected directory:

```sh
npx skills install <agent-name>
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
