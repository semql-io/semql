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

## Top grouped rows per partition

Select the top three sales reps in each region, using the catalog above:

```python
from semql import PartitionRank, RankOrder

ranked = catalog.compile(
    SemanticQuery(
        dimensions=["orders.region", "orders.rep_id"],
        measures=["orders.revenue"],
        rank=PartitionRank(
            partition_by=["orders.region"],
            order_by=[
                RankOrder(field="orders.revenue", direction="desc", nulls="last")
            ],
            count=3,
            ties="exact",
        ),
        order=[("orders.region", "asc"), ("orders.revenue", "desc")],
    ),
    viewer=viewer,
    context={"ctx.viewer_team": "EMEA-East"},
)
```

Ranking runs **after aggregation, selected derived measures, and HAVING**, but
before final ordering and any global limit/offset. `ties="exact"` selects up to
N rows per partition with deterministic remaining-group-key tie breakers;
`ties="include"` includes every tie at the Nth boundary (`RANK`, not
`DENSE_RANK`). Null scores default to `NULLS LAST`. Ranking requires grouped,
single-backend output; comparison, dense-time fill, and genuine multi-backend
federation are unsupported. It is a `SemanticQuery` API, not a semantic-SQL
parser/serializer round-trip. See the
[migration guide](docs/migrations/planner-contracts-ranking.md) and
[`demos/partitioned_ranking_demo.py`](demos/partitioned_ranking_demo.py).

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

For offline, end-to-end recipes covering trusted request scope, semantic
comparison, alias-aware enrichment, explicit federation, cache partitioning,
and stream cleanup, see [`demos/integration_recipes.py`](demos/integration_recipes.py).
Run it against installed packages (not editable workspace imports) using the
steps in the [0.7-to-0.8 migration guide](docs/migrations/0.7-to-0.8.md).
For the next release's budget-admission and prompt-result changes, see the
[0.8-to-0.9 migration guide](docs/migrations/0.8-to-0.9.md).

For loss-aware introspection reports, descriptive catalog provenance/digests,
and opt-in execution admission, deadline, row and value-byte limits, see the
[reliable integration guide](docs/migrations/reliable-integration.md).
Programmer-authored declarations remain trusted; resource limits are optional.

## The four-role prompt pipeline

Bring your own LLM. Each role pairs a **prompt-fragment builder**
(splices into your system prompt) with a **typed Pydantic output**
(the structured response the LLM returns).

| Role | Fragment builder | Output model |
|---|---|---|
| Router | `build_router_prompt_fragment` | `RouterDecision` |
| Query Generator | `build_query_generator_prompt_fragment` | `PlannerResponse` (plan / context / clarification / cannot-plan) |
| Presenter | `build_presenter_prompt_fragment` | `Presentation` |
| Drilldown | `build_drilldown_prompt_fragment` | `DrilldownSuggestions` |

See `demos/pipeline_demo.py` for the full Router → Generator →
Compile → Presenter → Drilldown flow against a small realistic
catalog. Run with `uv run demos/pipeline_demo.py`.

## Fixed-member cohorts

To measure the same selected entities in another period, use a single-hop
`CohortQuery`, not independent rankings for each period. Given a catalog with
`stores.id`, `stores.region`, `stores.period`, and `stores.conversion`:

```python
from semql import (
    Cohort, CohortMeasurement, CohortQuery, Filter,
    PartitionRank, RankOrder, SemanticQuery, compile_cohort_query,
)
from semql_engine import run_cohort

request = CohortQuery(
    cohort=Cohort(
        source=SemanticQuery(
            dimensions=["stores.id", "stores.region"],
            measures=["stores.conversion"],
            filters=[Filter(dimension="stores.period", op="eq", values=["prior"])],
            rank=PartitionRank(
                partition_by=["stores.region"],
                order_by=[RankOrder(field="stores.conversion", direction="desc")],
                count=1,
            ),
        ),
        member_keys=["stores.id"],
        attributes=["stores.region"],
        max_members=100,
    ),
    measurements=[
        CohortMeasurement(
            name="current",
            query=SemanticQuery(
                dimensions=["stores.id"],
                measures=["stores.conversion"],
                filters=[Filter(dimension="stores.period", op="eq", values=["current"])],
            ),
            key_mapping={"stores.id": "stores.id"},
        ),
    ],
)
plan = compile_cohort_query(request, catalog.as_dict(), viewer=viewer)
result = run_cohort(plan, engine)
```

Execution is staged on the engine's registered adapters, even on one backend.
The engine validates unique, non-NULL member keys and row bounds before dependent
measurement I/O; it never silently truncates or substitutes new winners.
Selection-time attributes remain separate from measurement-time attributes.
Missing measurements preserve the member with NULL metrics and explicit
`presence`/`coverage` evidence.

Add named `CohortComparison` values for NULL-preserving differences, ratios, or
percentage changes. Do not sum finalized member rates or distinct counts.
This API does not promise snapshots, persistence, nested cohorts, or an arbitrary
stage DAG. Child query compiler/backend restrictions still apply.

Generation is opt-in with `build_generation_contract(..., output_model=CohortQuery)`;
the default planner response is unchanged. See the
[cohort contract guide](docs/migrations/cohort-spine.md) and run
`uv run python demos/cohort_spine_demo.py` for a full authorized example.

## Surface to LLMs

```python
from semql import SchemaProfile
from semql_prompt import build_generation_contract

contract = build_generation_contract(
    catalog.as_dict(), viewer=viewer, profile=SchemaProfile.STRICT,
)
prompt_fragment = contract.text
response_schema = contract.schema
losses = contract.report.removed_features, contract.report.weakened_constraints
```

Supply this paired prompt/schema to your own structured-output client, then
call `contract.validate_payload(parsed_payload)`. `SchemaProfile.TOOL` preserves
recursive Boolean filters and semi-joins; `STRICT` omits `where` and `semi_joins`
and weakens tuple-position schema validation for restrictive decoders. Inspect
the loss report and still validate canonically. You can explicitly disable
`SemanticFeature.RANK`, `WHERE`, or `SEMI_JOINS` for a request.

A profile describes **representation**, not compiler support or authorization.
Catalog visibility is viewer-filtered; every executable proposal still needs
current host authorization and compilation. Adaptive and repair prompts include
mandatory output schemas, and continuation locks cannot be trimmed away to make
a budget fit. Check the returned `fits` before calling a model; the adaptive and
repair demos use `max_tokens=40_000`, not a guaranteed minimum for other catalogs.

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

## Minimal example: provider-neutral text-to-SQL chatbot

The generator now returns a `PlannerResponse` with four peer outcomes, not an
empty `QueryPlan` to signal failure. The function below consumes a parsed JSON
payload from your own client, using the same generation contract supplied to
that client. It executes only plan outcomes and leaves context retrieval,
clarification, rendering, and model calls with the host.

```python
from semql import Catalog
from semql_prompt import (
    CannotPlan, GenerationContract, NeedContext, NeedUserClarification,
    PlannerResponse, PlanOutcome,
)


def answer(payload, catalog: Catalog, contract: GenerationContract, run_sql,
           *, viewer, context):
    response = contract.validate_payload(payload)
    assert isinstance(response, PlannerResponse)
    outcome = response.outcome
    if isinstance(outcome, NeedContext):
        return {"kind": outcome.kind, "requests": outcome.requests}
    if isinstance(outcome, NeedUserClarification):
        return {
            "kind": outcome.kind,
            "decision_id": outcome.decision_id,
            "prompt": outcome.prompt,
            "choices": outcome.choices,
        }
    if isinstance(outcome, CannotPlan):
        return {"kind": outcome.kind, "diagnostics": outcome.diagnostics}
    assert isinstance(outcome, PlanOutcome)
    compiled = [
        catalog.compile(step.query, viewer=viewer, context=context)
        for step in outcome.plan.steps
    ]
    return {
        "kind": "plan",
        "results": [
            {"label": step.label or "Result",
             "rows": run_sql(query.sql, query.params)}
            for step, query in zip(outcome.plan.steps, compiled, strict=True)
        ],
    }
```

Parsing does not authorize a query or make model-selected context requests
trusted. For a resumed conversation, use host-confirmed typed decisions and
`accept_continuation`, rather than executing a newly parsed plan directly.
The host owns correlation, retrieval, confirmation, budgets, and termination.
See the [migration guide](docs/migrations/planner-contracts-ranking.md) and
[`demos/planner_continuation_demo.py`](demos/planner_continuation_demo.py).

Router, Presenter, and Drilldown still use `RouterDecision`, `Presentation`,
and `DrilldownSuggestions` from `semql.plan`. `QueryPlan` remains the immutable
plan payload inside `PlanOutcome`; it is no longer the generator's default
outer response. `demos/pipeline_demo.py` illustrates the four-role flow without
requiring a model provider.

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
