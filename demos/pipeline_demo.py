#!/usr/bin/env -S uv run
"""End-to-end walk through the four prompt-pipeline roles.

Runs without any LLM. The LLM outputs (RouterDecision, QueryPlan,
Presentation, DrilldownSuggestions) are stubbed inline so you can see
the full data flow — fragment in, structured value out, compiled SQL,
narrative back — without needing an API key. Drop in pydantic-ai
Agents to make it real.

Run with: ``uv run demos/pipeline_demo.py``
"""

from __future__ import annotations

import json
import textwrap

from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    DrilldownSuggestion,
    DrilldownSuggestions,
    Filter,
    Measure,
    Presentation,
    QueryPlan,
    QueryStep,
    RouterDecision,
    ScopePredicate,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    View,
)
from semql.introspect import iter_cubes
from semql_prompt import (
    build_drilldown_prompt_fragment,
    build_presenter_prompt_fragment,
    build_query_generator_prompt_fragment,
    build_router_prompt_fragment,
)

# ---------------------------------------------------------------------------
# A small realistic catalog
# ---------------------------------------------------------------------------

ORDERS = Cube(
    name="orders",
    dialect=Dialect.POSTGRES,
    table="orders",
    alias="o",
    primary_key="id",
    description="One row per customer order.",
    display_name="Orders",
    measures=[
        Measure(
            name="revenue",
            sql="{o}.amount",
            agg="sum",
            unit="currency",
            description="Total order revenue.",
        ),
        Measure(name="count", sql="*", agg="count", unit="count"),
    ],
    dimensions=[
        Dimension(name="id", sql="{o}.id", type="number"),
        Dimension(name="region", sql="{o}.region", type="string"),
        Dimension(name="status", sql="{o}.status", type="string"),
        Dimension(name="rep_id", sql="{o}.rep_id", type="string", foreign_key="reps"),
    ],
    time_dimensions=[
        TimeDimension(
            name="created_at",
            sql="{o}.created_at",
            granularities=("day", "week", "month"),
        ),
    ],
    drill_paths=[["region", "status"]],
    scope="my_team",
)

REPS = Cube(
    name="reps",
    dialect=Dialect.POSTGRES,
    table="reps",
    alias="r",
    primary_key="id",
    description="Sales reps and their team affiliation.",
    display_name="Sales Reps",
    measures=[Measure(name="count", sql="*", agg="count")],
    dimensions=[
        Dimension(name="id", sql="{r}.id", type="string"),
        Dimension(name="name", sql="{r}.name", type="string"),
        Dimension(name="team", sql="{r}.team", type="string"),
    ],
)

FINANCE_LEDGER = Cube(
    name="finance_ledger",
    dialect=Dialect.POSTGRES,
    table="finance_ledger",
    alias="fl",
    description="Confidential GL entries; restricted to the finance role.",
    required_roles=["finance"],
    measures=[Measure(name="amount", sql="{fl}.amount", agg="sum", unit="currency")],
    dimensions=[Dimension(name="account", sql="{fl}.account", type="string")],
)

REVENUE_VIEW = View(
    name="revenue_overview",
    display_name="Revenue Overview",
    description="Curated revenue facade for line-of-business questions.",
    fields={
        "revenue": "orders.revenue",
        "region": "orders.region",
        "created_at": "orders.created_at",
    },
)


def my_team_scope(_cube: Cube, viewer: AuthContext) -> ScopePredicate | None:
    """Sales reps only see orders their team owns. Admins see all."""
    if "admin" in viewer.roles:
        return None
    return ScopePredicate(
        sql=("{o}.rep_id IN (SELECT id FROM reps WHERE team = {ctx.viewer_team})"),
        ctx_keys=["ctx.viewer_team"],
    )


CATALOG = Catalog(
    cubes=[ORDERS, REPS, FINANCE_LEDGER],
    views=[REVENUE_VIEW],
    scope_fns={"my_team": my_team_scope},
)


# ---------------------------------------------------------------------------
# The viewer
# ---------------------------------------------------------------------------

VIEWER = AuthContext(
    viewer_id="alice@example.com",
    roles=["sales"],  # no "finance" — finance_ledger should be hidden
    metadata={},
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def banner(label: str) -> None:
    print(f"\n{'═' * 72}\n  {label}\n{'═' * 72}")


def section(label: str) -> None:
    print(f"\n── {label} " + "─" * (66 - len(label)))


def show_dict(d: dict[str, object]) -> None:
    print(json.dumps(d, indent=2, default=str))


# ---------------------------------------------------------------------------
# 1. ROUTER
# ---------------------------------------------------------------------------

QUESTION = "How did revenue trend by month in my region last quarter?"


def stage_router() -> RouterDecision:
    banner("STAGE 1 / ROUTER")
    print(f'Question: "{QUESTION}"\n')

    section("Router prompt fragment (sent to LLM)")
    fragment = build_router_prompt_fragment(CATALOG.as_dict(), viewer=VIEWER)
    # Trim for display but show the structural sections.
    head, _, tail = fragment.partition("## Output")
    print(textwrap.indent(head.rstrip(), "  "))
    print("  ... [output schema follows]")

    section("Catalog surface visible to this viewer")
    for c in iter_cubes(CATALOG, viewer=VIEWER):
        print(f"  - {c.name} ({c.dialect.value})")
    print("\n  (finance_ledger is hidden — viewer lacks 'finance' role)")

    section("Stubbed LLM output (what pydantic-ai would parse)")
    decision = RouterDecision(
        route_to="semantic",
        cubes=["orders"],
        views=["revenue_overview"],
        reasoning="Revenue-by-time question maps cleanly to the orders cube.",
    )
    print(decision.model_dump_json(indent=2))
    return decision


# ---------------------------------------------------------------------------
# 2. QUERY GENERATOR
# ---------------------------------------------------------------------------


def stage_generator(decision: RouterDecision) -> QueryPlan:
    banner("STAGE 2 / QUERY GENERATOR")
    section("Generator prompt fragment (scoped to router's picks)")
    fragment = build_query_generator_prompt_fragment(
        CATALOG.as_dict(),
        scope_to=decision.cubes + decision.views,
        views={"revenue_overview": REVENUE_VIEW},
        viewer=VIEWER,
    )
    head, _, tail = fragment.partition("## Output")
    cat_block, _, _ = head.partition("## When to fall back")
    print(textwrap.indent(cat_block.rstrip(), "  "))
    print("  ... [raw-fallback rule + output schema follow]")

    section("Stubbed LLM output (QueryPlan)")
    plan = QueryPlan(
        steps=[
            QueryStep(
                query=SemanticQuery(
                    measures=["orders.revenue"],
                    time_dimension=TimeWindow(
                        dimension="orders.created_at",
                        granularity="month",
                        range=("2026-01-01", "2026-04-01"),
                        fill_nulls_with=0,
                    ),
                ),
                intent="headline",
                label="Monthly revenue (Q1 2026)",
            ),
            QueryStep(
                query=SemanticQuery(
                    measures=["orders.revenue"],
                    time_dimension=TimeWindow(
                        dimension="orders.created_at",
                        granularity="month",
                        range=("2025-10-01", "2026-01-01"),
                        fill_nulls_with=0,
                    ),
                ),
                intent="compare",
                label="Monthly revenue (Q4 2025) — prior period",
            ),
        ],
        reasoning="Headline + prior-quarter compare for trend context.",
    )
    print(plan.model_dump_json(indent=2))
    return plan


# ---------------------------------------------------------------------------
# 3. COMPILE EACH STEP
# ---------------------------------------------------------------------------


def stage_compile(plan: QueryPlan) -> list[tuple[QueryStep, str, dict[str, object]]]:
    banner("STAGE 3 / COMPILE (real SQL emission, viewer + scope applied)")
    out: list[tuple[QueryStep, str, dict[str, object]]] = []
    for i, step in enumerate(plan.steps, start=1):
        section(f"Step {i}: {step.label}  [intent={step.intent}]")
        compiled = CATALOG.compile(
            step.query,
            viewer=VIEWER,
            context={"ctx.viewer_team": "EMEA-East"},
        )
        print(f"  SQL: {compiled.sql}\n")
        print(f"  Params: {compiled.params}")
        print(f"  Output columns: {compiled.columns}")
        out.append((step, compiled.sql, dict(compiled.params)))
    return out


# ---------------------------------------------------------------------------
# 4. PRESENTER (with fake result rows)
# ---------------------------------------------------------------------------


def stage_presenter(
    plan: QueryPlan, results: list[tuple[QueryStep, str, dict[str, object]]]
) -> Presentation:
    banner("STAGE 4 / PRESENTER")
    # Simulate the executed-row summaries the caller would feed in.
    section("Caller-supplied result summary (would come from DB rows)")
    summary = textwrap.dedent("""\
        Q1 2026 monthly revenue:
          2026-01: $182_400
          2026-02: $211_500
          2026-03: $239_900   (+18% MoM)
        Q4 2025 monthly revenue (compare):
          2025-10: $148_900
          2025-11: $163_200
          2025-12: $191_400
        QoQ growth: +29%, primarily from March.
    """).rstrip()
    print(textwrap.indent(summary, "  "))

    section("Presenter prompt fragment")
    fragment = build_presenter_prompt_fragment(
        query_labels=[step.label or "" for step, _, _ in results],
        result_summary=summary,
    )
    head, _, _ = fragment.partition("## Output")
    print(textwrap.indent(head.rstrip(), "  "))
    print("  ... [output schema follows]")

    section("Stubbed LLM output (Presentation)")
    presentation = Presentation(
        summary=(
            "Revenue grew 29% quarter over quarter, reaching $633.8K in Q1 2026 "
            "versus $503.5K in Q4 2025. March was the strongest month at $239.9K, "
            "up 18% from February."
        ),
        highlights=[
            "March hit a new monthly high of $239.9K (+18% MoM).",
            "Every month of Q1 outperformed the corresponding Q4 month.",
        ],
        caveats=[
            "Numbers are scoped to your team (EMEA-East).",
            "March's spike was concentrated in three large deals; check sustainability.",
        ],
    )
    print(presentation.model_dump_json(indent=2))
    return presentation


# ---------------------------------------------------------------------------
# 5. DRILLDOWN (on the March outlier)
# ---------------------------------------------------------------------------


def stage_drilldown() -> DrilldownSuggestions:
    banner("STAGE 5 / DRILLDOWN (anchored to the March outlier)")
    focused = {"month": "2026-03", "revenue": "$239_900"}

    section("Drilldown prompt fragment")
    fragment = build_drilldown_prompt_fragment(ORDERS, focused_row=focused)
    head, _, _ = fragment.partition("## Output")
    print(textwrap.indent(head.rstrip(), "  "))
    print("  ... [output schema follows]")

    section("Stubbed LLM output (DrilldownSuggestions)")
    suggestions = DrilldownSuggestions(
        focus="March 2026 revenue spike ($239.9K)",
        suggestions=[
            DrilldownSuggestion(
                label="Break March down by region",
                query=SemanticQuery(
                    measures=["orders.revenue"],
                    dimensions=["orders.region"],
                    time_dimension=TimeWindow(
                        dimension="orders.created_at",
                        granularity="month",
                        range=("2026-03-01", "2026-04-01"),
                    ),
                ),
                rationale="Catalog declares region→status drill_path; region first.",
            ),
            DrilldownSuggestion(
                label="Top 10 orders in March",
                query=SemanticQuery(
                    dimensions=["orders.id", "orders.region", "orders.status"],
                    filters=[
                        Filter(
                            dimension="orders.created_at",
                            op="gte",
                            values=["2026-03-01"],
                        ),
                        Filter(
                            dimension="orders.created_at",
                            op="lt",
                            values=["2026-04-01"],
                        ),
                    ],
                    order=[("orders.id", "desc")],
                    limit=10,
                    ungrouped=True,
                ),
                rationale="Identify the three large deals the Presenter caveat flagged.",
            ),
            DrilldownSuggestion(
                label="Daily revenue trend in March",
                query=SemanticQuery(
                    measures=["orders.revenue"],
                    time_dimension=TimeWindow(
                        dimension="orders.created_at",
                        granularity="day",
                        range=("2026-03-01", "2026-04-01"),
                        fill_nulls_with=0,
                    ),
                ),
                rationale="Spot whether the spike was sustained or concentrated.",
            ),
        ],
    )
    print(suggestions.model_dump_json(indent=2))
    return suggestions


# ---------------------------------------------------------------------------
# Drive it
# ---------------------------------------------------------------------------


def main() -> None:
    decision = stage_router()
    plan = stage_generator(decision)
    results = stage_compile(plan)
    stage_presenter(plan, results)
    stage_drilldown()
    banner("DONE")


if __name__ == "__main__":
    main()
