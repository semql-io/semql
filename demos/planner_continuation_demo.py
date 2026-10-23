"""Two host-owned continuations with real query execution; no model or provider calls."""

from __future__ import annotations

import duckdb
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Lookup,
    Measure,
    QueryPlan,
    QueryStep,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    resolve_outcome,
)
from semql_prompt.continuation import (
    AllowedDecision,
    ContinuationState,
    DecisionChoice,
    FilterValueTarget,
    MeasureChoiceTarget,
    accept_continuation,
    render_continuation,
)
from semql_prompt.outcomes import (
    ClarificationChoice,
    ContextCandidate,
    ContextRequest,
    ContextResult,
    NeedContext,
    NeedUserClarification,
    PlannerResponse,
    PlanOutcome,
)


def main() -> None:
    viewer = AuthContext(viewer_id="reader")
    catalog = Catalog(
        [
            Cube(
                name="sales",
                table="sales",
                alias="s",
                dialect=Dialect.DUCKDB,
                security_sql="{s}.owner = {ctx.viewer_id}",
                dimensions=[Dimension(name="category", sql="{s}.category", type="string")],
                time_dimensions=[TimeDimension(name="day", sql="{s}.day", type="date")],
                measures=[
                    Measure(name="gross", sql="{s}.gross", agg="sum"),
                    Measure(name="net", sql="{s}.net", agg="sum"),
                ],
            )
        ],
        lookups=[
            Lookup(
                dimension="sales.category",
                values=("retail_a", "retail_b"),
                labels={"retail_a": "Retail", "retail_b": "Retail"},
            )
        ],
    )
    base = SemanticQuery(
        measures=["sales.gross"],
        time_dimension=TimeWindow(dimension="sales.day", range=("2026-10-01", "2026-11-01")),
        filters=[Filter(dimension="sales.category", op="eq", values=["retail_b"])],
    )
    baseline = QueryPlan(steps=[QueryStep(query=base, intent="headline")])
    request = ContextRequest(
        request_id="metric-lookup", decision_id="metric", kind="definition", target="sales.net"
    )
    # The host retrieves an authorized definition. This fixture is not a model response.
    definition = ContextResult(
        request_id=request.request_id,
        decision_id=request.decision_id,
        status="resolved",
        complete=True,
        candidates=(
            ContextCandidate(
                id="net",
                label="Revenue after refunds",
                value="sales.net",
                match_kind="definition",
                authoritative=True,
            ),
        ),
    )
    need_context = PlannerResponse(outcome=NeedContext(requests=(request,), results=(definition,)))
    print("Definition request and host-supplied result:", need_context.model_dump_json())
    metric_state = ContinuationState(
        request_id="metric-run",
        revision=1,
        baseline=baseline,
        authorized_cubes=("sales",),
        visibility_revision="catalog-1",
        context_requests=(request,),
        context_results=(definition,),
        decisions=(
            AllowedDecision(
                decision_id="metric",
                target=MeasureChoiceTarget(step_index=0, measure_index=0),
                choices=(DecisionChoice(id="net", value=definition.candidates[0].value),),
                confirmed_choice_id="net",
            ),
        ),
    )
    metric_query = base.model_copy(update={"measures": ["sales.net"]})
    metric_response = PlannerResponse(
        request_id=metric_state.request_id,
        revision=metric_state.revision,
        outcome=PlanOutcome(
            plan=QueryPlan(steps=[QueryStep(query=metric_query, intent="headline")])
        ),
    )
    accepted_metric = accept_continuation(
        metric_state,
        metric_response,
        catalog=catalog,
        viewer=viewer,
        current_authorized_cubes=("sales",),
        current_visibility_revision="catalog-1",
    )
    assert accepted_metric.accepted

    lookup = resolve_outcome(catalog, "sales.category", "Retail")
    assert lookup.status == "ambiguous" and lookup.total_candidates == 2
    clarification = NeedUserClarification(
        decision_id="category",
        prompt="Which Retail category?",
        choices=tuple(
            ClarificationChoice(id=value, label=f"Retail ({value})", value=value)
            for value in lookup.values
        ),
    )
    print("Ambiguous labels remain distinct:", clarification.model_dump_json())
    # This is an explicit caller-provided user answer, not an autonomous model choice.
    user_answer = "retail_b"
    unresolved = base.model_copy(
        update={"filters": [Filter(dimension="sales.category", op="eq", values=["Retail"])]}
    )
    category_state = ContinuationState(
        request_id="category-run",
        revision=1,
        baseline=QueryPlan(steps=[QueryStep(query=unresolved, intent="headline")]),
        authorized_cubes=("sales",),
        visibility_revision="catalog-1",
        decisions=(
            AllowedDecision(
                decision_id="category",
                target=FilterValueTarget(step_index=0, filter_index=0, value_index=0),
                choices=(DecisionChoice(id=user_answer, value=user_answer),),
                confirmed_choice_id=user_answer,
            ),
        ),
    )
    category_response = PlannerResponse(
        request_id=category_state.request_id,
        revision=category_state.revision,
        outcome=PlanOutcome(plan=baseline),
    )
    accepted_category = accept_continuation(
        category_state,
        category_response,
        catalog=catalog,
        viewer=viewer,
        current_authorized_cubes=("sales",),
        current_visibility_revision="catalog-1",
    )
    assert accepted_category.accepted
    required_context = render_continuation(
        category_state,
        catalog=catalog,
        viewer=viewer,
        current_authorized_cubes=("sales",),
        current_visibility_revision="catalog-1",
    )
    assert "2026-10-01" in required_context
    assert base.time_dimension is not None
    drifted = base.model_copy(
        update={
            "time_dimension": base.time_dimension.model_copy(
                update={"range": ("2026-09-01", "2026-11-01")}
            )
        }
    )
    drift_response = category_response.model_copy(
        update={
            "outcome": PlanOutcome(
                plan=QueryPlan(steps=[QueryStep(query=drifted, intent="headline")])
            )
        }
    )
    rejection = accept_continuation(
        category_state,
        drift_response,
        catalog=catalog,
        viewer=viewer,
        current_authorized_cubes=("sales",),
        current_visibility_revision="catalog-1",
    )
    assert not rejection.accepted and rejection.diagnostics[0].code == "continuation_drift"
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE sales(category VARCHAR, day DATE, gross INT, net INT, owner VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO sales VALUES (?, ?, ?, ?, ?)",
            [
                ("retail_a", "2026-10-10", 100, 90, "reader"),
                ("retail_b", "2026-10-10", 20, 15, "reader"),
                ("retail_b", "2026-09-10", 300, 250, "reader"),
                ("retail_b", "2026-10-10", 400, 350, "someone-else"),
            ],
        )
        metric = accepted_metric.compiled[0]
        category = accepted_category.compiled[0]
        assert connection.execute(metric.sql, metric.params).fetchall() == [(15,)]
        assert connection.execute(category.sql, category.params).fetchall() == [(20,)]
    print("Definition continuation executed: net=15; category continuation executed: gross=20.")
    print("Changed locked date range rejected; current viewer authorization reapplied.")


if __name__ == "__main__":
    main()
