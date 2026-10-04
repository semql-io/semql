"""Exercise caller-owned query and prompt repair without a model client.

The structured outputs below are explicitly caller-provided samples. This
example demonstrates prompt construction and host-side validation only.
"""

from __future__ import annotations

import html
import json

from semql import (
    AuthContext,
    Catalog,
    Cube,
    Diagnostic,
    DiagnosticCategory,
    Dialect,
    Dimension,
    Measure,
    QueryPlan,
    SemanticQuery,
    UnknownIdentifierError,
)
from semql_prompt import AdaptivePromptPolicy, AdaptivePromptRequest, build_adaptive_prompt
from semql_prompt.repair import (
    PromptAugmentation,
    PromptExample,
    RepairGuidance,
    apply_prompt_augmentation,
    build_prompt_repair_prompt,
    build_query_repair_prompt,
)


def main() -> None:
    catalog = Catalog(
        [
            Cube(
                name="orders",
                dialect=Dialect.DUCKDB,
                table="orders",
                alias="o",
                description="Customer orders with revenue and region.",
                measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
                dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            )
        ]
    )
    viewer = AuthContext(viewer_id="repair-demo-user")
    policy = AdaptivePromptPolicy(top_k=3, max_tokens=4_000)
    question = "Show total order revenue by region."
    original = build_adaptive_prompt(
        catalog,
        AdaptivePromptRequest(question=question, scope_to=("orders",)),
        viewer=viewer,
        policy=policy,
    )
    assert original.fits
    print(f"Original adaptive prompt: fits={original.fits}; cubes={original.selected_cubes}")

    failed_query = SemanticQuery(measures=["orders.revnue"], dimensions=["orders.region"])
    try:
        catalog.compile(failed_query, viewer=viewer)
    except UnknownIdentifierError as error:
        # Construct a new, allowlisted diagnostic. Never forward exception text,
        # the unknown spelling, or arbitrary exception payload to the prompt.
        diagnostic = Diagnostic(
            code="UnknownIdentifierError",
            reason="unknown_field",
            references=("orders.revenue",) if error.hint == "revenue" else (),
            category=DiagnosticCategory.REFERENCE,
        )
    else:
        raise AssertionError("the deliberately misspelled query unexpectedly compiled")

    request = AdaptivePromptRequest(
        question=question,
        scope_to=("orders",),
        previous_output=failed_query.model_dump_json(),
        diagnostics=(diagnostic,),
    )
    failed_output_json = failed_query.model_dump_json()

    query_repair = build_query_repair_prompt(
        catalog,
        request,
        original_prompt=original.text,
        viewer=viewer,
        policy=policy,
    )
    prompt_repair = build_prompt_repair_prompt(
        catalog,
        request,
        original_prompt=original.text,
        viewer=viewer,
        policy=policy,
    )
    for result in (query_repair, prompt_repair):
        # Decode only for this host-side evidence check, never before model use.
        assert original.text in html.unescape(result.text)
        assert failed_output_json in result.text
        assert "unknown field" in result.text
    assert query_repair.target == "query"
    assert prompt_repair.target == "prompt"
    print(
        f"Query repair: output_model={query_repair.output_model.__name__}; fits={query_repair.fits}"
    )
    print(
        f"Prompt repair: output_model={prompt_repair.output_model.__name__}; "
        f"fits={prompt_repair.fits}"
    )
    print("Both repair prompts retain the original prompt, failed query JSON, and safe diagnostic.")

    # Host-owned parsing of sample values; these are not model responses.
    query_plan_json = json.dumps(
        {
            "steps": [
                {
                    "query": {
                        "measures": ["orders.revenue"],
                        "dimensions": ["orders.region"],
                    },
                    "intent": "breakdown",
                    "label": "Revenue by region",
                }
            ],
            "reasoning": None,
        }
    )
    query_plan = QueryPlan.model_validate_json(query_plan_json)
    print("Caller-provided sample QueryPlan output parsed by the host (not a model response).")

    augmentation_json = json.dumps(
        {
            "guidance": ["catalog_references", "filter_types"],
            "examples": [
                {
                    "question": question,
                    "query": {
                        "measures": ["orders.revenue"],
                        "dimensions": ["orders.region"],
                    },
                }
            ],
        }
    )
    augmentation = PromptAugmentation.model_validate_json(augmentation_json)
    print(
        "Caller-provided sample PromptAugmentation output parsed by the host "
        "(not a model response)."
    )

    accepted = apply_prompt_augmentation(
        catalog,
        request,
        augmentation,
        viewer=viewer,
        policy=policy,
    )
    assert accepted.fits
    assert all(cube in original.selected_cubes for cube in accepted.selected_cubes)
    print(f"Accepted bounded augmentation: fits={accepted.fits}; cubes={accepted.selected_cubes}")

    compiled = catalog.compile(query_plan.steps[0].query, viewer=viewer)
    print("Corrected caller-provided QueryPlan compiles through Catalog.compile:")
    print(compiled.sql)

    malicious = PromptAugmentation(
        guidance=(RepairGuidance.CATALOG_REFERENCES,),
        examples=(
            PromptExample(
                question="Show restricted payroll data.",
                query=SemanticQuery(measures=["payroll.salary"]),
            ),
        ),
    )
    try:
        apply_prompt_augmentation(
            catalog,
            request,
            malicious,
            viewer=viewer,
            policy=policy,
        )
    except ValueError:
        print("Rejected unavailable/out-of-selection augmentation with generic ValueError.")
    else:
        raise AssertionError("unavailable/out-of-selection augmentation was unexpectedly accepted")


if __name__ == "__main__":
    main()
