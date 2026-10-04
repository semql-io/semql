"""Exercise deterministic adaptive prompts and caller-owned repair feedback."""

from __future__ import annotations

from semql import (
    Catalog,
    Cube,
    Diagnostic,
    DiagnosticCategory,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    UnknownIdentifierError,
)
from semql_prompt import (
    AdaptivePromptPolicy,
    AdaptivePromptRequest,
    build_adaptive_prompt,
)


def main() -> None:
    orders = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        description="Recognized sales revenue, organized by order region and month.",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    catalog = Catalog([orders])
    policy = AdaptivePromptPolicy(top_k=3, max_tokens=4_000)

    initial = build_adaptive_prompt(
        catalog,
        AdaptivePromptRequest(
            question="Show the trend.",
            conversation=("Show recognized revenue by month and region.",),
            retrieved_snippets=("Finance defines recognized revenue from paid orders.",),
        ),
    )
    assert initial.fits
    assert "Finance defines recognized revenue" in initial.text
    assert "<untrusted-data>" in initial.text
    print(f"Selection: {initial.selection_reason} -> {', '.join(initial.selected_cubes)}")
    print("Conversation and reference context are included as fenced data.")

    invalid = SemanticQuery(measures=["orders.revnue"])
    try:
        catalog.compile(invalid)
    except UnknownIdentifierError as error:
        reference = f"orders.{error.hint}" if error.hint == "revenue" else None
        diagnostic = Diagnostic(
            code=error.code,
            reason="unknown_field",
            references=(reference,) if reference is not None else (),
            category=DiagnosticCategory.REFERENCE,
        )
    else:
        raise AssertionError("the deliberately misspelled measure unexpectedly compiled")

    repair_prompt = build_adaptive_prompt(
        catalog,
        AdaptivePromptRequest(
            question="Show the trend.",
            conversation=("Show recognized revenue by month and region.",),
            previous_output=invalid.model_dump_json(),
            diagnostics=(diagnostic,),
        ),
        policy=policy,
    )
    assert "unknown field" in repair_prompt.text
    assert "orders.revenue" in repair_prompt.text

    corrected = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    compiled = catalog.compile(corrected)
    print("Repair feedback: safe unknown-field guidance with an authorized field reference.")
    print("Corrected caller query compiles:")
    print(compiled.sql)


if __name__ == "__main__":
    main()
