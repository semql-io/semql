from __future__ import annotations

from semql import (
    Coverage,
    Diagnostic,
    QueryLocation,
    SemanticAnalysis,
    SemanticAssumption,
    SemanticBinding,
    SemanticDerivation,
    SemanticJoin,
    SemanticNode,
    SemanticOutput,
    SemanticResult,
    render_analysis,
)


def test_render_analysis_describes_outputs_hidden_operands_grain_and_native_strategy() -> None:
    analysis = SemanticAnalysis(
        nodes=(
            SemanticNode(
                node_id="n1",
                kind="derived",
                aggregate="ratio",
                operand_ids=("n2", "n3"),
                unit="currency/units",
                unit_state="declared",
                period_role="current",
                catalog_ref="private.catalog.ref",
            ),
            SemanticNode(node_id="n2", kind="aggregate", aggregate="sum", unit="currency"),
            SemanticNode(node_id="n3", kind="aggregate", aggregate="count", unit="units"),
            SemanticNode(node_id="n4", kind="dimension"),
        ),
        outputs=(SemanticOutput(output_id="o1", node_id="n1", sql_alias="rate"),),
        result=SemanticResult(grain_kind="grouped", grain_node_ids=("n4",)),
        derivations=(
            SemanticDerivation(
                operation="rollup",
                recipe="stored_measure_reaggregation",
                logical_node_ids=("n1",),
                physical_refs=("private.physical.source",),
            ),
        ),
    )

    rendered = render_analysis(analysis)

    assert (
        '"rate": derived measure; aggregation ratio; unit "currency/units" '
        "(declared); period current."
    ) in rendered
    assert "Hidden operands:" in rendered
    assert ('Operand 1: aggregate; aggregation sum; unit "currency" (not established).') in rendered
    assert 'Operand 2: aggregate; aggregation count; unit "units" (not established).' in rendered
    assert "Result grain:" in rendered and "Kind: grouped." in rendered
    assert "Physical strategy: native query." in rendered
    assert "re-aggregates stored rollup measures" in rendered
    assert "private.catalog.ref" not in rendered
    assert "private.physical.source" not in rendered


def test_render_analysis_distinguishes_declared_joins_and_federated_strategy() -> None:
    analysis = SemanticAnalysis(
        nodes=(SemanticNode(node_id="n1", kind="aggregate", aggregate="sum", unit="currency"),),
        outputs=(SemanticOutput(output_id="o1", node_id="n1", sql_alias="amount"),),
        joins=(
            SemanticJoin(
                source_ref="private.left",
                target_ref="private.right",
                kind="inner",
                relationship="many_to_one",
            ),
            SemanticJoin(
                source_ref="private.north",
                target_ref="private.south",
                kind="left",
                relationship="one_to_many",
            ),
        ),
        assumptions=(
            SemanticAssumption(
                reference="join:private.north->private.south",
                kind="relationship",
                declared_value="one_to_many",
                verification="verified",
            ),
            SemanticAssumption(
                reference="secret.join.reference",
                kind="relationship",
                declared_value="many_to_one",
                verification="catalog_trust",
            ),
            SemanticAssumption(
                reference="join:private.left->private.right",
                kind="relationship",
                declared_value="many_to_one",
                verification="catalog_trust",
            ),
            SemanticAssumption(
                reference="private.fragment.key",
                kind="unique_merge_key",
                declared_value="unique",
                verification="verified",
            ),
        ),
        derivations=(
            SemanticDerivation(
                operation="sum",
                recipe="sum_partial_aggregates",
                physical_refs=("fragment:0.secret_column",),
            ),
        ),
        binding_dependencies=(
            SemanticBinding(slot="secret", kind="filter_value", reference="private.filter.value"),
        ),
        coverage=Coverage(status="partial", not_established=("unknown_internal_marker",)),
    )

    rendered = render_analysis(analysis)

    assert "Physical strategy: federated merge." in rendered
    assert "cardinality many-to-one (declared; runtime evidence unavailable)." in rendered
    assert "cardinality one-to-many (declared; runtime evidence unavailable)." in rendered
    assert "runtime-verified" not in rendered
    assert "uniqueness declared; runtime evidence unavailable" in rendered
    assert "Runtime evidence for one or more declared assumptions is unavailable." in rendered
    for secret in (
        "private.left",
        "private.right",
        "private.north",
        "private.south",
        "secret.join.reference",
        "many_to_one",
        "private.fragment.key",
        "fragment:0.secret_column",
        "private.filter.value",
    ):
        assert secret not in rendered
    assert "Additional semantic coverage is unavailable." in rendered


def test_render_analysis_escapes_untrusted_aliases_and_reports_broken_graph() -> None:
    analysis = SemanticAnalysis(
        nodes=(
            SemanticNode(node_id="duplicate", kind="dimension"),
            SemanticNode(node_id="duplicate", kind="dimension"),
        ),
        outputs=(
            SemanticOutput(output_id="o1", node_id="duplicate", sql_alias="region\nINJECT\x1b[31m"),
            SemanticOutput(output_id="o2", node_id="missing", sql_alias="missing output"),
        ),
        result=SemanticResult(grain_node_ids=("missing_grain",)),
        coverage=Coverage(status="unavailable", not_established=("analysis_unavailable",)),
    )

    rendered = render_analysis(analysis)

    assert '"region\\nINJECT\\u001b[31m"' in rendered
    assert "INJECT\n" not in rendered
    assert "semantics unavailable" in rendered
    assert "Duplicate semantic node identifiers" in rendered
    assert "Dangling semantic graph references" in rendered
    assert "Semantic analysis is unavailable." in rendered
    assert "Physical strategy: not established." in rendered
    assert "Physical execution strategy is not established." in rendered


def test_diagnostic_public_payload_includes_bounded_repair_and_optional_location() -> None:
    location = QueryLocation(section="filters", index=2)
    diagnostic = Diagnostic(
        code="unknown_field",
        reason="unknown_field",
        location=location,
    )

    payload = diagnostic.to_public_payload()

    assert payload["location"] == {"section": "filters", "index": 2}
    assert payload["category"] == "reference"
    assert payload["repair_hint"] == "Check the referenced field and its query section."
    assert "value" not in payload
    no_location = Diagnostic(code="unknown_field", reason="unknown_field")
    assert "location" not in no_location.to_public_payload()
