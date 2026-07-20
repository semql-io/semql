"""Unit tests for ``SemanticQuery.llm_json_schema`` / ``from_llm_payload``.

Some tool-use decoders are pickier than the Bedrock Converse validation layer
``tool_json_schema()`` targets — Bedrock Nova has been observed to 424 on
``prefixItems`` (fixed-length tuple schemas) and on any recursive ``$ref``
cycle in ``$defs``, not just a bare root ``$ref``. ``llm_json_schema()`` is
the stricter projection for that decoder; these tests pin its two guarantees
(no ``prefixItems``, no ``$defs`` cycle) and confirm the existing
``model_json_schema()`` / ``tool_json_schema()`` projections are untouched.
"""

from __future__ import annotations

import json
from typing import Any, cast

from semql.spec import SemanticQuery, TimeWindow


def _as_list(node: Any) -> list[Any]:  # noqa: ANN401 — arbitrary JSON-schema subtree
    """Typed identity — see ``semql._schema._as_list`` for why a plain
    ``cast("list[Any]", node)`` at the call site doesn't satisfy both
    pyright (strict) and mypy simultaneously here."""
    return cast("list[Any]", node)


def _def_ref_edges(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Build the ``$defs`` name -> referenced ``$defs`` names graph."""
    defs = cast("dict[str, Any]", schema.get("$defs", {}))
    edges: dict[str, set[str]] = {name: set() for name in defs}

    def collect(node: Any, out: set[str]) -> None:  # noqa: ANN401 — arbitrary JSON-schema subtree
        if isinstance(node, dict):
            obj = cast("dict[str, Any]", node)
            ref = obj.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                out.add(ref[len("#/$defs/") :])
            for value in obj.values():
                collect(value, out)
        elif isinstance(node, list):
            for value in _as_list(node):
                collect(value, out)

    for name, body in defs.items():
        collect(body, edges[name])
    return edges


def _has_cycle(edges: dict[str, set[str]]) -> bool:
    """True if any ``$defs`` name can reach itself along ``$ref`` edges."""

    def reaches(node: str, target: str, seen: frozenset[str]) -> bool:
        if node == target and seen:
            return True
        if node in seen:
            return False
        next_seen = seen | {node}
        return any(reaches(nxt, target, next_seen) for nxt in edges.get(node, ()))

    return any(
        any(reaches(nxt, name, frozenset()) for nxt in succs) for name, succs in edges.items()
    )


def test_llm_json_schema_has_no_prefix_items() -> None:
    """``TimeWindow.range`` / ``CompareWindow.range`` / ``order`` are tuples in
    ``model_json_schema()`` (rendered as ``prefixItems``); the LLM-safe
    projection must not contain that keyword anywhere."""
    schema = SemanticQuery.llm_json_schema()
    assert "prefixItems" not in json.dumps(schema)


def test_llm_json_schema_has_no_defs_cycle() -> None:
    """``BoolExpr`` self-references and ``SemanticQuery`` <-> ``SemiJoin``
    close a cycle through ``semi_joins[].source``; both must be gone."""
    schema = SemanticQuery.llm_json_schema()
    edges = _def_ref_edges(schema)
    assert not _has_cycle(edges), f"cyclic $defs graph: {edges}"


def test_llm_json_schema_drops_where_and_semi_joins() -> None:
    """``where`` (BoolExpr tree) and ``semi_joins`` (inner SemanticQuery) are
    the only paths into the recursive defs — both properties should be
    absent rather than left dangling."""
    schema = SemanticQuery.llm_json_schema()
    assert "where" not in schema["properties"]
    assert "semi_joins" not in schema["properties"]
    assert "where" not in schema.get("required", [])
    assert "semi_joins" not in schema.get("required", [])


def test_llm_json_schema_keeps_other_shape() -> None:
    """Non-recursive, non-tuple fields survive the projection unchanged in
    spirit — still present, still describing measures/dimensions/etc."""
    schema = SemanticQuery.llm_json_schema()
    for name in ("measures", "dimensions", "time_dimension", "filters", "having", "order", "limit"):
        assert name in schema["properties"], name


def test_model_json_schema_and_tool_json_schema_are_unchanged() -> None:
    """``llm_json_schema`` is purely additive: the existing projections must
    still contain the constructs it strips out."""
    raw = json.dumps(SemanticQuery.model_json_schema())
    tool = json.dumps(SemanticQuery.tool_json_schema())
    assert "prefixItems" in raw
    assert "prefixItems" in tool
    assert "BoolExpr" in raw
    assert "BoolExpr" in tool


def test_from_llm_payload_round_trips_to_canonical_query() -> None:
    """A payload shaped per ``llm_json_schema`` (plain-list ``range`` /
    ``order``, no ``where``) must coerce to the same ``SemanticQuery`` as
    hand-building it with tuples."""
    payload = {
        "measures": ["orders.revenue"],
        "dimensions": ["orders.region"],
        "time_dimension": {
            "dimension": "orders.created_at",
            "range": ["2026-01-01", "2026-02-01"],
        },
        "order": [["orders.region", "desc"]],
        "limit": 10,
    }
    got = SemanticQuery.from_llm_payload(payload)
    want = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
        order=[("orders.region", "desc")],
        limit=10,
    )
    assert got == want
