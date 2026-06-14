"""Tests for ``semql_prompt.bedrock`` — Bedrock Converse tool-schema adaptation.

The contract being pinned: Bedrock's Converse API rejects a tool ``inputSchema``
whose top level is a ``$ref`` (it demands ``type: "object"`` at the root), but
accepts internal ``$ref`` / ``$defs`` — including recursive cycles. So
:func:`flatten_root_ref` must rewrite ONLY the root and must NOT strip nested
refs, ``$defs``, or ``prefixItems`` tuples. Verified against the live Converse
API (Nova / Claude / Llama / Mistral / Qwen / gpt-oss) on 2026-06; these tests
guard the invariant offline.
"""

from __future__ import annotations

import json

import pytest
from semql import AuthContext, Catalog, Cube, Dialect, Dimension, Measure, flatten_root_ref
from semql.spec import BoolExpr, SemanticQuery
from semql_prompt.bedrock import to_bedrock_converse_tools


def _cube(name: str = "orders", required_roles: list[str] | None = None) -> Cube:
    return Cube(
        name=name,
        dialect=Dialect.POSTGRES,
        table=f"public.{name}",
        alias=name[:2],
        description=f"The {name} cube.",
        measures=[Measure(name="revenue", sql=f"{{{name[:2]}}}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql=f"{{{name[:2]}}}.region", type="string")],
        required_roles=required_roles or [],
    )


# ---------------------------------------------------------------------------
# Object-rooted models pass through untouched.
# ---------------------------------------------------------------------------


def test_semantic_query_already_object_rooted_is_preserved() -> None:
    """SemanticQuery's schema is object-rooted already, so flattening is a
    no-op that must preserve internal refs, $defs, and prefixItems verbatim."""
    raw = SemanticQuery.model_json_schema()
    out = flatten_root_ref(raw)

    assert out["type"] == "object"
    assert "$ref" not in out
    # Internal machinery is retained, not inlined away.
    assert set(out["$defs"]) == set(raw["$defs"])
    assert out["properties"]["where"]["anyOf"][0]["$ref"] == "#/$defs/BoolExpr"
    # prefixItems tuples (TimeWindow.range) survive — Bedrock accepts them.
    assert "prefixItems" in out["$defs"]["TimeWindow"]["properties"]["range"]


def test_flatten_does_not_mutate_input() -> None:
    raw = SemanticQuery.model_json_schema()
    before = repr(raw)
    flatten_root_ref(raw)
    assert repr(raw) == before


# ---------------------------------------------------------------------------
# Recursive root ($ref at top level) is the case that gets rewritten.
# ---------------------------------------------------------------------------


def test_boolexpr_recursive_root_is_flattened() -> None:
    """BoolExpr is self-referential, so Pydantic emits a bare root $ref —
    exactly what Bedrock rejects. After flattening the root must be an object
    with BoolExpr's own fields, while $defs and the recursive internal $ref
    stay intact so children still resolve."""
    raw = BoolExpr.model_json_schema()
    assert raw.get("$ref") == "#/$defs/BoolExpr"  # precondition: Pydantic shape
    assert raw.get("type") is None

    out = flatten_root_ref(raw)

    assert out["type"] == "object"
    assert "$ref" not in out  # no longer a root ref
    assert set(out["properties"]) == {"op", "children"}
    # $defs retained, and the recursion still points back into it.
    assert "BoolExpr" in out["$defs"]
    assert "#/$defs/BoolExpr" in json.dumps(out["$defs"]["BoolExpr"])


def test_root_ref_siblings_are_preserved_and_win() -> None:
    """Keywords beside a root $ref (description/default) must survive and
    override the spliced body."""
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {
            "Node": {
                "type": "object",
                "description": "inner",
                "properties": {"x": {"type": "string"}},
            }
        },
        "description": "outer",
    }
    out = flatten_root_ref(schema)
    assert out["type"] == "object"
    assert out["properties"] == {"x": {"type": "string"}}
    assert out["description"] == "outer"  # sibling wins over the def's "inner"
    assert out["$defs"]["Node"]["description"] == "inner"  # $defs untouched


def test_flatten_is_idempotent() -> None:
    once = flatten_root_ref(BoolExpr.model_json_schema())
    twice = flatten_root_ref(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Non-object roots cannot be Converse tool inputs — fail loudly.
# ---------------------------------------------------------------------------

_NON_OBJECT_ROOTS: list[dict[str, object]] = [
    {"type": "array", "items": {"type": "string"}},
    {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    {"$ref": "#/$defs/N", "$defs": {"N": {"type": "array", "items": {"type": "string"}}}},
]


@pytest.mark.parametrize("schema", _NON_OBJECT_ROOTS)
def test_non_object_root_raises(schema: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="object-rooted"):
        flatten_root_ref(schema)


@pytest.mark.parametrize(
    "ref",
    ["http://example.com/schema", "#/$defs/Missing", "#/nope/Node"],
)
def test_unresolvable_or_remote_ref_raises(ref: str) -> None:
    schema: dict[str, object] = {
        "$ref": ref,
        "$defs": {"Node": {"type": "object", "properties": {}}},
    }
    with pytest.raises(ValueError):
        flatten_root_ref(schema)


# ---------------------------------------------------------------------------
# to_bedrock_converse_tools: Converse-shaped, object-rooted, role-gated.
# ---------------------------------------------------------------------------


def test_to_bedrock_converse_tools_shape_and_root() -> None:
    catalog = Catalog([_cube("orders")])
    tools = to_bedrock_converse_tools(catalog)

    assert len(tools) == 1
    spec = tools[0]["toolSpec"]
    assert spec["name"] == "query_orders"
    assert spec["description"]
    schema = spec["inputSchema"]["json"]
    assert schema["type"] == "object"  # Converse's hard requirement
    assert "$ref" not in schema


def test_to_bedrock_converse_tools_respects_role_gating() -> None:
    """Role-gated cubes are excluded unless the viewer holds the role — the
    visibility split is inherited from to_openai_tools, so just confirm it
    survives the reshape."""
    catalog = Catalog([_cube("orders"), _cube("admin_cube", required_roles=["admin"])])

    public = {t["toolSpec"]["name"] for t in to_bedrock_converse_tools(catalog)}
    assert public == {"query_orders"}

    admin_view = AuthContext(viewer_id="u", roles=["admin"])
    privileged = {
        t["toolSpec"]["name"] for t in to_bedrock_converse_tools(catalog, viewer=admin_view)
    }
    assert privileged == {"query_orders", "query_admin_cube"}
