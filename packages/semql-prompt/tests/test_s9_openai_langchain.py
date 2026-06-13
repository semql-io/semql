"""S9 — OpenAI-function-format adapter + LangChain integration.

to_openai_function(cube, catalog) -> dict:
  {
    "type": "function",
    "function": {
      "name": f"query_{cube.name}",
      "description": render_tool_description(cube),
      "parameters": SemanticQuery.model_json_schema(),
    }
  }

to_openai_tools(catalog, *, viewer, policy) -> list[dict]
to_langchain_tools(catalog, *, viewer, policy) -> list[StructuredTool]  (soft import guard)
"""

from __future__ import annotations

import pytest
from semql import (
    AuthContext,
    Backend,
    Catalog,
    Cube,
    Dimension,
    Measure,
)
from semql_prompt import render_tool_description, to_langchain_tools, to_openai_tools


def _cube(name: str = "orders", required_roles: list[str] | None = None) -> Cube:
    return Cube(
        name=name,
        backend=Backend.POSTGRES,
        table=f"public.{name}",
        alias=name[:2],
        description=f"The {name} cube.",
        measures=[Measure(name="revenue", sql=f"{{{name[:2]}}}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql=f"{{{name[:2]}}}.region", type="string")],
        required_roles=required_roles or [],
    )


def _catalog() -> Catalog:
    return Catalog([_cube("orders"), _cube("admin_cube", required_roles=["admin"])])


# ---------------------------------------------------------------------------
# to_openai_function importable
# ---------------------------------------------------------------------------


def test_to_openai_function_importable() -> None:
    from semql_prompt import to_openai_function

    assert to_openai_function is not None


def test_to_openai_function_exported_from_semql_prompt() -> None:
    import semql_prompt

    assert hasattr(semql_prompt, "to_openai_function")


# ---------------------------------------------------------------------------
# to_openai_function format
# ---------------------------------------------------------------------------


def test_to_openai_function_returns_dict() -> None:
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    assert isinstance(result, dict)


def test_to_openai_function_type_field() -> None:
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    assert result["type"] == "function"


def test_to_openai_function_has_function_key() -> None:
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    assert "function" in result
    fn = result["function"]
    assert "name" in fn
    assert "description" in fn
    assert "parameters" in fn


def test_to_openai_function_name_is_query_prefix() -> None:
    from semql_prompt import to_openai_function

    cube = _cube("orders")
    result = to_openai_function(cube)
    assert result["function"]["name"] == "query_orders"


def test_to_openai_function_description_matches_render_tool_description() -> None:
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    assert result["function"]["description"] == render_tool_description(cube)


def test_to_openai_function_parameters_is_json_schema() -> None:
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    params = result["function"]["parameters"]
    assert isinstance(params, dict)
    # JSON Schema should have type or properties
    assert "properties" in params or "type" in params


def test_to_openai_function_parameters_no_blank_descriptions() -> None:
    """All properties in the parameters schema must have a non-empty description."""
    from semql_prompt import to_openai_function

    cube = _cube()
    result = to_openai_function(cube)
    params = result["function"]["parameters"]
    props = params.get("properties", {})
    for prop_name, prop_schema in props.items():
        desc = prop_schema.get("description", "")
        assert desc, f"Property {prop_name!r} has no description in OpenAI function schema"


# ---------------------------------------------------------------------------
# Catalog.to_openai_tools
# ---------------------------------------------------------------------------


def test_to_openai_tools_callable_on_catalog() -> None:
    cat = _catalog()
    assert isinstance(to_openai_tools(cat), list)


def test_catalog_to_openai_tools_returns_list_of_dicts() -> None:
    cat = _catalog()
    tools = to_openai_tools(
        cat,
    )
    assert isinstance(tools, list)
    assert all(isinstance(t, dict) for t in tools)


def test_catalog_to_openai_tools_excludes_role_gated_without_viewer() -> None:
    cat = _catalog()
    tools = to_openai_tools(
        cat,
    )
    names = [t["function"]["name"] for t in tools]
    assert "query_orders" in names
    assert "query_admin_cube" not in names


def test_catalog_to_openai_tools_includes_role_gated_with_viewer() -> None:
    cat = _catalog()
    viewer = AuthContext(viewer_id="u1", roles=["admin"])
    tools = to_openai_tools(cat, viewer=viewer)
    names = [t["function"]["name"] for t in tools]
    assert "query_admin_cube" in names


def test_catalog_to_openai_tools_all_have_type_function() -> None:
    cat = _catalog()
    for tool in to_openai_tools(
        cat,
    ):
        assert tool["type"] == "function"


# ---------------------------------------------------------------------------
# Catalog.to_langchain_tools — soft import guard
# ---------------------------------------------------------------------------


def test_to_langchain_tools_importable() -> None:
    from semql_prompt import to_langchain_tools

    assert to_langchain_tools is not None


def test_catalog_to_langchain_tools_raises_import_error_without_langchain() -> None:
    """Without langchain-core installed, raises ImportError with helpful message."""
    import sys

    # Temporarily hide langchain_core from imports
    langchain_mod = sys.modules.pop("langchain_core", None)
    langchain_tools_mod = sys.modules.pop("langchain_core.tools", None)
    try:
        cat = _catalog()
        # Re-import with hidden module
        with pytest.raises(ImportError, match="langchain"):
            to_langchain_tools(
                cat,
            )
    finally:
        if langchain_mod is not None:
            sys.modules["langchain_core"] = langchain_mod
        if langchain_tools_mod is not None:
            sys.modules["langchain_core.tools"] = langchain_tools_mod
