"""OpenAI-function-format adapter + LangChain integration.

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

from typing import Any, cast

import pytest
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
)
from semql_prompt import render_tool_description, to_langchain_tools, to_openai_tools


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


def test_to_openai_function_parameters_object_rooted_despite_recursion() -> None:
    """``SemanticQuery`` is recursive (``semi_joins[].source`` is itself a
    ``SemanticQuery``), so Pydantic emits a root ``$ref``. The projection must
    flatten it to an object-rooted schema while keeping the recursive
    ``$defs`` so the self-reference still resolves — otherwise OpenAI /
    Bedrock tool-calling rejects the schema."""
    from semql_prompt import to_openai_function

    params = to_openai_function(_cube())["function"]["parameters"]
    assert params.get("type") == "object"
    assert "$ref" not in params  # root ref flattened away
    assert "semi_joins" in params["properties"]
    # The recursion target survives, and the inner source still refs it.
    assert "SemanticQuery" in params["$defs"]
    source = params["$defs"]["SemiJoin"]["properties"]["source"]
    assert source["$ref"] == "#/$defs/SemanticQuery"


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


def test_catalog_to_langchain_tools_raises_import_error_without_langchain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency fails clearly, even when installed for other tests."""
    import sys

    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.tools", None)
    with pytest.raises(ImportError, match="langchain"):
        to_langchain_tools(_catalog())


# ---------------------------------------------------------------------------
# Tool-visibility unification (S12) — both exporters share one decision,
# and both expose saved queries.
# ---------------------------------------------------------------------------


def _catalog_with_saved() -> Catalog:
    from semql import SemanticQuery
    from semql.spec import SavedQuery

    return Catalog(
        [_cube("orders"), _cube("admin_cube", required_roles=["admin"])],
        saved_queries=[
            SavedQuery(name="public_report", query=SemanticQuery(measures=["orders.revenue"])),
            SavedQuery(
                name="admin_report",
                query=SemanticQuery(measures=["orders.revenue"]),
                required_roles=["admin"],
            ),
        ],
    )


def test_openai_tools_expose_saved_queries_with_role_gating() -> None:
    cat = _catalog_with_saved()
    anon = {t["function"]["name"] for t in to_openai_tools(cat)}
    assert "saved_public_report" in anon
    assert "saved_admin_report" not in anon  # role-gated, no viewer

    admin = {
        t["function"]["name"]
        for t in to_openai_tools(cat, viewer=AuthContext(viewer_id="u", roles=["admin"]))
    }
    assert "saved_admin_report" in admin


def test_openai_parameter_schemas_do_not_alias_across_saved_queries() -> None:
    from semql import SemanticQuery
    from semql.spec import SavedQuery

    catalog = Catalog(
        [_cube()],
        saved_queries=[
            SavedQuery(name="first", query=SemanticQuery()),
            SavedQuery(name="second", query=SemanticQuery()),
        ],
    )
    tools = to_openai_tools(catalog)
    first, second = (
        tool["function"] for tool in tools if tool["function"]["name"].startswith("saved_")
    )
    first["parameters"]["properties"]["limit"]["x-test-mutation"] = {"nested": []}
    assert "x-test-mutation" not in second["parameters"]["properties"]["limit"]


@pytest.mark.parametrize("profile", ["tool", "strict"])
def test_langchain_tools_generate_schema_and_invoke_public_contract(profile: str) -> None:
    pytest.importorskip("langchain_core")
    import duckdb

    # jsonschema supplies no inline typing; validate the actual provider schema.
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
    from semql import SavedQuery, SemanticQuery
    from semql.schema import SchemaProfile

    catalog = Catalog(
        [
            Cube(
                name="orders",
                dialect=Dialect.DUCKDB,
                table="orders",
                alias="o",
                measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
                dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
            )
        ],
        saved_queries=[
            SavedQuery(name="total", query=SemanticQuery(measures=["orders.revenue"])),
            SavedQuery(
                name="regions",
                query=SemanticQuery(
                    measures=["orders.revenue"],
                    dimensions=["orders.region"],
                    order=[("orders.region", "asc")],
                ),
            ),
        ],
    )
    # The optional adapter deliberately returns opaque objects without importing
    # LangChain into the package's public type surface.
    exported = cast(list[Any], to_langchain_tools(catalog, profile=SchemaProfile(profile)))
    tools = {tool.name: tool for tool in exported}
    cube_tool = tools["query_orders"]
    payload = {"measures": ["orders.revenue"]}
    validator = cast(Any, Draft202012Validator(cube_tool.args_schema.model_json_schema()))
    validator.validate(payload)
    if profile == "strict":
        with pytest.raises(ValueError, match="where"):
            cube_tool.invoke({**payload, "where": None})
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE orders(region VARCHAR, amount INTEGER)")
        connection.execute("INSERT INTO orders VALUES ('A', 10), ('A', 20), ('B', 5)")
        for name, arguments, expected in [
            ("query_orders", payload, [(35,)]),
            ("saved_total", {}, [(35,)]),
            ("saved_regions", {}, [("A", 30), ("B", 5)]),
        ]:
            compiled = tools[name].invoke(arguments)
            assert connection.execute(compiled["sql"], compiled["params"]).fetchall() == expected


def test_visible_tool_targets_is_the_shared_decision() -> None:
    """Both exporters route through _visible_tool_targets, so the cube +
    saved-query sets they expose are identical (no drift, and langchain
    no longer silently omits saved queries)."""
    from semql_prompt.catalog_tools import (
        _visible_tool_targets,  # pyright: ignore[reportPrivateUsage]
    )

    cat = _catalog_with_saved()
    viewer = AuthContext(viewer_id="u", roles=["admin"])
    cubes, saved = _visible_tool_targets(cat, viewer=viewer, only_exposed=True)
    cube_names = {c.name for c in cubes}
    saved_names = {sq.name for sq in saved}

    openai_names = {t["function"]["name"] for t in to_openai_tools(cat, viewer=viewer)}
    assert {f"query_{n}" for n in cube_names} <= openai_names
    assert {f"saved_{n}" for n in saved_names} <= openai_names
    # admin viewer sees the admin cube + both saved queries.
    assert "admin_cube" in cube_names
    assert saved_names == {"public_report", "admin_report"}
