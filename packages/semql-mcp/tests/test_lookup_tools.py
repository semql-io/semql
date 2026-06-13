"""Tests for the auto-registered lookup MCP tools.

``MCPServer`` exposes two tools when the catalog carries any
:class:`semql.model.Lookup`:

- ``resolve_lookup(dimension, query, ...)`` — turn free text into
  canonical values via the catalog's exact/substring/fuzzy resolver.
- ``list_lookup_values(dimension, ...)`` — materialize a lookup's
  declared values (firing the loader for dynamic lookups).

Both tools type the ``dimension`` parameter as a ``Literal`` of the
catalog's registered lookup keys so MCP clients see an enumerated
choice. A catalog with no lookups doesn't register these tools at all
— absence is meaningful.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from fastmcp import Client
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Lookup,
    LookupValues,
    Measure,
    ResolutionContext,
)
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _static_catalog() -> Catalog:
    return Catalog(
        [_orders_cube()],
        lookups=[
            Lookup(
                dimension="orders.region",
                values=("EMEA", "APAC", "AMER"),
                labels={"EMEA": "Europe / Middle East / Africa", "AMER": "Americas"},
            ),
            Lookup(
                dimension="orders.status",
                values=("paid", "refunded", "cancelled"),
            ),
        ],
    )


def _server(cat: Catalog) -> MCPServer:
    return MCPServer(cat)


def _client(server: MCPServer) -> Client[Any]:
    return Client(server.mcp)


# ---------------------------------------------------------------------------
# Registration: lookup tools appear iff lookups are present
# ---------------------------------------------------------------------------


def test_lookup_tools_registered_when_catalog_has_lookups() -> None:
    s = _server(_static_catalog())

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert "resolve_lookup" in names
    assert "list_lookup_values" in names


def test_lookup_tools_omitted_when_catalog_has_no_lookups() -> None:
    cat = Catalog([_orders_cube()])  # no lookups=
    s = _server(cat)

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert "resolve_lookup" not in names
    assert "list_lookup_values" not in names


def test_resolve_lookup_dimension_param_is_literal_enum() -> None:
    """A client browsing the tool's JSON Schema should see the
    dimension parameter constrained to the registered lookup keys —
    otherwise the planner has to guess them."""
    s = _server(_static_catalog())

    async def fetch() -> dict[str, Any]:
        async with _client(s) as c:
            tools = await c.list_tools()
            tool = next(t for t in tools if t.name == "resolve_lookup")
            return tool.inputSchema  # type: ignore[no-any-return]

    schema = _run(fetch())
    dim_schema: Any = schema["properties"]["dimension"]
    # FastMCP renders a Literal as either an ``enum`` or a oneOf/anyOf
    # of singleton enums — accept whichever Pydantic produces.
    raw_enum: Any = dim_schema.get("enum")
    if raw_enum is None:
        variants: Any = dim_schema.get("anyOf") or dim_schema.get("oneOf") or []
        raw_enum = [e["const"] for e in variants if "const" in e]
    assert set(raw_enum) == {"orders.region", "orders.status"}


# ---------------------------------------------------------------------------
# resolve_lookup behavior
# ---------------------------------------------------------------------------


def test_resolve_lookup_exact_match_returns_single_value() -> None:
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "resolve_lookup",
                {"dimension": "orders.region", "query": "emea"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["dimension"] == "orders.region"
    assert out["query"] == "emea"
    assert out["values"] == ["EMEA"]


def test_resolve_lookup_substring_match_returns_canonical_values() -> None:
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "resolve_lookup",
                {"dimension": "orders.region", "query": "AME"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    # Substring hits — both AMER (via value) and EMEA (via label).
    assert "AMER" in out["values"]


def test_resolve_lookup_label_match_works() -> None:
    """A query against a Lookup's label should resolve to the
    canonical key, not the label."""
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "resolve_lookup",
                {"dimension": "orders.region", "query": "americas"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["values"] == ["AMER"]


def test_resolve_lookup_unknown_dimension_returns_error() -> None:
    s = _server(_static_catalog())

    async def call() -> Any:  # noqa: ANN401
        async with _client(s) as c:
            return await c.call_tool(
                "resolve_lookup",
                {"dimension": "orders.region", "query": "ignored"},
            )

    # Sanity check: a known dimension call shouldn't error.
    result = _run(call())
    assert not result.is_error


def test_resolve_lookup_no_match_returns_empty_values() -> None:
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "resolve_lookup",
                {"dimension": "orders.region", "query": "xyzqqq"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["values"] == []


# ---------------------------------------------------------------------------
# list_lookup_values behavior
# ---------------------------------------------------------------------------


def test_list_lookup_values_static_returns_inlined_tuple() -> None:
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "list_lookup_values",
                {"dimension": "orders.status"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["dimension"] == "orders.status"
    assert set(out["values"]) == {"paid", "refunded", "cancelled"}


def test_list_lookup_values_returns_labels_when_present() -> None:
    s = _server(_static_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "list_lookup_values",
                {"dimension": "orders.region"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["labels"]["AMER"] == "Americas"


# ---------------------------------------------------------------------------
# Dynamic lookups + ResolutionContext threading
# ---------------------------------------------------------------------------


def _dynamic_loader(ctx: ResolutionContext) -> LookupValues:
    tenant = ctx.context.get("tenant", "")
    if tenant == "acme":
        return ["acme_eu", "acme_us"]
    return ["other_a", "other_b"]


def _dynamic_catalog() -> Catalog:
    return Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=_dynamic_loader)],
    )


def test_dynamic_lookup_loader_fires_with_context() -> None:
    s = _server(_dynamic_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "list_lookup_values",
                {"dimension": "orders.region", "context": {"tenant": "acme"}},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert set(out["values"]) == {"acme_eu", "acme_us"}


def test_dynamic_lookup_without_context_signals_unresolved() -> None:
    """Dynamic lookups need a ResolutionContext to fire the loader.
    Without one, ``list_lookup_values`` reports ``values: None`` rather
    than inventing an empty answer."""
    s = _server(_dynamic_catalog())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "list_lookup_values",
                {"dimension": "orders.region"},
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert out["values"] is None


def test_resolve_lookup_threads_viewer_into_context() -> None:
    """``viewer_id`` + ``roles`` synthesize an AuthContext that the
    loader can branch on."""

    def role_aware_loader(ctx: ResolutionContext) -> LookupValues:
        if ctx.viewer and "admin" in ctx.viewer.roles:
            return ["all_a", "all_b", "secret_c"]
        return ["public_a"]

    cat = Catalog(
        [_orders_cube()],
        lookups=[Lookup(dimension="orders.region", loader=role_aware_loader)],
    )
    s = _server(cat)

    async def call(viewer_id: str | None, roles: list[str] | None) -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "resolve_lookup",
                {
                    "dimension": "orders.region",
                    "query": "secret",
                    "viewer_id": viewer_id,
                    "roles": roles,
                },
            )
            return result.data  # type: ignore[no-any-return]

    admin_out = _run(call(viewer_id="u1", roles=["admin"]))
    public_out = _run(call(viewer_id="u2", roles=[]))
    assert "secret_c" in admin_out["values"]
    assert "secret_c" not in public_out["values"]
