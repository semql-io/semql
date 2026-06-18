"""Tests for the auto-registered saved-query MCP tools.

Each :class:`SavedQuery` on the catalog becomes a zero-arg
``saved_<name>`` MCP tool. The tool compiles the pre-baked
SemanticQuery and (when the server was constructed with an executor)
runs it, returning the same envelope shape as ``query_semantic``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

from fastmcp import Client
from semql import (
    AuthContext,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SavedQuery,
    SemanticQuery,
)
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )


def _paid_revenue_query() -> SavedQuery:
    return SavedQuery(
        name="paid_revenue_by_region",
        description="Revenue from paid orders, broken down by region.",
        query=SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
    )


def _catalog_with_saved() -> Catalog:
    return Catalog([_orders_cube()], saved_queries=[_paid_revenue_query()])


def _server(cat: Catalog, *, executor: Any = None, debug: bool = False) -> MCPServer:  # noqa: ANN401
    return MCPServer(cat, executor=executor, debug=debug)


def _client(server: MCPServer) -> Client[Any]:
    return Client(server.mcp)


# ---------------------------------------------------------------------------
# Registration: tool appears iff catalog has saved queries
# ---------------------------------------------------------------------------


def test_saved_query_tool_registered_when_catalog_has_one() -> None:
    s = _server(_catalog_with_saved())

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert "saved_paid_revenue_by_region" in names


def test_no_saved_query_tools_when_catalog_has_none() -> None:
    s = _server(Catalog([_orders_cube()]))

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert not any(n.startswith("saved_") for n in names)


def test_saved_query_tool_carries_description() -> None:
    """The SavedQuery.description shows up as the tool's docstring so
    a planner reading the tool catalog knows what each saved query
    answers."""
    s = _server(_catalog_with_saved())

    async def fetch() -> str | None:
        async with _client(s) as c:
            tools = await c.list_tools()
            tool = next((t for t in tools if t.name == "saved_paid_revenue_by_region"), None)
            assert tool is not None
            return tool.description  # type: ignore[no-any-return]

    desc = _run(fetch())
    assert desc is not None
    assert "Revenue from paid orders" in desc


def test_multiple_saved_queries_register_one_tool_each() -> None:
    q1 = SavedQuery(
        name="q_one",
        query=SemanticQuery(measures=["orders.revenue"]),
    )
    q2 = SavedQuery(
        name="q_two",
        query=SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"]),
    )
    cat = Catalog([_orders_cube()], saved_queries=[q1, q2])
    s = _server(cat)

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert "saved_q_one" in names
    assert "saved_q_two" in names


# ---------------------------------------------------------------------------
# Compile-only behavior (no executor)
# ---------------------------------------------------------------------------


def test_saved_query_tool_returns_compile_envelope() -> None:
    s = _server(_catalog_with_saved())

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool("saved_paid_revenue_by_region", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "sql" in out
    assert "params" in out
    assert "columns" in out
    assert "dialect" in out
    assert out["dialect"] == "postgres"
    assert "SUM" in out["sql"].upper()
    # The status='paid' filter from the saved query is in the params.
    assert "paid" in str(out["params"].values())


def test_saved_query_tool_accepts_context_kwarg() -> None:
    """A saved query on a tenancy-scoped cube needs the runtime tenant
    via ``context`` even though its semantic shape is baked in."""

    def _orders_with_tenancy() -> Cube:
        return Cube(
            name="orders",
            dialect=Dialect.POSTGRES,
            table="{tenant_schema}.orders",
            alias="o",
            tenancy="schema",
            measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        )

    sq = SavedQuery(
        name="tenant_revenue",
        query=SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"]),
    )
    cat = Catalog([_orders_with_tenancy()], saved_queries=[sq])
    s = _server(cat)

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "saved_tenant_revenue", {"context": {"tenant_schema": "acme"}}
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "acme.orders" in out["sql"]


# ---------------------------------------------------------------------------
# Executor-mode: rows come back too
# ---------------------------------------------------------------------------


def test_saved_query_tool_executes_when_executor_set() -> None:
    """When the server is constructed with an executor, the saved-query
    tool returns ``rows`` alongside the compile envelope."""

    def fake_executor(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        # Pretend the warehouse returned a couple of rows.
        return [{"region": "EU", "revenue": 600.0}, {"region": "US", "revenue": 50.0}]

    s = _server(_catalog_with_saved(), executor=fake_executor)

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool("saved_paid_revenue_by_region", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "rows" in out
    assert len(out["rows"]) == 2
    assert out["rows"][0]["region"] == "EU"


def test_saved_query_tool_execute_failure_carries_sql() -> None:
    """If the executor raises, the tool returns a structured error
    payload alongside the SQL — same shape as ``query_execute``. The
    raw driver text is redacted by default; debug mode surfaces it."""

    def boom(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        raise RuntimeError("connection refused")

    async def call(s: MCPServer) -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool("saved_paid_revenue_by_region", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call(_server(_catalog_with_saved(), executor=boom)))
    assert "error" in out
    assert out["error"]["code"] == "ExecutionError"
    assert "connection refused" not in out["error"]["message"]
    # SQL still in the envelope so the caller can debug.
    assert "sql" in out

    dbg = _run(call(_server(_catalog_with_saved(), executor=boom, debug=True)))
    assert dbg["error"]["code"] == "RuntimeError"
    assert "connection refused" in dbg["error"]["message"]


# ---------------------------------------------------------------------------
# Role enforcement (CAND-semql-mcp-saved-query-required-roles-bypass):
# saved-query tools must check sq.required_roles before compiling.
# ---------------------------------------------------------------------------


def _admin_saved_query() -> SavedQuery:
    return SavedQuery(
        name="admin_report",
        description="Admin-only report.",
        required_roles=["admin"],
        query=SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
        ),
    )


def _catalog_with_role_gated_saved() -> Catalog:
    return Catalog([_orders_cube()], saved_queries=[_admin_saved_query()])


def test_saved_query_required_roles_denied_for_non_admin() -> None:
    """A viewer without required_roles must receive an AuthError, not SQL."""
    low_viewer = AuthContext(viewer_id="u1", roles=["viewer"])
    s = MCPServer(
        _catalog_with_role_gated_saved(),
        viewer_provider=lambda: low_viewer,
    )

    async def call() -> dict[str, Any]:
        async with Client(s.mcp) as c:
            result = await c.call_tool("saved_admin_report", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "error" in out
    assert out["error"]["code"] == "AuthError"
    assert "sql" not in out  # must not have compiled at all


def test_saved_query_required_roles_allowed_for_admin() -> None:
    """A viewer with the required role must receive compiled SQL."""
    admin_viewer = AuthContext(viewer_id="u2", roles=["admin"])
    s = MCPServer(
        _catalog_with_role_gated_saved(),
        viewer_provider=lambda: admin_viewer,
    )

    async def call() -> dict[str, Any]:
        async with Client(s.mcp) as c:
            result = await c.call_tool("saved_admin_report", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "error" not in out
    assert "sql" in out


def test_saved_query_required_roles_denied_when_no_viewer() -> None:
    """No viewer (None) with required_roles must be rejected, not pass through."""
    s = MCPServer(
        _catalog_with_role_gated_saved(),
        viewer_provider=lambda: None,
        require_viewer=False,  # allow None viewer at the server level
    )

    async def call() -> dict[str, Any]:
        async with Client(s.mcp) as c:
            result = await c.call_tool("saved_admin_report", {})
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "error" in out
    assert out["error"]["code"] == "AuthError"
