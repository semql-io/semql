"""Tests for ``semql_mcp.MCPServer``.

The server wraps a ``semql.Catalog`` and exposes the compiler /
validator / prompt-renderer surfaces as MCP tools. It does NOT execute
SQL — semql's compiler is pure and so is this server. Callers run
the emitted SQL against whatever backend they own.

Tests use FastMCP's in-process ``Client`` so we exercise the real
tool-registration plumbing without a transport.
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
    Measure,
    SemanticQuery,
)
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    """asyncio.run wrapper typed via PEP 695 generic syntax."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _orders_catalog() -> Catalog:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        description="Order lines.",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
        ],
    )
    return Catalog([orders])


def _server() -> MCPServer:
    return MCPServer(_orders_catalog())


def _client(server: MCPServer) -> Client[Any]:
    return Client(server.mcp)


# ---------------------------------------------------------------------------
# Server constructs; tools registered
# ---------------------------------------------------------------------------


def test_server_constructs_from_catalog() -> None:
    s = _server()
    assert isinstance(s, MCPServer)


def test_server_exposes_expected_tools() -> None:
    s = _server()

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert {"query_semantic", "validate", "explain", "catalog_prompt"}.issubset(names)


# ---------------------------------------------------------------------------
# query_semantic
# ---------------------------------------------------------------------------


def test_query_semantic_returns_compiled_shape() -> None:
    s = _server()

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "query_semantic",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "sql" in out
    assert "params" in out
    assert "columns" in out
    assert "column_meta" in out
    assert "backend" in out
    assert out["backend"] == "postgres"
    assert "SUM" in out["sql"].upper()
    assert out["columns"] == ["region", "revenue"]
    # column_meta lines up 1:1 with columns and carries kind+unit info
    # so a consuming LLM knows how to render each value.
    assert [m["name"] for m in out["column_meta"]] == ["region", "revenue"]
    kinds = {m["name"]: m["kind"] for m in out["column_meta"]}
    assert kinds["region"] == "dimension"
    assert kinds["revenue"] == "measure"


def test_query_semantic_on_unknown_field_returns_error_payload() -> None:
    """The MCP tool surface should never crash the server — unknown
    identifiers come back as a structured error response."""
    s = _server()

    async def call() -> Any:  # noqa: ANN401
        async with _client(s) as c:
            spec = {"measures": ["orders.nonexistent"], "dimensions": []}
            return await c.call_tool("query_semantic", {"spec": spec})

    result = _run(call())
    # The result should indicate an error - either via is_error flag
    # or via the error field on the structured content.
    data: Any = result.data
    assert result.is_error or (isinstance(data, dict) and "error" in data)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def test_validate_returns_empty_list_for_valid_query() -> None:
    s = _server()

    async def call() -> list[Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "validate",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    errors = _run(call())
    assert errors == []


def test_validate_returns_errors_for_broken_query() -> None:
    s = _server()

    async def call() -> list[Any]:
        async with _client(s) as c:
            spec = {
                "measures": ["orders.no_such_measure", "nope.boom"],
                "dimensions": [],
            }
            result = await c.call_tool("validate", {"spec": spec})
            return result.data  # type: ignore[no-any-return]

    errors = _run(call())
    assert len(errors) >= 2
    codes = {e["code"] for e in errors}
    assert "unknown_field" in codes
    assert "unknown_cube" in codes


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------


def test_explain_returns_compiled_sql_string() -> None:
    s = _server()

    async def call() -> str:
        async with _client(s) as c:
            result = await c.call_tool(
                "explain",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    sql = _run(call())
    assert isinstance(sql, str)
    assert "SUM" in sql.upper()
    assert "GROUP BY" in sql.upper()


# ---------------------------------------------------------------------------
# catalog_prompt
# ---------------------------------------------------------------------------


def test_catalog_prompt_returns_planner_fragment() -> None:
    s = _server()

    async def call() -> str:
        async with _client(s) as c:
            result = await c.call_tool("catalog_prompt", {})
            return result.data  # type: ignore[no-any-return]

    text = _run(call())
    assert isinstance(text, str)
    assert "SEMANTIC CATALOG" in text
    assert "orders.revenue" in text


def test_catalog_prompt_respects_only_exposed_flag() -> None:
    hidden = Cube(
        name="internal",
        backend=Dialect.POSTGRES,
        table="internal",
        alias="i",
        expose_in_prompt=False,
        dimensions=[Dimension(name="x", sql="{i}.x", type="string")],
    )
    cat = Catalog([_orders_catalog().as_dict()["orders"], hidden])
    s = MCPServer(cat)

    async def call(only_exposed: bool) -> str:
        async with _client(s) as c:
            result = await c.call_tool("catalog_prompt", {"only_exposed": only_exposed})
            return result.data  # type: ignore[no-any-return]

    exposed_only = _run(call(True))
    full = _run(call(False))
    assert "### internal" not in exposed_only
    assert "### internal" in full


# ---------------------------------------------------------------------------
# Context threading — compile-time substitutions pass through
# ---------------------------------------------------------------------------


def test_query_semantic_accepts_context_kwarg() -> None:
    """The compiler's ``context`` dict (for ``{schema}`` and ``{ctx.X}``
    substitution) needs a way to reach the server side."""
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="{schema}.orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    s = MCPServer(Catalog([cube]))

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "query_semantic",
                {
                    "spec": SemanticQuery(
                        measures=["orders.count"], dimensions=["orders.region"]
                    ).model_dump(),
                    "context": {"schema": "prod"},
                },
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "prod.orders" in out["sql"]


# ---------------------------------------------------------------------------
# Stdio transport smoke — the run hook exists and is callable.
# ---------------------------------------------------------------------------


def test_server_exposes_run_method() -> None:
    s = _server()
    # Don't actually run — that would block on stdin. Just confirm the
    # API surface so the README's "launch over stdio" claim is grounded.
    assert callable(getattr(s, "run", None))


# ---------------------------------------------------------------------------
# Query execute mode — opt-in via executor kwarg
# ---------------------------------------------------------------------------


def test_query_execute_tool_not_registered_without_executor() -> None:
    s = _server()  # no executor
    assert s.executor is None

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    names = _run(fetch())
    assert "query_execute" not in names


def test_query_execute_registered_when_executor_supplied() -> None:
    def fake_exec(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001
        return []

    s = MCPServer(_orders_catalog(), executor=fake_exec)

    async def fetch() -> set[str]:
        async with _client(s) as c:
            tools = await c.list_tools()
            return {t.name for t in tools}

    assert "query_execute" in _run(fetch())


def test_query_execute_returns_rows_plus_envelope() -> None:
    """The exec response carries the SQL/params/columns envelope (so
    the caller sees what ran) plus the row payload."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_exec(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        calls.append((sql, params))
        return [
            {"region": "us", "revenue": 1000},
            {"region": "eu", "revenue": 700},
        ]

    s = MCPServer(_orders_catalog(), executor=fake_exec)

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "query_execute",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    # The executor saw the compiled SQL the server itself emitted.
    assert len(calls) == 1
    assert "SUM" in calls[0][0].upper()
    # Response envelope.
    assert out["columns"] == ["region", "revenue"]
    assert out["backend"] == "postgres"
    assert out["rows"] == [
        {"region": "us", "revenue": 1000},
        {"region": "eu", "revenue": 700},
    ]


def test_query_execute_compile_failure_surfaces_structured_error() -> None:
    def never_called(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001
        raise AssertionError("executor must not run when compile fails")

    s = MCPServer(_orders_catalog(), executor=never_called)

    async def call() -> Any:  # noqa: ANN401
        async with _client(s) as c:
            spec = {"measures": ["orders.no_such"], "dimensions": []}
            return await c.call_tool("query_execute", {"spec": spec})

    result = _run(call())
    data: Any = result.data
    assert result.is_error or (isinstance(data, dict) and "error" in data)


def test_query_execute_execute_failure_carries_sql_for_debugging() -> None:
    """When the executor raises, the response should still include the
    SQL we tried to run so the caller can inspect / replay it."""

    def boom(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001
        raise RuntimeError("connection refused")

    s = MCPServer(_orders_catalog(), executor=boom)

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "query_execute",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "error" in out
    assert out["error"]["code"] == "RuntimeError"
    assert "connection refused" in out["error"]["message"]
    # The compiled SQL is still in the response so the caller can
    # replay / inspect it.
    assert "sql" in out
    assert "SUM" in out["sql"].upper()


def test_query_semantic_compile_only_even_when_executor_set() -> None:
    """query_semantic stays compile-only regardless of executor —
    the two tools are explicit about whether they touch the DB."""

    def fake_exec(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001
        raise AssertionError("query_semantic must not execute")

    s = MCPServer(_orders_catalog(), executor=fake_exec)

    async def call() -> dict[str, Any]:
        async with _client(s) as c:
            result = await c.call_tool(
                "query_semantic",
                {
                    "spec": SemanticQuery(
                        measures=["orders.revenue"],
                        dimensions=["orders.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    out = _run(call())
    assert "rows" not in out
    assert "sql" in out
