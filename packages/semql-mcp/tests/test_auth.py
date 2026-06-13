"""A6 — the MCP server must enforce identity via a trusted provider.

Before the fix the server compiled every query with ``viewer=None``, so
``required_roles`` cube visibility and ``security_sql`` row scoping were
never enforced — any client could read every cube. Worse, the lookup
tools took client-asserted ``viewer_id`` / ``roles``, which on a networked
transport lets a client name itself.

The fix injects a ``viewer_provider`` (the deployer derives identity from
the transport's authenticated request context). It is threaded into every
``compile(viewer=...)`` and is authoritative over client-asserted values.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable
from typing import Any

import pytest
from fastmcp import Client
from semql import Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery
from semql.model import AuthContext
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _gated_catalog() -> Catalog:
    """A cube only viewers with the ``admin`` role may see."""
    secret = Cube(
        name="secret",
        backend=Dialect.POSTGRES,
        table="secret",
        alias="s",
        required_roles=["admin"],
        measures=[Measure(name="total", sql="{s}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{s}.region", type="string")],
    )
    return Catalog([secret])


def _call_query_semantic(server: MCPServer) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        async with Client(server.mcp) as c:
            result = await c.call_tool(
                "query_semantic",
                {
                    "spec": SemanticQuery(
                        measures=["secret.total"],
                        dimensions=["secret.region"],
                    ).model_dump()
                },
            )
            return result.data  # type: ignore[no-any-return]

    return _run(call())


def test_query_tool_refuses_when_provider_viewer_lacks_role() -> None:
    server = MCPServer(
        _gated_catalog(), viewer_provider=lambda: AuthContext(viewer_id="u", roles=[])
    )
    out = _call_query_semantic(server)
    assert "error" in out
    assert "not authorised" in out["error"]["message"]


def test_query_tool_allows_when_provider_viewer_has_role() -> None:
    server = MCPServer(
        _gated_catalog(),
        viewer_provider=lambda: AuthContext(viewer_id="u", roles=["admin"]),
    )
    out = _call_query_semantic(server)
    assert "error" not in out
    assert "SUM" in out["sql"].upper()


def test_no_provider_still_compiles_without_viewer() -> None:
    """Backwards-compatible stdio default: with no provider the query
    compiles with viewer=None (the historical behaviour). This is the
    single-tenant trust model the run() warning calls out."""
    server = MCPServer(_gated_catalog())
    out = _call_query_semantic(server)
    assert "error" not in out
    assert "SUM" in out["sql"].upper()


def test_resolve_viewer_provider_is_authoritative_over_client_asserted() -> None:
    """A configured provider wins; client-asserted viewer_id / roles (the
    lookup-tool params) are ignored — the whole point on a networked
    transport."""
    server = MCPServer(
        _gated_catalog(),
        viewer_provider=lambda: AuthContext(viewer_id="real", roles=["admin"]),
    )
    resolved = server._resolve_viewer(client_viewer_id="attacker", client_roles=["superuser"])
    assert resolved is not None
    assert resolved.viewer_id == "real"
    assert resolved.roles == ["admin"]


def test_resolve_viewer_falls_back_to_client_asserted_without_provider() -> None:
    server = MCPServer(_gated_catalog())
    resolved = server._resolve_viewer(client_viewer_id="u", client_roles=["analyst"])
    assert resolved is not None
    assert resolved.viewer_id == "u"
    assert resolved.roles == ["analyst"]
    assert server._resolve_viewer() is None


def test_run_warns_on_networked_transport_without_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServer(_gated_catalog())
    monkeypatch.setattr(server.mcp, "run", lambda **_: None)
    with pytest.warns(UserWarning, match="NOT enforced"):
        server.run(transport="http")


def test_run_does_not_warn_with_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServer(_gated_catalog(), viewer_provider=lambda: None)
    monkeypatch.setattr(server.mcp, "run", lambda **_: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        server.run(transport="http")
