"""The MCP server must enforce identity via a trusted provider.

Before the fix the server compiled every query with ``viewer=None``, so
``required_roles`` cube visibility and ``security_sql`` row scoping were
never enforced — any client could read every cube. Worse, the lookup
tools took client-asserted ``viewer_id`` / ``roles``, which on a networked
transport lets a client name itself.

The fix injects a ``viewer_provider`` (the deployer derives identity from
the transport's authenticated request context). It is threaded into every
``compile(viewer=...)`` and is authoritative over client-asserted values.
"""

# This module exercises MCPServer internals (e.g. ``_resolve_viewer``) on
# purpose, so cross-class private access is expected here.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable
from typing import Any

import pytest
from fastmcp import Client
from semql import Catalog, Cube, Dialect, Dimension, Measure, SemanticQuery
from semql.errors import AuthError
from semql.model import AuthContext
from semql_mcp import MCPServer


def _run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _noop(**_: object) -> None:
    """Typed no-op stand-in for ``mcp.run`` under monkeypatch (a bare
    ``lambda **_: None`` gives pyright an Unknown-typed parameter)."""


def _gated_catalog() -> Catalog:
    """A cube only viewers with the ``admin`` role may see."""
    secret = Cube(
        name="secret",
        dialect=Dialect.POSTGRES,
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
    # A viewer lacking the required role can't see the gated cube at all:
    # it is filtered from their catalog, so resolution reports "Unknown cube"
    # rather than confirming the cube exists ("not authorised"). This
    # hide-existence behaviour is intentional and matches the core compiler
    # (see semql/tests/test_auth.py). The denied query yields no SQL.
    assert "Unknown cube" in out["error"]["message"]
    assert "sql" not in out


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


def test_query_tool_refuses_when_provider_returns_none_by_default() -> None:
    """Fail-secure: a configured provider returning None means 'deny',
    not 'anonymous OK'. The query tool surfaces an AuthError payload
    rather than compiling with viewer=None (which would expose every
    cube without required_roles)."""
    server = MCPServer(_gated_catalog(), viewer_provider=lambda: None)
    out = _call_query_semantic(server)
    assert "error" in out
    assert out["error"]["code"] == "AuthError"


def test_require_viewer_false_allows_anonymous() -> None:
    """Opt-in anonymous: with require_viewer=False a provider returning
    None resolves to viewer=None (unscoped) instead of refusing."""
    server = MCPServer(_gated_catalog(), viewer_provider=lambda: None, require_viewer=False)
    assert server._resolve_viewer() is None


def test_resolve_viewer_raises_when_provider_returns_none() -> None:
    server = MCPServer(_gated_catalog(), viewer_provider=lambda: None)
    with pytest.raises(AuthError):
        server._resolve_viewer()


def test_run_warns_on_networked_transport_without_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServer(_gated_catalog())
    monkeypatch.setattr(server.mcp, "run", _noop)
    with pytest.warns(UserWarning, match="NOT enforced"):
        server.run(transport="http")


def test_run_does_not_warn_with_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    server = MCPServer(_gated_catalog(), viewer_provider=lambda: None)
    monkeypatch.setattr(server.mcp, "run", _noop)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        server.run(transport="http")
