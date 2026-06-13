# pyright: reportUnusedFunction=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportUnknownMemberType=false
# - reportUnusedFunction: FastMCP's @mcp.tool decorators register the
#   wrapped function with the server; pyright sees the local name as
#   "unused" because it can't follow the decorator's side effect.
# - reportUnknownParameterType / Argument / Variable / Missing: the
#   per-cube tool factory builds dynamic ``Literal[...]`` annotations
#   at runtime and attaches them via ``__annotations__``; the def
#   itself intentionally has no static type hints.
"""MCP server wrapping a ``semql.Catalog``.

The server exposes the compiler / validator / prompt-renderer surfaces
as MCP tools so an LLM (or any MCP client) can plan and reason about
semantic queries against the catalog.

By default the server is **compile-only**: ``semql`` is pure, so is
this server, and callers run the emitted SQL against whatever backend
they own. Pass an ``executor`` at construction to opt into row-returning
mode — a ``query_execute`` tool registers in addition to the
compile-only tools, runs the SQL against the executor, and returns
both the SQL and the rows. The executor is the only stateful surface
the server owns; everything else stays pure.

Tools always registered:
- ``query_semantic(spec, context?)`` — compile a SemanticQuery, return
  ``{sql, params, columns, backend}``.
- ``validate(spec)`` — collect-all static validation; returns a list
  of ``ValidationError`` records.
- ``explain(spec, context?)`` — same as ``query_semantic`` but returns
  just the SQL string.
- ``catalog_prompt(only_exposed=True, include_introspection=False)`` —
  planner prompt fragment.

Registered only when ``executor`` is supplied:
- ``query_execute(spec, context?)`` — compile + run; returns the
  ``query_semantic`` shape plus ``rows: list[dict]``.

Registered only when the catalog carries ``Lookup`` entries:
- ``resolve_lookup(dimension, query, context?, viewer_id?, roles?)`` —
  turn a free-text query into canonical dimension values via the
  configured lookup's exact / substring / fuzzy resolver.
- ``list_lookup_values(dimension, context?, viewer_id?, roles?)`` —
  materialize a lookup's full value set (firing the loader for
  dynamic lookups) so a planner can browse the vocabulary.

Registered when the catalog carries ``SavedQuery`` entries:
- ``saved_<name>(context?)`` — one zero-arg tool per saved query.
  Compiles the pre-baked ``SemanticQuery`` and (if an executor is
  configured) runs it. Documentation comes from
  ``SavedQuery.description``.

Transports: stdio (FastMCP default) plus anything FastMCP supports
out of the box. Use ``server.run(transport="stdio")`` for a
CLI-launched process, or pass ``server.mcp`` to a custom transport.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

from fastmcp import FastMCP
from semql import Catalog
from semql.lookups import materialize as materialize_lookup
from semql.lookups import resolve as resolve_lookup
from semql.model import AuthContext, Cube, ResolutionContext
from semql.spec import Filter, SemanticQuery, TimeWindow
from semql.validate import ValidationError
from semql.validate import validate as validate_query

Transport = Literal["stdio", "http", "sse", "streamable-http"]
"""FastMCP transport identifiers."""

Executor = Callable[[str, dict[str, Any]], list[dict[str, Any]]]
"""``(sql, params) -> rows`` — sync executor surface.

Callers provide their own database driver (psycopg, clickhouse-connect,
DuckDB, ...) and adapt its row shape to a list of dicts. The MCP
server never imports a database driver itself."""

ViewerProvider = Callable[[], "AuthContext | None"]
"""``() -> AuthContext | None`` — the trusted, per-request identity source.

This is the fix for the multi-tenant auth hole (A6). On a networked
transport (http / sse / streamable-http) the *client* cannot be trusted
to assert who it is, so tool parameters like ``viewer_id`` / ``roles`` are
not an authorization boundary. The deployer wires a ``viewer_provider``
that derives the verified identity from the transport's authenticated
request context (a validated bearer token, an mTLS client cert, a session)
and the server threads its result into every ``catalog.compile(viewer=...)``
call — so ``required_roles`` cube/field visibility and ``security_sql``
row-level scoping are actually enforced.

The provider is invoked inside each tool call, so it sees the current
request's context. Return ``None`` for an unauthenticated request. When a
provider is configured it is authoritative: client-asserted ``viewer_id`` /
``roles`` are ignored. With no provider (the stdio single-tenant default)
the server falls back to the client-asserted values, which is the only
mode where trusting them is safe."""


class MCPServer:
    """An MCP server exposing a SemQL ``Catalog`` to MCP clients.

    The ``mcp`` attribute is the underlying ``FastMCP`` instance — pass
    it to a ``fastmcp.Client`` for in-process testing, or call
    ``server.run(transport=...)`` to launch a real transport."""

    def __init__(
        self,
        catalog: Catalog,
        *,
        executor: Executor | None = None,
        viewer_provider: ViewerProvider | None = None,
        name: str = "semql",
    ) -> None:
        self.catalog = catalog
        self.executor = executor
        self.viewer_provider = viewer_provider
        self.mcp = FastMCP(name=name)
        self._register_tools()
        self._register_per_cube_tools()
        self._register_lookup_tools()
        self._register_saved_query_tools()

    def _resolve_viewer(
        self,
        client_viewer_id: str | None = None,
        client_roles: list[str] | None = None,
    ) -> AuthContext | None:
        """Resolve the authoritative viewer for a tool call.

        A configured ``viewer_provider`` wins outright — client-asserted
        ``viewer_id`` / ``roles`` are ignored, because on a networked
        transport the client can't be trusted to name itself. With no
        provider (stdio single-tenant), fall back to the client-asserted
        values; that's the only context where trusting them is safe."""
        if self.viewer_provider is not None:
            return self.viewer_provider()
        if client_viewer_id is not None:
            return AuthContext(viewer_id=client_viewer_id, roles=list(client_roles or []))
        return None

    def _register_tools(self) -> None:
        catalog = self.catalog
        executor = self.executor
        resolve_viewer = self._resolve_viewer

        @self.mcp.tool(
            name="query_semantic",
            description=(
                "Compile a SemanticQuery against the catalog and return "
                "the emitted SQL, the bound parameters, and the output "
                "column names. Pass ``context`` for ``{schema}`` / "
                "``{ctx.X}`` substitution at compile time."
            ),
        )
        def query_semantic(
            spec: SemanticQuery,
            context: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            try:
                compiled = catalog.compile(spec, context=context, viewer=resolve_viewer())
            except Exception as exc:
                return _error_payload(exc)
            return {
                "backend": compiled.backend.value,
                "sql": compiled.sql,
                "params": compiled.params,
                "columns": compiled.columns,
                "column_meta": [asdict(m) for m in compiled.column_meta],
            }

        @self.mcp.tool(
            name="validate",
            description=(
                "Run collect-all static validation. Returns a list of "
                "structured ValidationError records — empty when the "
                "query would compile cleanly."
            ),
        )
        def validate(spec: SemanticQuery) -> list[dict[str, Any]]:
            errors: list[ValidationError] = validate_query(spec, catalog)
            return [asdict(e) for e in errors]

        @self.mcp.tool(
            name="explain",
            description=(
                "Compile a SemanticQuery and return just the SQL string. "
                "Equivalent to ``query_semantic(...).sql`` — handy for "
                "debugging 'what would you have run' without the params "
                "envelope."
            ),
        )
        def explain(
            spec: SemanticQuery,
            context: dict[str, str] | None = None,
        ) -> str:
            try:
                compiled = catalog.compile(spec, context=context, viewer=resolve_viewer())
            except Exception as exc:
                return f"-- compile failed: {exc}"
            return compiled.sql

        @self.mcp.tool(
            name="catalog_prompt",
            description=(
                "Render the planner prompt fragment for this catalog — "
                "what an LLM planner would see to learn the catalog's "
                "vocabulary and the SemanticQuery contract."
            ),
        )
        def catalog_prompt(
            only_exposed: bool = True,
            include_introspection: bool = False,
        ) -> str:
            from semql_prompt import planner_prompt

            return planner_prompt(
                catalog,
                only_exposed=only_exposed,
                include_introspection=include_introspection,
            )

        if executor is not None:

            @self.mcp.tool(
                name="query_execute",
                description=(
                    "Compile a SemanticQuery, execute it against the "
                    "configured database, and return both the SQL/params "
                    "envelope and the resulting rows. Available only when "
                    "the server was constructed with an ``executor``. "
                    "Errors from compile or execute surface as a "
                    "structured ``{error}`` payload."
                ),
            )
            def query_execute(
                spec: SemanticQuery,
                context: dict[str, str] | None = None,
            ) -> dict[str, Any]:
                try:
                    compiled = catalog.compile(spec, context=context, viewer=resolve_viewer())
                except Exception as exc:
                    return _error_payload(exc)
                try:
                    rows = executor(compiled.sql, compiled.params)
                except Exception as exc:
                    return _error_payload(exc) | {
                        "sql": compiled.sql,
                        "params": compiled.params,
                    }
                return {
                    "backend": compiled.backend.value,
                    "sql": compiled.sql,
                    "params": compiled.params,
                    "columns": compiled.columns,
                    "column_meta": [asdict(m) for m in compiled.column_meta],
                    "rows": rows,
                }

    def _register_lookup_tools(self) -> None:
        """Register ``resolve_lookup`` + ``list_lookup_values`` when the
        catalog carries any :class:`semql.model.Lookup`.

        Skipped on empty-lookup catalogs so a server with no resolvable
        dimensions doesn't advertise misleading tools."""
        catalog = self.catalog
        resolve_viewer = self._resolve_viewer
        if not catalog.lookups:
            return

        dim_keys = tuple(sorted(catalog.lookups))
        dim_t = Literal[dim_keys]  # type: ignore[valid-type]

        def resolve_lookup_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202 — signature attached via __annotations__ below
            dimension,  # noqa: ANN001
            query,  # noqa: ANN001
            context=None,  # noqa: ANN001
            viewer_id=None,  # noqa: ANN001
            roles=None,  # noqa: ANN001
            max_candidates=5,  # noqa: ANN001
        ):
            try:
                ctx = _build_resolution_ctx(resolve_viewer(viewer_id, roles), context)
                values = resolve_lookup(
                    catalog,
                    dimension,
                    query,
                    ctx=ctx,
                    max_candidates=max_candidates,
                )
            except Exception as exc:
                return _error_payload(exc)
            return {"dimension": dimension, "query": query, "values": values}

        resolve_lookup_fn.__name__ = "resolve_lookup"
        resolve_lookup_fn.__doc__ = (
            "Turn a free-text ``query`` into a list of canonical "
            "values for ``dimension`` using the catalog's configured "
            "Lookup. Resolution is exact-case-insensitive → substring "
            "→ fuzzy. Pass ``context`` / ``viewer_id`` / ``roles`` for "
            "dynamic lookups whose loader needs a ResolutionContext."
            f"\n\nDimensions with lookups: {', '.join(dim_keys)}."
        )
        resolve_lookup_fn.__annotations__ = {
            "dimension": dim_t,
            "query": str,
            "context": dict[str, str] | None,
            "viewer_id": str | None,
            "roles": list[str] | None,
            "max_candidates": int,
            "return": dict[str, Any],
        }
        self.mcp.add_tool(resolve_lookup_fn)

        def list_lookup_values_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202 — signature attached via __annotations__ below
            dimension,  # noqa: ANN001
            context=None,  # noqa: ANN001
            viewer_id=None,  # noqa: ANN001
            roles=None,  # noqa: ANN001
        ):
            try:
                lookup = catalog.lookups[dimension]
                ctx = _build_resolution_ctx(resolve_viewer(viewer_id, roles), context)
                materialized = materialize_lookup(lookup, ctx)
            except Exception as exc:
                return _error_payload(exc)
            if materialized is None:
                # Dynamic lookup, no context — signal "values resolved
                # at runtime" rather than inventing an empty answer.
                return {"dimension": dimension, "values": None, "labels": None}
            values, labels = materialized
            return {"dimension": dimension, "values": values, "labels": labels}

        list_lookup_values_fn.__name__ = "list_lookup_values"
        list_lookup_values_fn.__doc__ = (
            "Return the full materialized value set of ``dimension``'s "
            "Lookup. Static lookups return their declared tuple; dynamic "
            "lookups fire the loader against the supplied context. "
            "Returns ``{values: None}`` when a dynamic loader can't "
            "run because no context was passed."
            f"\n\nDimensions with lookups: {', '.join(dim_keys)}."
        )
        list_lookup_values_fn.__annotations__ = {
            "dimension": dim_t,
            "context": dict[str, str] | None,
            "viewer_id": str | None,
            "roles": list[str] | None,
            "return": dict[str, Any],
        }
        self.mcp.add_tool(list_lookup_values_fn)

    def _register_saved_query_tools(self) -> None:
        """For each ``SavedQuery`` on the catalog, register a zero-arg
        ``saved_<name>`` MCP tool that compiles + (if an executor is
        configured) executes the saved query.

        Saved queries with non-empty ``required_roles`` aren't filtered
        here — visibility is enforced per-call at compile time via the
        catalog's policy / required_roles plumbing. The MCP server
        exposes every registered saved query as a tool; whether the
        viewer is *allowed* to call it surfaces as a structured error
        when they try."""
        catalog = self.catalog
        executor = self.executor
        if not catalog.saved_queries:
            return

        for sq in catalog.saved_queries.values():
            self.mcp.add_tool(_make_saved_query_tool(sq, catalog, executor, self._resolve_viewer))

    def _register_per_cube_tools(self) -> None:
        """For each exposed, non-META cube, register a ``query_<cube>``
        tool whose ``measures`` / ``dimensions`` / ``time_window.dimension``
        parameters are ``Literal``-typed enums of the cube's actual
        fields. Hidden cubes (``expose_in_prompt=False``) and META
        reflection cubes are skipped — multi-cube and introspection
        queries go through ``query_semantic``."""
        from semql import iter_cubes

        catalog = self.catalog
        executor = self.executor
        for cube in iter_cubes(catalog, only_exposed=True):
            self.mcp.add_tool(_make_query_cube_tool(cube, catalog, executor, self._resolve_viewer))

    def run(self, transport: Transport = "stdio", **kwargs: Any) -> None:  # noqa: ANN401
        """Launch the server on ``transport``.

        Defaults to stdio so a parent process can spawn the server and
        speak JSON-RPC over its stdin/stdout. Forwards remaining kwargs
        to FastMCP.

        On a networked transport with no ``viewer_provider`` configured,
        warns once: every request compiles with no viewer, so
        ``required_roles`` and ``security_sql`` scoping are not enforced
        and any client can read every cube. Safe only behind a trusted
        single-tenant boundary."""
        if transport != "stdio" and self.viewer_provider is None:
            warnings.warn(
                f"MCPServer.run(transport={transport!r}) with no "
                "viewer_provider: requests compile with no viewer, so "
                "required_roles / security_sql are NOT enforced and any "
                "client can read every cube. Pass viewer_provider= to "
                "derive the identity from the transport's authenticated "
                "request context.",
                stacklevel=2,
            )
        self.mcp.run(transport=transport, **kwargs)


def _build_resolution_ctx(
    viewer: AuthContext | None,
    context: dict[str, str] | None,
) -> ResolutionContext | None:
    """Assemble a :class:`ResolutionContext` from the resolved viewer and
    compile-time context.

    ``viewer`` has already been run through ``MCPServer._resolve_viewer``,
    so it reflects the trusted provider when one is configured. Returns
    ``None`` when neither a viewer nor a context is present — so static
    lookups don't allocate an empty envelope, and dynamic loaders see the
    explicit "no context" signal."""
    if viewer is None and not context:
        return None
    return ResolutionContext(viewer=viewer, context=dict(context or {}))


def _error_payload(exc: Exception) -> dict[str, Any]:
    """Turn an exception into a structured tool response.

    The MCP client should be able to surface the failure mode to the
    planner; raising would just crash the tool call. ``code`` matches
    SemQL's error-leaf class names so callers can branch on them
    without parsing the message."""
    return {
        "error": {
            "code": type(exc).__name__,
            "message": str(exc),
        }
    }


def _make_query_cube_tool(
    cube: Cube,
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
) -> Callable[..., dict[str, Any]]:
    """Build a per-cube ``query_<cube>`` tool function.

    The returned function has ``__name__`` set to ``query_<cube_name>``,
    ``__doc__`` set to the cube's description, and ``__annotations__``
    set to typed signatures whose ``measures`` / ``dimensions`` /
    ``time_window.dimension`` are ``Literal``-typed enums of the cube's
    actual field names. FastMCP reads those via ``inspect.signature``
    to generate a JSON Schema with the enum constraint.

    The body auto-prefixes the bare field names with ``cube.name.`` so
    the planner doesn't have to repeat the cube name."""
    cube_name = cube.name
    measure_names = tuple(m.name for m in cube.measures)
    dimension_names = tuple(d.name for d in cube.dimensions)
    time_dim_names = tuple(td.name for td in cube.time_dimensions)
    field_names = (*measure_names, *dimension_names, *time_dim_names)

    # Build Literal types at runtime. ``Literal[("a", "b")]`` syntax is
    # supported in Python 3.11+ via the subscription protocol. The
    # types are attached to the function's ``__annotations__`` below —
    # the ``def`` itself can't reference them directly because this
    # module uses ``from __future__ import annotations`` (annotations
    # would be unresolvable string forms).
    measure_t = list[Literal[measure_names]] if measure_names else list[str]  # type: ignore[valid-type]
    dim_t = list[Literal[dimension_names]] if dimension_names else list[str]  # type: ignore[valid-type]
    field_t = Literal[field_names] if field_names else str
    order_t = list[tuple[field_t, Literal["asc", "desc"]]]  # type: ignore[valid-type]

    def query_cube_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202 — signature attached via __annotations__ below
        measures=None,  # noqa: ANN001
        dimensions=None,  # noqa: ANN001
        filters=None,  # noqa: ANN001
        time_window=None,  # noqa: ANN001
        having=None,  # noqa: ANN001
        order=None,  # noqa: ANN001
        limit=None,  # noqa: ANN001
        offset=None,  # noqa: ANN001
        ungrouped=False,  # noqa: ANN001
        context=None,  # noqa: ANN001
    ):
        try:
            spec = SemanticQuery(
                measures=[f"{cube_name}.{m}" for m in (measures or [])],
                dimensions=[f"{cube_name}.{d}" for d in (dimensions or [])],
                filters=[
                    Filter(
                        dimension=_prefix(f.dimension, cube_name),
                        op=f.op,
                        values=f.values,
                    )
                    for f in (filters or [])
                ],
                time_dimension=_prefix_time_window(time_window, cube_name),
                having=[
                    Filter(dimension=h.dimension, op=h.op, values=h.values) for h in (having or [])
                ],
                order=[(o[0], o[1]) for o in (order or [])],
                limit=limit,
                offset=offset,
                ungrouped=ungrouped,
            )
        except Exception as exc:
            return _error_payload(exc)
        try:
            compiled = catalog.compile(spec, context=context, viewer=resolve_viewer())
        except Exception as exc:
            return _error_payload(exc)
        envelope: dict[str, Any] = {
            "backend": compiled.backend.value,
            "sql": compiled.sql,
            "params": compiled.params,
            "columns": compiled.columns,
            "column_meta": [asdict(m) for m in compiled.column_meta],
        }
        if executor is None:
            return envelope
        try:
            envelope["rows"] = executor(compiled.sql, compiled.params)
        except Exception as exc:
            return _error_payload(exc) | envelope
        return envelope

    query_cube_fn.__name__ = f"query_{cube_name}"
    # Render via the shared projection helper so the MCP tool docstring
    # and ``project_tool_descriptions`` can't drift apart — both go
    # through ``semql_prompt.render_tool_description``.
    from semql_prompt import render_tool_description

    query_cube_fn.__doc__ = render_tool_description(cube)
    query_cube_fn.__annotations__ = {
        "measures": measure_t | None,
        "dimensions": dim_t | None,
        "filters": list[Filter] | None,
        "time_window": TimeWindow | None,
        "having": list[Filter] | None,
        "order": order_t | None,
        "limit": int | None,
        "offset": int | None,
        "ungrouped": bool,
        "context": dict[str, str] | None,
        "return": dict[str, Any],
    }
    return query_cube_fn


def _make_saved_query_tool(
    sq: Any,  # noqa: ANN401 — semql.SavedQuery (imported below at runtime)
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
) -> Callable[..., dict[str, Any]]:
    """Build a zero-arg ``saved_<name>`` tool that compiles + executes
    a pre-baked SemanticQuery.

    Optional ``context`` kwarg threads through to ``catalog.compile``
    for tenant / schema substitution — the saved query itself is
    static but its tenancy still needs runtime binding. When the
    server was constructed with an executor, the tool also returns
    ``rows``; otherwise it's compile-only."""
    saved_name = sq.name

    def saved_query_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202 — signature attached via __annotations__ below
        context=None,  # noqa: ANN001
    ):
        try:
            compiled = catalog.compile(sq.query, context=context, viewer=resolve_viewer())
        except Exception as exc:
            return _error_payload(exc)
        envelope: dict[str, Any] = {
            "backend": compiled.backend.value,
            "sql": compiled.sql,
            "params": compiled.params,
            "columns": compiled.columns,
            "column_meta": [asdict(m) for m in compiled.column_meta],
        }
        if executor is None:
            return envelope
        try:
            envelope["rows"] = executor(compiled.sql, compiled.params)
        except Exception as exc:
            return _error_payload(exc) | envelope
        return envelope

    saved_query_fn.__name__ = f"saved_{saved_name}"
    base_doc = sq.description or f"Run the saved query named {saved_name!r}."
    saved_query_fn.__doc__ = (
        base_doc + "\n\nThis tool runs a pre-baked SemanticQuery — no measure / "
        "dimension arguments needed. Pass ``context`` only if the "
        "underlying cubes use tenancy / schema placeholders."
    )
    saved_query_fn.__annotations__ = {
        "context": dict[str, str] | None,
        "return": dict[str, Any],
    }
    return saved_query_fn


def _prefix(name: str, cube_name: str) -> str:
    """Auto-prefix a bare field name with the cube name.

    If the caller already qualified the name (``orders.region``), pass
    it through unchanged so cross-cube references in filters/having
    still work."""
    if "." in name:
        return name
    return f"{cube_name}.{name}"


def _prefix_time_window(tw: TimeWindow | None, cube_name: str) -> TimeWindow | None:
    if tw is None:
        return None
    return TimeWindow(
        dimension=_prefix(tw.dimension, cube_name),
        granularity=tw.granularity,
        range=tw.range,
    )


__all__ = ["Executor", "MCPServer", "ViewerProvider"]
