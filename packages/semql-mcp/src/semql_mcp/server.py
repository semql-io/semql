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

Always registered, marked **BETA** (see :data:`semql_mcp.viz.VIZ_BETA_NOTICE`):
- ``query_visualize(spec, n_rows=0, shape_stats=None, supported_charts=None, context=None)`` —
  compile a ``SemanticQuery``, run it through the visualiser
  (optionally executing against the configured executor), and
  return a :class:`semql.VizDecision` payload the
  ``ui://semql/chart`` iframe renders. Registered with
  ``AppConfig(visibility=["app"])`` so only MCP Apps-aware hosts
  see it; a plain chat client doesn't get the affordance.

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
from semql.compile import CompiledQuery
from semql.errors import AuthError, SemQLError
from semql.lookups import materialize as materialize_lookup
from semql.lookups import resolve as resolve_lookup
from semql.model import AuthContext, Cube, Entity, MutableEntity, Op, ResolutionContext
from semql.mutate import SemanticMutation
from semql.rows import EntityFetch, EntityList
from semql.safe import is_read_only_statement
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

This is the fix for the multi-tenant auth hole. On a networked
transport (http / sse / streamable-http) the *client* cannot be trusted
to assert who it is, so tool parameters like ``viewer_id`` / ``roles`` are
not an authorization boundary. The deployer wires a ``viewer_provider``
that derives the verified identity from the transport's authenticated
request context (a validated bearer token, an mTLS client cert, a session)
and the server threads its result into every ``catalog.compile(viewer=...)``
call — so ``required_roles`` cube/field visibility and ``security_sql``
row-level scoping are actually enforced.

The provider is invoked inside each tool call, so it sees the current
request's context. Returning ``None`` means "no identity"; by default
(``require_viewer=True``) that is a hard refusal — the tool returns an
``AuthError`` payload rather than silently granting unscoped access.
Do **not** conflate "auth failed" with "anonymous is fine": deny by
returning ``None`` (refused) or raising ``AuthError``; allow anonymous
(unscoped) access only by constructing the server with
``require_viewer=False``. When a provider is configured it is
authoritative: client-asserted ``viewer_id`` / ``roles`` are ignored.
With no provider (the stdio single-tenant default) the server falls back
to the client-asserted values, which is the only mode where trusting
them is safe."""


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
        require_viewer: bool = True,
        debug: bool = False,
        name: str = "semql",
        generic_entity_tools: bool = False,
    ) -> None:
        self.catalog = catalog
        self.executor = executor
        self.viewer_provider = viewer_provider
        self.require_viewer = require_viewer
        self.debug = debug
        self.generic_entity_tools = generic_entity_tools
        self.mcp = FastMCP(name=name)
        self._register_tools()
        self._register_per_cube_tools()
        self._register_lookup_tools()
        self._register_saved_query_tools()
        self._register_entity_tools()
        self._register_visualization_tools()

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
        values; that's the only context where trusting them is safe.

        Fail-secure: when a provider is configured and ``require_viewer``
        is set (the default), a provider returning ``None`` is a hard
        refusal — not a silent grant of unscoped access. Providers must
        signal "deny" by returning ``None`` (refused here) and may raise
        their own :class:`~semql.errors.AuthError`; "anonymous is OK" is
        an explicit opt-in via ``require_viewer=False``."""
        if self.viewer_provider is not None:
            viewer = self.viewer_provider()
            if viewer is None and self.require_viewer:
                raise AuthError(
                    "viewer_provider returned no identity and require_viewer "
                    "is set. Pass require_viewer=False to allow anonymous "
                    "(unscoped) access.",
                    reason="no_viewer",
                )
            return viewer
        if client_viewer_id is not None:
            return AuthContext(viewer_id=client_viewer_id, roles=list(client_roles or []))
        return None

    def _register_tools(self) -> None:
        catalog = self.catalog
        executor = self.executor
        resolve_viewer = self._resolve_viewer
        debug = self.debug

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
                return _error_payload(exc, debug=debug)
            return {
                "dialect": compiled.dialect.value,
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
                    return _error_payload(exc, debug=debug)
                try:
                    _guard_read_only(compiled)
                    rows = executor(compiled.sql, compiled.params)
                except Exception as exc:
                    return _error_payload(exc, debug=debug) | {
                        "sql": compiled.sql,
                        "params": compiled.params,
                    }
                return {
                    "dialect": compiled.dialect.value,
                    "sql": compiled.sql,
                    "params": compiled.params,
                    "columns": compiled.columns,
                    "column_meta": [asdict(m) for m in compiled.column_meta],
                    "rows": rows,
                }

    def _register_visualization_tools(self) -> None:
        """Wire ``query_visualize`` + the ``ui://semql/chart`` iframe.

        Both are registered via :func:`semql_mcp.viz.register_visualization_tools`
        to keep the MCP Apps integration in one place; the tool
        advertises ``app=AppConfig(resource_uri=CHART_RESOURCE_URI,
        visibility=["app"])`` so an MCP Apps-aware host pairs the tool
        with the iframe and a plain chat client (model-only) doesn't
        see the affordance.

        Marked **BETA** in the tool description and on the resource —
        the recommendation logic is stable, but the rendered shape
        and the per-host protocol details are still settling, so we
        want consumers to see the lifecycle state at a glance."""
        from semql_mcp.viz import register_visualization_tools

        register_visualization_tools(
            self.mcp,
            self.catalog,
            self._resolve_viewer,
            debug=self.debug,
            executor=self.executor,
        )

    def _register_lookup_tools(self) -> None:
        """Register ``resolve_lookup`` + ``list_lookup_values`` when the
        catalog carries any :class:`semql.model.Lookup`.

        Skipped on empty-lookup catalogs so a server with no resolvable
        dimensions doesn't advertise misleading tools."""
        catalog = self.catalog
        resolve_viewer = self._resolve_viewer
        debug = self.debug
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
                return _error_payload(exc, debug=debug)
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
                return _error_payload(exc, debug=debug)
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
            self.mcp.add_tool(
                _make_saved_query_tool(sq, catalog, executor, self._resolve_viewer, self.debug)
            )

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
            self.mcp.add_tool(
                _make_query_cube_tool(cube, catalog, executor, self._resolve_viewer, self.debug)
            )

    def _register_entity_tools(self) -> None:
        """Register row-mode entity tools (entities spec M5).

        Per-entity by default: ``get_<entity>`` + ``list_<entity>`` for
        every entity with a key, plus ``mutate_<entity>`` for each
        :class:`~semql.model.MutableEntity` — the latter only when the
        catalog opted into ``allow_mutations`` (gate 1). Set
        ``generic_entity_tools=True`` to collapse to three generic tools
        for very large catalogs (D6). The per-call role gate (gate 3) is
        enforced inside the compiler, so a viewer who can't see the cube
        gets an AuthError payload rather than rows."""
        catalog = self.catalog
        if not catalog.entities:
            return
        if self.generic_entity_tools:
            self._register_generic_entity_tools()
            return
        for entity in catalog.entities.values():
            if entity.key is not None:
                self.mcp.add_tool(
                    _make_get_entity_tool(
                        entity, catalog, self.executor, self._resolve_viewer, self.debug
                    )
                )
                self.mcp.add_tool(
                    _make_list_entity_tool(
                        entity, catalog, self.executor, self._resolve_viewer, self.debug
                    )
                )
            if isinstance(entity, MutableEntity) and catalog.allow_mutations:
                self.mcp.add_tool(
                    _make_mutate_entity_tool(
                        entity, catalog, self.executor, self._resolve_viewer, self.debug
                    )
                )

    def _register_generic_entity_tools(self) -> None:
        catalog = self.catalog
        executor = self.executor
        resolve_viewer = self._resolve_viewer
        debug = self.debug

        @self.mcp.tool(
            name="get_entity",
            description="Fetch one row of an entity by key. Pass the entity name and its key.",
        )
        def get_entity(
            entity: str,
            key: str | int,
            fields: list[str] | None = None,
            context: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            return _run_entity_read(
                lambda: catalog.fetch(
                    EntityFetch(entity=entity, key=key, fields=fields),
                    context=context,
                    viewer=resolve_viewer(),
                ),
                executor,
                debug,
            )

        @self.mcp.tool(
            name="list_entity",
            description="List rows of an entity through its allowlisted filters.",
        )
        def list_entity(
            entity: str,
            where: dict[str, Any] | None = None,
            time_range: tuple[str, str, str] | None = None,
            order: str | None = None,
            limit: int = 50,
            cursor: str | None = None,
            context: dict[str, str] | None = None,
        ) -> dict[str, Any]:
            return _run_entity_read(
                lambda: catalog.list_rows(
                    EntityList(
                        entity=entity,
                        where=where or {},
                        time_range=time_range,
                        order=order,
                        limit=limit,
                        cursor=cursor,
                    ),
                    context=context,
                    viewer=resolve_viewer(),
                ),
                executor,
                debug,
            )

        if catalog.allow_mutations:

            @self.mcp.tool(
                name="mutate",
                description=(
                    "Mutate an entity. Two-step: confirm=false (default) previews "
                    "the affected rows; confirm=true executes the DML."
                ),
            )
            def mutate(
                entity: str,
                operation: Op,
                values: dict[str, Any] | None = None,
                pk: dict[str, Any] | None = None,
                where: dict[str, Any] | None = None,
                confirm: bool = False,
                context: dict[str, str] | None = None,
            ) -> dict[str, Any]:
                return _run_entity_mutation(
                    SemanticMutation(
                        entity=entity,
                        operation=operation,
                        values=values or {},
                        pk=pk,
                        where=where,
                    ),
                    catalog,
                    executor,
                    resolve_viewer(),
                    context,
                    confirm,
                    debug,
                )

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


class ReadOnlyError(Exception):
    """Compiled SQL failed the pre-execution read-only guard.

    Raised at the execution choke point when ``compiled.sql`` (or one of
    its ``derived_sources``) isn't a single read-only SELECT. The happy
    path never trips this — the compiler emits SELECT by construction —
    but RawSQL escape hatches (``DerivedTable.sql``, ``with_ctes``,
    ``security_sql``, ``ScopePredicate.sql``) splice author-controlled
    strings in, so we re-check before anything reaches the driver."""


def _guard_read_only(compiled: CompiledQuery) -> None:
    """Refuse to execute anything that isn't a read-only SELECT.

    Defense-in-depth per PHILOSOPHY: "the defensive guarantee reaching
    the LLM consumer is implemented in the recipe." This is that recipe
    step for the row-returning tools."""
    for sql in (compiled.sql, *compiled.derived_sources):
        if not is_read_only_statement(sql, dialect=compiled.dialect.value):
            raise ReadOnlyError("Compiled SQL is not a read-only SELECT; refusing to execute.")


def _error_payload(exc: Exception, *, debug: bool = False) -> dict[str, Any]:
    """Turn an exception into a structured tool response.

    The MCP client should be able to surface the failure mode to the
    planner; raising would just crash the tool call. ``code`` matches
    SemQL's error-leaf class names so callers can branch on them
    without parsing the message.

    Trust boundary: SemQL's own structured errors (and the read-only
    guard) carry planner-facing messages by construction — they name
    catalog fields, never raw rows — so they pass through verbatim. An
    *arbitrary* executor / driver exception (``RuntimeError`` from a
    DB-API call, a psycopg ``ProgrammingError``, …) can leak table /
    column names or even row data in ``str(exc)``, so by default it is
    reduced to a generic ``ExecutionError`` message. Construct the
    server with ``debug=True`` to surface the raw text for local
    troubleshooting."""
    if isinstance(exc, SemQLError):
        return {"error": exc.to_payload()}
    if isinstance(exc, ReadOnlyError) or debug:
        return {"error": {"code": type(exc).__name__, "message": str(exc)}}
    return {
        "error": {
            "code": "ExecutionError",
            "message": "Execution failed. Enable server debug mode to see details.",
        }
    }


def _make_query_cube_tool(
    cube: Cube,
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
    debug: bool = False,
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
            return _error_payload(exc, debug=debug)
        try:
            compiled = catalog.compile(spec, context=context, viewer=resolve_viewer())
        except Exception as exc:
            return _error_payload(exc, debug=debug)
        envelope: dict[str, Any] = {
            "dialect": compiled.dialect.value,
            "sql": compiled.sql,
            "params": compiled.params,
            "columns": compiled.columns,
            "column_meta": [asdict(m) for m in compiled.column_meta],
        }
        if executor is None:
            return envelope
        try:
            _guard_read_only(compiled)
            envelope["rows"] = executor(compiled.sql, compiled.params)
        except Exception as exc:
            return _error_payload(exc, debug=debug) | envelope
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


def _py_type_for(dim_type: str) -> type:
    """Map a SemQL dimension type to the Python type a tool param uses."""
    if dim_type == "number":
        return float
    if dim_type == "bool":
        return bool
    return str


def _entity_output_fields(entity: Entity, catalog: Catalog) -> list[str]:
    if entity.fields:
        return list(entity.fields)
    primary = catalog.as_dict()[entity.cubes[0]]
    return [d.name for d in primary.dimensions]


def _run_entity_read(
    compile_fn: Callable[[], Any],
    executor: Executor | None,
    debug: bool,
) -> dict[str, Any]:
    """Compile (and optionally execute) a row-mode read, returning the
    envelope. Row execution is guarded read-only — entity reads are
    SELECTs by construction, but the guard is defence-in-depth."""
    try:
        compiled = compile_fn()
    except Exception as exc:
        return _error_payload(exc, debug=debug)
    envelope: dict[str, Any] = {
        "sql": compiled.sql,
        "params": compiled.params,
        "columns": compiled.columns,
        "plan": compiled.plan.model_dump(),
    }
    if executor is None or compiled.sql is None:
        return envelope
    try:
        if not is_read_only_statement(compiled.sql, dialect=compiled.plan.source.backend):
            raise ReadOnlyError("Compiled entity SQL is not a read-only SELECT; refusing.")
        envelope["rows"] = executor(compiled.sql, compiled.params)
    except Exception as exc:
        return _error_payload(exc, debug=debug) | envelope
    return envelope


def _run_entity_mutation(
    mutation: SemanticMutation,
    catalog: Catalog,
    executor: Executor | None,
    viewer: AuthContext | None,
    context: dict[str, str] | None,
    confirm: bool,
    debug: bool,
) -> dict[str, Any]:
    """Two-step mutation (§5 confirm loop). ``confirm=False`` (default)
    runs only the preview SELECT and reports the affected count; it never
    touches the DML. ``confirm=True`` re-checks the count against the cap
    (closing the TOCTOU window, A.4.4) and then executes the DML."""
    try:
        compiled = catalog.mutate(mutation, context=context, viewer=viewer)
    except Exception as exc:
        return _error_payload(exc, debug=debug)
    envelope: dict[str, Any] = {
        "operation": compiled.operation.value,
        "sql": compiled.sql,
        "preview_sql": compiled.preview_sql,
        "affects": compiled.affects,
    }
    if executor is None:
        # Compile-only mode: hand back the DML + preview for the caller to run.
        return envelope | {"confirmed": False, "executed": False}
    try:
        preview_rows = executor(compiled.preview_sql, compiled.preview_params)
    except Exception as exc:
        return _error_payload(exc, debug=debug) | envelope
    affected = len(preview_rows)
    cap = compiled.max_affected_rows
    if cap is not None and affected > cap:
        return envelope | {
            "error": {
                "code": "MutationCapExceeded",
                "message": (
                    f"Mutation would affect {affected} rows, exceeding the cap of "
                    f"{cap}. Narrow the target or raise max_mutation_rows."
                ),
            },
            "affected_rows": affected,
            "executed": False,
        }
    if not confirm:
        return envelope | {
            "confirmed": False,
            "executed": False,
            "affected_rows": affected,
            "preview_rows": preview_rows,
        }
    try:
        executor(compiled.sql, compiled.params)
    except Exception as exc:
        return _error_payload(exc, debug=debug) | envelope
    return envelope | {"confirmed": True, "executed": True, "affected_rows": affected}


def _make_get_entity_tool(
    entity: Entity,
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
    debug: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build a ``get_<entity>`` point-lookup tool. ``fields`` is a
    ``Literal`` enum of the entity's output columns."""
    name = entity.name
    field_names = tuple(_entity_output_fields(entity, catalog))
    fields_t = list[Literal[field_names]] if field_names else list[str]  # type: ignore[valid-type]

    def get_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202
        key,  # noqa: ANN001
        fields=None,  # noqa: ANN001
        context=None,  # noqa: ANN001
    ):
        return _run_entity_read(
            lambda: catalog.fetch(
                EntityFetch(entity=name, key=key, fields=fields),
                context=context,
                viewer=resolve_viewer(),
            ),
            executor,
            debug,
        )

    get_fn.__name__ = f"get_{name}"
    get_fn.__doc__ = entity.description or f"Fetch one {name} by its key."
    get_fn.__annotations__ = {
        "key": str | int,
        "fields": fields_t | None,
        "context": dict[str, str] | None,
        "return": dict[str, Any],
    }
    return get_fn


def _make_list_entity_tool(
    entity: Entity,
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
    debug: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build a ``list_<entity>`` tool with one typed parameter per
    allowlisted ``list_filter`` (so the JSON Schema reflects each filter's
    field type), plus limit / cursor / time_range / order."""
    import inspect

    name = entity.name
    by_name = catalog.as_dict()
    # local param name -> (qualified ref, python type)
    filter_map: dict[str, tuple[str, type]] = {}
    for ref in entity.list_filters:
        cube_name, _, dim_name = ref.partition(".")
        cube = by_name.get(cube_name)
        dim = next((d for d in cube.dimensions if d.name == dim_name), None) if cube else None
        py = _py_type_for(dim.type) if dim is not None else str
        filter_map[dim_name] = (ref, py)

    def list_fn(**kwargs: Any) -> dict[str, Any]:  # noqa: ANN401 — dynamic per-filter params via __signature__
        where: dict[str, Any] = {
            ref: kwargs[local]
            for local, (ref, _) in filter_map.items()
            if kwargs.get(local) is not None
        }
        return _run_entity_read(
            lambda: catalog.list_rows(
                EntityList(
                    entity=name,
                    where=where,
                    time_range=kwargs.get("time_range"),
                    order=kwargs.get("order"),
                    limit=kwargs.get("limit", 50),
                    cursor=kwargs.get("cursor"),
                ),
                context=kwargs.get("context"),
                viewer=resolve_viewer(),
            ),
            executor,
            debug,
        )

    # Annotations must agree with the synthesized signature: pydantic
    # (via FastMCP) reads parameter types out of ``__annotations__`` while
    # iterating ``inspect.signature``, so every signature param needs a
    # matching annotation entry or schema generation raises KeyError.
    annotations: dict[str, Any] = {local: py | None for local, (_, py) in filter_map.items()}
    annotations |= {
        "time_range": tuple[str, str, str] | None,
        "order": str | None,
        "limit": int,
        "cursor": str | None,
        "context": dict[str, str] | None,
        "return": dict[str, Any],
    }
    defaults: dict[str, Any] = {"limit": 50}
    params = [
        inspect.Parameter(
            pname,
            inspect.Parameter.KEYWORD_ONLY,
            default=defaults.get(pname),
            annotation=annotations[pname],
        )
        for pname in annotations
        if pname != "return"
    ]
    list_fn.__signature__ = inspect.Signature(params, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    list_fn.__annotations__ = annotations
    list_fn.__name__ = f"list_{name}"
    list_fn.__doc__ = entity.description or f"List {name} rows through allowlisted filters."
    return list_fn


def _make_mutate_entity_tool(
    entity: MutableEntity,
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
    debug: bool = False,
) -> Callable[..., dict[str, Any]]:
    """Build a ``mutate_<entity>`` two-step tool. ``operation`` is a
    ``Literal`` enum of the entity's permitted operations; the docstring
    lists the mutable fields and their types."""
    name = entity.name
    ops = tuple(sorted(o.value for o in entity.operations))
    op_t = Literal[ops]  # type: ignore[valid-type]
    field_doc = ", ".join(f"{n}: {f.type}" for n, f in entity.mutable_fields.items())

    def mutate_fn(  # type: ignore[no-untyped-def]  # noqa: ANN202
        operation,  # noqa: ANN001
        values=None,  # noqa: ANN001
        pk=None,  # noqa: ANN001
        where=None,  # noqa: ANN001
        confirm=False,  # noqa: ANN001
        context=None,  # noqa: ANN001
    ):
        try:
            mutation = SemanticMutation(
                entity=name,
                operation=Op(operation),
                values=values or {},
                pk=pk,
                where=where,
            )
        except Exception as exc:
            return _error_payload(exc, debug=debug)
        return _run_entity_mutation(
            mutation, catalog, executor, resolve_viewer(), context, confirm, debug
        )

    mutate_fn.__name__ = f"mutate_{name}"
    mutate_fn.__doc__ = (
        f"Mutate a {name}. Mutable fields — {field_doc}. Two-step: confirm=false "
        "(default) previews the affected rows; confirm=true executes the DML."
    )
    mutate_fn.__annotations__ = {
        "operation": op_t,
        "values": dict[str, Any] | None,
        "pk": dict[str, Any] | None,
        "where": dict[str, Any] | None,
        "confirm": bool,
        "context": dict[str, str] | None,
        "return": dict[str, Any],
    }
    return mutate_fn


def _make_saved_query_tool(
    sq: Any,  # noqa: ANN401 — semql.SavedQuery (imported below at runtime)
    catalog: Catalog,
    executor: Executor | None,
    resolve_viewer: ViewerProvider,
    debug: bool = False,
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
        viewer = resolve_viewer()
        if sq.required_roles:
            viewer_roles = viewer.roles if viewer is not None else []
            if not any(r in viewer_roles for r in sq.required_roles):
                return _error_payload(
                    AuthError(
                        f"Saved query {saved_name!r} requires role(s) "
                        f"{sorted(sq.required_roles)!r}.",
                        reason="forbidden",
                    ),
                    debug=debug,
                )
        try:
            compiled = catalog.compile(sq.query, context=context, viewer=viewer)
        except Exception as exc:
            return _error_payload(exc, debug=debug)
        envelope: dict[str, Any] = {
            "dialect": compiled.dialect.value,
            "sql": compiled.sql,
            "params": compiled.params,
            "columns": compiled.columns,
            "column_meta": [asdict(m) for m in compiled.column_meta],
        }
        if executor is None:
            return envelope
        try:
            _guard_read_only(compiled)
            envelope["rows"] = executor(compiled.sql, compiled.params)
        except Exception as exc:
            return _error_payload(exc, debug=debug) | envelope
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
