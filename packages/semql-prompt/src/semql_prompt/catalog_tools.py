"""Catalog-level prompt conveniences.

Each takes the catalog as its first argument and reads it through the
public surface (``catalog.as_dict()``, ``catalog.policy``,
``catalog.views`` / ``.lookups`` / ``.glossary`` / ``.relations`` /
``.saved_queries``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from semql.model import AuthContext, ResolutionContext

from semql_prompt.prompt import (
    CatalogPrompt,
    build_planner_prompt_segments,
    catalog_prompt_hash,
    project_tool_descriptions,
    render_saved_query_tool_description,
    render_tool_description,
    to_openai_function,
)

if TYPE_CHECKING:
    from semql import Catalog
    from semql.hooks import CubePromptHook
    from semql.retrieve import Retriever


def planner_prompt(
    catalog: Catalog,
    *,
    only_exposed: bool = True,
    include_introspection: bool = False,
    viewer: AuthContext | None = None,
    ctx: ResolutionContext | None = None,
    user_query: str | None = None,
    retriever: Retriever | None = None,
    top_k: int = 10,
    retrieval_threshold: int = 50,
    current_date: str | None = None,
    retrieved_snippets: list[str] | None = None,
    extra: str | None = None,
    cube_prompt_hooks: list[CubePromptHook] | None = None,
) -> str:
    """Render the planner prompt fragment for ``catalog``.

    When ``viewer`` is provided, the catalog block shrinks to the cubes
    the viewer is allowed to see. ``ctx`` is the resolution context for
    dimension-value lookups. Retrieval mode (S7) narrows the block to the
    top-``top_k`` cubes when ``user_query`` + ``retriever`` are set and the
    catalog exceeds ``retrieval_threshold`` questions."""
    segments = build_planner_prompt_segments(
        catalog.as_dict(),
        only_exposed=only_exposed,
        include_introspection=include_introspection,
        views=catalog.views,
        viewer=viewer,
        policy=catalog.policy,
        lookups=catalog.lookups,
        ctx=ctx,
        glossary=catalog.glossary,
        relations=catalog.relations,
        user_query=user_query,
        retriever=retriever,
        top_k=top_k,
        retrieval_threshold=retrieval_threshold,
        saved_queries=list(catalog.saved_queries.values()),
        cube_prompt_hooks=cube_prompt_hooks,
    )
    return segments.full(
        current_date=current_date,
        retrieved_snippets=retrieved_snippets,
        extra=extra,
    )


def planner_prompt_segments(
    catalog: Catalog,
    *,
    only_exposed: bool = True,
    include_introspection: bool = False,
    viewer: AuthContext | None = None,
    ctx: ResolutionContext | None = None,
    user_query: str | None = None,
    retriever: Retriever | None = None,
    top_k: int = 10,
    retrieval_threshold: int = 50,
) -> CatalogPrompt:
    """Render the planner prompt as a cacheable two-segment object — a
    viewer-invariant ``static`` segment plus a per-viewer ``overlay``."""
    return build_planner_prompt_segments(
        catalog.as_dict(),
        only_exposed=only_exposed,
        include_introspection=include_introspection,
        views=catalog.views,
        viewer=viewer,
        policy=catalog.policy,
        lookups=catalog.lookups,
        ctx=ctx,
        glossary=catalog.glossary,
        relations=catalog.relations,
        user_query=user_query,
        retriever=retriever,
        top_k=top_k,
        retrieval_threshold=retrieval_threshold,
        saved_queries=list(catalog.saved_queries.values()),
    )


def prompt_hash(
    catalog: Catalog,
    *,
    only_exposed: bool = True,
    ctx: ResolutionContext | None = None,
) -> str:
    """SHA256 hex digest of the static (viewer-invariant) prompt segment —
    stable across viewer changes, so it keys a prompt-fragment cache that a
    catalog mutation invalidates."""
    return catalog_prompt_hash(
        catalog.as_dict(),
        only_exposed=only_exposed,
        lookups=catalog.lookups,
        ctx=ctx,
        glossary=catalog.glossary,
        relations=catalog.relations,
    )


def to_openai_tools(
    catalog: Catalog,
    *,
    viewer: AuthContext | None = None,
    only_exposed: bool = True,
) -> list[dict[str, Any]]:
    """One OpenAI-format tool dict per visible cube + saved query.
    Role-gated cubes / saved queries are excluded unless ``viewer`` holds a
    matching role."""
    from semql.introspect import iter_cubes
    from semql.spec import SemanticQuery

    by_name = catalog.as_dict()
    proj = project_tool_descriptions(
        by_name,
        only_exposed=only_exposed,
        viewer=viewer,
        policy=catalog.policy,
    )
    visible_cubes = {**proj.invariant, **proj.viewer_gated}

    tools: list[dict[str, Any]] = [
        to_openai_function(c)
        for c in iter_cubes(
            by_name,
            include_meta=False,
            only_exposed=only_exposed,
            viewer=None,  # proj already determined visibility
            policy=None,
        )
        if c.name in visible_cubes
    ]

    for sq in catalog.saved_queries.values():
        if sq.required_roles and (
            viewer is None or not any(r in viewer.roles for r in sq.required_roles)
        ):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"saved_{sq.name}",
                    "description": render_saved_query_tool_description(sq),
                    "parameters": SemanticQuery.model_json_schema(),
                },
            }
        )
    return tools


def to_langchain_tools(
    catalog: Catalog,
    *,
    viewer: AuthContext | None = None,
    only_exposed: bool = True,
) -> list[object]:
    """One LangChain ``StructuredTool`` per visible cube. Requires
    ``langchain-core``; each tool compiles and returns
    ``{"sql": ..., "params": ...}``."""
    from typing import cast

    try:
        from langchain_core.tools import StructuredTool  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "langchain-core is required for to_langchain_tools(). "
            "Install it with: pip install langchain-core"
        ) from None
    structured_tool_cls = cast(Any, StructuredTool)

    from semql.introspect import iter_cubes
    from semql.spec import SemanticQuery

    cubes = list(
        iter_cubes(
            catalog.as_dict(),
            include_meta=False,
            only_exposed=only_exposed,
            viewer=viewer,
            policy=catalog.policy,
        )
    )
    tools: list[object] = []
    for cube in cubes:
        _cube_ref = cube

        def _run(query: SemanticQuery, *, _c: object = _cube_ref) -> dict[str, object]:
            compiled = catalog.compile(query, viewer=viewer)
            return {"sql": compiled.sql, "params": compiled.params}

        tools.append(
            structured_tool_cls.from_function(
                func=_run,
                name=f"query_{cube.name}",
                description=render_tool_description(cube),
                args_schema=SemanticQuery,
            )
        )
    return tools
