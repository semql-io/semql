"""Bedrock Converse tool-schema adaptation.

Bedrock's Converse API validates every tool ``inputSchema`` and requires the
TOP LEVEL to be an object schema carrying ``type: "object"``. A bare root
``$ref`` is rejected on every model family — Nova, Claude, Llama, Mistral,
Qwen, and gpt-oss all return the identical
``inputSchema.json.type must be one of the following: object`` error (verified
empirically against the live Converse API, 2026-06). The constraint lives in
the Converse tool layer, above the model, so it applies regardless of which
model you call.

Pydantic emits a root ``$ref`` whenever the *root* model is recursive (e.g. a
self-referential :class:`~semql.spec.BoolExpr`). Internal ``$ref`` / ``$defs``
— including recursive cycles — are accepted fine, so this module rewrites ONLY
the root and leaves the rest of the schema (nested refs, ``$defs``,
``prefixItems`` tuples) untouched. Fully inlining ``$defs`` would be both
unnecessary and impossible for a genuine recursive cycle.

:meth:`SemanticQuery.model_json_schema() <semql.spec.SemanticQuery>` is already
object-rooted, so :func:`flatten_root_ref` is a no-op for it; the rewrite bites
only when a recursive model is exported directly as a tool root, and the guard
keeps that from regressing into a silently-rejected schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# The root-$ref flattener lives in core (``semql.flatten_root_ref``) — its
# single public home, since the recursion is a property of a core type. Imported
# here only for ``to_bedrock_converse_tools``'s defensive re-flatten.
from semql import flatten_root_ref

from semql_prompt.catalog_tools import to_openai_tools

if TYPE_CHECKING:
    from semql import Catalog
    from semql.model import AuthContext


def to_bedrock_converse_tools(
    catalog: Catalog,
    *,
    viewer: AuthContext | None = None,
    only_exposed: bool = True,
) -> list[dict[str, Any]]:
    """One Bedrock Converse ``toolSpec`` per visible cube + saved query.

    Returns dicts shaped for ``toolConfig["tools"]`` in
    ``bedrock-runtime.converse(...)``::

        {"toolSpec": {"name": ..., "description": ...,
                      "inputSchema": {"json": <object-rooted schema>}}}

    Built by reshaping :func:`~semql_prompt.catalog_tools.to_openai_tools` — so
    cube / saved-query visibility, role gating, and descriptions stay defined in
    one place — and running each parameter schema through
    :func:`flatten_root_ref` for Converse root-``$ref`` compatibility.
    """
    tools: list[dict[str, Any]] = []
    for tool in to_openai_tools(catalog, viewer=viewer, only_exposed=only_exposed):
        fn = tool["function"]
        tools.append(
            {
                "toolSpec": {
                    "name": fn["name"],
                    "description": fn["description"],
                    "inputSchema": {"json": flatten_root_ref(fn["parameters"])},
                }
            }
        )
    return tools


__all__ = ["to_bedrock_converse_tools"]
