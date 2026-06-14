"""JSON-Schema post-processing for tool-calling projections.

Pydantic emits a *root* ``$ref`` whenever the root model is recursive — which
:class:`~semql.spec.SemanticQuery` is (``semi_joins[].source`` is itself a
``SemanticQuery``), and which a self-referential :class:`~semql.spec.BoolExpr`
exported as a tool root would also produce. A bare root ``$ref`` is not an
object-rooted schema, and OpenAI / Anthropic / Bedrock tool-calling all expect
``{"type": "object", "properties": {...}}`` at the top (Bedrock Converse
rejects a root ``$ref`` on every model family — verified against the live API,
2026-06).

:func:`flatten_root_ref` splices the referenced definition up to the root while
keeping ``$defs`` and every *internal* (recursive) ``$ref`` intact, so the
self-reference still resolves. Internal refs, ``$defs``, and ``prefixItems``
tuples are left untouched — only the root is rewritten. It is a no-op (a deep
copy) on an already object-rooted schema, and raises if the schema cannot be
made object-rooted (a ``RootModel`` over a union or scalar), since such a model
cannot be a tool input.

This lives in core because the recursion is a property of a core type;
:meth:`SemanticQuery.tool_json_schema` is the front door, and the prompt /
Bedrock projection layers reuse :func:`flatten_root_ref` for arbitrary schemas.
"""

from __future__ import annotations

import copy
from typing import Any, cast


def flatten_root_ref(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` whose top level is an object schema.

    If the root is a ``$ref`` into ``$defs`` (Pydantic's output for a recursive
    root model), splice the referenced definition's body up to the top level —
    keeping ``$defs`` intact so internal/recursive ``$ref``\\ s still resolve.
    Sibling keywords sitting next to the root ``$ref`` (``description``,
    ``default``, …) are preserved and win over the spliced body. If the root is
    already an object schema, it is returned unchanged (deep-copied).

    Raises:
        ValueError: if the schema cannot be made object-rooted — e.g. a
            ``RootModel`` over a union or scalar whose top level is ``anyOf`` or
            a non-object ``type``. Fail loudly rather than ship a schema the
            tool-calling API will reject at request time.
    """
    schema = copy.deepcopy(schema)
    ref = schema.get("$ref")
    if ref is not None:
        defs = schema.get("$defs")
        target = _resolve_local_ref(ref, schema)
        # Keywords beside the root $ref override the spliced body (JSON Schema
        # 2020-12 allows $ref siblings; Pydantic uses them for default/title).
        siblings = {k: v for k, v in schema.items() if k not in ("$ref", "$defs")}
        schema = {**target, **siblings}
        if defs is not None:
            # Retain $defs untouched — the spliced body (and any recursive
            # node) still references entries inside it.
            schema["$defs"] = defs
    if schema.get("type") != "object":
        raise ValueError(
            "flatten_root_ref: schema is not object-rooted after flattening "
            f"(top-level type={schema.get('type')!r}, keys={sorted(schema)}). "
            "Tool-calling APIs require an inputSchema whose root is "
            "type='object'; a RootModel over a union or scalar cannot be a "
            "tool input."
        )
    return schema


def _resolve_local_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local JSON-pointer ``$ref`` (e.g. ``#/$defs/Name``) in ``root``."""
    if not ref.startswith("#/"):
        raise ValueError(f"flatten_root_ref: only local '#/...' refs are supported, got {ref!r}.")
    node: Any = root
    for raw in ref[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")  # RFC 6901 unescape
        try:
            node = node[token]
        except (KeyError, TypeError):
            raise ValueError(f"flatten_root_ref: cannot resolve ref {ref!r}.") from None
    if not isinstance(node, dict):
        raise ValueError(f"flatten_root_ref: ref {ref!r} does not point at a schema object.")
    return cast("dict[str, Any]", node)


__all__ = ["flatten_root_ref"]
