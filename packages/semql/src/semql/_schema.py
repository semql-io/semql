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

Some tool-use decoders are pickier than the Converse validation layer that
:func:`flatten_root_ref` targets: Bedrock Nova's decoder has been observed to
reject ``prefixItems`` (fixed-length tuple schemas) and any *recursive*
``$ref`` cycle among ``$defs`` — not just a bare root ``$ref`` — with a 424
``ModelErrorException``. :func:`to_llm_safe_schema` produces a stricter,
Nova-safe projection on top of an already object-rooted schema: it rewrites
every ``prefixItems`` node into a plain ``items``-typed array, then finds and
removes every ``$defs`` entry that participates in a reference cycle (via a
generic cycle search over ``$ref`` edges, not a hardcoded name list) along
with any property that reaches it, and finally drops any ``$defs`` entry left
unreachable by that removal. :meth:`SemanticQuery.llm_json_schema` is the
front door for this projection.
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


def to_llm_safe_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` with no ``prefixItems`` and no recursive ``$ref``.

    Intended to run on an already object-rooted schema (typically the output
    of :func:`flatten_root_ref` / ``tool_json_schema()``); it does not itself
    handle a root ``$ref``. Two rewrites, both generic — neither is hardcoded
    to a field or ``$defs`` name:

    1. Every node with ``prefixItems`` (a JSON Schema fixed-length tuple, e.g.
       Pydantic's rendering of ``tuple[str, str]``) is rewritten to a plain
       array node: ``prefixItems`` is dropped and ``items`` is set to the
       tuple's single item schema (if every position shares one), or an
       ``anyOf`` of the distinct position schemas otherwise. ``minItems`` /
       ``maxItems`` are left as-is, so arity is still documented (though no
       longer enforced positionally — the 2-tuple's field meanings become a
       documentation-only convention at this projection).
    2. ``$defs`` is treated as a directed graph of ``$ref`` edges; any
       ``$defs`` entry that can reach itself (a self-loop like ``BoolExpr``,
       or a longer cycle like ``SemanticQuery`` <-> ``SemiJoin``) is removed,
       along with every property — anywhere in the schema — whose value
       (directly, or through ``items`` / ``anyOf`` / ``oneOf`` / ``allOf``)
       references a removed def. Removing those properties can leave other
       ``$defs`` entries unreferenced from the root; those are pruned too.

    For :class:`~semql.spec.SemanticQuery` this drops ``where`` (the
    self-referential ``BoolExpr`` tree — OR/NOT predicates are not
    expressible via this projection; compose them before calling, or use
    ``tool_json_schema()`` instead if the target decoder tolerates recursive
    refs) and ``semi_joins`` (whose inner ``source: SemanticQuery`` closes a
    cycle back through the ``SemanticQuery`` def once ``where`` and
    ``BoolExpr`` are out of the picture) — the rest of the shape, notably
    ``filters``/``having``/``time_dimension``/``compare``/``order``/``limit``,
    is preserved.
    """
    schema = copy.deepcopy(schema)
    _inline_prefix_items(schema)
    raw_defs = schema.get("$defs")
    if isinstance(raw_defs, dict) and raw_defs:
        defs = cast("dict[str, Any]", raw_defs)
        edges = _ref_edges(defs)
        cyclic = _cyclic_def_names(edges)
        if cyclic:
            for name in cyclic:
                defs.pop(name, None)
            _strip_properties_referencing(schema, cyclic)
            _prune_unreachable_defs(schema, defs)
        if not defs:
            schema.pop("$defs", None)
    return schema


# A JSON-Schema subtree, recursively: an object (``dict[str, JSONNode]``), an
# array (``list[JSONNode]``), or a leaf scalar. ``Any`` at the leaves is
# unavoidable — schema nodes really are heterogeneous — but every helper
# below re-annotates a value the moment ``isinstance`` narrows it to
# ``dict``/``list``, so pyright (strict mode) never has to infer through a
# bare ``Any`` and degrade to ``Unknown``.
JSONNode = Any


def _as_list(node: Any) -> list[Any]:  # noqa: ANN401 — arbitrary JSON-schema subtree
    """Identity, typed. A bare ``cast("list[Any]", node)`` at each call site
    below is what pyright (strict mode) needs to stop inferring ``Unknown``
    element types past an ``isinstance(node, list)`` narrowing of an
    ``Any``-typed value — but mypy considers that particular cast a no-op
    (``list[Any]`` narrowed from ``Any`` already reads as ``list[Any]`` to
    it) and flags it ``redundant-cast``. Routing through a real function
    call sidesteps the disagreement: pyright treats the return type as
    authoritative same as a cast, and mypy has no cast expression to flag.
    """
    return cast("list[Any]", node)


def _inline_prefix_items(node: JSONNode) -> None:
    """Recursively rewrite every ``prefixItems`` tuple node into ``items``, in place."""
    if isinstance(node, dict):
        obj = cast("dict[str, Any]", node)
        prefix = obj.pop("prefixItems", None)
        if prefix is not None:
            distinct: list[Any] = []
            for item_schema in cast("list[Any]", prefix):
                if item_schema not in distinct:
                    distinct.append(item_schema)
            obj["items"] = distinct[0] if len(distinct) == 1 else {"anyOf": distinct}
        for value in obj.values():
            _inline_prefix_items(value)
    elif isinstance(node, list):
        for value in _as_list(node):
            _inline_prefix_items(value)


def _local_def_ref(node: JSONNode) -> str | None:
    """Return the ``$defs`` name a node's ``$ref`` points at, if any."""
    if isinstance(node, dict):
        obj = cast("dict[str, Any]", node)
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            return ref[len("#/$defs/") :]
    return None


def _collect_def_refs(node: JSONNode, out: set[str]) -> None:
    """Collect every local ``$defs`` name reachable from ``node`` via ``$ref``."""
    if isinstance(node, dict):
        obj = cast("dict[str, Any]", node)
        name = _local_def_ref(obj)
        if name is not None:
            out.add(name)
        for value in obj.values():
            _collect_def_refs(value, out)
    elif isinstance(node, list):
        for value in _as_list(node):
            _collect_def_refs(value, out)


def _ref_edges(defs: dict[str, Any]) -> dict[str, set[str]]:
    """Build the directed graph of ``$defs`` name -> referenced ``$defs`` names."""
    edges: dict[str, set[str]] = {name: set() for name in defs}
    for name, body in defs.items():
        _collect_def_refs(body, edges[name])
    return edges


def _cyclic_def_names(edges: dict[str, set[str]]) -> set[str]:
    """Return every ``$defs`` name that can reach itself along ``$ref`` edges."""

    def reaches(node: str, target: str, seen: frozenset[str]) -> bool:
        if node == target and seen:
            return True
        if node in seen:
            return False
        next_seen = seen | {node}
        return any(reaches(nxt, target, next_seen) for nxt in edges.get(node, ()))

    return {name for name in edges if any(reaches(nxt, name, frozenset()) for nxt in edges[name])}


def _strip_properties_referencing(node: JSONNode, banned: set[str]) -> None:
    """Delete any ``properties`` entry (and matching ``required`` entry) whose
    schema references one of ``banned`` def names, recursing through the
    whole tree in place."""
    if isinstance(node, dict):
        obj = cast("dict[str, Any]", node)
        props = obj.get("properties")
        if isinstance(props, dict):
            prop_map = cast("dict[str, Any]", props)
            drop = [key for key, value in prop_map.items() if _references_any(value, banned)]
            for key in drop:
                del prop_map[key]
            required = obj.get("required")
            if isinstance(required, list):
                obj["required"] = [r for r in _as_list(required) if r not in drop]
        for value in obj.values():
            _strip_properties_referencing(value, banned)
    elif isinstance(node, list):
        for value in _as_list(node):
            _strip_properties_referencing(value, banned)


def _references_any(node: JSONNode, banned: set[str]) -> bool:
    """Whether ``node`` references (directly, or via items/anyOf/oneOf/allOf)
    a ``$defs`` name in ``banned``."""
    if not isinstance(node, dict):
        return False
    obj = cast("dict[str, Any]", node)
    name = _local_def_ref(obj)
    if name is not None and name in banned:
        return True
    if "items" in obj and _references_any(obj["items"], banned):
        return True
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in cast("list[Any]", obj.get(key, ())):
            if _references_any(sub, banned):
                return True
    return False


def _prune_unreachable_defs(schema: dict[str, Any], defs: dict[str, Any]) -> None:
    """Drop any ``$defs`` entry no longer reachable from the root, in place."""
    reachable: set[str] = set()
    frontier: set[str] = set()
    for key, value in schema.items():
        if key != "$defs":
            _collect_def_refs(value, frontier)
    while frontier:
        name = frontier.pop()
        if name in reachable or name not in defs:
            continue
        reachable.add(name)
        _collect_def_refs(defs[name], frontier)
    for name in list(defs):
        if name not in reachable:
            del defs[name]


__all__ = ["flatten_root_ref", "to_llm_safe_schema"]
