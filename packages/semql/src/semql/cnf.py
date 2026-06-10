"""CNF — Convert a ``BoolExpr`` to Conjunctive Normal Form.

CNF shape: top-level AND, every clause an OR of literals. Each
"literal" is either a :class:`~semql.spec.Filter` leaf or a
``BoolExpr(op="not")`` wrapping one — the emitter flips negated
filters (``eq`` ↔ ``neq``, ``in`` ↔ ``not_in``, ``is_null`` ↔
``not_null``).

Pure function: no I/O, no globals, deterministic. ``compile_query``
invokes :func:`to_cnf` on the plan's ``Predicate`` nodes so every
downstream consumer (federation, segment routing, pushdown) sees a
shape where each clause is independently movable.

Algorithm: recursive distribution + de-Morgan + dedup. See
``tests/test_cnf.py`` for the contract.
"""

from __future__ import annotations

from typing import Literal as PyLiteral
from typing import cast

from semql.spec import BoolExpr, Filter

_Literal = Filter | BoolExpr
"""A literal is a Filter leaf or a NOT(Filter) — both atomic from CNF's POV."""


def _to_cnf(node: _Literal) -> _Literal:
    """Recursively normalise ``node`` to CNF."""
    if isinstance(node, Filter):
        return node

    if node.op == "not":
        # NNF: push NOT in.
        inner = node.children[0]
        if isinstance(inner, Filter):
            # NOT(Filter) — leaf negation; emit as-is for the dialect to flip.
            return BoolExpr(op="not", children=[inner])
        if inner.op == "not":
            # NOT NOT x — double-negation collapse.
            return _to_cnf(inner.children[0])
        if inner.op == "and":
            # NOT (a AND b) -> OR(NOT a, NOT b)
            flipped: list[BoolExpr | Filter] = [
                BoolExpr(op="not", children=[c]) for c in inner.children
            ]
            return _to_cnf(BoolExpr(op="or", children=flipped))
        if inner.op == "or":
            # NOT (a OR b) -> AND(NOT a, NOT b)
            flipped = [BoolExpr(op="not", children=[c]) for c in inner.children]
            return _to_cnf(BoolExpr(op="and", children=flipped))
        raise AssertionError(f"unknown BoolExpr.op: {inner.op!r}")

    if node.op in ("and", "or"):
        # 1. Recurse into each child first.
        children = [_to_cnf(c) for c in node.children]
        # 2. Flatten same-op nests; dedup.
        children = _flatten_and_dedup(node.op, children)
        # 3. Distribute OR over AND.  After distribution the outer op
        #    becomes AND regardless of the input op — distribution
        #    converts ``OR(AND, AND, ...)`` into ``AND(OR, OR, ...)``.
        if node.op == "or":
            clauses = _distribute_or_over_and(children)
            return _rebuild("and", clauses)
        return _rebuild("and", children)

    raise AssertionError(f"unknown BoolExpr.op: {node.op!r}")


def _flatten_and_dedup(op: str, children: list[_Literal]) -> list[_Literal]:
    """Pull nested same-op nodes up to the surface; drop duplicates.

    A duplicate is the same literal at the same nesting level — two
    ``AND(a, a)`` collapses to ``AND(a)``. We key by ``model_dump()``
    for Filters (which are frozen Pydantic) and by ``repr`` for
    BoolExpr (cheap structural fingerprint that respects node shape).
    """
    flat: list[_Literal] = []
    seen: set[str] = set()
    for c in children:
        if isinstance(c, BoolExpr) and c.op == op:
            for grand in c.children:
                key = _key(grand)
                if key not in seen:
                    seen.add(key)
                    flat.append(grand)
        else:
            key = _key(c)
            if key not in seen:
                seen.add(key)
                flat.append(c)
    return flat


def _distribute_or_over_and(children: list[_Literal]) -> list[_Literal]:
    """``OR(AND(a, b), c)`` → ``AND(OR(a, c), OR(b, c))``.

    Recursively distributes until no OR has an AND child. The result
    is a list of OR-of-literals clauses; the outer wrap builds the
    top-level AND.
    """
    # Each child contributes a list of "literals" (itself, or its
    # children if it's an AND — that's the cross-product axis).
    child_literals: list[list[_Literal]] = []
    for c in children:
        if isinstance(c, BoolExpr) and c.op == "and":
            child_literals.append(list(c.children))
        else:
            child_literals.append([c])

    # Cartesian product: every combination of one literal per child
    # becomes an OR-clause. Combined into a list of OR nodes; the
    # outer caller wraps in an AND.
    if not child_literals:
        return []
    out: list[_Literal] = []
    for combo in _cartesian_product(child_literals):
        seen: set[str] = set()
        unique: list[_Literal] = []
        for lit in combo:
            key = _key(lit)
            if key not in seen:
                seen.add(key)
                unique.append(lit)
        if len(unique) == 1:
            out.append(unique[0])
        else:
            out.append(BoolExpr(op="or", children=unique))
    return out


def _cartesian_product(lists: list[list[_Literal]]) -> list[list[_Literal]]:
    """Cartesian product of a list of literal-lists. Returns all combinations."""
    if not lists:
        return [[]]
    result: list[list[_Literal]] = [[]]
    for lst in lists:
        new_result: list[list[_Literal]] = []
        for prefix in result:
            for item in lst:
                new_result.append(prefix + [item])
        result = new_result
    return result


def _rebuild(op: str, children: list[_Literal]) -> _Literal:
    """Build a BoolExpr; collapse singletons to their single child."""
    if len(children) == 0:
        # Empty AND/OR — convention: AND is a tautology, OR is a contradiction.
        # We never reach here on a well-formed input (BoolExpr validator
        # requires >= 2 children for "and"/"or"), but be safe.
        if op == "and":
            return Filter(dimension="__true__", op="eq", values=[True])
        return BoolExpr(
            op="or",
            children=cast(list[BoolExpr | Filter], []),
        )
    if len(children) == 1:
        return children[0]
    return BoolExpr(
        op=cast(PyLiteral["and", "or", "not"], op),
        children=children,
    )


def _key(node: _Literal) -> str:
    """Structural fingerprint for dedup.

    ``Filter`` is a frozen Pydantic model; ``model_dump()`` is a
    JSON-safe dict that hashes consistently. ``BoolExpr`` is
    recursive, so we use ``model_dump_json()`` to get a stable
    canonical form. Hashable via ``str``.
    """
    if isinstance(node, Filter):
        return node.model_dump_json()
    return node.model_dump_json()


def to_cnf(node: _Literal) -> _Literal:
    """Public entry point — normalise a tree (or single Filter) to CNF.

    Args:
        node: a :class:`~semql.spec.Filter` leaf or a
            :class:`~semql.spec.BoolExpr` subtree.

    Returns:
        A CNF ``BoolExpr`` (top-level AND, every clause an OR of
        literals) or a single ``Filter`` if the input collapsed
        to a single literal. Identity is preserved when the input
        is already in CNF and idempotent under rebuild.
    """
    return _to_cnf(node)
