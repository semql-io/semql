# mypy: disable-error-code=type-arg
# pyright: reportMissingTypeArgument=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnusedVariable=false, reportUnusedImport=false
"""CNF — BoolExpr where-tree normalisation.

Pure pre-pass that converts a ``BoolExpr`` into Conjunctive Normal
Form: a top-level AND of clauses, each clause an OR of literals.
The compiler's WHERE emission reads the tree, but downstream consumers
(distributive-mode federation, segment routing, pushdown detection)
need a shape where every clause is independently movable / inspectable.

Algorithm (standard recursive distribution + de-Morgan + dedup):

  1. **NNF** — push NOTs in with de-Morgan:
     - ``NOT (a AND b)`` -> ``OR(NOT a, NOT b)``
     - ``NOT (a OR b)`` -> ``AND(NOT a, NOT b)``
     - ``NOT NOT a`` -> ``a``
     - Leaf negation (e.g. ``NOT Filter``) is left to the dialect
       (compilers flip ``eq`` ↔ ``neq``, ``in`` ↔ ``not_in``,
       ``is_null`` ↔ ``not_null`` — no need to do it here).

  2. **Flatten** — collapse nested same-op nodes:
     - ``AND(a, AND(b, c))`` -> ``AND(a, b, c)``
     - ``OR(a, OR(b, c))`` -> ``OR(a, b, c)``
     - Deduplicate children (by ``model_dump()`` hash).

  3. **Distribute** — push OR over AND:
     - ``OR(AND(a, b), c)`` -> ``AND(OR(a, c), OR(b, c))``
     - Repeated until no OR has an AND child.

  4. **Identity** — drop tautologies / contradictions:
     - ``AND(..., True)`` -> the rest (Filter can't be True, so this
       is a no-op for our trees — but document the invariant).

  5. **Rebuild** — emit a new ``BoolExpr`` (or the original if it was
     already in CNF).

  The function is **pure**: no I/O, no globals, deterministic for
  the same input. ``compile_query`` invokes it on ``q.where`` (and on
  every flat ``Filter`` that the plan wraps in a ``Predicate`` node)
  so all emission sees the normalised tree.
"""

from __future__ import annotations

from semql.cnf import to_cnf
from semql.spec import BoolExpr, Filter, SemanticQuery

# ---------------------------------------------------------------------------
# Leaves pass through unchanged
# ---------------------------------------------------------------------------


def test_flat_filter_passes_through_unchanged() -> None:
    """A single Filter is already in CNF — to_cnf returns it as-is."""
    f = Filter(dimension="orders.region", op="eq", values=["emea"])
    out = to_cnf(f)
    assert out is f or out == f  # either identity or rebuilt equal


def test_and_of_filters_stays_flat() -> None:
    """``AND(a, b)`` is already CNF — output equals input structurally."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    tree = BoolExpr(op="and", children=[a, b])
    out = to_cnf(tree)
    # Output is an AND with exactly the same leaves.
    assert isinstance(out, BoolExpr)
    assert out.op == "and"
    leaves = [c for c in out.children if isinstance(c, Filter)]
    assert {str(leaf.dimension) for leaf in leaves} == {"orders.region", "orders.status"}


# ---------------------------------------------------------------------------
# Distribution: OR(AND, ...) → AND(OR, OR, ...)
# ---------------------------------------------------------------------------


def test_or_of_and_distributes_to_cnf() -> None:
    """``OR(AND(a, b), AND(c, d))`` → ``AND(OR(a, c), OR(a, d), OR(b, c), OR(b, d))``."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    c = Filter(dimension="orders.region", op="eq", values=["west"])
    d = Filter(dimension="orders.status", op="eq", values=["pending"])

    # (region=emea AND status=paid) OR (region=west AND status=pending)
    left = BoolExpr(op="and", children=[a, b])
    right = BoolExpr(op="and", children=[c, d])
    tree = BoolExpr(op="or", children=[left, right])

    out = to_cnf(tree)
    assert isinstance(out, BoolExpr)
    assert out.op == "and"  # top-level AND after distribution
    # Each clause is an OR of two Filters.
    clauses = out.children
    assert all(isinstance(cl, BoolExpr) and cl.op == "or" for cl in clauses)
    # 2 × 2 = 4 distributed clauses.
    assert len(clauses) == 4


def test_or_of_single_and_distributes() -> None:
    """``OR(AND(a, b), c)`` → ``AND(OR(a, c), OR(b, c))``."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    c = Filter(dimension="orders.is_paid", op="eq", values=[True])

    left = BoolExpr(op="and", children=[a, b])
    tree = BoolExpr(op="or", children=[left, c])

    out = to_cnf(tree)
    assert isinstance(out, BoolExpr)
    assert out.op == "and"
    clauses = out.children
    assert len(clauses) == 2
    # Each clause is an OR(a or b, c). The OR's children are the
    # distributed cross-product: (a OR c) and (b OR c).
    assert all(isinstance(cl, BoolExpr) and cl.op == "or" for cl in clauses)
    flat_literals = [c for cl in clauses if isinstance(cl, BoolExpr) for c in cl.children]
    assert a in flat_literals
    assert b in flat_literals
    # c is in both clauses.
    for cl in clauses:
        if isinstance(cl, BoolExpr):
            assert c in cl.children


# ---------------------------------------------------------------------------
# Flatten + dedup
# ---------------------------------------------------------------------------


def test_nested_and_flattens() -> None:
    """``AND(a, AND(b, c))`` → ``AND(a, b, c)``."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    c = Filter(dimension="orders.is_paid", op="eq", values=[True])
    tree = BoolExpr(op="and", children=[a, BoolExpr(op="and", children=[b, c])])

    out = to_cnf(tree)
    assert isinstance(out, BoolExpr)
    assert out.op == "and"
    leaves = [child for child in out.children if isinstance(child, Filter)]
    assert len(leaves) == 3
    assert {leaf.dimension for leaf in leaves} == {
        "orders.region",
        "orders.status",
        "orders.is_paid",
    }


def test_duplicate_filters_dedup() -> None:
    """Two identical Filter leaves under AND collapse to one."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.region", op="eq", values=["emea"])
    tree = BoolExpr(op="and", children=[a, b])

    out = to_cnf(tree)
    # After dedup the only child is the lone Filter; _rebuild returns
    # the single child directly (BoolExpr requires >=2 children).
    if isinstance(out, Filter):
        assert out is a
    else:
        assert isinstance(out, BoolExpr)
        assert out.op == "and"
        leaves = [c for c in out.children if isinstance(c, Filter)]
        assert len(leaves) == 1


# ---------------------------------------------------------------------------
# Negation pushes in (NNF)
# ---------------------------------------------------------------------------


def test_not_pushed_in_via_demorgan() -> None:
    """``NOT(AND(a, b))`` → ``OR(NOT a, NOT b)`` (NNF).

    Leaf-level negation (NOT Filter) is left to the dialect; the
    normalised tree contains a BoolExpr ``not`` with a single Filter
    child, which the emitter flips.
    """
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    inner = BoolExpr(op="and", children=[a, b])
    not_node = BoolExpr(op="not", children=[inner])

    out = to_cnf(not_node)
    assert isinstance(out, BoolExpr)
    # NOT(AND) -> OR(NOT a, NOT b) (NNF). A single OR-clause is its
    # own AND — top-level is OR, with two NOT(Filter) children.
    assert out.op == "or"
    nots = [c for c in out.children if isinstance(c, BoolExpr) and c.op == "not"]
    assert len(nots) == 2


def test_double_negation_collapsed() -> None:
    """``NOT NOT a`` → ``a`` (collapses to the leaf)."""
    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    not_a = BoolExpr(op="not", children=[a])
    not_not_a = BoolExpr(op="not", children=[not_a])

    out = to_cnf(not_not_a)
    # After double-negation collapse, the leaf is back at the top.
    # Could be a single Filter (pass-through) or AND([Filter]).
    if isinstance(out, Filter):
        assert out is a
    else:
        assert isinstance(out, BoolExpr)
        assert out.op == "and"
        leaves = [c for c in out.children if isinstance(c, Filter)]
        assert len(leaves) == 1
        assert leaves[0] is a


# ---------------------------------------------------------------------------
# Integration: plan-level predicates are normalised to CNF
# ---------------------------------------------------------------------------


def test_plan_predicates_are_cnf_normalised(catalog: dict) -> None:
    """The plan carries CNF-normalised predicates after ``to_logical_plan``.

    Downstream consumers (distributive-mode federation, segment
    routing, pushdown) expect a normalised shape. ``to_logical_plan``
    applies the CNF pre-pass on every Predicate it builds.
    """
    from semql.logical import to_logical_plan
    from semql.spec import BoolExpr

    a = Filter(dimension="orders.region", op="eq", values=["emea"])
    b = Filter(dimension="orders.status", op="eq", values=["paid"])
    c = Filter(dimension="orders.region", op="eq", values=["west"])
    d = Filter(dimension="orders.status", op="eq", values=["pending"])
    left = BoolExpr(op="and", children=[a, b])
    right = BoolExpr(op="and", children=[c, d])
    q_tree = BoolExpr(op="or", children=[left, right])

    q = SemanticQuery(
        measures=["orders.revenue"],
        where=q_tree,
    )
    plan = to_logical_plan(q, catalog)
    # The plan should carry exactly one Predicate node whose expr is
    # in CNF — top-level AND, 4 OR-clauses of 2 literals each.
    assert len(plan.filters) == 1
    expr = plan.filters[0].expr
    assert isinstance(expr, BoolExpr)
    assert expr.op == "and"
    assert len(expr.children) == 4
    for cl in expr.children:
        assert isinstance(cl, BoolExpr)
        assert cl.op == "or"
