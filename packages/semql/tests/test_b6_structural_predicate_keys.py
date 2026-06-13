"""B6 (W2 stage 1) — predicate resolution must be structural, not by
object identity.

Architecture review B6: ``_CompileEnv`` resolves a filter leaf to its
catalog field via object identity — ``f is leaf`` against
``filter_resolutions`` and ``id(leaf)`` against ``where_leaf_resolutions``
(``compile.py``/``_resolve.py``). That only works because
``to_logical_plan``'s CNF pass reuses the *same* ``Filter`` objects the
resolver walked. Any IR transform that *copies* a ``Filter`` — which the
federation split-point and the distributive where-tree routing both do
(rebuilding clauses from CNF) — yields a leaf whose ``id()`` is absent,
so emission raises ``CompileError`` even though the leaf names a perfectly
resolvable ``cube.field``.

The field a leaf resolves to depends only on ``leaf.dimension`` (the
resolver calls ``resolve_with_views(leaf.dimension)``), so the fix keys
resolution by ``dimension``. These tests pin that a *copied* leaf — a new
object with the same ``dimension`` — resolves identically.
"""

from __future__ import annotations

import pytest
from semql._resolve import walk_where_leaves
from semql.compile import _CompileEnv
from semql.errors import CompileError
from semql.model import Cube, Dialect, Dimension, Measure
from semql.spec import BoolExpr, Filter, SemanticQuery


def _catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Dialect.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
    )
    return {"orders": orders}


def _env(q: SemanticQuery, catalog: dict[str, Cube]) -> _CompileEnv:
    return _CompileEnv(
        q,
        catalog,
        context=None,
        group_by_alias=True,
        having_alias=False,
        dialects=None,
        views=None,
        viewer=None,
        policy=None,
        scope_fns=None,
        allow_unbounded_ungrouped=False,
    )


def test_copied_where_leaf_resolves() -> None:
    """A where-tree leaf copied into a new object (as CNF routing does)
    still resolves to its field."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.region", op="eq", values=["ca"]),
            ],
        ),
    )
    env = _env(q, _catalog())
    assert q.where is not None
    original = walk_where_leaves(q.where)[0]
    copied = original.model_copy()
    assert copied is not original  # genuinely a different object
    fld = env._lookup_filter_field(copied)
    assert fld.name == "region"


def test_copied_flat_filter_resolves() -> None:
    """A flat filter copied into a new object resolves too — the old
    ``f is leaf`` identity check would miss it."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
    )
    env = _env(q, _catalog())
    copied = q.filters[0].model_copy()
    assert copied is not q.filters[0]
    fld = env._lookup_filter_field(copied)
    assert fld.name == "status"


def test_unresolvable_leaf_still_raises() -> None:
    """The refusal path is preserved: a leaf naming a field the plan
    never resolved is still a CompileError."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
    )
    env = _env(q, _catalog())
    stray = Filter(dimension="orders.region", op="eq", values=["us"])
    with pytest.raises(CompileError, match=r"(?i)could not be resolved"):
        env._lookup_filter_field(stray)
