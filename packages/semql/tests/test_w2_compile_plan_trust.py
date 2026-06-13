"""W2 stage 4 — ``compile_plan`` must *trust* the plan it is given.

Review B1 / architecture A1 (the finish): ``compile_plan`` used to
reverse-engineer a ``SemanticQuery`` from the plan and then re-run
``to_logical_plan`` inside ``_CompileEnv`` — throwing the caller's plan
away.  That is lossless for an *untransformed* plan (re-planning the
reconstructed query reproduces it), so the byte-equality gate passed.
But it silently discards any plan→plan *transform* the caller applied:
a scan rewritten to a different physical table (rollup / partition
routing), a join dropped (the federation split-point), a predicate
pushed down.  Those transforms are the whole reason the split-point
feeds a plan rather than a query.

These tests pin the load-bearing invariant: emission reads the plan it
is handed.  A scan rewritten on the plan reaches the FROM clause, and
the result diverges from re-planning the original query — proving the
plan, not a re-derived query, drives emission.
"""

from __future__ import annotations

from dataclasses import replace

from semql.compile import compile_plan, compile_query
from semql.logical import to_logical_plan
from semql.model import Cube, Dialect, Dimension, Measure
from semql.spec import Filter, SemanticQuery

from .conftest import CONTEXT


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


def test_compile_plan_honours_rewritten_scan() -> None:
    """A scan rewritten to a different physical table — the kind of
    transform rollup / partition routing performs — must reach the
    emitted FROM clause."""
    catalog = _catalog()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    plan = to_logical_plan(q, catalog)

    rewritten = catalog["orders"].model_copy(update={"table": "prod.orders_2024"})
    transformed = replace(plan, scans=[replace(s, cube=rewritten) for s in plan.scans])

    cq = compile_plan(transformed, catalog, context=CONTEXT)
    assert "orders_2024" in cq.sql, cq.sql
    # The transform is genuinely load-bearing: re-planning the original
    # query never produces the rewritten table.
    assert "orders_2024" not in compile_query(q, catalog, context=CONTEXT).sql


def test_compile_plan_honours_pushed_down_predicate() -> None:
    """A predicate added directly to ``plan.filters`` (not present on the
    originating query) must appear in the WHERE clause."""
    catalog = _catalog()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"])
    plan = to_logical_plan(q, catalog)
    assert plan.filters == []

    from semql.logical import Predicate

    pushed = Filter(dimension="orders.status", op="eq", values=["paid"])
    transformed = replace(plan, filters=[Predicate(expr=pushed)])

    cq = compile_plan(transformed, catalog, context=CONTEXT)
    assert "status" in cq.sql
    assert "paid" in str(cq.params.values())
    # Re-planning the original (filter-free) query has no WHERE on status.
    assert "status" not in compile_query(q, catalog, context=CONTEXT).sql


def test_compile_plan_still_byte_equal_untransformed() -> None:
    """The trust change must not move the untransformed round-trip: an
    unmodified plan still compiles byte-identically to the query path."""
    catalog = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
    )
    plan = to_logical_plan(q, catalog)
    via_plan = compile_plan(plan, catalog, context=CONTEXT)
    via_query = compile_query(q, catalog, context=CONTEXT)
    assert via_plan.sql == via_query.sql
    assert via_plan.params == via_query.params
    assert via_plan.columns == via_query.columns
