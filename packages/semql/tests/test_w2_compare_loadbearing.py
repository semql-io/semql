"""W2 stage 3b — the compare-mode emitter must read its ranges from the
plan, not recompute them.

Review B1: the compare prior-range math was computed *twice* — once in
``to_logical_plan`` (stored on ``CompareSplit.prior_range``) and again in
``_emit_compare_query`` (recomputed from ``q.compare``). The plan node
was write-only; the emitter ignored it. That's duplicated semantic logic
and means a plan transform that rewrites the compare window can't affect
the output.

This pins the plan node as load-bearing: overriding
``plan.compare.prior_range`` must change the emitted prior CTE.
"""

from __future__ import annotations

from dataclasses import replace

from semql.compile import _CompileEnv, _emit_compare_query
from semql.model import Cube, Dialect, Measure, TimeDimension
from semql.spec import CompareWindow, SemanticQuery, TimeWindow


def _catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Dialect.POSTGRES,
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
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


def test_compare_emitter_reads_prior_range_from_plan() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2024-01-01", "2024-04-01"),
        ),
        compare=CompareWindow(mode="previous_period"),
    )
    env = _env(q, _catalog())
    assert env.plan.compare is not None

    # Override the plan's prior range with a sentinel the recompute-from-
    # q.compare path would never produce.
    sentinel = ("1999-07-07", "1999-08-08")
    env.plan = replace(env.plan, compare=replace(env.plan.compare, prior_range=sentinel))

    out = _emit_compare_query(env)
    blob = out.sql + " " + " ".join(str(v) for v in out.params.values())
    assert "1999-07-07" in blob
    assert "1999-08-08" in blob
