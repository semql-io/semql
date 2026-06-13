"""A1 — ``compile_plan`` must be a *lossless* round-trip.

Architecture review (2026-06) A1: ``compile_plan`` re-derives a
``SemanticQuery`` from the plan but copied back only
measures/dimensions/time/compare/order/limit — silently dropping
``filters``, ``where``, ``having``, ``segments``, ``ungrouped``,
``left_joins``, ``derived_measures`` and ``aliases``.  Because the
re-derived query is then *re-planned* inside ``_CompileEnv``, every
dropped field vanishes from the emitted SQL: the plan path returned
different (wrong) SQL than the spec-tree path for any non-trivial
query.

The existing ``test_compile_plan_byte_equal_to_compile_query`` only
covered the trivial measures+dimensions shape, so the divergence went
unnoticed.  These tests pin the invariant the docstring already
promises: ``compile_plan(to_logical_plan(q)) == compile_query(q)``,
byte-for-byte, for each previously-dropped field and for all of them
combined.
"""

from __future__ import annotations

from collections.abc import Mapping

from semql.compile import compile_plan, compile_query
from semql.logical import to_logical_plan
from semql.model import (
    Cube,
    Dialect,
    Dimension,
    Measure,
    Segment,
)
from semql.model import (
    Join as ModelJoin,
)
from semql.spec import (
    BoolExpr,
    Filter,
    InlineDerived,
    SemanticQuery,
)

from .conftest import CONTEXT


def _catalog() -> dict[str, Cube]:
    orders = Cube(
        name="orders",
        alias="o",
        table="prod.orders",
        backend=Dialect.POSTGRES,
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="cnt", sql="*", agg="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
        ],
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
        joins=[
            ModelJoin(to="customers", relationship="many_to_one", on="{o}.customer_id = {c}.id")
        ],
    )
    customers = Cube(
        name="customers",
        alias="c",
        table="prod.customers",
        backend=Dialect.POSTGRES,
        dimensions=[
            Dimension(name="name", sql="{c}.name", type="string"),
            Dimension(name="tier", sql="{c}.tier", type="string"),
        ],
    )
    return {"orders": orders, "customers": customers}


def _assert_round_trip(catalog: Mapping[str, Cube], q: SemanticQuery) -> None:
    """The plan path and the spec-tree path must agree byte-for-byte."""
    cat = dict(catalog)
    plan = to_logical_plan(q, cat)
    via_plan = compile_plan(plan, cat, context=CONTEXT)
    via_query = compile_query(q, cat, context=CONTEXT)
    assert via_plan.sql == via_query.sql, (
        f"compile_plan dropped state:\n  via_plan : {via_plan.sql}\n  via_query: {via_query.sql}"
    )
    assert via_plan.columns == via_query.columns
    assert via_plan.params == via_query.params
    assert via_plan.column_meta == via_query.column_meta


def test_round_trip_flat_filter() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
        ),
    )


def test_round_trip_where_tree() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["ca"]),
                ],
            ),
        ),
    )


def test_round_trip_flat_filter_and_where_tree() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["ca"]),
                ],
            ),
        ),
    )


def test_round_trip_segment() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            segments=["orders.paid"],
        ),
    )


def test_round_trip_having() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            having=[Filter(dimension="orders.revenue", op="gt", values=[1000])],
        ),
    )


def test_round_trip_derived_measures() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue", "orders.cnt"],
            dimensions=["orders.region"],
            derived_measures=[
                InlineDerived(
                    name="avg_order", op="ratio", operands=["orders.revenue", "orders.cnt"]
                )
            ],
        ),
    )


def test_round_trip_left_join() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            filters=[Filter(dimension="customers.name", op="eq", values=["alice"])],
            left_joins=["customers"],
        ),
    )


def test_round_trip_ungrouped() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            dimensions=["orders.region", "orders.status"],
            ungrouped=True,
            limit=10,
        ),
    )


def test_round_trip_aliases() -> None:
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            aliases={"net": "orders.revenue"},
        ),
    )


def test_round_trip_kitchen_sink() -> None:
    """Everything the re-derivation used to drop, in one query."""
    _assert_round_trip(
        _catalog(),
        SemanticQuery(
            measures=["orders.revenue", "orders.cnt"],
            dimensions=["orders.region"],
            filters=[
                Filter(dimension="orders.status", op="eq", values=["paid"]),
                Filter(dimension="customers.name", op="eq", values=["alice"]),
            ],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["ca"]),
                ],
            ),
            segments=["orders.paid"],
            having=[Filter(dimension="orders.revenue", op="gt", values=[1000])],
            derived_measures=[
                InlineDerived(
                    name="avg_order", op="ratio", operands=["orders.revenue", "orders.cnt"]
                )
            ],
            left_joins=["customers"],
            aliases={"net": "orders.revenue"},
            order=[("orders.revenue", "desc")],
            limit=50,
            offset=5,
        ),
    )
