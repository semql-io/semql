"""Tests for ``SemanticQuery.where`` — boolean predicate trees.

The flat ``filters`` list is implicit AND. To express
``(region = 'us' AND status = 'paid') OR plan = 'enterprise'`` the
planner builds a ``BoolExpr`` tree and assigns it to ``where``.

Tree shape: ``BoolExpr(op="and"|"or"|"not", children=[Filter | BoolExpr])``.
``not`` takes exactly one child; ``and`` / ``or`` take two or more.

``where`` composes with the existing flat ``filters`` list via implicit
AND — both apply when both are set. ``required_filters`` enforcement
still considers ``filters`` only, since a Filter buried inside an OR
doesn't actually constrain the column.
"""

from __future__ import annotations

from dataclasses import replace

import duckdb
import pytest
from semql import (
    BoolExpr,
    Catalog,
    CompileError,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    Segment,
    SemanticQuery,
)
from semql.compile import compile_plan
from semql.logical import Predicate, to_logical_plan
from semql.spec import FilterOp


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="plan", sql="{o}.plan", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
    )


def _boolean_orders_cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="plan", sql="{o}.plan", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        segments=[Segment(name="paid", sql="{o}.status = 'paid'")],
    )


def _boolean_rows() -> list[tuple[int, str | None, str | None, str | None, int | None]]:
    return [
        (1, "us", "paid", "basic", 10),
        (2, "us", "open", "basic", 20),
        (3, "eu", "paid", "enterprise", 30),
        (4, "eu", "open", "basic", 40),
        (5, None, "paid", "enterprise", 50),
        (6, "us", None, "enterprise", None),
        (7, "ca", "paid", None, 70),
        (8, None, None, None, None),
    ]


def _execute_membership(catalog: Catalog, query: SemanticQuery) -> list[int]:
    compiled = catalog.compile(query)
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE orders(id INTEGER, region VARCHAR, status VARCHAR, "
            "plan VARCHAR, amount INTEGER)"
        )
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _boolean_rows())
        return sorted(
            row[0] for row in connection.execute(compiled.sql, compiled.params).fetchall()
        )


def _filter(field: str, op: FilterOp, value: str | int) -> Filter:
    return Filter(dimension=f"orders.{field}", op=op, values=[value])


@pytest.mark.parametrize(
    ("tree", "oracle"),
    [
        (
            BoolExpr(
                op="and",
                children=[
                    _filter("region", "eq", "us"),
                    BoolExpr(
                        op="or",
                        children=[
                            _filter("status", "eq", "paid"),
                            _filter("plan", "eq", "enterprise"),
                        ],
                    ),
                ],
            ),
            "region = 'us' AND (status = 'paid' OR plan = 'enterprise')",
        ),
        (
            BoolExpr(
                op="or",
                children=[
                    _filter("region", "eq", "us"),
                    BoolExpr(
                        op="and",
                        children=[
                            _filter("status", "eq", "paid"),
                            _filter("plan", "eq", "enterprise"),
                        ],
                    ),
                ],
            ),
            "region = 'us' OR (status = 'paid' AND plan = 'enterprise')",
        ),
        (
            BoolExpr(
                op="not",
                children=[
                    BoolExpr(
                        op="or",
                        children=[
                            _filter("region", "eq", "us"),
                            BoolExpr(
                                op="and",
                                children=[
                                    _filter("status", "eq", "paid"),
                                    _filter("plan", "eq", "enterprise"),
                                ],
                            ),
                        ],
                    )
                ],
            ),
            "NOT (region = 'us' OR (status = 'paid' AND plan = 'enterprise'))",
        ),
        (
            BoolExpr(
                op="not",
                children=[
                    BoolExpr(
                        op="not",
                        children=[
                            BoolExpr(
                                op="or",
                                children=[
                                    _filter("region", "eq", "us"),
                                    _filter("status", "eq", "paid"),
                                ],
                            )
                        ],
                    )
                ],
            ),
            "region = 'us' OR status = 'paid'",
        ),
    ],
)
def test_boolean_tree_membership_matches_independent_duckdb_sql(
    tree: BoolExpr, oracle: str
) -> None:
    catalog = Catalog([_boolean_orders_cube()])
    actual = _execute_membership(
        catalog,
        SemanticQuery(dimensions=["orders.id"], where=tree),
    )
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE orders(id INTEGER, region VARCHAR, status VARCHAR, "
            "plan VARCHAR, amount INTEGER)"
        )
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _boolean_rows())
        expected = sorted(
            row[0] for row in connection.execute(f"SELECT id FROM orders WHERE {oracle}").fetchall()
        )
    assert actual == expected


def test_boolean_tree_membership_composes_with_flat_filter_and_segment() -> None:
    catalog = Catalog([_boolean_orders_cube()])
    tree = BoolExpr(
        op="or",
        children=[
            BoolExpr(
                op="and",
                children=[_filter("region", "eq", "us"), _filter("plan", "eq", "basic")],
            ),
            _filter("region", "eq", "eu"),
        ],
    )
    actual = _execute_membership(
        catalog,
        SemanticQuery(
            dimensions=["orders.id"],
            filters=[_filter("amount", "gt", 15)],
            segments=["orders.paid"],
            where=tree,
        ),
    )
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE orders(id INTEGER, region VARCHAR, status VARCHAR, "
            "plan VARCHAR, amount INTEGER)"
        )
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _boolean_rows())
        expected = sorted(
            row[0]
            for row in connection.execute(
                "SELECT id FROM orders WHERE amount > 15 AND status = 'paid' "
                "AND ((region = 'us' AND plan = 'basic') OR region = 'eu')"
            ).fetchall()
        )
    assert actual == expected


def test_compile_plan_preserves_compound_not_grouping() -> None:
    catalog = Catalog([_boolean_orders_cube()])
    query = SemanticQuery(dimensions=["orders.id"])
    plan = to_logical_plan(query, catalog.as_dict())
    tree = BoolExpr(
        op="not",
        children=[
            BoolExpr(
                op="or",
                children=[
                    _filter("region", "eq", "us"),
                    BoolExpr(
                        op="and",
                        children=[
                            _filter("status", "eq", "paid"),
                            _filter("plan", "eq", "enterprise"),
                        ],
                    ),
                ],
            )
        ],
    )
    plan = replace(plan, filters=[Predicate(tree)])
    compiled = compile_plan(plan, catalog.as_dict())
    with duckdb.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE orders(id INTEGER, region VARCHAR, status VARCHAR, "
            "plan VARCHAR, amount INTEGER)"
        )
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", _boolean_rows())
        actual = sorted(
            row[0] for row in connection.execute(compiled.sql, compiled.params).fetchall()
        )
        expected = sorted(
            row[0]
            for row in connection.execute(
                "SELECT id FROM orders WHERE NOT "
                "(region = 'us' OR (status = 'paid' AND plan = 'enterprise'))"
            ).fetchall()
        )
    assert actual == expected


# ---------------------------------------------------------------------------
# Model — BoolExpr construction.
# ---------------------------------------------------------------------------


def test_boolexpr_and_with_filter_children() -> None:
    expr = BoolExpr(
        op="and",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            Filter(dimension="orders.status", op="eq", values=["paid"]),
        ],
    )
    assert expr.op == "and"
    assert len(expr.children) == 2


def test_boolexpr_or_with_filter_children() -> None:
    expr = BoolExpr(
        op="or",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            Filter(dimension="orders.region", op="eq", values=["eu"]),
        ],
    )
    assert expr.op == "or"


def test_boolexpr_not_with_single_child() -> None:
    expr = BoolExpr(
        op="not",
        children=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    assert expr.op == "not"
    assert len(expr.children) == 1


def test_boolexpr_nested_tree() -> None:
    inner = BoolExpr(
        op="and",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            Filter(dimension="orders.status", op="eq", values=["paid"]),
        ],
    )
    outer = BoolExpr(
        op="or",
        children=[
            inner,
            Filter(dimension="orders.plan", op="eq", values=["enterprise"]),
        ],
    )
    assert outer.op == "or"
    assert isinstance(outer.children[0], BoolExpr)
    assert isinstance(outer.children[1], Filter)


def test_boolexpr_is_frozen() -> None:
    expr = BoolExpr(
        op="and",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            Filter(dimension="orders.status", op="eq", values=["paid"]),
        ],
    )
    with pytest.raises(Exception):  # noqa: B017, BLE001 — Pydantic raises ValidationError on frozen mutation; the exact class isn't load-bearing here.
        expr.op = "or"


def test_boolexpr_not_rejects_multiple_children() -> None:
    with pytest.raises(ValueError, match=r"(?i)not"):
        BoolExpr(
            op="not",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.status", op="eq", values=["paid"]),
            ],
        )


def test_boolexpr_and_or_require_at_least_two_children() -> None:
    with pytest.raises(ValueError, match=r"(?i)and|or|children"):
        BoolExpr(
            op="and",
            children=[Filter(dimension="orders.region", op="eq", values=["us"])],
        )


# ---------------------------------------------------------------------------
# Spec — SemanticQuery.where field.
# ---------------------------------------------------------------------------


def test_semantic_query_where_defaults_to_none() -> None:
    q = SemanticQuery(measures=["orders.count"])
    assert q.where is None


def test_semantic_query_accepts_where_tree() -> None:
    expr = BoolExpr(
        op="or",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            Filter(dimension="orders.region", op="eq", values=["eu"]),
        ],
    )
    q = SemanticQuery(measures=["orders.count"], where=expr)
    assert q.where is expr


# ---------------------------------------------------------------------------
# Compiler — emit AND / OR / NOT into the WHERE clause.
# ---------------------------------------------------------------------------


def test_or_tree_emits_or_in_sql() -> None:
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["eu"]),
                ],
            ),
        )
    )
    assert " OR " in out.sql
    assert "us" in out.params.values()
    assert "eu" in out.params.values()


def test_and_tree_emits_and_in_sql() -> None:
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="and",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.status", op="eq", values=["paid"]),
                ],
            ),
        )
    )
    assert "o.region" in out.sql
    assert "o.status" in out.sql
    assert "us" in out.params.values()
    assert "paid" in out.params.values()


def test_not_tree_emits_not_in_sql() -> None:
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="not",
                children=[Filter(dimension="orders.region", op="eq", values=["us"])],
            ),
        )
    )
    assert "NOT" in out.sql.upper()
    assert "us" in out.params.values()


def test_where_composes_with_flat_filters_via_and() -> None:
    """``filters`` (implicit AND) and ``where`` (tree) both apply — the
    tree's predicate ANDs with each filter and with base_predicate."""
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.status", op="eq", values=["paid"])],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["eu"]),
                ],
            ),
        )
    )
    assert "o.status" in out.sql
    assert "o.region" in out.sql
    assert "paid" in out.params.values()
    assert "us" in out.params.values()
    assert "eu" in out.params.values()


def test_where_tree_values_are_parameter_bound() -> None:
    """Filter values inside the tree never appear as SQL literals."""
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["eu"]),
                ],
            ),
        )
    )
    assert "'us'" not in out.sql
    assert "'eu'" not in out.sql
    assert "us" in out.params.values()
    assert "eu" in out.params.values()


def test_where_tree_filter_type_check_applies() -> None:
    """Type-mismatch in a tree leaf still raises a FilterTypeError."""
    from semql import FilterTypeError

    cat = Catalog([_orders_cube()])
    with pytest.raises(FilterTypeError):
        cat.compile(
            SemanticQuery(
                measures=["orders.count"],
                where=BoolExpr(
                    op="or",
                    children=[
                        Filter(dimension="orders.amount", op="gt", values=["abc"]),
                        Filter(dimension="orders.amount", op="lt", values=[100]),
                    ],
                ),
            )
        )


def test_where_tree_unknown_field_raises_compile_error() -> None:
    cat = Catalog([_orders_cube()])
    with pytest.raises(CompileError):
        cat.compile(
            SemanticQuery(
                measures=["orders.count"],
                where=BoolExpr(
                    op="or",
                    children=[
                        Filter(dimension="orders.region", op="eq", values=["us"]),
                        Filter(dimension="orders.nonexistent", op="eq", values=["x"]),
                    ],
                ),
            )
        )


def test_where_in_op_inside_tree() -> None:
    """``IN`` inside a tree leaf — multiple values still bind one each."""
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="in", values=["us", "ca"]),
                    Filter(dimension="orders.plan", op="eq", values=["enterprise"]),
                ],
            ),
        )
    )
    assert " OR " in out.sql
    assert "us" in out.params.values()
    assert "ca" in out.params.values()
    assert "enterprise" in out.params.values()


def test_where_alone_no_flat_filters() -> None:
    """``where`` works as the only predicate source."""
    cat = Catalog([_orders_cube()])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.count"],
            where=BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.region", op="eq", values=["us"]),
                    Filter(dimension="orders.region", op="eq", values=["eu"]),
                ],
            ),
        )
    )
    assert " WHERE " in out.sql.upper()
    assert " OR " in out.sql
