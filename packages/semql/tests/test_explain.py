"""Tests for the public ``Catalog.explain()`` entry point.

``explain()`` is the explainability win the LogicalPlan IR was
designed to enable.  It returns a human-readable string of the
LogicalPlan repr so a user (or the MCP ``explain`` tool) can
inspect what the compiler will do *before* the SQL is emitted.

The output is the same ``repr(plan)`` the plan-snapshot tests
already pin, so a regression in either surface is caught
automatically.
"""

from __future__ import annotations

from semql import Catalog, Cube, Dialect, Dimension, Measure
from semql.spec import SemanticQuery


def _orders_catalog() -> Catalog:
    return Catalog(
        [
            Cube(
                name="orders",
                alias="o",
                table="prod.orders",
                backend=Dialect.POSTGRES,
                dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
                measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
            )
        ]
    )


def test_explain_returns_plan_repr() -> None:
    """explain() returns a string of the LogicalPlan repr."""
    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
    )
    output = catalog.explain(query)

    assert isinstance(output, str)
    assert "LogicalPlan(root=orders)" in output
    assert "scans=[Scan(orders as o)]" in output
    assert "aggregate=" in output
    assert "project=" in output


def test_explain_resolves_against_catalog() -> None:
    """explain() raises the same resolution errors as compile()."""
    catalog = _orders_catalog()
    bad_query = SemanticQuery(measures=["orders.nonexistent_measure"])
    try:
        catalog.explain(bad_query)
    except Exception as exc:
        # Same error category as compile — an UnknownIdentifier or
        # CompileError pointing at the field.
        assert "nonexistent_measure" in str(exc) or "not" in str(exc)
    else:
        raise AssertionError("explain() should raise on unresolved field")


def test_explain_does_not_emit_sql() -> None:
    """explain() returns a plan repr, not a CompiledQuery."""
    catalog = _orders_catalog()
    query = SemanticQuery(measures=["orders.revenue"])
    output = catalog.explain(query)
    # No "SELECT" — the plan repr is the surface.
    assert "SELECT" not in output
    # No bound parameters, no params dict, just the IR shape.
    assert "params" not in output


def test_explain_with_where_tree() -> None:
    """explain() captures where-tree predicates as Predicate nodes."""
    from semql.spec import BoolExpr, Filter

    catalog = _orders_catalog()
    query = SemanticQuery(
        measures=["orders.revenue"],
        where=BoolExpr(
            op="or",
            children=[
                Filter(dimension="orders.region", op="eq", values=["us"]),
                Filter(dimension="orders.region", op="eq", values=["ca"]),
            ],
        ),
    )
    output = catalog.explain(query)
    assert "Predicate" in output
    assert "BoolExpr" in output
