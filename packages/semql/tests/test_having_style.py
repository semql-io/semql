"""HAVING reference style — accept both bare and qualified.

The rest of the spec is qualified (cube.field). HAVING currently only
accepts bare measure names. We accept both shapes so the planner can
emit the more natural qualified form without an extra translation step.
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    CompileError,
    Cube,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
)
from semql.model import Dialect


def _cat() -> Catalog:
    orders = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    return Catalog([orders])


def test_having_accepts_bare_measure_name() -> None:
    """Backwards-compat: the existing behavior of bare measure names
    in HAVING must keep working."""
    cat = _cat()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="revenue", op="gt", values=[100])],
    )
    out = cat.compile(q)
    assert "HAVING" in out.sql


def test_having_accepts_qualified_measure_name() -> None:
    """New: qualified cube.measure names also resolve."""
    cat = _cat()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="orders.revenue", op="gt", values=[100])],
    )
    out = cat.compile(q)
    assert "HAVING" in out.sql


def test_having_qualified_and_bare_produce_equivalent_sql() -> None:
    """Pinning equivalence: the qualified form should not generate any
    extra/different SQL relative to the bare form."""
    cat = _cat()
    bare = cat.compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            having=[Filter(dimension="revenue", op="gt", values=[100])],
        )
    )
    qualified = cat.compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            having=[Filter(dimension="orders.revenue", op="gt", values=[100])],
        )
    )
    assert bare.sql == qualified.sql


def test_having_rejects_unknown_qualified_measure() -> None:
    cat = _cat()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="orders.profit", op="gt", values=[100])],
    )
    with pytest.raises(CompileError, match="HAVING"):
        cat.compile(q)


def test_having_rejects_qualified_dimension_reference() -> None:
    """HAVING is only valid for measures — qualified or otherwise."""
    cat = _cat()
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="orders.region", op="eq", values=["us"])],
    )
    with pytest.raises(CompileError, match="HAVING"):
        cat.compile(q)
