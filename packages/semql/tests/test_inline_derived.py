"""Tests for F10 — inline derived measures (``InlineDerived``).

Three ops over existing catalog measures, composed at query time:

- ``ratio``: ``num_agg(num) / NULLIF(den_agg(den), 0) AS name``
- ``sum``: ``a_agg(a) + b_agg(b) + ... AS name``
- ``diff``: ``a_agg(a) - b_agg(b) AS name``

Operands may span cubes when the join graph connects them
unambiguously and fan-safely (C17): a cross-cube ratio compiles over a
``one_to_one`` join, but is refused when the operand cube is
unreachable, reached two equal-cost ways (ambiguous), or joined so
that an additive operand fans out. The derived measure is addressable
in ``order`` and ``having`` by its ``name``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    InlineDerived,
    Join,
    Measure,
    SemanticQuery,
)
from semql.errors import CompileError


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
            Measure(name="active_seconds", sql="{o}.active_seconds", agg="sum"),
            Measure(name="logged_seconds", sql="{o}.logged_seconds", agg="sum"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )


def _customers_cube() -> Cube:
    return Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="customers",
        alias="c",
        primary_key="id",
        measures=[Measure(name="signups", sql="*", agg="count")],
        dimensions=[Dimension(name="id", sql="{c}.id", type="number")],
    )


def _cat() -> Catalog:
    return Catalog([_orders_cube(), _customers_cube()])


# ---------------------------------------------------------------------------
# Model — arity validators
# ---------------------------------------------------------------------------


def test_inline_ratio_requires_two_operands() -> None:
    with pytest.raises(ValidationError, match=r"(?i)exactly two operands"):
        InlineDerived(name="aov", op="ratio", operands=["orders.revenue"])


def test_inline_diff_requires_two_operands() -> None:
    with pytest.raises(ValidationError, match=r"(?i)exactly two operands"):
        InlineDerived(
            name="idle",
            op="diff",
            operands=["orders.logged_seconds"],
        )


def test_inline_sum_requires_at_least_two_operands() -> None:
    with pytest.raises(ValidationError, match=r"(?i)at least two operands"):
        InlineDerived(name="total", op="sum", operands=["orders.revenue"])


# ---------------------------------------------------------------------------
# Compile — ratio
# ---------------------------------------------------------------------------


def test_inline_ratio_emits_div_with_nullif_guard() -> None:
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="aov", op="ratio", operands=["orders.revenue", "orders.count"]),
        ],
    )
    out = _cat().compile(q)
    assert "aov" in out.columns
    # SUM(amount) / NULLIF(COUNT(*), 0) AS aov
    assert "NULLIF" in out.sql.upper()
    assert "aov" in out.sql


def test_inline_ratio_addressable_in_order() -> None:
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="aov", op="ratio", operands=["orders.revenue", "orders.count"]),
        ],
        order=[("aov", "desc")],
    )
    out = _cat().compile(q)
    assert "ORDER BY aov" in out.sql


def test_inline_ratio_addressable_in_having() -> None:
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="aov", op="ratio", operands=["orders.revenue", "orders.count"]),
        ],
        having=[Filter(dimension="aov", op="gt", values=[10])],
    )
    out = _cat().compile(q)
    assert "HAVING" in out.sql.upper()


# ---------------------------------------------------------------------------
# Compile — sum
# ---------------------------------------------------------------------------


def test_inline_sum_combines_two_measure_aggregates() -> None:
    """``total_seconds = active_seconds + logged_seconds`` —
    catalog-summed operands plus-composed at the outer SELECT."""
    q = SemanticQuery(
        measures=["orders.active_seconds", "orders.logged_seconds"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="total_seconds",
                op="sum",
                operands=["orders.active_seconds", "orders.logged_seconds"],
            ),
        ],
    )
    out = _cat().compile(q)
    assert "total_seconds" in out.columns
    # SUM(...) + SUM(...) AS total_seconds
    assert " + " in out.sql or " +\n" in out.sql


def test_inline_sum_three_operands() -> None:
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count", "orders.active_seconds"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="abs_total",
                op="sum",
                operands=[
                    "orders.revenue",
                    "orders.count",
                    "orders.active_seconds",
                ],
            ),
        ],
    )
    out = _cat().compile(q)
    # Three operands → two ``+`` connectors in the emitted expression.
    assert out.sql.count(" + ") >= 2


# ---------------------------------------------------------------------------
# Compile — diff
# ---------------------------------------------------------------------------


def test_inline_diff_subtracts_second_from_first() -> None:
    """``idle_seconds = logged_seconds - active_seconds`` — the canonical
    diff shape for activity / attendance analytics."""
    q = SemanticQuery(
        measures=["orders.logged_seconds", "orders.active_seconds"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="idle_seconds",
                op="diff",
                operands=["orders.logged_seconds", "orders.active_seconds"],
            ),
        ],
    )
    out = _cat().compile(q)
    assert "idle_seconds" in out.columns
    assert " - " in out.sql


# ---------------------------------------------------------------------------
# Validation — error paths
# ---------------------------------------------------------------------------


def test_inline_cross_cube_unreachable_rejected() -> None:
    """Cross-cube operands whose cubes share no join path are refused —
    ``orders`` and ``customers`` are disconnected in ``_cat()``, so the
    operand cube can't be pulled into the FROM clause."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="cross",
                op="ratio",
                operands=["orders.revenue", "customers.signups"],
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)no join path|unreachable"):
        _cat().compile(q)


# ---------------------------------------------------------------------------
# Cross-cube operands (C17) — reachable + unambiguous + fan-safe
# ---------------------------------------------------------------------------


def _orders_extras_cat() -> Catalog:
    """Orders ⋈ order_extras over a ``one_to_one`` join — neither side
    duplicates, so a cross-cube composition is fan-safe."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[
            Join(
                to="order_extras",
                relationship="one_to_one",
                on="{o}.id = {e}.order_id",
            )
        ],
    )
    extras = Cube(
        name="order_extras",
        dialect=Dialect.POSTGRES,
        table="order_extras",
        alias="e",
        measures=[Measure(name="bonus", sql="{e}.bonus", agg="sum")],
        dimensions=[Dimension(name="order_id", sql="{e}.order_id", type="number")],
    )
    return Catalog([orders, extras])


def test_inline_cross_cube_ratio_over_one_to_one_compiles() -> None:
    """A ratio whose denominator lives on a ``one_to_one``-joined cube
    compiles: the operand cube is pulled into the FROM clause even
    though it is referenced only through the derived measure."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="rev_per_bonus",
                op="ratio",
                operands=["orders.revenue", "order_extras.bonus"],
            ),
        ],
    )
    out = _orders_extras_cat().compile(q)
    assert "rev_per_bonus" in out.columns
    assert "NULLIF" in out.sql.upper()
    # The operand-only cube is joined in even though it is not projected.
    assert "order_extras" in out.sql
    assert "bonus" not in out.columns  # operand, not an output column


def test_inline_cross_cube_fanning_operand_rejected() -> None:
    """An additive operand that fans out across the join is refused, even
    when the fanning measure appears *only* as a derived operand (never
    in the query's own ``measures``)."""
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[
            Join(
                to="items",
                relationship="one_to_many",
                on="{o}.id = {i}.order_id",
            )
        ],
    )
    items = Cube(
        name="items",
        dialect=Dialect.POSTGRES,
        table="items",
        alias="i",
        measures=[Measure(name="qty", sql="{i}.quantity", agg="sum")],
        dimensions=[Dimension(name="sku", sql="{i}.sku", type="string")],
    )
    cat = Catalog([orders, items])
    q = SemanticQuery(
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="rev_per_qty",
                op="ratio",
                operands=["orders.revenue", "items.qty"],
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)fans out"):
        cat.compile(q)


def test_inline_cross_cube_ambiguous_path_rejected() -> None:
    """When the operand cube is reachable two equal-cost ways the
    derivation's value depends on which route the planner picks, so it is
    refused rather than silently resolved."""

    def _oo(name: str, alias: str, *, to: str | None = None) -> Cube:
        joins = [Join(to=to, relationship="one_to_one", on=f"{{{alias}}}.k = x")] if to else []
        return Cube(
            name=name,
            dialect=Dialect.POSTGRES,
            table=name,
            alias=alias,
            measures=[Measure(name="val", sql=f"{{{alias}}}.v", agg="sum")],
            dimensions=[Dimension(name="region", sql=f"{{{alias}}}.region", type="string")],
            joins=joins,
        )

    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        joins=[
            Join(to="b", relationship="one_to_one", on="{o}.k = x"),
            Join(to="c", relationship="one_to_one", on="{o}.k = x"),
        ],
    )
    b = _oo("b", "b", to="d")
    c = _oo("c", "c", to="d")
    d = _oo("d", "d")
    cat = Catalog([orders, b, c, d])
    q = SemanticQuery(
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="rev_per_d",
                op="ratio",
                operands=["orders.revenue", "d.val"],
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)ambiguous"):
        cat.compile(q)


def test_inline_operand_must_be_measure() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="weird",
                op="ratio",
                operands=["orders.revenue", "orders.region"],
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)not a measure"):
        _cat().compile(q)


def test_inline_name_collision_with_existing_column_rejected() -> None:
    """If the derived name matches a column already in the SELECT
    (a measure or dim), the compiler refuses — output columns must be
    unique."""
    q = SemanticQuery(
        measures=["orders.revenue", "orders.count"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(
                name="revenue",  # collides with the measure
                op="ratio",
                operands=["orders.revenue", "orders.count"],
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)collides"):
        _cat().compile(q)


def test_inline_ratio_operand_cant_be_another_ratio() -> None:
    """Nested-ratio composition is refused — the recursion semantics
    are ambiguous (do you SUM the ratios? AVG?). Flatten in the spec."""
    cube = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
            Measure(name="count", sql="*", agg="count"),
            Measure(
                name="aov",
                sql="",
                agg="ratio",
                numerator="revenue",
                denominator="count",
            ),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
    )
    cat = Catalog([cube])
    q = SemanticQuery(
        measures=["orders.aov"],
        dimensions=["orders.region"],
        derived_measures=[
            InlineDerived(name="nested", op="diff", operands=["orders.aov", "orders.revenue"]),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)ratio measure"):
        cat.compile(q)
