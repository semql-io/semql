"""Tests for F10 — inline derived measures (``InlineDerived``).

Three ops over existing catalog measures, composed at query time:

- ``ratio``: ``num_agg(num) / NULLIF(den_agg(den), 0) AS name``
- ``sum``: ``a_agg(a) + b_agg(b) + ... AS name``
- ``diff``: ``a_agg(a) - b_agg(b) AS name``

Phase A: every operand must resolve to a measure on the same cube;
cross-cube refs raise. The derived measure is addressable in
``order`` and ``having`` by its ``name``.
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
    Measure,
    SemanticQuery,
)
from semql.errors import CompileError


def _orders_cube() -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
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
        backend=Dialect.POSTGRES,
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


def test_inline_cross_cube_operands_rejected() -> None:
    """Phase A: every operand must be on the same cube."""
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
    with pytest.raises(CompileError, match=r"(?i)same cube"):
        _cat().compile(q)


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
        backend=Dialect.POSTGRES,
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
