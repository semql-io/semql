"""Pathological & edge-case coverage for SemanticQuery + Cube + Catalog.

Hand-written examples that pin *intended* behaviour at the boundaries:
empty queries, degenerate limits, hostile filter values, ambiguous
catalogs, and the refusal paths (alias collision, fan-out, left-join
dimension misuse). Properties pin invariants; these pin the named cases
a reader should be able to find.
"""

from __future__ import annotations

import time

import pytest
from semql import (
    MAX_UNGROUPED_ROWS,
    BoolExpr,
    Catalog,
    CompileError,
    Cube,
    Dialect,
    Dimension,
    Filter,
    FilterTypeError,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    UnknownIdentifierError,
    is_read_only_statement,
)
from semql.cnf import to_cnf

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _orders() -> Cube:
    return Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )


def _single() -> Catalog:
    return Catalog([_orders()])


# ---------------------------------------------------------------------------
# SemanticQuery degenerate shapes
# ---------------------------------------------------------------------------


def test_empty_query_refused() -> None:
    with pytest.raises(CompileError, match=r"(?i)empty"):
        _single().compile(SemanticQuery())


def test_limit_zero_compiles_to_limit_0() -> None:
    out = _single().compile(SemanticQuery(measures=["orders.revenue"], limit=0))
    assert "LIMIT 0" in out.sql


def test_huge_limit_is_preserved() -> None:
    out = _single().compile(SemanticQuery(measures=["orders.revenue"], limit=10_000_000))
    assert "LIMIT 10000000" in out.sql


def test_ungrouped_without_limit_refused_with_cap_in_message() -> None:
    with pytest.raises(CompileError, match=rf"{MAX_UNGROUPED_ROWS}"):
        _single().compile(SemanticQuery(dimensions=["orders.region"], ungrouped=True))


def test_ungrouped_at_the_cap_compiles() -> None:
    out = _single().compile(
        SemanticQuery(dimensions=["orders.region"], ungrouped=True, limit=MAX_UNGROUPED_ROWS)
    )
    assert "GROUP BY" not in out.sql
    assert f"LIMIT {MAX_UNGROUPED_ROWS}" in out.sql


def test_unknown_field_names_the_ref_and_lists_known() -> None:
    with pytest.raises(UnknownIdentifierError) as ei:
        _single().compile(SemanticQuery(measures=["orders.nope"]))
    msg = str(ei.value)
    assert "nope" in msg
    # The error is actionable — it enumerates the real fields.
    assert "revenue" in msg


def test_filter_targeting_a_measure_is_refused() -> None:
    with pytest.raises((FilterTypeError, CompileError)):
        _single().compile(
            SemanticQuery(
                measures=["orders.revenue"],
                filters=[Filter(dimension="orders.revenue", op="gt", values=[1])],
            )
        )


def test_having_on_non_measure_refused() -> None:
    with pytest.raises(CompileError, match=r"(?i)having"):
        _single().compile(
            SemanticQuery(
                dimensions=["orders.region"],
                having=[Filter(dimension="region", op="gt", values=[1])],
            )
        )


def test_deeply_nested_boolexpr_filter_compiles() -> None:
    where = BoolExpr(
        op="and",
        children=[
            Filter(dimension="orders.region", op="eq", values=["us"]),
            BoolExpr(
                op="or",
                children=[
                    Filter(dimension="orders.status", op="eq", values=["a"]),
                    BoolExpr(
                        op="not",
                        children=[Filter(dimension="orders.status", op="eq", values=["b"])],
                    ),
                ],
            ),
        ],
    )
    out = _single().compile(SemanticQuery(measures=["orders.count"], where=where))
    assert is_read_only_statement(out.sql)
    # Three leaf values → three bound params, none inlined.
    assert len(out.params) == 3


def test_equal_time_window_endpoints_compile() -> None:
    """Half-open [t, t) is an empty but legal window — must not crash."""
    out = _single().compile(
        SemanticQuery(
            measures=["orders.count"],
            time_dimension=TimeWindow(
                dimension="orders.created_at", range=("2026-01-01", "2026-01-01")
            ),
        )
    )
    assert is_read_only_statement(out.sql)


# ---------------------------------------------------------------------------
# Hostile filter values must round-trip as bind params, never as SQL text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "'; DROP TABLE orders; --",
        "' OR '1'='1",
        "%(p0)s",  # the Postgres placeholder syntax itself — must come back as data
        "@p",  # other driver paramstyles
        "?",
        "{o}.amount",  # a SemQL substitution token — must NOT be interpolated
        "\\",
        "100%",
        "ʼ＇﻿🦊",  # homoglyph quotes + zero-width + emoji
        "x" * 2000,  # oversized
    ],
)
def test_hostile_filter_value_is_parameterised(raw: str) -> None:
    # Sentinel trick (property-testing.md §1.4): wrap the value in a unique
    # marker. A plain ``raw not in sql`` check false-positives when ``raw``
    # equals the placeholder syntax (e.g. ``%(p0)s``); the marker can't
    # coincide, so its absence proves the value was bound, not spliced.
    marker = "⟦S⟧"
    sentinel = f"{marker}{raw}{marker}"
    out = _single().compile(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension="orders.region", op="eq", values=[sentinel])],
        )
    )
    assert sentinel in out.params.values()
    assert marker not in out.sql


# ---------------------------------------------------------------------------
# Cube / Catalog construction & multi-cube refusals
# ---------------------------------------------------------------------------


def test_duplicate_cube_names_refused_at_construction() -> None:
    with pytest.raises(ValueError, match=r"(?i)duplicate cube"):
        Catalog([_orders(), _orders()])


def test_cubes_sharing_an_alias_refused_when_joined() -> None:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="cid", sql="{o}.cid", type="string")],
        joins=[Join(to="customers", relationship="many_to_one", on="{o}.cid = {o}.id")],
    )
    customers = Cube(
        name="customers",
        dialect=Dialect.POSTGRES,
        table="customers",
        alias="o",  # collision
        dimensions=[Dimension(name="name", sql="{o}.name", type="string")],
    )
    with pytest.raises(CompileError, match=r"(?i)share.*alias"):
        Catalog([orders, customers]).compile(
            SemanticQuery(measures=["orders.revenue"], dimensions=["customers.name"])
        )


def _fanout_catalog() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="id", sql="{o}.id", type="string")],
        joins=[Join(to="line_items", relationship="one_to_many", on="{o}.id = {li}.oid")],
    )
    line_items = Cube(
        name="line_items",
        dialect=Dialect.POSTGRES,
        table="line_items",
        alias="li",
        measures=[Measure(name="qty", sql="{li}.qty", agg="sum")],
        dimensions=[Dimension(name="sku", sql="{li}.sku", type="string")],
    )
    return Catalog([orders, line_items])


def test_additive_measure_fans_out_across_one_to_many_refused() -> None:
    with pytest.raises(CompileError, match=r"(?i)fans?\s+out"):
        _fanout_catalog().compile(
            SemanticQuery(measures=["orders.revenue"], dimensions=["line_items.sku"])
        )


def test_left_joined_cube_dimension_in_dimensions_refused() -> None:
    with pytest.raises(CompileError, match=r"(?i)dimensions"):
        _fanout_catalog().compile(
            SemanticQuery(
                measures=["orders.revenue"],
                dimensions=["line_items.sku"],
                left_joins=["line_items"],
            )
        )


# ---------------------------------------------------------------------------
# Blowup / recursion tripwires (W5/§7). These are structural guards, not
# latency benchmarks: the wall-clock bound is deliberately loose (an
# exponential regression takes minutes or OOMs, not milliseconds), so it
# never flakes but still trips on genuine blowup. The hard assertion is on
# the *shape* — bounded clause count, no ``RecursionError``.
# ---------------------------------------------------------------------------


def _or_leaf(i: int) -> Filter:
    return Filter(dimension="orders.status", op="eq", values=[f"v{i}"])


def _cnf_clause_count(node: BoolExpr | Filter) -> int:
    """CNF is a top-level AND of OR-clauses. A bare Filter or a single
    OR node is one clause; an AND has one clause per child."""
    if isinstance(node, Filter):
        return 1
    return len(node.children) if node.op == "and" else 1


def test_cnf_nested_or_stays_flat_and_bounded() -> None:
    """A 30-leaf right-nested OR must collapse to a single flat clause,
    not distribute into an exponential AND-of-ORs. If flattening ever
    regresses this either explodes the clause count or times out."""
    node: BoolExpr | Filter = _or_leaf(29)
    for i in range(28, -1, -1):
        node = BoolExpr(op="or", children=[_or_leaf(i), node])

    start = time.perf_counter()
    out = to_cnf(node)
    elapsed = time.perf_counter() - start

    # Pure OR never distributes: 30 leaves -> exactly one OR-clause.
    assert _cnf_clause_count(out) == 1
    assert isinstance(out, BoolExpr) and out.op == "or"
    assert len(out.children) == 30
    assert elapsed < 1.0


def test_deep_filter_nesting_survives_recursion() -> None:
    """A filter tree far deeper than any human-written query must compile
    without tripping Python's recursion limit. 250 levels is well past
    real usage yet clear of the ~900-frame ceiling, so a regression that
    adds a recursive frame per level surfaces here before production."""
    depth = 250
    where = BoolExpr(op="and", children=[_or_leaf(0), _or_leaf(1)])
    for i in range(2, depth):
        where = BoolExpr(op="and", children=[_or_leaf(i), where])

    out = _single().compile(SemanticQuery(measures=["orders.count"], where=where))
    assert is_read_only_statement(out.sql)
    assert len(out.params) == depth
