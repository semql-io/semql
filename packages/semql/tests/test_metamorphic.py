"""DuckDB-backed metamorphic suite (W5/§10.6).

Metamorphic testing needs no known-good answer: it asserts a
*relationship* between the results of two related queries that must
hold for any correct engine, then executes both against a real
in-process DuckDB oracle. This is the project's first wrong-results
oracle — snapshot/property tests pin SQL *shape*; these pin SQL
*meaning*.

Five relations, each a classic:

- **P15 — TLP (ternary logic partitioning).** For any predicate ``p``
  over a nullable column, the rows split three disjoint ways:
  ``p`` true, ``p`` false (``NOT p``), and ``p`` NULL (``col IS NULL``).
  The three counts must sum to the unfiltered count.
- **P16 — filter monotonicity.** Conjoining a filter can only shrink
  the result: ``count(a AND b) <= min(count(a), count(b))``.
- **P17 — aggregate/ungrouped consistency.** An additive measure summed
  over its GROUP BY partition equals the same measure computed
  ungrouped (the NULL group is a real group and counts).
- **P18 — segment additivity.** Two complementary segments that
  partition the non-null rows split every additive measure exactly.
- **P19 — limit/order.** ``ORDER BY k LIMIT n`` returns the first
  ``min(n, total)`` rows of the fully-ordered result — a genuine
  prefix, same rows in the same order.

The seed data deliberately carries NULLs in every nullable column so
the three-valued-logic paths (P15) and the NULL group (P17) are
actually exercised.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from hypothesis import given
from hypothesis import strategies as st
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
)
from semql.model import Segment

duckdb = pytest.importorskip("duckdb")


# ---------------------------------------------------------------------------
# Catalog + seed oracle
# ---------------------------------------------------------------------------

_REGIONS: list[str | None] = ["us", "eu", "apac", None]
_STATUSES: list[str | None] = ["shipped", "pending", "cancelled", None]
# One deterministic row per (region, status) combo, cycling an amount
# that includes a NULL so SUM's null-skipping and count(*)'s null-keeping
# diverge (which the relations must respect).
_AMOUNTS: list[int | None] = [100, 20, 60, None, 80, 10, 50, 5]


def _seed_rows() -> list[tuple[int, str | None, str | None, int | None]]:
    rows: list[tuple[int, str | None, str | None, int | None]] = []
    rid = 1
    for i, region in enumerate(_REGIONS):
        for j, status in enumerate(_STATUSES):
            amount = _AMOUNTS[(i * len(_STATUSES) + j) % len(_AMOUNTS)]
            rows.append((rid, region, status, amount))
            rid += 1
    return rows


def _metamorphic_catalog() -> Catalog:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        primary_key="id",
        measures=[
            Measure(name="count", sql="*", agg="count"),
            Measure(name="revenue", sql="{o}.amount", agg="sum"),
        ],
        dimensions=[
            Dimension(name="id", sql="{o}.id", type="number"),
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        # big + small partition exactly the rows whose amount is non-null.
        segments=[
            Segment(name="big", sql="{o}.amount >= 50"),
            Segment(name="small", sql="{o}.amount < 50"),
        ],
    )
    return Catalog([cube])


class _Oracle:
    """A seeded DuckDB connection plus a compile-and-execute helper."""

    def __init__(self) -> None:
        self.catalog = _metamorphic_catalog()
        self.con = duckdb.connect()
        self.con.execute(
            "CREATE TABLE orders(id INTEGER, region VARCHAR, status VARCHAR, amount INTEGER)"
        )
        self.con.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?)",
            _seed_rows(),
        )

    def run(self, query: SemanticQuery) -> list[tuple[Any, ...]]:
        out = self.catalog.compile(query)
        return cast(
            "list[tuple[Any, ...]]",
            self.con.execute(out.sql, out.params).fetchall(),
        )

    def scalar(self, query: SemanticQuery) -> Any:
        """First column of the single aggregate row (0 if empty)."""
        rows = self.run(query)
        return rows[0][0] if rows else 0


@pytest.fixture(scope="module")
def oracle() -> _Oracle:
    # Module scope: built once, read-only thereafter — safe to reuse
    # across Hypothesis examples (no per-example function-scoped-fixture
    # health-check warning, and no re-seed cost per example).
    return _Oracle()


# ---------------------------------------------------------------------------
# P15 — Ternary Logic Partitioning
# ---------------------------------------------------------------------------

_NULLABLE_DIMS = ["orders.region", "orders.status"]


@given(
    dim=st.sampled_from(_NULLABLE_DIMS),
    value=st.sampled_from(["us", "eu", "apac", "shipped", "pending", "cancelled", "nonexistent"]),
)
def test_p15_tlp_partitions_the_row_set(oracle: _Oracle, dim: str, value: str) -> None:
    """``p`` / ``NOT p`` / ``p IS NULL`` are disjoint and exhaustive, so
    their counts sum to the unfiltered count regardless of predicate."""
    total = oracle.scalar(SemanticQuery(measures=["orders.count"]))
    n_true = oracle.scalar(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension=dim, op="eq", values=[value])],
        )
    )
    n_false = oracle.scalar(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension=dim, op="neq", values=[value])],
        )
    )
    n_null = oracle.scalar(
        SemanticQuery(
            measures=["orders.count"],
            filters=[Filter(dimension=dim, op="is_null", values=[])],
        )
    )
    assert n_true + n_false + n_null == total


# ---------------------------------------------------------------------------
# P16 — Filter monotonicity
# ---------------------------------------------------------------------------


def _candidate_filters() -> list[Filter]:
    return [
        Filter(dimension="orders.region", op="eq", values=["us"]),
        Filter(dimension="orders.status", op="eq", values=["shipped"]),
        Filter(dimension="orders.status", op="neq", values=["cancelled"]),
        Filter(dimension="orders.amount", op="gte", values=[50]),
        Filter(dimension="orders.amount", op="lt", values=[50]),
        Filter(dimension="orders.region", op="not_null", values=[]),
    ]


@given(a=st.sampled_from(_candidate_filters()), b=st.sampled_from(_candidate_filters()))
def test_p16_conjoining_a_filter_never_grows_the_result(
    oracle: _Oracle, a: Filter, b: Filter
) -> None:
    """A conjunctive filter is monotone: the AND of two predicates
    selects no more rows than either predicate alone."""
    n_a = oracle.scalar(SemanticQuery(measures=["orders.count"], filters=[a]))
    n_b = oracle.scalar(SemanticQuery(measures=["orders.count"], filters=[b]))
    n_ab = oracle.scalar(SemanticQuery(measures=["orders.count"], filters=[a, b]))
    assert n_ab <= min(n_a, n_b)


# ---------------------------------------------------------------------------
# P17 — Aggregate / ungrouped consistency
# ---------------------------------------------------------------------------


@given(
    dim=st.sampled_from(["orders.region", "orders.status"]),
    measure=st.sampled_from(["orders.count", "orders.revenue"]),
)
def test_p17_grouped_measure_sums_to_ungrouped_total(
    oracle: _Oracle, dim: str, measure: str
) -> None:
    """An additive measure grouped by any dimension sums back to its
    ungrouped total — the NULL bucket is a real group and is counted."""
    ungrouped = oracle.scalar(SemanticQuery(measures=[measure]))
    grouped = oracle.run(SemanticQuery(measures=[measure], dimensions=[dim]))
    # The measure is the last column (dimension is projected first).
    assert sum(row[-1] for row in grouped) == ungrouped


# ---------------------------------------------------------------------------
# P18 — Segment additivity
# ---------------------------------------------------------------------------


@given(measure=st.sampled_from(["orders.count", "orders.revenue"]))
def test_p18_complementary_segments_partition_every_measure(oracle: _Oracle, measure: str) -> None:
    """``big`` (amount >= 50) and ``small`` (amount < 50) partition the
    amount-non-null rows, so any additive measure over the two segments
    sums to the measure over that same non-null population."""
    big = oracle.scalar(SemanticQuery(measures=[measure], segments=["orders.big"]))
    small = oracle.scalar(SemanticQuery(measures=[measure], segments=["orders.small"]))
    whole = oracle.scalar(
        SemanticQuery(
            measures=[measure],
            filters=[Filter(dimension="orders.amount", op="not_null", values=[])],
        )
    )
    assert big + small == whole


# ---------------------------------------------------------------------------
# P19 — Limit / order
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=0, max_value=6),
    direction=st.sampled_from(["asc", "desc"]),
)
def test_p19_limit_returns_a_prefix_of_the_ordered_result(
    oracle: _Oracle, n: int, direction: str
) -> None:
    """``ORDER BY revenue <dir> LIMIT n`` returns the first
    ``min(n, total)`` rows of the fully-ordered result — identical rows
    in identical order."""
    order: list[tuple[str, Any]] = [("orders.revenue", direction), ("orders.region", "asc")]
    full = oracle.run(
        SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"], order=order)
    )
    limited = oracle.run(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            order=order,
            limit=n,
        )
    )
    assert len(limited) == min(n, len(full))
    assert limited == full[:n]
