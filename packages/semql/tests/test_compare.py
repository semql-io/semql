"""Tests for compare windows.

Compare semantics: two CTEs (``current``, ``prior``) over the same
inner query body with different time ranges, joined via FULL OUTER
JOIN on the dimensions. The outer SELECT exposes — per measure —
``{m}_current``, ``{m}_prior``, ``{m}_delta``, ``{m}_pct_change``.
These four column names are part of the compiler's contract: ``order``
and ``having`` may reference them.

``previous_period`` derives the prior window from the current window
at compile time (subtract the duration). ``explicit`` requires the
caller to supply ``range``.
"""

from __future__ import annotations

import duckdb
import pytest
from semql import (
    Catalog,
    CompareWindow,
    CompileError,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)


def _cat() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
        table="orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    return Catalog([orders])


def _basic_compare_query() -> SemanticQuery:
    return SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_compare_without_time_dimension_rejected() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        compare=CompareWindow(),
    )
    with pytest.raises(CompileError, match=r"(?i)compare.*time_dimension"):
        _cat().compile(q)


def test_compare_without_measures_rejected() -> None:
    """Compare with no measures has nothing to delta — reject."""
    q = SemanticQuery(
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
        ),
        compare=CompareWindow(),
    )
    with pytest.raises(CompileError, match=r"(?i)compare.*measure"):
        _cat().compile(q)


def test_compare_ungrouped_rejected() -> None:
    """ungrouped is row-listing; compare aggregates by definition."""
    q = SemanticQuery(
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
        ),
        compare=CompareWindow(),
        ungrouped=True,
        limit=10,
    )
    with pytest.raises(CompileError):
        _cat().compile(q)


def test_compare_explicit_without_range_rejected() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
        ),
        compare=CompareWindow(mode="explicit"),
    )
    with pytest.raises(CompileError, match=r"explicit"):
        _cat().compile(q)


# ---------------------------------------------------------------------------
# Happy path — SQL structure
# ---------------------------------------------------------------------------


def test_compare_emits_cte_shell() -> None:
    out = _cat().compile(_basic_compare_query())
    sql = out.sql
    assert "WITH " in sql.upper()
    assert "current" in sql
    assert "prior" in sql
    assert "FULL OUTER JOIN" in sql.upper()


def test_compare_outer_select_coalesces_dimensions() -> None:
    out = _cat().compile(_basic_compare_query())
    # Either order of the join sides is acceptable.
    assert "COALESCE" in out.sql.upper()
    assert "region" in out.sql


def test_compare_outputs_four_columns_per_measure() -> None:
    out = _cat().compile(_basic_compare_query())
    expected = {"region", "revenue_current", "revenue_prior", "revenue_delta", "revenue_pct_change"}
    assert expected.issubset(set(out.columns))


def test_compare_dimension_column_appears_first() -> None:
    out = _cat().compile(_basic_compare_query())
    assert out.columns[0] == "region"


def test_compare_pct_change_has_div_by_zero_guard() -> None:
    """The pct_change column cannot perform naked division — it must
    guard against zero prior (``CASE WHEN`` or ``NULLIF`` are both
    acceptable shapes)."""
    out = _cat().compile(_basic_compare_query())
    sql_upper = out.sql.upper()
    assert "CASE" in sql_upper or "NULLIF" in sql_upper


# ---------------------------------------------------------------------------
# previous_period range computation
# ---------------------------------------------------------------------------


def test_compare_previous_period_derives_prior_range() -> None:
    """For a 31-day current window, prior starts 31 days earlier and
    ends at the current window's start."""
    out = _cat().compile(_basic_compare_query())
    values = [str(v) for v in out.params.values()]
    # current bounds (literal forms accepted) + prior start
    assert any("2026-01-01" in v for v in values)
    assert any("2026-02-01" in v for v in values)
    assert any("2025-12-01" in v for v in values)


def test_compare_binds_four_distinct_time_params() -> None:
    """Two CTEs × two bounds = four time-param bindings. Filters bind
    once per value; time-window bounds bind per CTE."""
    out = _cat().compile(_basic_compare_query())
    time_like = [
        v
        for v in out.params.values()
        if isinstance(v, str) and (v.startswith("2025") or v.startswith("2026"))
    ]
    assert len(time_like) == 4


def test_compare_explicit_mode_uses_supplied_range() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
        ),
        compare=CompareWindow(mode="explicit", range=("2025-01-01", "2025-02-01")),
    )
    out = _cat().compile(q)
    values = [str(v) for v in out.params.values()]
    assert any("2025-01-01" in v for v in values)
    assert any("2025-02-01" in v for v in values)


# ---------------------------------------------------------------------------
# Filter values shared between both CTEs
# ---------------------------------------------------------------------------


def test_compare_filter_value_bound_once_referenced_twice() -> None:
    """A filter ``region='us'`` shows up in BOTH CTE bodies, but the
    value binds once. The placeholder is referenced twice in the SQL,
    once per CTE."""
    q = _basic_compare_query().model_copy(
        update={
            "filters": [Filter(dimension="orders.region", op="eq", values=["us"])],
        }
    )
    out = _cat().compile(q)
    # `us` appears exactly once among the bound values.
    us_bindings = [v for v in out.params.values() if v == "us"]
    assert len(us_bindings) == 1
    # The placeholder shows up twice (one per CTE WHERE).
    placeholder_us = next(name for name, val in out.params.items() if val == "us")
    assert out.sql.count(f"%({placeholder_us})s") == 2


# ---------------------------------------------------------------------------
# Multiple measures — all four columns per measure
# ---------------------------------------------------------------------------


def test_compare_multiple_measures_each_gets_four_columns() -> None:
    q = _basic_compare_query().model_copy(update={"measures": ["orders.revenue", "orders.count"]})
    out = _cat().compile(q)
    for m in ("revenue", "count"):
        for suffix in ("current", "prior", "delta", "pct_change"):
            assert f"{m}_{suffix}" in out.columns, f"missing {m}_{suffix}"


# ---------------------------------------------------------------------------
# Order + Having against compare column names
# ---------------------------------------------------------------------------


def test_compare_order_by_delta_column() -> None:
    q = _basic_compare_query().model_copy(update={"order": [("revenue_delta", "desc")]})
    out = _cat().compile(q)
    assert "ORDER BY" in out.sql.upper()
    assert "revenue_delta" in out.sql


def test_compare_order_by_pct_change_column() -> None:
    q = _basic_compare_query().model_copy(update={"order": [("revenue_pct_change", "asc")]})
    out = _cat().compile(q)
    assert "revenue_pct_change" in out.sql


def test_compare_order_by_unknown_column_rejected() -> None:
    q = _basic_compare_query().model_copy(update={"order": [("nonexistent_col", "asc")]})
    with pytest.raises(CompileError, match=r"(?i)order"):
        _cat().compile(q)


# ---------------------------------------------------------------------------
# Time-breakdown (granularity) in compare
# ---------------------------------------------------------------------------


def test_compare_with_granularity_includes_time_column() -> None:
    """A compare with granularity adds the time bucket as a dimension —
    COALESCE on it joins current/prior. The output retains the
    ``{time_dim}_{gran}`` column."""
    q = _basic_compare_query().model_copy(
        update={
            "time_dimension": TimeWindow(
                dimension="orders.created_at",
                granularity="day",
                range=("2026-01-01", "2026-02-01"),
            ),
        }
    )
    out = _cat().compile(q)
    assert "created_at_day" in out.columns
    assert "revenue_current" in out.columns


# ---------------------------------------------------------------------------
# F3 — synthetic ``compare.<measure>.<facet>`` refs in order / having
# ---------------------------------------------------------------------------


def test_compare_order_by_synthetic_delta_ref() -> None:
    """``compare.revenue.delta`` is the readable form that rewrites to
    the ``revenue_delta`` outer column."""
    q = _basic_compare_query().model_copy(update={"order": [("compare.revenue.delta", "desc")]})
    out = _cat().compile(q)
    assert "ORDER BY" in out.sql.upper()
    assert "revenue_delta" in out.sql


def test_compare_order_by_synthetic_pct_change_ref() -> None:
    q = _basic_compare_query().model_copy(update={"order": [("compare.revenue.pct_change", "asc")]})
    out = _cat().compile(q)
    assert "revenue_pct_change" in out.sql


def test_compare_order_by_synthetic_current_and_prior_refs() -> None:
    """The synthetic form covers all four compare-derived columns,
    not just delta."""
    q = _basic_compare_query().model_copy(
        update={
            "order": [
                ("compare.revenue.current", "desc"),
                ("compare.revenue.prior", "asc"),
            ]
        }
    )
    out = _cat().compile(q)
    assert "revenue_current" in out.sql
    assert "revenue_prior" in out.sql


@pytest.mark.parametrize(
    ("facet", "threshold", "expected_regions"),
    [
        ("current", 6, {"north"}),
        ("prior", 4, {"north"}),
        ("delta", 0, {"north", "current-only"}),
        ("pct_change", 50, {"north"}),
    ],
)
def test_compare_output_filters_execute_after_projection_with_order_and_limit(
    facet: str, threshold: float, expected_regions: set[str]
) -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(),
        having=[Filter(dimension=f"compare.revenue.{facet}", op="gt", values=[threshold])],
        order=[("compare.revenue.delta", "desc")],
        limit=1,
    )
    compiled = Catalog([cube]).compile(query)
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE orders (region VARCHAR, amount DOUBLE, created_at TIMESTAMP)")
        con.execute(
            "INSERT INTO orders VALUES "
            "('north', 10, '2026-01-10'), ('north', 5, '2025-12-10'), "
            "('current-only', 3, '2026-01-11'), ('prior-only', 4, '2025-12-11')"
        )
        rows = con.execute(compiled.sql, compiled.params).fetchall()
        assert len(rows) == 1
        assert rows[0][0] in expected_regions
        assert rows[0][1:] == (
            10.0 if rows[0][0] == "north" else 3.0,
            5.0 if rows[0][0] == "north" else None,
            5.0 if rows[0][0] == "north" else 3.0,
            100.0 if rows[0][0] == "north" else None,
        )
    finally:
        con.close()


def test_compare_null_safe_identity_for_multiple_keys_and_one_sided_groups() -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="channel", sql="{o}.channel", type="string"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region", "orders.channel"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(),
    )
    compiled = Catalog([cube]).compile(query)
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            "CREATE TABLE orders "
            "(region VARCHAR, channel VARCHAR, amount DOUBLE, created_at TIMESTAMP)"
        )
        con.execute(
            "INSERT INTO orders VALUES "
            "(NULL, NULL, 5, '2026-01-10'), (NULL, NULL, 2, '2025-12-10'), "
            "(NULL, 'web', 7, '2026-01-11'), (NULL, 'web', 3, '2025-12-11'), "
            "('west', NULL, 11, '2026-01-12'), ('east', 'store', 13, '2025-12-12')"
        )
        rows = con.execute(compiled.sql, compiled.params).fetchall()
        actual = {
            (region, channel): (current, prior, delta, pct)
            for region, channel, current, prior, delta, pct in rows
        }
        assert actual == {
            (None, None): (5.0, 2.0, 3.0, 150.0),
            (None, "web"): (7.0, 3.0, 4.0, 400.0 / 3.0),
            ("west", None): (11.0, None, 11.0, None),
            ("east", "store"): (None, 13.0, -13.0, None),
        }
        assert len(rows) == len(actual)
    finally:
        con.close()


@pytest.mark.parametrize("facet", ["current", "prior", "delta", "pct_change"])
@pytest.mark.parametrize("grouped", [False, True], ids=["no-dimension", "grouped"])
def test_compare_filters_execute_for_all_facets_and_grouping_shapes(
    facet: str, grouped: bool
) -> None:
    cube = Cube(
        name="orders",
        dialect=Dialect.DUCKDB,
        table="orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    filt = Filter(dimension=f"compare.sales.{facet}", op="gt", values=[0])
    query = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"] if grouped else [],
        aliases={"sales": "orders.revenue"},
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
        ),
        compare=CompareWindow(),
        having=[filt],
        order=[("compare.sales.delta", "desc")],
        limit=2,
    )
    compiled = Catalog([cube]).compile(query)
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE orders (region VARCHAR, amount DOUBLE, created_at TIMESTAMP)")
        con.execute(
            "INSERT INTO orders VALUES "
            "('a', 10, '2026-01-10'), ('a', 5, '2025-12-10'), "
            "('b', 2, '2026-01-11'), ('b', 4, '2025-12-11')"
        )
        cursor = con.execute(compiled.sql, compiled.params)
        names = [item[0] for item in cursor.description]
        rows = cursor.fetchall()
        facet_index = {
            "current": names.index("sales_current"),
            "prior": names.index("sales_prior"),
            "delta": names.index("sales_delta"),
            "pct_change": names.index("sales_pct_change"),
        }[facet]
        assert len(rows) == (1 if facet in {"delta", "pct_change"} else (2 if grouped else 1))
        assert rows[0][facet_index] is not None and rows[0][facet_index] > 0
        assert [meta.name for meta in compiled.column_meta] == compiled.columns
        assert "sales_current" in names and "sales_delta" in names
        assert all(len(row) == len(names) for row in rows)
        if grouped:
            assert [row[0] for row in rows] == (
                ["a"] if facet in {"delta", "pct_change"} else ["a", "b"]
            )
    finally:
        con.close()


def test_compare_synthetic_ref_unknown_measure_rejected() -> None:
    """``compare.<X>.delta`` where X is not in the query's measures
    raises with a pointer at the actual measure list."""
    q = _basic_compare_query().model_copy(update={"order": [("compare.nonexistent.delta", "desc")]})
    with pytest.raises(CompileError, match=r"(?i)not in this query's measures"):
        _cat().compile(q)


def test_compare_synthetic_ref_unknown_facet_rejected() -> None:
    """``compare.X.<facet>`` where facet isn't current/prior/delta/pct_change."""
    q = _basic_compare_query().model_copy(update={"order": [("compare.revenue.banana", "desc")]})
    with pytest.raises(CompileError, match=r"(?i)unknown compare facet"):
        _cat().compile(q)


def test_compare_synthetic_ref_in_order_without_compare_rejected() -> None:
    """Using the synthetic form when ``compare`` isn't set surfaces a
    pointed error rather than the cryptic 'unknown cube compare'."""
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        order=[("compare.revenue.delta", "desc")],
    )
    with pytest.raises(CompileError, match=r"(?i)only valid when.*compare"):
        _cat().compile(q)


def test_compare_synthetic_ref_in_having_without_compare_rejected() -> None:
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        having=[Filter(dimension="compare.revenue.delta", op="gt", values=[100])],
    )
    with pytest.raises(CompileError, match=r"(?i)only valid when.*compare"):
        _cat().compile(q)
