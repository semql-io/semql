"""Tests for ``collapse_unrequested_grain`` — the top-N-per-period foot-gun.

A granularity on a top-N shape (order + limit, neither referencing the
time bucket) fans the result out to top-N *per bucket*. The transform
strips ``time_dimension.granularity`` (keeping ``range`` and everything
else) exactly when that structural signal holds, and is a strict no-op
(same object back) otherwise. Pure ``SemanticQuery -> SemanticQuery``
tests plus the opt-in ``autoplan(..., collapse_unrequested_grain=True)``
wiring and one end-to-end compile check.
"""

from __future__ import annotations

from typing import Any

from semql.autoplan import autoplan, collapse_unrequested_grain
from semql.compile import compile_query
from semql.model import Cube
from semql.spec import CompareWindow, InlineDerived, SemanticQuery, TimeWindow

from .conftest import CONTEXT

_RANGE = ("2026-01-01", "2026-02-01")


def _topn(**overrides: Any) -> SemanticQuery:
    """The motivating repro: 'single most-active entity over the window'
    authored with a per-day grain that fans it out to one row per day."""
    base: dict[str, Any] = {
        "measures": ["user_events.duration"],
        "dimensions": ["user_events.identity_id"],
        "time_dimension": TimeWindow(
            dimension="user_events.start_time",
            granularity="day",
            range=_RANGE,
        ),
        "order": [("user_events.duration", "desc")],
        "limit": 1,
    }
    base.update(overrides)
    return SemanticQuery(**base)


# ---------------------------------------------------------------------------
# The repro: grain stripped, everything else preserved
# ---------------------------------------------------------------------------


def test_repro_strips_granularity_and_preserves_everything_else() -> None:
    q = _topn()
    out = collapse_unrequested_grain(q)

    assert out is not q, "signal holds — must return a new object"
    assert out.time_dimension is not None
    assert out.time_dimension.granularity is None
    assert out.time_dimension.range == _RANGE
    assert out.time_dimension.dimension == "user_events.start_time"
    assert out.measures == q.measures
    assert out.dimensions == q.dimensions
    assert out.order == q.order
    assert out.limit == q.limit

    # Original is unmutated (frozen model, but pin it anyway).
    assert q.time_dimension is not None
    assert q.time_dimension.granularity == "day"


def test_derived_measures_alone_count_as_aggregation() -> None:
    q = _topn(
        measures=[],
        derived_measures=[
            InlineDerived(
                name="total",
                op="sum",
                operands=["user_events.duration", "user_events.idle"],
            )
        ],
        order=[("total", "desc")],
    )
    out = collapse_unrequested_grain(q)
    assert out is not q
    assert out.time_dimension is not None
    assert out.time_dimension.granularity is None


# ---------------------------------------------------------------------------
# Grain preserved: the time bucket IS referenced by an order key
# ---------------------------------------------------------------------------


def test_order_by_qualified_time_ref_keeps_grain() -> None:
    q = _topn(order=[("user_events.start_time", "asc"), ("user_events.duration", "desc")])
    assert collapse_unrequested_grain(q) is q


def test_order_by_bare_field_name_keeps_grain() -> None:
    q = _topn(order=[("start_time", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_order_by_bucket_column_keeps_grain() -> None:
    q = _topn(order=[("start_time_day", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_order_by_qualified_bucket_column_keeps_grain() -> None:
    q = _topn(order=[("user_events.start_time_day", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_order_by_other_granularity_bucket_suffix_keeps_grain() -> None:
    # Conservative match: any granularity suffix reads as "bucket requested".
    q = _topn(order=[("start_time_week", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_order_by_alias_of_time_dimension_keeps_grain() -> None:
    q = _topn(
        aliases={"when": "user_events.start_time"},
        order=[("when", "desc")],
    )
    assert collapse_unrequested_grain(q) is q


# ---------------------------------------------------------------------------
# Grain preserved: signal incomplete or grain explicitly wanted
# ---------------------------------------------------------------------------


def test_no_order_keeps_grain() -> None:
    q = _topn(order=[])
    assert collapse_unrequested_grain(q) is q


def test_no_limit_keeps_grain() -> None:
    q = _topn(limit=None)
    assert collapse_unrequested_grain(q) is q


def test_no_granularity_is_noop() -> None:
    q = _topn(
        time_dimension=TimeWindow(dimension="user_events.start_time", range=_RANGE),
    )
    assert collapse_unrequested_grain(q) is q


def test_no_time_dimension_is_noop() -> None:
    q = _topn(time_dimension=None)
    assert collapse_unrequested_grain(q) is q


def test_fill_nulls_with_keeps_grain() -> None:
    # fill_nulls_with REQUIRES granularity — stripping would build an
    # invalid TimeWindow, and per-bucket gap-filling is the strongest
    # signal the grain is wanted.
    q = _topn(
        dimensions=[],
        time_dimension=TimeWindow(
            dimension="user_events.start_time",
            granularity="day",
            range=_RANGE,
            fill_nulls_with=0,
        ),
    )
    assert collapse_unrequested_grain(q) is q


def test_ungrouped_keeps_grain() -> None:
    q = _topn(measures=[], ungrouped=True, order=[("user_events.identity_id", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_no_measures_keeps_grain() -> None:
    q = _topn(measures=[], order=[("user_events.identity_id", "asc")])
    assert collapse_unrequested_grain(q) is q


def test_compare_mode_keeps_grain() -> None:
    # Compare mode joins current/prior on the time bucket — the grain
    # participates in row identity, so it is left as authored.
    q = _topn(compare=CompareWindow(mode="previous_period"))
    assert collapse_unrequested_grain(q) is q


# ---------------------------------------------------------------------------
# autoplan wiring (opt-in flag; default byte-for-byte unchanged)
# ---------------------------------------------------------------------------


def _orders_topn(**overrides: Any) -> SemanticQuery:
    base: dict[str, Any] = {
        "measures": ["orders.revenue"],
        "dimensions": ["orders.region"],
        "time_dimension": TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=_RANGE,
        ),
        "order": [("orders.revenue", "desc")],
        "limit": 1,
    }
    base.update(overrides)
    return SemanticQuery(**base)


def test_autoplan_flag_off_returns_query_unchanged(catalog: dict[str, Cube]) -> None:
    q = _orders_topn()  # would be collapsed if the flag were on
    plan = autoplan(q, catalog)
    assert plan.query is q
    assert plan.query.time_dimension is not None
    assert plan.query.time_dimension.granularity == "day"


def test_autoplan_flag_on_strips_granularity(catalog: dict[str, Cube]) -> None:
    q = _orders_topn()
    plan = autoplan(q, catalog, collapse_unrequested_grain=True)
    assert plan.query is not q
    assert plan.query.time_dimension is not None
    assert plan.query.time_dimension.granularity is None
    assert plan.query.time_dimension.range == _RANGE
    # The collapse records no CrossSourceDecision — that type is about
    # foreign-cube routing; callers wanting the reason use the function.
    assert plan.decisions == ()


def test_autoplan_flag_on_leaves_requested_grain(catalog: dict[str, Cube]) -> None:
    q = _orders_topn(order=[("orders.created_at", "asc")])
    plan = autoplan(q, catalog, collapse_unrequested_grain=True)
    assert plan.query is q


# ---------------------------------------------------------------------------
# End-to-end: the collapsed query compiles without a time-bucket GROUP BY
# ---------------------------------------------------------------------------


def test_collapsed_query_compiles_without_time_bucket(catalog: dict[str, Cube]) -> None:
    q = _orders_topn()
    collapsed = collapse_unrequested_grain(q)
    out = compile_query(collapsed, catalog, context=CONTEXT)

    assert "created_at_day" not in out.columns
    assert "created_at_day" not in out.sql
    assert "date_trunc" not in out.sql.lower()
    # The window still filters: both range params are bound.
    assert set(out.params.values()) == set(_RANGE)
    # Aggregation over the whole window remains grouped by the entity dim.
    assert "GROUP BY" in out.sql
    assert out.columns == ["region", "revenue"]
