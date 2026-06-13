"""A2 — partition routing must compare timestamps by *instant*, not bytes.

Architecture review (2026-06) A2 flagged two things about
time-partitioned source routing (:mod:`semql.partition`):

1. ``TimeWindow.range`` was documented "inclusive (start, end)", but the
   compiler emits ``time_dim >= start AND time_dim < end``
   (``compile.py``) — a *half-open* ``[start, end)`` window. The routing
   in ``_ranges_intersect`` is also half-open, so the two agree; the
   docstring was simply wrong. (Fixed in ``spec.py``.)

2. ``_ranges_intersect`` and the ``TimePartitionedSource`` range-ordering
   validator compared the endpoints as **lexical strings**. For
   zero-padded same-offset ISO-8601 that happens to match chronological
   order, so the bug stayed hidden — but the moment two endpoints carry
   different UTC offsets (or differing precision) the byte order diverges
   from the instant order. A query window expressed in ``-05:00`` whose
   instants land entirely inside the ``live`` source was routed to the
   *wrong* table: ``live`` (every matching row) was skipped and
   ``archive`` (no matching row) was scanned, so the query silently
   returned empty.

These tests pin instant-based comparison: routing follows the real
chronological position of the endpoints regardless of textual offset or
precision.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimePartition,
    TimePartitionedSource,
    TimeWindow,
)
from semql.partition import _ranges_intersect, select_physical_sources


def _utc_split_cube() -> Cube:
    """orders split at the 2024-01-01T00:00:00Z instant, written in UTC."""
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        alias="o",
        time_partition=TimePartition(time_dimension="placed_at"),
        physical_sources=[
            TimePartitionedSource(
                name="archive",
                table="orders_archive",
                range_start=None,
                range_end="2024-01-01T00:00:00+00:00",
            ),
            TimePartitionedSource(
                name="live",
                table="orders_live",
                range_start="2024-01-01T00:00:00+00:00",
                range_end=None,
            ),
        ],
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="placed_at", sql="{o}.placed_at")],
    )


# ---------------------------------------------------------------------------
# Unit — _ranges_intersect compares instants, not bytes
# ---------------------------------------------------------------------------


def test_ranges_intersect_honours_utc_offsets() -> None:
    """A window written in -05:00 whose instants are all after the UTC
    split must intersect the *live* range and miss the *archive* range,
    even though its text sorts before '2024-...'."""
    # [2024-01-01T01:00Z, 2024-01-01T04:00Z) — entirely inside live.
    q = ("2023-12-31T20:00:00-05:00", "2023-12-31T23:00:00-05:00")
    assert _ranges_intersect(q, ("2024-01-01T00:00:00+00:00", None)) is True
    assert _ranges_intersect(q, (None, "2024-01-01T00:00:00+00:00")) is False


def test_ranges_intersect_treats_equal_instants_as_half_open() -> None:
    """'2024-01-01' and '2024-01-01T00:00:00' are the same instant; the
    half-open touching rule must hold across the textual difference."""
    # Query high == source low (same instant, different text): half-open
    # → no intersection (the boundary belongs to the lower source).
    assert _ranges_intersect(("2023-01-01", "2024-01-01"), ("2024-01-01T00:00:00", None)) is False
    # Query high one instant past the boundary → intersects.
    assert (
        _ranges_intersect(("2023-01-01", "2024-01-01T00:00:01"), ("2024-01-01T00:00:00", None))
        is True
    )


# ---------------------------------------------------------------------------
# End-to-end — the wrong-table routing as a compiled query
# ---------------------------------------------------------------------------


def test_offset_window_routes_to_live_not_archive() -> None:
    """The compiled query for an offset window inside live must scan
    orders_live and not orders_archive."""
    cube = _utc_split_cube()
    matched = select_physical_sources(
        cube,
        TimeWindow(
            dimension="orders.placed_at",
            range=("2023-12-31T20:00:00-05:00", "2023-12-31T23:00:00-05:00"),
        ),
    )
    assert [s.name for s in matched] == ["live"]

    compiled = Catalog(cubes=[cube]).compile(
        SemanticQuery(
            measures=["orders.revenue"],
            time_dimension=TimeWindow(
                dimension="orders.placed_at",
                range=("2023-12-31T20:00:00-05:00", "2023-12-31T23:00:00-05:00"),
            ),
        )
    )
    assert "orders_live" in compiled.sql
    assert "orders_archive" not in compiled.sql
    assert compiled.physical_sources_hit == ("live",)


# ---------------------------------------------------------------------------
# Model validation — range ordering is an instant check, not a byte check
# ---------------------------------------------------------------------------


def test_range_ordering_rejects_offset_inverted_endpoints() -> None:
    """range_start < range_end lexically but start is the *later* instant
    once offsets are applied → an empty interval that must be refused."""
    # start = 2024-01-01T00:00Z ; end = 2024-01-01T02:00+05:00 = 2023-12-31T21:00Z
    with pytest.raises(ValidationError, match=r"(?i)must be strictly less than"):
        TimePartitionedSource(
            name="inverted",
            table="orders_x",
            range_start="2024-01-01T00:00:00+00:00",
            range_end="2024-01-01T02:00:00+05:00",
        )
