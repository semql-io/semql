"""Tests for F1 — anti-join via ``SemanticQuery.left_joins``.

The canonical absent-row pattern: identify entities (spine) whose
fact records are missing for a window. The catalog declares the FK
on the fact side; ``left_joins=["fact_cube"]`` lets the BFS walk
bidirectionally so the spine→facts edge resolves, and the standard
``Filter(op="is_null")`` then expresses the anti-join.
"""

from __future__ import annotations

import pytest
from semql import (
    Backend,
    Catalog,
    Cube,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
)
from semql.errors import CompileError


def _identity_cube() -> Cube:
    """Spine: one row per employee."""
    return Cube(
        name="identity",
        backend=Backend.POSTGRES,
        table="identities",
        alias="id",
        primary_key="id",
        dimensions=[
            Dimension(name="id", sql="{id}.id", type="number"),
            Dimension(name="name", sql="{id}.name", type="string"),
            Dimension(name="region", sql="{id}.region", type="string"),
        ],
    )


def _punch_log_cube() -> Cube:
    """Fact: one row per punch-in event. FK declared on this side."""
    return Cube(
        name="user_punch_log",
        backend=Backend.POSTGRES,
        table="user_punch_log",
        alias="upl",
        primary_key="id",
        measures=[Measure(name="punch_count", sql="*", agg="count")],
        dimensions=[
            Dimension(name="id", sql="{upl}.id", type="number"),
            Dimension(
                name="identity_id",
                sql="{upl}.identity_id",
                type="number",
                foreign_key="identity",
            ),
            Dimension(name="created_at", sql="{upl}.created_at", type="time"),
        ],
    )


def _cat() -> Catalog:
    return Catalog([_identity_cube(), _punch_log_cube()])


# ---------------------------------------------------------------------------
# Happy path: anti-join compiles end-to-end
# ---------------------------------------------------------------------------


def test_anti_join_via_is_null_resolves_with_left_joins() -> None:
    """The user story: who didn't punch in. ``identity`` is the FROM
    root; ``user_punch_log`` named in ``left_joins`` so the BFS walks
    the reverse edge (FK on the fact side); ``is_null`` filter on the
    punch_log column gives the anti-join."""
    q = SemanticQuery(
        dimensions=["identity.name"],
        filters=[
            Filter(
                dimension="user_punch_log.created_at",
                op="is_null",
                values=[],
            )
        ],
        left_joins=["user_punch_log"],
    )
    out = _cat().compile(q)
    # The compile reached user_punch_log via the reverse-walked join.
    assert "LEFT JOIN" in out.sql.upper()
    assert "user_punch_log" in out.sql
    assert "IS NULL" in out.sql.upper()


def test_left_joins_works_alongside_measures() -> None:
    """A spine query with a measure on the fact side: how many punches
    each employee made, including zero for those with none. (Without
    a fill, employees with no punches drop because the COUNT(*) is 0
    not NULL — but the row survives.)"""
    q = SemanticQuery(
        measures=["user_punch_log.punch_count"],
        dimensions=["identity.name"],
        left_joins=["user_punch_log"],
    )
    out = _cat().compile(q)
    assert "LEFT JOIN" in out.sql.upper()


# ---------------------------------------------------------------------------
# Without left_joins, the spine→facts path is unreachable
# ---------------------------------------------------------------------------


def test_spine_to_facts_without_left_joins_fails() -> None:
    """The catalog has a forward edge ``user_punch_log → identity``
    (FK on the fact side). Without ``left_joins``, BFS from
    ``identity`` to ``user_punch_log`` finds no forward path and
    raises."""
    q = SemanticQuery(
        dimensions=["identity.name"],
        filters=[
            Filter(
                dimension="user_punch_log.created_at",
                op="is_null",
                values=[],
            )
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)no join path"):
        _cat().compile(q)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_left_joins_unknown_cube_rejected() -> None:
    q = SemanticQuery(
        dimensions=["identity.name"],
        left_joins=["does_not_exist"],
    )
    with pytest.raises(CompileError, match=r"(?i)not in the catalog"):
        _cat().compile(q)


def test_left_joined_cube_in_dimensions_rejected() -> None:
    """Putting a left-joined cube's dim in ``dimensions`` would make
    GROUP BY a NULL bucket — refuse with a pointer at the anti-join
    pattern."""
    q = SemanticQuery(
        dimensions=["identity.name", "user_punch_log.created_at"],
        left_joins=["user_punch_log"],
    )
    with pytest.raises(CompileError, match=r"(?i)appear in.*dimensions"):
        _cat().compile(q)


def test_left_joins_idempotent_with_existing_forward_edge() -> None:
    """A plain forward-edge query (no ``left_joins``) compiles cleanly —
    bidirectional BFS is a superset of forward BFS, never a refusal. With
    D9 the edge honours ``Join.kind``: an un-listed forward edge is an
    INNER JOIN (the LEFT spine is opt-in via ``left_joins``)."""
    catalog = Catalog([_identity_cube(), _punch_log_cube()])
    # The forward edge is user_punch_log → identity, reached from the
    # FROM root user_punch_log without any left_joins hint.
    q = SemanticQuery(
        measures=["user_punch_log.punch_count"],
        dimensions=["identity.region"],
    )
    out = catalog.compile(q)
    assert "INNER JOIN" in out.sql.upper()
    assert "LEFT JOIN" not in out.sql.upper()
