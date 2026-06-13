"""Tests for offset / pagination on SemanticQuery."""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    CompileError,
    Cube,
    Dimension,
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


def test_query_accepts_offset_field() -> None:
    q = SemanticQuery(measures=["orders.revenue"], limit=10, offset=20)
    assert q.offset == 20


def test_query_offset_defaults_to_none() -> None:
    q = SemanticQuery(measures=["orders.revenue"])
    assert q.offset is None


def test_compile_emits_limit_offset_when_both_set() -> None:
    cat = _cat()
    q = SemanticQuery(
        measures=["orders.revenue"], dimensions=["orders.region"], limit=10, offset=20
    )
    out = cat.compile(q)
    assert "LIMIT 10" in out.sql
    assert "OFFSET 20" in out.sql
    # OFFSET must follow LIMIT.
    assert out.sql.index("LIMIT 10") < out.sql.index("OFFSET 20")


def test_compile_omits_offset_when_offset_is_none() -> None:
    cat = _cat()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"], limit=10)
    out = cat.compile(q)
    assert "LIMIT 10" in out.sql
    assert "OFFSET" not in out.sql


def test_compile_omits_offset_when_offset_is_zero() -> None:
    """offset=0 means 'no skip' — don't emit OFFSET 0 noise."""
    cat = _cat()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"], limit=10, offset=0)
    out = cat.compile(q)
    assert "OFFSET" not in out.sql


def test_offset_without_limit_rejected() -> None:
    cat = _cat()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.region"], offset=20)
    with pytest.raises(CompileError, match="offset"):
        cat.compile(q)


def test_negative_offset_rejected() -> None:
    with pytest.raises(ValueError):
        SemanticQuery(measures=["orders.revenue"], limit=10, offset=-1)
