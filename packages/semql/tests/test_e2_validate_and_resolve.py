"""E2 — validate_and_resolve convenience function.

Thin wrapper over resolve_query; exposes it under a name that reads as
the "validation + resolution" stage in the three-stage pipeline:
  SemanticQuery → validate_and_resolve → ResolvedQuery → compile_query → Compiled
"""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Measure,
    SemanticQuery,
)
from semql.errors import ResolveError
from semql.introspect import ResolvedQuery


def _catalog() -> Catalog:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
    )
    return Catalog([cube])


def test_validate_and_resolve_returns_resolved_query() -> None:
    from semql import validate_and_resolve

    cat = _catalog()
    q = SemanticQuery(measures=["orders.revenue"], dimensions=["orders.status"])
    result = validate_and_resolve(q, cat)
    assert isinstance(result, ResolvedQuery)
    assert len(result.measures) == 1
    assert result.measures[0][0].name == "orders"
    assert result.measures[0][1].name == "revenue"
    assert len(result.dimensions) == 1


def test_validate_and_resolve_raises_resolve_error_on_unknown_field() -> None:
    from semql import validate_and_resolve

    cat = _catalog()
    q = SemanticQuery(measures=["orders.unicorn"])
    with pytest.raises(ResolveError):
        validate_and_resolve(q, cat)


def test_validate_and_resolve_accepts_catalog_dict() -> None:
    """Works with raw dict[str, Cube] as well as a Catalog instance."""
    from semql import validate_and_resolve

    cat = _catalog()
    q = SemanticQuery(measures=["orders.revenue"])
    result = validate_and_resolve(q, cat.as_dict())
    assert isinstance(result, ResolvedQuery)


def test_validate_and_resolve_exported_from_root() -> None:
    """Must be importable directly from semql."""
    import semql

    assert hasattr(semql, "validate_and_resolve")
    assert callable(semql.validate_and_resolve)
