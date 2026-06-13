"""H5 — SemanticQueryDefaults declarative compile defaults.

Adds SemanticQueryDefaults (frozen Pydantic model) to spec.py.
Wires as query_defaults kwarg on Catalog.compile(). Merge priority:
  query's own value > per-call query_defaults > catalog query_defaults > None.
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
    TimeDimension,
)
from semql.spec import SemanticQueryDefaults, TimeWindow


def _catalog() -> Catalog:
    cube = Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="public.orders",
        alias="o",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="status", sql="{o}.status", type="string")],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                granularities=("day", "week"),
            )
        ],
    )
    return Catalog([cube])


# ---------------------------------------------------------------------------
# SemanticQueryDefaults model
# ---------------------------------------------------------------------------


def test_defaults_model_exists_and_is_frozen() -> None:
    d = SemanticQueryDefaults(limit=100)
    assert d.limit == 100
    with pytest.raises(Exception):  # noqa: B017 — pydantic frozen raises ValidationError
        d.limit = 200


def test_defaults_all_fields_none_by_default() -> None:
    d = SemanticQueryDefaults()
    assert d.limit is None
    assert d.time_window is None
    assert d.granularity is None


# ---------------------------------------------------------------------------
# Catalog-level defaults
# ---------------------------------------------------------------------------


def test_catalog_default_limit_fills_when_query_limit_none() -> None:
    cat = _catalog()
    q = SemanticQuery(measures=["orders.revenue"])
    compiled = cat.compile(
        q,
        query_defaults=SemanticQueryDefaults(limit=500),
    )
    assert "LIMIT 500" in compiled.sql


def test_query_explicit_limit_overrides_catalog_default() -> None:
    cat = _catalog()
    q = SemanticQuery(measures=["orders.revenue"], limit=10)
    compiled = cat.compile(
        q,
        query_defaults=SemanticQueryDefaults(limit=500),
    )
    assert "LIMIT 10" in compiled.sql
    assert "LIMIT 500" not in compiled.sql


def test_no_defaults_compile_unchanged() -> None:
    cat = _catalog()
    q = SemanticQuery(measures=["orders.revenue"])
    without_defaults = cat.compile(q)
    with_empty_defaults = cat.compile(q, query_defaults=SemanticQueryDefaults())
    assert without_defaults.sql == with_empty_defaults.sql


# ---------------------------------------------------------------------------
# Granularity fill
# ---------------------------------------------------------------------------


def test_default_granularity_fills_into_existing_time_window() -> None:
    cat = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            range=("2026-01-01", "2026-02-01"),
            # no granularity set
        ),
    )
    compiled = cat.compile(
        q,
        query_defaults=SemanticQueryDefaults(granularity="week"),
    )
    assert "week" in compiled.sql.lower() or "date_trunc" in compiled.sql.lower()


def test_default_granularity_ignored_when_query_already_has_one() -> None:
    cat = _catalog()
    q = SemanticQuery(
        measures=["orders.revenue"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="day",
            range=("2026-01-01", "2026-02-01"),
        ),
    )
    compiled = cat.compile(
        q,
        query_defaults=SemanticQueryDefaults(granularity="week"),
    )
    assert "day" in compiled.sql.lower() or "date_trunc" in compiled.sql.lower()


# ---------------------------------------------------------------------------
# Exported from semql
# ---------------------------------------------------------------------------


def test_semantic_query_defaults_exported_from_semql() -> None:
    import semql

    assert hasattr(semql, "SemanticQueryDefaults")
