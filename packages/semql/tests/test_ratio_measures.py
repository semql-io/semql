"""Tests for ``Measure(agg='ratio', ...)`` — derived measures.

A ratio measure has no column of its own. It names two other
measures on the same cube and emits their aggregates divided:

    SUM(numerator) / NULLIF(SUM(denominator), 0)

The numerator / denominator references are bare names (e.g.
``"completed"``) — a ratio composes measures on the *same* cube,
not across cubes. Filtered measures compose freely as either side
of a ratio.

NULLIF guards divide-by-zero. The result is NULL when the
denominator is zero — same convention as the compare ``pct_change``
expression.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql import Catalog, CompileError, Cube, Dialect, Dimension, Measure, SemanticQuery
from semql_prompt import planner_prompt


def _funnel() -> Cube:
    return Cube(
        name="funnel",
        backend=Dialect.POSTGRES,
        table="funnel",
        alias="f",
        measures=[
            Measure(name="started", sql="*", agg="count"),
            Measure(
                name="completed",
                sql="*",
                agg="count",
                filter="{f}.status = 'complete'",
            ),
            Measure(
                name="conversion_rate",
                sql="",
                agg="ratio",
                numerator="completed",
                denominator="started",
                format="percent",
            ),
        ],
        dimensions=[Dimension(name="cohort", sql="{f}.cohort", type="string")],
    )


# ---------------------------------------------------------------------------
# Model — Measure(agg='ratio') field shape + validation.
# ---------------------------------------------------------------------------


def test_ratio_measure_constructs_with_numerator_and_denominator() -> None:
    m = Measure(
        name="rate",
        sql="",
        agg="ratio",
        numerator="x",
        denominator="y",
    )
    assert m.agg == "ratio"
    assert m.numerator == "x"
    assert m.denominator == "y"


def test_ratio_measure_requires_numerator_and_denominator() -> None:
    with pytest.raises(ValidationError, match=r"(?i)numerator|denominator"):
        Measure(name="rate", sql="", agg="ratio")
    with pytest.raises(ValidationError, match=r"(?i)denominator"):
        Measure(name="rate", sql="", agg="ratio", numerator="x")


def test_non_ratio_measure_rejects_numerator_and_denominator() -> None:
    """``numerator``/``denominator`` only make sense for ``agg='ratio'``."""
    with pytest.raises(ValidationError, match=r"(?i)ratio"):
        Measure(name="bad", sql="*", agg="sum", numerator="x", denominator="y")


def test_non_ratio_measure_defaults_numerator_denominator_to_none() -> None:
    m = Measure(name="count", sql="*", agg="count")
    assert m.numerator is None
    assert m.denominator is None


# ---------------------------------------------------------------------------
# Compiler — emit SUM(num) / NULLIF(SUM(den), 0).
# ---------------------------------------------------------------------------


def test_ratio_compiles_to_division_with_nullif_guard() -> None:
    out = Catalog([_funnel()]).compile(SemanticQuery(measures=["funnel.conversion_rate"]))
    # Both components rendered as their own aggregates.
    assert "COUNT(*)" in out.sql
    # NULLIF guard around the denominator.
    assert "NULLIF" in out.sql.upper()
    # Divided.
    assert "/" in out.sql
    # Aliased to the ratio measure name.
    assert "AS conversion_rate" in out.sql or "conversion_rate" in out.sql


def test_ratio_with_filtered_numerator_emits_filter_inside_division() -> None:
    """The numerator references ``completed``, which is itself a
    filtered measure. The FILTER clause must end up *inside* the
    numerator of the ratio, not at the WHERE level."""
    out = Catalog([_funnel()]).compile(SemanticQuery(measures=["funnel.conversion_rate"]))
    assert "FILTER" in out.sql.upper()
    assert "f.status" in out.sql


def test_ratio_with_groupby_dimension() -> None:
    out = Catalog([_funnel()]).compile(
        SemanticQuery(
            measures=["funnel.conversion_rate"],
            dimensions=["funnel.cohort"],
        )
    )
    assert "GROUP BY" in out.sql.upper()
    assert "f.cohort" in out.sql
    assert "NULLIF" in out.sql.upper()


def test_ratio_alongside_component_measures() -> None:
    """A query can ask for the ratio and its components in one
    pass — common shape for funnel dashboards."""
    out = Catalog([_funnel()]).compile(
        SemanticQuery(measures=["funnel.started", "funnel.completed", "funnel.conversion_rate"])
    )
    assert "started" in out.sql
    assert "completed" in out.sql
    assert "conversion_rate" in out.sql or "NULLIF" in out.sql.upper()


# ---------------------------------------------------------------------------
# Compile-time validation — references must resolve on the same cube.
# ---------------------------------------------------------------------------


def test_ratio_with_unknown_numerator_raises_compile_error() -> None:
    cube = Cube(
        name="funnel",
        backend=Dialect.POSTGRES,
        table="funnel",
        alias="f",
        measures=[
            Measure(name="started", sql="*", agg="count"),
            Measure(
                name="rate",
                sql="",
                agg="ratio",
                numerator="nonexistent",
                denominator="started",
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)numerator|nonexistent"):
        Catalog([cube]).compile(SemanticQuery(measures=["funnel.rate"]))


def test_ratio_with_unknown_denominator_raises_compile_error() -> None:
    cube = Cube(
        name="funnel",
        backend=Dialect.POSTGRES,
        table="funnel",
        alias="f",
        measures=[
            Measure(name="completed", sql="*", agg="count"),
            Measure(
                name="rate",
                sql="",
                agg="ratio",
                numerator="completed",
                denominator="nonexistent",
            ),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)denominator|nonexistent"):
        Catalog([cube]).compile(SemanticQuery(measures=["funnel.rate"]))


def test_ratio_cannot_reference_another_ratio() -> None:
    """Ratios reference leaf measures only — chaining ratios is a
    different feature (deferred) and would need cycle detection."""
    cube = Cube(
        name="funnel",
        backend=Dialect.POSTGRES,
        table="funnel",
        alias="f",
        measures=[
            Measure(name="a", sql="*", agg="count"),
            Measure(name="b", sql="*", agg="count"),
            Measure(name="r1", sql="", agg="ratio", numerator="a", denominator="b"),
            Measure(name="r2", sql="", agg="ratio", numerator="r1", denominator="b"),
        ],
    )
    with pytest.raises(CompileError, match=r"(?i)ratio"):
        Catalog([cube]).compile(SemanticQuery(measures=["funnel.r2"]))


# ---------------------------------------------------------------------------
# Prompt fragment surfaces ratio measures.
# ---------------------------------------------------------------------------


def test_prompt_marks_ratio_measures() -> None:
    rendered = planner_prompt(Catalog([_funnel()]))
    assert "conversion_rate" in rendered
    assert "agg=ratio" in rendered
