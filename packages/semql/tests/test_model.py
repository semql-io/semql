"""Unit tests for ``semql.model``.

The model defines the catalog's value-object types; their invariants
(frozen, type literals, defaults) are part of the public surface and
gate every downstream layer. Catching a literal-renaming or
frozen-flag-flip here is cheaper than catching it via compile errors
in user catalogs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from semql.model import (
    Cube,
    Dialect,
    Dimension,
    Join,
    Measure,
    TimeDimension,
)

# ---------------------------------------------------------------------------
# Cube.field_names()
# ---------------------------------------------------------------------------


def test_field_names_collects_all_kinds() -> None:
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="t",
        alias="a",
        measures=[Measure(name="m1", sql="x", agg="sum")],
        dimensions=[Dimension(name="d1", sql="x", type="string")],
        time_dimensions=[TimeDimension(name="t1", sql="x")],
    )
    assert cube.field_names() == {"m1", "d1", "t1"}


def test_field_names_empty_cube_returns_empty_set() -> None:
    cube = Cube(name="c", dialect=Dialect.POSTGRES, table="t", alias="a")
    assert cube.field_names() == set()


def test_field_names_dedupes_implicitly() -> None:
    """Same identifier across kinds collapses into the set — surfaces
    name collisions the compiler later prefixes with the cube name."""
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="t",
        alias="a",
        measures=[Measure(name="count", sql="*", agg="count")],
        dimensions=[Dimension(name="count", sql="x", type="number")],
    )
    assert cube.field_names() == {"count"}


# ---------------------------------------------------------------------------
# Pydantic validation — invalid literals raise ValidationError
# ---------------------------------------------------------------------------


def test_measure_rejects_unknown_agg() -> None:
    with pytest.raises(ValidationError):
        Measure(name="m", sql="x", agg="quantile_99.999")  # type: ignore[arg-type]


def test_dimension_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Dimension(name="d", sql="x", type="datetime")  # type: ignore[arg-type]


def test_cube_rejects_unknown_chart_type() -> None:
    with pytest.raises(ValidationError):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="t",
            alias="a",
            default_chart_type="treemap",  # type: ignore[arg-type]
        )


def test_cube_rejects_unknown_backend() -> None:
    with pytest.raises(ValidationError):
        Cube(name="c", dialect="elasticsearch", table="t", alias="a")  # type: ignore[arg-type]


def test_join_rejects_unknown_relationship() -> None:
    with pytest.raises(ValidationError):
        Join(to="x", relationship="many_to_many", on="...")  # type: ignore[arg-type]


def test_time_dimension_rejects_unknown_granularity() -> None:
    with pytest.raises(ValidationError):
        TimeDimension(
            name="td",
            sql="x",
            granularities=("yearly",),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# frozen=True invariants — Measure / Dimension / TimeDimension / Join
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instance",
    [
        Measure(name="m", sql="x", agg="sum"),
        Dimension(name="d", sql="x", type="string"),
        TimeDimension(name="td", sql="x"),
        Join(to="o", relationship="many_to_one", on="..."),
    ],
)
def test_frozen_models_reject_field_mutation(instance: object) -> None:
    """Pydantic ``frozen=True`` raises ValidationError on attribute set."""
    with pytest.raises(ValidationError):
        instance.name = "renamed"  # type: ignore[attr-defined]


def test_cube_is_frozen() -> None:
    """Cube is a frozen catalog value type like every other (AGENTS.md):
    it must not be able to drift out of sync with the catalog that
    validated it. Mutation raises; build a fresh one (or model_copy)."""
    cube = Cube(name="c", dialect=Dialect.POSTGRES, table="t", alias="a")
    with pytest.raises(ValidationError):
        cube.description = "set later"


def test_required_filters_must_name_real_field() -> None:
    """A required_filters typo is caught at construction, not as a
    misleading 'requires a filter on regn' at every query."""
    with pytest.raises(ValidationError, match="required_filters"):
        Cube(
            name="c",
            dialect=Dialect.POSTGRES,
            table="t",
            alias="a",
            dimensions=[Dimension(name="region", sql="{a}.region", type="string")],
            required_filters=["regn"],
        )


def test_required_filters_accepts_dimension_and_time_dimension() -> None:
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="t",
        alias="a",
        dimensions=[Dimension(name="region", sql="{a}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{a}.created_at")],
        required_filters=["region", "created_at"],
    )
    assert cube.required_filters == ["region", "created_at"]


def test_cube_keywords_deduped_and_frozen() -> None:
    """Keyword normalisation/dedupe still happens — now in a
    mode='before' validator since the instance is frozen."""
    cube = Cube(
        name="c",
        dialect=Dialect.POSTGRES,
        table="t",
        alias="a",
        keywords=["Sales", "sales", "revenue"],
    )
    assert cube.keywords == ["sales", "revenue"]


# ---------------------------------------------------------------------------
# Default-value invariants — empty lists, None, default granularities
# ---------------------------------------------------------------------------


def test_cube_defaults() -> None:
    cube = Cube(name="c", dialect=Dialect.POSTGRES, table="t", alias="a")
    assert cube.measures == []
    assert cube.dimensions == []
    assert cube.time_dimensions == []
    assert cube.joins == []
    assert cube.required_filters == []
    assert cube.base_predicate is None
    assert cube.description == ""
    assert cube.display_name is None
    assert cube.default_chart_type is None
    assert cube.expose_in_prompt is True
    assert cube.metadata == {}


def test_measure_defaults() -> None:
    m = Measure(name="m", sql="x", agg="count")
    assert m.unit is None
    assert m.description == ""
    assert m.display_name is None
    assert m.format is None
    assert m.metadata == {}


def test_dimension_defaults() -> None:
    d = Dimension(name="d", sql="x", type="string")
    assert d.description == ""
    assert d.display_name is None
    assert d.metadata == {}


def test_time_dimension_defaults() -> None:
    td = TimeDimension(name="td", sql="x")
    assert td.type == "time"
    assert td.granularities == ("hour", "day", "week", "month", "quarter", "year")
    assert td.description == ""
    assert td.display_name is None
    assert td.metadata == {}


def test_separate_cubes_dont_share_default_lists() -> None:
    """Mutable defaults must be per-instance, not shared. Pydantic v2
    handles this correctly by default; pin the invariant so a refactor
    doesn't smear state across catalog cubes."""
    a = Cube(name="a", dialect=Dialect.POSTGRES, table="t", alias="a")
    b = Cube(name="b", dialect=Dialect.POSTGRES, table="t", alias="b")
    assert a.measures is not b.measures
    assert a.dimensions is not b.dimensions
    assert a.required_filters is not b.required_filters


# ---------------------------------------------------------------------------
# Dialect enum
# ---------------------------------------------------------------------------


def test_backend_has_expected_members() -> None:
    expected = {
        "postgres",
        "clickhouse",
        "duckdb",
        "bigquery",
        "snowflake",
        # R1 first-class analytics engines.
        "redshift",
        "trino",
        "databricks",
        # R1 experimental OLTP engines (opt-in via experimental_dialects()).
        "sqlserver",
        "mysql",
        "oracle",
        "meta",
    }
    assert {b.value for b in Dialect} == expected


def test_backend_is_strenum() -> None:
    """Dialect.POSTGRES compares equal to "postgres" — useful for
    serialisation and JSON round-trips."""
    # mypy's narrowing sees Dialect.POSTGRES as Literal[Dialect.POSTGRES]
    # and Literal["postgres"] as non-overlapping; the runtime StrEnum
    # behaviour is exactly what's being asserted.
    assert str(Dialect.POSTGRES) == "postgres"
    assert isinstance(Dialect.POSTGRES, str)
