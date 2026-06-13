"""Tests for the user-owned ``metadata`` field on catalog types.

The metadata field is a k8s-annotation-style escape hatch: opaque
``dict[str, str]`` that callers can stash any context they want in
(ownership tags, lineage IDs, feature-flag hints, presentation data
for a downstream UI). SemQL guarantees it will *not*:

- Influence compiled SQL.
- Surface in rendered prompt fragments.
- Appear in the reflection META cubes.

It must, however, round-trip cleanly through ``model_copy``,
serialisation, and the eventual catalog documentation generator.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from semql import (
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Join,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
)
from semql.compile import compile_query
from semql.introspect import build_meta_values
from semql_prompt import planner_prompt


def _orders(**meta_overrides: dict[str, str]) -> Cube:
    return Cube(
        name="orders",
        backend=Dialect.POSTGRES,
        table="orders",
        alias="o",
        measures=[
            Measure(
                name="revenue",
                sql="{o}.amount",
                agg="sum",
                unit="currency",
                metadata=meta_overrides.get("measure_meta", {}),
            ),
        ],
        dimensions=[
            Dimension(
                name="region",
                sql="{o}.region",
                type="string",
                metadata=meta_overrides.get("dim_meta", {}),
            ),
        ],
        time_dimensions=[
            TimeDimension(
                name="created_at",
                sql="{o}.created_at",
                metadata=meta_overrides.get("td_meta", {}),
            ),
        ],
        joins=[],
        metadata=meta_overrides.get("cube_meta", {}),
    )


# ---------------------------------------------------------------------------
# Field exists; defaults to {}; accepts arbitrary string maps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Cube, Measure, Dimension, TimeDimension, Join])
def test_metadata_defaults_to_empty_dict(cls: type[BaseModel]) -> None:
    """Every catalog type carries a ``metadata`` field that defaults to {}."""
    assert "metadata" in cls.model_fields, f"{cls.__name__} is missing the metadata field"


def test_cube_metadata_round_trips_through_init() -> None:
    cube = _orders(cube_meta={"owner": "team-data", "tier": "gold"})
    assert cube.metadata == {"owner": "team-data", "tier": "gold"}


def test_measure_metadata_round_trips() -> None:
    m = Measure(
        name="x",
        sql="{o}.x",
        agg="sum",
        metadata={"presentation/format": "currency_usd"},
    )
    assert m.metadata == {"presentation/format": "currency_usd"}


def test_dimension_metadata_round_trips() -> None:
    d = Dimension(
        name="region",
        sql="{o}.region",
        type="string",
        metadata={"lineage": "warehouse.dim_region"},
    )
    assert d.metadata == {"lineage": "warehouse.dim_region"}


def test_time_dimension_metadata_round_trips() -> None:
    td = TimeDimension(name="ts", sql="{o}.ts", metadata={"timezone-hint": "UTC"})
    assert td.metadata == {"timezone-hint": "UTC"}


def test_join_metadata_round_trips() -> None:
    j = Join(
        to="customers",
        relationship="many_to_one",
        on="{o}.cid = {c}.id",
        metadata={"reviewed-by": "carol"},
    )
    assert j.metadata == {"reviewed-by": "carol"}


# ---------------------------------------------------------------------------
# Round-trip through model_copy / model_dump.
# ---------------------------------------------------------------------------


def test_metadata_survives_model_copy() -> None:
    cube = _orders(cube_meta={"k": "v"})
    copy = cube.model_copy(update={"alias": "ord"})
    assert copy.metadata == {"k": "v"}
    assert copy.alias == "ord"


def test_metadata_survives_dump_validate_round_trip() -> None:
    cube = _orders(cube_meta={"k": "v"}, dim_meta={"lineage": "x"})
    dumped = cube.model_dump()
    rebuilt = Cube.model_validate(dumped)
    assert rebuilt.metadata == {"k": "v"}
    assert rebuilt.dimensions[0].metadata == {"lineage": "x"}


# ---------------------------------------------------------------------------
# Opacity contract — metadata cannot leak into compiled SQL / params / columns.
# ---------------------------------------------------------------------------


def _compile_with(cube: Cube) -> tuple[str, dict[str, object], list[str]]:
    cat = Catalog([cube])
    out = cat.compile(
        SemanticQuery(
            measures=["orders.revenue"],
            dimensions=["orders.region"],
            time_dimension=TimeWindow(
                dimension="orders.created_at", range=("2026-01-01", "2026-02-01")
            ),
            filters=[Filter(dimension="orders.region", op="eq", values=["us"])],
        )
    )
    return out.sql, out.params, out.columns


def test_cube_metadata_does_not_affect_compiled_output() -> None:
    plain_sql, plain_params, plain_cols = _compile_with(_orders())
    with_meta_sql, with_meta_params, with_meta_cols = _compile_with(
        _orders(cube_meta={"owner": "team-data", "secret-token": "DO-NOT-LEAK"})
    )
    assert plain_sql == with_meta_sql
    assert plain_params == with_meta_params
    assert plain_cols == with_meta_cols
    assert "DO-NOT-LEAK" not in with_meta_sql


def test_measure_metadata_does_not_affect_compiled_output() -> None:
    plain_sql, _, _ = _compile_with(_orders())
    with_meta_sql, _, _ = _compile_with(
        _orders(measure_meta={"presentation/format": "currency_usd"})
    )
    assert plain_sql == with_meta_sql


def test_dimension_metadata_does_not_affect_compiled_output() -> None:
    plain_sql, _, _ = _compile_with(_orders())
    with_meta_sql, _, _ = _compile_with(_orders(dim_meta={"lineage": "x.y"}))
    assert plain_sql == with_meta_sql


# ---------------------------------------------------------------------------
# Opacity contract — metadata cannot leak into prompt fragments.
# ---------------------------------------------------------------------------


def test_cube_metadata_does_not_appear_in_prompt() -> None:
    cube_plain = _orders()
    cube_meta = _orders(cube_meta={"secret-marker": "META-LEAK-SENTINEL"})
    plain_prompt = planner_prompt(Catalog([cube_plain]))
    meta_prompt = planner_prompt(Catalog([cube_meta]))
    assert plain_prompt == meta_prompt
    assert "META-LEAK-SENTINEL" not in meta_prompt


def test_measure_metadata_does_not_appear_in_prompt() -> None:
    cube_meta = _orders(measure_meta={"presentation/format": "PROMPT-LEAK-SENTINEL"})
    prompt = planner_prompt(Catalog([cube_meta]))
    assert "PROMPT-LEAK-SENTINEL" not in prompt


# ---------------------------------------------------------------------------
# Opacity contract — metadata cannot leak through META reflection cubes.
# ---------------------------------------------------------------------------


def test_metadata_not_surfaced_in_catalog_cubes_meta() -> None:
    cat = Catalog([_orders(cube_meta={"owner": "META-LEAK"})])
    sql = build_meta_values("catalog_cubes", cat.as_dict())
    assert "META-LEAK" not in sql


def test_metadata_not_surfaced_in_catalog_measures_meta() -> None:
    cat = Catalog([_orders(measure_meta={"k": "META-LEAK"})])
    sql = build_meta_values("catalog_measures", cat.as_dict())
    assert "META-LEAK" not in sql


def test_metadata_not_surfaced_in_catalog_dimensions_meta() -> None:
    cat = Catalog([_orders(dim_meta={"k": "META-LEAK"})])
    sql = build_meta_values("catalog_dimensions", cat.as_dict())
    assert "META-LEAK" not in sql


# ---------------------------------------------------------------------------
# Type safety — metadata is dict[str, str]; non-string values are rejected.
# ---------------------------------------------------------------------------


def test_metadata_rejects_non_string_values() -> None:
    with pytest.raises((ValueError, TypeError)):
        Cube(
            name="bad",
            backend=Dialect.POSTGRES,
            table="bad",
            alias="b",
            metadata={"k": 42},  # type: ignore[dict-item]
        )


def test_metadata_separate_instances_dont_share_default() -> None:
    """Default-factory invariant: each instance gets a fresh dict.
    Mutating one cube's metadata mustn't change another's."""
    a = Cube(name="a", backend=Dialect.POSTGRES, table="a", alias="a")
    b = Cube(name="b", backend=Dialect.POSTGRES, table="b", alias="b")
    assert a.metadata == {}
    assert b.metadata == {}
    # If they shared a mutable default, this would smear:
    assert a.metadata is not b.metadata


# Reference compile_query so the unused-import linter doesn't strip it
# when this module is refactored later.
_ = compile_query
