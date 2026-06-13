"""Tests for ``Cube.drill_paths`` — hierarchies for UI consumers.

A drill path is an ordered list of dimensions that form a
hierarchy: ``["country", "state", "city"]``. A frontend that
renders a result grouped by country can offer to drill into the
next level. The compiler doesn't read drill paths; they're
metadata for downstream consumers.

The validator just enforces that every dimension name in every
path resolves to a real dimension on the cube — so a typo in the
catalog fails fast instead of producing a broken UI affordance.
"""

from __future__ import annotations

import pytest
from semql import Cube, Dialect, Dimension


def _geo_cube(drills: list[list[str]] | None = None) -> Cube:
    return Cube(
        name="sales",
        backend=Dialect.POSTGRES,
        table="sales",
        alias="s",
        dimensions=[
            Dimension(name="country", sql="{s}.country", type="string"),
            Dimension(name="state", sql="{s}.state", type="string"),
            Dimension(name="city", sql="{s}.city", type="string"),
        ],
        drill_paths=drills if drills is not None else [],
    )


def test_drill_paths_defaults_to_empty_list() -> None:
    cube = Cube(name="x", backend=Dialect.POSTGRES, table="x", alias="x")
    assert cube.drill_paths == []


def test_cube_accepts_single_hierarchy() -> None:
    cube = _geo_cube([["country", "state", "city"]])
    assert cube.drill_paths == [["country", "state", "city"]]


def test_cube_accepts_multiple_hierarchies() -> None:
    cube = _geo_cube([["country", "state"], ["country", "city"]])
    assert len(cube.drill_paths) == 2


def test_drill_path_with_unknown_dimension_raises() -> None:
    with pytest.raises(ValueError, match=r"(?i)drill_path|unknown|dimension"):
        _geo_cube([["country", "nonexistent"]])


def test_drill_path_must_have_at_least_one_entry() -> None:
    with pytest.raises(ValueError, match=r"(?i)drill_path|empty"):
        _geo_cube([[]])


def test_drill_path_must_be_unique_per_step() -> None:
    """A path with a repeated dim like ``[country, country]`` isn't a
    hierarchy — flag it so a copy-paste typo doesn't ship."""
    with pytest.raises(ValueError, match=r"(?i)drill_path|duplicate|repeat"):
        _geo_cube([["country", "country"]])
