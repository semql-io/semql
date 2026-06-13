"""Tests for the uuid Dimension type subtype."""

from __future__ import annotations

import pytest
from semql import (
    Catalog,
    Cube,
    Dimension,
    Filter,
    FilterTypeError,
    Measure,
    SemanticQuery,
)
from semql.model import Backend
from semql_prompt import planner_prompt


def _cat() -> Catalog:
    users = Cube(
        name="users",
        backend=Backend.POSTGRES,
        table="users",
        alias="u",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[
            Dimension(name="id", sql="{u}.id", type="uuid"),
            Dimension(name="name", sql="{u}.name", type="string"),
        ],
    )
    return Catalog([users])


def test_dimension_accepts_uuid_type() -> None:
    d = Dimension(name="id", sql="{u}.id", type="uuid")
    assert d.type == "uuid"


def test_filter_uuid_eq_accepts_valid_uuid() -> None:
    cat = _cat()
    q = SemanticQuery(
        measures=["users.count"],
        filters=[
            Filter(
                dimension="users.id",
                op="eq",
                values=["550e8400-e29b-41d4-a716-446655440000"],
            )
        ],
    )
    out = cat.compile(q)
    # UUID is sent as a bound param, not inlined.
    assert "550e8400-e29b-41d4-a716-446655440000" in out.params.values()


def test_filter_uuid_eq_rejects_non_uuid_string() -> None:
    cat = _cat()
    q = SemanticQuery(
        measures=["users.count"],
        filters=[Filter(dimension="users.id", op="eq", values=["not-a-uuid"])],
    )
    with pytest.raises(FilterTypeError, match="UUID"):
        cat.compile(q)


def test_filter_uuid_eq_rejects_non_string() -> None:
    cat = _cat()
    q = SemanticQuery(
        measures=["users.count"],
        filters=[Filter(dimension="users.id", op="eq", values=[42])],
    )
    with pytest.raises(FilterTypeError):
        cat.compile(q)


def test_filter_uuid_in_accepts_list_of_uuids() -> None:
    cat = _cat()
    uuids = [
        "550e8400-e29b-41d4-a716-446655440000",
        "00112233-4455-6677-8899-aabbccddeeff",
    ]
    q = SemanticQuery(
        measures=["users.count"],
        filters=[Filter(dimension="users.id", op="in", values=list(uuids))],
    )
    out = cat.compile(q)
    bound = list(out.params.values())
    for u in uuids:
        assert u in bound


def test_prompt_fragment_surfaces_uuid_type() -> None:
    cat = _cat()
    prompt = planner_prompt(
        cat,
    )
    # The catalog render shows `type=uuid` so the LLM knows to quote
    # and bind literally rather than parse.
    assert "type=uuid" in prompt
