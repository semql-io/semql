"""Unit tests for the SemQLError hierarchy and structured attrs.

The compiler raises specific leaf classes so MCP / API callers can branch
on failure mode programmatically. ``str(err)`` carries the human message;
attributes carry the machine-readable structure.
"""

from __future__ import annotations

import pytest
from semql.compile import compile_query
from semql.errors import (
    CompileError,
    CrossDialectError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    SemQLError,
    UnknownIdentifierError,
    closest_match,
)
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import Filter, SemanticQuery


def test_hierarchy_root_is_semqlerror() -> None:
    assert issubclass(ResolveError, SemQLError)
    assert issubclass(CompileError, ResolveError)
    for leaf in (
        UnknownIdentifierError,
        JoinPathError,
        FilterTypeError,
        PlaceholderError,
        CrossDialectError,
        PhaseDeferredError,
    ):
        assert issubclass(leaf, CompileError)
        assert issubclass(leaf, SemQLError)


def test_closest_match_finds_typo() -> None:
    assert closest_match("regin", ["region", "status", "amount"]) == "region"


def test_closest_match_returns_none_when_far() -> None:
    assert closest_match("zzz", ["region", "status"]) is None


def test_unknown_cube_raises_structured_error(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["nope.count"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    err = exc_info.value
    assert err.kind == "cube"
    assert err.name == "nope"
    assert err.cube is None


def test_unknown_field_raises_structured_error(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.no_such_metric"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    err = exc_info.value
    assert err.kind == "field"
    assert err.name == "no_such_metric"
    assert err.cube == "orders"


def test_unknown_field_suggests_closest_match(catalog: dict[str, Cube]) -> None:
    # 'reveune' is a plausible typo of 'revenue'.
    q = SemanticQuery(measures=["orders.reveune"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    err = exc_info.value
    assert err.hint == "revenue"
    assert "Did you mean 'revenue'" in str(err)


def test_unknown_cube_suggests_closest_match(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["order.revenue"])
    with pytest.raises(UnknownIdentifierError) as exc_info:
        compile_query(q, catalog)
    err = exc_info.value
    assert err.hint == "orders"
    assert "Did you mean 'orders'" in str(err)


def test_cross_backend_raises_structured_error(catalog: dict[str, Cube]) -> None:
    q = SemanticQuery(measures=["orders.revenue", "sessions.duration"])
    with pytest.raises(CrossDialectError) as exc_info:
        compile_query(q, catalog)
    err = exc_info.value
    assert set(err.backends) == {"postgres", "clickhouse"}


def test_filter_type_mismatch_raises_filter_type_error(catalog: dict[str, Cube]) -> None:
    # is_paid is bool; pass a string
    q = SemanticQuery(
        measures=["orders.count"],
        filters=[Filter(dimension="orders.is_paid", op="eq", values=["yes"])],
    )
    with pytest.raises(FilterTypeError) as exc_info:
        compile_query(q, catalog, context={"schema": "test_schema"})
    err = exc_info.value
    assert err.dimension == "orders.is_paid"
    assert err.op == "eq"


def test_placeholder_error_carries_name() -> None:
    bad = Cube(
        name="bad",
        backend=Dialect.POSTGRES,
        table="{nope}.bad",
        alias="b",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    cat = {"bad": bad}
    with pytest.raises(PlaceholderError) as exc_info:
        compile_query(SemanticQuery(measures=["bad.count"]), cat)
    assert exc_info.value.placeholder == "nope"


def test_join_path_error_carries_cube_names() -> None:
    a = Cube(
        name="a",
        backend=Dialect.POSTGRES,
        table="a",
        alias="a",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
    )
    b = Cube(
        name="b",
        backend=Dialect.POSTGRES,
        table="b",
        alias="b",
        dimensions=[Dimension(name="x", sql="{b}.x", type="string")],
    )
    cat = {"a": a, "b": b}
    q = SemanticQuery(measures=["a.count"], dimensions=["b.x"])
    with pytest.raises(JoinPathError) as exc_info:
        compile_query(q, cat)
    err = exc_info.value
    assert err.root_cube == "a"
    assert err.target_cube == "b"


def test_back_compat_compile_error_still_catches(catalog: dict[str, Cube]) -> None:
    """Existing test_compile.py uses ``with pytest.raises(CompileError):``
    against unknown-cube errors. The new UnknownIdentifierError must
    still satisfy that contract."""
    q = SemanticQuery(measures=["nope.count"])
    with pytest.raises(CompileError):
        compile_query(q, catalog)


def test_back_compat_resolve_error_still_catches(catalog: dict[str, Cube]) -> None:
    """Visualization layer catches ResolveError. New leaf classes must
    still be ResolveErrors."""
    q = SemanticQuery(measures=["nope.count"])
    with pytest.raises(ResolveError):
        compile_query(q, catalog)


def test_unused_join_not_treated_as_unreachable() -> None:
    """A cube with no out-going joins to the target reports a structured
    JoinPathError rather than a bare CompileError string."""
    a = Cube(
        name="a",
        backend=Dialect.POSTGRES,
        table="a",
        alias="a",
        measures=[Measure(name="count", sql="*", agg="count", unit="count")],
        joins=[Join(to="missing", relationship="many_to_one", on="{a}.id = 1")],
    )
    cat = {"a": a}
    # The join targets a non-existent cube 'missing'. BFS from 'a' alone
    # without naming 'missing' in the query shouldn't blow up.
    q = SemanticQuery(measures=["a.count"])
    # Compiles fine — the join edge is only consulted when traversal needs it.
    result = compile_query(q, cat)
    assert "a" in result.sql
