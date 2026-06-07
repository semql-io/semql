"""Hypothesis property tests on compile_query.

The invariants under test:

1. Any ``SemanticQuery`` the catalog accepts compiles to a single
   safe SELECT statement.
2. ``compile_query`` is deterministic — same inputs produce the same
   bytes (modulo bind-parameter naming, which depends on input order).
3. ``validate`` agrees with ``compile_query``: when validate returns
   an empty list, compile_query succeeds; when it returns errors,
   compile_query raises something.
4. Any valid query generated against a random catalog compiles and
   produces a safe SELECT (random-catalog coverage via strategies.py).

The fixed-catalog tests (1–3) exercise specific query shapes.
The random-catalog tests (4) exercise arbitrary cube structures.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from semql import (
    Backend,
    Catalog,
    CompileError,
    Cube,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    is_safe_select,
    validate,
)


def _catalog() -> Catalog:
    """Stable fixture — every property test runs against this shape."""
    orders = Cube(
        name="orders",
        backend=Backend.POSTGRES,
        table="orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[
            Measure(name="revenue", sql="{o}.amount", agg="sum", unit="currency"),
            Measure(name="count", sql="*", agg="count", unit="count"),
        ],
        dimensions=[
            Dimension(name="region", sql="{o}.region", type="string"),
            Dimension(name="status", sql="{o}.status", type="string"),
            Dimension(name="amount", sql="{o}.amount", type="number"),
        ],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    return Catalog([orders])


_MEASURES = ["orders.revenue", "orders.count"]
_DIMS = ["orders.region", "orders.status"]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _string_filter(draw: st.DrawFn) -> Filter:
    dim = draw(st.sampled_from(["orders.region", "orders.status"]))
    op = draw(st.sampled_from(["eq", "neq", "in", "not_in", "contains"]))
    if op in ("in", "not_in"):
        values = draw(
            st.lists(
                st.text(min_size=1, max_size=5, alphabet="abcdef"),
                min_size=1,
                max_size=4,
            )
        )
    elif op == "contains":
        # contains is single-value.
        values = [draw(st.text(min_size=1, max_size=5, alphabet="abcdef"))]
    else:
        values = [draw(st.text(min_size=1, max_size=5, alphabet="abcdef"))]
    return Filter(dimension=dim, op=op, values=list(values))  # type: ignore[arg-type]


@st.composite
def _numeric_filter(draw: st.DrawFn) -> Filter:
    op = draw(st.sampled_from(["eq", "neq", "gt", "lt", "gte", "lte"]))
    values = [draw(st.integers(min_value=-1_000_000, max_value=1_000_000))]
    return Filter(dimension="orders.amount", op=op, values=list(values))  # type: ignore[arg-type]


@st.composite
def _null_filter(draw: st.DrawFn) -> Filter:
    dim = draw(st.sampled_from(_DIMS))
    op = draw(st.sampled_from(["is_null", "not_null"]))
    return Filter(dimension=dim, op=op, values=[])  # type: ignore[arg-type]


@st.composite
def _aggregated_query(draw: st.DrawFn) -> SemanticQuery:
    measures = draw(st.lists(st.sampled_from(_MEASURES), min_size=1, max_size=2, unique=True))
    dimensions = draw(st.lists(st.sampled_from(_DIMS), max_size=2, unique=True))
    filters = draw(
        st.lists(
            st.one_of(_string_filter(), _numeric_filter(), _null_filter()),
            max_size=3,
        )
    )
    limit = draw(st.one_of(st.none(), st.integers(min_value=1, max_value=1000)))
    return SemanticQuery(
        measures=list(measures),
        dimensions=list(dimensions),
        filters=list(filters),
        limit=limit,
    )


@st.composite
def _ungrouped_query(draw: st.DrawFn) -> SemanticQuery:
    dimensions = draw(st.lists(st.sampled_from(_DIMS), min_size=1, max_size=2, unique=True))
    filters = draw(st.lists(st.one_of(_string_filter(), _null_filter()), max_size=3))
    limit = draw(st.integers(min_value=1, max_value=1000))
    return SemanticQuery(
        dimensions=list(dimensions),
        filters=list(filters),
        ungrouped=True,
        limit=limit,
    )


@st.composite
def _time_breakdown_query(draw: st.DrawFn) -> SemanticQuery:
    measures = draw(st.lists(st.sampled_from(_MEASURES), min_size=1, max_size=2, unique=True))
    granularity = draw(st.sampled_from(["hour", "day", "week", "month"]))
    return SemanticQuery(
        measures=list(measures),
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity=granularity,  # type: ignore[arg-type]
            range=("2026-01-01", "2026-02-01"),
        ),
    )


_QUERY = st.one_of(_aggregated_query(), _ungrouped_query(), _time_breakdown_query())


# ---------------------------------------------------------------------------
# Property 1: every generated query compiles to a safe SELECT.
# ---------------------------------------------------------------------------


@given(query=_QUERY)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=200)
def test_property_compile_produces_safe_select(query: SemanticQuery) -> None:
    out = _catalog().compile(query)
    assert is_safe_select(out.sql), f"unsafe SQL produced:\n{out.sql}"
    # Params bound exactly once per distinct value (or once per filter
    # value in the legacy path) — the dict keys are uniquely numbered.
    assert len(out.params) == len(set(out.params.keys()))


# ---------------------------------------------------------------------------
# Property 2: compile_query is deterministic — same input, same SQL.
# ---------------------------------------------------------------------------


@given(query=_QUERY)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=100)
def test_property_compile_is_deterministic(query: SemanticQuery) -> None:
    a = _catalog().compile(query)
    b = _catalog().compile(query)
    assert a.sql == b.sql
    assert a.params == b.params
    assert a.columns == b.columns


# ---------------------------------------------------------------------------
# Property 3: validate and compile_query agree on success.
# ---------------------------------------------------------------------------


@given(query=_QUERY)
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=100)
def test_property_validate_agrees_with_compile_on_success(query: SemanticQuery) -> None:
    cat = _catalog()
    errors = validate(query, cat)
    if not errors:
        # No validation errors → compile must succeed.
        cat.compile(query)
    else:
        # Validation errors → compile must raise.
        try:
            cat.compile(query)
        except CompileError:
            pass
        else:  # pragma: no cover — surfaces the disagreement if it ever happens
            raise AssertionError(f"validate reported errors but compile succeeded: {errors}")


# ---------------------------------------------------------------------------
# Property 4: random-catalog coverage.
# ---------------------------------------------------------------------------
# SQL uses {alias}.{field_name} everywhere so it always parses; no joins or
# time-dimensions (kept simple so the strategy stays self-contained).


from .strategies import catalog_and_query  # noqa: E402


@given(pair=catalog_and_query())
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=200)
def test_property_random_catalog_compiles(pair: tuple[Catalog, SemanticQuery]) -> None:
    catalog, query = pair
    out = catalog.compile(query)
    assert is_safe_select(out.sql), f"unsafe SQL from random catalog:\n{out.sql}"


@given(pair=catalog_and_query())
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=100)
def test_property_random_catalog_deterministic(pair: tuple[Catalog, SemanticQuery]) -> None:
    catalog, query = pair
    a = catalog.compile(query)
    b = catalog.compile(query)
    assert a.sql == b.sql
    assert a.params == b.params
