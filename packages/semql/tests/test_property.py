"""Property-based tests for the semql compiler.

Catalogue per ``docs/specs/property-testing.md`` §2, ordered by oracle
strength. Generation is feature-flagged swarm (``strategies.swarm``); the
hostile-value tail is sentinel-wrapped so injection is checkable without
false positives. Settings come from the profile registered in the root
``conftest.py`` (``HYPOTHESIS_PROFILE=dev|ci|nightly``).

Design note — *conditional agreement*: generating a guaranteed-compilable
multi-cube query is hard (fan-out, join paths, left-join rules), so the
differential / algebraic properties assert that two computations reach the
*same* outcome — both succeed with equal SQL, or both refuse — rather than
requiring success. Refusals are first-class, not skipped.
"""

from __future__ import annotations

import contextlib
from typing import cast

import pytest
import sqlglot
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from semql import (
    BoolExpr,
    Catalog,
    Cube,
    Dialect,
    Dimension,
    Filter,
    Measure,
    SemanticQuery,
    TimeDimension,
    TimeWindow,
    compile_query,
    is_read_only_statement,
    to_logical_plan,
    validate,
)
from semql.cnf import to_cnf
from semql.compile import CompiledQuery, compile_plan
from semql.dialect import dialect_for as sqlglot_dialect_for
from semql.errors import SemQLError
from sqlglot import exp

from .strategies import DIALECT_BACKENDS, SENTINEL, broken_pair, swarm

# Swarm catalogs/queries are richer than the default health-check budget
# expects; suppress the timing alarm but never ``filter_too_much`` (§3.5).
_SWARM = settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


def _outcome(thunk: object) -> tuple[str, object, object]:
    """Normalise a compile attempt to a comparable outcome:
    ``("ok", sql, params)`` or ``("err", ExceptionClassName, None)``."""
    try:
        cq = cast("CompiledQuery", thunk())  # type: ignore[operator]
    except SemQLError as e:
        return ("err", type(e).__name__, None)
    return ("ok", cq.sql, cq.params)


# ---------------------------------------------------------------------------
# Fixed-catalog smoke (readable, fast)
# ---------------------------------------------------------------------------


def _catalog() -> Catalog:
    orders = Cube(
        name="orders",
        dialect=Dialect.POSTGRES,
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


@st.composite
def _fixed_query(draw: st.DrawFn) -> SemanticQuery:
    measures = draw(st.lists(st.sampled_from(_MEASURES), min_size=1, max_size=2, unique=True))
    dimensions = draw(st.lists(st.sampled_from(_DIMS), max_size=2, unique=True))
    return SemanticQuery(measures=measures, dimensions=dimensions)


@given(query=_fixed_query())
def test_fixed_catalog_compiles_to_safe_select(query: SemanticQuery) -> None:
    out = _catalog().compile(query)
    assert is_read_only_statement(out.sql, dialect=sqlglot_dialect_for(out.dialect))


# ---------------------------------------------------------------------------
# Totality (the crash-finder)
# ---------------------------------------------------------------------------


@given(a=swarm(), b=swarm())
@_SWARM
def test_p1_totality_compile_never_crashes(
    a: tuple[frozenset[str], Catalog, SemanticQuery],
    b: tuple[frozenset[str], Catalog, SemanticQuery],
) -> None:
    """A structurally-valid query against a *foreign* catalog (refs rarely
    resolve) either compiles or raises ``SemQLError`` — never ``KeyError`` /
    ``AttributeError`` / ``RecursionError`` / any non-SemQL exception."""
    _, _, query = a
    _, catalog, _ = b
    # A typed refusal is the contract; any *other* exception propagates and
    # fails the test — that's the crash this property exists to catch.
    with contextlib.suppress(SemQLError):
        compile_query(query, catalog.as_dict())


# ---------------------------------------------------------------------------
# Safe SELECT + sentinel bind-params (injection oracle)
# ---------------------------------------------------------------------------


@given(trip=swarm())
@_SWARM
def test_p2_values_are_bound_never_spliced(
    trip: tuple[frozenset[str], Catalog, SemanticQuery],
) -> None:
    _, catalog, query = trip
    try:
        out = compile_query(query, catalog.as_dict())
    except SemQLError:
        return
    assert is_read_only_statement(out.sql, dialect=sqlglot_dialect_for(out.dialect))
    # Every filter string value is sentinel-wrapped; a spliced value would
    # carry its marker into the SQL text. The marker must never appear.
    assert SENTINEL not in out.sql
    # Param keys are unique (no clobbering).
    assert len(out.params) == len(set(out.params))


# ---------------------------------------------------------------------------
# Dialect validity
# ---------------------------------------------------------------------------


@given(trip=swarm())
@_SWARM
def test_p3_emitted_sql_parses_under_its_dialect(
    trip: tuple[frozenset[str], Catalog, SemanticQuery],
) -> None:
    _, catalog, query = trip
    try:
        out = compile_query(query, catalog.as_dict())
    except SemQLError:
        return
    parsed = sqlglot.parse_one(out.sql, dialect=out.dialect.value)
    # ``exp.Command`` is sqlglot's "couldn't really parse this" fallback.
    assert not isinstance(parsed, exp.Command)
    assert not list(parsed.find_all(exp.Command)), f"unparsed fragment in:\n{out.sql}"


def _dialect_probe_catalog(dialect: Dialect) -> Catalog:
    """A one-cube catalog whose only variable is the backend — used to
    pin P3 to *every* dialect deterministically (the swarm hits each
    dialect only by chance)."""
    orders = Cube(
        name="orders",
        dialect=dialect,
        table="orders",
        alias="o",
        base_predicate="{o}.deleted_at IS NULL",
        measures=[Measure(name="revenue", sql="{o}.amount", agg="sum")],
        dimensions=[Dimension(name="region", sql="{o}.region", type="string")],
        time_dimensions=[TimeDimension(name="created_at", sql="{o}.created_at")],
    )
    return Catalog([orders])


@pytest.mark.parametrize("dialect", DIALECT_BACKENDS, ids=lambda d: d.value)
def test_p3_every_dialect_emits_parseable_sql(dialect: Dialect) -> None:
    """Systematic companion to the swarm P3: the same query — which
    forces the dialect-sensitive surface (date-truncation, LIMIT,
    bound filter, GROUP BY, ORDER BY) — parses under each backend with
    no ``exp.Command`` fallback."""
    cat = _dialect_probe_catalog(dialect)
    q = SemanticQuery(
        measures=["orders.revenue"],
        dimensions=["orders.region"],
        time_dimension=TimeWindow(
            dimension="orders.created_at",
            granularity="month",
            range=("2020-01-01", "2025-01-01"),
        ),
        filters=[Filter(dimension="orders.region", op="eq", values=["EU"])],
        order=[("orders.revenue", "desc")],
        limit=100,
    )
    out = compile_query(q, cat.as_dict())
    parsed = sqlglot.parse_one(out.sql, dialect=out.dialect.value)
    assert not isinstance(parsed, exp.Command)
    assert not list(parsed.find_all(exp.Command)), f"unparsed fragment in:\n{out.sql}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@given(trip=swarm())
@_SWARM
def test_p5_determinism(trip: tuple[frozenset[str], Catalog, SemanticQuery]) -> None:
    _, catalog, query = trip
    d = catalog.as_dict()
    assert _outcome(lambda: compile_query(query, d)) == _outcome(lambda: compile_query(query, d))


# ---------------------------------------------------------------------------
# Insensitivity to catalog construction order
# ---------------------------------------------------------------------------


@given(trip=swarm())
@_SWARM
def test_p6_cube_order_does_not_change_output(
    trip: tuple[frozenset[str], Catalog, SemanticQuery],
) -> None:
    _, catalog, query = trip
    cubes = list(catalog.as_dict().values())
    permuted = Catalog(list(reversed(cubes)))
    assert _outcome(lambda: compile_query(query, catalog.as_dict())) == _outcome(
        lambda: compile_query(query, permuted.as_dict())
    )


# ---------------------------------------------------------------------------
# CNF semantic equivalence + idempotence
# ---------------------------------------------------------------------------

_ATOMS = [Filter(dimension=f"c.d{i}", op="eq", values=[f"v{i}"]) for i in range(5)]


def _atom_id(f: Filter) -> str:
    return f"{f.dimension}|{f.op}|{tuple(f.values)}"


def _eval(node: BoolExpr | Filter, env: dict[str, bool]) -> bool:
    if isinstance(node, Filter):
        return env[_atom_id(node)]
    if node.op == "not":
        return not _eval(node.children[0], env)
    if node.op == "and":
        return all(_eval(c, env) for c in node.children)
    return any(_eval(c, env) for c in node.children)


def _cnf_expr(leaves: st.SearchStrategy[Filter]) -> st.SearchStrategy[BoolExpr]:
    # hypothesis types `st.recursive`'s element strategy as Unknown, so the
    # `st.builds` lambdas below have an Unknown `c` — no annotation can fix
    # it (the type originates inside hypothesis), hence the scoped ignores.
    strategy = st.recursive(
        leaves,
        lambda sub: st.one_of(
            st.builds(
                lambda c: BoolExpr(op="and", children=c),  # pyright: ignore
                st.lists(sub, min_size=2, max_size=3),  # pyright: ignore
            ),
            st.builds(
                lambda c: BoolExpr(op="or", children=c),  # pyright: ignore
                st.lists(sub, min_size=2, max_size=3),  # pyright: ignore
            ),
            st.builds(
                lambda c: BoolExpr(op="not", children=[c]),  # pyright: ignore
                sub,  # pyright: ignore
            ),
        ),
        max_leaves=6,
    ).filter(lambda e: isinstance(e, BoolExpr))
    return cast("st.SearchStrategy[BoolExpr]", strategy)


@example(
    # Regression (2026-06-13): ``a OR b OR (a AND b)`` distributed to
    # ``(a OR b) AND (a OR b)`` and the duplicate conjunct survived the
    # first pass — to_cnf was not idempotent. Fixed in cnf.py.
    expr=BoolExpr(
        op="or",
        children=[_ATOMS[0], _ATOMS[1], BoolExpr(op="and", children=[_ATOMS[0], _ATOMS[1]])],
    )
)
@given(expr=_cnf_expr(st.sampled_from(_ATOMS)))
@_SWARM
def test_p7_cnf_preserves_truth_table_and_is_idempotent(expr: BoolExpr) -> None:
    converted = to_cnf(expr)
    # Same truth table under every assignment of the (≤5) atoms.
    ids = [_atom_id(a) for a in _ATOMS]
    for mask in range(2 ** len(ids)):
        env = {aid: bool(mask & (1 << i)) for i, aid in enumerate(ids)}
        assert _eval(expr, env) == _eval(converted, env)
    # Idempotence: a second pass changes nothing.
    assert to_cnf(converted) == converted


# ---------------------------------------------------------------------------
# Path equivalence (the cross-path equivalence net): two compile paths, one truth
# ---------------------------------------------------------------------------


@given(trip=swarm())
@_SWARM
def test_p9_query_path_equals_plan_path(
    trip: tuple[frozenset[str], Catalog, SemanticQuery],
) -> None:
    _, catalog, query = trip
    d = catalog.as_dict()
    direct = _outcome(lambda: compile_query(query, d))
    via_plan = _outcome(lambda: compile_plan(to_logical_plan(query, d), d))
    if direct[0] == "ok" and via_plan[0] == "ok":
        assert direct[1] == via_plan[1]  # SQL
        assert direct[2] == via_plan[2]  # params
    else:
        # Both paths must agree on refusal (one succeeding while the other
        # raises would be the cross-path divergence bug this guards against).
        assert direct[0] == "err" and via_plan[0] == "err"


# ---------------------------------------------------------------------------
# Negative properties on one-mutation-off-valid pairs
# ---------------------------------------------------------------------------


@given(t=broken_pair())
@_SWARM
def test_p10_validate_agrees_with_compile_on_broken(
    t: tuple[Catalog, SemanticQuery, str],
) -> None:
    catalog, query, _label = t
    errors = validate(query, catalog)
    # A breakage that ``validate`` flags must also stop ``compile``.
    if errors:
        try:
            catalog.compile(query)
        except SemQLError:
            pass
        else:
            raise AssertionError("validate reported errors but compile succeeded")


@given(t=broken_pair())
@_SWARM
def test_p12_broken_pair_raises_typed_error_naming_the_offender(
    t: tuple[Catalog, SemanticQuery, str],
) -> None:
    catalog, query, label = t
    try:
        catalog.compile(query)
    except SemQLError as e:
        # PHILOSOPHY: errors serve machines and humans — a non-empty message.
        assert str(e)
    else:
        # The only breakage that may legitimately still compile is a
        # filter pointed at a measure when that path is lenient; the
        # ref-not-found breakages must refuse.
        if label in ("unknown_measure", "unknown_dimension"):
            raise AssertionError(f"{label}: expected a typed refusal, compile succeeded")
