"""Cross-backend semi-join: staged value-list filtering.

A :class:`semql.spec.SemiJoin` restricts an outer dimension to the value set
produced by an inner query. Unlike a federated join, nothing crosses the
backend boundary as a join: the inner ``source`` runs first on its own
backend(s), its ``select`` column is projected to a de-duplicated literal
list, and that list is spliced into the outer query as an ``in`` / ``not_in``
``Filter``. The product is a :class:`SemiJoinPlan` — the compiled inner plans
plus a recipe (``bindings`` + ``bind_outer``) for folding their results into
the outer query.

Execution is inherently sequential (every inner runs before the outer can be
compiled, because the outer's ``IN`` list isn't known until the inner has
run), so this is *not* a single :class:`FederatedPlan`. The executor that
drives the stages lives in ``semql_engine`` (``run_semi_join``); this module
only compiles.

Type safety mirrors a cross-backend bridge join: the outer ``dimension`` and
the inner ``select`` dimension must share an accepted type (see
:func:`semql.federate._accepted_types` / ``Dimension.coerce_to``), or the
compiler refuses with ``cross_cube_type_coercion`` rather than shipping a
value list that the outer backend would silently coerce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from semql.errors import FederationError
from semql.federate import (
    # ``_accepted_types`` is the single source of truth for the cross-backend
    # type-coercion rule, shared with bridge joins; reusing it keeps semi-joins
    # and joins from diverging on what counts as a safe key comparison.
    FederationMode,
    _accepted_types,  # pyright: ignore[reportPrivateUsage]
    compile_federated_query,
)
from semql.refs import field_of, is_qualified, parse_qualified_ref
from semql.spec import Filter, SemanticQuery

if TYPE_CHECKING:
    from collections.abc import Callable

    from semql.backend import DialectStrategy
    from semql.compile import ColumnMeta
    from semql.federate import FederatedPlan
    from semql.introspect import PolicyFn, ScopeFn
    from semql.model import AuthContext, Cube, Dialect, Dimension, View


SEMI_JOIN_PLAN_VERSION = 2


@dataclass(frozen=True)
class SemiJoinBinding:
    """How one resolved semi-join folds its inner result into the outer query.

    ``select_column`` is the output column of the matching inner plan to
    project to a value list; that list becomes a ``Filter(bind_dimension,
    op, values)`` AND-composed onto the outer query."""

    bind_dimension: str
    op: Literal["in", "not_in"]
    select_column: str


@dataclass(frozen=True)
class SemiJoinPlan:
    """A staged plan: run each inner, project its ``select_column``, bind the
    resulting value lists as ``in`` / ``not_in`` filters, then compile + run
    the outer query.

    ``inner_plans[i]`` pairs with ``bindings[i]``. ``bind_outer`` recompiles
    the outer query with the executor-supplied literal filters spliced in and
    returns the final :class:`FederatedPlan` to run. ``columns`` /
    ``column_meta`` describe the final output shape — identical to the outer
    query's, since an ``IN`` filter never changes the projection.

    ``outer_plan`` is a preflight-only shape that retains requirements from
    every binding dimension, including filter-only foreign cubes. Its output
    shape matches the final outer query; the structural predicates are never
    executed.
    Frozen, like every node in the compile IR; ``bind_outer`` is excluded from
    equality/repr (it's an opaque closure over the catalog + compile options)."""

    inner_plans: tuple[FederatedPlan, ...]
    outer_plan: FederatedPlan = field(kw_only=True)
    bindings: tuple[SemiJoinBinding, ...]
    bind_outer: Callable[[list[Filter]], FederatedPlan] = field(compare=False, repr=False)
    columns: list[str]
    column_meta: list[ColumnMeta]
    version: int = SEMI_JOIN_PLAN_VERSION


def _resolve_dimension(catalog: dict[str, Cube], ref: str, role: str) -> tuple[Cube, Dimension]:
    """Resolve a qualified ``cube.field`` ref to its (cube, dimension)."""
    if not is_qualified(ref):
        raise FederationError(
            f"SemiJoin {role} reference {ref!r} must be qualified as ``cube.field``.",
            reason="unqualified_or_unknown_reference",
        )
    parsed = parse_qualified_ref(ref)
    cube_name, fld = parsed.cube, parsed.field
    cube = catalog.get(cube_name)
    if cube is None:
        raise FederationError(
            f"SemiJoin {role} reference {ref!r} names unknown cube {cube_name!r}.",
            reason="unqualified_or_unknown_reference",
        )
    for d in cube.dimensions:
        if d.name == fld:
            return cube, d
    raise FederationError(
        f"SemiJoin {role} reference {ref!r}: cube {cube_name!r} has no dimension {fld!r}.",
        reason="unqualified_or_unknown_reference",
    )


def _check_key_types(catalog: dict[str, Cube], dimension: str, select: str) -> None:
    """Refuse a semi-join whose outer dimension and inner ``select`` key have
    incompatible declared types — the value list would be silently coerced on
    the outer backend (a uuid matched as text, a number read from a string),
    dropping or inventing matches. Same rule as a cross-backend bridge join."""
    _, outer = _resolve_dimension(catalog, dimension, "dimension")
    _, inner = _resolve_dimension(catalog, select, "select")
    if _accepted_types(outer) & _accepted_types(inner):
        return
    raise FederationError(
        f"Semi-join would compare {dimension} (type={outer.type!r}) against the value list "
        f"from {select} (type={inner.type!r}); the list would be silently coerced, which can "
        f"drop or invent matches. Make the types match, or opt in by setting "
        f"coerce_to={inner.type!r} on {dimension} (or coerce_to={outer.type!r} on {select}).",
        reason="cross_cube_type_coercion",
    )


def _resolve_select_column(select: str, inner: FederatedPlan) -> str:
    """The inner plan's output column carrying the ``select`` dimension.

    Dimensions project under their bare field name (``employees.id`` ->
    ``id``). Refuse unless that name appears exactly once in the inner
    plan's columns."""
    col = field_of(select)
    if inner.columns.count(col) != 1:
        raise FederationError(
            f"Semi-join inner query must project {select!r} as exactly one output column "
            f"{col!r}; inner plan columns are {inner.columns!r}.",
            reason="semi_join_select_not_projected",
        )
    return col


def compile_semi_join_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    dialects: dict[Dialect, DialectStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    mode: FederationMode = "distributive",
) -> SemiJoinPlan:
    """Compile a ``SemanticQuery`` carrying ``semi_joins`` into a staged
    :class:`SemiJoinPlan`. Compile options pass through to every stage (inner
    sources and the outer query)."""
    if not q.semi_joins:
        raise FederationError(
            "compile_semi_join_query requires q.semi_joins; compile a plain query with "
            "compile_query / compile_federated_query.",
            reason="semi_join_absent",
        )

    def _compile(query: SemanticQuery) -> FederatedPlan:
        return compile_federated_query(
            query,
            catalog,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            views=views,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
            mode=mode,
        )

    inner_plans: list[FederatedPlan] = []
    bindings: list[SemiJoinBinding] = []
    for sj in q.semi_joins:
        _check_key_types(catalog, sj.dimension, sj.select)
        inner = _compile(sj.source)
        bindings.append(
            SemiJoinBinding(
                bind_dimension=sj.dimension,
                op=sj.op,
                select_column=_resolve_select_column(sj.select, inner),
            )
        )
        inner_plans.append(inner)

    residual = q.model_copy(update={"semi_joins": []})

    def bind_outer(extra_filters: list[Filter]) -> FederatedPlan:
        bound = residual.model_copy(update={"filters": [*residual.filters, *extra_filters]})
        return _compile(bound)

    # Filter-only cubes can change federation strategy and its metadata. Keep
    # the real residual output separately from the dependency-only template.
    shape = _compile(residual)
    preflight_query = residual.model_copy(
        update={
            "filters": [
                *residual.filters,
                *(Filter(dimension=binding.bind_dimension, op="not_null") for binding in bindings),
            ]
        }
    )
    preflight_shape = _compile(preflight_query)
    return SemiJoinPlan(
        inner_plans=tuple(inner_plans),
        outer_plan=preflight_shape,
        bindings=tuple(bindings),
        bind_outer=bind_outer,
        columns=shape.columns,
        column_meta=shape.column_meta,
    )


__all__ = [
    "SEMI_JOIN_PLAN_VERSION",
    "SemiJoinBinding",
    "SemiJoinPlan",
    "compile_semi_join_query",
]
