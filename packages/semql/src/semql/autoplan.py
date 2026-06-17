"""Auto-Planner — choose a cross-source pushdown strategy and rewrite for it.

Scope (this module): a flat :class:`~semql.spec.SemanticQuery` whose measures
live on one *primary* backend and which carries a filter on a **filter-only
foreign cube** — a cube referenced *only* by ``q.filters``, on a different
backend, directly bridged to the primary by a simple-equality ``Join``. For each
such cube the planner picks a strategy and, when it chooses ``semi_join``,
rewrites that cube's filter into an injected :class:`~semql.spec.SemiJoin` (the
foreign filter moves into the semi-join's inner ``source``; the outer keeps its
measures + dimensions). A ``bridge_merge`` decision leaves the filter in place
for :func:`~semql.federate.compile_federated_query`.

Why: a flat cross-source filter already federates via *bridge-merge* — ship
every fact row into the merge layer and join against the matching dimension
rows. A semi-join instead resolves the foreign filter to a value list and pushes
``key IN (...)`` into the fact fragment, so the fact scan is selective. Both
return identical rows (see ``semql-engine/tests/test_autoplan_equivalence.py``);
the choice is about cost.

Strategy rule (P2 — operator + ``size_hint`` heuristic):
- non-distributive measure (not sum/count/avg) -> ``semi_join`` (bridge-merge
  refuses it; semi-join keeps the aggregation on the primary backend);
- else all foreign filters selective (eq/in, or a Lookup-backed dimension) ->
  ``semi_join`` (few keys, e.g. ``name = 'Nikhil'``);
- else a broad filter: ``size_hint`` above ``semi_join_max`` -> ``bridge_merge``
  (value list too large to ship); at/below the threshold or unknown ->
  ``semi_join`` (key set bounded; the foreign/metadata cube is usually small).

Invariants:

- **Sans-io.** Operates on already-resolved ``Filter.values`` and never executes
  or resolves anything. Value resolution (human phrase -> canonical key) happens
  at the I/O edge *before* this runs.
- **Non-destructive deferral.** Anything outside P1 scope is returned unchanged
  for :func:`~semql.federate.compile_federated_query` to bridge-merge or refuse:
  foreign cubes contributing output dimensions/measures, foreign refs inside
  ``q.where``, indirect bridges, ``q.compare``, caller-supplied ``semi_joins``
  (Option C override), or measures spanning multiple backends.

The rewritten query carries ``semi_joins`` and must be compiled with
:func:`~semql.semijoin.compile_semi_join_query` (not
:func:`~semql.federate.compile_federated_query`, which refuses ``semi_joins``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from semql.federate import (
    _Bridge,  # pyright: ignore[reportPrivateUsage]
    _find_bridges,  # pyright: ignore[reportPrivateUsage]
)
from semql.model import Cube, Lookup
from semql.spec import Filter, FilterOp, SemanticQuery, SemiJoin

Strategy = Literal["semi_join", "bridge_merge", "attach"]
"""Cross-source pushdown strategy. P1/P2 emit ``semi_join`` / ``bridge_merge``;
``attach`` is reserved for P3 (see the design spec)."""

# Default ceiling on a semi-join's value list. A foreign cube whose row count
# exceeds this AND whose filter isn't provably selective routes to bridge-merge
# instead — the IN list would be too large to ship. Override per call.
SEMI_JOIN_MAX = 10_000

# Aggregations federation can re-aggregate at the merge (distributive across
# sources). Anything else (count_distinct, min, max, median, ratio, …) is
# refused by bridge-merge — see federate.py — so it must use semi-join, which
# keeps the aggregation on the primary backend.
_DISTRIBUTIVE_AGGS = frozenset({"sum", "count", "avg"})

# Operators that bound the resolved key set to "few" — a selective filter the
# planner is happy to push as a value list regardless of foreign cube size.
_SELECTIVE_OPS: frozenset[FilterOp] = frozenset({"eq", "in"})


@dataclass(frozen=True)
class CrossSourceDecision:
    """Why the planner routed one foreign cube the way it did. ``reason`` is a
    human-facing sentence safe to surface to a caller / LLM."""

    foreign_cube: str
    strategy: Strategy
    reason: str


@dataclass(frozen=True)
class AutoPlan:
    """The planner verdict: a (possibly rewritten) query plus the per-foreign-cube
    decisions. ``query is q`` (unchanged) when nothing was planned."""

    query: SemanticQuery
    decisions: tuple[CrossSourceDecision, ...] = field(default=())


def _cube_of(ref: str) -> str:
    """Owning cube name of a qualified ``cube.field`` reference."""
    return ref.split(".", 1)[0]


def _bridge_to_kept(bridges: list[_Bridge], foreign: str, primary_kept: set[str]) -> _Bridge | None:
    """A bridge directly connecting ``foreign`` to a kept cube on the primary
    backend, or ``None``. ``_find_bridges`` only returns cross-dialect bridges,
    so the kept side is guaranteed to differ in backend from ``foreign``."""
    for b in bridges:
        names = {b.left_cube.name, b.right_cube.name}
        if foreign in names and (names - {foreign}) & primary_kept:
            return b
    return None


def _orient(bridge: _Bridge, foreign: str) -> tuple[str, str]:
    """Return ``(outer_key_ref, inner_key_ref)`` for a semi-join, where the
    inner (``select``) key is on the ``foreign`` cube and the outer
    (``dimension``) key is on the primary side."""
    if bridge.left_cube.name == foreign:
        inner = f"{bridge.left_cube.name}.{bridge.left_dim}"
        outer = f"{bridge.right_cube.name}.{bridge.right_dim}"
    else:
        inner = f"{bridge.right_cube.name}.{bridge.right_dim}"
        outer = f"{bridge.left_cube.name}.{bridge.left_dim}"
    return outer, inner


def _all_measures_distributive(q: SemanticQuery, catalog: dict[str, Cube]) -> bool:
    """True iff every measure aggregates distributively (sum/count/avg) — the
    precondition for bridge-merge. An unresolvable measure counts as
    non-distributive (conservative: forces the safe semi-join path)."""
    for ref in q.measures:
        cube = catalog.get(_cube_of(ref))
        if cube is None:
            return False
        name = ref.split(".", 1)[1] if "." in ref else ref
        m = next((x for x in cube.measures if x.name == name), None)
        if m is None or m.agg not in _DISTRIBUTIVE_AGGS:
            return False
    return True


def _filter_is_selective(f: Filter, lookups: dict[str, Lookup]) -> bool:
    """A filter bounds the resolved key set to "few" when it is an eq/in test or
    targets a Lookup-backed dimension (bounded vocabulary)."""
    return f.op in _SELECTIVE_OPS or f.dimension in lookups


def _decide_strategy(
    foreign: str,
    ffilters: list[Filter],
    *,
    distributive: bool,
    size_hint: int | None,
    semi_join_max: int,
    lookups: dict[str, Lookup],
) -> tuple[Strategy, str]:
    """Pick semi_join vs bridge_merge for one filter-only foreign cube, with a
    human-facing reason. See module docstring for the rule."""
    if not distributive:
        return "semi_join", (
            f"{foreign!r} filters a non-distributive measure; bridge-merge would "
            f"refuse it, so it is pushed as a value-list semi-join."
        )
    if all(_filter_is_selective(f, lookups) for f in ffilters):
        return "semi_join", (
            f"{foreign!r} is filtered selectively (eq/in or a lookup); pushed as a "
            f"value-list semi-join."
        )
    if size_hint is not None and size_hint > semi_join_max:
        return "bridge_merge", (
            f"{foreign!r} has a broad filter and size_hint {size_hint} > "
            f"{semi_join_max}; the value list could be large, so it ships through "
            f"the bridge-merge instead."
        )
    hint = "unknown size_hint" if size_hint is None else f"size_hint {size_hint}"
    return "semi_join", (
        f"{foreign!r} has a broad filter but {hint} (<= {semi_join_max}); the key "
        f"set is bounded, so it is pushed as a value-list semi-join."
    )


def autoplan(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    semi_join_max: int = SEMI_JOIN_MAX,
    lookups: dict[str, Lookup] | None = None,
) -> AutoPlan:
    """Plan cross-source filter pushdown for ``q``. See module docstring."""
    lookups = lookups or {}
    # Defer anything beyond a flat measure+filter query to federation untouched.
    if q.semi_joins or q.where is not None or q.compare is not None:
        return AutoPlan(query=q)
    if not q.measures or not q.filters:
        return AutoPlan(query=q)

    # The primary backend is the dialect of the measure-bearing cubes. P1
    # supports a single primary dialect (mirrors federation's "measures on one
    # backend" rule); defer when measures span dialects.
    measure_cubes = {_cube_of(m) for m in q.measures}
    primary_dialects = {catalog[c].dialect for c in measure_cubes if c in catalog}
    if len(primary_dialects) != 1:
        return AutoPlan(query=q)
    (primary_dialect,) = tuple(primary_dialects)

    # Cubes that project output (kept), vs cubes referenced only by filters.
    kept: set[str] = set(measure_cubes)
    kept.update(_cube_of(d) for d in q.dimensions)
    if q.time_dimension is not None:
        kept.add(_cube_of(q.time_dimension.dimension))
    # Kept cubes on the primary backend are valid semi-join "outer" hosts.
    primary_kept = {c for c in kept if c in catalog and catalog[c].dialect == primary_dialect}

    filters_by_cube: dict[str, list[Filter]] = {}
    for f in q.filters:
        filters_by_cube.setdefault(_cube_of(f.dimension), []).append(f)

    # Filter-only foreign cube: referenced only by filters, in the catalog, on a
    # different backend than the measures.
    foreign = [
        name
        for name in filters_by_cube
        if name not in kept and name in catalog and catalog[name].dialect != primary_dialect
    ]
    if not foreign:
        return AutoPlan(query=q)

    touched = [catalog[n] for n in (primary_kept | set(foreign)) if n in catalog]
    bridges = _find_bridges(touched, catalog)
    distributive = _all_measures_distributive(q, catalog)

    injected: list[SemiJoin] = []
    decisions: list[CrossSourceDecision] = []
    consumed: set[int] = set()  # id() of filters folded into a semi-join inner

    for fcube in foreign:
        bridge = _bridge_to_kept(bridges, fcube, primary_kept)
        if bridge is None:
            # No direct simple-equality bridge to a primary-backend kept cube.
            # P1 doesn't plan indirect paths — leave it for federation.
            continue
        ffilters = filters_by_cube[fcube]
        strategy, reason = _decide_strategy(
            fcube,
            ffilters,
            distributive=distributive,
            size_hint=catalog[fcube].size_hint,
            semi_join_max=semi_join_max,
            lookups=lookups,
        )
        decisions.append(CrossSourceDecision(foreign_cube=fcube, strategy=strategy, reason=reason))
        if strategy != "semi_join":
            # bridge_merge: leave the foreign filter in place; federation's
            # bridge-merge (now INNER for filter-only, Gap C) handles it.
            continue
        outer_key, inner_key = _orient(bridge, fcube)
        injected.append(
            SemiJoin(
                dimension=outer_key,
                op="in",
                select=inner_key,
                source=SemanticQuery(dimensions=[inner_key], filters=list(ffilters)),
            )
        )
        consumed.update(id(f) for f in ffilters)

    if not injected:
        # Nothing rewritten — query is unchanged, but record any bridge_merge
        # decisions so the caller can see how each foreign cube was routed.
        return AutoPlan(query=q, decisions=tuple(decisions))

    remaining = [f for f in q.filters if id(f) not in consumed]
    rewritten = q.model_copy(update={"filters": remaining, "semi_joins": injected})
    return AutoPlan(query=rewritten, decisions=tuple(decisions))


__all__ = ["SEMI_JOIN_MAX", "AutoPlan", "CrossSourceDecision", "Strategy", "autoplan"]
