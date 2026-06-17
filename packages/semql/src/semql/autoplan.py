"""Auto-Planner — choose a cross-source pushdown strategy and rewrite for it.

P1 scope (this module): a flat :class:`~semql.spec.SemanticQuery` whose measures
live on one *primary* backend and which carries a filter on a **filter-only
foreign cube** — a cube referenced *only* by ``q.filters``, on a different
backend, directly bridged to the primary by a simple-equality ``Join`` — is
rewritten so each such foreign cube becomes an injected :class:`~semql.spec.SemiJoin`.
The foreign filters move into the semi-join's inner ``source``; the outer query
keeps its measures + dimensions and drops the foreign filters.

Why: a flat cross-source filter already federates today, but via *bridge-merge*
— ship every fact row into the merge layer and only then join against the one
matching dimension row. A semi-join instead resolves the foreign filter to a
value list and pushes ``key IN (...)`` into the fact fragment, so the fact scan
is selective. Both return identical rows (see
``semql-engine/tests/test_autoplan_equivalence.py``); the rewrite is purely a
selectivity choice.

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
from semql.model import Cube
from semql.spec import Filter, SemanticQuery, SemiJoin

Strategy = Literal["semi_join", "bridge_merge", "attach"]
"""Cross-source pushdown strategy. P1 only emits ``semi_join``; ``bridge_merge``
and ``attach`` are reserved for P2/P3 (see the design spec)."""


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


def autoplan(q: SemanticQuery, catalog: dict[str, Cube]) -> AutoPlan:
    """Plan cross-source filter pushdown for ``q``. See module docstring."""
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

    injected: list[SemiJoin] = []
    decisions: list[CrossSourceDecision] = []
    consumed: set[int] = set()  # id() of filters folded into a semi-join inner

    for fcube in foreign:
        bridge = _bridge_to_kept(bridges, fcube, primary_kept)
        if bridge is None:
            # No direct simple-equality bridge to a primary-backend kept cube.
            # P1 doesn't plan indirect paths — leave it for federation.
            continue
        outer_key, inner_key = _orient(bridge, fcube)
        ffilters = filters_by_cube[fcube]
        injected.append(
            SemiJoin(
                dimension=outer_key,
                op="in",
                select=inner_key,
                source=SemanticQuery(dimensions=[inner_key], filters=list(ffilters)),
            )
        )
        decisions.append(
            CrossSourceDecision(
                foreign_cube=fcube,
                strategy="semi_join",
                reason=(
                    f"{fcube!r} is filter-only and on a different backend than the "
                    f"measures; pushed across as a value-list semi-join "
                    f"({outer_key} IN {inner_key})."
                ),
            )
        )
        consumed.update(id(f) for f in ffilters)

    if not injected:
        return AutoPlan(query=q)

    remaining = [f for f in q.filters if id(f) not in consumed]
    rewritten = q.model_copy(update={"filters": remaining, "semi_joins": injected})
    return AutoPlan(query=rewritten, decisions=tuple(decisions))


__all__ = ["AutoPlan", "CrossSourceDecision", "Strategy", "autoplan"]
