"""LogicalPlan IR — the middle representation between a
``SemanticQuery`` and the sqlglot ``exp.Select`` AST the compiler
emits.

The node vocabulary is small on purpose.  Each node is a frozen
dataclass so plan→plan transforms are pure functions and the plan
is hashable for memoisation in tests.

Two layers of dataclass:

- *Pure data nodes* (``Scan``, ``Join``, ``Filter``, ``TimeBreakdown``,
  ``OrderBy``, ``Limit``, ``ColumnRef``, ``Aggregate``, ``Project``,
  ``CompareSplit``) describe the query shape.
- *Aggregator nodes* (``LogicalPlan``) bundle a query into a unit the
  emitter reads.

``Filter.expr`` carries the spec's ``BoolExpr | Filter`` tree, NOT a
resolved SQL string.  This is the meaningful change from the
intermediate scaffolding: plan-level transforms (CNF normalization,
predicate pushdown) operate on the tree without round-tripping
through sqlglot.  The existing ``_compile_where_tree`` helper in
``compile.py`` keeps its emission logic; the wire-up (Stage 2) just
passes the plan-stored tree to it.

Stage 4 introduces ``apply_rollup_to_plan`` — a plan→plan transform
that replaces one ``Scan`` with a synthetic one pointing at the
rollup's physical table, leaving the original catalog untouched.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from semql.errors import CompileError, JoinPathError
from semql.model import (
    Backend,
    Cube,
    Dimension,
    GranularityLiteral,
    Measure,
    Provenance,
    Rollup,
    TimeDimension,
    View,
)
from semql.model import Join as ModelJoin
from semql.spec import BoolExpr, Filter, SemanticQuery, TimeWindow

if TYPE_CHECKING:
    from semql._resolve import _ResolvedFields


# ---------------------------------------------------------------------------
# Pure data nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnRef:
    """A projected output column — the (cube, field, alias) triple.

    ``kind`` is the role the column plays in the output: ``dimension``,
    ``time``, ``measure``, or ``computed`` (a derived or inline
    measure).  Carrying it on the plan lets the emitter skip the
    re-derivation it otherwise has to do per column.

    ``field`` is the resolved :class:`Measure` / :class:`Dimension` /
    :class:`TimeDimension` from the catalog. Carrying the resolved
    object (not just the field name) means the emitter can render
    the column's SQL straight from ``field.sql`` without a re-lookup.
    ``None`` for the synthetic time-breakdown column whose field
    name is a derived bucket name (``"created_at_month"``) — the
    emitter reads ``time_breakdown.granularity`` from the plan's
    :class:`Aggregate.time` node instead.
    """

    cube: Cube
    field_name: str
    alias: str
    kind: Literal["dimension", "time", "measure", "computed"]
    field: Measure | Dimension | TimeDimension | None = None

    @property
    def provenance(self) -> Provenance:
        """How much this output column can be trusted (C3), derived from
        :attr:`kind`: a catalog measure is ``VERIFIED``, a derived/inline
        measure is ``COMPOSED``, a raw dimension or time bucket is
        ``DIMENSION``."""
        if self.kind == "measure":
            return Provenance.VERIFIED
        if self.kind == "computed":
            return Provenance.COMPOSED
        return Provenance.DIMENSION


@dataclass(frozen=True)
class Scan:
    cube: Cube
    alias: str

    def __repr__(self) -> str:
        return f"Scan({self.cube.name} as {self.alias})"


@dataclass(frozen=True)
class Join:
    left: Cube
    right: Cube
    on: str
    kind: Literal["inner", "left"]
    model: ModelJoin

    def __repr__(self) -> str:
        return f"Join({self.left.name} -> {self.right.name}, kind={self.kind})"


@dataclass(frozen=True)
class Predicate:
    """A predicate that ANDs into the WHERE clause at emission time.

    ``expr`` is the spec tree, not a SQL string.  Holding the tree
    lets plan-level transforms (CNF, pushdown) operate on the
    logical form; the emitter resolves field types and binds values
    when it walks the tree.

    Named ``Predicate`` (not ``Filter``) to avoid colliding with
    :class:`semql.spec.Filter` in the public surface — that name is
    reserved for the spec's predicate value type.
    """

    expr: BoolExpr | Filter

    def __repr__(self) -> str:
        return f"Predicate({self.expr!r})"


@dataclass(frozen=True)
class TimeBreakdown:
    cube: Cube
    field_name: str
    granularity: GranularityLiteral

    def __repr__(self) -> str:
        return f"TimeBreakdown({self.cube.name}.{self.field_name} @ {self.granularity})"


@dataclass(frozen=True)
class Aggregate:
    group_by: list[str]  # cube-qualified refs
    measures: list[str]  # cube-qualified refs
    time: TimeBreakdown | None
    derived: tuple[object, ...]  # InlineDerived list; tuple for hashability

    def __repr__(self) -> str:
        return (
            f"Aggregate(group_by={self.group_by}, measures={self.measures}, "
            f"time={self.time}, derived={list(self.derived)})"
        )


@dataclass(frozen=True)
class Project:
    columns: list[ColumnRef]

    def __repr__(self) -> str:
        cols = ", ".join(f"{c.kind}:{c.alias}" for c in self.columns)
        return f"Project([{cols}])"


@dataclass(frozen=True)
class OrderBy:
    keys: list[tuple[str, Literal["asc", "desc"]]]

    def __repr__(self) -> str:
        return f"OrderBy({self.keys})"


@dataclass(frozen=True)
class Limit:
    limit: int | None
    offset: int | None

    def __repr__(self) -> str:
        return f"Limit(limit={self.limit}, offset={self.offset})"


@dataclass(frozen=True)
class CompareSplit:
    """Compare-mode wrapper.  The plan is a *template* — the emitter
    instantiates it twice with different time-window ranges.

    ``current_range`` is the range from ``plan.time_window``; ``prior_range``
    is either the explicit ``range`` on ``CompareWindow`` (mode='explicit')
    or ``previous_period`` (current_range duration, shifted back).

    The inner ``plan`` is stored once and shared; it's frozen and
    immutable, so a shared reference is safe.
    """

    plan: LogicalPlan
    current_range: tuple[str, str]
    prior_range: tuple[str, str]

    def __repr__(self) -> str:
        return f"CompareSplit(current_range={self.current_range}, prior_range={self.prior_range})"


# ---------------------------------------------------------------------------
# Top-level plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogicalPlan:
    scans: list[Scan]
    joins: list[Join]
    filters: list[Predicate]
    aggregate: Aggregate | None
    project: Project
    order: OrderBy
    limit: Limit
    touched: list[Cube]
    root: Cube
    time_window: TimeWindow | None
    compare: CompareSplit | None = None
    # Carry-through query state that the emitter still reads from the
    # source ``SemanticQuery`` rather than walking the plan: post-agg
    # ``having`` predicates, named ``segments``, and I14 output
    # ``aliases``.  Held here (tuples for immutability) so the plan is
    # a *lossless* representation of the query — ``compile_plan`` can
    # reconstruct the query faithfully instead of silently dropping
    # them (architecture review A1).  Not rendered in ``__repr__`` to
    # keep the plan-snapshot contract stable.
    having: tuple[Filter, ...] = ()
    segments: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()

    def __repr__(self) -> str:
        lines = [
            f"LogicalPlan(root={self.root.name})",
            f"  scans=[{', '.join(repr(s) for s in self.scans)}]",
            f"  joins=[{', '.join(repr(j) for j in self.joins)}]",
            f"  filters=[{', '.join(repr(f) for f in self.filters)}]",
            f"  aggregate={self.aggregate!r}",
            f"  project={self.project!r}",
            f"  order={self.order!r}",
            f"  limit={self.limit!r}",
            f"  time_window={self.time_window!r}",
            f"  compare={self.compare!r}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def to_logical_plan(
    query: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    views: dict[str, View] | None = None,
    resolved: _ResolvedFields | None = None,
) -> LogicalPlan:
    """Lower a ``SemanticQuery`` to a ``LogicalPlan`` IR.

    This stage handles field resolution, join path discovery, and
    logical structure (aggregation vs ungrouped), but does NOT emit
    backend-specific SQL.

    Pass a pre-computed ``resolved`` (``_ResolvedFields``) to skip
    resolution — used by ``_CompileEnv`` so it can run this
    builder *after* its own resolution pass without redoing the
    work.
    """
    if resolved is None:
        from semql._resolve import walk_query_fields

        views_map = views or {}
        resolved, diagnostics = walk_query_fields(query, catalog, views_map=views_map)
        if diagnostics:
            lines = [f"  - {d.message}" for d in diagnostics]
            raise CompileError(
                f"SemanticQuery has {len(diagnostics)} resolution errors:\n" + "\n".join(lines)
            )

    if not resolved.touched:
        raise CompileError("Could not determine any cubes from the query.")

    # 1. Join graph (delegates to the existing BFS helper).
    left_join_cubes = set(query.left_joins)
    cubes_in_from, join_edges = build_join_graph(
        resolved.touched, catalog, left_join_cubes=left_join_cubes
    )

    # 2. Scans & Joins.
    scans = [Scan(cube=c, alias=c.alias) for c in cubes_in_from]
    joins: list[Join] = []
    for left, right, j in join_edges:
        joins.append(
            Join(
                left=left,
                right=right,
                on=j.on,
                kind="left" if right.name in left_join_cubes else "inner",
                model=j,
            )
        )

    # 3. Filters — carry the spec tree, not a SQL string. Apply the
    # CNF pre-pass so every downstream consumer (federation, segment
    # routing, pushdown) sees a top-level AND of OR-clauses. The
    # pass is pure: a tree already in CNF is rebuilt equal to itself.
    from semql.cnf import to_cnf

    filters: list[Predicate] = [Predicate(expr=to_cnf(f)) for f in query.filters]
    if query.where is not None:
        filters.append(Predicate(expr=to_cnf(query.where)))

    # 4. Aggregate. ``ungrouped=True`` is row-listing mode — no
    # GROUP BY, no measure aggregation; the emitter still needs the
    # Project to render correctly.  In that case ``aggregate`` is
    # None and the emitter skips the GROUP BY stage.
    aggregate: Aggregate | None = None
    if not query.ungrouped:
        time_breakdown: TimeBreakdown | None = None
        if (
            query.time_dimension is not None
            and query.time_dimension.granularity is not None
            and resolved.time_cube is not None
            and resolved.time_dim is not None
        ):
            time_breakdown = TimeBreakdown(
                cube=resolved.time_cube,
                field_name=resolved.time_dim.name,
                granularity=query.time_dimension.granularity,
            )
        aggregate = Aggregate(
            group_by=list(query.dimensions),
            measures=list(query.measures),
            time=time_breakdown,
            derived=tuple(query.derived_measures),
        )

    # 5. Project — ColumnRef per output column.  Carrying kind lets
    # the emitter skip per-column re-derivation.
    collisions = output_column_collisions(
        [dim.name for _, dim in resolved.dim_fields],
        [m.name for _, m in resolved.measure_fields],
    )
    project_cols: list[ColumnRef] = []
    for cube, dim in resolved.dim_fields:
        alias = output_alias(cube.name, dim.name, collisions)
        project_cols.append(
            ColumnRef(cube=cube, field_name=dim.name, alias=alias, kind="dimension", field=dim)
        )
    if aggregate is not None and aggregate.time is not None and resolved.time_cube is not None:
        # The time-breakdown column's "field" is the source
        # TimeDimension — the emitter reads the granularity from
        # ``aggregate.time.granularity`` and renders the truncation
        # expression itself.  Carrying the source TimeDimension on
        # the ColumnRef keeps the resolved-field invariant uniform.
        source_td = next(
            (
                td
                for td in resolved.time_cube.time_dimensions
                if td.name == aggregate.time.field_name
            ),
            None,
        )
        project_cols.append(
            ColumnRef(
                cube=resolved.time_cube,
                field_name=aggregate.time.field_name,
                alias=f"{aggregate.time.field_name}_{aggregate.time.granularity}",
                kind="time",
                field=source_td,
            )
        )
    for cube, m in resolved.measure_fields:
        alias = output_alias(cube.name, m.name, collisions)
        project_cols.append(
            ColumnRef(cube=cube, field_name=m.name, alias=alias, kind="measure", field=m)
        )
    project = Project(columns=project_cols)

    # 6. Order / Limit.
    order = OrderBy(keys=list(query.order))
    limit = Limit(limit=query.limit, offset=query.offset)

    # 7. Build the inner LogicalPlan (the template).  When
    # compare-mode wraps it, the outer plan carries the
    # ``CompareSplit`` and a reference to this inner plan.
    inner = LogicalPlan(
        scans=scans,
        joins=joins,
        filters=filters,
        aggregate=aggregate,
        project=project,
        order=order,
        limit=limit,
        touched=resolved.touched,
        root=cubes_in_from[0],
        time_window=query.time_dimension,
        having=tuple(query.having),
        segments=tuple(query.segments),
        aliases=tuple(query.aliases.items()),
    )

    # 8. Compare-mode wrapper.  The emitter reads the inner plan
    # twice, binding each range to the inner select it builds.
    compare: CompareSplit | None = None
    if query.compare is not None and query.time_dimension is not None:
        from datetime import datetime

        current_range = query.time_dimension.range
        if query.compare.mode == "previous_period":
            cs = datetime.fromisoformat(current_range[0])
            ce = datetime.fromisoformat(current_range[1])
            duration = ce - cs
            prior_range: tuple[str, str] = (
                (cs - duration).isoformat(),
                cs.isoformat(),
            )
        else:
            assert query.compare.range is not None
            prior_range = query.compare.range
        compare = CompareSplit(
            plan=inner,
            current_range=current_range,
            prior_range=prior_range,
        )

    return LogicalPlan(
        scans=scans,
        joins=joins,
        filters=filters,
        aggregate=aggregate,
        project=project,
        order=order,
        limit=limit,
        touched=resolved.touched,
        root=cubes_in_from[0],
        time_window=query.time_dimension,
        compare=compare,
        having=tuple(query.having),
        segments=tuple(query.segments),
        aliases=tuple(query.aliases.items()),
    )


def output_column_collisions(
    dim_field_names: Sequence[str],
    measure_field_names: Sequence[str],
) -> set[str]:
    """Resolved field names that appear more than once across the
    projected dimensions + measures.

    The collision set is computed on *resolved field names* — not the
    query's ref locals — so referencing a dimension by an I7 input-alias
    (``orders.territory`` for the ``region`` dimension) can't defeat the
    prefix. This is the single basis shared by the plan's projection and
    the emitter (review B1: the convention used to exist twice)."""
    counts = Counter([*dim_field_names, *measure_field_names])
    return {n for n, c in counts.items() if c > 1}


def output_alias(cube_name: str, field_name: str, collisions: set[str]) -> str:
    """The output-column alias for a projected field: cube-prefixed when
    the field name collides across cubes, else the bare field name."""
    return f"{cube_name}_{field_name}" if field_name in collisions else field_name


# ---------------------------------------------------------------------------
# Graph helpers (unchanged)
# ---------------------------------------------------------------------------


def build_join_graph(
    touched: list[Cube],
    catalog: dict[str, Cube],
    *,
    left_join_cubes: set[str] | None = None,
) -> tuple[list[Cube], list[tuple[Cube, Cube, ModelJoin]]]:
    left_set: set[str] = left_join_cubes or set()
    # Root the FROM clause at a cube that is *not* left-joined, so the
    # left-joined cubes land on the ``right`` side of an edge and the
    # logical plan stamps them ``kind="left"`` (the spine pattern: keep
    # all root rows, optionally match the fact). Falls back to the first
    # touched cube when none qualifies — unchanged for the no-left-join
    # default, where every edge is an inner join.
    root = next((c for c in touched if c.name not in left_set), touched[0])
    join_edges: list[tuple[Cube, Cube, ModelJoin]] = []
    cubes_in_from: list[Cube] = [root]
    for c in touched:
        if c is root:
            continue
        path = find_join_path(
            root.name,
            c.name,
            catalog,
            bidirectional=c.name in left_set,
        )
        cursor = root
        for next_name, j in path:
            tgt = catalog[next_name]
            if tgt not in cubes_in_from:
                join_edges.append((cursor, tgt, j))
                cubes_in_from.append(tgt)
            cursor = tgt
    return cubes_in_from, join_edges


def find_join_path(
    root: str,
    target: str,
    catalog: dict[str, Cube],
    *,
    bidirectional: bool = False,
) -> list[tuple[str, ModelJoin]]:
    if root == target:
        return []
    visited: set[str] = {root}
    queue: list[tuple[str, list[tuple[str, ModelJoin]]]] = [(root, [])]
    while queue:
        current, path = queue.pop(0)
        for j in catalog[current].joins:
            if j.to in visited:
                continue
            new_path = path + [(j.to, j)]
            if j.to == target:
                return new_path
            visited.add(j.to)
            queue.append((j.to, new_path))
        if bidirectional:
            for other_name, other_cube in catalog.items():
                if other_name in visited:
                    continue
                for j in other_cube.joins:
                    if j.to != current:
                        continue
                    new_path = path + [(other_name, j)]
                    if other_name == target:
                        return new_path
                    visited.add(other_name)
                    queue.append((other_name, new_path))
    raise JoinPathError(
        f"No join path from cube {root!r} to {target!r}. "
        "Declare a Join in the catalog or restructure the query.",
        root_cube=root,
        target_cube=target,
    )


# ---------------------------------------------------------------------------
# Plan→plan transforms
# ---------------------------------------------------------------------------


def apply_rollup_to_plan(
    plan: LogicalPlan,
    cube: Cube,
    rollup: Rollup,
) -> LogicalPlan:
    """Return a new ``LogicalPlan`` with the matched ``Scan`` rewritten
    to read the rollup's physical table.

    The original plan and the original catalog are untouched.  The
    transformed plan carries a synthetic Cube — a model_copy of the
    original with ``table`` pointing at the rollup and field SQLs
    rewritten to the rollup's bucketed columns.

    The transform is paired with the existing ``pick_rollup`` selector
    (in ``semql.rollup``) — the caller is expected to have already
    verified the rollup covers the query.  The transform itself does
    not re-verify; it just performs the structural rewrite.

    The synthetic Cube uses the same name and alias as the original
    so downstream emission sees an unchanged view of "this is the
    orders cube" — the only difference is where the FROM points and
    which columns the field SQLs reference.
    """
    # Build the synthetic cube (the rewrite of `cube` reading the
    # rollup table).  Mirrors ``apply_rollup`` in semql.rollup but
    # the result is a *single cube value* — we don't touch a
    # catalog dict.
    alias = cube.alias
    new_measures: list[Measure] = []
    for m in cube.measures:
        if m.name in rollup.measures:
            new_measures.append(m.model_copy(update={"sql": f"{{{alias}}}.{m.name}"}))

    new_dims: list[Dimension] = []
    for d in cube.dimensions:
        if d.name in rollup.dimensions:
            new_dims.append(d.model_copy(update={"sql": f"{{{alias}}}.{d.name}"}))

    new_time_dims: list[TimeDimension] = []
    if rollup.time_dimension is not None and rollup.granularity is not None:
        bucket_col = f"{rollup.time_dimension}_{rollup.granularity}"
        for td in cube.time_dimensions:
            if td.name == rollup.time_dimension:
                new_time_dims.append(
                    td.model_copy(
                        update={
                            "sql": f"{{{alias}}}.{bucket_col}",
                            "granularities": (rollup.granularity,),
                        }
                    )
                )

    new_cube = cube.model_copy(
        update={
            "table": rollup.physical_table,
            "source": None,
            "measures": new_measures,
            "dimensions": new_dims,
            "time_dimensions": new_time_dims,
            "joins": [],
            "segments": [],
            "rollups": [],
        }
    )

    # Replace the matched Scan with a Scan reading the synthetic
    # cube.  Scans are matched by cube name — the caller's
    # pick_rollup already verified the query references this cube.
    new_scans: list[Scan] = []
    for scan in plan.scans:
        if scan.cube.name == cube.name:
            new_scans.append(Scan(cube=new_cube, alias=scan.alias))
        else:
            new_scans.append(scan)

    # Joins in the plan reference the original cube by name in their
    # ``left`` / ``right`` slots.  A rollup routing applies only when
    # the cube is single (Phase 1 — ``pick_rollup`` refuses
    # multi-cube queries), so the joins list is empty in practice.
    # We still rebuild it defensively to keep the dataclass
    # invariants intact.
    new_joins: list[Join] = list(plan.joins)

    return replace(plan, scans=new_scans, joins=new_joins)


def partition_scans(plan: LogicalPlan) -> dict[Backend, LogicalPlan]:
    """Partition a ``LogicalPlan`` by the backend of each ``Scan``'s cube.

    Returns a dict keyed by ``Backend`` whose values are fresh
    ``LogicalPlan`` instances, each containing only the scans /
    joins that touch cubes on that backend.  Cross-backend joins in
    the original plan are dropped from the per-partition output
    (the federation merge step stitches the per-partition fragments
    together via ``MergePlan`` and ``BridgeJoin``).

    A single-backend plan returns a one-entry dict whose value is
    the input plan (no rewrap needed for the common case).

    This is the federation split-point helper.  ``compile_federated_query``
    can use it to derive the per-partition fragments that each
    backend's ``compile_query(sub_query, scoped_catalog)`` compiles
    independently, replacing the ad-hoc ``_build_partition_sub_query``
    path that duplicates join-graph + filter logic today.
    """
    by_backend: dict[Backend, list[Scan]] = {}
    for scan in plan.scans:
        by_backend.setdefault(scan.cube.backend, []).append(scan)

    if len(by_backend) == 1:
        (single,) = by_backend.values()
        # Single-backend: no partitioning needed.
        return {single[0].cube.backend: plan}

    out: dict[Backend, LogicalPlan] = {}
    for backend, scans in by_backend.items():
        scanned_names = {s.cube.name for s in scans}
        # Joins touching this partition: only ones whose BOTH sides
        # are on this backend are kept.  Cross-backend joins are
        # bridges handled by the merge step.
        kept_joins = [
            j for j in plan.joins if j.left.name in scanned_names and j.right.name in scanned_names
        ]
        # Filters stay with the partition whose scans own the
        # touched cube.  A full predicate-router is a future pass;
        # for now the helper's job is to give each backend the
        # minimum surface to compile against.
        out[backend] = replace(plan, scans=scans, joins=kept_joins)
    return out


def apply_partition_to_plan(plan: LogicalPlan, cube: Cube) -> LogicalPlan:
    """Route a time-partitioned cube through the matching physical
    sources. Pure plan→plan transform.

    Mirrors :func:`apply_rollup_to_plan` in shape — the matched
    ``Scan`` is replaced with a ``Scan`` whose cube is a synthetic
    :class:`Cube` (a fresh ``model_copy``) carrying a
    :class:`PartitionedScan` on its ``partitioned_scan`` slot.  The
    emitter reads that slot and emits the unioned subquery in
    place of ``cube.table``.

    The original cube, the original catalog, and the input plan
    are all untouched.  A future plan transform can keep
    inspecting the "un-routed" path without confusing it with the
    routed one.

    Empty-match case: when no source's range intersects the
    query's ``TimeWindow.range``, the synthetic cube carries
    ``PartitionedScan(is_empty=True)`` and the emitter emits a
    zero-row subquery (``SELECT 1 WHERE FALSE``) so the outer
    query predicates still apply and the result is empty by
    construction.
    """
    # Late import — partition module imports model; logical is the
    # other side of the cycle.  The function is invoked after both
    # modules are fully imported (the test harness / catalog
    # construction always happens after the import graph is
    # complete).
    from semql.model import PartitionedScan
    from semql.partition import select_physical_sources

    if not cube.physical_sources:
        raise ValueError(
            f"apply_partition_to_plan: cube {cube.name!r} has no "
            f"physical_sources — nothing to route."
        )
    if cube.time_partition is None:
        raise ValueError(
            f"apply_partition_to_plan: cube {cube.name!r} declares "
            f"physical_sources but no time_partition."
        )

    matched = select_physical_sources(cube, plan.time_window)
    if not matched:
        partitioned = PartitionedScan(sources=(), is_empty=True)
    else:
        # Stable ordering — same source declaration order on every
        # call.  ``select_physical_sources`` already returns
        # declaration order, but enforce it again for the
        # contract.
        ordered = tuple(next(s for s in cube.physical_sources if s.name == m.name) for m in matched)
        partitioned = PartitionedScan(sources=ordered, is_empty=False)

    # Synthetic cube — a model_copy with ``partitioned_scan`` set
    # and the cube's normal table field cleared (so the emitter's
    # table-name fallback doesn't trip on a stale value).
    new_cube = cube.model_copy(
        update={
            "table": "",
            "source": None,
            "physical_sources": [],
            "time_partition": None,
            "partitioned_scan": partitioned,
        }
    )

    new_scans: list[Scan] = []
    for scan in plan.scans:
        if scan.cube.name == cube.name:
            new_scans.append(Scan(cube=new_cube, alias=scan.alias))
        else:
            new_scans.append(scan)

    return replace(plan, scans=new_scans)


__all__ = [
    "Aggregate",
    "ColumnRef",
    "CompareSplit",
    "Join",
    "Limit",
    "LogicalPlan",
    "OrderBy",
    "Predicate",
    "Project",
    "Rollup",  # re-export for callers of apply_rollup_to_plan
    "Scan",
    "TimeBreakdown",
    "apply_partition_to_plan",
    "apply_rollup_to_plan",
    "partition_scans",
    "to_logical_plan",
]
