"""Cross-source compilation: emit per-backend fragments + a structured
merge plan when a query's touched cubes span backends.

The sans-io counterpart of :func:`semql.compile.compile_query` for
federated queries. Each fragment is a normal :class:`CompiledQuery` for
its backend; the cross-fragment merge is described structurally by
:class:`MergeSpec` (dimensions, bridge joins, per-measure re-aggregation
recipes, residual predicates). This module is dialect-agnostic — it
emits no merge SQL. The executor's package (``semql_engine.merge``)
renders the spec to DuckDB SQL, the federation lingua franca; a custom
executor may consume the spec however it likes.

v1 restrictions, refused with :class:`FederationError`:

- Measures must live on a single "primary" backend partition. Dim
  cubes on other backends contribute lookup attributes only. The one
  exception is cross-backend *symmetric aggregation*: two or more
  additive (``sum`` / ``count``) measures on different backends that
  all conform to one shared bridge cube and group by a bridge
  dimension. Those pre-aggregate per fact on their own backend and
  LEFT-join onto the bridge at the merge — fan-safe because each fact
  reaches the grouping grain before the cross-fragment join (see
  :func:`_detect_cross_backend_symmetric`).
- Bridge joins between partitions must be equality on a single column
  pair, with both sides declared as ``Dimension``\\s on their cubes
  (the federation layer cannot project columns the catalog hasn't
  named).
- ``q.where`` (the boolean predicate tree) and ``q.compare`` are not
  supported in cross-source v1.
- Aggregations: ``sum`` / ``count`` distribute (sum-of-sums at merge);
  ``avg`` is decomposed into ``(sum, count)`` in the primary fragment
  and recomposed at the merge step. ``count_distinct`` / ``min`` /
  ``max`` / ``ratio`` need raw rows; refused in sans-io. The in-process
  executor (``semql_engine``) can handle these via raw-row streaming —
  call it via its own helper instead.

Single-backend queries should call :func:`compile_query` directly.
:func:`compile_federated_query` will also succeed on single-backend
queries (returning a degenerate one-fragment plan), but the simpler
type is preferred whenever federation isn't needed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Literal, Protocol, cast

from semql.cnf import to_cnf as core_to_cnf
from semql.compile import ColumnMeta, CompiledQuery, compile_query
from semql.errors import FederationError
from semql.introspect import resolve_query
from semql.model import Cube, Dialect, Dimension, Join, Measure
from semql.spec import BoolExpr, Filter, SemanticQuery, TimeWindow

FederationMode = Literal["distributive", "raw_rows"]
"""Federation compile-path selector.

- ``"distributive"`` (default) — fragments aggregate locally (SUM /
  COUNT / decomposed AVG); merge re-aggregates with SUM. Refuses
  ``count_distinct`` / ``min`` / ``max`` / ``ratio`` because they
  don't distribute under SUM.
- ``"raw_rows"`` — fragments emit ungrouped rows; merge applies the
  full aggregation. Lifts the non-distributive-agg restriction and
  the ``having`` restriction. Costs more bytes (full join cardinality
  on the wire) so callers should opt in deliberately.
"""

if TYPE_CHECKING:
    from semql.backend import DialectStrategy
    from semql.introspect import PolicyFn, ScopeFn
    from semql.model import AuthContext, View


# ---------------------------------------------------------------------------
# Public IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FragmentColumn:
    fragment_index: int  # 0-based into FederatedPlan.fragments
    column_name: str  # name in that fragment's output


@dataclass(frozen=True)
class BridgeJoin:
    left: FragmentColumn
    right: FragmentColumn
    join_kind: Literal["left", "full_outer"] = "left"


@dataclass(frozen=True)
class DimensionOutput:
    output_name: str
    sources: list[FragmentColumn]  # COALESCE order for full-outer
    column_meta: ColumnMeta
    # Time bucketing grain, set only for a time dimension the *merge*
    # buckets (raw_rows mode renders ``date_trunc(grain, source)``).
    # ``None`` for ordinary dimensions and for distributive-mode time
    # columns, which the fragment already bucketed — those pass through
    # under their own name. Carried on the spec so a renderer can
    # reconstruct the bucket without the original SemanticQuery.
    time_grain: str | None = None


# How the merge re-aggregates a measure column. Spans both federation
# modes: distributive folds count into ``sum`` (fragment pre-counted),
# while raw_rows streams the un-aggregated source and applies the real
# aggregate at the merge — so the raw aggregates (``avg``, the
# percentiles) appear here too. Every value is a recipe a renderer can
# turn into a merge-side expression from the ``*_source`` columns alone.
MergeAgg = Literal[
    "sum",  # SUM(source)
    "count",  # COUNT(source) -- raw_rows; distributive folds count into sum
    "min",  # MIN(source)
    "max",  # MAX(source)
    "avg",  # AVG(source) -- raw_rows
    "count_distinct",  # COUNT(DISTINCT source)
    "median",  # PERCENTILE_CONT(0.5)  -- raw_rows
    "p75",  # PERCENTILE_CONT(0.75) -- raw_rows
    "p90",  # PERCENTILE_CONT(0.90) -- raw_rows
    "p95",  # PERCENTILE_CONT(0.95) -- raw_rows
    "avg_recomposed",  # SUM(sum_src) / NULLIF(SUM(cnt_src), 0)
    "ratio",  # agg(num) / NULLIF(agg(den), 0)
    "passthrough",  # single-fragment; no re-aggregation
]


@dataclass(frozen=True)
class MeasureOutput:
    output_name: str
    merge_agg: MergeAgg
    column_meta: ColumnMeta
    source: FragmentColumn | None = None
    sum_source: FragmentColumn | None = None  # avg_recomposed
    count_source: FragmentColumn | None = None  # avg_recomposed
    numerator: FragmentColumn | None = None  # ratio
    denominator: FragmentColumn | None = None  # ratio
    # Per-side aggregates for ``ratio`` — the merge applies these to the
    # raw numerator / denominator columns before dividing. ``None`` for
    # every other ``merge_agg``.
    numerator_agg: MergeAgg | None = None
    denominator_agg: MergeAgg | None = None


# The aggregates a raw-rows fragment streams un-applied for the merge to
# apply. Every member is a ``MergeAgg`` literal; the set is the merge's
# supported raw aggregates (``ratio`` recomposes from these, never
# itself). Used to narrow a cube measure's ``agg`` string into the typed
# ``MergeAgg`` recorded on the spec — replaces the old ``type: ignore``.
_RAW_MERGE_AGGS: frozenset[str] = frozenset(
    {"sum", "count", "count_distinct", "avg", "min", "max", "median", "p75", "p90", "p95"}
)


def _as_merge_agg(agg: str) -> MergeAgg:
    if agg not in _RAW_MERGE_AGGS:
        raise FederationError(
            f"agg={agg!r} has no raw-rows merge recipe.", reason="unsupported_agg"
        )
    return cast("MergeAgg", agg)


# A cross-partition residual clause resolved to fragment coordinates.
# Each literal is ``(negated, fragment_index, column_name, op, values)``;
# a clause is an OR of literals; the spec holds an AND of clauses. Plain
# hashable tuples so :class:`MergeSpec` stays frozen, and self-contained
# so a renderer emits the post-join WHERE with no catalog lookup.
_ResolvedCrossLiteral = tuple[bool, int, str, str, tuple[object, ...]]
_ResolvedCrossClause = tuple[_ResolvedCrossLiteral, ...]


@dataclass(frozen=True)
class MergeSpec:
    primary_index: int
    bridges: list[BridgeJoin]
    dimensions: list[DimensionOutput]
    measures: list[MeasureOutput]
    having: list[Filter]
    order_by: list[tuple[str, Literal["asc", "desc"]]]
    limit: int | None
    offset: int | None
    mode: Literal["distributive", "raw_rows"]
    # CNF clauses that touch more than one partition, resolved to
    # fragment coordinates: each literal is
    # ``(negated, fragment_index, column_name, op, values)``. The merge
    # applies them as a post-join WHERE — AND across clauses, OR within a
    # clause. Self-contained: a renderer needs no catalog or cube→index
    # map. ``op`` / ``values`` mirror :class:`semql.spec.Filter`.
    cross_partition_clauses: tuple[_ResolvedCrossClause, ...] = ()


# Format version of the federation plan IR.  Bumped when the
# fragment / merge_spec shape changes in a way an out-of-tree executor
# would need to know about (``MergeSpec`` travels across package
# versions — see the executor's ``ANN401`` note).  Stamped on every
# :class:`FederatedPlan` so a consumer can detect a compiler / executor
# skew instead of mis-reading a changed shape.
#
# v2: ``FederatedPlan.merge`` (the rendered DuckDB ``MergePlan``) was
# removed — the plan now carries only the structured ``merge_spec``, and
# the spec gained ``DimensionOutput.time_grain``, the widened
# ``MeasureOutput.merge_agg`` (+ ``numerator_agg`` / ``denominator_agg``
# for ratios), and fragment-resolved ``cross_partition_clauses``.
FEDERATED_PLAN_VERSION = 2


@dataclass(frozen=True)
class FederatedPlan:
    """A query that touches cubes on multiple backends.

    ``fragments[i]`` is a :class:`CompiledQuery` to run against its
    respective backend. An executor loads each result into a DuckDB
    table named ``frag_i`` and applies ``merge_spec`` to produce the
    final shape. The plan is dialect-agnostic: it carries the structured
    :class:`MergeSpec`, not merge SQL — rendering the DuckDB merge SQL
    lives in ``semql_engine.merge`` (a custom executor may consume the
    spec however it likes).

    ``columns`` and ``column_meta`` describe the final output shape
    after the merge — same role they play on :class:`CompiledQuery`.

    Frozen — like every other node in the federation IR.  Derive a
    tweaked plan with :func:`dataclasses.replace`, never by attribute
    assignment, so the fragments can't silently desync from the spec.
    ``version`` carries :data:`FEDERATED_PLAN_VERSION`.
    """

    fragments: list[CompiledQuery]
    merge_spec: MergeSpec
    columns: list[str]
    column_meta: list[ColumnMeta]
    version: int = FEDERATED_PLAN_VERSION


# ---------------------------------------------------------------------------
# Bridge join parsing — v1 supports only ``{a}.col = {b}.col`` shape.
# ---------------------------------------------------------------------------


_BRIDGE_RE = re.compile(
    r"^\s*\{(?P<la>[a-z_][a-z0-9_]*)\}\.(?P<lc>[a-z_][a-z0-9_]*)\s*"
    r"=\s*"
    r"\{(?P<ra>[a-z_][a-z0-9_]*)\}\.(?P<rc>[a-z_][a-z0-9_]*)\s*$"
)


@dataclass(frozen=True)
class _Bridge:
    """A cross-backend join edge with its keys extracted into structured form.

    ``left_cube`` is the cube the ``Join`` is declared on; ``right_cube``
    is its target. ``left_col`` / ``right_col`` are the column names on
    each side. ``left_dim`` / ``right_dim`` are the dimension names that
    expose those columns (the catalog must declare them — see module
    docstring)."""

    left_cube: Cube
    right_cube: Cube
    left_col: str
    right_col: str
    left_dim: str
    right_dim: str


def _find_dim_for_column(cube: Cube, alias: str, column: str) -> str:
    """Find the dimension on ``cube`` whose ``sql`` exposes ``{alias}.column``.

    Used to translate a bridge join's column reference into a
    Dimension name the per-fragment sub-query can project. We compare
    the dimension's ``sql`` after substituting the alias — exact text
    match. Returns the dimension name. Raises :class:`FederationError`
    if no dimension exposes that column."""
    target = f"{alias}.{column}"
    for d in cube.dimensions:
        # Dimension SQL uses {alias} placeholders; substitute and trim
        # whitespace for the comparison. A user can write ``{o}.id`` or
        # ``{o}. id`` — normalise.
        rendered = d.sql.replace("{" + alias + "}", alias).strip()
        if rendered == target:
            return d.name
    raise FederationError(
        f"Cube {cube.name!r}: bridge join references column {target!r} "
        f"but no dimension on this cube exposes that column. Declare a "
        f"Dimension with sql={target!r} so the federation layer can "
        f"project the join key.",
        reason="join_key_not_a_dimension",
    )


def _parse_bridge(left_cube: Cube, right_cube: Cube, join: Join) -> _Bridge:
    """Parse a cross-backend ``Join`` into structured bridge keys.

    Refuses anything more complex than ``{a}.col = {b}.col``."""
    m = _BRIDGE_RE.match(join.on)
    if m is None:
        raise FederationError(
            f"Cross-backend join from {left_cube.name!r} to "
            f"{right_cube.name!r}: ``on`` must be a single column "
            f"equality of the form ``{{a}}.col = {{b}}.col``. Got: "
            f"{join.on!r}.",
            reason="bridge_join_not_simple_equality",
        )
    la, lc, ra, rc = m.group("la", "lc", "ra", "rc")
    if {la, ra} != {left_cube.alias, right_cube.alias}:
        raise FederationError(
            f"Cross-backend join from {left_cube.name!r} to "
            f"{right_cube.name!r}: aliases in ``on`` ({la!r}, {ra!r}) "
            f"don't match the cubes' aliases "
            f"({left_cube.alias!r}, {right_cube.alias!r}).",
            reason="bridge_alias_mismatch",
        )
    # Normalise so left/right match left_cube/right_cube.
    if la == right_cube.alias:
        la, lc, ra, rc = ra, rc, la, lc

    left_dim = _find_dim_for_column(left_cube, left_cube.alias, lc)
    right_dim = _find_dim_for_column(right_cube, right_cube.alias, rc)
    _check_bridge_key_types(left_cube, left_dim, right_cube, right_dim)
    return _Bridge(
        left_cube=left_cube,
        right_cube=right_cube,
        left_col=lc,
        right_col=rc,
        left_dim=left_dim,
        right_dim=right_dim,
    )


def _accepted_types(dim: Dimension) -> set[str]:
    """The set of types a dimension is willing to be compared as: its own
    declared ``type`` plus any ``coerce_to`` opt-in."""
    types: set[str] = {dim.type}
    if dim.coerce_to is not None:
        types.add(dim.coerce_to)
    return types


def _check_bridge_key_types(
    left_cube: Cube,
    left_dim: str,
    right_cube: Cube,
    right_dim: str,
) -> None:
    """Refuse a cross-backend bridge join whose two keys have
    incompatible declared types.

    The merge equates the keys with a bare ``a.k = b.k``; if the types
    differ the underlying engine coerces one side silently, which can
    drop or invent matches (a ``uuid`` compared as text, a number read
    from a string). We refuse unless the catalog author opts in via
    ``Dimension.coerce_to`` on one side — see :func:`_accepted_types`."""
    left = _dim_by_name(left_cube, left_dim)
    right = _dim_by_name(right_cube, right_dim)
    if _accepted_types(left) & _accepted_types(right):
        return
    raise FederationError(
        f"Cross-backend join from {left_cube.name!r} to {right_cube.name!r}: join key "
        f"{left_cube.name}.{left_dim} (type={left.type!r}) would be silently coerced to "
        f"compare with {right_cube.name}.{right_dim} (type={right.type!r}). Make the types "
        f"match, or opt in by setting coerce_to={right.type!r} on {left_cube.name}.{left_dim} "
        f"(or coerce_to={left.type!r} on {right_cube.name}.{right_dim}).",
        reason="cross_cube_type_coercion",
    )


def _dim_by_name(cube: Cube, name: str) -> Dimension:
    """Look up a declared dimension by name. The caller has already
    resolved ``name`` via :func:`_find_dim_for_column`, so it always
    exists; the guard is defensive."""
    for d in cube.dimensions:
        if d.name == name:
            return d
    raise FederationError(  # pragma: no cover — name came from this cube's dimensions
        f"Cube {cube.name!r}: bridge key dimension {name!r} not found.",
        reason="join_key_not_a_dimension",
    )


# ---------------------------------------------------------------------------
# Partitioning + sub-query synthesis
# ---------------------------------------------------------------------------


def _touched(q: SemanticQuery, catalog: dict[str, Cube]) -> list[Cube]:
    resolved = resolve_query(q, catalog)
    touched = list(resolved.touched_cubes)

    # ``resolve_query`` reports the cubes named by measures / dimensions,
    # but a cube can also enter a query through a *filter* alone — e.g.
    # "count orders, filtered to a customer tier" where the tier lives in
    # another backend and there is no grouping dimension on it. Such a
    # cube must count toward the touched set, or the federation entry gate
    # sees a single backend, declines to federate, and the single-backend
    # compiler then pulls the foreign cube onto the join path as a bridge
    # (invalid cross-dialect SQL). Add any filter- / where-referenced cube
    # that resolve_query didn't already surface, preserving order.
    seen = {c.name for c in touched}
    for cube_name in _filter_where_cube_names(q):
        if cube_name not in seen and cube_name in catalog:
            touched.append(catalog[cube_name])
            seen.add(cube_name)
    return touched


def _filter_where_cube_names(q: SemanticQuery) -> list[str]:
    """Cube names referenced by ``q.filters`` and the ``q.where`` tree.

    Filter dimensions are qualified ``cube.field`` refs; the where-tree is
    flattened to its ``Filter`` leaves. Order-preserving, may contain
    duplicates (the caller de-dups)."""
    names: list[str] = [f.dimension.split(".", 1)[0] for f in q.filters]
    if q.where is not None:
        for clause in _to_cnf(q.where):
            for _negated, lit in clause:
                names.append(lit.dimension.split(".", 1)[0])
    return names


def _find_bridges(
    touched: list[Cube],
    catalog: dict[str, Cube],
) -> list[_Bridge]:
    """Walk every join declared on a touched cube; keep the ones that
    cross a backend boundary into another touched cube. A join's
    reverse direction is treated as the same bridge — we de-duplicate
    by ``(min, max)`` of the cube names."""
    in_scope = {c.name for c in touched}
    by_name = {c.name: c for c in touched}
    seen: set[tuple[str, str]] = set()
    bridges: list[_Bridge] = []
    for cube in touched:
        for j in cube.joins:
            if j.to not in in_scope:
                continue
            target = by_name[j.to]
            if target.dialect == cube.dialect:
                continue
            key = tuple(sorted((cube.name, target.name)))
            if key in seen:
                continue
            seen.add((key[0], key[1]))
            bridges.append(_parse_bridge(cube, target, j))
    return bridges


def _resolve_field_to_cube(ref: str, catalog: dict[str, Cube]) -> Cube:
    """Identify the cube that owns the field referenced by ``ref``.

    ``ref`` is the qualified ``cube.field`` form used in
    ``SemanticQuery.measures`` / ``dimensions`` / ``filters``."""
    if "." in ref:
        cube_name = ref.split(".", 1)[0]
        if cube_name in catalog:
            return catalog[cube_name]
    raise FederationError(
        f"Reference {ref!r} must be qualified as ``cube.field`` and resolve to a known cube.",
        reason="unqualified_or_unknown_reference",
    )


_AVG_DECOMP_SUM_SUFFIX = "__avg_sum"
_AVG_DECOMP_COUNT_SUFFIX = "__avg_count"


def _decompose_avg_measure(m: Measure) -> tuple[Measure, Measure]:
    """For ``agg='avg'`` measures, emit a ``sum`` and a ``count`` pair
    the fragment can compute. The merge step recomposes the average as
    ``SUM(sum) / NULLIF(SUM(count), 0)``."""
    sum_m = m.model_copy(
        update={
            "name": m.name + _AVG_DECOMP_SUM_SUFFIX,
            "agg": "sum",
            "display_name": None,
            "format": None,
            "unit": m.unit,
            "display_unit": None,
        }
    )
    count_m = m.model_copy(
        update={
            "name": m.name + _AVG_DECOMP_COUNT_SUFFIX,
            "agg": "count",
            "display_name": None,
            "format": None,
            "unit": "count",
            "display_unit": None,
        }
    )
    return sum_m, count_m


@dataclass
class _PartitionPlan:
    """The per-backend sub-query we'll compile into a fragment.

    Carries enough metadata to drive the merge step: which dims are
    "projected" (visible in the final output), which are bridge keys
    (consumed by joins, not projected), which measure names came from
    avg-decomposition (need recomposing), and which measure names are
    pass-through.
    """

    dialect: Dialect
    cubes: list[Cube]
    sub_query: SemanticQuery
    # Map from the original measure ref ("orders.revenue") to the
    # column name(s) in the fragment's output.
    measure_columns: dict[str, str]
    # Original avg measures decomposed: ref → (sum_col, count_col).
    avg_columns: dict[str, tuple[str, str]]
    # Map from the original dim ref ("customers.region") to the column
    # name in the fragment's output. Only dims this partition owns.
    dim_columns: dict[str, str]
    # Bridge dim columns this fragment exposes (column names in the
    # fragment output). Keyed by the dim ref ("orders.customer_id")
    # so the merge step can wire them up.
    bridge_columns: dict[str, str]


def _build_partition_sub_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    partition_cubes: list[Cube],
    primary_partition: Dialect,
    bridges: list[_Bridge],
) -> _PartitionPlan:
    """Construct the ``SemanticQuery`` that will compile into this
    partition's fragment.

    Includes:
    - Dimensions this partition owns (from ``q.dimensions``).
    - Time dimension if it resolves to a cube in this partition.
    - Measures (only on the primary partition).
    - Filters whose dimension belongs to a cube in this partition.
    - Extra "bridge" dimensions for every join key on a bridge that
      touches a cube in this partition — so the merge step can join
      fragments on those columns.
    - Segments whose cube lives in this partition.

    The where-tree is routed separately, after all partitions are
    built (``_route_where_distributive``), because a cross-partition
    clause has to force-project columns across several fragments at
    once.  Refuses queries with measures on non-primary partitions.
    """
    partition_names = {c.name for c in partition_cubes}
    dialect = partition_cubes[0].dialect
    is_primary = dialect is primary_partition

    sub_measures: list[str] = []
    measure_columns: dict[str, str] = {}
    avg_columns: dict[str, tuple[str, str]] = {}
    # Synthetic Measures we add to the cube for avg decomposition.
    synthetic_measures: dict[str, list[Measure]] = {}

    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        if owner.name not in partition_names:
            if is_primary:
                # measure resolves to a non-primary partition — refused
                raise FederationError(
                    f"Measure {ref!r} resolves to cube {owner.name!r} on "
                    f"backend {owner.dialect.value!r}, which is not the "
                    f"primary measure partition {primary_partition.value!r}. "
                    f"All measures must live on a single backend in v1; the "
                    f"in-process executor can lift this restriction.",
                    reason="measure_on_non_primary_partition",
                )
            continue
        # ref belongs to this partition.
        m_name = ref.rsplit(".", 1)[1]
        m = next(x for x in owner.measures if x.name == m_name)
        if m.agg in ("sum", "count"):
            sub_measures.append(ref)
            measure_columns[ref] = m_name
        elif m.agg == "avg":
            sum_m, count_m = _decompose_avg_measure(m)
            synthetic_measures.setdefault(owner.name, []).extend([sum_m, count_m])
            sub_measures.append(f"{owner.name}.{sum_m.name}")
            sub_measures.append(f"{owner.name}.{count_m.name}")
            avg_columns[ref] = (sum_m.name, count_m.name)
        else:
            raise FederationError(
                f"Measure {ref!r} uses agg={m.agg!r}, which is not "
                f"distributive across sources. v1 supports sum, count, "
                f"and avg; the in-process executor handles the rest via "
                f"raw-row streaming.",
                reason=f"non_distributive_aggregation:{m.agg}",
            )

    sub_dimensions: list[str] = []
    dim_columns: dict[str, str] = {}
    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        if owner.name in partition_names:
            sub_dimensions.append(ref)
            dim_columns[ref] = ref.rsplit(".", 1)[1]

    # Time dimension: include if it resolves to this partition. The
    # time window itself stays on the fragment that owns the time dim
    # (it doesn't need to be re-applied at merge).
    sub_time_dim: TimeWindow | None = None
    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        if td_cube.name in partition_names:
            sub_time_dim = q.time_dimension
            # Add the bucketed time column to dim_columns so the merge
            # step can group by it. The compiler aliases it
            # ``<td_name>_<granularity>`` when granularity is set, else
            # ``<td_name>``.
            td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
            if q.time_dimension.granularity is not None:
                td_col = f"{td_name}_{q.time_dimension.granularity}"
            else:
                td_col = td_name
            dim_columns[q.time_dimension.dimension] = td_col

    # Filters whose dimension lives on this partition.
    sub_filters: list[Filter] = []
    for f in q.filters:
        owner = _resolve_field_to_cube(f.dimension, catalog)
        if owner.name in partition_names:
            sub_filters.append(f)

    # Segments route to the partition owning the segment's cube. A
    # segment's SQL references a single cube's alias, so it always
    # belongs to exactly one partition — the compiler applies it
    # inside that fragment; the merge step never re-applies it.
    sub_segments = [s for s in q.segments if s.split(".", 1)[0] in partition_names]

    # Bridge keys. For every bridge that touches a cube in this
    # partition, expose the appropriate side as a dimension.
    bridge_columns: dict[str, str] = {}
    for b in bridges:
        if b.left_cube.name in partition_names:
            ref = f"{b.left_cube.name}.{b.left_dim}"
            if ref not in sub_dimensions:
                sub_dimensions.append(ref)
            bridge_columns[ref] = b.left_dim
        if b.right_cube.name in partition_names:
            ref = f"{b.right_cube.name}.{b.right_dim}"
            if ref not in sub_dimensions:
                sub_dimensions.append(ref)
            bridge_columns[ref] = b.right_dim

    # If we synthesised measures (avg decomposition), splice them into
    # a model_copy of the owning cube so the compiler can resolve them.
    if synthetic_measures:
        for cube in list(partition_cubes):
            extra = synthetic_measures.get(cube.name)
            if not extra:
                continue
            patched = cube.model_copy(update={"measures": [*cube.measures, *extra]})
            # Replace in the partition list and let _scoped_catalog
            # pick up the patched version.
            partition_cubes[partition_cubes.index(cube)] = patched

    sub_query = SemanticQuery(
        measures=sub_measures,
        dimensions=sub_dimensions,
        time_dimension=sub_time_dim,
        filters=sub_filters,
        segments=sub_segments,
        # ``where`` is routed after partition construction
        # (``_route_where_distributive``); HAVING stays a merge-only
        # concern and is refused in distributive mode upstream.
        order=[],
        # Ordering & limits happen at merge.
    )
    return _PartitionPlan(
        dialect=dialect,
        cubes=partition_cubes,
        sub_query=sub_query,
        measure_columns=measure_columns,
        avg_columns=avg_columns,
        dim_columns=dim_columns,
        bridge_columns=bridge_columns,
    )


def _scoped_catalog(partition_cubes: list[Cube]) -> dict[str, Cube]:
    """Return a catalog restricted to one partition's cubes, with
    intra-partition joins preserved and cross-partition joins stripped.

    The compiler will BFS the join graph during sub-query compilation
    and would otherwise stumble on joins pointing at cubes that aren't
    in this fragment's scope."""
    in_scope = {c.name for c in partition_cubes}
    scoped: dict[str, Cube] = {}
    for c in partition_cubes:
        local_joins = [j for j in c.joins if j.to in in_scope]
        if local_joins != list(c.joins):
            scoped[c.name] = c.model_copy(update={"joins": local_joins})
        else:
            scoped[c.name] = c
    return scoped


# ---------------------------------------------------------------------------
# Merge SQL emission
# ---------------------------------------------------------------------------


class _MergePartition(Protocol):
    """The slice of a partition plan the merge assembler reads — shared by
    the distributive and raw-rows plans so one assembler serves both."""

    dialect: Dialect
    cubes: list[Cube]
    dim_columns: dict[str, str]
    bridge_columns: dict[str, str]


def _build_merge_joins(
    partitions: Sequence[_MergePartition],
    bridges: list[_Bridge],
    primary_idx: int,
) -> list[BridgeJoin]:
    """Resolve the bridge graph into an ordered list of LEFT JOINs.

    BFS out from the primary partition so each :class:`BridgeJoin` has
    its ``left`` (host) side already joined when the ``right`` (new) side
    is added — the renderer joins them in this order. Raises if the
    partitions don't form a connected graph.
    """
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    bridge_joins: list[BridgeJoin] = []
    joined: set[int] = {primary_idx}
    pending = list(bridges)
    while pending:
        progress = False
        remaining: list[_Bridge] = []
        for b in pending:
            left_idx, right_idx = cube_to_idx[b.left_cube.name], cube_to_idx[b.right_cube.name]
            if left_idx in joined and right_idx not in joined:
                new_idx, host_idx = right_idx, left_idx
            elif right_idx in joined and left_idx not in joined:
                new_idx, host_idx = left_idx, right_idx
            else:
                remaining.append(b)
                continue
            if host_idx == cube_to_idx[b.left_cube.name]:
                h_cube, h_dim = b.left_cube.name, b.left_dim
                n_cube, n_dim = b.right_cube.name, b.right_dim
            else:
                h_cube, h_dim = b.right_cube.name, b.right_dim
                n_cube, n_dim = b.left_cube.name, b.left_dim
            host_col = partitions[host_idx].bridge_columns[f"{h_cube}.{h_dim}"]
            new_col = partitions[new_idx].bridge_columns[f"{n_cube}.{n_dim}"]
            bridge_joins.append(
                BridgeJoin(
                    left=FragmentColumn(host_idx, host_col),
                    right=FragmentColumn(new_idx, new_col),
                )
            )
            joined.add(new_idx)
            progress = True
        pending = remaining
        if not progress and pending:
            raise FederationError(
                "Federated plan has disconnected backend partitions.",
                reason="disconnected_partitions",
            )
    return bridge_joins


# (output alias, fragment source column, bucket grain or None) for the
# query's time dimension at the merge. Grain None = the fragment already
# bucketed (distributive passthrough); set = the merge buckets (raw_rows).
_TimeOutput = tuple[str, FragmentColumn, str | None]


def _build_merge_spec(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    primary_partition: Dialect,
    partitions: Sequence[_MergePartition],
    bridges: list[_Bridge],
    output_column_meta: list[ColumnMeta],
    *,
    mode: Literal["distributive", "raw_rows"],
    measure_outputs: list[MeasureOutput],
    time_output: _TimeOutput | None,
    cross_partition_clauses: _Cnf | None = None,
) -> MergeSpec:
    """Assemble the structured :class:`MergeSpec` — the authoritative,
    dialect-agnostic federation plan. No SQL: rendering DuckDB merge SQL
    from this spec lives in ``semql_engine.merge`` (see PHILOSOPHY — the
    core compiler plans, the executor's package renders the dialect).

    Universal across both federation modes; the mode-specific pieces
    (measure re-aggregation, time bucketing) arrive pre-built as
    ``measure_outputs`` / ``time_output``. Output column order is
    dimensions, then the time column, then measures — the contract the
    renderer's positional GROUP BY relies on.
    """
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    primary_idx = next(i for i, p in enumerate(partitions) if p.dialect == primary_partition)

    dimension_outputs: list[DimensionOutput] = []
    for ref in q.dimensions:
        idx = cube_to_idx[_resolve_field_to_cube(ref, catalog).name]
        col = partitions[idx].dim_columns[ref]
        alias = ref.rsplit(".", 1)[1]
        meta = next(m for m in output_column_meta if m.name == alias)
        dimension_outputs.append(
            DimensionOutput(output_name=alias, sources=[FragmentColumn(idx, col)], column_meta=meta)
        )

    if time_output is not None:
        time_alias, time_source, time_grain = time_output
        meta = next(m for m in output_column_meta if m.name == time_alias)
        dimension_outputs.append(
            DimensionOutput(
                output_name=time_alias,
                sources=[time_source],
                column_meta=meta,
                time_grain=time_grain,
            )
        )

    bridge_joins = _build_merge_joins(partitions, bridges, primary_idx)
    resolved_cross = _resolve_cross_clauses(cross_partition_clauses or [], cube_to_idx)

    # HAVING refusal (compile-time): every HAVING target must be one of
    # the selected measures — the renderer references it by output alias.
    if q.having:
        selected = {m.output_name for m in measure_outputs}
        for hf in q.having:
            alias = hf.dimension.rsplit(".", 1)[1]
            if alias not in selected:
                raise FederationError(
                    f"HAVING references {hf.dimension!r}, which is not in query.measures.",
                    reason="having_unknown_measure",
                )

    return MergeSpec(
        primary_index=primary_idx,
        bridges=bridge_joins,
        dimensions=dimension_outputs,
        measures=measure_outputs,
        having=list(q.having),
        order_by=list(q.order),
        limit=q.limit,
        offset=q.offset,
        mode=mode,
        cross_partition_clauses=resolved_cross,
    )


def _build_distributive_spec(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    primary_partition: Dialect,
    partitions: list[_PartitionPlan],
    bridges: list[_Bridge],
    output_column_meta: list[ColumnMeta],
    *,
    cross_partition_clauses: _Cnf | None = None,
) -> MergeSpec:
    """Distributive merge spec: fragments pre-aggregate (SUM / pre-counted
    / decomposed AVG), so each measure re-aggregates with SUM at the merge
    (``avg_recomposed`` for averages). Delegates assembly to
    :func:`_build_merge_spec`."""
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}

    time_output: _TimeOutput | None = None
    if q.time_dimension is not None:
        idx = cube_to_idx[_resolve_field_to_cube(q.time_dimension.dimension, catalog).name]
        td_col = partitions[idx].dim_columns[q.time_dimension.dimension]
        # Distributive buckets in the fragment; the merge passes the
        # column through (grain None).
        time_output = (td_col, FragmentColumn(idx, td_col), None)

    measure_outputs: list[MeasureOutput] = []
    for ref in q.measures:
        idx = cube_to_idx[_resolve_field_to_cube(ref, catalog).name]
        plan = partitions[idx]
        m_name = ref.rsplit(".", 1)[1]
        meta = next(m for m in output_column_meta if m.name == m_name)
        if ref in plan.avg_columns:
            sum_col, count_col = plan.avg_columns[ref]
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="avg_recomposed",
                    sum_source=FragmentColumn(idx, sum_col),
                    count_source=FragmentColumn(idx, count_col),
                    column_meta=meta,
                )
            )
        else:
            col = plan.measure_columns[ref]
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="sum",  # both sum and count reduce to SUM at merge
                    source=FragmentColumn(idx, col),
                    column_meta=meta,
                )
            )

    return _build_merge_spec(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_column_meta,
        mode="distributive",
        measure_outputs=measure_outputs,
        time_output=time_output,
        cross_partition_clauses=cross_partition_clauses,
    )


def _resolve_cross_clauses(
    clauses: _Cnf,
    cube_to_idx: dict[str, int],
) -> tuple[_ResolvedCrossClause, ...]:
    """Resolve a cross-partition CNF clause list to fragment coordinates.

    Each ``Filter.dimension`` (``cube.dim``) becomes
    ``(negated, fragment_index, column_name, op, values)``. The column
    name is the unqualified dimension — the where-router force-projects
    such a dimension under that bare name (see ``_route_where_clauses``),
    so the merge can reference ``f{idx}.{column_name}`` directly. Hashable
    so :class:`MergeSpec` stays frozen, and self-describing so a renderer
    needs neither the catalog nor the cube→index map.
    """
    out: list[_ResolvedCrossClause] = []
    for clause in clauses:
        resolved: list[_ResolvedCrossLiteral] = []
        for neg, f in clause:
            cube_name, dim_name = f.dimension.split(".", 1)
            resolved.append((neg, cube_to_idx[cube_name], dim_name, f.op, tuple(f.values)))
        out.append(tuple(resolved))
    return tuple(out)


# ---------------------------------------------------------------------------
# Raw-row mode — lifts non-distributive aggregations and having
# ---------------------------------------------------------------------------


# Synthetic-dim name prefix for a measure's raw source SQL in raw_rows mode.
_RAW_MEASURE_PREFIX = "__rm_"


def _raw_measure_col(measure_name: str) -> str:
    """Output-column name carrying a measure's raw source value."""
    return _RAW_MEASURE_PREFIX + measure_name


@dataclass
class _RawRowPartitionPlan:
    """Per-backend sub-query for raw-row federation."""

    dialect: Dialect
    cubes: list[Cube]
    sub_query: SemanticQuery
    raw_measure_columns: dict[str, tuple[str, str]]
    ratio_measure_columns: dict[str, tuple[str, str, str, str]]
    dim_columns: dict[str, str]
    bridge_columns: dict[str, str]
    time_col: str | None = None
    time_grain: str | None = None
    time_dim_ref: str | None = None


# ---------------------------------------------------------------------------
# CNF conversion — splits a where-tree across federation partitions
# ---------------------------------------------------------------------------


_CnfLiteral = tuple[bool, Filter]
_CnfClause = list[_CnfLiteral]
_Cnf = list[_CnfClause]


def _literal_to_tuple(lit: BoolExpr | Filter) -> _CnfLiteral:
    """Map a core-CNF literal to federation's ``(negated, Filter)`` form.

    The core engine emits a literal as either a bare ``Filter`` or a
    ``BoolExpr(op="not", children=[Filter])`` — the only two literal
    shapes :func:`semql.cnf.to_cnf` produces."""
    if isinstance(lit, Filter):
        return (False, lit)
    inner = lit.children[0]
    if lit.op != "not" or not isinstance(inner, Filter):  # pragma: no cover - guard
        raise FederationError(
            f"unexpected CNF literal shape: {lit!r}",
            reason="cnf_internal",
        )
    return (True, inner)


def _clause_node_to_clause(node: BoolExpr | Filter) -> _CnfClause:
    """Flatten one top-level AND child (a literal or an OR of literals)
    into a list of ``(negated, Filter)`` literals."""
    if isinstance(node, BoolExpr) and node.op == "or":
        return [_literal_to_tuple(c) for c in node.children]
    return [_literal_to_tuple(node)]


def _to_cnf(node: BoolExpr | Filter) -> _Cnf:
    """Normalise ``node`` to CNF via the shared :func:`semql.cnf.to_cnf`
    engine, then flatten its ``AND(OR(...), ...)`` tree into the
    ``(negated, Filter)`` clause list the router and emitters consume.

    Delegating to the core engine drops a duplicate CNF implementation and
    gives federation its dedup + idempotence (``a OR a`` → ``a``) for free.
    """
    normalised = core_to_cnf(node)
    if isinstance(normalised, BoolExpr) and normalised.op == "and":
        return [_clause_node_to_clause(child) for child in normalised.children]
    return [_clause_node_to_clause(normalised)]


def _clause_to_boolexpr(clause: _CnfClause) -> BoolExpr | Filter:
    def lit(neg: bool, f: Filter) -> BoolExpr | Filter:
        return BoolExpr(op="not", children=[f]) if neg else f

    if len(clause) == 1:
        return lit(*clause[0])
    return BoolExpr(op="or", children=[lit(n, f) for n, f in clause])


def _partition_for_dim(
    dim_ref: str,
    cube_to_partition: dict[str, int],
) -> int | None:
    cube_name = dim_ref.split(".", 1)[0]
    return cube_to_partition.get(cube_name)


def _clauses_to_boolexpr(clauses: _Cnf) -> BoolExpr | Filter:
    branches = [_clause_to_boolexpr(c) for c in clauses]
    if len(branches) == 1:
        return branches[0]
    return BoolExpr(op="and", children=branches)


def _route_where_clauses(
    where: BoolExpr | Filter,
    cube_to_idx: dict[str, int],
    dim_columns_per_partition: list[dict[str, str]],
) -> tuple[dict[int, _Cnf], _Cnf, dict[int, list[str]]]:
    """Split a where-tree into per-partition CNF clauses + a cross-
    partition residual.

    A clause whose literals all resolve to one partition is applied
    inside that fragment (``per_partition``); a clause spanning
    partitions becomes a residual the merge step applies post-join
    (``cross``), and every dimension such a clause references that the
    fragment doesn't already project is force-projected
    (``extra_dims``) so the merge SELECT can see it.

    Pure computation shared by the distributive and raw_rows routing —
    neither plan dataclass leaks in, only the per-partition
    ``dim_columns`` maps and the cube→partition index.
    """
    cnf = _to_cnf(where)
    per_partition: dict[int, _Cnf] = {}
    cross: _Cnf = []
    extra_dims: dict[int, list[str]] = {}

    for clause in cnf:
        idxs: set[int] = set()
        for _negated, lit in clause:
            owner_idx = _partition_for_dim(lit.dimension, cube_to_idx)
            if owner_idx is None:
                raise FederationError(
                    f"where: dimension {lit.dimension!r} resolutions error.",
                    reason="where_references_untouched_cube",
                )
            idxs.add(owner_idx)
        if len(idxs) == 1:
            (idx,) = tuple(idxs)
            per_partition.setdefault(idx, []).append(clause)
        else:
            cross.append(clause)
            for _negated, lit in clause:
                idx = cube_to_idx[lit.dimension.split(".", 1)[0]]
                already = dim_columns_per_partition[idx]
                pending = extra_dims.setdefault(idx, [])
                if lit.dimension not in already and lit.dimension not in pending:
                    pending.append(lit.dimension)
    return per_partition, cross, extra_dims


def _route_where_tree(
    where: BoolExpr | Filter,
    partitions: list[_RawRowPartitionPlan],
) -> tuple[list[_RawRowPartitionPlan], _Cnf]:
    cube_to_idx: dict[str, int] = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    per_partition, cross, extra_dims = _route_where_clauses(
        where, cube_to_idx, [p.dim_columns for p in partitions]
    )

    out: list[_RawRowPartitionPlan] = []
    for idx, p in enumerate(partitions):
        clauses_for_p = per_partition.get(idx)
        dims_for_p = extra_dims.get(idx, [])
        if not clauses_for_p and not dims_for_p:
            out.append(p)
            continue
        sub_q = p.sub_query
        sub_q_updates: dict[str, object] = {}
        if clauses_for_p:
            new_where: BoolExpr | Filter = _clauses_to_boolexpr(clauses_for_p)
            if sub_q.where is not None:
                new_where = BoolExpr(op="and", children=[sub_q.where, new_where])
            sub_q_updates["where"] = new_where
        new_dim_columns = dict(p.dim_columns)
        if dims_for_p:
            existing_dims = list(sub_q.dimensions)
            to_add = [d for d in dims_for_p if d not in existing_dims]
            sub_q_updates["dimensions"] = existing_dims + to_add
            for d in to_add:
                new_dim_columns[d] = d.rsplit(".", 1)[1]
        new_sub_q = sub_q.model_copy(update=sub_q_updates)
        out.append(
            _RawRowPartitionPlan(
                dialect=p.dialect,
                cubes=p.cubes,
                sub_query=new_sub_q,
                raw_measure_columns=p.raw_measure_columns,
                ratio_measure_columns=p.ratio_measure_columns,
                dim_columns=new_dim_columns,
                bridge_columns=p.bridge_columns,
                time_col=p.time_col,
                time_grain=p.time_grain,
                time_dim_ref=p.time_dim_ref,
            )
        )
    return out, cross


def _route_where_distributive(
    where: BoolExpr | Filter,
    partitions: list[_PartitionPlan],
) -> tuple[list[_PartitionPlan], _Cnf]:
    """Distributive analogue of :func:`_route_where_tree`.

    Single-partition clauses AND into the owning fragment's ``where``;
    a cross-partition clause stays as a residual the merge applies
    post-join.  Dimensions a cross clause references but the fragment
    doesn't already project are added to the fragment's GROUP BY (and
    its ``dim_columns`` map) so the merge SELECT can reference them —
    correct for the distributive sum/count aggs, since grouping by an
    extra key and re-summing at merge is exact.
    """
    cube_to_idx: dict[str, int] = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    per_partition, cross, extra_dims = _route_where_clauses(
        where, cube_to_idx, [p.dim_columns for p in partitions]
    )

    out: list[_PartitionPlan] = []
    for idx, p in enumerate(partitions):
        clauses_for_p = per_partition.get(idx)
        dims_for_p = extra_dims.get(idx, [])
        if not clauses_for_p and not dims_for_p:
            out.append(p)
            continue
        sub_q = p.sub_query
        sub_q_updates: dict[str, object] = {}
        if clauses_for_p:
            new_where: BoolExpr | Filter = _clauses_to_boolexpr(clauses_for_p)
            if sub_q.where is not None:
                new_where = BoolExpr(op="and", children=[sub_q.where, new_where])
            sub_q_updates["where"] = new_where
        new_dim_columns = dict(p.dim_columns)
        if dims_for_p:
            existing_dims = list(sub_q.dimensions)
            to_add = [d for d in dims_for_p if d not in existing_dims]
            sub_q_updates["dimensions"] = existing_dims + to_add
            for d in to_add:
                new_dim_columns[d] = d.rsplit(".", 1)[1]
        new_sub_q = sub_q.model_copy(update=sub_q_updates)
        out.append(
            _PartitionPlan(
                dialect=p.dialect,
                cubes=p.cubes,
                sub_query=new_sub_q,
                measure_columns=p.measure_columns,
                avg_columns=p.avg_columns,
                dim_columns=new_dim_columns,
                bridge_columns=p.bridge_columns,
            )
        )
    return out, cross


_RAW_TIME_PREFIX = "__rt_"


def _raw_time_col(time_dim_name: str) -> str:
    return _RAW_TIME_PREFIX + time_dim_name


def _projected_measure_sql(m: Measure) -> str:
    is_count_star = m.agg == "count" and m.sql.strip() == "*"
    body = "1" if is_count_star else m.sql
    if m.filter:
        return f"CASE WHEN {m.filter} THEN {body} ELSE NULL END"
    return body


def _build_partition_sub_query_raw_rows(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    partition_cubes: list[Cube],
    primary_partition: Dialect,
    bridges: list[_Bridge],
) -> _RawRowPartitionPlan:
    partition_names = {c.name for c in partition_cubes}
    dialect = partition_cubes[0].dialect
    is_primary = dialect is primary_partition
    raw_measure_columns: dict[str, tuple[str, str]] = {}
    ratio_measure_columns: dict[str, tuple[str, str, str, str]] = {}
    synthetic_dims: dict[str, list[Dimension]] = {}
    projected_raw: set[tuple[str, str]] = set()

    def _register_raw(owner_cube_name: str, m: Measure) -> tuple[str, str]:
        is_count_star_no_filter = m.agg == "count" and m.sql.strip() == "*" and not m.filter
        if is_count_star_no_filter:
            return ("", "count")
        raw_col = _raw_measure_col(m.name)
        key = (owner_cube_name, m.name)
        if key not in projected_raw:
            projected_raw.add(key)
            raw_dim = Dimension(name=raw_col, sql=_projected_measure_sql(m), type="number")
            synthetic_dims.setdefault(owner_cube_name, []).append(raw_dim)
        return (raw_col, m.agg)

    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        if owner.name not in partition_names:
            if is_primary:
                raise FederationError(
                    f"Measure {ref!r} non-primary partition.", reason="measure_non_primary"
                )
            continue
        m_name = ref.rsplit(".", 1)[1]
        m = next(x for x in owner.measures if x.name == m_name)
        if m.agg == "ratio":
            assert m.numerator is not None and m.denominator is not None
            num_m = next(x for x in owner.measures if x.name == m.numerator)
            den_m = next(x for x in owner.measures if x.name == m.denominator)
            if num_m.agg == "ratio" or den_m.agg == "ratio":
                raise FederationError(
                    "Nested ratio measures are not supported in raw_rows federation.",
                    reason="nested_ratio_in_raw_rows",
                )
            num_col, num_agg = _register_raw(owner.name, num_m)
            den_col, den_agg = _register_raw(owner.name, den_m)
            ratio_measure_columns[ref] = (num_col, num_agg, den_col, den_agg)
            continue
        raw_col, agg = _register_raw(owner.name, m)
        raw_measure_columns[ref] = (raw_col, agg)

    sub_dimensions: list[str] = []
    dim_columns: dict[str, str] = {}
    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        if owner.name in partition_names:
            sub_dimensions.append(ref)
            dim_columns[ref] = ref.rsplit(".", 1)[1]

    time_col: str | None = None
    time_grain: str | None = None
    time_dim_ref: str | None = None
    extra_filters: list[Filter] = []
    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        if td_cube.name in partition_names:
            td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
            td_field = next(t for t in td_cube.time_dimensions if t.name == td_name)
            raw_name = _raw_time_col(td_name)
            synthetic_dims.setdefault(td_cube.name, []).append(
                Dimension(name=raw_name, sql=td_field.sql, type="time")
            )
            time_col, time_grain = raw_name, q.time_dimension.granularity
            time_dim_ref = q.time_dimension.dimension
            sub_dimensions.append(f"{td_cube.name}.{raw_name}")
            start, end = q.time_dimension.range
            extra_filters.append(
                Filter(dimension=f"{td_cube.name}.{raw_name}", op="gte", values=[start])
            )
            extra_filters.append(
                Filter(dimension=f"{td_cube.name}.{raw_name}", op="lt", values=[end])
            )

    if synthetic_dims:
        for i, cube in enumerate(partition_cubes):
            if extra := synthetic_dims.get(cube.name):
                partition_cubes[i] = cube.model_copy(
                    update={"dimensions": [*cube.dimensions, *extra]}
                )

    sub_filters: list[Filter] = [
        f for f in q.filters if _resolve_field_to_cube(f.dimension, catalog).name in partition_names
    ] + extra_filters
    sub_segments = [s for s in q.segments if s.split(".", 1)[0] in partition_names]
    bridge_columns: dict[str, str] = {}
    for b in bridges:
        if b.left_cube.name in partition_names:
            ref = f"{b.left_cube.name}.{b.left_dim}"
            if ref not in sub_dimensions:
                sub_dimensions.append(ref)
            bridge_columns[ref] = b.left_dim
        if b.right_cube.name in partition_names:
            ref = f"{b.right_cube.name}.{b.right_dim}"
            if ref not in sub_dimensions:
                sub_dimensions.append(ref)
            bridge_columns[ref] = b.right_dim

    for ref, (col, _) in raw_measure_columns.items():
        if col:
            sub_dimensions.append(f"{_resolve_field_to_cube(ref, catalog).name}.{col}")
    for _, (num_col, _, den_col, _) in ratio_measure_columns.items():
        for cube_name in partition_names:
            for col in (num_col, den_col):
                if col and any(d.name == col for d in synthetic_dims.get(cube_name, [])):
                    qualified = f"{cube_name}.{col}"
                    if qualified not in sub_dimensions:
                        sub_dimensions.append(qualified)

    sub_query = SemanticQuery(
        measures=[],
        dimensions=sub_dimensions,
        filters=sub_filters,
        segments=sub_segments,
        ungrouped=True,
    )
    return _RawRowPartitionPlan(
        dialect=dialect,
        cubes=partition_cubes,
        sub_query=sub_query,
        raw_measure_columns=raw_measure_columns,
        ratio_measure_columns=ratio_measure_columns,
        dim_columns=dim_columns,
        bridge_columns=bridge_columns,
        time_col=time_col,
        time_grain=time_grain,
        time_dim_ref=time_dim_ref,
    )


def _build_raw_rows_spec(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    primary_partition: Dialect,
    partitions: list[_RawRowPartitionPlan],
    bridges: list[_Bridge],
    output_column_meta: list[ColumnMeta],
    *,
    cross_partition_clauses: _Cnf | None = None,
) -> MergeSpec:
    """Raw-rows merge spec: fragments stream ungrouped rows, so the merge
    applies the full aggregation (and time bucketing, and HAVING). The
    measure ``merge_agg`` carries the raw aggregate (``sum`` / ``avg`` /
    a percentile / …); ``ratio`` records its per-side aggregates.
    Delegates assembly to :func:`_build_merge_spec`."""
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}

    time_output: _TimeOutput | None = None
    if q.time_dimension is not None:
        idx = cube_to_idx[_resolve_field_to_cube(q.time_dimension.dimension, catalog).name]
        plan = partitions[idx]
        assert plan.time_col is not None
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        # The merge buckets raw timestamps; the output alias carries the
        # grain (e.g. ``created_at_day``) when one is set.
        time_alias = f"{td_name}_{plan.time_grain}" if plan.time_grain else td_name
        time_output = (time_alias, FragmentColumn(idx, plan.time_col), plan.time_grain)

    measure_outputs: list[MeasureOutput] = []
    for ref in q.measures:
        idx = cube_to_idx[_resolve_field_to_cube(ref, catalog).name]
        plan = partitions[idx]
        m_name = ref.rsplit(".", 1)[1]
        meta = next(m for m in output_column_meta if m.name == m_name)
        if ref in plan.ratio_measure_columns:
            num_col, num_agg, den_col, den_agg = plan.ratio_measure_columns[ref]
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="ratio",
                    numerator=FragmentColumn(idx, num_col),
                    denominator=FragmentColumn(idx, den_col),
                    numerator_agg=_as_merge_agg(num_agg),
                    denominator_agg=_as_merge_agg(den_agg),
                    column_meta=meta,
                )
            )
        else:
            col, agg = plan.raw_measure_columns[ref]
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg=_as_merge_agg(agg),
                    source=FragmentColumn(idx, col),
                    column_meta=meta,
                )
            )

    return _build_merge_spec(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_column_meta,
        mode="raw_rows",
        measure_outputs=measure_outputs,
        time_output=time_output,
        cross_partition_clauses=cross_partition_clauses,
    )


def _merge_meta_for_dim(
    ref: str,
    partitions: Sequence[_MergePartition],
    fragments: list[CompiledQuery],
    cube_to_idx: dict[str, int],
) -> ColumnMeta:
    """Output ColumnMeta for a merged dimension — inherits the fragment's
    column meta (kind/unit/display) under the unqualified output name."""
    idx = cube_to_idx[ref.split(".", 1)[0]]
    col = partitions[idx].dim_columns[ref]
    for cm in fragments[idx].column_meta:
        if cm.name == col:
            return ColumnMeta(
                name=ref.rsplit(".", 1)[1],
                kind=cm.kind,
                display_name=cm.display_name,
                unit=cm.unit,
                display_unit=cm.display_unit,
                format=cm.format,
            )
    return ColumnMeta(name=ref.rsplit(".", 1)[1], kind="dimension")


def _merge_meta_for_measure(ref: str, catalog: dict[str, Cube]) -> ColumnMeta:
    """Output ColumnMeta for a merged measure — carries the cube measure's
    unit/format so the merged result presents like a single-source query."""
    owner = _resolve_field_to_cube(ref, catalog)
    m_name = ref.rsplit(".", 1)[1]
    m = next(x for x in owner.measures if x.name == m_name)
    return ColumnMeta(
        name=m_name,
        kind="measure",
        display_name=m.display_name or m_name.replace("_", " ").title(),
        unit=m.unit,
        display_unit=m.display_unit,
        format=m.format,
    )


def _merge_output_columns(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    partitions: Sequence[_MergePartition],
    fragments: list[CompiledQuery],
    *,
    time_output: tuple[str, ColumnMeta] | None,
) -> tuple[list[str], list[ColumnMeta]]:
    """The user-facing output columns + meta for a merged plan: dims, then
    the (mode-specific) time column, then measures. Shared by both
    pipelines — only ``time_output`` differs (raw-rows buckets the name)."""
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    columns = [r.rsplit(".", 1)[1] for r in q.dimensions]
    meta = [_merge_meta_for_dim(r, partitions, fragments, cube_to_idx) for r in q.dimensions]
    if time_output is not None:
        col, cm = time_output
        columns.append(col)
        meta.append(cm)
    for r in q.measures:
        columns.append(r.rsplit(".", 1)[1])
        meta.append(_merge_meta_for_measure(r, catalog))
    return columns, meta


def _compile_raw_rows(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    grouped: dict[Dialect, list[Cube]],
    backend_order: list[Dialect],
    primary_partition: Dialect,
    bridges: list[_Bridge],
    *,
    context: dict[str, str] | None,
    group_by_alias: bool,
    having_alias: bool,
    dialects: dict[Dialect, DialectStrategy] | None,
    viewer: AuthContext | None,
    policy: PolicyFn | None,
    scope_fns: dict[str, ScopeFn] | None,
) -> FederatedPlan:
    partitions = [
        _build_partition_sub_query_raw_rows(
            q, catalog, list(grouped[b]), primary_partition, bridges
        )
        for b in backend_order
    ]
    cross_partition_clauses: _Cnf = []
    if q.where is not None:
        partitions, cross_partition_clauses = _route_where_tree(q.where, partitions)

    fragments = [
        compile_query(
            p.sub_query,
            _scoped_catalog(p.cubes),
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
            _allow_unbounded_ungrouped=True,
        )
        for p in partitions
    ]

    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    time_output: tuple[str, ColumnMeta] | None = None
    if q.time_dimension:
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        plan = partitions[cube_to_idx[q.time_dimension.dimension.split(".", 1)[0]]]
        # Raw-rows buckets at merge, so the output column carries the grain.
        td_col = f"{td_name}_{plan.time_grain}" if plan.time_grain else td_name
        time_output = (td_col, ColumnMeta(name=td_col, kind="time", display_name=td_col))
    output_columns, output_column_meta = _merge_output_columns(
        q, catalog, partitions, fragments, time_output=time_output
    )

    merge_spec = _build_raw_rows_spec(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_column_meta,
        cross_partition_clauses=cross_partition_clauses,
    )
    return FederatedPlan(
        fragments=fragments,
        merge_spec=merge_spec,
        columns=output_columns,
        column_meta=output_column_meta,
    )


# ---------------------------------------------------------------------------
# Cross-backend symmetric aggregation
#
# The federated twin of :func:`semql.logical._detect_symmetric_agg`. Two or
# more additive-measure facts that conform to one shared bridge cube, but live
# on *different* backends, can still be combined safely: pre-aggregate each
# fact to the conformed key on its own backend, then LEFT-join the per-fact
# results onto the bridge (the entity universe) at the merge. Because each
# fact aggregates to the grouping grain *before* the cross-fragment join, there
# is no fan-out — this is categorically different from the join-before-aggregate
# chasm trap. Making the bridge the primary fragment keeps the merge to plain
# LEFT joins with single-source dimensions, the shapes the renderer already
# emits.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SymmetricFact:
    cube: Cube
    key_dim: str  # dimension on the fact exposing its conformed key to the bridge
    measure_refs: tuple[str, ...]  # qualified measure refs owned by this fact


@dataclass(frozen=True)
class _CrossBackendSymmetric:
    bridge: Cube
    bridge_key_dim: str  # dimension on the bridge exposing the conformed key
    facts: tuple[_SymmetricFact, ...]


def _detect_cross_backend_symmetric(
    q: SemanticQuery, catalog: dict[str, Cube], touched: list[Cube]
) -> _CrossBackendSymmetric | None:
    """Recognise the cross-backend symmetric shape, or ``None`` to fall back
    to the ``measures_span_backends`` refusal.

    Conservative on purpose — mirrors the single-backend detector: two or more
    fact cubes each carrying an additive (``sum`` / ``count``) measure, all the
    *many* side of a join to one shared bridge cube that is the only non-fact
    touched cube; every projected dimension on the bridge; conformed keys are
    simple ``{a}.col = {b}.col`` equality on the same bridge column; no time
    breakdown, where-tree, derived measures, segments, having, compare, or
    ungrouped. Anything outside that stays a refusal so the emit path never
    sees a half-supported query."""
    if (
        q.ungrouped
        or q.where is not None
        or q.derived_measures
        or q.segments
        or q.having
        or q.compare is not None
        or q.time_dimension is not None
    ):
        return None
    if not q.measures:
        return None
    # Require a grouping dimension on the bridge. Without one the bridge isn't
    # even referenced (so it never enters ``touched``), and pure cross-backend
    # totals are two independent scalar aggregates — a simpler plan than this
    # symmetric one. Refuse that here; it can grow its own path later.
    if not q.dimensions:
        return None

    fact_order: list[str] = []
    fact_by_name: dict[str, Cube] = {}
    measures_by_fact: dict[str, list[str]] = {}
    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        m_name = ref.rsplit(".", 1)[1]
        m = next((x for x in owner.measures if x.name == m_name), None)
        if m is None or m.agg not in ("sum", "count"):
            return None
        if owner.name not in fact_by_name:
            fact_by_name[owner.name] = owner
            fact_order.append(owner.name)
        measures_by_fact.setdefault(owner.name, []).append(ref)
    if len(fact_order) < 2:
        return None
    # Must genuinely span backends — otherwise a non-federated path applies.
    if len({fact_by_name[n].dialect for n in fact_order}) < 2:
        return None

    # Shared many-side parent (the bridge), from declared joins among touched
    # cubes — same basis as the chasm-trap guard.
    in_scope = {c.name for c in touched}
    parents: dict[str, set[str]] = {}
    join_by_pair: dict[tuple[str, str], Join] = {}
    for cube in touched:
        for j in cube.joins:
            if j.to not in in_scope:
                continue
            join_by_pair[(cube.name, j.to)] = j
            if j.relationship == "many_to_one":
                parents.setdefault(cube.name, set()).add(j.to)
            elif j.relationship == "one_to_many":
                parents.setdefault(j.to, set()).add(cube.name)
    shared = set(parents.get(fact_order[0], set()))
    for name in fact_order[1:]:
        shared &= parents.get(name, set())
    shared -= set(fact_order)
    if len(shared) != 1:
        return None
    bridge_name = next(iter(shared))
    if bridge_name not in catalog:
        return None
    bridge = catalog[bridge_name]
    # The bridge must be the only non-fact touched cube.
    if {c.name for c in touched} != set(fact_order) | {bridge_name}:
        return None
    # Every projected dimension must live on the bridge.
    for ref in q.dimensions:
        if _resolve_field_to_cube(ref, catalog).name != bridge_name:
            return None

    facts: list[_SymmetricFact] = []
    bridge_key_dims: set[str] = set()
    for name in fact_order:
        fact_cube = fact_by_name[name]
        fwd = join_by_pair.get((name, bridge_name))
        rev = join_by_pair.get((bridge_name, name))
        # _parse_bridge validates simple equality + key-type compatibility. A
        # non-simple join means the shape isn't eligible (fall back to the
        # generic refusal); a real type-coercion hazard is worth surfacing.
        try:
            if fwd is not None:
                br = _parse_bridge(fact_cube, bridge, fwd)
                fact_dim, bridge_dim = br.left_dim, br.right_dim
            elif rev is not None:
                br = _parse_bridge(bridge, fact_cube, rev)
                fact_dim, bridge_dim = br.right_dim, br.left_dim
            else:
                return None
        except FederationError as exc:
            if exc.reason == "cross_cube_type_coercion":
                raise
            return None
        bridge_key_dims.add(bridge_dim)
        facts.append(
            _SymmetricFact(
                cube=fact_cube, key_dim=fact_dim, measure_refs=tuple(measures_by_fact[name])
            )
        )
    # All facts must conform on the same bridge column.
    if len(bridge_key_dims) != 1:
        return None
    return _CrossBackendSymmetric(
        bridge=bridge, bridge_key_dim=bridge_key_dims.pop(), facts=tuple(facts)
    )


def _compile_cross_backend_symmetric(
    q: SemanticQuery,
    sym: _CrossBackendSymmetric,
    *,
    context: dict[str, str] | None,
    group_by_alias: bool,
    having_alias: bool,
    dialects: dict[Dialect, DialectStrategy] | None,
    views: dict[str, View] | None,
    viewer: AuthContext | None,
    policy: PolicyFn | None,
    scope_fns: dict[str, ScopeFn] | None,
) -> FederatedPlan:
    """Emit the cross-backend symmetric plan: a bridge fragment (the entity
    universe + the requested dimensions) plus one pre-aggregated fragment per
    fact, LEFT-joined to the bridge on the conformed key at the merge."""

    def _compile(sub: SemanticQuery, cubes: list[Cube]) -> CompiledQuery:
        return compile_query(
            sub,
            _scoped_catalog(cubes),
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            views=views,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )

    # Fragment 0 (primary): the bridge, projecting requested dims + the
    # conformed key (needed for the join, not necessarily output).
    bridge_key_ref = f"{sym.bridge.name}.{sym.bridge_key_dim}"
    bridge_dims = list(q.dimensions)
    if bridge_key_ref not in bridge_dims:
        bridge_dims.append(bridge_key_ref)
    bridge_frag = _compile(SemanticQuery(measures=[], dimensions=bridge_dims), [sym.bridge])
    fragments: list[CompiledQuery] = [bridge_frag]

    bridges: list[BridgeJoin] = []
    measure_outputs: list[MeasureOutput] = []
    for idx, fact in enumerate(sym.facts, start=1):
        fact_key_ref = f"{fact.cube.name}.{fact.key_dim}"
        frag = _compile(
            SemanticQuery(measures=list(fact.measure_refs), dimensions=[fact_key_ref]),
            [fact.cube],
        )
        fragments.append(frag)
        bridges.append(
            BridgeJoin(
                left=FragmentColumn(0, sym.bridge_key_dim),
                right=FragmentColumn(idx, fact.key_dim),
            )
        )
        for ref in fact.measure_refs:
            m_name = ref.rsplit(".", 1)[1]
            meta = next(m for m in frag.column_meta if m.name == m_name)
            # Each fragment already aggregated to the conformed key; the merge
            # sums per group (identity when grouping by the key, a real fold
            # when grouping by a coarser bridge dimension). ``count`` reduces
            # to a sum-of-counts, exactly as the distributive path does.
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="sum",
                    source=FragmentColumn(idx, m_name),
                    column_meta=meta,
                )
            )

    dimension_outputs: list[DimensionOutput] = []
    for ref in q.dimensions:
        alias = ref.rsplit(".", 1)[1]
        meta = next(m for m in bridge_frag.column_meta if m.name == alias)
        dimension_outputs.append(
            DimensionOutput(output_name=alias, sources=[FragmentColumn(0, alias)], column_meta=meta)
        )

    merge_spec = MergeSpec(
        primary_index=0,
        bridges=bridges,
        dimensions=dimension_outputs,
        measures=measure_outputs,
        having=[],
        order_by=list(q.order),
        limit=q.limit,
        offset=q.offset,
        mode="distributive",
        cross_partition_clauses=(),
    )
    output_columns = [d.output_name for d in dimension_outputs]
    output_columns += [m.output_name for m in measure_outputs]
    output_column_meta = [d.column_meta for d in dimension_outputs]
    output_column_meta += [m.column_meta for m in measure_outputs]
    return FederatedPlan(
        fragments=fragments,
        merge_spec=merge_spec,
        columns=output_columns,
        column_meta=output_column_meta,
    )


def compile_federated_query(
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
) -> FederatedPlan:
    if q.compare is not None:
        raise FederationError(
            "Federated compare-mode is not supported in v1.",
            reason="compare_in_federated",
        )
    if q.having and mode == "distributive":
        raise FederationError(
            "Federated queries cannot use HAVING in distributive mode.",
            reason="having_in_distributive_federated",
        )

    touched = _touched(q, catalog)
    if not touched:
        raise FederationError("Empty query.", reason="empty")
    backends_seen = {c.dialect for c in touched}

    if len(backends_seen) == 1:
        c = compile_query(
            q,
            catalog,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            views=views,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )
        return FederatedPlan(
            fragments=[c],
            merge_spec=MergeSpec(
                primary_index=0,
                bridges=[],
                dimensions=[
                    DimensionOutput(
                        output_name=cm.name, sources=[FragmentColumn(0, cm.name)], column_meta=cm
                    )
                    for cm in c.column_meta
                    if cm.kind in ("dimension", "time")
                ],
                measures=[
                    MeasureOutput(
                        output_name=cm.name,
                        merge_agg="passthrough",
                        source=FragmentColumn(0, cm.name),
                        column_meta=cm,
                    )
                    for cm in c.column_meta
                    if cm.kind in ("measure", "computed")
                ],
                having=[],
                order_by=[],
                limit=None,
                offset=None,
                mode="distributive",
                cross_partition_clauses=(),
            ),
            columns=c.columns,
            column_meta=c.column_meta,
        )

    if q.measures:
        primary_partition = _resolve_field_to_cube(q.measures[0], catalog).dialect
        measures_span = any(
            _resolve_field_to_cube(ref, catalog).dialect is not primary_partition
            for ref in q.measures[1:]
        )
        if measures_span:
            # Cross-backend additive measures conforming to one shared bridge
            # are fan-safe — pre-aggregate per fact, then LEFT-join onto the
            # bridge. Anything else still refuses.
            sym = _detect_cross_backend_symmetric(q, catalog, touched)
            if sym is not None:
                return _compile_cross_backend_symmetric(
                    q,
                    sym,
                    context=context,
                    group_by_alias=group_by_alias,
                    having_alias=having_alias,
                    dialects=dialects,
                    views=views,
                    viewer=viewer,
                    policy=policy,
                    scope_fns=scope_fns,
                )
            raise FederationError("Measures span backends.", reason="measures_span_backends")
    else:
        primary_partition = touched[0].dialect

    bridges = _find_bridges(touched, catalog)
    if not bridges:
        raise FederationError("No cross-backend join.", reason="no_cross_backend_join")

    grouped: dict[Dialect, list[Cube]] = {}
    for cube in touched:
        grouped.setdefault(cube.dialect, []).append(cube)
    backend_order = [primary_partition] + [b for b in grouped if b is not primary_partition]

    if mode == "raw_rows":
        return _compile_raw_rows(
            q,
            catalog,
            grouped,
            backend_order,
            primary_partition,
            bridges,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )

    partitions = [
        _build_partition_sub_query(q, catalog, list(grouped[b]), primary_partition, bridges)
        for b in backend_order
    ]
    # Route the where-tree: per-partition clauses fold into each
    # fragment, the cross-partition residual rides into the merge SQL.
    cross_partition_clauses: _Cnf = []
    if q.where is not None:
        partitions, cross_partition_clauses = _route_where_distributive(q.where, partitions)
    fragments = [
        compile_query(
            p.sub_query,
            _scoped_catalog(p.cubes),
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            dialects=dialects,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )
        for p in partitions
    ]
    cube_to_idx = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    time_output: tuple[str, ColumnMeta] | None = None
    if q.time_dimension:
        ref = q.time_dimension.dimension
        # Distributive already buckets in the fragment, so the merge
        # passes the dimension column through under its bucketed name
        # (e.g. ``created_at_day``). The fragment's column meta is named
        # for the bare dimension, so rename it to the bucketed output
        # column — otherwise the SELECT-list alias and its meta disagree
        # and the assembler can't find the meta (granularity != None).
        td_col = partitions[cube_to_idx[ref.split(".", 1)[0]]].dim_columns[ref]
        td_meta = _merge_meta_for_dim(ref, partitions, fragments, cube_to_idx)
        time_output = (td_col, dc_replace(td_meta, name=td_col))
    output_columns, output_column_meta = _merge_output_columns(
        q, catalog, partitions, fragments, time_output=time_output
    )

    merge_spec = _build_distributive_spec(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_column_meta,
        cross_partition_clauses=cross_partition_clauses,
    )
    return FederatedPlan(
        fragments=fragments,
        merge_spec=merge_spec,
        columns=output_columns,
        column_meta=output_column_meta,
    )


__all__ = [
    "FEDERATED_PLAN_VERSION",
    "FederatedPlan",
    "FederationError",
    "FederationMode",
    "compile_federated_query",
]
