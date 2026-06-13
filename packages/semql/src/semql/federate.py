"""Cross-source compilation: emit per-backend fragments + a DuckDB
merge plan when a query's touched cubes span backends.

The sans-io counterpart of :func:`semql.compile.compile_query` for
federated queries. Each fragment is a normal :class:`CompiledQuery` for its
backend; the merge SQL is always DuckDB dialect — DuckDB is the lingua
franca of the federation layer (both for our in-process executor and
for sans-io callers who want to materialise results into DuckDB-
compatible tooling).

v1 restrictions, refused with :class:`FederationError`:

- Measures must live on a single "primary" backend partition. Dim
  cubes on other backends contribute lookup attributes only.
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
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Literal

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
class MergePlan:
    """How to combine per-fragment result sets into the final result.

    ``sql`` references fragments as DuckDB tables named ``frag_0``,
    ``frag_1``, … indexed into ``FederatedPlan.fragments``. The
    executor (or any caller doing the merge manually) must materialise
    each fragment under that name before running ``sql``.
    """

    sql: str
    params: dict[str, object] = dc_field(default_factory=lambda: {})


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


@dataclass(frozen=True)
class MeasureOutput:
    output_name: str
    merge_agg: Literal[
        "sum",  # SUM(source)
        "count",  # SUM(source) -- fragment pre-counted
        "min",  # raw_rows only
        "max",  # raw_rows only
        "count_distinct",  # raw_rows only
        "avg_recomposed",  # SUM(sum_src) / NULLIF(SUM(cnt_src), 0)
        "ratio",  # SUM(num) / NULLIF(SUM(den), 0)
        "passthrough",  # single-fragment; no re-aggregation
    ]
    column_meta: ColumnMeta
    source: FragmentColumn | None = None
    sum_source: FragmentColumn | None = None  # avg_recomposed
    count_source: FragmentColumn | None = None  # avg_recomposed
    numerator: FragmentColumn | None = None  # ratio
    denominator: FragmentColumn | None = None  # ratio


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
    # CNF clauses that touch more than one partition. Each clause is
    # a list of (negated: bool, dimension: str, op: str, values: tuple)
    # tuples. The merge applies them as a post-join WHERE combined
    # with AND across clauses and OR within a clause. Used by the
    # distributive-mode where-tree lift (v1 carryover from raw_rows).
    cross_partition_clauses: tuple[tuple[tuple[bool, str, str, tuple[object, ...]], ...], ...] = ()


# Format version of the federation plan IR.  Bumped when the
# fragment / merge / merge_spec shape changes in a way an out-of-tree
# executor would need to know about (``MergeSpec`` travels across
# package versions — see the executor's ``ANN401`` note).  Stamped on
# every :class:`FederatedPlan` so a consumer can detect a compiler /
# executor skew instead of mis-reading a changed shape.
FEDERATED_PLAN_VERSION = 1


@dataclass(frozen=True)
class FederatedPlan:
    """A query that touches cubes on multiple backends.

    ``fragments[i]`` is a :class:`CompiledQuery` to run against its
    respective backend. Results are loaded into a DuckDB table named
    ``frag_i`` and the ``merge.sql`` produces the final shape.

    ``columns`` and ``column_meta`` describe the final output shape
    after the merge — same role they play on :class:`CompiledQuery`.

    Frozen — like every other node in the federation IR.  Derive a
    tweaked plan with :func:`dataclasses.replace`, never by attribute
    assignment, so the fragments can't silently desync from the merge.
    ``version`` carries :data:`FEDERATED_PLAN_VERSION`.
    """

    fragments: list[CompiledQuery]
    merge: MergePlan
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
    return list(resolved.touched_cubes)


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


def _quote_ident(name: str) -> str:
    """DuckDB identifier quoting. Cube/dimension names already match
    ``[a-z_][a-z0-9_]*`` so quoting is conservative-but-safe."""
    return f'"{name}"'


def _emit_merge_sql(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    primary_partition: Dialect,
    partitions: list[_PartitionPlan],
    bridges: list[_Bridge],
    output_columns: list[str],
    output_column_meta: list[ColumnMeta],
    *,
    cross_partition_clauses: _Cnf | None = None,
) -> tuple[str, MergeSpec, dict[str, object]]:
    """Generate the DuckDB merge SQL.

    Frags are aliased ``f0``, ``f1``, ... matching the partitions order.
    The primary fragment is the FROM target; every other fragment
    LEFT JOINs onto it via the appropriate bridge.

    Returns ``(sql, spec, params)`` — the params map carries any
    cross-partition filter values, bound as ``$name`` placeholders
    rather than inlined as literals.
    """
    binder = _MergeBinder()
    # Index the partitions by backend for join-key lookup.
    by_dialect: dict[Dialect, tuple[int, _PartitionPlan]] = {
        p.dialect: (i, p) for i, p in enumerate(partitions)
    }
    primary_idx, _primary_plan = by_dialect[primary_partition]

    def frag_alias(idx: int) -> str:
        return f"f{idx}"

    # Map cube → fragment index, so we know which alias to reference
    # for any (cube.field) ref in the final SELECT.
    cube_to_idx: dict[str, int] = {}
    for i, p in enumerate(partitions):
        for c in p.cubes:
            cube_to_idx[c.name] = i

    # SELECT list — one expression per output column the user asked for.
    select_exprs: list[str] = []
    dimension_outputs: list[DimensionOutput] = []
    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        col = plan.dim_columns[ref]
        alias = ref.rsplit(".", 1)[1]
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(col)} AS {_quote_ident(alias)}")
        # In v1 distributive, a dimension always comes from exactly one source.
        meta = next(m for m in output_column_meta if m.name == alias)
        dimension_outputs.append(
            DimensionOutput(output_name=alias, sources=[FragmentColumn(idx, col)], column_meta=meta)
        )

    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx = cube_to_idx[td_cube.name]
        plan = partitions[idx]
        td_col = plan.dim_columns[q.time_dimension.dimension]
        td_alias = td_col
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(td_col)} AS {_quote_ident(td_alias)}")
        meta = next(m for m in output_column_meta if m.name == td_alias)
        dimension_outputs.append(
            DimensionOutput(
                output_name=td_alias, sources=[FragmentColumn(idx, td_col)], column_meta=meta
            )
        )

    # Measures (re-aggregation).
    measure_outputs: list[MeasureOutput] = []
    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        m_name = ref.rsplit(".", 1)[1]
        meta = next(m for m in output_column_meta if m.name == m_name)
        if ref in plan.avg_columns:
            sum_col, count_col = plan.avg_columns[ref]
            expr = (
                f"SUM({frag_alias(idx)}.{_quote_ident(sum_col)}) / "
                f"NULLIF(SUM({frag_alias(idx)}.{_quote_ident(count_col)}), 0)"
            )
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
            expr = f"SUM({frag_alias(idx)}.{_quote_ident(col)})"
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="sum",  # both sum and count reduce to SUM at merge
                    source=FragmentColumn(idx, col),
                    column_meta=meta,
                )
            )
        select_exprs.append(f"{expr} AS {_quote_ident(m_name)}")

    # FROM + JOINs.
    from_clause = f"frag_{primary_idx} AS {frag_alias(primary_idx)}"
    joins_sql: list[str] = []
    bridge_joins: list[BridgeJoin] = []
    joined: set[int] = {primary_idx}
    pending = list(bridges)
    while pending:
        progress = False
        remaining: list[_Bridge] = []
        for b in pending:
            left_idx = cube_to_idx[b.left_cube.name]
            right_idx = cube_to_idx[b.right_cube.name]
            if left_idx in joined and right_idx not in joined:
                new_idx = right_idx
                left_alias = frag_alias(left_idx)
                right_alias = frag_alias(new_idx)
                left_col = partitions[left_idx].bridge_columns[f"{b.left_cube.name}.{b.left_dim}"]
                right_col = partitions[new_idx].bridge_columns[f"{b.right_cube.name}.{b.right_dim}"]
                bridge_joins.append(
                    BridgeJoin(
                        left=FragmentColumn(left_idx, left_col),
                        right=FragmentColumn(new_idx, right_col),
                    )
                )
            elif right_idx in joined and left_idx not in joined:
                new_idx = left_idx
                left_alias = frag_alias(right_idx)
                right_alias = frag_alias(new_idx)
                left_col = partitions[right_idx].bridge_columns[
                    f"{b.right_cube.name}.{b.right_dim}"
                ]
                right_col = partitions[new_idx].bridge_columns[f"{b.left_cube.name}.{b.left_dim}"]
                bridge_joins.append(
                    BridgeJoin(
                        left=FragmentColumn(right_idx, left_col),
                        right=FragmentColumn(new_idx, right_col),
                    )
                )
            else:
                remaining.append(b)
                continue
            joins_sql.append(
                f"LEFT JOIN frag_{new_idx} AS {right_alias} "
                f"ON {left_alias}.{_quote_ident(left_col)} = "
                f"{right_alias}.{_quote_ident(right_col)}"
            )
            joined.add(new_idx)
            progress = True
        pending = remaining
        if not progress and pending:
            raise FederationError(
                "Federated plan has disconnected backend partitions.",
                reason="disconnected_partitions",
            )

    # GROUP BY all non-measure SELECT items.
    n_group = len(q.dimensions) + (1 if q.time_dimension is not None else 0)
    group_by = ", ".join(str(i + 1) for i in range(n_group))

    sql = f"SELECT {', '.join(select_exprs)} FROM {from_clause}"
    if joins_sql:
        sql += " " + " ".join(joins_sql)
    if cross_partition_clauses:
        cube_to_idx_m = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
        clauses_sql = " AND ".join(
            _emit_cross_partition_clause(c, cube_to_idx_m, binder) for c in cross_partition_clauses
        )
        sql += f" WHERE {clauses_sql}"
    if q.measures and n_group > 0:
        sql += f" GROUP BY {group_by}"

    if q.order:
        order_terms: list[str] = []
        for ref, direction in q.order:
            alias = ref.rsplit(".", 1)[-1] if "." in ref else ref
            order_terms.append(f"{_quote_ident(alias)} {'DESC' if direction == 'desc' else 'ASC'}")
        sql += f" ORDER BY {', '.join(order_terms)}"

    if q.limit is not None:
        sql += f" LIMIT {int(q.limit)}"
    if q.offset is not None and q.offset > 0:
        sql += f" OFFSET {int(q.offset)}"

    spec = MergeSpec(
        primary_index=primary_idx,
        bridges=bridge_joins,
        dimensions=dimension_outputs,
        measures=measure_outputs,
        having=list(q.having),
        order_by=list(q.order),
        limit=q.limit,
        offset=q.offset,
        mode="distributive",
        cross_partition_clauses=_serialise_cnf(cross_partition_clauses or []),
    )
    return sql, spec, binder.params


def _serialise_cnf(
    clauses: _Cnf,
) -> tuple[tuple[tuple[bool, str, str, tuple[object, ...]], ...], ...]:
    """Serialise a CNF clause list into the MergeSpec-friendly form.

    Each clause is a list of literals; each literal is a (negated,
    dimension, op, values) tuple. The serialised form is hashable
    so MergeSpec can stay frozen.
    """
    out: list[tuple[tuple[bool, str, str, tuple[object, ...]], ...]] = []
    for clause in clauses:
        out.append(tuple((neg, f.dimension, f.op, tuple(f.values)) for neg, f in clause))
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


def _negate_tree(node: BoolExpr | Filter) -> BoolExpr | Filter:
    if isinstance(node, Filter):
        return BoolExpr(op="not", children=[node])
    if node.op == "not":
        return node.children[0]
    new_op: str = "and" if node.op == "or" else "or"
    return BoolExpr(
        op=new_op,  # type: ignore[arg-type]
        children=[_negate_tree(c) for c in node.children],
    )


def _to_cnf(node: BoolExpr | Filter) -> _Cnf:
    if isinstance(node, Filter):
        return [[(False, node)]]
    if node.op == "not":
        inner = node.children[0]
        if isinstance(inner, Filter):
            return [[(True, inner)]]
        return _to_cnf(_negate_tree(inner))
    if node.op == "and":
        out: _Cnf = []
        for child in node.children:
            out.extend(_to_cnf(child))
        return out
    child_cnfs = [_to_cnf(c) for c in node.children]
    result = child_cnfs[0]
    for nxt in child_cnfs[1:]:
        result = [lc + rc for lc in result for rc in nxt]
    return result


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


class _MergeBinder:
    """Bind merge-SQL filter values as DuckDB named parameters.

    The merge step runs in-process DuckDB over fragment result sets;
    DuckDB reads ``$name`` placeholders against the params mapping. Every
    cross-partition residual / HAVING value flows through here so it
    binds rather than inlining — ``Filter.values`` are LLM/user-derived,
    and the federation invariant is "values bind as parameters, never as
    literals." Names are ``m0``, ``m1``, … kept distinct from the
    ``p*`` fragment params (which live in separate dicts regardless)."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}

    def bind(self, value: object) -> str:
        name = f"m{len(self.params)}"
        self.params[name] = value
        return f"${name}"


def _emit_filter_predicate(col_expr: str, f: Filter, binder: _MergeBinder) -> str:
    op = f.op
    vals = list(f.values)
    if op == "eq":
        return f"{col_expr} = {binder.bind(vals[0])}"
    if op == "neq":
        return f"{col_expr} <> {binder.bind(vals[0])}"
    if op == "gt":
        return f"{col_expr} > {binder.bind(vals[0])}"
    if op == "gte":
        return f"{col_expr} >= {binder.bind(vals[0])}"
    if op == "lt":
        return f"{col_expr} < {binder.bind(vals[0])}"
    if op == "lte":
        return f"{col_expr} <= {binder.bind(vals[0])}"
    if op == "in":
        items = ", ".join(binder.bind(v) for v in vals)
        return f"{col_expr} IN ({items})"
    if op == "not_in":
        items = ", ".join(binder.bind(v) for v in vals)
        return f"{col_expr} NOT IN ({items})"
    if op == "is_null":
        return f"{col_expr} IS NULL"
    if op == "not_null":
        return f"{col_expr} IS NOT NULL"
    if op == "contains":
        return f"{col_expr} ILIKE {binder.bind('%' + str(vals[0]) + '%')}"
    raise FederationError(f"Filter op {op!r} unsupported.", reason="unsupported_op")


def _emit_cross_partition_clause(
    clause: _CnfClause,
    cube_to_idx: dict[str, int],
    binder: _MergeBinder,
) -> str:
    lits: list[str] = []
    for negated, f in clause:
        cube_name, dim_name = f.dimension.split(".", 1)
        idx = cube_to_idx[cube_name]
        col_expr = f"f{idx}.{_quote_ident(dim_name)}"
        pred = _emit_filter_predicate(col_expr, f, binder)
        if negated:
            pred = f"NOT ({pred})"
        lits.append(pred)
    if len(lits) == 1:
        return lits[0]
    return f"({' OR '.join(lits)})"


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


def _emit_merge_sql_raw_rows(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    primary_partition: Dialect,
    partitions: list[_RawRowPartitionPlan],
    bridges: list[_Bridge],
    output_columns: list[str],
    output_column_meta: list[ColumnMeta],
    *,
    cross_partition_clauses: _Cnf | None = None,
) -> tuple[str, MergeSpec, dict[str, object]]:
    binder = _MergeBinder()
    by_dialect: dict[Dialect, tuple[int, _RawRowPartitionPlan]] = {
        p.dialect: (i, p) for i, p in enumerate(partitions)
    }
    primary_idx, _ = by_dialect[primary_partition]

    def frag_alias(idx: int) -> str:
        return f"f{idx}"

    cube_to_idx: dict[str, int] = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
    select_exprs: list[str] = []
    dimension_outputs: list[DimensionOutput] = []
    n_group_dims = 0

    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        idx, plan = cube_to_idx[owner.name], partitions[cube_to_idx[owner.name]]
        col, alias = plan.dim_columns[ref], ref.rsplit(".", 1)[1]
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(col)} AS {_quote_ident(alias)}")
        meta = next(m for m in output_column_meta if m.name == alias)
        dimension_outputs.append(
            DimensionOutput(output_name=alias, sources=[FragmentColumn(idx, col)], column_meta=meta)
        )
        n_group_dims += 1

    time_alias: str | None = None
    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx, plan = cube_to_idx[td_cube.name], partitions[cube_to_idx[td_cube.name]]
        assert plan.time_col is not None
        raw_ref = f"{frag_alias(idx)}.{_quote_ident(plan.time_col)}"
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        bucket = f"date_trunc('{plan.time_grain}', {raw_ref})" if plan.time_grain else raw_ref
        time_alias = f"{td_name}_{plan.time_grain}" if plan.time_grain else td_name
        select_exprs.append(f"{bucket} AS {_quote_ident(time_alias)}")
        meta = next(m for m in output_column_meta if m.name == time_alias)
        dimension_outputs.append(
            DimensionOutput(
                output_name=time_alias,
                sources=[FragmentColumn(idx, plan.time_col)],
                column_meta=meta,
            )
        )
        n_group_dims += 1

    measure_outputs: list[MeasureOutput] = []
    selected_measure_aliases = {ref.rsplit(".", 1)[1] for ref in q.measures}
    for having_filter in q.having:
        alias = having_filter.dimension.rsplit(".", 1)[1]
        if alias not in selected_measure_aliases:
            raise FederationError(
                f"HAVING references {having_filter.dimension!r}, which is not in query.measures.",
                reason="having_unknown_measure",
            )

    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        idx, plan = cube_to_idx[owner.name], partitions[cube_to_idx[owner.name]]
        m_name = ref.rsplit(".", 1)[1]
        meta = next(m for m in output_column_meta if m.name == m_name)
        if ref in plan.ratio_measure_columns:
            num_col, num_agg, den_col, den_agg = plan.ratio_measure_columns[ref]
            num_expr, den_expr = (
                _raw_agg_expr(num_agg, num_col, frag_alias(idx)),
                _raw_agg_expr(den_agg, den_col, frag_alias(idx)),
            )
            expr = f"{num_expr} / NULLIF({den_expr}, 0)"
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg="ratio",
                    numerator=FragmentColumn(idx, num_col),
                    denominator=FragmentColumn(idx, den_col),
                    column_meta=meta,
                )
            )
        else:
            col, agg = plan.raw_measure_columns[ref]
            expr = _raw_agg_expr(agg, col, frag_alias(idx))
            measure_outputs.append(
                MeasureOutput(
                    output_name=m_name,
                    merge_agg=agg,  # type: ignore[arg-type]
                    source=FragmentColumn(idx, col),
                    column_meta=meta,
                )
            )
        select_exprs.append(f"{expr} AS {_quote_ident(m_name)}")

    from_clause = f"frag_{primary_idx} AS {frag_alias(primary_idx)}"
    joins_sql: list[str] = []
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
            host_alias, new_alias = frag_alias(host_idx), frag_alias(new_idx)

            # Resolve which side of the bridge corresponds to which fragment
            if host_idx == cube_to_idx[b.left_cube.name]:
                h_cube, h_dim = b.left_cube.name, b.left_dim
                n_cube, n_dim = b.right_cube.name, b.right_dim
            else:
                h_cube, h_dim = b.right_cube.name, b.right_dim
                n_cube, n_dim = b.left_cube.name, b.left_dim

            host_col = partitions[host_idx].bridge_columns[f"{h_cube}.{h_dim}"]
            new_col = partitions[new_idx].bridge_columns[f"{n_cube}.{n_dim}"]

            joins_sql.append(
                f"LEFT JOIN frag_{new_idx} AS {new_alias} "
                f"ON {host_alias}.{_quote_ident(host_col)} = "
                f"{new_alias}.{_quote_ident(new_col)}"
            )
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
            raise FederationError("Disconnected partitions.", reason="disconnected")

    sql = f"SELECT {', '.join(select_exprs)} FROM {from_clause}"
    if joins_sql:
        sql += " " + " ".join(joins_sql)
    if cross_partition_clauses:
        clauses_sql = " AND ".join(
            _emit_cross_partition_clause(c, cube_to_idx, binder) for c in cross_partition_clauses
        )
        sql += f" WHERE {clauses_sql}"
    if q.measures and n_group_dims > 0:
        group_cols = ", ".join(str(i + 1) for i in range(n_group_dims))
        sql += f" GROUP BY {group_cols}"
    if q.having:
        having_sql = " AND ".join(
            _emit_having_term(hf.dimension.rsplit(".", 1)[1], hf, binder) for hf in q.having
        )
        sql += f" HAVING {having_sql}"
    if q.order:
        order_terms: list[str] = []
        for r, d in q.order:
            alias = r.rsplit(".", 1)[-1]
            dir_str = "DESC" if d == "desc" else "ASC"
            order_terms.append(f"{_quote_ident(alias)} {dir_str}")
        sql += f" ORDER BY {', '.join(order_terms)}"
    if q.limit:
        sql += f" LIMIT {int(q.limit)}"
    if q.offset:
        sql += f" OFFSET {int(q.offset)}"

    spec = MergeSpec(
        primary_index=primary_idx,
        bridges=bridge_joins,
        dimensions=dimension_outputs,
        measures=measure_outputs,
        having=list(q.having),
        order_by=list(q.order),
        limit=q.limit,
        offset=q.offset,
        mode="raw_rows",
        cross_partition_clauses=_serialise_cnf(cross_partition_clauses or []),
    )
    return sql, spec, binder.params


def _raw_agg_expr(agg: str, col: str, frag_alias: str) -> str:
    if not col:
        return "COUNT(*)"
    ref = f"{frag_alias}.{_quote_ident(col)}"
    if agg == "sum":
        return f"SUM({ref})"
    if agg == "count":
        return f"COUNT({ref})"
    if agg == "count_distinct":
        return f"COUNT(DISTINCT {ref})"
    if agg == "avg":
        return f"AVG({ref})"
    if agg == "min":
        return f"MIN({ref})"
    if agg == "max":
        return f"MAX({ref})"
    _q = {"median": 0.5, "p75": 0.75, "p90": 0.90, "p95": 0.95}
    if agg in _q:
        return f"PERCENTILE_CONT({_q[agg]}) WITHIN GROUP (ORDER BY {ref})"
    raise FederationError(f"agg={agg!r} unsupported.", reason="unsupported_agg")


def _emit_having_term(alias: str, f: Filter, binder: _MergeBinder) -> str:
    col, op, val = _quote_ident(alias), f.op, binder.bind(f.values[0])
    if op == "gt":
        return f"{col} > {val}"
    if op == "gte":
        return f"{col} >= {val}"
    if op == "lt":
        return f"{col} < {val}"
    if op == "lte":
        return f"{col} <= {val}"
    if op == "eq":
        return f"{col} = {val}"
    if op == "neq":
        return f"{col} <> {val}"
    raise FederationError(f"HAVING op {op!r} unsupported.", reason="unsupported_having_op")


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

    def _meta_for_dim(ref: str) -> ColumnMeta:
        idx, plan = cube_to_idx[ref.split(".", 1)[0]], partitions[cube_to_idx[ref.split(".", 1)[0]]]
        col = plan.dim_columns[ref]
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

    def _meta_for_measure(ref: str) -> ColumnMeta:
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

    output_columns = [r.rsplit(".", 1)[1] for r in q.dimensions]
    output_column_meta = [_meta_for_dim(r) for r in q.dimensions]
    if q.time_dimension:
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        plan = partitions[cube_to_idx[q.time_dimension.dimension.split(".", 1)[0]]]
        td_col = f"{td_name}_{plan.time_grain}" if plan.time_grain else td_name
        output_columns.append(td_col)
        output_column_meta.append(ColumnMeta(name=td_col, kind="time", display_name=td_col))
    for r in q.measures:
        output_columns.append(r.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_measure(r))

    merge_sql, merge_spec, merge_params = _emit_merge_sql_raw_rows(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_columns,
        output_column_meta,
        cross_partition_clauses=cross_partition_clauses,
    )
    return FederatedPlan(
        fragments=fragments,
        merge=MergePlan(sql=merge_sql, params=merge_params),
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
            merge=MergePlan(sql="SELECT * FROM frag_0"),
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
        for ref in q.measures[1:]:
            if _resolve_field_to_cube(ref, catalog).dialect is not primary_partition:
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

    def _meta_for_dim(ref: str) -> ColumnMeta:
        idx, plan = cube_to_idx[ref.split(".", 1)[0]], partitions[cube_to_idx[ref.split(".", 1)[0]]]
        col = plan.dim_columns[ref]
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

    def _meta_for_measure(ref: str) -> ColumnMeta:
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

    output_columns = [r.rsplit(".", 1)[1] for r in q.dimensions]
    output_column_meta = [_meta_for_dim(r) for r in q.dimensions]
    if q.time_dimension:
        td_col = partitions[cube_to_idx[q.time_dimension.dimension.split(".", 1)[0]]].dim_columns[
            q.time_dimension.dimension
        ]
        output_columns.append(td_col)
        output_column_meta.append(_meta_for_dim(q.time_dimension.dimension))
    for r in q.measures:
        output_columns.append(r.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_measure(r))

    merge_sql, merge_spec, merge_params = _emit_merge_sql(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_columns,
        output_column_meta,
        cross_partition_clauses=cross_partition_clauses,
    )
    return FederatedPlan(
        fragments=fragments,
        merge=MergePlan(sql=merge_sql, params=merge_params),
        merge_spec=merge_spec,
        columns=output_columns,
        column_meta=output_column_meta,
    )


__all__ = [
    "FEDERATED_PLAN_VERSION",
    "FederatedPlan",
    "FederationError",
    "FederationMode",
    "MergePlan",
    "compile_federated_query",
]
