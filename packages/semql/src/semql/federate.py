"""Cross-source compilation: emit per-backend fragments + a DuckDB
merge plan when a query's touched cubes span backends.

The sans-io counterpart of :func:`semql.compile.compile_query` for
federated queries. Each fragment is a normal :class:`Compiled` for its
backend; the merge SQL is always DuckDB dialect — DuckDB is the lingua
franca of the federation layer (both for our in-process executor and
for sans-io callers who want to materialise results into DuckDB-
compatible tooling).

v1 restrictions, refused with :class:`FederationError`:

- Measures must live on a single "primary" backend partition. Dim
  cubes on other backends contribute lookup attributes only.
- Bridge joins between partitions must be equality on a single column
  pair, with both sides declared as ``Dimension``\\s on their cubes
  (the federation layer cannot project columns the catalogue hasn't
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

from semql.compile import ColumnMeta, Compiled, compile_query
from semql.errors import FederationError
from semql.introspect import resolve_query
from semql.model import Backend, Cube, Dimension, Join, Measure
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
    from semql.backend import BackendStrategy
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


@dataclass
class FederatedPlan:
    """A query that touches cubes on multiple backends.

    ``fragments[i]`` is a :class:`Compiled` to run against its
    respective backend. Results are loaded into a DuckDB table named
    ``frag_i`` and the ``merge.sql`` produces the final shape.

    ``columns`` and ``column_meta`` describe the final output shape
    after the merge — same role they play on :class:`Compiled`.
    """

    fragments: list[Compiled]
    merge: MergePlan
    columns: list[str]
    column_meta: list[ColumnMeta]


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
    expose those columns (the catalogue must declare them — see module
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
    return _Bridge(
        left_cube=left_cube,
        right_cube=right_cube,
        left_col=lc,
        right_col=rc,
        left_dim=left_dim,
        right_dim=right_dim,
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
            if target.backend == cube.backend:
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
    if "." not in ref:
        raise FederationError(
            f"Reference {ref!r} must be qualified as ``cube.field``.",
            reason="unqualified_reference",
        )
    cube_name = ref.split(".", 1)[0]
    if cube_name not in catalog:
        raise FederationError(
            f"Reference {ref!r}: unknown cube {cube_name!r}.",
            reason="unknown_cube",
        )
    return catalog[cube_name]


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

    backend: Backend
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
    primary_partition: Backend,
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

    Refuses cross-partition where-trees, segments, and queries with
    measures on non-primary partitions.
    """
    partition_names = {c.name for c in partition_cubes}
    backend = partition_cubes[0].backend
    is_primary = backend is primary_partition

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
                    f"backend {owner.backend.value!r}, which is not the "
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
        # No segments / where / having in v1 federated subqueries.
        order=[],
        # Ordering & limits happen at merge.
    )
    return _PartitionPlan(
        backend=backend,
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
    primary_partition: Backend,
    partitions: list[_PartitionPlan],
    bridges: list[_Bridge],
    output_columns: list[str],
) -> str:
    """Generate the DuckDB merge SQL.

    Frags are aliased ``f0``, ``f1``, ... matching the partitions order.
    The primary fragment is the FROM target; every other fragment
    LEFT JOINs onto it via the appropriate bridge.
    """
    # Index the partitions by backend for join-key lookup.
    by_backend: dict[Backend, tuple[int, _PartitionPlan]] = {
        p.backend: (i, p) for i, p in enumerate(partitions)
    }
    primary_idx, _primary_plan = by_backend[primary_partition]

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
    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        # The dim column in the fragment's output keeps its original
        # dim name (compiler emits the field name unless there's a
        # collision; cross-fragment we control naming).
        col = plan.dim_columns[ref]
        alias = ref.rsplit(".", 1)[1]
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(col)} AS {_quote_ident(alias)}")

    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx = cube_to_idx[td_cube.name]
        plan = partitions[idx]
        td_col = plan.dim_columns[q.time_dimension.dimension]
        # Output column name matches compile_query's convention.
        td_alias = td_col
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(td_col)} AS {_quote_ident(td_alias)}")

    # Measures (re-aggregation).
    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        m_name = ref.rsplit(".", 1)[1]
        if ref in plan.avg_columns:
            sum_col, count_col = plan.avg_columns[ref]
            expr = (
                f"SUM({frag_alias(idx)}.{_quote_ident(sum_col)}) / "
                f"NULLIF(SUM({frag_alias(idx)}.{_quote_ident(count_col)}), 0)"
            )
        else:
            col = plan.measure_columns[ref]
            # sum-of-sums and sum-of-counts both reduce to SUM at merge.
            expr = f"SUM({frag_alias(idx)}.{_quote_ident(col)})"
        select_exprs.append(f"{expr} AS {_quote_ident(m_name)}")

    # FROM + JOINs.
    from_clause = f"frag_{primary_idx} AS {frag_alias(primary_idx)}"
    joins_sql: list[str] = []
    joined: set[int] = {primary_idx}
    # Walk bridges in declaration order; each bridge connects exactly
    # one already-joined fragment to a not-yet-joined fragment.
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
            elif right_idx in joined and left_idx not in joined:
                new_idx = left_idx
                left_alias = frag_alias(right_idx)
                right_alias = frag_alias(new_idx)
                left_col = partitions[right_idx].bridge_columns[
                    f"{b.right_cube.name}.{b.right_dim}"
                ]
                right_col = partitions[new_idx].bridge_columns[f"{b.left_cube.name}.{b.left_dim}"]
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
            unreached = [f"{b.left_cube.name}<->{b.right_cube.name}" for b in pending]
            raise FederationError(
                "Federated plan has disconnected backend partitions — "
                "the bridge join graph is not a connected tree rooted at "
                f"the primary partition. Unreached bridges: {unreached}. "
                "Add a join from a primary-partition cube to each "
                "satellite, or run them as separate queries.",
                reason="disconnected_partitions",
            )

    # GROUP BY all non-measure SELECT items.
    n_group = len(q.dimensions) + (1 if q.time_dimension is not None else 0)
    group_by = ", ".join(str(i + 1) for i in range(n_group))

    sql = f"SELECT {', '.join(select_exprs)} FROM {from_clause}"
    if joins_sql:
        sql += " " + " ".join(joins_sql)
    if q.measures and n_group > 0:
        sql += f" GROUP BY {group_by}"

    # ORDER BY + LIMIT/OFFSET — apply against output column aliases.
    if q.order:
        order_terms: list[str] = []
        for ref, direction in q.order:
            # Resolve output column name.
            alias = ref.rsplit(".", 1)[-1] if "." in ref else ref
            if alias not in output_columns:
                raise FederationError(
                    f"ORDER BY {ref!r}: not in the federated output columns {output_columns}.",
                    reason="order_by_unknown_column",
                )
            order_terms.append(f"{_quote_ident(alias)} {'DESC' if direction == 'desc' else 'ASC'}")
        sql += f" ORDER BY {', '.join(order_terms)}"

    if q.limit is not None:
        sql += f" LIMIT {int(q.limit)}"
    if q.offset is not None and q.offset > 0:
        sql += f" OFFSET {int(q.offset)}"

    return sql


# ---------------------------------------------------------------------------
# Raw-row mode — lifts non-distributive aggregations and having
# ---------------------------------------------------------------------------


# Synthetic-dim name prefix for a measure's raw source SQL in raw_rows mode.
# Picked so it can't collide with a user-declared dimension under semql's
# ``[a-z_][a-z0-9_]*`` naming rules.
_RAW_MEASURE_PREFIX = "__rm_"


def _raw_measure_col(measure_name: str) -> str:
    """Output-column name carrying a measure's raw source value."""
    return _RAW_MEASURE_PREFIX + measure_name


@dataclass
class _RawRowPartitionPlan:
    """Per-backend sub-query for raw-row federation.

    Mirrors :class:`_PartitionPlan` but the sub-query is ``ungrouped=True``
    — no aggregation in the fragment — and the merge step does all the
    work. ``raw_measure_columns`` maps each original measure ref to the
    column name + aggregator the merge SQL should apply.

    ``ratio_measure_columns`` carries the recursive expansion of a
    ratio measure: ``(num_col, num_agg, den_col, den_agg)``. Both
    operands are also projected via their own ``__rm_*`` raw column,
    and the merge composes ``<num_agg>(num_col) / NULLIF(<den_agg>(den_col), 0)``.

    ``time_grain`` is set when the query has a ``time_dimension`` in
    raw-row mode — the fragment emits the raw timestamp + the range
    filter, the merge applies ``date_trunc(grain, ...)`` for
    grouping. ``None`` when there is no time_dimension.
    """

    backend: Backend
    cubes: list[Cube]
    sub_query: SemanticQuery
    # Map measure ref ("orders.revenue") to (raw_col_name, agg). When
    # ``raw_col_name`` is empty the merge should use ``COUNT(*)``
    # rather than aggregating a column (the case for ``COUNT(*)``
    # measures that have no underlying column).
    raw_measure_columns: dict[str, tuple[str, str]]
    ratio_measure_columns: dict[str, tuple[str, str, str, str]]
    dim_columns: dict[str, str]
    bridge_columns: dict[str, str]
    # Time-dimension support in raw-row mode. When set, the fragment
    # projects the raw timestamp under ``time_col``; the merge applies
    # ``date_trunc(time_grain, frag.time_col)`` and groups by the
    # bucket. ``None`` when the query has no ``time_dimension`` on
    # this partition.
    time_col: str | None = None
    time_grain: str | None = None
    # Original ``cube.dim`` ref of the time_dimension so the merge
    # can route output meta correctly.
    time_dim_ref: str | None = None


# ---------------------------------------------------------------------------
# CNF conversion — splits a where-tree across federation partitions
# ---------------------------------------------------------------------------


# A literal is ``(negated, Filter)``; a clause is a disjunction of literals;
# the CNF is a conjunction of clauses. Kept as plain Python types so the
# routing logic doesn't have to import a dedicated AST.
_CnfLiteral = tuple[bool, Filter]
_CnfClause = list[_CnfLiteral]
_Cnf = list[_CnfClause]


def _negate_tree(node: BoolExpr | Filter) -> BoolExpr | Filter:
    """Apply De Morgan's: return the negated form of ``node``.

    Filter leaves get wrapped in a ``not`` BoolExpr; nested NOTs
    cancel; AND/OR swap. The recursion bottoms out at filter leaves —
    the CNF walker handles the leaf negation as a polarity bit on the
    literal."""
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
    """Convert a where-tree to CNF.

    Walks the BoolExpr / Filter tree, pushing NOTs to filter leaves
    via De Morgan's and distributing OR over AND. Returns a list of
    clauses (each a list of literals). Worst-case size is the
    product of all OR-branch sizes — fine for typical query trees,
    pathological for deeply nested ORs; we accept that since
    SemanticQuery callers don't tend to write deep boolean trees."""
    if isinstance(node, Filter):
        return [[(False, node)]]
    if node.op == "not":
        # Negation: recurse on the De Morgan'd child.
        inner = node.children[0]
        if isinstance(inner, Filter):
            return [[(True, inner)]]
        return _to_cnf(_negate_tree(inner))
    if node.op == "and":
        out: _Cnf = []
        for child in node.children:
            out.extend(_to_cnf(child))
        return out
    # op == "or": fold-distribute children's CNFs.
    child_cnfs = [_to_cnf(c) for c in node.children]
    result = child_cnfs[0]
    for nxt in child_cnfs[1:]:
        result = [lc + rc for lc in result for rc in nxt]
    return result


def _clause_to_boolexpr(clause: _CnfClause) -> BoolExpr | Filter:
    """Render a CNF clause back as a BoolExpr (or bare Filter when single).

    Used to attach a single-partition clause to a fragment's
    ``SemanticQuery.where`` — the fragment's compile path handles the
    BoolExpr natively."""

    def lit(neg: bool, f: Filter) -> BoolExpr | Filter:
        return BoolExpr(op="not", children=[f]) if neg else f

    if len(clause) == 1:
        return lit(*clause[0])
    return BoolExpr(op="or", children=[lit(n, f) for n, f in clause])


def _partition_for_dim(
    dim_ref: str,
    cube_to_partition: dict[str, int],
) -> int | None:
    """Which partition owns this ``cube.dim`` reference? ``None`` when
    the cube isn't in any partition (defensive — caller should have
    validated)."""
    cube_name = dim_ref.split(".", 1)[0]
    return cube_to_partition.get(cube_name)


def _clauses_to_boolexpr(clauses: _Cnf) -> BoolExpr | Filter:
    """AND-join a list of CNF clauses back into a single where-tree."""
    branches = [_clause_to_boolexpr(c) for c in clauses]
    if len(branches) == 1:
        return branches[0]
    return BoolExpr(op="and", children=branches)


def _route_where_tree(
    where: BoolExpr | Filter,
    partitions: list[_RawRowPartitionPlan],
) -> tuple[list[_RawRowPartitionPlan], _Cnf]:
    """Split a where-tree across partitions via CNF conversion.

    Returns ``(updated_partitions, cross_partition_clauses)``. Each
    clause whose literals all live on one partition gets AND-composed
    into that partition's ``sub_query.where``; clauses with leaves on
    multiple partitions stay in ``cross_partition_clauses`` and the
    merge emits them post-JOIN. Cross-partition leaf dimensions get
    auto-projected on their owning partition so the merge SQL can
    reference them.
    """
    cube_to_idx: dict[str, int] = {c.name: i for i, p in enumerate(partitions) for c in p.cubes}
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
                    f"where: dimension {lit.dimension!r} resolves to a "
                    "cube that isn't in any partition. The query's "
                    "where-tree must only reference cubes the query "
                    "actually touches.",
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
                if lit.dimension not in partitions[idx].dim_columns:
                    extra_dims.setdefault(idx, []).append(lit.dimension)

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
                backend=p.backend,
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


def _emit_filter_predicate(col_expr: str, f: Filter) -> str:
    """Render a single ``Filter`` as a DuckDB-dialect SQL predicate.

    Used by the cross-partition where-tree emit path. Values inline
    because the merge SQL is one self-contained string with no
    parameter envelope; the values originated from the user-supplied
    SemanticQuery and were typed-checked at construction time.
    """
    op = f.op
    vals = list(f.values)
    if op == "eq":
        return f"{col_expr} = {_lit(vals[0])}"
    if op == "neq":
        return f"{col_expr} <> {_lit(vals[0])}"
    if op == "gt":
        return f"{col_expr} > {_lit(vals[0])}"
    if op == "gte":
        return f"{col_expr} >= {_lit(vals[0])}"
    if op == "lt":
        return f"{col_expr} < {_lit(vals[0])}"
    if op == "lte":
        return f"{col_expr} <= {_lit(vals[0])}"
    if op == "in":
        items = ", ".join(_lit(v) for v in vals)
        return f"{col_expr} IN ({items})"
    if op == "not_in":
        items = ", ".join(_lit(v) for v in vals)
        return f"{col_expr} NOT IN ({items})"
    if op == "is_null":
        return f"{col_expr} IS NULL"
    if op == "not_null":
        return f"{col_expr} IS NOT NULL"
    if op == "contains":
        return f"{col_expr} ILIKE {_lit('%' + str(vals[0]) + '%')}"
    raise FederationError(
        f"Filter op {op!r} is not yet supported in cross-partition where-tree predicates.",
        reason=f"unsupported_cross_partition_op:{op}",
    )


def _emit_cross_partition_clause(
    clause: _CnfClause,
    cube_to_idx: dict[str, int],
) -> str:
    """Render one CNF clause as a merge-level disjunction.

    Each literal becomes ``f<idx>.<col> <op> <val>``; negated literals
    wrap in ``NOT (...)``. Clauses with multiple literals are wrapped
    in parens so the outer AND has the right binding."""
    lits: list[str] = []
    for negated, f in clause:
        cube_name, dim_name = f.dimension.split(".", 1)
        idx = cube_to_idx[cube_name]
        col_expr = f"f{idx}.{_quote_ident(dim_name)}"
        pred = _emit_filter_predicate(col_expr, f)
        if negated:
            pred = f"NOT ({pred})"
        lits.append(pred)
    if len(lits) == 1:
        return lits[0]
    return f"({' OR '.join(lits)})"


_RAW_TIME_PREFIX = "__rt_"


def _raw_time_col(time_dim_name: str) -> str:
    """Column name for a raw timestamp in raw_rows time-dimension mode."""
    return _RAW_TIME_PREFIX + time_dim_name


def _projected_measure_sql(m: Measure) -> str:
    """SQL projected for a measure's raw column in raw_rows mode.

    Filtered measures wrap in ``CASE WHEN <filter> THEN <sql> ELSE NULL END``
    — standard aggs (``SUM``/``AVG``/``COUNT``/``MIN``/``MAX``/
    ``COUNT(DISTINCT)``) all ignore NULL, so the case-when projection
    composes filter semantics into the merge agg without a separate
    FILTER clause. Filtered ``COUNT(*)`` measures (``sql="*"``) project
    a literal ``1`` so the merge can ``COUNT(<col>)`` over non-NULL.
    """
    is_count_star = m.agg == "count" and m.sql.strip() == "*"
    body = "1" if is_count_star else m.sql
    if m.filter:
        return f"CASE WHEN {m.filter} THEN {body} ELSE NULL END"
    return body


def _build_partition_sub_query_raw_rows(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    partition_cubes: list[Cube],
    primary_partition: Backend,
    bridges: list[_Bridge],
) -> _RawRowPartitionPlan:
    """Construct an ungrouped per-partition fragment for raw-row mode.

    Selects: every output dim this partition owns + every bridge key
    that touches a partition cube + a synthetic dimension exposing
    each measure's raw source SQL (wrapped in CASE-WHEN when the
    measure declares a ``filter=``). Ratio measures expand into their
    numerator + denominator raw columns and the merge recomposes the
    ratio. Time-dimension queries project the raw timestamp + push
    the time range as a Filter; the merge applies ``date_trunc()`` for
    grouping. The compiler runs with ``_allow_unbounded_ungrouped=True``
    because raw-row federation inherently produces fragment-cardinality
    rows.
    """
    partition_names = {c.name for c in partition_cubes}
    backend = partition_cubes[0].backend
    is_primary = backend is primary_partition

    raw_measure_columns: dict[str, tuple[str, str]] = {}
    ratio_measure_columns: dict[str, tuple[str, str, str, str]] = {}
    synthetic_dims: dict[str, list[Dimension]] = {}
    projected_raw: set[tuple[str, str]] = set()  # (cube_name, measure_name) dedupe

    def _register_raw(owner_cube_name: str, m: Measure) -> tuple[str, str]:
        """Project a measure's raw column on its owning cube.

        Returns ``(raw_col_name, agg)``. For COUNT(*) measures with no
        filter the raw col is ``""`` — the merge emits ``COUNT(*)``.
        Same measure registered twice (e.g. via a direct request *and*
        as a ratio operand) projects only once."""
        is_count_star_no_filter = m.agg == "count" and m.sql.strip() == "*" and not m.filter
        if is_count_star_no_filter:
            return ("", "count")
        raw_col = _raw_measure_col(m.name)
        key = (owner_cube_name, m.name)
        if key not in projected_raw:
            projected_raw.add(key)
            raw_dim = Dimension(
                name=raw_col,
                sql=_projected_measure_sql(m),
                type="number",
            )
            synthetic_dims.setdefault(owner_cube_name, []).append(raw_dim)
        return (raw_col, m.agg)

    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        if owner.name not in partition_names:
            if is_primary:
                raise FederationError(
                    f"Measure {ref!r} resolves to cube {owner.name!r} on "
                    f"backend {owner.backend.value!r}, which is not the "
                    f"primary measure partition {primary_partition.value!r}. "
                    f"All measures must live on one backend; the engine "
                    f"can't cross-source-join row-level measure values "
                    f"yet.",
                    reason="measure_on_non_primary_partition",
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
                    f"Measure {ref!r}: nested ratio measures (a ratio "
                    "operand that is itself a ratio) are not supported "
                    "in raw_rows federation mode. Flatten the ratio in "
                    "the catalog or compose client-side.",
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

    # Time dimension: project the raw timestamp via a synthetic
    # ``__rt_<name>`` Dimension (so the fragment compiler treats it
    # exactly like any other dim selection) and push the range as a
    # pair of Filter objects on the same synthetic dim so the
    # fragment runs ``WHERE __rt_x >= start AND __rt_x < end``. The
    # original granularity carries on the partition plan for the
    # merge step to apply ``date_trunc``.
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
            raw_dim = Dimension(name=raw_name, sql=td_field.sql, type="time")
            synthetic_dims.setdefault(td_cube.name, []).append(raw_dim)
            time_col = raw_name
            time_grain = q.time_dimension.granularity
            time_dim_ref = q.time_dimension.dimension
            sub_dimensions.append(f"{td_cube.name}.{raw_name}")
            start, end = q.time_dimension.range
            extra_filters.append(
                Filter(
                    dimension=f"{td_cube.name}.{raw_name}",
                    op="gte",
                    values=[start],
                )
            )
            extra_filters.append(
                Filter(
                    dimension=f"{td_cube.name}.{raw_name}",
                    op="lt",
                    values=[end],
                )
            )

    if synthetic_dims:
        for cube in list(partition_cubes):
            extra = synthetic_dims.get(cube.name)
            if not extra:
                continue
            patched = cube.model_copy(update={"dimensions": [*cube.dimensions, *extra]})
            partition_cubes[partition_cubes.index(cube)] = patched

    sub_filters: list[Filter] = []
    for f in q.filters:
        owner = _resolve_field_to_cube(f.dimension, catalog)
        if owner.name in partition_names:
            sub_filters.append(f)
    sub_filters.extend(extra_filters)

    # Segments route by qualified name: ``cube.segment`` belongs to
    # exactly one cube → exactly one partition. The fragment
    # compiler accepts ``segments=`` and AND-composes their
    # predicates into WHERE alongside ``filters``.
    sub_segments: list[str] = []
    for seg_ref in q.segments:
        seg_cube_name = seg_ref.split(".", 1)[0]
        if seg_cube_name in partition_names:
            sub_segments.append(seg_ref)

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

    # Project synthetic raw-measure dims (skip the synthetic-less
    # COUNT(*) cases).
    for ref, (col, _agg) in raw_measure_columns.items():
        if not col:
            continue
        owner = _resolve_field_to_cube(ref, catalog)
        sub_dimensions.append(f"{owner.name}.{col}")
    # Ratio operands also need their raw columns projected — the
    # ``_register_raw`` call above already added them to
    # ``synthetic_dims``, but the fragment SemanticQuery still has to
    # name them in its ``dimensions`` list.
    for _ref, (num_col, _na, den_col, _da) in ratio_measure_columns.items():
        for cube_name in partition_names:
            for col in (num_col, den_col):
                if not col:
                    continue
                qualified = f"{cube_name}.{col}"
                if qualified in sub_dimensions:
                    continue
                # Only add if this cube actually projects the col
                # (synthetic_dims tracks this).
                if any(d.name == col for d in synthetic_dims.get(cube_name, [])):
                    sub_dimensions.append(qualified)

    sub_query = SemanticQuery(
        measures=[],
        dimensions=sub_dimensions,
        filters=sub_filters,
        segments=sub_segments,
        ungrouped=True,
        order=[],
    )
    return _RawRowPartitionPlan(
        backend=backend,
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
    primary_partition: Backend,
    partitions: list[_RawRowPartitionPlan],
    bridges: list[_Bridge],
    output_columns: list[str],
    *,
    cross_partition_clauses: _Cnf | None = None,
) -> str:
    """Generate the DuckDB merge SQL for raw-row federation.

    Structure: FROM primary fragment LEFT JOIN satellites on bridge
    columns, optional WHERE for cross-partition where-tree clauses,
    GROUP BY output dims, SELECT dims + per-measure aggregation
    expression. ``having`` (if any) lives at this layer against the
    recomposed measure aliases. Cross-partition clauses are emitted
    post-JOIN / pre-GROUP-BY so they prune rows before aggregation.
    """
    by_backend: dict[Backend, tuple[int, _RawRowPartitionPlan]] = {
        p.backend: (i, p) for i, p in enumerate(partitions)
    }
    primary_idx, _primary_plan = by_backend[primary_partition]

    def frag_alias(idx: int) -> str:
        return f"f{idx}"

    cube_to_idx: dict[str, int] = {}
    for i, p in enumerate(partitions):
        for c in p.cubes:
            cube_to_idx[c.name] = i

    select_exprs: list[str] = []
    n_group_dims = 0
    for ref in q.dimensions:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        col = plan.dim_columns[ref]
        alias = ref.rsplit(".", 1)[1]
        select_exprs.append(f"{frag_alias(idx)}.{_quote_ident(col)} AS {_quote_ident(alias)}")
        n_group_dims += 1

    # Time-dimension bucket — the partition that owns the time dim
    # exposes the raw timestamp; the merge buckets via date_trunc.
    time_alias: str | None = None
    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx = cube_to_idx[td_cube.name]
        plan = partitions[idx]
        assert plan.time_col is not None, (
            "time_dimension was on this query but the partition plan "
            "didn't surface a time_col — partition build is out of sync."
        )
        raw_ref = f"{frag_alias(idx)}.{_quote_ident(plan.time_col)}"
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        if plan.time_grain is not None:
            bucket = f"date_trunc('{plan.time_grain}', {raw_ref})"
            time_alias = f"{td_name}_{plan.time_grain}"
        else:
            bucket = raw_ref
            time_alias = td_name
        select_exprs.append(f"{bucket} AS {_quote_ident(time_alias)}")
        n_group_dims += 1

    # Per-measure aggregation in the merge — the heart of raw-row mode.
    for ref in q.measures:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        m_name = ref.rsplit(".", 1)[1]
        if ref in plan.ratio_measure_columns:
            num_col, num_agg, den_col, den_agg = plan.ratio_measure_columns[ref]
            num_expr = _raw_agg_expr(num_agg, num_col, frag_alias(idx))
            den_expr = _raw_agg_expr(den_agg, den_col, frag_alias(idx))
            expr = f"{num_expr} / NULLIF({den_expr}, 0)"
        else:
            col, agg = plan.raw_measure_columns[ref]
            expr = _raw_agg_expr(agg, col, frag_alias(idx))
        select_exprs.append(f"{expr} AS {_quote_ident(m_name)}")

    # FROM + LEFT JOIN chain — identical shape to the distributive path.
    from_clause = f"frag_{primary_idx} AS {frag_alias(primary_idx)}"
    joins_sql: list[str] = []
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
            elif right_idx in joined and left_idx not in joined:
                new_idx = left_idx
                left_alias = frag_alias(right_idx)
                right_alias = frag_alias(new_idx)
                left_col = partitions[right_idx].bridge_columns[
                    f"{b.right_cube.name}.{b.right_dim}"
                ]
                right_col = partitions[new_idx].bridge_columns[f"{b.left_cube.name}.{b.left_dim}"]
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
            unreached = [f"{b.left_cube.name}<->{b.right_cube.name}" for b in pending]
            raise FederationError(
                "Federated plan has disconnected backend partitions — "
                "the bridge join graph is not a connected tree rooted at "
                f"the primary partition. Unreached bridges: {unreached}.",
                reason="disconnected_partitions",
            )

    group_by = ", ".join(str(i + 1) for i in range(n_group_dims))

    sql = f"SELECT {', '.join(select_exprs)} FROM {from_clause}"
    if joins_sql:
        sql += " " + " ".join(joins_sql)

    if cross_partition_clauses:
        where_terms = [
            _emit_cross_partition_clause(clause, cube_to_idx) for clause in cross_partition_clauses
        ]
        sql += f" WHERE {' AND '.join(where_terms)}"

    if q.measures and n_group_dims > 0:
        sql += f" GROUP BY {group_by}"

    # HAVING — raw-row mode applies it at merge against the recomposed
    # measure aliases. Each Filter's ``dimension`` must name one of the
    # measure refs in ``q.measures``; we resolve to the output alias.
    if q.having:
        having_terms: list[str] = []
        for hf in q.having:
            if hf.dimension not in q.measures:
                raise FederationError(
                    f"HAVING dimension {hf.dimension!r} in raw_rows mode "
                    "must reference one of the query's measures by "
                    f"qualified name. Known measures: {q.measures}.",
                    reason="having_unknown_measure",
                )
            alias = hf.dimension.rsplit(".", 1)[1]
            having_terms.append(_emit_having_term(alias, hf))
        sql += f" HAVING {' AND '.join(having_terms)}"

    if q.order:
        order_terms: list[str] = []
        for ref, direction in q.order:
            alias = ref.rsplit(".", 1)[-1] if "." in ref else ref
            if alias not in output_columns:
                raise FederationError(
                    f"ORDER BY {ref!r}: not in the federated output columns {output_columns}.",
                    reason="order_by_unknown_column",
                )
            order_terms.append(f"{_quote_ident(alias)} {'DESC' if direction == 'desc' else 'ASC'}")
        sql += f" ORDER BY {', '.join(order_terms)}"

    if q.limit is not None:
        sql += f" LIMIT {int(q.limit)}"
    if q.offset is not None and q.offset > 0:
        sql += f" OFFSET {int(q.offset)}"

    return sql


def _raw_agg_expr(agg: str, col: str, frag_alias: str) -> str:
    """Render an aggregation expression over a fragment's raw column.

    ``col`` may be the empty string — only valid for ``agg == "count"``,
    meaning the measure was a ``COUNT(*)`` with no underlying column;
    the merge emits ``COUNT(*)`` directly.
    """
    if not col:
        if agg != "count":
            raise FederationError(
                f"Internal: empty raw column for agg={agg!r}; only COUNT(*) "
                "measures may omit a raw column.",
                reason=f"unimplemented_agg_in_raw_rows:{agg}",
            )
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
    raise FederationError(
        f"agg={agg!r} is not implemented in raw_rows merge.",
        reason=f"unimplemented_agg_in_raw_rows:{agg}",
    )


def _emit_having_term(alias: str, f: Filter) -> str:
    """Render a single HAVING predicate in DuckDB dialect.

    Used by raw_rows mode; the alias references a measure output column
    on the merge SELECT. Values inline because the merge SQL is a
    single string with no separate params envelope — and the values
    come from the user-supplied SemanticQuery's HAVING filters which
    were validated upstream."""
    col = _quote_ident(alias)
    op = f.op
    vals = list(f.values)
    if op == "gt":
        return f"{col} > {_lit(vals[0])}"
    if op == "gte":
        return f"{col} >= {_lit(vals[0])}"
    if op == "lt":
        return f"{col} < {_lit(vals[0])}"
    if op == "lte":
        return f"{col} <= {_lit(vals[0])}"
    if op == "eq":
        return f"{col} = {_lit(vals[0])}"
    if op == "neq":
        return f"{col} <> {_lit(vals[0])}"
    raise FederationError(
        f"HAVING operator {op!r} is not yet supported in raw_rows federation mode.",
        reason=f"unsupported_having_op:{op}",
    )


def _lit(v: object) -> str:
    """Emit a literal in DuckDB dialect — numeric or single-quoted string."""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


def _compile_raw_rows(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    grouped: dict[Backend, list[Cube]],
    backend_order: list[Backend],
    primary_partition: Backend,
    bridges: list[_Bridge],
    *,
    context: dict[str, str] | None,
    group_by_alias: bool,
    having_alias: bool,
    strategies: dict[Backend, BackendStrategy] | None,
    viewer: AuthContext | None,
    policy: PolicyFn | None,
    scope_fns: dict[str, ScopeFn] | None,
) -> FederatedPlan:
    """Multi-backend raw-row compile path.

    Each partition emits an ungrouped fragment (raw rows + bridge keys
    + synthetic raw-measure dims); the merge step joins them and
    applies the final aggregation (including non-distributive aggs and
    ``having``)."""
    partitions: list[_RawRowPartitionPlan] = []
    for backend in backend_order:
        plan = _build_partition_sub_query_raw_rows(
            q, catalog, list(grouped[backend]), primary_partition, bridges
        )
        partitions.append(plan)

    # CNF-split the where-tree (if any): single-partition clauses
    # AND-compose into their partition's fragment; cross-partition
    # clauses emit at merge-time.
    cross_partition_clauses: _Cnf = []
    if q.where is not None:
        partitions, cross_partition_clauses = _route_where_tree(q.where, partitions)

    fragments: list[Compiled] = []
    for plan in partitions:
        scoped = _scoped_catalog(plan.cubes)
        c = compile_query(
            plan.sub_query,
            scoped,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            strategies=strategies,
            views=None,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
            _allow_unbounded_ungrouped=True,
        )
        fragments.append(c)

    output_columns: list[str] = []
    output_column_meta: list[ColumnMeta] = []
    cube_to_idx: dict[str, int] = {}
    for i, p in enumerate(partitions):
        for cube in p.cubes:
            cube_to_idx[cube.name] = i

    def _meta_for_dim(ref: str) -> ColumnMeta:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
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

    for ref in q.dimensions:
        output_columns.append(ref.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_dim(ref))

    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx = cube_to_idx[td_cube.name]
        plan = partitions[idx]
        td_name = q.time_dimension.dimension.rsplit(".", 1)[1]
        td_col = f"{td_name}_{plan.time_grain}" if plan.time_grain is not None else td_name
        output_columns.append(td_col)
        output_column_meta.append(ColumnMeta(name=td_col, kind="time", display_name=td_col))

    for ref in q.measures:
        output_columns.append(ref.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_measure(ref))

    merge_sql = _emit_merge_sql_raw_rows(
        q,
        catalog,
        primary_partition,
        partitions,
        bridges,
        output_columns,
        cross_partition_clauses=cross_partition_clauses,
    )
    return FederatedPlan(
        fragments=fragments,
        merge=MergePlan(sql=merge_sql),
        columns=output_columns,
        column_meta=output_column_meta,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compile_federated_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    strategies: dict[Backend, BackendStrategy] | None = None,
    views: dict[str, View] | None = None,
    viewer: AuthContext | None = None,
    policy: PolicyFn | None = None,
    scope_fns: dict[str, ScopeFn] | None = None,
    mode: FederationMode = "distributive",
) -> FederatedPlan:
    """Compile a query whose touched cubes span multiple backends.

    Returns a :class:`FederatedPlan` with one :class:`Compiled`
    fragment per backend and a DuckDB merge SQL that joins them.

    Single-backend queries succeed too — the returned plan has a single
    fragment and a trivial merge — but :func:`compile_query` is the
    simpler API for that case.

    See the module docstring for the v1 restriction set. Refusals are
    :class:`FederationError`\\s with a structured ``reason`` attribute
    for programmatic handling.
    """
    if q.compare is not None:
        raise FederationError(
            "Federated compare-mode is not supported in v1. Run the "
            "current and prior windows as separate federated queries "
            "and diff client-side, or use the in-process executor.",
            reason="compare_in_federated",
        )
    if q.where is not None and mode == "distributive":
        # Raw-row mode splits the where-tree into CNF, routes
        # single-partition clauses to their owning fragment, and lifts
        # cross-partition clauses (where two leaves on different
        # partitions are OR-joined) into the merge SQL. Distributive
        # mode's compile path doesn't ship the routing logic — the
        # path's primary-fragment-does-aggregation shape would also
        # need work to handle predicates that move rows in or out of
        # the count.
        raise FederationError(
            "Federated queries cannot use ``where`` (the boolean "
            "predicate tree) in distributive mode — pass "
            "``mode='raw_rows'`` to enable CNF-based partition "
            "routing, or use flat ``filters`` instead.",
            reason="where_tree_in_distributive_federated",
        )
    if q.segments and mode == "distributive":
        # Raw-row mode routes single-partition segments to their owning
        # fragment (the segment's qualified ``cube.segment`` names
        # exactly one cube → exactly one partition). Distributive
        # mode's partition builder doesn't ship segment routing yet —
        # follow-up if anyone needs it.
        raise FederationError(
            "Federated queries cannot reference segments in distributive "
            "mode — pass ``mode='raw_rows'`` (which routes single-partition "
            "segments to their owning fragment) or inline the segment's "
            "predicate as a flat Filter on the same cube.",
            reason="segments_in_distributive_federated",
        )
    if q.having and mode == "distributive":
        raise FederationError(
            "Federated queries cannot use HAVING in distributive mode "
            "— the having predicate would have to run against "
            "re-aggregated measures at the merge step, which raw_rows "
            'mode now lifts. Pass ``mode="raw_rows"`` to '
            "compile_federated_query.",
            reason="having_in_distributive_federated",
        )

    touched = _touched(q, catalog)
    if not touched:
        raise FederationError(
            "Query is empty — no cubes touched. Federation needs at least one cube reference.",
            reason="empty_query",
        )
    backends_seen = {c.backend for c in touched}

    # Degenerate single-backend case: delegate to compile_query and wrap.
    if len(backends_seen) == 1:
        c = compile_query(
            q,
            catalog,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            strategies=strategies,
            views=views,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )
        return FederatedPlan(
            fragments=[c],
            merge=MergePlan(sql="SELECT * FROM frag_0"),
            columns=c.columns,
            column_meta=c.column_meta,
        )

    # Pick the primary partition — the backend that owns any measure.
    # If there are no measures, the primary is the backend of the first
    # touched cube (deterministic; doesn't affect correctness).
    primary_partition: Backend
    if q.measures:
        m_owner = _resolve_field_to_cube(q.measures[0], catalog)
        primary_partition = m_owner.backend
        # Refuse if any other measure resolves to a different backend.
        for ref in q.measures[1:]:
            owner = _resolve_field_to_cube(ref, catalog)
            if owner.backend is not primary_partition:
                raise FederationError(
                    f"Measures span multiple backends: {ref!r} resolves "
                    f"to {owner.backend.value!r} but the primary "
                    f"partition (from the first measure) is "
                    f"{primary_partition.value!r}. v1 federated queries "
                    f"require all measures to live on one backend.",
                    reason="measures_span_backends",
                )
    else:
        primary_partition = touched[0].backend

    # Find every cross-backend bridge.
    bridges = _find_bridges(touched, catalog)
    if not bridges:
        raise FederationError(
            "Touched cubes span multiple backends but the catalogue "
            "declares no cross-backend Join between them. Add a "
            "many_to_one Join (or a foreign_key dimension that "
            "auto-derives one).",
            reason="no_cross_backend_join",
        )

    # Group cubes by backend (preserving first-mention order).
    grouped: dict[Backend, list[Cube]] = {}
    for cube in touched:
        grouped.setdefault(cube.backend, []).append(cube)

    # Build per-partition sub-queries. We sort so the primary partition
    # comes first — the merge SQL FROMs the primary and LEFT JOINs the
    # satellites.
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
            strategies=strategies,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )

    partitions: list[_PartitionPlan] = []
    for backend in backend_order:
        plan = _build_partition_sub_query(
            q, catalog, list(grouped[backend]), primary_partition, bridges
        )
        partitions.append(plan)

    # Compile each fragment.
    fragments: list[Compiled] = []
    for plan in partitions:
        scoped = _scoped_catalog(plan.cubes)
        c = compile_query(
            plan.sub_query,
            scoped,
            context=context,
            group_by_alias=group_by_alias,
            having_alias=having_alias,
            strategies=strategies,
            views=None,
            viewer=viewer,
            policy=policy,
            scope_fns=scope_fns,
        )
        fragments.append(c)

    # Compose output columns + column_meta.
    output_columns: list[str] = []
    output_column_meta: list[ColumnMeta] = []
    cube_to_idx: dict[str, int] = {}
    for i, p in enumerate(partitions):
        for cube in p.cubes:
            cube_to_idx[cube.name] = i

    def _meta_for_dim(ref: str) -> ColumnMeta:
        owner = _resolve_field_to_cube(ref, catalog)
        idx = cube_to_idx[owner.name]
        plan = partitions[idx]
        col = plan.dim_columns[ref]
        for cm in fragments[idx].column_meta:
            if cm.name == col:
                # Re-name to the final output column name (no cube prefix).
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
        # Pull display info from the original Measure declaration.
        m = next(x for x in owner.measures if x.name == m_name)
        return ColumnMeta(
            name=m_name,
            kind="measure",
            display_name=m.display_name or m_name.replace("_", " ").title(),
            unit=m.unit,
            display_unit=m.display_unit,
            format=m.format,
        )

    for ref in q.dimensions:
        output_columns.append(ref.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_dim(ref))

    if q.time_dimension is not None:
        td_cube = _resolve_field_to_cube(q.time_dimension.dimension, catalog)
        idx = cube_to_idx[td_cube.name]
        td_col = partitions[idx].dim_columns[q.time_dimension.dimension]
        output_columns.append(td_col)
        output_column_meta.append(_meta_for_dim(q.time_dimension.dimension))

    for ref in q.measures:
        output_columns.append(ref.rsplit(".", 1)[1])
        output_column_meta.append(_meta_for_measure(ref))

    merge_sql = _emit_merge_sql(q, catalog, primary_partition, partitions, bridges, output_columns)
    return FederatedPlan(
        fragments=fragments,
        merge=MergePlan(sql=merge_sql),
        columns=output_columns,
        column_meta=output_column_meta,
    )


__all__ = [
    "FederatedPlan",
    "FederationError",
    "FederationMode",
    "MergePlan",
    "compile_federated_query",
]
