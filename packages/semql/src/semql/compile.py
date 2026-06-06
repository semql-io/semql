"""Pure compiler from `SemanticQuery` to backend SQL.

The compiler has no I/O. It reads the catalogue, resolves identifiers,
emits a parameterised SQL string + params dict + output column list,
and raises `CompileError` with a precise message on unknown references,
unreachable joins, or unsupported shapes.

Scope (Phase 1):
- Single-backend queries (cross-backend rejected).
- `base_predicate` lifted to the outer WHERE.
- Parameterised `Filter.values` (positional names `p0`, `p1`, ...);
  Postgres renders `%(name)s`, ClickHouse renders `{name:Type}`.
- Time-window pre-resolved as ISO strings, exclusive end.
- Granularity truncation for time dimensions.
- `ungrouped=True` row listings with a hard 1000-row cap.
- `having` on measure aliases.
- `context` dict substitutes `{key}` placeholders in table/SQL strings
  (e.g. `{"schema": "mydb"}` resolves `{schema}.orders`).

Out of scope (deferred):
- `compare` CTE shell (current/prior FULL OUTER JOIN).
- Cross-backend merge.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from semql._resolve import resolve_field as _resolve_field_raw
from semql.backend import BackendStrategy, strategy_for
from semql.errors import (
    CompileError,
    CrossBackendError,
    FilterTypeError,
    JoinPathError,
    PhaseDeferredError,
    PlaceholderError,
    ResolveError,
    UnknownIdentifierError,
)
from semql.model import Backend, Cube, Dimension, Join, Measure, TimeDimension
from semql.spec import Filter, SemanticQuery

MAX_UNGROUPED_ROWS = 1000


@dataclass
class Compiled:
    backend: Backend
    sql: str
    params: dict[str, Any]
    columns: list[str]


def _resolve_field(
    qualified: str,
    catalog: dict[str, Cube],
) -> tuple[Cube, Measure | Dimension | TimeDimension]:
    try:
        return _resolve_field_raw(qualified, catalog)
    except CompileError:
        raise
    except ResolveError as exc:
        raise CompileError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Cube graph BFS — find a single join path from a root through `joins` edges.
# ---------------------------------------------------------------------------


def _find_join_path(
    root: str,
    target: str,
    catalog: dict[str, Cube],
) -> list[Join]:
    if root == target:
        return []
    visited: set[str] = {root}
    queue: list[tuple[str, list[Join]]] = [(root, [])]
    while queue:
        current, path = queue.pop(0)
        for j in catalog[current].joins:
            if j.to in visited:
                continue
            new_path = path + [j]
            if j.to == target:
                return new_path
            visited.add(j.to)
            queue.append((j.to, new_path))
    raise JoinPathError(
        f"No join path from cube {root!r} to {target!r}. "
        "Declare a Join in the catalogue or restructure the query.",
        root_cube=root,
        target_cube=target,
    )


# ---------------------------------------------------------------------------
# Placeholder substitution — `{key}` in SQL fragments resolved from
# `cube_aliases` (alias → alias, cube_name → alias) and `context`.
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _resolve_sql(
    sql: str,
    cube_aliases: dict[str, str],
    context: dict[str, str],
) -> str:
    """Resolve `{key}` placeholders in a SQL fragment.

    Priority: cube aliases (both alias and cube name map to alias), then
    caller-supplied `context`. ClickHouse typed param placeholders like
    `{p0:String}` are left untouched — the inner regex requires a plain
    identifier (no `:Type` suffix). Unknown placeholders raise CompileError."""
    lookup: dict[str, str] = dict(context)
    for cube_name, alias in cube_aliases.items():
        lookup[alias] = alias
        lookup[cube_name] = alias

    def _repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name not in lookup:
            raise PlaceholderError(
                f"Unknown placeholder {{{name}}} in catalogue SQL. Known: {sorted(lookup)}.",
                placeholder=name,
                known=sorted(lookup),
            )
        return lookup[name]

    return _PLACEHOLDER_RE.sub(_repl, sql)


# ---------------------------------------------------------------------------
# Aggregation rendering
# ---------------------------------------------------------------------------

_AGG_FN: dict[str, str] = {
    "sum": "SUM",
    "count": "COUNT",
    "count_distinct": "COUNT(DISTINCT {x})",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
}


def _render_agg(m: Measure, sql_expr: str) -> str:
    template = _AGG_FN[m.agg]
    if "{x}" in template:
        return template.replace("{x}", sql_expr)
    return f"{template}({sql_expr})"


# ---------------------------------------------------------------------------
# Filter rendering — operator shape only. The dialect-specific bits
# (placeholder syntax, `contains` substring shape) are delegated to the
# backend strategy.
# ---------------------------------------------------------------------------


def _render_filter(
    f: Filter,
    field_sql: str,
    field_type: str,
    strategy: BackendStrategy,
    bind: Callable[[Any, str], str],  # noqa: ANN401 — see compile_query's bind closure
) -> str:
    op = f.op
    if op == "is_null":
        return f"{field_sql} IS NULL"
    if op == "not_null":
        return f"{field_sql} IS NOT NULL"

    try:
        f.validate_for_type(field_type)
    except ValueError as exc:
        raise FilterTypeError(
            str(exc),
            dimension=f.dimension,
            op=f.op,
            value=f.values[0] if f.values else None,
        ) from exc

    if op == "contains":
        return strategy.emit_contains(field_sql, str(f.values[0]), bind)

    placeholders = [bind(v, field_type) for v in f.values]

    if op == "eq":
        return f"{field_sql} = {placeholders[0]}"
    if op == "neq":
        return f"{field_sql} <> {placeholders[0]}"
    if op == "gt":
        return f"{field_sql} > {placeholders[0]}"
    if op == "lt":
        return f"{field_sql} < {placeholders[0]}"
    if op == "gte":
        return f"{field_sql} >= {placeholders[0]}"
    if op == "lte":
        return f"{field_sql} <= {placeholders[0]}"
    if op == "in":
        return f"{field_sql} IN ({', '.join(placeholders)})"
    if op == "not_in":
        return f"{field_sql} NOT IN ({', '.join(placeholders)})"

    raise CompileError(f"Unsupported filter op: {op!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compile_query(
    q: SemanticQuery,
    catalog: dict[str, Cube],
    *,
    context: dict[str, str] | None = None,
    group_by_alias: bool = True,
    having_alias: bool = False,
    strategies: dict[Backend, BackendStrategy] | None = None,
) -> Compiled:
    """Compile a SemanticQuery to a Compiled bundle.

    `catalog` — dict of cube name → Cube (build from `Catalog.as_dict()`).
    `context` — optional string substitutions applied to `{key}` placeholders
        in table names and SQL expressions (e.g. `{"schema": "mydb"}`).
    `group_by_alias` — when True (default), GROUP BY references the SELECT
        output alias. Set False to repeat the resolved expression.
    `having_alias` — when False (default), HAVING repeats the aggregate
        expression. Set True only when you control the backend.
    `strategies` — optional per-backend strategy overrides. Pass a
        ``RecordingStrategy`` for tests; pass a custom Snowflake / BigQuery
        adapter for out-of-tree backends without touching the global registry."""
    ctx = context or {}

    if q.compare is not None:
        raise PhaseDeferredError(
            "compare windows are not yet supported by the compiler (Phase 2).",
            feature="compare",
        )

    if q.offset is not None and q.offset > 0 and q.limit is None:
        raise CompileError(
            "SemanticQuery has offset set without limit. "
            "OFFSET is only meaningful in combination with LIMIT."
        )

    if q.ungrouped and (q.limit is None or q.limit > MAX_UNGROUPED_ROWS):
        raise CompileError(
            f"Ungrouped query requires limit <= {MAX_UNGROUPED_ROWS}. Got limit={q.limit}."
        )

    if not q.measures and not q.dimensions and q.time_dimension is None:
        raise CompileError(
            "SemanticQuery is empty — at least one measure, dimension, "
            "or time_dimension is required."
        )

    # 1. Resolve references; gather touched cubes.
    measure_fields: list[tuple[Cube, Measure]] = []
    for ref in q.measures:
        cube, fld = _resolve_field(ref, catalog)
        if not isinstance(fld, Measure):
            raise CompileError(f"{ref!r} is not a measure on cube {cube.name!r}.")
        measure_fields.append((cube, fld))

    dim_fields: list[tuple[Cube, Dimension]] = []
    for ref in q.dimensions:
        cube, fld = _resolve_field(ref, catalog)
        if not isinstance(fld, Dimension):
            raise CompileError(f"{ref!r} is not a dimension on cube {cube.name!r}.")
        dim_fields.append((cube, fld))

    time_cube: Cube | None = None
    time_dim: TimeDimension | None = None
    if q.time_dimension is not None:
        tcube, tfld = _resolve_field(q.time_dimension.dimension, catalog)
        if not isinstance(tfld, TimeDimension):
            raise CompileError(f"{q.time_dimension.dimension!r} is not a time dimension.")
        gran = q.time_dimension.granularity
        if gran is not None and gran not in tfld.granularities:
            raise CompileError(
                f"Granularity {gran!r} not supported on {q.time_dimension.dimension!r}. "
                f"Allowed: {tfld.granularities}."
            )
        time_cube, time_dim = tcube, tfld

    touched: list[Cube] = []
    for c, _ in [*measure_fields, *dim_fields]:
        if c not in touched:
            touched.append(c)
    if time_cube is not None and time_cube not in touched:
        touched.append(time_cube)

    filter_resolutions: list[tuple[Filter, Cube, Dimension | Measure | TimeDimension]] = []
    for f in q.filters:
        c, fld = _resolve_field(f.dimension, catalog)
        filter_resolutions.append((f, c, fld))
        if c not in touched:
            touched.append(c)

    if not touched:
        raise CompileError("Could not determine any cubes from the query.")

    # 1b. Required-filter enforcement.
    filter_dims = {f.dimension for f in q.filters}
    for cube in touched:
        for req in cube.required_filters:
            if f"{cube.name}.{req}" not in filter_dims:
                raise CompileError(
                    f"Cube {cube.name!r} requires a filter on {req!r}. "
                    f"Add Filter(dimension='{cube.name}.{req}', op=..., values=[...])."
                )

    # 2. Single-backend check.
    backends = {c.backend for c in touched}
    if len(backends) > 1:
        backend_names = sorted(b.value for b in backends)
        raise CrossBackendError(
            "Cross-backend queries are not yet supported (Phase 2). "
            f"Touched backends: {backend_names}.",
            backends=backend_names,
        )
    backend = next(iter(backends))

    # 3. Pick root cube; BFS over joins to reach the rest.
    root = touched[0]
    join_edges: list[tuple[Cube, Cube, Join]] = []
    cubes_in_from: list[Cube] = [root]
    for c in touched:
        if c is root:
            continue
        path = _find_join_path(root.name, c.name, catalog)
        cursor = root
        for j in path:
            tgt = catalog[j.to]
            if tgt not in cubes_in_from:
                join_edges.append((cursor, tgt, j))
                cubes_in_from.append(tgt)
            cursor = tgt

    # 4. Build alias map for placeholder substitution + bind/strategy closures.
    cube_aliases: dict[str, str] = {c.name: c.alias for c in cubes_in_from}
    strategy = strategy_for(backend, strategies)
    params: dict[str, Any] = {}

    def bind(value: Any, dim_type: str) -> str:  # noqa: ANN401 — Filter values are arbitrary literals
        name = f"p{len(params)}"
        params[name] = value
        return strategy.placeholder(name, dim_type)

    def resolve_in_ctx(sql: str) -> str:
        return _resolve_sql(sql, cube_aliases, ctx)

    # 5. Emit FROM + JOINs.
    from_clause = strategy.emit_source(root, catalog, resolve_in_ctx)
    join_clauses: list[str] = []
    for _, tgt, j in join_edges:
        tgt_strategy = strategy_for(tgt.backend, strategies)
        target_source = tgt_strategy.emit_source(tgt, catalog, resolve_in_ctx)
        on_sql = _resolve_sql(j.on, cube_aliases, ctx)
        join_clauses.append(f"LEFT JOIN {target_source} ON {on_sql}")

    # 6. Compose SELECT projections.
    select_items: list[str] = []
    columns: list[str] = []

    proposed_names: list[str] = []
    proposed_names.extend(d.name for _, d in dim_fields)
    proposed_names.extend(m.name for _, m in measure_fields)
    name_counts = Counter(proposed_names)
    collisions = {n for n, c in name_counts.items() if c > 1}

    def _col(cube: Cube, field_name: str) -> str:
        return f"{cube.name}_{field_name}" if field_name in collisions else field_name

    for cube, dim in dim_fields:
        sql_expr = _resolve_sql(dim.sql, cube_aliases, ctx)
        col_name = _col(cube, dim.name)
        select_items.append(f"{sql_expr} AS {col_name}")
        columns.append(col_name)

    has_time_breakdown = (
        time_cube is not None
        and time_dim is not None
        and q.time_dimension is not None
        and q.time_dimension.granularity is not None
    )
    if has_time_breakdown:
        assert time_dim is not None and q.time_dimension is not None
        granularity = q.time_dimension.granularity
        assert granularity is not None
        sql_expr = _resolve_sql(time_dim.sql, cube_aliases, ctx)
        sql_expr = strategy.trunc(granularity, sql_expr)
        col_name = f"{time_dim.name}_{granularity}"
        select_items.append(f"{sql_expr} AS {col_name}")
        columns.append(col_name)

    measure_alias_map: dict[str, str] = {}
    for cube, m in measure_fields:
        sql_expr = _resolve_sql(m.sql, cube_aliases, ctx)
        agg_sql = _render_agg(m, sql_expr)
        col_name = _col(cube, m.name)
        select_items.append(f"{agg_sql} AS {col_name}")
        columns.append(col_name)
        measure_alias_map[m.name] = agg_sql
        if col_name != m.name:
            measure_alias_map[col_name] = agg_sql

    if not select_items:
        raise CompileError("Compiled query has no SELECT projections.")

    # 7. WHERE: filters + time window + base_predicates.
    where_terms: list[str] = []
    for f, _cube, fld in filter_resolutions:
        fld_sql = _resolve_sql(fld.sql, cube_aliases, ctx)
        if isinstance(fld, Dimension):
            fld_type = fld.type
        elif isinstance(fld, TimeDimension):
            fld_type = "time"
        else:
            fld_type = "string"
        where_terms.append(_render_filter(f, fld_sql, fld_type, strategy, bind))

    if q.time_dimension is not None and time_dim is not None:
        td_sql = _resolve_sql(time_dim.sql, cube_aliases, ctx)
        where_terms.append(
            f"{td_sql} >= {bind(q.time_dimension.range[0], 'time')} "
            f"AND {td_sql} < {bind(q.time_dimension.range[1], 'time')}"
        )

    for cube in cubes_in_from:
        if cube.base_predicate and cube.backend is not Backend.META:
            where_terms.insert(0, _resolve_sql(cube.base_predicate, cube_aliases, ctx))

    # 8. GROUP BY.
    group_by_items: list[str] = []
    if not q.ungrouped and measure_fields:
        for i, (_cube, dim) in enumerate(dim_fields):
            if group_by_alias:
                group_by_items.append(columns[i])
            else:
                group_by_items.append(_resolve_sql(dim.sql, cube_aliases, ctx))
        if has_time_breakdown:
            assert time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            if group_by_alias:
                group_by_items.append(f"{time_dim.name}_{granularity}")
            else:
                sql_expr = _resolve_sql(time_dim.sql, cube_aliases, ctx)
                sql_expr = strategy.trunc(granularity, sql_expr)
                group_by_items.append(sql_expr)

    select_keyword = "SELECT"
    if not q.ungrouped and not measure_fields:
        select_keyword = "SELECT DISTINCT"

    # 9. HAVING. Accept either bare (`revenue`) or qualified (`orders.revenue`)
    # measure references; the qualified form is split on '.' and the short
    # name looked up in the alias map.
    having_terms: list[str] = []
    for hf in q.having:
        lookup_name = hf.dimension
        if lookup_name not in measure_alias_map and "." in lookup_name:
            lookup_name = lookup_name.rsplit(".", 1)[-1]
        if lookup_name in measure_alias_map:
            target = lookup_name if having_alias else measure_alias_map[lookup_name]
            having_terms.append(_render_filter(hf, target, "number", strategy, bind))
        else:
            raise CompileError(
                f"HAVING references {hf.dimension!r}, which is not a measure in this query."
            )

    # 10. ORDER BY.
    order_items: list[str] = []
    for ref, direction in q.order:
        if ref in measure_alias_map or ref in columns:
            order_items.append(f"{ref} {direction.upper()}")
            continue
        try:
            cube, fld = _resolve_field(ref, catalog)
        except CompileError as exc:
            raise CompileError(
                f"ORDER BY {ref!r}: must reference an output column or a known cube.field. ({exc})"
            ) from exc
        order_items.append(f"{_resolve_sql(fld.sql, cube_aliases, ctx)} {direction.upper()}")

    # 11. Assemble.
    parts: list[str] = [
        f"{select_keyword} {', '.join(select_items)}",
        f"FROM {from_clause}",
    ]
    parts.extend(join_clauses)
    if where_terms:
        parts.append("WHERE " + " AND ".join(where_terms))
    if group_by_items:
        parts.append("GROUP BY " + ", ".join(group_by_items))
    if having_terms:
        parts.append("HAVING " + " AND ".join(having_terms))
    if order_items:
        parts.append("ORDER BY " + ", ".join(order_items))
    if q.limit is not None:
        parts.append(f"LIMIT {int(q.limit)}")
    if q.offset is not None and q.offset > 0:
        parts.append(f"OFFSET {int(q.offset)}")

    return Compiled(backend=backend, sql="\n".join(parts), params=params, columns=columns)


__all__ = [
    "Compiled",
    "CompileError",
    "CrossBackendError",
    "FilterTypeError",
    "JoinPathError",
    "MAX_UNGROUPED_ROWS",
    "PhaseDeferredError",
    "PlaceholderError",
    "UnknownIdentifierError",
    "compile_query",
]
