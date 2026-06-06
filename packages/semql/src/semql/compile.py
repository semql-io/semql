"""Pure compiler from `SemanticQuery` to backend SQL.

The compiler has no I/O. It reads the catalogue, resolves identifiers,
emits a parameterised SQL string + params dict + output column list,
and raises ``CompileError`` (or a more specific leaf) on unknown
references, unreachable joins, or unsupported shapes.

The body composes a sqlglot ``exp.Select`` AST and renders it via the
target backend's dialect. Per-cube fragments (dim SQL, measure SQL,
``base_predicate``, ``Join.on``) are parsed by sqlglot under the
catalogue cube's declared backend; dialect-specific shapes
(``placeholder``, ``trunc``, ``contains``, ``emit_source``) come from
the ``BackendStrategy`` and slot into the AST as nodes.

Scope (Phase 1):
- Single-backend queries (cross-backend rejected).
- ``base_predicate`` lifted to the outer WHERE.
- Parameterised ``Filter.values`` (positional names ``p0``, ``p1``, …).
- Time-window pre-resolved as ISO strings, exclusive end.
- Granularity truncation for time dimensions.
- ``ungrouped=True`` row listings with a hard 1000-row cap.
- ``having`` on measure aliases (bare or qualified).
- ``context`` dict substitutes ``{key}`` placeholders in table/SQL strings.

Out of scope (deferred):
- ``compare`` CTE shell (current/prior FULL OUTER JOIN).
- Cross-backend merge.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from semql._resolve import resolve_field as _resolve_field_raw
from semql.backend import BackendStrategy, strategy_for

# Importing from semql.dialect also registers the ClickHouse placeholder
# override against the ``clickhouse`` dialect name (side effect on import).
from semql.dialect import dialect_for
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
    """Resolve ``{key}`` placeholders in a SQL fragment.

    Priority: cube aliases (both alias and cube name map to alias), then
    caller-supplied ``context``. ClickHouse typed param placeholders like
    ``{p0:String}`` are left untouched — the inner regex requires a plain
    identifier (no ``:Type`` suffix). Unknown placeholders raise
    ``PlaceholderError``."""
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
# Aggregation — measure agg → sqlglot node
# ---------------------------------------------------------------------------


def _agg_node(m: Measure, expr_node: exp.Expression) -> exp.Expression:
    """Wrap an inner expression in the measure's aggregate function."""
    agg = m.agg
    if agg == "sum":
        return exp.Sum(this=expr_node)
    if agg == "count":
        return exp.Count(this=expr_node)
    if agg == "count_distinct":
        return exp.Count(this=expr_node, distinct=True)
    if agg == "avg":
        return exp.Avg(this=expr_node)
    if agg == "min":
        return exp.Min(this=expr_node)
    if agg == "max":
        return exp.Max(this=expr_node)
    raise CompileError(f"Unsupported aggregate: {agg!r}")  # pragma: no cover


_FILTER_BINOPS: dict[str, type[exp.Expression]] = {
    "eq": exp.EQ,
    "neq": exp.NEQ,
    "gt": exp.GT,
    "lt": exp.LT,
    "gte": exp.GTE,
    "lte": exp.LTE,
}


def _filter_node(
    f: Filter,
    field: exp.Expression,
    field_type: str,
    strategy: BackendStrategy,
    bind: Callable[[Any, str], exp.Placeholder],
) -> exp.Expression:
    """Build a predicate node for a Filter.

    Dialect-specific shapes (``contains``) go through the strategy. Type
    checks raise ``FilterTypeError`` with structured attrs."""
    op = f.op
    if op == "is_null":
        return exp.Is(this=field, expression=exp.Null())
    if op == "not_null":
        return exp.Not(this=exp.Is(this=field, expression=exp.Null()))

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
        return strategy.emit_contains(field, str(f.values[0]), bind)

    placeholders: list[exp.Placeholder] = [bind(v, field_type) for v in f.values]

    if op in _FILTER_BINOPS:
        return _FILTER_BINOPS[op](this=field, expression=placeholders[0])

    in_args: list[exp.Expression] = list(placeholders)
    if op == "in":
        return exp.In(this=field, expressions=in_args)
    if op == "not_in":
        return exp.Not(this=exp.In(this=field, expressions=in_args))

    raise CompileError(f"Unsupported filter op: {op!r}")  # pragma: no cover


def _parse_fragment(sql: str, dialect: str) -> exp.Expression:
    """Parse a catalogue SQL fragment into a sqlglot AST node.

    Used for dim/measure/time-dimension expressions, ``Join.on``, and
    ``base_predicate``. Any parse failure surfaces as a ``CompileError``
    naming the offending fragment so the catalogue author can fix it."""
    try:
        return sqlglot.parse_one(sql, dialect=dialect)  # type: ignore[return-value]
    except ParseError as exc:
        raise CompileError(
            f"Could not parse catalogue SQL fragment {sql!r} under dialect {dialect!r}: {exc}"
        ) from exc


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

    ``catalog`` — dict of cube name → Cube (build from ``Catalog.as_dict()``).
    ``context`` — optional string substitutions applied to ``{key}``
        placeholders in table names and SQL expressions
        (e.g. ``{"schema": "mydb"}``).
    ``group_by_alias`` — when True (default), GROUP BY references the
        SELECT output alias. Set False to repeat the resolved expression.
    ``having_alias`` — when False (default), HAVING repeats the aggregate
        expression. Set True only when you control the backend.
    ``strategies`` — optional per-backend strategy overrides. Pass a
        ``RecordingStrategy`` for tests; pass a custom Snowflake / BigQuery
        adapter for out-of-tree backends without touching the global registry.
    """
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

    # 4. Aliases, strategy, bind closure.
    cube_aliases: dict[str, str] = {c.name: c.alias for c in cubes_in_from}
    strategy = strategy_for(backend, strategies)
    dialect = dialect_for(backend)
    params: dict[str, Any] = {}

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        name = f"p{len(params)}"
        params[name] = value
        return strategy.placeholder(name, dim_type)

    def resolve_in_ctx(sql: str) -> str:
        return _resolve_sql(sql, cube_aliases, ctx)

    def parse(sql: str) -> exp.Expression:
        return _parse_fragment(resolve_in_ctx(sql), dialect)

    # 5. Compose the Select node — FROM + JOINs.
    select_node = exp.Select()
    select_node = select_node.from_(strategy.emit_source(root, catalog, resolve_in_ctx))
    for _, tgt, j in join_edges:
        tgt_strategy = strategy_for(tgt.backend, strategies)
        target_source = tgt_strategy.emit_source(tgt, catalog, resolve_in_ctx)
        select_node = select_node.join(target_source, on=parse(j.on), join_type="left")

    # 6. SELECT projections (dimensions, time-breakdown, measures).
    columns: list[str] = []
    proposed_names: list[str] = []
    proposed_names.extend(d.name for _, d in dim_fields)
    proposed_names.extend(m.name for _, m in measure_fields)
    name_counts = Counter(proposed_names)
    collisions = {n for n, c in name_counts.items() if c > 1}

    def col_name_for(cube: Cube, field_name: str) -> str:
        return f"{cube.name}_{field_name}" if field_name in collisions else field_name

    for cube, dim in dim_fields:
        col_name = col_name_for(cube, dim.name)
        select_node = select_node.select(exp.alias_(parse(dim.sql), col_name))
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
        trunc_node = strategy.trunc(granularity, parse(time_dim.sql))
        col_name = f"{time_dim.name}_{granularity}"
        select_node = select_node.select(exp.alias_(trunc_node, col_name))
        columns.append(col_name)

    measure_alias_map: dict[str, exp.Expression] = {}
    for cube, m in measure_fields:
        if m.sql == "*":
            inner: exp.Expression = exp.Star()
        else:
            inner = parse(m.sql)
        agg_node = _agg_node(m, inner)
        col_name = col_name_for(cube, m.name)
        select_node = select_node.select(exp.alias_(agg_node, col_name))
        columns.append(col_name)
        measure_alias_map[m.name] = agg_node
        if col_name != m.name:
            measure_alias_map[col_name] = agg_node

    if not columns:
        raise CompileError("Compiled query has no SELECT projections.")

    if not q.ungrouped and not measure_fields:
        select_node.set("distinct", exp.Distinct())

    # 7. WHERE: base_predicates + filters + time window.
    where_terms: list[exp.Expression] = []
    for cube in cubes_in_from:
        if cube.base_predicate and cube.backend is not Backend.META:
            where_terms.append(parse(cube.base_predicate))

    for f, _cube, fld in filter_resolutions:
        fld_node = parse(fld.sql)
        if isinstance(fld, Dimension):
            fld_type = fld.type
        elif isinstance(fld, TimeDimension):
            fld_type = "time"
        else:
            fld_type = "string"
        where_terms.append(_filter_node(f, fld_node, fld_type, strategy, bind))

    if q.time_dimension is not None and time_dim is not None:
        td_node_start = parse(time_dim.sql)
        td_node_end = parse(time_dim.sql)
        where_terms.append(
            exp.GTE(this=td_node_start, expression=bind(q.time_dimension.range[0], "time"))
        )
        where_terms.append(
            exp.LT(this=td_node_end, expression=bind(q.time_dimension.range[1], "time"))
        )

    if where_terms:
        select_node = select_node.where(*where_terms)

    # 8. GROUP BY.
    if not q.ungrouped and measure_fields:
        for i, (_cube, dim) in enumerate(dim_fields):
            if group_by_alias:
                select_node = select_node.group_by(exp.column(columns[i]))
            else:
                select_node = select_node.group_by(parse(dim.sql))
        if has_time_breakdown:
            assert time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            if group_by_alias:
                select_node = select_node.group_by(exp.column(f"{time_dim.name}_{granularity}"))
            else:
                select_node = select_node.group_by(strategy.trunc(granularity, parse(time_dim.sql)))

    # 9. HAVING — bare or qualified measure reference.
    for hf in q.having:
        lookup_name = hf.dimension
        if lookup_name not in measure_alias_map and "." in lookup_name:
            lookup_name = lookup_name.rsplit(".", 1)[-1]
        if lookup_name not in measure_alias_map:
            raise CompileError(
                f"HAVING references {hf.dimension!r}, which is not a measure in this query."
            )
        target_node: exp.Expression
        if having_alias:
            target_node = exp.column(lookup_name)
        else:
            target_node = measure_alias_map[lookup_name].copy()
        select_node = select_node.having(_filter_node(hf, target_node, "number", strategy, bind))

    # 10. ORDER BY.
    for ref, direction in q.order:
        if ref in measure_alias_map or ref in columns:
            order_target: exp.Expression = exp.column(ref)
        else:
            try:
                _, fld = _resolve_field(ref, catalog)
            except CompileError as exc:
                raise CompileError(
                    f"ORDER BY {ref!r}: must reference an output column or "
                    f"a known cube.field. ({exc})"
                ) from exc
            order_target = parse(fld.sql)
        select_node = select_node.order_by(
            exp.Ordered(this=order_target, desc=(direction == "desc"))
        )

    # 11. LIMIT / OFFSET.
    if q.limit is not None:
        select_node = select_node.limit(int(q.limit))
    if q.offset is not None and q.offset > 0:
        select_node = select_node.offset(int(q.offset))

    sql = select_node.sql(dialect=dialect, pretty=False, normalize_functions=False)
    return Compiled(backend=backend, sql=sql, params=params, columns=columns)


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
