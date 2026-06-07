# pyright: reportPrivateImportUsage=false
# sqlglot's AST types (Expression, Placeholder, ...) are imported via
# ``from sqlglot import exp``; they aren't in ``sqlglot.expressions.__all__``
# but are public by convention and by sqlglot's type stubs.
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
from datetime import datetime
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
from semql.model import Backend, Cube, Dimension, Join, Measure, Segment, TimeDimension
from semql.spec import BoolExpr, Filter, SemanticQuery

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
# ``{ctx.X}`` — caller-context placeholders inside ``security_sql``.
# Substituted with a bound parameter so the value never appears as a
# SQL literal.
_CTX_PLACEHOLDER_RE = re.compile(r"\{ctx\.([a-z_][a-z0-9_]*)\}")


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
        # sqlglot's exp.Count(..., distinct=True) is ignored on emit;
        # COUNT(DISTINCT x) requires wrapping the arg in exp.Distinct.
        return exp.Count(this=exp.Distinct(expressions=[expr_node]))
    if agg == "avg":
        return exp.Avg(this=expr_node)
    if agg == "min":
        return exp.Min(this=expr_node)
    if agg == "max":
        return exp.Max(this=expr_node)
    if agg == "ratio":
        # Ratio measures are composed in build_inner — _agg_node isn't
        # the right seam (it sees one expression, ratios need two).
        raise CompileError(  # pragma: no cover — caller routes around this.
            f"Measure {m.name!r}: ratio aggregation is computed in build_inner, not _agg_node."
        )
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


def _walk_where_leaves(expr: BoolExpr | Filter) -> list[Filter]:
    """Return all ``Filter`` leaves from a where tree, depth-first."""
    if isinstance(expr, Filter):
        return [expr]
    leaves: list[Filter] = []
    for child in expr.children:
        leaves.extend(_walk_where_leaves(child))
    return leaves


def _compile_where_tree(
    expr: BoolExpr | Filter,
    leaf_to_node: Callable[[Filter], exp.Expression],
) -> exp.Expression:
    """Build a sqlglot predicate node from a where tree.

    ``leaf_to_node`` resolves each Filter leaf to its compiled AST
    fragment (the caller owns the bind closure + type resolution)."""
    if isinstance(expr, Filter):
        return leaf_to_node(expr)
    children_nodes = [_compile_where_tree(c, leaf_to_node) for c in expr.children]
    if expr.op == "not":
        return exp.Not(this=children_nodes[0])
    combiner: type[exp.Expression] = exp.And if expr.op == "and" else exp.Or
    result = children_nodes[0]
    for n in children_nodes[1:]:
        result = combiner(this=result, expression=n)
    return result


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
        if q.time_dimension is None:
            raise CompileError(
                "compare requires a time_dimension on the query — the "
                "current and prior windows are derived from it."
            )
        if not q.measures:
            raise CompileError(
                "compare requires at least one measure — without a "
                "measure there is nothing to delta between current and prior."
            )
        if q.ungrouped:
            raise CompileError(
                "compare is incompatible with ungrouped=True — compare aggregates by definition."
            )
        if q.compare.mode == "explicit" and q.compare.range is None:
            raise CompileError(
                "compare(mode='explicit') requires a range. Pass "
                "CompareWindow(mode='explicit', range=(start, end))."
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

    # 1d. Resolve where-tree leaves up front so cubes referenced inside
    # OR / NOT branches are added to ``touched`` and the join graph
    # reaches them. The compiled predicate is built later in
    # ``build_inner`` (per CTE in compare mode).
    where_leaves: list[Filter] = _walk_where_leaves(q.where) if q.where is not None else []
    where_leaf_resolutions: dict[int, tuple[Cube, Dimension | Measure | TimeDimension]] = {}
    for leaf in where_leaves:
        c, fld = _resolve_field(leaf.dimension, catalog)
        where_leaf_resolutions[id(leaf)] = (c, fld)
        if c not in touched:
            touched.append(c)

    # 1c. Resolve segments — reusable named predicates declared on the
    # cube. ``cube.segment`` syntax matches the rest of the spec; the
    # predicate's SQL slots into the WHERE clause via ``build_inner``.
    segment_resolutions: list[tuple[Cube, Segment]] = []
    for seg_ref in q.segments:
        if "." not in seg_ref:
            raise CompileError(
                f"Segment reference {seg_ref!r} must be qualified as 'cube.segment'."
            )
        cube_name, seg_name = seg_ref.rsplit(".", 1)
        if cube_name not in catalog:
            raise CompileError(f"Segment reference {seg_ref!r}: unknown cube {cube_name!r}.")
        cube_obj = catalog[cube_name]
        match = next((s for s in cube_obj.segments if s.name == seg_name), None)
        if match is None:
            known = ", ".join(s.name for s in cube_obj.segments) or "(none)"
            raise CompileError(
                f"Segment reference {seg_ref!r}: cube {cube_name!r} has no segment "
                f"{seg_name!r}. Known segments: {known}."
            )
        segment_resolutions.append((cube_obj, match))
        if cube_obj not in touched:
            touched.append(cube_obj)

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
    # Memoize ``(value, dim_type)`` → placeholder so a filter value
    # referenced in both the current and prior CTEs binds once and the
    # same placeholder is reused — the database round-trip stays cheap
    # and the planner's intent (one filter, one bound value) is preserved.
    _binds: dict[tuple[Any, str], str] = {}

    def bind(value: Any, dim_type: str) -> exp.Placeholder:  # noqa: ANN401
        key = (value, dim_type)
        if key in _binds:
            return strategy.placeholder(_binds[key], dim_type)
        name = f"p{len(params)}"
        params[name] = value
        _binds[key] = name
        return strategy.placeholder(name, dim_type)

    def resolve_in_ctx(sql: str) -> str:
        return _resolve_sql(sql, cube_aliases, ctx)

    def parse(sql: str) -> exp.Expression:
        return _parse_fragment(resolve_in_ctx(sql), dialect)

    def _resolve_security_sql(cube: Cube, raw: str) -> str:
        """Substitute ``{alias}`` and ``{ctx.X}`` placeholders in a
        ``security_sql`` fragment, binding ``{ctx.X}`` values as
        parameters so they never appear as SQL literals."""
        resolved = resolve_in_ctx(raw)

        def _ctx_repl(m: re.Match[str]) -> str:
            key = "ctx." + m.group(1)
            if key not in ctx:
                raise CompileError(
                    f"Cube {cube.name!r} security_sql references "
                    f"{{{key}}} but no {key!r} value was provided in "
                    "compile context."
                )
            return bind(ctx[key], "string").sql(dialect=dialect, normalize_functions=False)

        return _CTX_PLACEHOLDER_RE.sub(_ctx_repl, resolved)

    def wrap_for_tenancy(cube: Cube, source: exp.Expression) -> exp.Expression:
        """Wrap a cube's FROM source in an isolation subquery.

        Two predicates may apply: tenancy (DISCRIMINATOR cubes) and
        ``security_sql`` (caller-attached RLS). Both live *inside* the
        alias the outer query sees so a malformed outer ``OR``
        predicate can't smuggle in rows the policy excludes. The
        predicates AND-compose. SCHEMA / NONE cubes with no
        ``security_sql`` pass through unchanged."""
        predicates: list[exp.Expression] = []

        if cube.tenancy == "discriminator":
            if "tenant" not in ctx:
                raise CompileError(
                    f"Cube {cube.name!r} declares tenancy='discriminator' but "
                    "no 'tenant' value was provided in compile context. "
                    "Pass context={'tenant': <tenant_id>, ...}."
                )
            assert cube.tenancy_column is not None
            predicates.append(
                exp.EQ(
                    this=exp.column(cube.tenancy_column, table=cube.alias),
                    expression=bind(ctx["tenant"], "string"),
                )
            )

        if cube.security_sql:
            resolved_sql = _resolve_security_sql(cube, cube.security_sql)
            predicates.append(_parse_fragment(resolved_sql, dialect))

        if not predicates:
            return source

        where_expr = predicates[0]
        for p in predicates[1:]:
            where_expr = exp.And(this=where_expr, expression=p)

        inner = exp.Select().select(exp.Star()).from_(source).where(where_expr)
        return exp.Subquery(
            this=inner,
            alias=exp.TableAlias(this=exp.to_identifier(cube.alias)),
        )

    # 5. Column name allocation (collision-prefix shared across compare CTEs).
    proposed_names: list[str] = []
    proposed_names.extend(d.name for _, d in dim_fields)
    proposed_names.extend(m.name for _, m in measure_fields)
    name_counts = Counter(proposed_names)
    collisions = {n for n, c in name_counts.items() if c > 1}

    def col_name_for(cube: Cube, field_name: str) -> str:
        return f"{cube.name}_{field_name}" if field_name in collisions else field_name

    dim_col_names: list[str] = [col_name_for(c, d.name) for c, d in dim_fields]
    measure_col_names: list[str] = [col_name_for(c, m.name) for c, m in measure_fields]

    has_time_breakdown = (
        time_cube is not None
        and time_dim is not None
        and q.time_dimension is not None
        and q.time_dimension.granularity is not None
    )
    time_col_name: str | None = None
    if has_time_breakdown:
        assert time_dim is not None and q.time_dimension is not None
        granularity = q.time_dimension.granularity
        assert granularity is not None
        time_col_name = f"{time_dim.name}_{granularity}"

    def build_measure_expr(cube_owner: Cube, m: Measure) -> exp.Expression:
        """Compose the SELECT expression for one measure.

        Routes ratio measures to a recursive division over their
        numerator / denominator (resolved by name on the same cube);
        everything else flows through ``_agg_node`` with an optional
        ``FILTER (WHERE ...)`` wrapper for filtered measures.

        Lifted outside ``build_inner`` so the HAVING / ORDER-BY alias
        map below can reuse the same composition path."""
        if m.agg == "ratio":
            assert m.numerator is not None and m.denominator is not None
            num_m = next(
                (x for x in cube_owner.measures if x.name == m.numerator),
                None,
            )
            if num_m is None:
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio numerator "
                    f"{m.numerator!r} is not a measure on cube {cube_owner.name!r}."
                )
            den_m = next(
                (x for x in cube_owner.measures if x.name == m.denominator),
                None,
            )
            if den_m is None:
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio denominator "
                    f"{m.denominator!r} is not a measure on cube {cube_owner.name!r}."
                )
            if num_m.agg == "ratio" or den_m.agg == "ratio":
                raise CompileError(
                    f"Measure {cube_owner.name}.{m.name}: ratio measures "
                    "cannot reference another ratio measure. Use leaf "
                    "(sum / count / avg / ...) measures as numerator "
                    "and denominator."
                )
            num_node = build_measure_expr(cube_owner, num_m)
            den_node = build_measure_expr(cube_owner, den_m)
            # Div(num, NULLIF(den, 0)) — NULLIF returns NULL on zero
            # denominator, propagating to a NULL result rather than
            # raising. Matches the pct_change convention in the
            # compare path.
            return exp.Div(
                this=num_node,
                expression=exp.Anonymous(
                    this="NULLIF",
                    expressions=[den_node, exp.Literal.number(0)],
                ),
            )

        if m.sql == "*":
            inner: exp.Expression = exp.Star()
        else:
            inner = parse(m.sql)
        agg = _agg_node(m, inner)
        if m.filter:
            # ``COUNT(*) FILTER (WHERE <pred>)`` — sqlglot renders
            # natively on PG / CH / DuckDB / BigQuery and transpiles
            # to ``COUNT(IFF(...))`` on Snowflake.
            agg = exp.Filter(
                this=agg,
                expression=exp.Where(this=parse(m.filter)),
            )
        return agg

    def build_inner(time_range: tuple[str, str]) -> exp.Select:
        """Build the inner Select used as a query body or compare CTE.

        Filter values bind via the shared closure so they're deduplicated
        across the current/prior CTEs in compare mode; only the time
        window bindings differ per call."""
        sel = exp.Select()
        sel = sel.from_(wrap_for_tenancy(root, strategy.emit_source(root, catalog, resolve_in_ctx)))
        for _, tgt, j in join_edges:
            tgt_strategy = strategy_for(tgt.backend, strategies)
            target_source = wrap_for_tenancy(
                tgt, tgt_strategy.emit_source(tgt, catalog, resolve_in_ctx)
            )
            sel = sel.join(target_source, on=parse(j.on), join_type="left")

        for (_cube, dim), col_name in zip(dim_fields, dim_col_names, strict=True):
            sel = sel.select(exp.alias_(parse(dim.sql), col_name))

        if has_time_breakdown:
            assert time_dim is not None and q.time_dimension is not None
            granularity = q.time_dimension.granularity
            assert granularity is not None
            assert time_col_name is not None
            trunc_node = strategy.trunc(granularity, parse(time_dim.sql))
            sel = sel.select(exp.alias_(trunc_node, time_col_name))

        for (cube_owner, m), col_name in zip(measure_fields, measure_col_names, strict=True):
            sel = sel.select(exp.alias_(build_measure_expr(cube_owner, m), col_name))

        if not q.ungrouped and not measure_fields:
            sel.set("distinct", exp.Distinct())

        where_terms: list[exp.Expression] = []
        for cube_w in cubes_in_from:
            if cube_w.base_predicate and cube_w.backend is not Backend.META:
                where_terms.append(parse(cube_w.base_predicate))

        for f, _cube, fld in filter_resolutions:
            fld_node = parse(fld.sql)
            if isinstance(fld, Dimension):
                fld_type = fld.type
            elif isinstance(fld, TimeDimension):
                fld_type = "time"
            else:
                fld_type = "string"
            where_terms.append(_filter_node(f, fld_node, fld_type, strategy, bind))

        # Segment predicates AND-compose with filters. Same resolution
        # path as ``Join.on`` / ``base_predicate`` SQL fragments.
        for _seg_cube, segment in segment_resolutions:
            where_terms.append(parse(segment.sql))

        # Boolean predicate tree — ANDs with the flat ``filters`` and
        # segments. Leaves resolve through the shared bind closure so
        # values get parameter-bound the same as flat-filter values.
        if q.where is not None:

            def _leaf_to_node(leaf: Filter) -> exp.Expression:
                _cube, fld = where_leaf_resolutions[id(leaf)]
                fld_node = parse(fld.sql)
                if isinstance(fld, Dimension):
                    fld_type = fld.type
                elif isinstance(fld, TimeDimension):
                    fld_type = "time"
                else:
                    fld_type = "string"
                return _filter_node(leaf, fld_node, fld_type, strategy, bind)

            where_terms.append(_compile_where_tree(q.where, _leaf_to_node))

        if time_dim is not None:
            where_terms.append(
                exp.GTE(this=parse(time_dim.sql), expression=bind(time_range[0], "time"))
            )
            where_terms.append(
                exp.LT(this=parse(time_dim.sql), expression=bind(time_range[1], "time"))
            )

        if where_terms:
            sel = sel.where(*where_terms)

        if not q.ungrouped and measure_fields:
            for i, (_cube, dim) in enumerate(dim_fields):
                if group_by_alias:
                    sel = sel.group_by(exp.column(dim_col_names[i]))
                else:
                    sel = sel.group_by(parse(dim.sql))
            if has_time_breakdown:
                assert time_dim is not None and q.time_dimension is not None
                granularity = q.time_dimension.granularity
                assert granularity is not None
                assert time_col_name is not None
                if group_by_alias:
                    sel = sel.group_by(exp.column(time_col_name))
                else:
                    sel = sel.group_by(strategy.trunc(granularity, parse(time_dim.sql)))

        return sel

    # 6. Compare branch — build two CTEs and an outer SELECT.
    if q.compare is not None:
        assert q.time_dimension is not None
        current_range = q.time_dimension.range
        if q.compare.mode == "previous_period":
            cs = datetime.fromisoformat(current_range[0])
            ce = datetime.fromisoformat(current_range[1])
            duration = ce - cs
            prior_range: tuple[str, str] = ((cs - duration).isoformat(), cs.isoformat())
        else:
            assert q.compare.range is not None
            prior_range = q.compare.range

        current_inner = build_inner(current_range)
        prior_inner = build_inner(prior_range)

        outer = exp.Select()
        outer = outer.with_("current", current_inner)
        outer = outer.with_("prior", prior_inner)

        # Outer column construction: COALESCE'd dims (then time bucket),
        # then per-measure current/prior/delta/pct_change.
        outer_columns: list[str] = []
        for col_name in dim_col_names:
            outer = outer.select(
                exp.alias_(
                    exp.Anonymous(
                        this="COALESCE",
                        expressions=[
                            exp.column(col_name, table="current"),
                            exp.column(col_name, table="prior"),
                        ],
                    ),
                    col_name,
                )
            )
            outer_columns.append(col_name)
        if has_time_breakdown:
            assert time_col_name is not None
            outer = outer.select(
                exp.alias_(
                    exp.Anonymous(
                        this="COALESCE",
                        expressions=[
                            exp.column(time_col_name, table="current"),
                            exp.column(time_col_name, table="prior"),
                        ],
                    ),
                    time_col_name,
                )
            )
            outer_columns.append(time_col_name)

        for col_name in measure_col_names:
            cur_ref = exp.column(col_name, table="current")
            pri_ref = exp.column(col_name, table="prior")
            cur_col = f"{col_name}_current"
            pri_col = f"{col_name}_prior"
            delta_col = f"{col_name}_delta"
            pct_col = f"{col_name}_pct_change"

            outer = outer.select(exp.alias_(cur_ref.copy(), cur_col))
            outer = outer.select(exp.alias_(pri_ref.copy(), pri_col))

            # delta: COALESCE both to 0 so the diff is meaningful when
            # one period is missing the entity.
            coalesced_cur = exp.Anonymous(
                this="COALESCE", expressions=[cur_ref.copy(), exp.Literal.number(0)]
            )
            coalesced_pri = exp.Anonymous(
                this="COALESCE", expressions=[pri_ref.copy(), exp.Literal.number(0)]
            )
            outer = outer.select(
                exp.alias_(
                    exp.Sub(this=coalesced_cur, expression=coalesced_pri),
                    delta_col,
                )
            )

            # pct_change: divide-by-zero guard. CASE WHEN prior > 0 THEN
            # (current - prior) * 100.0 / prior ELSE NULL END.
            # ``exp.Paren`` around the Sub is load-bearing — sqlglot's
            # renderer doesn't infer precedence for binary ops, so
            # ``Mul(Sub(a, b), 100)`` would render as ``a - b * 100``
            # (parsed as ``a - (b*100)``) without the explicit paren.
            pct_expr = exp.Case(
                ifs=[
                    exp.If(
                        this=exp.GT(this=pri_ref.copy(), expression=exp.Literal.number(0)),
                        true=exp.Div(
                            this=exp.Mul(
                                this=exp.Paren(
                                    this=exp.Sub(
                                        this=cur_ref.copy(),
                                        expression=pri_ref.copy(),
                                    )
                                ),
                                expression=exp.Literal.number(100.0),
                            ),
                            expression=pri_ref.copy(),
                        ),
                    )
                ],
                default=exp.Null(),
            )
            outer = outer.select(exp.alias_(pct_expr, pct_col))

            outer_columns.extend([cur_col, pri_col, delta_col, pct_col])

        outer = outer.from_(exp.to_table("current"))
        join_dims = dim_col_names + ([time_col_name] if time_col_name else [])
        if join_dims:
            on_expr: exp.Expression | None = None
            for jd in join_dims:
                eq = exp.EQ(
                    this=exp.column(jd, table="current"),
                    expression=exp.column(jd, table="prior"),
                )
                on_expr = eq if on_expr is None else exp.And(this=on_expr, expression=eq)
            outer = outer.join(exp.to_table("prior"), on=on_expr, join_type="full outer")
        else:
            # No dims, no time bucket — cross product of single rows.
            outer = outer.join(
                exp.to_table("prior"),
                on=exp.Boolean(this=True),
                join_type="full outer",
            )

        for ref, direction in q.order:
            if ref in outer_columns:
                order_target_node: exp.Expression = exp.column(ref)
            else:
                raise CompileError(
                    f"ORDER BY {ref!r}: in compare mode, only the outer "
                    f"output column names are addressable "
                    f"(e.g. {measure_col_names[0]}_delta). Got {ref!r}."
                )
            outer = outer.order_by(exp.Ordered(this=order_target_node, desc=(direction == "desc")))

        if q.limit is not None:
            outer = outer.limit(int(q.limit))
        if q.offset is not None and q.offset > 0:
            outer = outer.offset(int(q.offset))

        sql = outer.sql(dialect=dialect, pretty=False, normalize_functions=False)
        return Compiled(backend=backend, sql=sql, params=params, columns=outer_columns)

    # 7. Non-compare path — single inner Select with order/having/limit
    # applied directly.
    assert q.time_dimension is None or time_dim is not None
    time_range_for_query: tuple[str, str] | None = (
        q.time_dimension.range if q.time_dimension is not None else None
    )
    # If there's no time_dim, build_inner skips the time-window WHERE; the
    # explicit tuple is only consulted when ``time_dim`` is set.
    select_node = build_inner(time_range_for_query or ("", ""))

    # Re-derive output columns + measure alias map for HAVING / ORDER BY.
    columns: list[str] = list(dim_col_names)
    if has_time_breakdown and time_col_name is not None:
        columns.append(time_col_name)
    columns.extend(measure_col_names)

    if not columns:
        raise CompileError("Compiled query has no SELECT projections.")

    measure_alias_map: dict[str, exp.Expression] = {}
    for (cube_owner, m), col_name in zip(measure_fields, measure_col_names, strict=True):
        agg_node = build_measure_expr(cube_owner, m)
        measure_alias_map[m.name] = agg_node
        if col_name != m.name:
            measure_alias_map[col_name] = agg_node

    # 8. HAVING — bare or qualified measure reference.
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

    # 9. ORDER BY.
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

    # 10. LIMIT / OFFSET.
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
